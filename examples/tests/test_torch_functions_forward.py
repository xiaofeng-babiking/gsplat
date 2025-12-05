import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import time
import random
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from typing import OrderedDict, Optional
from datasets.colmap import Parser as ColmapParser
from datasets.colmap import Dataset as ColmapDataset
from gsplat.rendering import rasterization, rasterize_to_pixels, spherical_harmonics
from gsplat.logger import create_logger
from fused_ssim import FusedSSIMMap, fused_ssim
from torchmetrics import StructuralSimilarityIndexMeasure
from gsplat.optimizers.torch_functions_forward import *
from gsplat.cuda._torch_impl import (
    _quat_scale_to_matrix,
    _persp_proj,
    _world_to_cam,
    _fully_fused_projection,
)

COLMAP_DATA_PATH = os.getenv("COLMAP_DATA_PATH")
if not COLMAP_DATA_PATH:
    raise ValueError("Please set the Environment Variable 'COLMAP_DATA_PATH'!")
GSPLAT_CHECKPOINT_FILE = os.getenv("GSPLAT_CHECKPOINT_FILE")
if not GSPLAT_CHECKPOINT_FILE:
    raise ValueError("Please set the Environment Variable 'GSPLAT_CHECKPOINT_FILE'!")
RESOLUTION_FACTOR = 4  # use 1/4-resolution images by default
LOGGER = create_logger(
    name=os.path.splitext(os.path.basename(__file__))[0], level="INFO"
)


def generate_render_data_sample(batch_mode: bool = True):
    """Generate rendering sample data to evaluate SSIM."""
    LOGGER.info(
        f"Start to parse COLMAP data from {os.path.basename(COLMAP_DATA_PATH)}..."
    )
    parser = ColmapParser(
        data_dir=COLMAP_DATA_PATH,
        factor=RESOLUTION_FACTOR,
        normalize=True,
        test_every=8,
    )
    dataset = ColmapDataset(parser, split="test")
    n_imgs = len(dataset)
    LOGGER.info(f"Loaded {n_imgs} views from {os.path.basename(COLMAP_DATA_PATH)}.")

    LOGGER.info(
        f"Start to load gsplat checkpoint from {os.path.basename(GSPLAT_CHECKPOINT_FILE)}..."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splats = torch.load(GSPLAT_CHECKPOINT_FILE)["splats"]
    splats = OrderedDict({k: v.to(device) for k, v in splats.items()})
    sh_deg = int(np.sqrt(splats["sh0"].shape[1] + splats["shN"].shape[1])) - 1
    LOGGER.info(
        f"Loaded {len(splats['means'])} Splats from {os.path.basename(GSPLAT_CHECKPOINT_FILE)}."
    )

    LOGGER.info(f"Start to concatenate groundtruth images and test views...")
    img_ids = []
    gt_imgs = []
    Ks = []
    viewmats = []
    for data_item in tqdm(dataset, desc="Iterate over the dataset...", total=n_imgs):
        img_id = data_item["image_id"]
        img_ids.append(img_id)

        gt_img = data_item["image"].to(device)[None, ...]
        gt_img = gt_img.float() / 255.0
        # image dimension format NCHW
        gt_img = torch.permute(gt_img, dims=[0, 3, 1, 2])
        gt_imgs.append(gt_img)

        # Dimension=[1, 3, 3]
        K = data_item["K"][None, ...]
        K = K.to(device)
        Ks.append(K)

        # Dimension=[1, 4, 4]
        viewmat = data_item["camtoworld"][None, ...]
        viewmat = viewmat.to(device)
        viewmats.append(viewmat)

    img_ids = torch.tensor(img_ids, dtype=torch.int32)
    gt_imgs = torch.cat(gt_imgs, dim=0)
    Ks = torch.cat(Ks, dim=0)
    viewmats = torch.linalg.inv(torch.cat(viewmats, dim=0))

    img_h, img_w = gt_imgs.shape[2:]
    assert gt_imgs.shape[0] == n_imgs

    LOGGER.info(f"Start to render {n_imgs} images with resolution={img_w}x{img_h}...")
    if batch_mode:
        rd_imgs, rd_alphas, rd_metas = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
            Ks=Ks,
            viewmats=viewmats,
            width=img_w,
            height=img_h,
            sh_degree=sh_deg,
            packed=True,
        )
    else:
        rd_imgs = []
        rd_alphas = []
        rd_metas = []
        for i in range(n_imgs):
            rd_img, rd_alpha, rd_meta = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
                Ks=Ks[i].unsqueeze(0),
                viewmats=viewmats[i].unsqueeze(0),
                width=img_w,
                height=img_h,
                sh_degree=sh_deg,
                packed=True,
            )
            rd_imgs.append(rd_img)
            rd_alphas.append(rd_alpha)
            rd_metas.append(rd_meta)
        rd_imgs = torch.concatenate(rd_imgs, dim=0)
        rd_alphas = torch.concatenate(rd_alphas, dim=0)
    LOGGER.info(f"Finished rendering {n_imgs} images.")

    rd_imgs = torch.permute(rd_imgs, dims=[0, 3, 1, 2])
    rd_imgs = torch.clamp(rd_imgs, min=0.0, max=1.0)

    RenderDataSample = namedtuple(
        typename="RenderDataSample",
        field_names=[
            "image_id",
            "camera_matrix",
            "view_matrix",
            "groundtruth_image",
            "render_image",
            "render_alpha",
            "render_metadata",
        ],
    )
    data = RenderDataSample(
        image_id=img_ids,
        camera_matrix=Ks,
        view_matrix=viewmats,
        groundtruth_image=gt_imgs,
        render_image=rd_imgs,
        render_alpha=rd_alphas,
        render_metadata=rd_metas,
    )
    return data, splats


def test_rasterization_tile_forward():
    """Test rasterization tile forward."""
    data, splats = generate_render_data_sample(batch_mode=False)

    rd_imgs = data.render_image.contiguous()
    rd_metas = data.render_metadata
    cam_mats = data.camera_matrix.contiguous()
    view_mats = data.view_matrix.contiguous()

    cam_idx = 3  # random.choice(list(range(cam_mats.shape[0])))

    rd_img = rd_imgs[cam_idx, :, :, :][None]
    cam_mat = cam_mats[cam_idx, :, :][None]
    view_mat = view_mats[cam_idx, :, :][None]
    rd_meta = rd_metas[cam_idx]
    img_w = rd_meta["width"]
    img_h = rd_meta["height"]

    del rd_imgs, cam_mats, view_mats, rd_metas
    torch.cuda.empty_cache()

    gauss_ids = rd_meta["gaussian_ids"]
    LOGGER.info(f"Random select camera-{cam_idx}, #Splats={len(gauss_ids)}.")

    means3d = splats["means"][gauss_ids]
    scales = torch.exp(splats["scales"][gauss_ids])
    opacities = torch.sigmoid(splats["opacities"][gauss_ids])
    quats = splats["quats"][gauss_ids]
    sh_coeffs = torch.cat([splats["sh0"][gauss_ids], splats["shN"][gauss_ids]], dim=1)

    # 1. test means 3d to 2d projection
    means2d_fwd = project_means_3d_to_2d(means3d, view_mat, cam_mat)
    means2d_meta = rd_meta["means2d"]
    assert torch.allclose(
        means2d_fwd, means2d_meta, rtol=1e-4, atol=1e-3
    ), f"Project means 3d to 2d failed!"

    # 2. test spherical harmonics colors
    sh_deg = int(np.sqrt(sh_coeffs.shape[-2])) - 1
    view_dirs = means3d[None, :, :] - torch.linalg.inv(view_mat)[:, :3, 3][:, None, :]
    sh_colors_fwd = combine_sh_colors_from_coefficients(view_dirs, sh_coeffs)
    sh_colors_meta = spherical_harmonics(sh_deg, view_dirs, sh_coeffs[None])
    assert torch.allclose(
        sh_colors_fwd, sh_colors_meta, atol=1e-4, rtol=1e-3
    ), f"Spherical harmonics colors failed!"

    # 3. test inverse covariance 2d
    covars3d_fwd = compute_covariance_3d(quats, scales)
    covars3d_meta = _quat_scale_to_matrix(quats, scales)
    assert torch.allclose(
        covars3d_fwd, covars3d_meta, atol=1e-4, rtol=1e-3
    ), "Covariance 3D failed!"

    means3d_cam, covars3d_cam = _world_to_cam(means3d, covars3d_meta, view_mat)
    means2d_meta, covars2d_meta = _persp_proj(
        means3d_cam, covars3d_cam, cam_mat, img_w, img_h
    )
    covars2d_fwd, conics2d_fwd = project_covariances_3d_to_2d(
        view_mat, cam_mat, means3d, covars3d_fwd, img_w, img_h
    )
    assert torch.allclose(
        covars2d_fwd, covars2d_meta, atol=1e-4, rtol=1e-1
    ), "Covariance 2D failed!"

    conics2d_meta = rd_meta["conics"]

if __name__ == "__main__":
    test_rasterization_tile_forward()
