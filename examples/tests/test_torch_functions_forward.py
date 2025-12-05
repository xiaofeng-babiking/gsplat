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
from typing import OrderedDict
from datasets.colmap import Parser as ColmapParser
from datasets.colmap import Dataset as ColmapDataset
from gsplat.rendering import rasterization, spherical_harmonics, fully_fused_projection
from gsplat.logger import create_logger
from torchmetrics import StructuralSimilarityIndexMeasure
from gsplat.optimizers.torch_functions_forward import *

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

    view_idx = 3  # random.choice(list(range(cam_mats.shape[0])))

    rd_img = rd_imgs[view_idx, :, :, :][None]
    rd_img = rd_img.permute([0, 2, 3, 1])  # NCHW -> NHWC
    cam_mat = cam_mats[view_idx, :, :][None]
    view_mat = view_mats[view_idx, :, :][None]
    rd_meta = rd_metas[view_idx]
    img_w = rd_meta["width"]
    img_h = rd_meta["height"]
    tile_size = rd_meta["tile_size"]
    tile_w = int(math.ceil(img_w / tile_size))
    tile_h = int(math.ceil(img_h / tile_size))
    isect_offsets = rd_meta["isect_offsets"]
    assert isect_offsets.shape == (1, tile_h, tile_w)
    flatten_ids = rd_meta["flatten_ids"]

    del rd_imgs, cam_mats, view_mats, rd_metas
    torch.cuda.empty_cache()

    device = rd_img.device

    gauss_ids = rd_meta["gaussian_ids"]
    LOGGER.info(f"Random select View-{view_idx}, #Splats={len(gauss_ids)}.")

    means3d = splats["means"][gauss_ids]
    scales = torch.exp(splats["scales"][gauss_ids])
    opacities = torch.sigmoid(splats["opacities"][gauss_ids])
    assert torch.allclose(opacities, rd_meta["opacities"], rtol=1e-3, atol=1e-3)
    quats = splats["quats"][gauss_ids]
    sh_coeffs = torch.cat([splats["sh0"][gauss_ids], splats["shN"][gauss_ids]], dim=1)

    # 1. test spherical harmonics colors
    sh_deg = int(np.sqrt(sh_coeffs.shape[-2])) - 1
    view_dirs = means3d[None, :, :] - torch.linalg.inv(view_mat)[:, :3, 3][:, None, :]
    sh_colors_fwd = combine_sh_colors_from_coefficients(view_dirs, sh_coeffs)
    sh_colors_meta = spherical_harmonics(sh_deg, view_dirs, sh_coeffs[None])
    assert torch.allclose(
        sh_colors_fwd, sh_colors_meta, atol=1e-4, rtol=1e-3
    ), f"Spherical harmonics colors failed!"

    # 2. test 3D to 2D projection
    covars3d_fwd = compute_covariance_3d(quats, scales)
    assert torch.allclose(
        covars3d_fwd, covars3d_fwd.transpose(-1, -2)
    ), f"Covariance 3D NOT symmetric!"

    means2d_fwd, conics2d_fwd, depths_fwd, radii_fwd, _ = project_gaussians_3d_to_2d(
        view_mat, cam_mat, means3d, covars3d_fwd, img_w, img_h, opacities=opacities
    )
    assert torch.allclose(opacities, rd_meta["opacities"], atol=1e-4, rtol=1e-3)

    means2d_meta = rd_meta["means2d"]
    assert torch.allclose(
        means2d_fwd[0], means2d_meta, atol=1e-4, rtol=1e-2
    ), f"Means 2D projection failed!"

    conics2d_meta = rd_meta["conics"]
    assert torch.allclose(
        conics2d_fwd[0], conics2d_meta, atol=1e-4, rtol=1e-3
    ), "Inverse covariance 2D failed!"

    depths_meta = rd_meta["depths"]
    assert torch.allclose(
        depths_fwd[0], depths_meta, atol=1e-4, rtol=1e-3
    ), f"Depth 2D projection failed!"

    radii_meta = rd_meta["radii"]
    torch.allclose(radii_fwd[0], radii_meta), f"Radius int failed!"

    # 3. test whole image rendering
    from gsplat.optimizers.sparse_newton import GSGroupNewtonOptimizer

    cache = GSGroupNewtonOptimizer._cache_tile_to_splat_alphas(
        gauss_ids,
        means2d_fwd[0],
        conics2d_fwd[0],
        opacities,
        img_h,
        img_w,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    ssimer = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    fwd_img = torch.zeros(size=[1, img_h, img_w, 3], dtype=torch.float32, device=device)
    isect_offsets = isect_offsets[0].flatten()
    for tile_x in range(tile_w):
        for tile_y in range(tile_h):
            start = time.time()
            tile_idx = tile_y * tile_w + tile_x

            isect_start = isect_offsets[tile_idx]
            isect_end = (
                isect_offsets[tile_idx + 1]
                if tile_idx < tile_h * tile_w - 1
                else len(flatten_ids)
            )

            if isect_start >= isect_end:
                continue

            flat_idxs = flatten_ids[isect_start:isect_end]

            tile_rgb, tile_a, tile_bbox = rasterize_to_pixels_tile_forward(
                tile_x,
                tile_y,
                tile_size,
                img_w,
                img_h,
                means3d[flat_idxs],
                quats[flat_idxs],
                scales[flat_idxs],
                opacities[flat_idxs],
                sh_coeffs[flat_idxs],
                cam_mat,
                view_mat,
            )
            end = time.time()
            tile_elapsed = float(end - start)

            tile_rgb = torch.clamp(tile_rgb, min=0.0, max=1.0)
            tile_xmin, tile_ymin, crop_w, crop_h = tile_bbox

            tile_alphas = cache[tile_idx]["tile_alphas"]
            blend_alphas = cache[tile_idx]["blend_alphas"]
            flat_idxs = cache[tile_idx]["tile_splat_indices"]
            cache_rgb = torch.sum(
                tile_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
                * blend_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
                * sh_colors_fwd[0, flat_idxs, :][None, None, :, :],  # [1, 1, tK, 3]
                dim=-2,
            )[None]
            cache_rgb = torch.clamp(cache_rgb, min=0.0, max=1.0)

            rd_rgb = rd_img[
                :, tile_ymin : (tile_ymin + crop_h), tile_xmin : (tile_xmin + crop_w), :
            ]

            ssim = ssimer(rd_rgb.permute([0, 3, 1, 2]), cache_rgb.permute([0, 3, 1, 2]))
            LOGGER.info(
                f"Tile=({tile_x}, {tile_y}), Elapsed={tile_elapsed:.4f} seconds, SSIM={ssim:.4f}."
            )

            fwd_img[
                :, tile_ymin : (tile_ymin + crop_h), tile_xmin : (tile_xmin + crop_w), :
            ] = tile_rgb[:, :crop_h, :crop_w, :]


if __name__ == "__main__":
    test_rasterization_tile_forward()
