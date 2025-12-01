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
from gsplat.rendering import rasterization, spherical_harmonics, rasterize_to_pixels
from gsplat.logger import create_logger
from gsplat.optimizers.sparse_newton import (
    GroupSSIMLoss,
    GSGroupNewtonOptimizer,
    raster_to_pixels_torch,
)
from fused_ssim import FusedSSIMMap, fused_ssim
from torchmetrics import StructuralSimilarityIndexMeasure

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


def parse_fused_ssim_gauss_filter_1d(
    marco_str: Optional[str] = None,
) -> torch.FloatTensor:
    """ "Parse Gaussian filter 1D from fused-ssim macro string."""
    if marco_str is None:
        marco_str = """
            #define G_00 0.001028380123898387f
            #define G_01 0.0075987582094967365f
            #define G_02 0.036000773310661316f
            #define G_03 0.10936068743467331f
            #define G_04 0.21300552785396576f
            #define G_05 0.26601171493530273f
            #define G_06 0.21300552785396576f
            #define G_07 0.10936068743467331f
            #define G_08 0.036000773310661316f
            #define G_09 0.0075987582094967365f
            #define G_10 0.001028380123898387f
        """
    pattern = re.compile(r"#define\s+G_([0-9]+)\s+(.*)f", re.MULTILINE)
    matches = [(int(i), float(x)) for i, x in pattern.findall(marco_str.strip())]

    filter = torch.tensor([x for _, x in sorted(matches)], dtype=torch.float32)
    return filter


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


def test_fused_ssim_forward():
    """Test forward pass of Fused SSIM."""
    data, _ = generate_render_data_sample(batch_mode=True)

    gt_imgs = data.groundtruth_image.contiguous()
    rd_imgs = data.render_image.contiguous()

    device = gt_imgs.device

    torch_ssim = StructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction="none"
    ).to(device)
    ssim_0 = torch_ssim(gt_imgs, rd_imgs)
    LOGGER.info(f"Metrics average SSIM score: {ssim_0.mean():.4f}.")

    # ssim_1 = FusedSSIMMap.apply(0.01**2, 0.01**2, gt_imgs, rd_imgs, "valid", False)
    # ssim_1 = ssim_1.mean(dim=[1, 2, 3])
    ssim_1 = np.array(
        [
            fused_ssim(x[None, ...], y[None, ...], padding="valid", train=False).item()
            for x, y in zip(rd_imgs, gt_imgs)
        ],
        dtype=np.float32,
    )

    start = time.time()
    ssim_1_batch = (
        FusedSSIMMap.apply(0.01**2, 0.03**2, rd_imgs, gt_imgs, "valid", False)
        .mean(dim=[1, 2, 3])
        .cpu()
        .detach()
        .numpy()
    )
    end = time.time()
    elapsed_1 = float(end - start)
    assert np.allclose(
        ssim_1, ssim_1_batch, atol=1e-4, rtol=1e-1
    ), "Fused SSIM abnormal batch-mode behavior!"

    filter = parse_fused_ssim_gauss_filter_1d().to(device)
    group_ssim = GroupSSIMLoss(
        filter=filter,
        c1=0.01**2,
        c2=0.03**2,
        padding="valid",
        reduction="mean",
    )
    group_ssim = group_ssim.to(device)
    ssim_2 = np.array(
        [
            group_ssim(x[None, ...], y[None, ...]).item()
            for x, y in zip(rd_imgs, gt_imgs)
        ],
        dtype=np.float32,
    )
    start = time.time()
    ssim_2_batch = group_ssim(gt_imgs, rd_imgs).cpu().detach().numpy()
    end = time.time()
    elapsed_2 = float(end - start)
    assert np.allclose(
        ssim_2, ssim_2_batch, atol=1e-4, rtol=1e-1
    ), f"GroupSSIM abnormal batch-mode behavior!"

    # SSIMScore = 1 - SSIMSLoss
    assert np.allclose(
        ssim_1, 1.0 - ssim_2, atol=1e-6, rtol=1e-4
    ), f"Fused and Group SSIM forward mismatch!"

    LOGGER.info(
        f"Forward time fused={elapsed_1:.6f}s, group={elapsed_2:.6f}s "
        + f"({elapsed_2 / elapsed_1:.4f} slower)."
    )

    torch.cuda.empty_cache()


def test_fused_ssim_backward():
    """Test backward pass of Fused SSIM."""
    data, _ = generate_render_data_sample(batch_mode=True)

    gt_imgs = data.groundtruth_image.contiguous()
    rd_imgs = data.render_image.contiguous()
    rd_imgs.requires_grad = True

    device = gt_imgs.device

    ssim_loss = fused_ssim(rd_imgs, gt_imgs, padding="valid")
    ssim_loss.require_grad = True

    start = time.time()
    ssim_loss.backward()
    end = time.time()
    torch_elapsed = float(end - start)
    torch_jacob = rd_imgs.grad.clone().cpu().detach().numpy()

    rd_imgs = rd_imgs.data.clone().detach()
    rd_imgs.requires_grad = False
    rd_imgs.grad = None
    torch.cuda.empty_cache()

    filter = parse_fused_ssim_gauss_filter_1d().to(device)
    start = time.time()
    group_jacob, group_hess = GSGroupNewtonOptimizer._backward_ssim_to_render(
        rd_imgs, gt_imgs, filter=filter, padding="valid", with_hessian=True, eps=1e-15
    )
    end = time.time()
    group_elapsed = float(end - start)
    group_jacob = group_jacob.cpu().detach().numpy()

    assert not np.any(
        np.isnan(group_jacob) | np.isinf(group_jacob)
    ), "Group SSIM jacobian contains NaN or Inf!"

    if group_hess is not None:
        group_hess = group_hess.cpu().detach().numpy()

        assert group_hess.shape == group_jacob.shape
        f"Group SSIM jacobian and hessian shape mismatch!"

        assert not np.any(
            np.isnan(group_hess) | np.isinf(group_hess)
        ), "Group SSIM jacobian contains NaN or Inf!"

    torch_jacob *= torch_jacob.size
    group_jacob *= group_jacob.size

    # SSIMScore = 1 - SSIMSLoss
    rerr = np.abs(torch_jacob - (-group_jacob)) / np.abs(torch_jacob)
    bad_rerr_ratio = len(np.where(rerr > 1e-3)[0]) / torch_jacob.size
    assert bad_rerr_ratio < 1.5e-2, f"Fused and Group SSIM backward mismatch!"

    LOGGER.info(
        f"Backward time fused={torch_elapsed:.6f}s, group={group_elapsed:.6f}s "
        + f"({group_elapsed / torch_elapsed:.4f} slower)."
        + f"Ratio relative error > 1e-3: {bad_rerr_ratio * 100.0:.4f}%."
    )
    torch.cuda.empty_cache()


def test_render_color_forward():
    """Test backward pass of spherical harmonics color."""
    data, splats = generate_render_data_sample(batch_mode=False)

    gt_imgs = data.groundtruth_image.contiguous()
    # rd_imgs = data.render_image.contiguous()
    rd_metas = data.render_metadata
    # cam_mats = data.camera_matrix.contiguous()
    view_mats = data.view_matrix.contiguous()

    n_cams = gt_imgs.shape[0]
    cam_idx = random.choice(list(range(n_cams)))

    view_mat = view_mats[cam_idx]

    rd_meta = rd_metas[cam_idx]

    gt_img = gt_imgs[cam_idx]
    img_h, img_w, _ = gt_img.shape

    n_splats = len(rd_meta["means2d"])

    gauss_idxs = rd_meta["gaussian_ids"]

    view_dirs = splats["means"][gauss_idxs] - torch.linalg.inv(view_mat)[:3, 3][None]
    view_masks = (rd_meta["radii"] > 0).all(dim=-1)
    sh_coeffs = torch.cat([splats["sh0"][gauss_idxs], splats["shN"][gauss_idxs]], dim=1)
    rd_colors = spherical_harmonics(
        degrees_to_use=int(np.sqrt(sh_coeffs.shape[-2])) - 1,
        dirs=view_dirs,
        coeffs=sh_coeffs,
        masks=view_masks,
    )
    rd_colors = torch.clamp_min(rd_colors + 0.5, 0.0)

    tile_size = rd_meta["tile_size"]
    tile_h = int(np.ceil(rd_meta["height"] / tile_size))
    assert tile_h == rd_meta["tile_height"]
    tile_w = int(np.ceil(rd_meta["width"] / tile_size))
    assert tile_w == rd_meta["tile_width"]

    isect_offsets = rd_meta["isect_offsets"][0].flatten()
    assert len(isect_offsets) == tile_h * tile_w

    assert len(torch.unique(rd_meta["flatten_ids"])) == len(rd_meta["means2d"])

    start = time.time()
    rd_img, rd_alphas = rasterize_to_pixels(
        means2d=rd_meta["means2d"],
        conics=rd_meta["conics"],
        colors=rd_colors,
        opacities=rd_meta["opacities"],
        image_height=rd_meta["height"],
        image_width=rd_meta["width"],
        tile_size=rd_meta["tile_size"],
        isect_offsets=rd_meta["isect_offsets"],
        flatten_ids=rd_meta["flatten_ids"],
        packed=True,
    )
    end = time.time()
    cuda_elapsed = float(end - start)
    LOGGER.info(
        f"CUDA #Splats={n_splats}, Image={img_w}x{img_h}, Elapsed={cuda_elapsed:.6f} seconds."
    )

    start = time.time()
    fwd_img, fwd_alphas = raster_to_pixels_torch(
        means2d=rd_meta["means2d"],
        conics=rd_meta["conics"],
        colors=rd_colors,
        opacities=rd_meta["opacities"],
        image_height=rd_meta["height"],
        image_width=rd_meta["width"],
        tile_size=rd_meta["tile_size"],
        isect_offsets=rd_meta["isect_offsets"],
        flatten_ids=rd_meta["flatten_ids"],
        packed=True,
    )
    end = time.time()
    torch_elapsed = float(end - start)
    LOGGER.info(
        f"Torch #Splats={n_splats}, Elapsed={torch_elapsed:.6f} seconds"
        + f"(Slower={(torch_elapsed / cuda_elapsed):.4f})"
    )

    assert not torch.isinf(fwd_img).any()
    assert not torch.isnan(fwd_img).any()

    rd_img = torch.clamp(rd_img, min=0.0, max=1.0)
    fwd_img = torch.clamp(fwd_img, min=0.0, max=1.0)
    ssim_measure = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=11,
        k1=0.01,
        k2=0.03,
        sigma=1.5,
        reduction="none",
    ).to(rd_img.device)
    rgb_ssim_metric = (
        ssim_measure(rd_img.permute([0, 3, 1, 2]), fwd_img.permute([0, 3, 1, 2]))
        .mean()
        .item()
    )
    rgb_l2_metric = torch.linalg.norm(rd_img - fwd_img, dim=-1).mean().item()
    assert rgb_ssim_metric > 0.99 and rgb_l2_metric < 0.005

    alpha_mean_err = torch.abs(rd_alphas - fwd_alphas).mean().item()
    assert alpha_mean_err < 0.003


if __name__ == "__main__":
    test_render_color_forward()
    test_fused_ssim_forward()
    test_fused_ssim_backward()
