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
from gsplat.optimizers.sparse_newton import (
    GroupSSIMLoss,
    GSGroupNewtonOptimizer,
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

    device = gt_imgs.device

    ssim_loss_func = lambda rd_imgs: fused_ssim(rd_imgs, gt_imgs, padding="valid")

    start = time.time()
    torch_jacob = torch.autograd.functional.jacobian(
        ssim_loss_func, rd_imgs, vectorize=False, create_graph=True
    )
    # NOTE. OOM Dim -> [N, H, W, C, 3]^2
    # torch_hess = torch.autograd.functional.hessian(
    #     ssim_loss_func, rd_imgs, vectorize=False, create_graph=True
    # )
    end = time.time()
    torch_elapsed = float(end - start)
    torch.cuda.empty_cache()

    filter = parse_fused_ssim_gauss_filter_1d().to(device)
    start = time.time()
    group_jacob, group_hess = GSGroupNewtonOptimizer._backward_ssim_to_render(
        rd_imgs, gt_imgs, filter=filter, padding="valid", with_hessian=True, eps=1e-15
    )
    end = time.time()
    group_elapsed = float(end - start)

    assert not torch.any(
        torch.isnan(group_jacob) | torch.isinf(group_jacob)
    ), "Group SSIM jacobian contains NaN or Inf!"

    assert not torch.any(
        torch.isnan(group_hess) | torch.isinf(group_hess)
    ), "Group SSIM jacobian contains NaN or Inf!"

    torch_jacob *= torch_jacob.numel()
    group_jacob *= group_jacob.numel()

    # SSIMScore = 1 - SSIMSLoss
    rerr = torch.abs(torch_jacob - (-group_jacob)) / torch.abs(torch_jacob)
    bad_rerr_ratio = len(torch.where(rerr > 1e-3)[0]) / torch_jacob.numel()
    assert bad_rerr_ratio < 1.5e-2, f"Fused and Group SSIM backward mismatch!"

    LOGGER.info(
        f"Backward time fused={torch_elapsed:.6f}s, group={group_elapsed:.6f}s "
        + f"({group_elapsed / torch_elapsed:.4f} slower)."
        + f"Ratio relative error > 1e-3: {bad_rerr_ratio * 100.0:.4f}%."
    )
    torch.cuda.empty_cache()


def test_render_color_forward():
    """Test pytorch tile-based alpha-blending i.e. forward rendering."""
    data, splats = generate_render_data_sample(batch_mode=False)

    rd_imgs = data.render_image.contiguous()
    rd_metas = data.render_metadata
    view_mats = data.view_matrix.contiguous()

    n_cams = rd_imgs.shape[0]
    cam_idx = random.choice(list(range(n_cams)))

    rd_img = rd_imgs[cam_idx][None].permute(0, 2, 3, 1)  # NCHW -> NHWC
    device = rd_img.device

    view_mat = view_mats[cam_idx][None]  # [1, 4, 4]

    rd_meta = rd_metas[cam_idx]
    n_splats = len(rd_meta["means2d"])

    gauss_ids = rd_meta["gaussian_ids"]
    means3d = splats["means"][gauss_ids]
    sh_coeffs = torch.cat([splats["sh0"][gauss_ids], splats["shN"][gauss_ids]], dim=1)
    means2d = rd_meta["means2d"]
    radii = rd_meta["radii"]
    opacities = rd_meta["opacities"]
    conics = rd_meta["conics"]
    assert (
        len(means3d)
        == len(sh_coeffs)
        == len(means2d)
        == len(radii)
        == len(opacities)
        == len(conics)
    ), f"Mismatch between number of splats!"

    img_h = rd_meta["height"]
    img_w = rd_meta["width"]
    tile_size = rd_meta["tile_size"]
    isect_offsets = rd_meta["isect_offsets"]
    flatten_ids = rd_meta["flatten_ids"]

    start = time.time()
    sh_colors_cache = GSGroupNewtonOptimizer._cache_sh_colors(
        means3d=means3d,
        view_mats=view_mat,
        sh_coeffs=sh_coeffs,
        radii=radii,
    )
    end = time.time()
    sh_deg = int(np.sqrt(sh_coeffs.shape[-2])) - 1
    sh_colors_cache_elapsed = float(end - start)
    LOGGER.info(
        f"SH-Colors-Cache #Splats={len(torch.unique(gauss_ids))}, SH-Degree={sh_deg}, Elapsed={sh_colors_cache_elapsed:.6f} seconds."
    )

    start = time.time()
    cuda_img, cuda_alphas = rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=sh_colors_cache[0],
        opacities=opacities,
        image_height=img_h,
        image_width=img_w,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        packed=True,
    )
    cuda_img = torch.clamp(cuda_img, min=0.0, max=1.0)
    end = time.time()
    cuda_elapsed = float(end - start)
    LOGGER.info(
        f"CUDA #Splats={n_splats}, Image={img_w}x{img_h}, Elapsed={cuda_elapsed:.6f} seconds."
    )
    assert torch.allclose(cuda_img, rd_img, atol=1e-4, rtol=1e-6)

    start = time.time()
    tile_to_alphas_cache = GSGroupNewtonOptimizer._cache_tile_to_splat_alphas(
        gauss_ids=gauss_ids,
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        img_height=img_h,
        img_width=img_w,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
    )
    end = time.time()
    tile_to_alpha_cache_elapsed = float(end - start)
    LOGGER.info(
        f"Tile-to-Alpha-Cache #Splats={n_splats}, Elapsed={tile_to_alpha_cache_elapsed:.6f} seconds"
    )

    start = time.time()
    n_imgs = sh_colors_cache.shape[0]
    torch_img = torch.zeros(
        size=[n_imgs, img_h, img_w, 3], device=device, dtype=torch.float32
    )
    torch_alphas = torch.zeros(
        size=[n_imgs, img_h, img_w, 1], device=device, dtype=torch.float32
    )

    for _, tile_meta in tile_to_alphas_cache.items():
        img_idx = tile_meta["image_index"]
        tile_x = tile_meta["tile_x"]
        tile_y = tile_meta["tile_y"]
        tile_size = tile_meta["tile_size"]
        tile_splat_idxs = tile_meta["tile_splat_indices"]
        tile_alphas = tile_meta["tile_alphas"]
        blend_alphas = tile_meta["blend_alphas"]

        tile_img = torch.sum(
            tile_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
            * blend_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
            * sh_colors_cache[img_idx, tile_splat_idxs, :][
                None, None, :, :
            ],  # [1, 1, tK, 3]
            dim=-2,
        )
        tile_alphas = 1.0 - blend_alphas[:, :, -1, None]

        paste_xmin = tile_x * tile_size
        paste_width = min(tile_size, img_w - paste_xmin)
        paste_ymin = tile_y * tile_size
        paste_height = min(tile_size, img_h - paste_ymin)

        torch_img[
            img_idx,
            paste_ymin : paste_ymin + paste_height,
            paste_xmin : paste_xmin + paste_width,
            :,
        ] = tile_img[:paste_height, :paste_width, :]
        torch_alphas[
            img_idx,
            paste_ymin : paste_ymin + paste_height,
            paste_xmin : paste_xmin + paste_width,
            :,
        ] = tile_alphas[:paste_height, :paste_width, :]

    torch_img = torch.clamp(torch_img, min=0.0, max=1.0)
    end = time.time()
    torch_elapsed = float(end - start)

    n_splats_per_tile = np.mean(
        [v["tile_splat_indices"].shape[0] for v in tile_to_alphas_cache.values()]
    )
    LOGGER.info(
        f"Torch #Splats={n_splats}, Image={img_w}x{img_h}, "
        + f"#SplatsPerTile={n_splats_per_tile:.4f}, TileSize={tile_size}, "
        + f"Elapsed={torch_elapsed:.6f} seconds ({torch_elapsed / cuda_elapsed:.4f} slower)."
    )

    ssim_measure = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=11,
        k1=0.01,
        k2=0.03,
        sigma=1.5,
        reduction="none",
    ).to(device)
    rgb_ssim_metric = (
        ssim_measure(cuda_img.permute([0, 3, 1, 2]), torch_img.permute([0, 3, 1, 2]))
        .mean()
        .item()
    )
    rgb_l2_metric = torch.linalg.norm(cuda_img - torch_img, dim=-1).mean().item()
    assert rgb_ssim_metric > 0.99 and rgb_l2_metric < 0.005

    alpha_mean_err = torch.abs(cuda_alphas - torch_alphas).mean().item()
    assert alpha_mean_err < 0.003


def test_backward_render_to_sh_color_tile():
    """Test Backward pass of rendered pixels w.r.t spherical harmonics colors."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tile_size = 16

    n_channels = 3

    # Per-tile number of splats ~= 1000
    n_splats = np.random.randint(900, 1200)

    sh_colors = torch.rand(
        size=[n_splats, n_channels], device=device, dtype=torch.float32
    )
    tile_alphas = torch.rand(
        size=[tile_size, tile_size, n_splats],
        device=device,
        dtype=torch.float32,
    )

    blend_alphas = torch.cumprod(1.0 - tile_alphas, dim=-1)
    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
    blend_alphas[:, :, 0] = 1.0

    alpha_blend_func = lambda colors: torch.sum(
        tile_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
        * blend_alphas[:, :, :, None]  # [tile_size, tile_size, tK, 1]
        * colors[None, None, :, :],  # [1, 1, tK, 3]
        dim=-2,
    )

    start = time.time()
    tile_jacob, tile_hess = GSGroupNewtonOptimizer._backward_render_to_sh_color_tile(
        n_channels,
        tile_alphas.clone().detach(),
        blend_alphas.clone().detach(),
        with_hessian=False,
    )
    end = time.time()
    assert tile_hess is None
    ours_elapsed = float(end - start)

    start = time.time()
    torch_jacob = torch.autograd.functional.jacobian(
        alpha_blend_func,
        sh_colors,
        vectorize=False,
        create_graph=True,
    )
    end = time.time()
    torch_jacob_elapsed = float(end - start)

    start = time.time()
    torch_hess = torch.autograd.functional.hessian(
        lambda x: alpha_blend_func(x).sum(),
        sh_colors,
        vectorize=False,
        create_graph=True,
    )
    assert torch.allclose(torch_hess, torch.zeros_like(torch_hess))
    end = time.time()
    torch_hess_elapsed = float(end - start)
    LOGGER.info(
        f"#Splats={n_splats}, TileSize={tile_size}, OursJ={ours_elapsed:.6f} secs, TorchJ={torch_jacob_elapsed:.6f} secs, TorchH={torch_hess_elapsed:.6f} secs."
    )

    rgb_idxs = [0, 1, 2]
    torch_jacob = torch_jacob.permute(
        [0, 1, 3, 2, 4]
    )  # [tile_size, tile_size, 3, tK, 3] -> [tile_size, tile_size, tK, 3]
    torch_jacob = torch_jacob[:, :, :, rgb_idxs, rgb_idxs]
    assert torch.allclose(tile_jacob, torch_jacob)


def test_backward_sh_color_to_position():
    """Test backward pass of Spherical Harmonics w.r.t view directions."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_views = np.random.randint(5, 10)
    n_splats = np.random.randint(100, 300)
    n_channels = 3

    sh_deg = 3

    sh_coeffs = torch.rand(
        size=[n_splats, (sh_deg + 1) ** 2, n_channels],
        device=device,
        dtype=torch.float32,
    )

    view_mats = (
        torch.randn(
            size=[n_views, 4, 4],
            device=device,
            dtype=torch.float32,
        )
        * 100.0
        + 33.0
    )
    means3d = (
        torch.randn(
            size=[n_splats, 3],
            device=device,
            dtype=torch.float32,
        )
        * 10.0
        + 3.0
    )

    start = time.time()
    sh_deriv_funcs = GSGroupNewtonOptimizer._cache_sh_derivative_functions(
        sh_deg=sh_deg
    )
    end = time.time()
    sh_deriv_elapsed = float(end - start)
    LOGGER.info(
        f"SH Degree={sh_deg}, Cache SH Derivatives={sh_deriv_elapsed:.6f} seconds."
    )

    start = time.time()
    ours_jacob, _ = GSGroupNewtonOptimizer._backward_sh_color_to_position(
        view_mats,
        means3d,
        sh_coeffs,
        sh_deriv_funcs=sh_deriv_funcs,
        with_hessian=True,
    )
    end = time.time()
    ours_jacob_elapsed = float(end - start)
    LOGGER.info(
        f"#Views={n_views}, Splats={n_splats}, OursJacobian={ours_jacob_elapsed:.6f} seconds."
    )

    forward_func = lambda positions: spherical_harmonics(
        degrees_to_use=sh_deg,
        dirs=positions[None, :, :] - torch.linalg.inv(view_mats)[:, :3, 3][:, None, :],
        coeffs=sh_coeffs[None].repeat([n_views, 1, 1, 1]),
    )
    torch_jacob = torch.autograd.functional.jacobian(
        forward_func,
        means3d,
        vectorize=False,
        create_graph=True,
    )
    torch_jacob = torch_jacob.permute([0, 1, 3, 2, 4])
    splat_idxs = list(range(n_splats))
    torch_jacob = torch_jacob[:, splat_idxs, splat_idxs, :, :]
    assert torch.allclose(
        ours_jacob, torch_jacob, atol=1e-3, rtol=1e-2
    ), f"{__name__}: Jacobian mismatch!"


if __name__ == "__main__":
    test_backward_sh_color_to_position()
    test_backward_render_to_sh_color_tile()
    test_fused_ssim_backward()
    test_render_color_forward()
    test_fused_ssim_forward()
