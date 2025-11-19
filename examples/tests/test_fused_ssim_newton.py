import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from typing import OrderedDict, Optional, Literal
from datasets.colmap import Parser as ColmapParser
from datasets.colmap import Dataset as ColmapDataset
from gsplat.rendering import rasterization
from gsplat.logger import create_logger
from gsplat.optimizers.sparse_newton import GSGroupSSIMLoss
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


def parse_fused_ssim_gauss_filter_1d(marco_str: Optional[str] = None):
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

    filter_1d = torch.tensor([x for _, x in sorted(matches)], dtype=torch.float32)
    return filter_1d


def generate_render_data_sample():
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
    )
    LOGGER.info(f"Finished rendering image-{img_id}.")

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
    return data


def test_fused_ssim_forward():
    """Test forward pass of Fused SSIM."""
    data = generate_render_data_sample()

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

    ssim_1_batch = (
        FusedSSIMMap.apply(0.01**2, 0.03**2, rd_imgs, gt_imgs, "valid", False)
        .mean(dim=[1, 2, 3])
        .cpu()
        .detach()
        .numpy()
    )
    assert np.allclose(
        ssim_1, ssim_1_batch, atol=1e-4, rtol=1e-1
    ), "Fused SSIM abnormal batch-mode behavior!"

    group_ssim = GSGroupSSIMLoss(
        gauss_filter_1d=parse_fused_ssim_gauss_filter_1d(),
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
    ssim_2_batch = group_ssim(gt_imgs, rd_imgs).cpu().detach().numpy()
    assert np.allclose(
        ssim_2, ssim_2_batch, atol=1e-4, rtol=1e-1
    ), f"Customized SSIM abnormal batch-mode behavior!"

    assert np.allclose(
        ssim_1, 1.0 - ssim_2, atol=1e-4, rtol=1e-1
    ), f"Fused and customized SSIM mismatch!"

    torch.cuda.empty_cache()


def test_fused_ssim_backward():
    """Test backward pass of Fused SSIM."""
    pass


if __name__ == "__main__":
    test_fused_ssim_forward()
    test_fused_ssim_backward()
