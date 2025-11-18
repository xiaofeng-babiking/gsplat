import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import torch
import cv2 as cv
import numpy as np
from scipy.optimize import curve_fit
from typing import OrderedDict, List, Literal
from datasets.colmap import Parser as ColmapParser
from datasets.colmap import Dataset as ColmapDataset
from gsplat.rendering import rasterization
from gsplat.logger import create_logger
from fused_ssim import fused_ssim
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


def fit_fused_ssim_gaussian_distribution(xdata: np.ndarray, ydata: np.ndarray):
    """Fit Gaussian Distribution sigma from fused-ssim."""

    gauss_filter_1d = lambda x, mu, sigma: (
        1.0
        / (np.sqrt(2 * np.pi * sigma**2))
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )

    popt, _ = curve_fit(gauss_filter_1d, xdata=xdata, ydata=ydata)

    opt_mu, opt_sigma = popt
    return opt_mu, opt_sigma


class CustomizedSSIM(torch.nn.Module):
    def __init__(
        self,
        ksize: int = 11,
        sigma=1.5,
        c1=0.01**2,
        c2=0.03**2,
        padding: Literal["valid", "same"] = "valid",
        reduction: Literal["mean", "sum", "none"] = "mean",
        *args,
        **kwargs,
    ) -> None:
        """Customized SSIM module initialization."""
        super().__init__(*args, **kwargs)
        self._ksize = ksize
        self._sigma = sigma
        self._c1 = c1
        self._c2 = c2
        self._padding = padding
        self._reduction = reduction

        self._kernel = torch.nn.Parameter(
            self.create_gaussian_kernel(), requires_grad=False
        )

    def create_gaussian_kernel(self) -> torch.FloatTensor:
        """Create a gaussian filter."""
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

        gauss_1d = torch.tensor([x for _, x in sorted(matches)], dtype=torch.float32)
        kernel = gauss_1d.reshape(-1, 1) * gauss_1d.reshape(1, -1)
        return kernel[None, None, ...]

    def forward(
        self, src_img: torch.FloatTensor, dst_img: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute SSIM between two images.

        Args:
            [1] src_img: torch.FloatTensor, source image, NCHW format.
            [2] dst_img: torch.FloatTensor, target image, NCHW format.

        Return:
            [1] ssim: torch.FloatTensor, SSIM score between two images.
        """
        assert (
            src_img.shape == dst_img.shape
        ), "source and target image must have the same shape!"

        groups = src_img.shape[1]

        src_mean = torch.conv2d(
            src_img,
            torch.tile(self._kernel, (groups, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=groups,
        )
        dst_mean = torch.conv2d(
            dst_img,
            torch.tile(self._kernel, (groups, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=groups,
        )

        src_var = (
            torch.conv2d(
                src_img**2,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - src_mean**2
        )
        dst_var = (
            torch.conv2d(
                dst_img**2,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - dst_mean**2
        )
        src_dst_covar = (
            torch.conv2d(
                src_img * dst_img,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - src_mean * dst_mean
        )

        a = self._c1 + src_mean**2 + dst_mean**2
        b = self._c2 + src_var + dst_var
        c = self._c1 + 2.0 * src_mean * dst_mean
        d = self._c2 + 2.0 * src_dst_covar
        ssim_map = (c * d) / (a * b)

        half_ksize = self._ksize // 2
        if self._padding == "valid":
            ssim_map = ssim_map[:, :, half_ksize:-half_ksize, half_ksize:-half_ksize]

        if self._reduction == "mean":
            ssim_score = torch.mean(ssim_map, dim=[1, 2, 3])
        elif self._reduction == "sum":
            ssim_score = torch.sum(ssim_map, dim=[1, 2, 3])
        elif self._reduction == "none":
            ssim_score = ssim_map
        else:
            raise ValueError(f"Invalid reduction mode: {self._reduction}!")
        return ssim_score


def test_fused_ssim_forward():
    parser = ColmapParser(
        data_dir=COLMAP_DATA_PATH,
        factor=RESOLUTION_FACTOR,
        normalize=True,
        test_every=8,
    )
    dataset = ColmapDataset(parser, split="test")
    n_imgs = len(dataset)
    LOGGER.info(f"Loaded {n_imgs} views from {os.path.basename(COLMAP_DATA_PATH)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splats = torch.load(GSPLAT_CHECKPOINT_FILE)["splats"]
    splats = OrderedDict({k: v.to(device) for k, v in splats.items()})

    sh_deg = int(np.sqrt(splats["sh0"].shape[1] + splats["shN"].shape[1])) - 1
    LOGGER.info(
        f"Loaded {len(splats['means'])} Splats from {os.path.basename(GSPLAT_CHECKPOINT_FILE)}."
    )

    # test fused ssim forward pass one-by-one
    ssim_sum = 0.0
    for img_cnt, data_item in enumerate(dataset):
        img_id = data_item["image_id"]
        gt_rgb = data_item["image"].to(device)[None, ...]
        gt_rgb = gt_rgb.float() / 255.0
        # image dimension format NCHW
        gt_rgb = torch.permute(gt_rgb, dims=[0, 3, 1, 2])

        img_h, img_w = gt_rgb.shape[-2:]

        # Dimension=[1, 3, 3]
        Ks = data_item["K"][None, ...]
        Ks = Ks.to(device)
        # Dimension=[1, 4, 4]
        viewmats = torch.linalg.inv(data_item["camtoworld"][None, ...])
        viewmats = viewmats.to(device)
        LOGGER.info(f"Loaded image-{img_id} with resolution={img_w}x{img_h}.")

        LOGGER.info(f"Start rendering image-{img_id} ({img_cnt + 1}/{n_imgs})...")
        rd_rgb, rd_a, rd_meta = rasterization(
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

        rd_rgb = torch.permute(rd_rgb, dims=[0, 3, 1, 2])
        rd_rgb = torch.clamp(rd_rgb, min=0.0, max=1.0)

        torch_ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11, k1=0.01, k2=0.03, sigma=1.5
        ).to(device)
        ssim_0 = torch_ssim(gt_rgb, rd_rgb)
        ssim_sum += ssim_0.item()
        LOGGER.info(f"Computed image-{img_id} SSIM={ssim_0.item():.4f}.")

        ssim_1 = fused_ssim(gt_rgb, rd_rgb, padding="valid", train=False).item()

        custom_ssim = CustomizedSSIM(ksize=11, sigma=1.5, padding="valid")
        custo_ssim = custom_ssim.to(device)
        ssim_2 = custom_ssim(gt_rgb, rd_rgb).item()

        assert (
            abs(ssim_1 - ssim_2) < 1e-4
        ), f"Fused VS Customized SSIM mismatch ({ssim_1:.6f} VS {ssim_2:.6f})!"
        # torch.cuda.empty_cache()

    ssim_avg = ssim_sum / n_imgs
    LOGGER.info(
        f"Average SSIM={ssim_avg:.4f} for Dataset={os.path.basename(COLMAP_DATA_PATH)}."
    )


if __name__ == "__main__":
    test_fused_ssim_forward()
