import os
import math
import time
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm
from typing import Literal
from datasets.colmap import Parser as ColmapParser
from datasets.colmap import Dataset as ColmapDataset
from gsplat.rendering import rasterization
from gsplat.logger import create_logger
from collections import OrderedDict
from torchmetrics import StructuralSimilarityIndexMeasure
from gsplat.optimizers.torch_functions_forward import (
    rasterize_to_pixels_tile_forward,
    get_tile_size,
)

LOGGER = create_logger(
    name=os.path.splitext(os.path.basename(__file__))[0], level="INFO"
)

import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_string(
    "data_path",
    "/home/babiking/mnt/Datasets.writable-by-babiking/mipnerf360/garden",
    "Path to COLMAP data.",
)
gflags.DEFINE_string(
    "checkpoint_file",
    "/home/babiking/mnt/Codebases.writable-by-babiking/gsplat/examples/results/garden_default_adam/ckpts/ckpt_6999_rank0.pt",
    "Checkpoint file path of trained model.",
)
gflags.DEFINE_integer(
    "resolution_factor", 4, "Resolution factor for downsampling images."
)
gflags.DEFINE_string("data_split", "test", "Dataset split train or test.")
gflags.DEFINE_integer("test_every", 8, "Test every this many frames.")
gflags.DEFINE_integer("train_steps", 10, "Number of train iteration steps.")
gflags.DEFINE_float("noise_level", 0.5, "Random noise level.")
gflags.DEFINE_string("output_path", "./outputs", "Path to save training outputs.")


def load_3dgs_checkpoint(
    chkpt_file: str,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    """Load a checkpoint file and return its contents."""
    splats = torch.load(chkpt_file)["splats"]
    splats = OrderedDict({k: v.to(device) for k, v in splats.items()})

    means3d = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    sh_coeffs = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return means3d, quats, scales, opacities, sh_coeffs


def load_colmap_dataset(
    data_path: str,
    resolution_factor: int,
    data_split: Literal["train", "test"] = "test",
    test_every: int = 8,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    """Load COLMAP dataset."""
    parser = ColmapParser(
        data_dir=data_path,
        factor=resolution_factor,
        normalize=True,
        test_every=test_every,
    )
    dataset = ColmapDataset(parser, split=data_split)

    img_ids = []
    gt_imgs = []
    cam_mats = []
    view_mats = []
    for data_item in dataset:
        img_id = data_item["image_id"]
        img_ids.append(img_id)

        gt_img = data_item["image"].to(device)[None, ...]
        gt_img = gt_img.float() / 255.0
        # image dimension format NCHW
        gt_img = torch.permute(gt_img, dims=[0, 3, 1, 2])
        gt_imgs.append(gt_img)

        # Dimension=[1, 3, 3]
        cam_mat = data_item["K"][None, ...]
        cam_mat = cam_mat.to(device)
        cam_mats.append(cam_mat)

        # Dimension=[1, 4, 4]
        view_mat = data_item["camtoworld"][None, ...]
        view_mat = view_mat.to(device)
        view_mats.append(view_mat)

    img_ids = torch.tensor(img_ids, dtype=torch.int32)
    gt_imgs = torch.cat(gt_imgs, dim=0)
    cam_mats = torch.cat(cam_mats, dim=0)
    view_mats = torch.linalg.inv(torch.cat(view_mats, dim=0))
    return img_ids, gt_imgs, cam_mats, view_mats


def train():
    FLAGS(sys.argv)

    out_path = FLAGS.output_path
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_ids, gt_imgs, cam_mats, view_mats = load_colmap_dataset(
        data_path=FLAGS.data_path,
        resolution_factor=FLAGS.resolution_factor,
        data_split=FLAGS.data_split,
        test_every=FLAGS.test_every,
        device=device,
    )

    _, _, img_h, img_w = gt_imgs.shape
    LOGGER.info(f"Loda dataset from {len(img_ids)} views with size={img_w}x{img_h}.")

    means3d, quats, scales, opacities, sh_coeffs = load_3dgs_checkpoint(
        chkpt_file=FLAGS.checkpoint_file,
        device=device,
    )
    sh_deg = int(math.sqrt(sh_coeffs.shape[-2])) - 1
    LOGGER.info(
        f"Load trained 3DGS model with {means3d.shape[0]} splats with degree={sh_deg}."
    )

    means3d = means3d + torch.randn_like(means3d) * FLAGS.noise_level
    LOGGER.info(f"Noise ({FLAGS.noise_level}) added into parameters.")

    view_idx = random.choice(list(range(len(img_ids))))

    gt_img = gt_imgs[view_idx].unsqueeze(0)
    rd_img, _, rd_meta = rasterization(
        means=means3d,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        Ks=cam_mats[view_idx].unsqueeze(0),
        viewmats=view_mats[view_idx].unsqueeze(0),
        width=img_w,
        height=img_h,
        sh_degree=sh_deg,
        packed=True,
    )
    rd_img = torch.clamp(rd_img, min=0.0, max=1.0)
    rd_img = rd_img.permute([0, 3, 1, 2])

    l2_loss_func = lambda x, y: torch.mean((x - y) ** 2)
    ssim_loss_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    init_l2_loss = l2_loss_func(rd_img, gt_img)
    init_ssim_loss = ssim_loss_func(rd_img, gt_img)
    LOGGER.info(
        f"Image Initial Loss: L2={init_l2_loss.item():.6f}, SSIM={init_ssim_loss.item():.6f}."
    )

    gauss_ids = rd_meta["gaussian_ids"]
    means3d = means3d[gauss_ids]
    quats = quats[gauss_ids]
    scales = scales[gauss_ids]
    opacities = opacities[gauss_ids]
    sh_coeffs = sh_coeffs[gauss_ids]
    isect_offsets = rd_meta["isect_offsets"]
    flatten_ids = rd_meta["flatten_ids"]
    LOGGER.info(f"View={view_idx}, #Splats={len(gauss_ids)}.")

    tile_size = rd_meta["tile_size"]
    tile_w = int(math.ceil(img_w / tile_size))
    tile_h = int(math.ceil(img_h / tile_size))
    assert isect_offsets.shape == (1, tile_h, tile_w)
    isect_offsets = isect_offsets[0].flatten()

    for tile_x in range(tile_w):
        for tile_y in range(tile_h):
            tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
                tile_x, tile_y, tile_size, img_w, img_h
            )

            tile_gt_img = gt_img[
                :, :, tile_ymin : tile_ymin + crop_h, tile_xmin : tile_xmin + crop_w
            ]

            tile_idx = tile_y * tile_w + tile_x
            isect_start = isect_offsets[tile_idx]
            isect_end = (
                isect_offsets[tile_idx + 1]
                if tile_idx < tile_w * tile_h - 1
                else len(flatten_ids)
            )
            flat_idxs = flatten_ids[isect_start:isect_end]

            tile_means3d = means3d[flat_idxs]
            tile_quats = quats[flat_idxs]
            tile_scales = scales[flat_idxs]
            tile_opacities = opacities[flat_idxs]
            tile_sh_coeffs = sh_coeffs[flat_idxs]

            render_func = lambda _tile_means: rasterize_to_pixels_tile_forward(
                tile_x,
                tile_y,
                tile_size,
                img_w,
                img_h,
                _tile_means,
                tile_quats,
                tile_scales,
                tile_opacities,
                tile_sh_coeffs,
                cam_mats[view_idx][None],
                view_mats[view_idx][None],
            )

            forward_func = lambda _tile_means: l2_loss_func(
                render_func(_tile_means)[0], tile_gt_img
            )

            n_splats = len(flat_idxs)
            LOGGER.info(
                f"View={view_idx}, Tile=({tile_x}, {tile_y}), #Splats={n_splats}."
            )

            for i in tqdm(
                range(FLAGS.train_steps),
                total=FLAGS.train_steps,
                desc=f"train tile=({tile_x}, {tile_y}) 2nd-order...",
            ):
                l2_loss = forward_func(tile_means3d)
                LOGGER.info(
                    f"View={view_idx}, Tile=({tile_x}, {tile_y}), Step={i}, L2={l2_loss:.4f}."
                )

                start = time.time()
                # Means3d Dim = [tK, 3]
                jacob = torch.autograd.functional.jacobian(forward_func, tile_means3d)
                jacob = jacob.reshape([n_splats * 3, 1])
                end = time.time()
                jacob_elapsed = float(end - start)
                LOGGER.info(
                    f"View={view_idx}, Tile=({tile_x}, {tile_y}), Step={i}, Jacobian={jacob_elapsed:.4f} seconds."
                )

                start = time.time()
                hess = torch.autograd.functional.hessian(forward_func, tile_means3d)
                hess = hess.reshape([n_splats * 3, n_splats * 3])
                end = time.time()
                hess_elapsed = float(end - start)
                LOGGER.info(
                    f"View={view_idx}, Tile=({tile_x}, {tile_y}), Step={i}, Hessian={hess_elapsed:.4f} seconds."
                )

                assert torch.any(torch.abs(hess) >= 0.0)
                
                start = time.time()
                hess_inv = torch.linalg.pinv(hess)
                end = time.time()
                hess_inv_elapsed = float(end - start)
                LOGGER.info(
                    f"View={view_idx}, Tile=({tile_x}, {tile_y}), Step={i}, InverseHessian={hess_inv_elapsed:.4f} seconds."
                )

                delta_means3d = -torch.matmul(hess_inv, jacob).reshape([n_splats, 3])
                tile_means3d = tile_means3d + delta_means3d


if __name__ == "__main__":
    train()
