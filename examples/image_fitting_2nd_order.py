import os
import sys
import math
import time
import random
import numpy as np
from enum import Enum
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from PIL import Image
from tqdm import tqdm
from gsplat.rendering import rasterization
from gsplat.optimizers.torch_functions_forward import (
    rasterize_to_pixels_tile_forward,
    get_tile_size,
)
from gsplat.optimizers.selective_adam import SelectiveAdam
from gsplat.logger import create_logger

LOGGER = create_logger(name=os.path.basename(__file__), level="INFO")

import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_string(
    "image",
    "/home/babiking/mnt/Datasets.writable-by-babiking/example/MooreThreads.png",
    "Image file path.",
)
gflags.DEFINE_integer("points", 2000, "Number of initial points.")
gflags.DEFINE_integer("degree", 3, "Spherical harmonics rotation degree.")
gflags.DEFINE_integer("width", 256, "Cropped image width.")
gflags.DEFINE_integer("height", 256, "Cropped image height.")
gflags.DEFINE_float("fov_x", 90.0, "FOV along X-axis in degrees.")
gflags.DEFINE_integer("steps", 9000, "Number of training iterations.")
gflags.DEFINE_string(
    "solver_type",
    "ADAM",
    "Solver type, 1st or 2nd order optimizer, Literal['ADAM', 'NEWTON'].",
)
gflags.DEFINE_string(
    "solver_level",
    "IMAGE",
    "Solver level, paramerater update granularity, Literal['IMAGE', 'TILE', 'PIXEL', 'RAY']",
)
gflags.DEFINE_float("learn_rate", 0.01, "Learning rate.")
gflags.DEFINE_float("sample_ratio", 1.0, "Sample ratio for multiple tiles.")
gflags.DEFINE_string(
    "output",
    "/home/babiking/mnt/Codebases.writable-by-babiking/gsplat/examples/image_fit_2nd_order",
    "Output path.",
)


class GSParameters(Enum):
    POSITION = 0
    ROTATION = 1  # orientations of each 3D gaussian ellipsoid
    SCALE = 2
    OPACITY = 3
    COLOR = 4
    VIEW = 5
    CAMERA = 6


class SolverLevel(Enum):
    IMAGE = 0
    TILE = 1
    PIXEL = 2  # infinite ray along pixel reprojection
    RAY = 3  # pixel with depth truncation


class SolverType(Enum):
    ADAM = 0
    NEWTON = 1


class SimpleTrainer:
    """Trains random 3DGS gaussians to fit an image."""

    def __init__(
        self,
        gt_img: torch.FloatTensor,
        num_pnts: int = 2000,
        fov_x: float = 90.0,
        sh_deg: int = 3,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self._device = device

        # data format NCHW
        self._gt_img = gt_img.to(device=self._device)
        self._num_chs = self._gt_img.shape[1]  # number of color channels

        self._num_pnts = num_pnts
        self._fov_x = fov_x / 180.0 * math.pi  # degrees to radians
        self._img_h, self._img_w = self._gt_img.shape[-2:]
        self._focal = 0.5 * float(self._img_w) / math.tan(0.5 * self._fov_x)

        self._sh_deg = sh_deg

        self._loss_fn = torch.nn.MSELoss()

        self._init_gs_params()

    def _init_gs_params(self):
        """Initialize random gaussian splats' parameters."""
        self._means3d = 2.0 * (
            torch.rand(self._num_pnts, 3, dtype=torch.float32, device=self._device)
            - 0.5
        )

        u = torch.rand(self._num_pnts, 1, dtype=torch.float32, device=self._device)
        v = torch.rand(self._num_pnts, 1, dtype=torch.float32, device=self._device)
        w = torch.rand(self._num_pnts, 1, dtype=torch.float32, device=self._device)
        self._quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )

        self._scales = torch.rand(
            self._num_pnts, 3, dtype=torch.float32, device=self._device
        )
        self._opacities = torch.ones(
            (self._num_pnts), dtype=torch.float32, device=self._device
        )
        self._sh_coeffs = torch.rand(
            (self._num_pnts, (self._sh_deg + 1) ** 2, self._num_chs),
            dtype=torch.float32,
            device=self._device,
        )

        self._view_mats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )[None]

        self._cam_mats = torch.tensor(
            [
                [self._focal, 0.0, self._img_w / 2],
                [0.0, self._focal, self._img_h / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )[None]

        self._params = torch.nn.ParameterDict(
            {
                GSParameters.POSITION.name: self._means3d.requires_grad_(True),
                GSParameters.ROTATION.name: self._quats.requires_grad_(True),
                GSParameters.SCALE.name: self._scales.requires_grad_(True),
                GSParameters.OPACITY.name: self._opacities.requires_grad_(True),
                GSParameters.COLOR.name: self._sh_coeffs.requires_grad_(True),
                GSParameters.VIEW.name: self._view_mats.requires_grad_(False),
                GSParameters.CAMERA.name: self._cam_mats.requires_grad_(False),
            }
        )

    def train(
        self,
        steps: int,
        out_path: str,
        tile_size: int = 16,
        init_lr: float = 0.01,
        groups: List[GSParameters] = [
            GSParameters.POSITION,
            GSParameters.ROTATION,
            GSParameters.SCALE,
            GSParameters.OPACITY,
            GSParameters.COLOR,
        ],
        solver_level: SolverLevel = SolverLevel.IMAGE,
        solver_type: SolverType = SolverType.ADAM,
        sample_ratio: float = 1.0,
    ):
        """Train."""
        os.makedirs(out_path, exist_ok=True)
        writer = tensorboard.SummaryWriter(
            log_dir=os.path.join(out_path, f"{solver_level.name}-{solver_type.name}")
        )

        n_splats = self._means3d.shape[0]
        device = self._means3d.device

        optimizers = (
            {
                group.name: (
                    SelectiveAdam(
                        params=[
                            {
                                "params": self._params[group.name],
                                "lr": init_lr,
                                "name": group.name,
                            }
                        ],
                        eps=1e-8,
                        betas=(0.9, 0.999),
                    )
                )
                for group in groups
            }
            if solver_type != SolverType.NEWTON
            else None
        )

        for i in range(steps + 1):
            if i == 0:
                writer.add_image(
                    "Groundtruth", self._gt_img[0], global_step=i, dataformats="CHW"
                )
                writer.add_scalar("TileSize", tile_size, global_step=i)
                writer.add_scalar("LearnRate", init_lr, global_step=i)

            rd_img, _, rd_meta = rasterization(
                means=self._params[GSParameters.POSITION.name],
                quats=self._params[GSParameters.ROTATION.name]
                / self._params[GSParameters.ROTATION.name].norm(dim=-1, keepdim=True),
                scales=self._params[GSParameters.SCALE.name],
                opacities=torch.sigmoid(self._params[GSParameters.OPACITY.name]),
                colors=self._params[GSParameters.COLOR.name],
                viewmats=self._params[GSParameters.VIEW.name],
                Ks=self._params[GSParameters.CAMERA.name],
                width=self._img_w,
                height=self._img_h,
                sh_degree=self._sh_deg,
                tile_size=tile_size,
                packed=True,
            )
            rd_img = rd_img.permute([0, 3, 1, 2])
            if i % 300 == 0:
                writer.add_image("Render", rd_img[0], global_step=i, dataformats="CHW")

            loss = self._loss_fn(self._gt_img, rd_img)

            LOGGER.info(
                f"Step={i}, Image={self._img_w}x{self._img_h}, Loss={loss.item():.6f}"
            )
            writer.add_scalar("Loss", loss, global_step=i)

            if solver_type != SolverType.NEWTON:
                [optimizers[group.name].zero_grad() for group in groups]
                loss.backward()

            if solver_level == SolverLevel.IMAGE:
                if solver_type == SolverType.ADAM:
                    [
                        optimizers[group.name].step(
                            visibility=torch.ones(
                                [n_splats], dtype=torch.bool, device=device
                            )
                        )
                        for group in groups
                    ]
                else:
                    raise NotImplementedError(
                        f"NOT support solver={solver_type} (ADAM only) for level={solver_level}!"
                    )
            elif solver_level == SolverLevel.TILE or solver_level == SolverLevel.PIXEL:
                tile_w = rd_meta["tile_width"]
                tile_h = rd_meta["tile_height"]

                n_tiles = tile_h * tile_w
                tile_idxs = random.sample(
                    list(range(n_tiles)), k=int(n_tiles * sample_ratio)
                )

                isect_offsets = rd_meta["isect_offsets"]
                isect_offsets = isect_offsets[0].flatten()

                flatten_ids = rd_meta["flatten_ids"]

                for tile_idx in tqdm(
                    tile_idxs, desc=f"loop over step={i:04d} tiles..."
                ):
                    tile_y = tile_idx // tile_w
                    tile_x = tile_idx % tile_w

                    isect_start = isect_offsets[tile_idx]
                    isect_end = (
                        isect_offsets[tile_idx + 1]
                        if tile_idx < tile_h * tile_w - 1
                        else len(flatten_ids)
                    )

                    if isect_start >= isect_end:
                        continue

                    flat_idxs = flatten_ids[isect_start:isect_end]

                    tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
                        tile_x, tile_y, tile_size, tile_size, self._img_w, self._img_h
                    )

                    tile_gt_img = self._gt_img[
                        :,
                        :,
                        tile_ymin : tile_ymin + crop_h,
                        tile_xmin : tile_xmin + crop_w,
                    ]

                    tile_rd_img, _, rd_meta = rasterization(
                        means=self._params[GSParameters.POSITION.name][flat_idxs],
                        quats=self._params[GSParameters.ROTATION.name][flat_idxs]
                        / self._params[GSParameters.ROTATION.name][flat_idxs].norm(
                            dim=-1, keepdim=True
                        ),
                        scales=self._params[GSParameters.SCALE.name][flat_idxs],
                        opacities=torch.sigmoid(
                            self._params[GSParameters.OPACITY.name][flat_idxs]
                        ),
                        colors=self._params[GSParameters.COLOR.name][flat_idxs],
                        viewmats=self._params[GSParameters.VIEW.name],
                        Ks=self._params[GSParameters.CAMERA.name],
                        width=self._img_w,
                        height=self._img_h,
                        sh_degree=self._sh_deg,
                        tile_size=tile_size,
                        packed=True,
                    )
                    tile_rd_img = tile_rd_img.permute([0, 3, 1, 2])
                    tile_rd_img = tile_rd_img[
                        :,
                        :,
                        tile_ymin : tile_ymin + crop_h,
                        tile_xmin : tile_xmin + crop_w,
                    ]

                    visibles = torch.zeros(
                        size=[n_splats], dtype=torch.bool, device=device
                    )
                    visibles[flat_idxs] = True

                    if solver_type == SolverType.ADAM:
                        tile_loss = self._loss_fn(tile_rd_img, tile_gt_img)
                        for group in groups:
                            optimizers[group.name].zero_grad()
                        tile_loss.backward()

                        for group in groups:
                            optimizers[group.name].step(visibility=visibles)
                            optimizers[group.name].state[self._params[group.name]][
                                "step"
                            ] -= 1.0

                # NOTE. step only increase for each iteration, NOT each tile
                for group in groups:
                    optimizers[group.name].state[self._params[group.name]][
                        "step"
                    ] += 1.0

            elif solver_level == SolverLevel.RAY:
                raise NotImplementedError(
                    f"Extra depth prior required for level={solver_level}!"
                )

            #         render_fn = lambda x: rasterize_to_pixels_tile_forward(
            #             tile_x,
            #             tile_y,
            #             tile_size,
            #             tile_size,
            #             self._img_w,
            #             self._img_h,
            #             (
            #                 x
            #                 if group == GSParameters.POSITION
            #                 else self._params[GSParameters.POSITION.name][flat_idxs]
            #             ),
            #             (
            #                 x / x.norm(dim=-1, keepdim=True)
            #                 if group == GSParameters.ROTATION
            #                 else (
            #                     self._params[GSParameters.ROTATION.name][flat_idxs]
            #                     / self._params[GSParameters.ROTATION.name][
            #                         flat_idxs
            #                     ].norm(dim=-1, keepdim=True)
            #                 )
            #             ),
            #             (
            #                 x
            #                 if group == GSParameters.SCALE
            #                 else self._params[GSParameters.SCALE.name][flat_idxs]
            #             ),
            #             torch.sigmoid(
            #                 x
            #                 if group == GSParameters.OPACITY
            #                 else self._params[GSParameters.OPACITY.name][flat_idxs]
            #             ),
            #             (
            #                 x
            #                 if group == GSParameters.COLOR
            #                 else self._params[GSParameters.COLOR.name][flat_idxs]
            #             ),
            #             self._params[GSParameters.CAMERA.name],
            #             self._params[GSParameters.VIEW.name],
            #         )

        writer.close()


def read_image_file(
    img_file: str, img_w: Optional[int] = None, img_h: Optional[int] = None
):
    img = Image.open(img_file, mode="r").convert("RGB")

    raw_w, raw_h = img.size

    if img_w is None or img_h is None:
        img_w, img_h = img.size

    rem_w = raw_w % img_w if raw_w > img_w else 0
    rem_h = raw_h % img_h if raw_h > img_h else 0

    crop_l = rem_w // 2
    crop_t = rem_h // 2
    crop_r = raw_w - (rem_w - crop_l)
    crop_b = raw_h - (rem_h - crop_t)

    img = img.crop((crop_l, crop_t, crop_r, crop_b))
    assert img.size[0] % img_w == 0
    assert img.size[1] % img_h == 0

    img = img.resize((img_w, img_h))

    img_tensor = torch.tensor(
        np.array(img, dtype=np.float32) / 255.0, dtype=torch.float32
    )
    img_tensor = img_tensor.reshape([img_h, img_w, -1])
    # NCHW, i.e. [1, H, W, C]
    img_tensor = img_tensor.permute([2, 0, 1])[None]
    return img_tensor


def main():
    FLAGS(sys.argv)

    img_file = FLAGS.image
    img_w = FLAGS.width
    img_h = FLAGS.height
    num_pnts = FLAGS.points
    fov_x = FLAGS.fov_x
    sh_deg = FLAGS.degree
    solver_type = SolverType[FLAGS.solver_type.upper()]
    solver_level = SolverLevel[FLAGS.solver_level.upper()]
    init_lr = FLAGS.learn_rate
    sample_ratio = FLAGS.sample_ratio
    LOGGER.info(f"File={img_file}.")
    LOGGER.info(
        f"Image={img_w}x{img_h}, #Splats={num_pnts}, FOV={fov_x:.2f}°, Degree={sh_deg}."
    )
    LOGGER.info(
        f"SolverLevel={solver_level}, SolverType={solver_type}, "
        + f"LearnRate={init_lr:.4f}, SampleRatio={sample_ratio:.4f}."
    )

    gt_img = read_image_file(img_file, img_w, img_h)

    trainer = SimpleTrainer(
        gt_img=gt_img, num_pnts=num_pnts, fov_x=fov_x, sh_deg=sh_deg
    )
    trainer.train(
        steps=FLAGS.steps,
        out_path=FLAGS.output,
        init_lr=init_lr,
        solver_level=solver_level,
        solver_type=solver_type,
        sample_ratio=sample_ratio,
    )


if __name__ == "__main__":
    main()
