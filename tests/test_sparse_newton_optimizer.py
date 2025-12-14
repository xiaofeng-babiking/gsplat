import pytest
import os
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from gsplat.optimizers.torch_functions_forward import (
    blend_sh_colors_with_alphas,
    compute_sh_colors,
    compute_gaussian_weights_2d_tile,
    project_points_3d_to_2d,
)
from gsplat.optimizers.sparse_newton import (
    compute_blend_alphas,
    cache_sh_derivative_functions,
    _backward_render_to_sh_colors,
    _backward_sh_colors_to_positions,
    _backward_render_to_gaussians2d,
    _backward_gaussians2d_to_means2d,
    _backward_means2d_to_means3d,
    _backward_conics2d_to_covars2d,
)
from gsplat.logger import create_logger

LOGGER = create_logger(name=os.path.basename(__file__), level="INFO")

mN = 7  # number of camera views
tH = 16  # tile size along height dimension
tW = 16  # tile size along width dimension
tK = 12  # number of gaussian splats
mL = 3  # rotation degree of spherical harmonics
mC = 3  # number of RGB color channels
KWARGS = {
    "dtype": torch.float32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


def test_backward_render_to_sh_colors():
    """Test backward from render color to SH color."""
    name = "From_RENDER_To_SH_COLORS"

    alphas = torch.randn(size=[mN, tH, tW, tK], **KWARGS)

    sh_colors = torch.randn(size=[mN, tK, mC], **KWARGS)

    masks = torch.randint(low=0, high=2, size=[mN, tK], **KWARGS).bool()

    forward_fn = lambda x: blend_sh_colors_with_alphas(x, alphas, masks)[0]

    # Dim = [mN, tH, tW, mC, mN, tK, mC]
    jacob_auto = torch.autograd.functional.jacobian(forward_fn, sh_colors)
    # re-indexing autograd jacobian
    ns, cs = torch.meshgrid(
        [
            torch.arange(0, mN, **KWARGS).int(),
            torch.arange(0, mC, **KWARGS).int(),
        ],
        indexing="ij",
    )
    # Dim = [mN, tH, tW, mC, mN, tK, mC] -> [mN, tH, tW, tK, mC]
    jacob_auto = jacob_auto[ns, :, :, cs, ns, :, cs]
    jacob_auto = jacob_auto.permute([0, 2, 3, 4, 1])

    start = time.time()
    jacob_ours, _ = _backward_render_to_sh_colors(
        num_chs=mC,
        alphas=alphas,
        masks=masks,
    )
    end = time.time()
    assert torch.allclose(
        jacob_auto, jacob_ours, rtol=1e-3
    ), f"{name} jacobian wrong values!"
    assert jacob_ours.shape == (mN, tH, tW, tK, mC), f"{name} jacobian wrong dimension!"
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tH}, {tW}, {mC}], Input=[{mN}, {tK}, {mC}], "
        + f"Jacobian=[{mN}, {tH}, {tW} {tK}, {mC}], Hessian={None}, "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


def test_backward_sh_colors_to_positions():
    """Test backward from SH colors to positions."""
    name = "From_SH_COLORS_To_POSITIONS"

    view_mats = np.eye(4, dtype=np.float32)
    view_mats = np.tile(view_mats[None], [mN, 1, 1])

    view_quats = np.random.uniform(size=[mN, 4])
    view_rot_mats = np.stack(
        [ScipyRotation.from_quat(q).as_matrix() for q in view_quats], axis=0
    )
    view_mats[:, :3, :3] = view_rot_mats
    view_mats[:, :3, 3] = np.random.normal(size=[mN, 3]).astype(np.float32)
    view_mats = torch.tensor(view_mats, **KWARGS)

    means3d = torch.randn(size=[tK, 3], **KWARGS)

    sh_coeffs = torch.randn(size=[tK, (mL + 1) ** 2, mC], **KWARGS)

    sh_colors = compute_sh_colors(means3d, view_mats, sh_coeffs)
    assert sh_colors.shape == (mN, tK, mC)

    # Output: Dim = [mN, tK, mC]
    # Input:  Dim = [tK, 3]
    # Jacobian: Dim = [mN, tK, mC, tK, 3] -> [mN, tK, mC, 3]
    jacob_auto = torch.autograd.functional.jacobian(
        lambda x: compute_sh_colors(x, view_mats, sh_coeffs, False), means3d
    )
    idxs = [i for i in range(tK)]
    jacob_auto = jacob_auto[:, idxs, :, idxs, :]
    jacob_auto = jacob_auto.permute([1, 0, 2, 3])

    # Output: Dim = [mN, tK, mC]
    # Input:  Dim = [tK, 3]
    # Hessian: Dim = [mN, tK, mC, tK, 3, tK, 3] -> [mN, tK, mC, 3, 3]
    hess_auto = torch.autograd.functional.hessian(
        lambda x: compute_sh_colors(x, view_mats, sh_coeffs, False).sum(), means3d
    )
    idxs = [i for i in range(tK)]
    # Dim = [tK, 3, 3]
    hess_auto = hess_auto[idxs, :, idxs, :]

    start = time.time()
    sh_deriv_funcs = cache_sh_derivative_functions(sh_deg=mL)
    end = time.time()
    LOGGER.info(
        f"Cache SH derivative functions, Elapsed={float(end - start):.6f} seconds."
    )

    start = time.time()
    jacob_ours, hess_ours = _backward_sh_colors_to_positions(
        means3d, view_mats, sh_coeffs, sh_deriv_funcs
    )
    end = time.time()
    assert torch.allclose(
        jacob_auto, jacob_ours, rtol=1e-3
    ), f"{name} jacobian wrong values!"
    assert torch.allclose(
        hess_auto, hess_ours.sum(dim=[0, 2]), rtol=1e-3
    ), f"{name} hessian wrong values!"
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tK}, {mC}], Input=[{tK}, 3], "
        + f"Jacobian=[{mN}, {tK}, {mC}, 3], Hessian=[{mN}, {tK}, {mC}, 3, 3], "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


def test_backward_render_to_gaussians2d():
    """Test backward render to gaussian 2D weights."""
    name = "From_RENDER_to_GAUSSIANS2D"

    sh_colors = torch.rand(size=[mN, tK, mC], **KWARGS)

    gaussians2d = torch.rand(size=[mN, tH, tW, tK], **KWARGS)

    opacities = torch.rand(size=[tK], **KWARGS)

    forward_fn = lambda _gaussians2d: blend_sh_colors_with_alphas(
        sh_colors, alphas=_gaussians2d * opacities
    )[0]

    # Dim = [mN, tH, tW, mC, mN, tH, tW, tK]
    jacob_auto = torch.autograd.functional.jacobian(forward_fn, gaussians2d)
    ns, hs, ws = torch.meshgrid(
        [
            torch.arange(0, mN, **KWARGS).int(),
            torch.arange(0, tH, **KWARGS).int(),
            torch.arange(0, tW, **KWARGS).int(),
        ],
        indexing="ij",
    )
    jacob_auto = jacob_auto[ns, hs, ws, :, ns, hs, ws, :]

    alphas = gaussians2d * opacities
    alphas, blend_alphas = compute_blend_alphas(alphas)

    start = time.time()
    jacob_ours, _ = _backward_render_to_gaussians2d(
        sh_colors,
        opacities,
        alphas=gaussians2d * opacities,
        blend_alphas=blend_alphas,
    )
    end = time.time()
    assert torch.allclose(
        jacob_auto, jacob_ours, rtol=1e-2
    ), f"{name} jacobian wrong values!"
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tH}, {tW}, {mC}], Input=[{mN}, {tH}, {tW}, {tK}], "
        + f"Jacobian=[{mN}, {tH}, {tW}, {mC}, {tK}], Hessian={None}, "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


def test_backward_gaussians2d_to_means2d():
    """Test backward gaussian 2D weights to mean 2D locations."""
    name = "From_GAUSSIANS2D_To_MEANS2D"

    img_w = 1920
    img_h = 1080

    tile_x = int(np.random.randint(low=0, high=np.ceil(img_w / tW)))
    tile_y = int(np.random.randint(low=0, high=np.ceil(img_h / tH)))

    means2d_x = (torch.rand(size=[mN, tK], **KWARGS) + tile_x) * tW
    means2d_y = (torch.rand(size=[mN, tK], **KWARGS) + tile_y) * tH
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)
    conics2d = torch.rand(size=[mN, tK, 3], **KWARGS) + 1.0

    gausses2d, _ = compute_gaussian_weights_2d_tile(
        tile_x, tile_y, tW, tH, img_w, img_h, means2d, conics2d
    )

    # Output: Dim = [mN, tH, tW, tK]
    # Input: Dim = [mN, tK, 2]
    # Jacobian: Dim = [mN, tH, tW, tK, mN, tK, 2] -> [mN, tH, tW, tK, 2]
    jacob_auto = torch.autograd.functional.jacobian(
        lambda x: compute_gaussian_weights_2d_tile(
            tile_x, tile_y, tW, tH, img_w, img_h, x, conics2d
        )[0],
        means2d,
    )
    ns, ks = torch.meshgrid(
        [
            torch.arange(0, mN, **KWARGS).int(),
            torch.arange(0, tK, **KWARGS).int(),
        ],
        indexing="ij",
    )
    jacob_auto = jacob_auto[ns, :, :, ks, ns, ks, :]
    jacob_auto = jacob_auto.permute([0, 2, 3, 1, 4])
    jacob_mask = torch.logical_not(torch.isnan(jacob_auto) | torch.isinf(jacob_auto))

    # Hessian: Dim = [mN, tK, 2, mN, tK, 2]
    hess_auto = torch.autograd.functional.hessian(
        lambda x: compute_gaussian_weights_2d_tile(
            tile_x, tile_y, tW, tH, img_w, img_h, x, conics2d
        )[0].sum(),
        means2d,
    )
    # Hessian: Dim = [mN, tK, 2, 2]
    hess_auto = hess_auto[ns, ks, :, ns, ks, :]

    start = time.time()
    jacob_ours, hess_ours = _backward_gaussians2d_to_means2d(
        tile_x, tile_y, tW, tH, img_w, img_h, gausses2d, means2d, conics2d
    )
    end = time.time()
    assert torch.allclose(
        jacob_auto[jacob_mask], jacob_ours[jacob_mask], rtol=1e-3
    ), f"{name} jacobian wrong values!"
    # Dim [mN, tH, tW, tK, 2, 2] -> [mN, tK, 2, 2]
    hess_ours = hess_ours.sum(dim=[1, 2])
    hess_mask = torch.logical_not(
        torch.isnan(hess_auto)
        | torch.isinf(hess_auto)
        | (torch.abs(hess_auto) > 1e3)
        | torch.isnan(hess_ours)
        | torch.isinf(hess_ours)
        | (torch.abs(hess_ours) > 1e3)
    )
    assert torch.allclose(
        hess_auto[hess_mask], hess_ours[hess_mask], rtol=1e-1
    ), f"{name} hessian wrong values!"
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tH}, {tW}, {tK}], Input=[{mN}, {tK}, 2], "
        + f"Jacobian=[{mN}, {tH}, {tW}, {tK}, 2], Hessian=[{mN}, {tH}, {tW}, {tK}, 2, 2], "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


def test_backward_means2d_to_means3d():
    """Test backward mean 2D positions to mean 3D positions."""
    name = "From_MEANS2D_To_MEANS3D"

    img_w = 1920
    img_h = 1080

    fx, fy = 512.0, 512.0
    cx, cy = img_w / 2.0, img_h / 2.0

    cam_mats = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        **KWARGS,
    )
    cam_mats = cam_mats[None].repeat(mN, 1, 1)

    scale = 10.0

    means2d = torch.rand(size=[mN, tK, 2], **KWARGS)
    means2d[:, :, 0] = means2d[:, :, 0] * img_w
    means2d[:, :, 1] = means2d[:, :, 1] * img_h

    cam_zs = torch.rand(size=[mN, tK], **KWARGS) * scale + 1e-12
    cam_xs = (means2d[:, :, 0] - cx) / fx * cam_zs
    cam_ys = (means2d[:, :, 1] - cy) / fy * cam_zs

    # Dim = [mN, tK, 4]
    cam_means3d = torch.stack(
        [cam_xs, cam_ys, cam_zs, torch.ones([mN, tK], **KWARGS)], dim=-1
    )

    view_mats = np.eye(4, dtype=np.float32)
    view_mats = np.tile(view_mats[None], [mN, 1, 1])

    view_quats = np.random.uniform(size=[mN, 4])
    view_rot_mats = np.stack(
        [ScipyRotation.from_quat(q).as_matrix() for q in view_quats], axis=0
    )
    view_mats[:, :3, :3] = view_rot_mats
    view_mats[:, :3, 3] = np.random.normal(size=[mN, 3]).astype(np.float32) * scale
    view_mats = torch.tensor(view_mats, **KWARGS)

    # Dim = [mN, 4, 4] @ [mN, tK, 4] -> [mN, tK, 4]
    means3d = torch.einsum("ijk,ilk->ilj", torch.linalg.inv(view_mats), cam_means3d)
    means3d = means3d[:, :, :3] / (means3d[:, :, 3][..., None] + 1e-12)

    view_idx = int(np.random.randint(0, mN))
    means3d = means3d[view_idx]

    _, mean2d_proj = project_points_3d_to_2d(
        means3d, view_mats[view_idx][None], cam_mats[view_idx][None]
    )

    assert torch.allclose(mean2d_proj, means2d[view_idx], rtol=1e-3)

    # Output: Dim = [mN, tK, 2]
    # Input:  Dim = [tK, 3]
    # Jacobian: Dim = [mN, tK, 2, 3]
    jacob_auto = torch.autograd.functional.jacobian(
        lambda x: project_points_3d_to_2d(
            x, view_mats[view_idx][None], cam_mats[view_idx][None]
        )[1],
        means3d,
    )
    idxs = [i for i in range(tK)]
    jacob_auto = jacob_auto[:, idxs, :, idxs, :]
    jacob_auto = jacob_auto.permute([1, 0, 2, 3])

    hess_auto = torch.autograd.functional.hessian(
        lambda x: project_points_3d_to_2d(
            x, view_mats[view_idx][None], cam_mats[view_idx][None]
        )[1].sum(),
        means3d,
    )
    hess_auto = hess_auto[idxs, :, idxs, :]

    start = time.time()
    jacob_ours, hess_ours = _backward_means2d_to_means3d(
        view_mats[view_idx][None], cam_mats[view_idx][None], means3d
    )
    end = time.time()
    assert torch.allclose(
        jacob_auto, jacob_ours, rtol=1e-3
    ), f"{name} jacobian wrong values!"
    hess_ours = hess_ours.sum(dim=[0, 2])
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tK}, 2], Input=[{tK}, 3], "
        + f"Jacobian=[{mN}, {tK}, 2, 3], Hessian=[{mN}, {tK}, 2, 3, 3], "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


def test_backward_conics2d_to_covars2d():
    """Test backward 2D gaussian inverse covariance to covariance matrices."""
    name = "From_CONICS2D_TO_COVARS2D"

    covars2d = torch.rand(size=[mN, tK, 3], **KWARGS)
    covars2d_mat = torch.zeros(
        size=list(covars2d.shape[:-1]) + [2, 2],
        dtype=covars2d.dtype,
        device=covars2d.device,
    )
    covars2d_mat[..., 0, 0] = covars2d[..., 0]
    covars2d_mat[..., 0, 1] = covars2d[..., 1]
    covars2d_mat[..., 1, 0] = covars2d[..., 1]
    covars2d_mat[..., 1, 1] = covars2d[..., 2]

    conics2d_mat = torch.linalg.inv(covars2d_mat)
    conics2d = torch.zeros(size=list(conics2d_mat.shape[:-2]) + [3], **KWARGS)
    conics2d[..., 0] = conics2d_mat[..., 0, 0]
    conics2d[..., 1] = conics2d_mat[..., 0, 1]
    conics2d[..., 2] = conics2d_mat[..., 1, 1]

    jacob_auto = torch.autograd.functional.jacobian(
        lambda x: torch.linalg.inv(x), covars2d_mat
    )
    ns, ks = torch.meshgrid(
        [
            torch.arange(mN, **KWARGS).int(),
            torch.arange(tK, **KWARGS).int(),
        ],
        indexing="ij",
    )
    jacob_auto = jacob_auto[ns, ks, :, :, ns, ks, :, :]

    hess_auto = torch.autograd.functional.hessian(
        lambda x: torch.linalg.inv(x).sum(), covars2d_mat
    )
    hess_auto = hess_auto[ns, ks, :, :, ns, ks, :, :]

    start = time.time()
    jacob_ours, hess_ours = _backward_conics2d_to_covars2d(conics2d)
    end = time.time()
    hess_ours = hess_ours.sum(dim=[2, 3])
    assert torch.allclose(
        jacob_auto, jacob_ours, rtol=1e-3
    ), f"{name} jacobian wrong values!"
    assert torch.allclose(
        hess_auto, hess_ours, rtol=1e-3
    ), f"{name} hessian wrong values!"
    LOGGER.info(
        f"Backward={name}, "
        + f"Output=[{mN}, {tK}, 2, 2], Input=[{mN}, {tK}, 2, 2], "
        + f"Jacobian=[{mN}, {tK}, 2, 2, 2, 2], Hessian=[{mN}, {tK}, 2, 2, 2, 2, 2, 2], "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


if __name__ == "__main__":
    test_backward_conics2d_to_covars2d()
    test_backward_means2d_to_means3d()
    test_backward_gaussians2d_to_means2d()
    test_backward_render_to_gaussians2d()
    test_backward_sh_colors_to_positions()
    test_backward_render_to_sh_colors()
