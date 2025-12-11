import pytest
import os
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from gsplat.optimizers.torch_functions_forward import (
    blend_sh_colors_with_alphas,
    compute_sh_colors,
)
from gsplat.optimizers.sparse_newton import (
    cache_sh_derivative_functions,
    _backward_render_to_sh_colors,
    _backward_sh_colors_to_positions,
)
from gsplat.logger import create_logger

LOGGER = create_logger(name=os.path.basename(__file__), level="INFO")

mN = 7  # number of camera views
tH = 16  # tile size along height dimension
tW = 16  # tile size along width dimension
tK = 92  # number of gaussian splats
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


if __name__ == "__main__":
    test_backward_sh_colors_to_positions()
    test_backward_render_to_sh_colors()
