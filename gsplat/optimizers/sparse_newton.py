import math
import torch
from typing import Optional, Callable, Dict, List
from sympy import symbols, simplify, diff, lambdify

SPHERICAL_HARMONICS_CONSTANTS = [
    +0.2820947917738781,  # Y(0, 0), 1.0 / 2.0 * sqrt(1.0 / pi)
    -0.4886025119029199,  # Y(1, -1), sqrt(3.0 / (4.0 * pi))
    +0.4886025119029199,  # Y(1, 0), sqrt(3.0 / (4.0 * pi))
    -0.4886025119029199,  # Y(1, 1), sqrt(3.0 / (4.0 * pi))
    +1.0925484305920792,  # Y(2, -2), 1.0 * / 2.0 * sqrt(15.0 / pi)
    -1.0925484305920792,  # Y(2, -1), 1.0 * / 2.0 * sqrt(15.0 / pi)
    +0.3153915652525201,  # Y(2, 0), 1.0 * / 4.0 * sqrt(5.0 / pi)
    -1.0925484305920792,  # Y(2, 1), 1.0 * / 2.0 * sqrt(15.0 / pi)
    +0.5462742152960395,  # Y(2, 2), 1.0 * / 4.0 * sqrt(15.0 / pi)
    -0.5900435899266435,  # Y(3, -3), 1.0 * / 4.0 * sqrt(35.0 / (2.0 * pi))
    +2.8906114426405540,  # Y(3, -2), 1.0 * / 2.0 * sqrt(105.0 / pi)
    -0.4570457994644658,  # Y(3, -1), 1.0 * / 4.0 * sqrt(21.0 / (2.0 * pi))
    +0.3731763325901154,  # Y(3, 0), 1.0 * / 4.0 * sqrt(7.0 / pi)
    -0.4570457994644658,  # Y(3, 1), 1.0 * / 4.0 * sqrt(21.0 / (2.0 * pi))
    +1.4453057213202770,  # Y(3, 2), 1.0 * / 4.0 * sqrt(105.0 / pi)
    -0.5900435899266435,  # Y(3, 3), 1.0 * / 4.0 * sqrt(35.0 / (2.0 * pi))
]


def _backward_render_to_sh_colors(
    num_chs: int,
    alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    masks: Optional[torch.Tensor] = None,  # Dim = [mN, tK]
):
    """Backward pass of rendered pixels w.r.t spherical harmonics colors.

    Args:
        [1] num_chs: Number of color channels, int.
        [2] alphas: Pixelwise alphas within a tile, Dim=[mN, tH, tW, tK].
        [3] masks: Valid splats maks, Dim = [mN, tK].

    Return:
        [1] jacob: [mN, tH, tK, mC]
            For each pixel (i, y, x), render = (R, G, B), sh_color = (R, G, B)
        [2] hess: [mN, tH, tK, mC, mC]

    For each pixel (m, n) within this tile,
        RGB = SUM_k^{tK}(G(k) * σ(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - G(j) * σ(j)))

        ∂(RGB) / ∂(SH(k)) = G(k) * σ(k) * CUMPROD_j^{k - 1}(1.0 - G(j) * σ(j)))
            where, alphas(k) = G(k) * σ(k)

        ∂²(RGB) / ∂(SH(k))² = 0.0
    """
    alphas = torch.clamp_max(alphas, 0.9999)

    # Dim = [mN, tH, tW, tK]
    blend_alphas = torch.cumprod(1.0 - alphas, dim=-1)
    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
    blend_alphas[:, :, :, 0] = 1.0

    # Output:  Dim = [mN, tH, tW, mC]
    # Input:   Dim = [tK, mC]
    # Jacobian Dim = [mN, tH, tW, mC, tK, mC] -> [mN, tH, tW, tK, mC]
    jacob = (
        alphas
        * blend_alphas
        * (masks[:, None, None, :].float() if masks is not None else 1.0)
    )
    jacob = jacob[:, :, :, :, None].repeat([1, 1, 1, 1, num_chs])

    hess = None
    return jacob, hess


def cache_sh_derivative_functions(sh_deg: int):
    """Cache derivative functions of spherical harmonics."""
    assert sh_deg <= 3, "Only support Spherical Harmonics rotation degree <= 3."

    x, y, z = symbols("x y z")

    r = (x**2 + y**2 + z**2) ** (1 / 2)

    # Reference:
    #   [1] https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    sh_exprs = [
        1.0,  # Y(0, 0)
        y / r,  # Y(1, -1)
        z / r,  # Y(1, 0)
        x / r,  # Y(1, 1)
        x * y / r**2,  # Y(2, -2)
        y * z / r**2,  # Y(2, -1)
        (3 * z**2 - r**2) / r**2,  # Y(2, 0)
        x * z / r**2,  # Y(2, 1)
        (x**2 - y**2) / r**2,  # Y(2, 2)
        y * (3 * x**2 - y**2) / r**3,  # Y(3, -3)
        x * y * z / r**3,  # Y(3, -2)
        y * (5 * z**2 - r**2) / r**3,  # Y(3, -1)
        (5 * z**3 - 3 * z * r**2) / r**3,  # Y(3, 0)
        x * (5 * z**2 - r**2) / r**3,  # Y(3, 1)
        (x**2 - y**2) * z / r**3,  # Y(3, 2)
        x * (x**2 - 3 * y**2) / r**3,  # Y(3, 3)
    ]

    sh_deriv_funcs = []
    for i in range((sh_deg + 1) ** 2):
        sh_expr = sh_exprs[i]

        sh_deriv_dict = {"index": i}

        # 1st order derivatives
        for u in [x, y, z]:
            du_expr = simplify(diff(sh_expr, u))
            sh_deriv_dict[f"d{u}"] = lambdify(
                [x, y, z],
                du_expr,
                modules=["torch"],
            )

        # 2nd order derivatives
        for u, v in [(x, x), (x, y), (x, z), (y, y), (y, z), (z, z)]:
            dudv_expr = simplify(diff(sh_expr, u, v))
            sh_deriv_dict[f"d{u}d{v}"] = lambdify(
                [x, y, z],
                dudv_expr,
                modules=["torch"],
            )

        sh_deriv_funcs.append(sh_deriv_dict)
    return sh_deriv_funcs


def _backward_sh_colors_to_positions(
    means3d: torch.Tensor,  # Dim = [tK, 3]
    view_mats: torch.Tensor,  # Dim = [mN, 4, 4]
    sh_coeffs: torch.Tensor,  # Dim = [tK, (L+1)^2, mC]
    sh_deriv_funcs: List[Dict[str, Callable]],  # Dim = [(L+1)^2]
    sh_consts: Optional[torch.Tensor] = None,  # Dim = [(L+1)^2]
    masks: Optional[torch.BoolTensor] = None,  # Dim = [tK]
):
    """Backward from spherical colors to positions."""
    sh_deg = int(math.sqrt(sh_coeffs.shape[-2])) - 1
    assert sh_deg <= 3, "Only support Spherical Harmonics rotation degree <= 3."

    if sh_consts is None:
        sh_consts = torch.tensor(
            SPHERICAL_HARMONICS_CONSTANTS[: ((sh_deg + 1) ** 2)],
            dtype=sh_coeffs.dtype,
            device=sh_coeffs.device,
        )

    # camera positions in world space
    cam_cents = torch.linalg.inv(view_mats)[:, :3, 3]

    view_dirs = means3d[None, :, :] - cam_cents[:, None, :]  # Dim = [mN, mK, 3]

    # REFERENCE:
    #   [1] https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    # SH-RGB = SUM_l_{0}^{L}(SUM_m_{-l}^{l}(C(l, m) * Y(x, y, z, l, m)))
    #   where,
    #       1. l - rotation degree 0 <= l <= L
    #       2. m - rotation order -l <= m <= l
    #       3. (x, y, z) - unit vector from camera center to splat position
    #               * (x, y, z) = means3d - cam_center
    #       4. C(l, m) - spherical harmonics coefficient
    #       5. Y(x, y, z, l, m) - spherical harmonics function

    # ∂(SH-RGB) / ∂(means3d) =
    #    (∂c / ∂Y) @ (∂Y / ∂r) @ (∂r / ∂p)

    n_views = view_mats.shape[0]
    n_splats = means3d.shape[0]

    # SH-RGB            Dim = [mN, tK, mC]
    # SH-Value          Dim = [mN, tK, (L+1)^2]
    # SH-coefficients   Dim = [1,  tK, (L+1)^2, mC]
    # Jacobian-RGB-to-Value Dim = [mN, tK, mC, (L+1)^2]
    jacob_c_to_v = sh_coeffs.permute([0, 2, 1])
    jacob_c_to_v = jacob_c_to_v[None].repeat([n_views, 1, 1, 1])

    # SH-Value          Dim = [mN, tK, (L+1)^2]
    # Directions        Dim = [mN, tK, 3]
    # SH-Value-to-Direction-jacobian Dim = [mN, tK, (L+1)^2, 3]
    x = view_dirs[:, :, 0]
    y = view_dirs[:, :, 1]
    z = view_dirs[:, :, 2]

    jacob_v_to_r = torch.zeros(
        size=[n_views, n_splats, (sh_deg + 1) ** 2, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    for i in range((sh_deg + 1) ** 2):
        sh_deriv_dict = sh_deriv_funcs[i]
        assert sh_deriv_dict["index"] == i

        jacob_v_to_r[:, :, i, 0] = sh_deriv_dict["dx"](x, y, z)
        jacob_v_to_r[:, :, i, 1] = sh_deriv_dict["dy"](x, y, z)
        jacob_v_to_r[:, :, i, 2] = sh_deriv_dict["dz"](x, y, z)
    # Dim = [mN, tK, (L+1)^2, 3] * [1, 1, (L+1)^2, 1]
    jacob_v_to_r = jacob_v_to_r * sh_consts[None, None, :, None]
    # Output: SH-RGB,  Dim = [mN, tK, mC]
    # Input:  means3d, Dim = [tK, 3]
    # Jacobian: Dim = [mN, tK, mC, 3]
    jacob = torch.einsum("ijkl,ijlm->ijkm", jacob_c_to_v, jacob_v_to_r)

    # ∂²(SH-RGB) / ∂(means3d)²
    # = (∂c / ∂Y) @ (∂²Y / ∂p²) s.t. ∂²c / ∂Y² = 0
    # = (∂c / ∂Y) @ (∂²Y / ∂r²) s.t. ∂²r / ∂p² = 0
    hess_v_to_r = torch.zeros(
        [n_views, n_splats, (sh_deg + 1) ** 2, 3, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    for i in range((sh_deg + 1) ** 2):
        sh_deriv_dict = sh_deriv_funcs[i]
        assert sh_deriv_dict["index"] == i

        dxdx = sh_deriv_dict["dxdx"](x, y, z)
        dxdy = sh_deriv_dict["dxdy"](x, y, z)
        dxdz = sh_deriv_dict["dxdz"](x, y, z)
        dydy = sh_deriv_dict["dydy"](x, y, z)
        dydz = sh_deriv_dict["dydz"](x, y, z)
        dzdz = sh_deriv_dict["dzdz"](x, y, z)

        hess_v_to_r[:, :, i, 0, 0] = dxdx
        hess_v_to_r[:, :, i, 0, 1] = dxdy
        hess_v_to_r[:, :, i, 0, 2] = dxdz
        hess_v_to_r[:, :, i, 1, 0] = dxdy
        hess_v_to_r[:, :, i, 1, 1] = dydy
        hess_v_to_r[:, :, i, 1, 2] = dydz
        hess_v_to_r[:, :, i, 2, 0] = dxdz
        hess_v_to_r[:, :, i, 2, 1] = dydz
        hess_v_to_r[:, :, i, 2, 2] = dzdz

    # Dim = [mN, tK, (L + 1)^2, 3, 3]
    hess_v_to_r = hess_v_to_r * sh_consts[None, None, :, None, None]

    # Output: SH-RGB,  Dim = [mN, tK, mC]
    # Input:  means3d, Dim = [tK, 3]
    # Hessian: Dim = [mN, tK, mC, 3, 3] <- [mN, tK, mC, (L+1)^2, 1, 1] * [mN, tK, 1, (L + 1)^2, 3, 3]
    hess = torch.sum(
        jacob_c_to_v[:, :, :, :, None, None] * hess_v_to_r[:, :, None, :, :, :],
        dim=-3,
    )

    if masks is not None:
        jacob = jacob * masks[:, :, None, None].float()
        hess = hess * masks[:, :, None, None, None].float()
    return jacob, hess
