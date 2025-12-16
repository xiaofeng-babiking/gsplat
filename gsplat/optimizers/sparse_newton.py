import math
import torch
from typing import Optional, Callable, Dict, List, Any
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


def get_tile_size(
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
):
    """Get valid tile size within image."""
    tile_xmin = tile_x * tile_size_w
    tile_ymin = tile_y * tile_size_h
    tile_xmax = min(img_w, tile_xmin + tile_size_w)
    tile_ymax = min(img_h, tile_ymin + tile_size_h)

    crop_w = tile_xmax - tile_xmin
    crop_h = tile_ymax - tile_ymin
    return tile_xmin, tile_ymin, crop_w, crop_h


def compute_blend_alphas(alphas: torch.Tensor, clamped: bool = True):
    """Compute blended alphas by cumprod."""
    if clamped:
        alphas = torch.clamp_max(alphas, 0.9999)

    # Dim = [mN, tH, tW, tK]
    blend_alphas = torch.cumprod(1.0 - alphas, dim=-1)
    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
    blend_alphas[:, :, :, 0] = 1.0
    return alphas, blend_alphas


def _backward_render_to_sh_colors(
    num_chs: int,
    alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    blend_alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
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
    # Jacobian-SH-Value-to-Direction Dim = [mN, tK, (L+1)^2, 3]
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


def _backward_render_to_gaussians2d(
    sh_colors: torch.Tensor,
    opacities: torch.Tensor,
    alphas: torch.Tensor,
    blend_alphas: torch.Tensor,
    masks: Optional[torch.BoolTensor] = None,
):
    """Backward render colors to gaussian 2d weights.
    Args:
        [1] sh_colors: Spherical harmonic colors of each pixel, Dim = [mN, tK, mC].
        [2] opacities: Opacity of each pixel, Dim = [tK].
        [3] alphas: Alphas of each pixel on the tile, Dim = [mN, tH, tW, tK].
        [4] blend_alphas: Blended alphas, torch.Tensor, Dim = [mN, tH, tW, tK].
        [4] masks: Masks of valid gaussian splats, Dim = [mN, tK].

    Returns:
        [1] jacob: Jacobian matrix of each pixel w.r.t. Gaussian kernel 2D weights.
                Output: Dim = [mN, tH, tW, mC]
                Input:  Dim = [mN, tH, tW, tK]
                Jacobian: Dim = [mN, tH, tW, tK, mC]
        [2] hess: Hessian matrix of each pixel w.r.t. Gaussian kernel 2D weights.

    For pixel (m, n),

    RGB = SUM_k^{tK}(α(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - α(j)))
        where, α(j) = G(j) * σ(j)

    ∂(RGB) / ∂(G(0))
        = ∂(RGB)/ ∂(α(0)) @ ∂(α(0)) / ∂(G(0))
        = ∂(RGB)/ ∂(α(0)) * σ(0)
        = σ(0) * (
            + α(0) * SH(0) * 1.0 / α(0)
            - α(1) * SH(1) * CUMPROD_j^{0}(1.0 - α(j)) / (1 - α(0))
            - α(2) * SH(2) * CUMPROD_j^{1}(1.0 - α(j)) / (1 - α(0))
            ...
            - α(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - α(j)) / (1 - α(0))
        )

    ∂(RGB) / ∂(G(1))
        = σ(1) * (
            + α(1) * SH(1) * CUMPROD_j^{0}(1.0 - α(j)) / α(1)
            - α(2) * SH(2) * CUMPROD_j^{0}(1.0 - α(j)) / (1 - α(1))
            - α(3) * SH(3) * CUMPROD_j^{1}(1.0 - α(j)) / (1 - α(1))
            ...
            - α(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - α(j)) / (1 - α(1))
        )


    ...

    Let β(k) = α(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - α(j)),
    Then,
        S = [σ(0), σ(1), ..., σ(k)].T
        B = [β(0), β(1), ..., β(k)].T
        W = [
                [1.0 / α(0), -1.0 / (1.0 - α(0)),    ...     ,                       -1.0 / (1.0 - α(0))],
                [0.0,            1.0 / α(1)    , -1.0 / (1.0 - α(1)),      ... ,     -1.0 / (1.0 - α(1))],
                ...
                [0.0,     0.0    ,    ...     ,                  1.0 / α(k - 1), -1.0 / (1.0 - α(k - 1))],
                [0.0,     0.0    ,    ...     ,                         0.0,                  1.0 / α(k)],
        ]
        ∂(RGB) / ∂(G(k)) = S * W @ B
    """
    # alphas:    Dim = [mN, tH, tW, tK]
    # sh-colors: Dim = [mN, tK, mC]
    # betas:     Dim = [mN, tH, tW, tK, mC]
    betas = (
        alphas[:, :, :, :, None]  # Dim = [mN, tH, tW, tK, 1]
        * blend_alphas[:, :, :, :, None]  # Dim = [mN, tH, tW, tK, 1]
        * sh_colors[:, None, None, :, :]  # Dim = [mN, 1,  1,  tK, mC]
    )

    tile_size_h, tile_size_w = alphas.shape[1:3]
    n_splats = len(opacities)

    # Dim = [tH, tW, tK, tK]
    omegas = torch.ones(
        size=[tile_size_h, tile_size_w, n_splats, n_splats],
        dtype=alphas.dtype,
        device=alphas.device,
    )
    omegas = torch.triu(omegas, diagonal=0)
    # Dim = [mN, tH, tW, tK, tK] <- [1, tH, tW, tK, tK] / [mN, tH, tW, tK, 1]
    omegas = -omegas[None, :, :, :, :] / (1.0 - alphas[:, :, :, :, None])

    splat_idxs = list(range(n_splats))
    omegas[:, :, :, splat_idxs, splat_idxs] = 1.0 / (alphas + 1e-12)

    # Output: Dim = [mN, tH, tW, mC]
    # Input:  Dim = [mN, tH, tW, tK]
    # Jacobian: Dim = [mN, tH, tW, tK, mC] <- [mN, tH, tW, tK, tK] @ [mN, tH, tW, tK, mC]
    jacob = torch.einsum("ijklm,ijkmn->ijkln", omegas, betas)
    # Dim = [mN, tH, tW, mC, tK]
    jacob = jacob.permute([0, 1, 2, 4, 3])
    jacob = jacob * opacities

    hess = None

    if masks is not None:
        jacob = jacob * masks[:, None, None, None, :].float()
    return jacob, hess


def _backward_gaussians2d_to_means2d(
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
    gaussians2d: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    means2d: torch.Tensor,  # Dim = [mN, tK, 2]
    conics2d: torch.Tensor,  # Dim = [mN, tK, 3]
):
    """Backward from 2D Gaussian weights to 2D positions.

    Args:
        [1] tile_x: Tile column index, int.
        [2] tile_y: Tile row index, int.
        [3] tile_size_w: Tile width, int.
        [4] tile_size_h: Tile height, int.
        [4] img_w: Image width, int.
        [5] img_h: Image height, int.
        [6] gaussians2d: 2D gaussian weights of each pixel, Dim = [mN, tH, tW, tK].
        [7] means2d: 2D positions of each Gaussian kernel, Dim = [mN, tK, 2].
        [8] conics2d: Inverse covariance matrices of each Gaussian kernel, Dim = [mN, tK, 3].

    Returns:
        [1] jacob: Jacobian matrix of Gaussian kernel parameters w.r.t. means2d.
                Output: Dim = [mN, tH, tW, tK]
                Input:  Dim = [mN, tK, 2]
                Jacobian: Dim = [mN, tH, tW, tK, 2]
        [2] hess: Hessian matrix of Gaussian kernel parameters w.r.t. means2d.
                Hessian:  Dim = [mN, tH, tW, tK, 2, 2]

    G(k, m, n) = exp(-0.5 * (x(m, n) - means2d(k)).T @ conics2d @ (x(m, n) - means2d(k)))

    ∂(G(k, m, n)) / ∂(means2d(k))
        = -G(k, m, n) * conics2d.T * (means2d(k) - x(m, n))
    ∂²(G(k, m, n)) / ∂(means2d(k))²
        = G(k, m, n) * (conics2d.T * (means2d(k) - x(m, n)).T @ (conics2d.T * (means2d(k) - x(m, n))
            - G(k, m, n) * conics2d
    """
    tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
        tile_x, tile_y, tile_size_w, tile_size_h, img_w, img_h
    )

    tile_ys, tile_xs = torch.meshgrid(
        torch.arange(crop_h, dtype=means2d.dtype, device=means2d.device),
        torch.arange(crop_w, dtype=means2d.dtype, device=means2d.device),
        indexing="ij",
    )

    tile_xs = tile_xs + tile_xmin + 0.5
    tile_ys = tile_ys + tile_ymin + 0.5

    tile_xs = means2d[:, :, 0][:, None, None, :] - tile_xs[None, :, :, None]
    tile_ys = means2d[:, :, 1][:, None, None, :] - tile_ys[None, :, :, None]

    # Dim = [mN, tH, tW, tK, 2]
    term_0 = torch.stack(
        [
            tile_xs * conics2d[:, :, 0][:, None, None, :]
            + tile_ys * conics2d[:, :, 1][:, None, None, :],
            tile_xs * conics2d[:, :, 1][:, None, None, :]
            + tile_ys * conics2d[:, :, 2][:, None, None, :],
        ],
        dim=-1,
    )
    jacob = -gaussians2d[:, :, :, :, None] * term_0

    hess = gaussians2d[:, :, :, :, None, None] * torch.einsum(
        "ijklmn,ijklnp->ijklmp",
        term_0[:, :, :, :, :, None],
        term_0[:, :, :, :, None, :],
    )
    # Dim = [mN, tH, tW, tK, 3]
    term_1 = gaussians2d[:, :, :, :, None] * conics2d[:, None, None, :, :]
    term_1 = torch.stack(
        [
            term_1[:, :, :, :, 0],
            term_1[:, :, :, :, 1],
            term_1[:, :, :, :, 1],
            term_1[:, :, :, :, 2],
        ],
        dim=-1,
    )
    term_1 = term_1.reshape(list(term_1.shape[:-1]) + [2, 2])
    # Dim = [mN, tH, tW, tK, 2, 2]
    hess = hess - term_1
    return jacob, hess


def _backward_means2d_to_means3d(
    view_mats: torch.Tensor, cam_mats: torch.Tensor, means3d: torch.Tensor
):
    """Backward from 2D mean position to 3D mean position."""

    # Output: Dim = [mN, tK, 2]
    # Input:  Dim = [tK, 3]
    # Jacobian: Dim = [mN, tK, 2, 3]
    proj_mats = torch.eye(4, dtype=cam_mats.dtype, device=cam_mats.device)
    proj_mats = proj_mats[None].repeat(cam_mats.shape[0], 1, 1)
    # Dim = [mN, 4, 4]
    proj_mats[:, :3, :3] = cam_mats

    # Dim = [mN, 4, 4]
    pw = torch.einsum("ijk,ikl->ijl", proj_mats, view_mats)

    pk = torch.concatenate(
        [
            means3d,
            torch.ones(
                size=[means3d.shape[0], 1], dtype=means3d.dtype, device=means3d.device
            ),
        ],
        dim=-1,
    )

    # Dim = [mN, 4, 4] @ [tK, 4, 1] -> [mN, tK, 4, 1]
    h = torch.einsum("ijk,lkm->iljm", pw, pk[:, :, None]).squeeze(-1)
    # Dim = [mN, tK, 1]
    hx = h[:, :, 0][:, :, None]
    hy = h[:, :, 1][:, :, None]
    hz = h[:, :, 2][:, :, None] + 1e-12

    # Dim = [mN, 1, 3]
    pw_0 = pw[:, 0, :3][:, None, :]
    pw_1 = pw[:, 1, :3][:, None, :]
    pw_2 = pw[:, 2, :3][:, None, :]

    # Dim = [mN, tK, 2, 3]
    jacob = torch.zeros(
        size=[view_mats.shape[0], means3d.shape[0], 2, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    jacob[:, :, 0, :] = 1.0 / hz * (pw_0 - hx / hz * pw_2)
    jacob[:, :, 1, :] = 1.0 / hz * (pw_1 - hy / hz * pw_2)

    # Hessian: Dim = [mN, tK, 2, 3, 3]
    hess = torch.zeros(
        size=[view_mats.shape[0], means3d.shape[0], 2, 3, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    outer_fn = lambda x, y: torch.einsum(
        "ijk,ikl->ijl", x.reshape([-1, 3, 1]), y.reshape([-1, 1, 3])
    )
    hess[:, :, 0, :, :] = -1.0 / (hz**2 + 1e-12)[:, :, :, None] * (
        outer_fn(pw_2, pw_0) + outer_fn(pw_0, pw_2)
    ) + 2.0 / (hz**3 + 1e-12)[:, :, :, None] * hx[:, :, :, None] * outer_fn(
        pw_2, pw_2
    )
    hess[:, :, 1, :, :] = -1.0 / (hz**2 + 1e-12)[:, :, :, None] * (
        outer_fn(pw_2, pw_1) + outer_fn(pw_1, pw_2)
    ) + 2.0 / (hz**3 + 1e-12)[:, :, :, None] * hy[:, :, :, None] * outer_fn(
        pw_2, pw_2
    )
    return jacob, hess


def _backward_conics2d_to_covars2d(conics2d: torch.Tensor):
    """Backward from inverse covariance matrices to covariance matrices."""
    conics2d_mat = conics2d[..., [0, 1, 1, 2]].reshape(
        list(conics2d.shape[:-1]) + [2, 2]
    )

    i, j, p, l = torch.meshgrid(
        [
            torch.arange(2, dtype=conics2d.dtype, device=conics2d.device).int()
            for _ in range(4)
        ],
        indexing="ij",
    )
    jacob = torch.zeros(
        size=[*conics2d.shape[:-1], 2, 2, 2, 2],
        dtype=conics2d.dtype,
        device=conics2d.device,
    )
    jacob[..., i, j, p, l] = -conics2d_mat[..., i, p] * conics2d_mat[..., l, j]

    i, j, p, l, g, h = torch.meshgrid(
        [
            torch.arange(2, dtype=conics2d.dtype, device=conics2d.device).int()
            for _ in range(6)
        ],
        indexing="ij",
    )
    hess = torch.zeros(
        size=[*conics2d.shape[:-1], 2, 2, 2, 2, 2, 2],
        dtype=conics2d.dtype,
        device=conics2d.device,
    )
    hess[..., i, j, p, l, g, h] = (
        conics2d_mat[..., i, g] * conics2d_mat[..., h, p] * conics2d_mat[..., l, j]
        + conics2d_mat[..., i, p] * conics2d_mat[..., l, g] * conics2d_mat[..., h, j]
    )
    return jacob, hess


def _backward_gaussians2d_to_covars2d(
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
    gaussians2d: torch.Tensor,
    means2d: torch.Tensor,
    conics2d: torch.Tensor,
):
    """Backward from Gaussian 2D weights to 2D covariance matrices.

    Args:
        [1] tile_x: The x coordinate of the tile, int.
        [2] tile_y: The y coordinate of the tile, int.
        [3] tile_size_w: The width of the tile, int.
        [4] tile_size_h: The height of the tile, int.
        [5] gaussian2d: The Gaussian 2D weights, torch.Tensor.
        [6] means2d: The 2D mean positions, torch.Tensor.
        [7] conics2d: The 2D inverse of covariance matrices, torch.Tensor.

    Returns:
        [1] jacob: The Jacobian of the backward pass, torch.Tensor.
            Output: Dim = [mN, tH, tW, tK]
            Input:  Dim = [mN, tK, 2, 2]
            Jacobian: Dim = [mN, tH, tW, tK, 2, 2]
        [2] hess: The Hessian of the backward pass, torch.Tensor.
            Output: Dim = [mN, tH, tW, tK]
            Input:  Dim = [mN, tK, 2, 2]
            Hessian: Dim = [mN, tH, tW, tK, 2, 2, 2, 2]
    """
    tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
        tile_x, tile_y, tile_size_w, tile_size_h, img_w, img_h
    )

    tile_ys, tile_xs = torch.meshgrid(
        torch.arange(crop_h, dtype=means2d.dtype, device=means2d.device),
        torch.arange(crop_w, dtype=means2d.dtype, device=means2d.device),
        indexing="ij",
    )

    tile_xs = tile_xs + tile_xmin + 0.5
    tile_ys = tile_ys + tile_ymin + 0.5
    # Dim = [tH, tW, 2]
    tile_pixels = torch.stack([tile_xs, tile_ys], dim=-1)
    # Dim = [mN, 1, 1, tK, 2] - [1, tH, tW, 1, 2] -> [mN, tH, tW, tK, 2]
    tile_pixels = means2d[:, None, None, :, :] - tile_pixels[None, :, :, None, :]

    # jacobian conics2d to covars2d: Dim = [mN, tK, 2, 2, 2, 2]
    # hessian conics2d to covars2d: Dim = [mN, tK, 2, 2, 2, 2, 2, 2]
    jacob_conics2d, hess_conics2d = _backward_conics2d_to_covars2d(conics2d)

    # Dim = [mN, tH, tW, tK, 2] @  [mN, tK, 2, 2, 2, 2] @ [mN, tH, tW, tK, 2]
    #   -> [mN, tH, tW, tK, 2, 2]
    term_0 = torch.einsum(
        "nhwka,nkabcd,nhwkb->nhwkcd", tile_pixels, jacob_conics2d, tile_pixels
    )
    # Dim = [mN, tH, tW, tK] * [mN, tH, tW, tK, 2, 2]
    jacob = -0.5 * gaussians2d[:, :, :, :, None, None] * term_0

    # Dim = [mN, tH, tW, tK, 2, 2] -> [mN, tH, tW, tK, 2, 2, 2, 2]
    batch_dims = list(term_0.shape[:-2])
    term_0_sq = torch.einsum(
        "...ij,...jk->...ik",
        term_0.reshape(batch_dims + [4, 1]),
        term_0.reshape(batch_dims + [1, 4]),
    )
    term_0_sq = term_0_sq.reshape(batch_dims + [2, 2, 2, 2])

    # Dim = [mN, tH, tW, tK, 2ᵃ] @ [mN, tK, 2ᵃ, 2ᵇ, 2, 2, 2, 2] @ [mN, tH, tW, tK, 2ᵇ]
    #    -> [mN, tH, tW, tK, 2, 2, 2, 2]
    term_1 = torch.einsum(
        "nhwka,nkabpquv,nhwkb->nhwkpquv", tile_pixels, hess_conics2d, tile_pixels
    )

    # Dim = [mN, tH, tW, tK, 1, 1, 1, 1] * [mN, tH, tW, tK, 2ᵃ, 2ᵇ, 2ᵍ, 2ʰ]
    hess = (
        0.25 * gaussians2d[:, :, :, :, None, None, None, None] * term_0_sq
        - 0.5 * gaussians2d[:, :, :, :, None, None, None, None] * term_1
    )
    return jacob, hess


def _backward_covars2d_to_pinhole(
    cam_covars3d: torch.Tensor,  # Dim = [mN, tK, 3, 3]
    pin_jacob: torch.Tensor,  # Dim = [mN, tK, 2, 3]
):
    """Backward from 2D covariance matrices to pinhole jacobian."""
    # Output: Dim = [mN, tK, 2, 2]
    # Input:  Dim = [tK, 2, 3]
    # Jacobian: Dim = [mN, tK, 2, 2, 2, 3]
    jacob = torch.zeros(
        size=list(cam_covars3d.shape[:-2]) + [2, 2, 2, 3],
        dtype=cam_covars3d.dtype,
        device=cam_covars3d.device,
    )

    # Dim = [mN, tK, 3]
    pin_0 = pin_jacob[:, :, 0, :]
    pin_1 = pin_jacob[:, :, 1, :]

    # ∂Σ(0, 0) / ∂J(0) = J(0) @ (A + A.T)
    # Dim = [mN, tK, 3] @ [mN, tK, 3, 3]
    jacob[:, :, 0, 0, 0, :] = torch.einsum(
        "nka,nkab->nkb", pin_0, (cam_covars3d + cam_covars3d.transpose(-1, -2))
    )
    # ∂Σ(0, 0) / ∂J(1) = 0
    # ∂Σ(0, 1) / ∂J(0) = J(1) @ A.T
    jacob[:, :, 0, 1, 0, :] = torch.einsum(
        "nka,nkab->nkb", pin_1, cam_covars3d.transpose(-1, -2)
    )
    # ∂Σ(0, 1) / ∂J(1) = J(0) @ A
    jacob[:, :, 0, 1, 1, :] = torch.einsum("nka,nkab->nkb", pin_0, cam_covars3d)
    # ∂Σ(1, 0) / ∂J(0) = ∂(J(1) @ A @ J(0).T) / ∂J(0) = J(1) @ A, actually A = A.T
    jacob[:, :, 1, 0, 0, :] = torch.einsum("nka,nkab->nkb", pin_1, cam_covars3d)
    # ∂Σ(1, 0) / ∂J(1) = ∂(J(1) @ A @ J(0).T) / ∂J(1) = J(0) @ A.T, actually A = A.T
    jacob[:, :, 1, 0, 1, :] = torch.einsum(
        "nka,nkab->nkb", pin_0, cam_covars3d.transpose(-1, -2)
    )
    # ∂Σ(1, 1) / ∂J(0) = 0
    # ∂Σ(1, 1) / ∂J(1) = J(1) @ (A + A.T)
    jacob[:, :, 1, 1, 1, :] = torch.einsum(
        "nka,nkab->nkb", pin_1, (cam_covars3d + cam_covars3d.transpose(-1, -2))
    )

    # Hessian: Dim = [mN, tK, 2, 2, 2, 3, 2, 3]
    hess = torch.zeros(
        size=list(cam_covars3d.shape[:-2]) + [2, 2, 2, 3, 2, 3],
        dtype=cam_covars3d.dtype,
        device=cam_covars3d.device,
    )
    # ∂²Σ(0, 0) / ∂J(0)² = A + A.T
    hess[:, :, 0, 0, 0, :, 0, :] = cam_covars3d + cam_covars3d.transpose(-1, -2)
    # ∂²Σ(0, 0) / ∂J(0)∂J(1) = 0
    # ∂²Σ(0, 0) / ∂J(1)∂J(0) = 0
    # ∂²Σ(0, 0) / ∂J(1)² = 0

    # ∂²Σ(0, 1) / ∂J(0)² = 0
    # ∂²Σ(0, 1) / ∂J(0)∂J(1) = A.T
    hess[:, :, 0, 1, 0, :, 1, :] = cam_covars3d.transpose(-1, -2)
    # ∂²Σ(0, 1) / ∂J(1)∂J(0) = A
    hess[:, :, 0, 1, 1, :, 0, :] = cam_covars3d
    # ∂²Σ(0, 1) / ∂J(1)² = 0

    # ∂²Σ(1, 0) / ∂J(0)² = 0
    # ∂²Σ(1, 0) / ∂J(0)∂J(1) = A
    hess[:, :, 1, 0, 0, :, 1, :] = cam_covars3d
    # ∂²Σ(1, 0) / ∂J(1)∂J(0) = A.T
    hess[:, :, 1, 0, 1, :, 0, :] = cam_covars3d.transpose(-1, -2)
    # ∂²Σ(1, 0) / ∂J(1)² = 0

    # ∂²Σ(1, 1) / ∂J(0)² = 0
    # ∂²Σ(1, 1) / ∂J(0)∂J(1) = 0
    # ∂²Σ(1, 1) / ∂J(1)∂J(0) = 0
    # ∂²Σ(1, 1) / ∂J(1)² = A + A.T
    hess[:, :, 1, 1, 1, :, 1, :] = cam_covars3d + cam_covars3d.transpose(-1, -2)
    return jacob, hess


def _backward_pinhole_to_means3d(
    view_mats: torch.Tensor,
    cam_mats: torch.Tensor,
    means3d: torch.Tensor,
):
    """Backward from pinhole jacobian to 3D positions in the world frame.

    J = [[fx / z,    0,    -fx * x / z^2],
         [0,      fy / z,  -fy * y / z^2]]
    where (x, y, z) is the 3D position in the camera frame.
    """
    # Dim = [mN, 1]
    fx = cam_mats[:, 0, 0][:, None]
    fy = cam_mats[:, 1, 1][:, None]

    cam_means3d = (
        torch.sum(view_mats[:, None, :3, :3] * means3d[None, :, None, :], dim=-1)
        + view_mats[:, :3, 3][:, None, :]
    )
    # Dim = [mN, tK]
    cam_xs = cam_means3d[:, :, 0]
    cam_ys = cam_means3d[:, :, 1]
    cam_zs = cam_means3d[:, :, 2]

    # Dim = [mN, tK, 2, 3, 3]
    batch_dims = [view_mats.shape[0], means3d.shape[0]]
    # Output: Dim = [mN, tK, 2, 3]
    # Input: Dim = [mN, tK, 3]
    # Jacobian-Pinhole-CameraMeans3d: Dim = [mN, tK, 2, 3, 3]
    jacob_p_to_c = torch.zeros(
        size=batch_dims + [2, 3, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    eps = 1e-12
    # ∂J(0, 0) / ∂X(2) = -fx / z²
    jacob_p_to_c[:, :, 0, 0, 2] = -fx / (cam_zs**2 + eps)
    # ∂J(0, 2) / ∂X(0) = -fx / z²
    jacob_p_to_c[:, :, 0, 2, 0] = -fx / (cam_zs**2 + eps)
    # ∂J(0, 2) / ∂X(2) = 2.0 * fx * x / z³
    jacob_p_to_c[:, :, 0, 2, 2] = 2.0 * fx * cam_xs / (cam_zs**3 + eps)
    # ∂J(1, 1) / ∂X(2) = -fy / z²
    jacob_p_to_c[:, :, 1, 1, 2] = -fy / (cam_zs**2 + eps)
    # ∂J(1, 2) / ∂X(1) = -fy / z²
    jacob_p_to_c[:, :, 1, 2, 1] = -fy / (cam_zs**2 + eps)
    # ∂J(1, 2) / ∂X(2) = 2.0 * fy * y / z³
    jacob_p_to_c[:, :, 1, 2, 2] = 2.0 * fy * cam_ys / (cam_zs**3 + eps)

    # Hessian-Pinhole-CameraMeans3d: Dim = [mN, tK, 2, 3, 3, 3]
    hess_p_to_c = torch.zeros(
        size=batch_dims + [2, 3, 3, 3],
        dtype=means3d.dtype,
        device=means3d.device,
    )
    # ∂J(0, 0) / ∂X(2) = -fx / z²
    #   ∂²J(0, 0) / ∂X(2)∂X(2) = 2.0 * fx / z³
    hess_p_to_c[:, :, 0, 0, 2, 2] = 2.0 * fx / (cam_zs**3 + eps)
    # ∂J(0, 2) / ∂X(0) = -fx / z²
    #   ∂²J(0, 2) / ∂X(0)∂X(2) = 2.0 * fx / z³
    hess_p_to_c[:, :, 0, 2, 0, 2] = 2.0 * fx / (cam_zs**3 + eps)
    # ∂J(0, 2) / ∂X(2) = 2.0 * fx * x / z³
    #   ∂²J(0, 2) / ∂X(2)∂X(0) = 2.0 * fx / z³
    #   ∂²J(0, 2) / ∂X(2)∂X(2) = -6.0 * fx * x / z⁴
    hess_p_to_c[:, :, 0, 2, 2, 0] = 2.0 * fx / (cam_zs**3 + eps)
    hess_p_to_c[:, :, 0, 2, 2, 2] = -6.0 * fx * cam_xs / (cam_zs**4 + eps)
    # ∂J(1, 1) / ∂X(2) = -fy / z²
    #   ∂²J(1, 1) / ∂X(2)∂X(2) = 2.0 * fy / z³
    hess_p_to_c[:, :, 1, 1, 2, 2] = 2.0 * fy / (cam_zs**3 + eps)
    # ∂J(1, 2) / ∂X(1) = -fy / z²
    #   ∂²J(1, 2) / ∂X(1)∂X(2) = 2.0 * fy / z³
    hess_p_to_c[:, :, 1, 2, 1, 2] = 2.0 * fy / (cam_zs**3 + eps)
    # ∂J(1, 2) / ∂X(2) = 2.0 * fy * y / z³
    #   ∂²J(1, 2) / ∂X(2)∂X(2) = -6.0 * fy * y / z⁴
    #   ∂²J(1, 2) / ∂X(2)∂X(1) = 2.0 * fy / z³
    hess_p_to_c[:, :, 1, 2, 2, 2] = -6.0 * fy * cam_ys / (cam_zs**4 + eps)
    hess_p_to_c[:, :, 1, 2, 2, 1] = 2.0 * fy / (cam_zs**3 + eps)

    # Output: Dim = [mN, tK, 3]
    # Input: Dim = [mN, tK, 3]
    # Jacobian-CameraMeans3d-WorldMeans3d: Dim = [mN, tK, 3, 3]
    # X = R @ X' + t
    # ∂X / ∂X' = R
    # ∂²X / ∂X'² = 0
    # ∂J / ∂X' = (∂J / ∂X) @ (∂X / ∂X')
    # ∂²J / ∂X'²
    #   = (∂X / ∂X').T @ (∂²J / ∂X²) @ (∂X / ∂X') + (∂J / ∂X) @ (∂²X / ∂X'²)
    #   = (∂X / ∂X').T @ (∂²J / ∂X²) @ (∂X / ∂X')

    # Dim = [mN, 3, 3]
    rot_mats = view_mats[:, :3, :3]
    # Dim = [mN, tK, 2, 3, 3] @ [mN, 3, 3] -> [mN, tK, 2, 3, 3]
    jacob = torch.einsum("nkabc,ncd->nkabd", jacob_p_to_c, rot_mats)

    # Dim = [mN, 3, 3ᵃ] @ [mN, tK, 2, 3, 3ᵃ, 3ᵇ] @ [mN, 3ᵇ, 3] -> [mN, tK, 2, 3, 3, 3]
    hess = torch.einsum(
        "ica,ikmnab,ibd->ikmncd", rot_mats.transpose(-1, -2), hess_p_to_c, rot_mats
    )
    return jacob, hess


def _backward_render_to_means3d(
    num_chs: int,
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
    view_mats: torch.Tensor,  # Dim = [mN, 4, 4]
    cam_mats: torch.Tensor,  # Dim = [mN, 3, 3]
    means3d: torch.Tensor,  # Dim = [tK, 3]
    opacities: torch.Tensor,  # Dim = [tK]
    sh_coeffs: torch.Tensor,  # Dim = [tK, (L + 1)^2, 3]
    sh_deriv_funcs: List[Dict[str, Callable]],  # Dim = [(L + 1)^2]
    alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    blend_alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    sh_colors: torch.Tensor,  # Dim = [mN, tK, 3]
    gaussians2d: torch.Tensor,  # Dim = [mN, tH, tW, tK]
    means2d: torch.Tensor,  # Dim = [mN, tK, 2]
    conics2d: torch.Tensor,  # Dim = [mN, tK, 3]
    cam_covars3d: torch.Tensor,  # Dim = [mN, tK, 3, 3]
    pin_jacob: torch.Tensor,  # Dim = [mN, tK, 2, 3]
    tile_pixels: torch.Tensor,  # Dim = [mN, tH, tW, 2]
    masks: Optional[torch.Tensor] = None,  # Dim = [mN, tK]
):
    """Backward from render RGB to 3D positions in the world frame.
    Symbols:
        c:  pixel-wise tile-level render RGB, Dim = [mN, tH, tW, mC]
        ck: splat-wise SH RGB, Dim = [mN, tK, mC]
        pk: splat-wise 3D centers in the world frame, Dim = [tK, 3]
        gk: splat-wise tile-level 2D gaussian weights, Dim = [mN, tK, tH, tW]
        uk: i.e. πk, splat-wise 2D centers in the image plane, Dim = [mN, tK, 2]
        sk: i.e. Σk, splat-wise 2D covariance matrices in the image plane, Dim = [mN, tK, 2, 2]
        jk: splat-wise pinhole-projection 2x3 jacobian-approximate matrices, Dim = [mN, tK, 2, 3]
    """
    # c: Dim = [mN, tH, tW, mC]
    # ck: Dim = [mN, tK, mC]
    # j_c_ck: Dim = [mN, tH, tW, tK, mC, mC] -> RGB diagonal -> [mN, tH, tW, tK, mC]
    j_c_ck, _ = _backward_render_to_sh_colors(num_chs, alphas, blend_alphas, masks=None)

    # ck: Dim = [mN, tK, mC]
    # pk: Dim = [tK, 3]
    # j_ck_pk: Dim = [mN, tK, mC, 3]
    # h_ck_pk: Dim = [mN, tK, mC, 3, 3]
    j_ck_pk, h_ck_pk = _backward_sh_colors_to_positions(
        means3d,
        view_mats,
        sh_coeffs,
        sh_deriv_funcs,
        None,
        masks=None,
    )

    # c: Dim = [mN, tH, tW, mC]
    # gk: Dim = [mN, tH, tW, tK]
    # j_c_gk: Dim = [mN, tH, tW, tK, mC]
    j_c_gk, _ = _backward_render_to_gaussians2d(
        sh_colors, opacities, alphas, blend_alphas, masks=None
    )

    # gk: Dim = [mN, tH, tW, tK]
    # uk: Dim = [mN, tK, 2]
    # j_gk_uk: Dim = [mN, tH, tW, tK, 2]
    # h_gk_uk: Dim = [mN, tH, tW, tK, 2, 2]
    j_gk_uk, h_gk_uk = _backward_gaussians2d_to_means2d(
        tile_x,
        tile_y,
        tile_size_w,
        tile_size_h,
        img_w,
        img_h,
        gaussians2d,
        means2d,
        conics2d,
    )

    # uk: Dim = [mN, tK, 2]
    # pk: Dim = [tK, 3]
    # j_uk_pk: Dim = [mN, tK, 2, 3]
    # h_uk_pk: Dim = [mN, tK, 2, 3, 3]
    j_uk_pk, h_uk_pk = _backward_means2d_to_means3d(view_mats, cam_mats, means3d)

    # gk: Dim = [mN, tH, tW, tK]
    # sk: Dim = [mN, tK, 2, 2]
    # j_gk_sk: Dim = [mN, tH, tW, tK, 2, 2]
    # h_gk_sk: Dim = [mN, tH, tW, tK, 2, 2, 2, 2]
    j_gk_sk, h_gk_sk = _backward_gaussians2d_to_covars2d(
        tile_x,
        tile_y,
        tile_size_w,
        tile_size_h,
        img_w,
        img_h,
        gaussians2d,
        means2d,
        conics2d,
    )

    # sk: Dim = [mN, tK, 2, 2]
    # jk: Dim = [mN, tK, 2, 3]
    # j_sk_jk: Dim = [mN, tK, 2, 2, 2, 3]
    # h_sk_jk: Dim = [mN, tK, 2, 2, 2, 3, 2, 3]
    j_sk_jk, h_sk_jk = _backward_covars2d_to_pinhole(cam_covars3d, pin_jacob)

    # jk: Dim = [mN, tK, 2, 3]
    # pk: Dim = [tK, 3]
    # j_jk_pk: Dim = [mN, tK, 2, 3, 3]
    # h_jk_pk: Dim = [mN, tK, 2, 3, 3, 3]
    j_jk_pk, h_jk_pk = _backward_pinhole_to_means3d(view_mats, cam_mats, means3d)
    # j_sk_pk
    #   = j_sk_jk @ j_jk_pk
    #   -> [mN, tK, 2, 2, 2, 3] @ [mN, tK, 2, 3, 3] -> [mN, tK, 2, 2, 3]
    j_sk_pk = torch.einsum("nkabpq,nkpqc->nkabc", j_sk_jk, j_jk_pk)

    # h_sk_pk
    #   = j_jk_pk.T @ h_sk_jk @ j_jk_pk + j_sk_jk @ h_jk_pk
    # h_sk_pk_0
    #   = j_jk_pk.T @ h_sk_jk @ j_jk_pk
    #   -> [mN, tK, 2ᵃ, 3ᵇ, 3] @ [mN, tK, 2, 2, 2ᵃ, 3ᵇ, 2ᶜ, 3ᵈ] @ [mN, tK, 2ᶜ, 3ᵈ, 3]
    #   -> [mN, tK, 2, 2, 3, 3]
    h_sk_pk_0 = torch.einsum("nkabe,nkpqabcd,nkcdf->nkpqef", j_jk_pk, h_sk_jk, j_jk_pk)
    # h_sk_pk_1
    #   = j_sk_jk @ h_jk_pk
    #   -> [mN, tK, 2, 2, 2, 3] @ [mN, tK, 2, 3, 3, 3]
    h_sk_pk_1 = torch.einsum("nkpqab,nkabcd->nkpqcd", j_sk_jk, h_jk_pk)
    h_sk_pk = h_sk_pk_0 + h_sk_pk_1

    # ∂G / ∂π = -G * Σ^-1 @ (π - x) where x is the pixel coordinates within a tile
    # ∂G / ∂π∂Σ
    #   = -(∂G / ∂Σ) * Σ^-1 @ (π - x) - G * (∂(Σ^-1) / ∂Σ) @ (π - x)
    # h_gk_uk_sk = -j_gk_sk @ conics2d @ (means2d - x) - j_gk_uk @ j_sk_inv_sk
    # h_gk_uk_sk_0
    #   = -j_gk_sk @ conics2d @ (means2d - x)
    #   = -j_gk_sk @ conics2d @ tile_pixels
    #   -> [mN, tH, tW, tK, 2ˢ, 2ˢ] * [mN, tK, 2ᵘ, 2] @ [mN, tK, 2]
    #   -> [mN, tH, tW, tK, 2ᵘ, 2ˢ, 2ˢ]
    h_gk_uk_sk_0 = -torch.einsum(
        "nhwkcd,nkab,nkb->nhwkacd", j_gk_sk, conics2d, tile_pixels
    )
    # sk_inv: Dim = [mN, tK, 2, 2]
    # sk: Dim = [mN, tK, 2, 2]
    # j_sk_inv_sk: Dim = [mN, tK, 2ᵃ, 2ᵇ, 2, 2]
    j_sk_inv_sk, _ = _backward_conics2d_to_covars2d(conics2d)
    # h_gk_uk_sk_1
    #   = -gaussians2d * tile_pixels.T @ j_sk_inv_sk
    #   -> [mN, tH, tW, tK] * [mN, tK, 2ᵃ, 2ᵇ, 2, 2] @ [mN, tK, 2ᵇ]
    h_gk_uk_sk_1 = -torch.einsum(
        "nhwk,nkabcd,nkb->nhwkacd", gaussians2d, j_sk_inv_sk, tile_pixels
    )
    h_gk_uk_sk = h_gk_uk_sk_0 + h_gk_uk_sk_1

    # jacob
    #   = j_c_ck @ j_ck_pk + j_c_gk @ (j_gk_uk @ j_uk_pk + j_gk_sk @ j_sk_pk)
    # jacob_0
    #   = j_c_ck @ j_ck_pk
    #   -> [mN, tH, tW, tK, mC, 1] * [mN, 1, 1, tK, mC, 3]
    #   -> [mN, tH, tW, tK, mC, 3]
    jacob_0 = j_c_ck[:, :, :, :, :, None] * j_ck_pk[:, None, None, :, :, :]
    # jacob_1_0
    #   = j_gk_uk @ j_uk_pk
    #   -> [mN, tH, tW, tK, 2] @ [mN, tK, 2, 3]
    jacob_1_0 = torch.einsum("nhwku,nkux->nhwkx", j_gk_uk, j_uk_pk)
    # jacob_1_1
    #   = j_gk_sk @ j_sk_pk
    #   -> [mN, tH, tW, tK, 2, 2] @ [mN, tK, 2, 2, 3]
    jacob_1_1 = torch.einsum("nhwkab,nkabx->nhwkx", j_gk_sk, j_sk_pk)
    # jacob_1
    #   = j_c_gk @ (j_gk_uk @ j_uk_pk + j_gk_sk @ j_sk_pk)
    #   -> [mN, tH, tW, tK, mC, 1] @ [mN, tH, tW, tK, 1, 3]
    jacob_1 = j_c_gk[:, :, :, :, :, None] * (jacob_1_0 + jacob_1_1)[:, :, :, :, None, :]
    # Dim = [mN, tH, tW, tK, mC, 3]
    jacob = jacob_0 + jacob_1

    # hess_0
    #   = j_c_ck @ h_ck_pk
    #   -> [mN, tH, tW, tK, mC, 1, 1] @ [mN, tK, mC, 3, 3]
    #   -> [mN, tH, tW, tK, mC, 3, 3]
    hess_0 = j_c_ck[:, :, :, :, :, None, None] * h_ck_pk[:, None, None, :, :, :, :]

    # hess_1_0
    #   = j_gk_uk @ h_uk_pk
    #   -> [mN, tH, tW, tK, 2] @ [mN, tK, 2, 3, 3]
    #   -> [mN, tH, tW, tK, 3, 3]
    hess_1_0 = torch.einsum("nhwku,nkuxy->nhwkxy", j_gk_uk, h_uk_pk)

    # hess_1_1
    #   = j_gk_sk @ h_sk_pk
    #   -> [mN, tH, tW, tK, 2, 2] @ [mN, tK, 2, 2, 3, 3]
    #   -> [mN, tH, tW, tK, 3, 3]
    hess_1_1 = torch.einsum("nhwkab,nkabxy->nhwkxy", j_gk_sk, h_sk_pk)

    # hess_1_2
    #   = j_uk_pk.T @ h_gk_uk @ j_uk_pk
    #   -> [mN, tH, tW, tK, 2, 3] @ [mN, tH, tW, tK, 2, 2] @ [mN, tH, tW, tK, 2, 3]
    hess_1_2 = torch.einsum("nhwkax,nhwkab,nhwkby->nhwkxy", j_uk_pk, h_gk_uk, j_uk_pk)

    # hess_1_3
    #   = j_sk_pk.T @ h_gk_sk @ j_sk_pk
    #   -> [mN, tK, 2ᵃ, 2ᵇ, 3] @ [mN, tH, tW, tK, 2ᵃ, 2ᵇ, 2ᶜ, 2ᵈ] @ [mN, tK, 2ᶜ, 2ᵈ, 3]
    hess_1_3 = torch.einsum("nkabx,nhwkabcd,nkcdy->nhwkxy", j_sk_pk, h_gk_sk, j_sk_pk)

    # hess_1_4
    #   = 2.0 * j_sk_pk.T @ h_gk_uk_sk @ j_uk_pk
    #   -> 2.0 * [mN, tK, 2, 2, 3] @ [mN, tH, tW, tK, 2, 2, 2] @ [mN, tK, 2, 3]
    #   -> [mN, tH, tW, tK, 3, 3]
    hess_1_4 = 2.0 * torch.einsum(
        "nkabx,nhwkuab,nkuy->nhwkxy", j_sk_pk, h_gk_uk_sk, j_uk_pk
    )
    # hess_1 = j_c_gk @ (*)
    #   -> [mN, tH, tW, tK, mC, 1, 1] * [mN, tH, tW, 1, tK, 3, 3]
    #   -> [mN, tH, tW, tK, mC, 3, 3]
    hess_1 = (
        j_c_gk[:, :, :, :, :, None, None]
        * (hess_1_0 + hess_1_1 + hess_1_2 + hess_1_3 + hess_1_4)[:, :, :, None, :, :, :]
    )
    hess = hess_0 + hess_1

    if masks is not None:
        # Dim = [mN, tH, tW, tK, mC, 3] * [mN, tK]
        jacob = jacob * masks[:, None, None, :, None, None].float()
        # Dim = [mN, tH, tW, tK, mC, 3, 3] * [mN, tK]
        hess = hess * masks[:, None, None, :, None, None, None].float()
    return jacob, hess


def _backward_l2_to_render(rd_imgs: torch.Tensor, gt_imgs: torch.Tensor):
    """Computes Jacobian matrix from L2 to rendering RGB pixels.

    Args:
        [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
        [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.
        [3] with_hessian: Whether to compute Hessian matrix, bool.

    Returns:
        [1] jacob: Jacobian matrix from L2 to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.
                L2 = MEAN(||rd_imgs - gt_imgs||^2)
                RGB =  { pixel=(r, g, b) | for any pixel in rd_imgs }
                RGB' = { pixel=(r, g, b) | for any pixel in gt_imgs }
                ∂L2 / ∂(RGB) -> partial derivative of L2 (scalar) with respect to C (matrix)

                ∂L2 / ∂(RGB)
                    = [∂L2 / ∂R, ∂L2 / ∂G, ∂L2 / ∂B]
                    = 2.0 * (rd_imgs - gt_imgs) / (N * C * H * W)
        [2] hess: Hessian matrix from L2 to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.
                ∂²L2 / ∂(RGB)²
                = [[∂²L2 / ∂R²,  ∂L2 / ∂R∂G, ∂L2 / ∂R∂B],
                   [∂²L2 / ∂R∂G, ∂²L2 / ∂G², ∂L2 / ∂G∂B],
                   [∂²L2 / ∂R∂B, ∂L2 / ∂G∂B, ∂²L2 / ∂B²]]
                Since R, G and B are independent with each other,
                = 2.0 * diagonal([1, 1, 1]).view(N, C, C, H, W)
    """
    n, c, h, w = rd_imgs.shape

    factor = 2.0 / (n * c * h * w)

    jacob = factor * (rd_imgs - gt_imgs)
    # Dim = [mN, tH, tW, mC]
    jacob = jacob.permute([0, 2, 3, 1])

    hess = 2.0
    return jacob, hess


class PositionNewton(torch.optim.Optimizer):
    def __init__(
        self,
        params: Dict[str, Any],
        sh_deg: int = 3,
        num_chs: int = 3,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(params, defaults)

        self._sh_deg = sh_deg
        self._sh_deriv_funcs = cache_sh_derivative_functions(sh_deg)

        self._num_chs = num_chs

    def step(
        self,
        visibility: torch.Tensor,
        rd_imgs: torch.Tensor,
        gt_imgs: torch.Tensor,
        tile_x: int,
        tile_y: int,
        tile_size_w: int,
        tile_size_h: int,
        img_w: int,
        img_h: int,
        view_mats: torch.Tensor,  # Dim = [mN, 4, 4]
        cam_mats: torch.Tensor,  # Dim = [mN, 3, 3]
        opacities: torch.Tensor,  # Dim = [tK]
        sh_coeffs: torch.Tensor,  # Dim = [tK, (L + 1)^2, 3]
        alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
        blend_alphas: torch.Tensor,  # Dim = [mN, tH, tW, tK]
        sh_colors: torch.Tensor,  # Dim = [mN, tK, 3]
        gaussians2d: torch.Tensor,  # Dim = [mN, tH, tW, tK]
        means2d: torch.Tensor,  # Dim = [mN, tK, 2]
        conics2d: torch.Tensor,  # Dim = [mN, tK, 3]
        cam_covars3d: torch.Tensor,  # Dim = [mN, tK, 3, 3]
        pin_jacob: torch.Tensor,  # Dim = [mN, tK, 2, 3]
        tile_pixels: torch.Tensor,  # Dim = [mN, tH, tW, 2]
        masks: Optional[torch.Tensor] = None,  # Dim = [mN, tK]
    ):
        for group in self.param_groups:
            name = group["name"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]

            j_l2_c, h_l2_c = _backward_l2_to_render(rd_imgs, gt_imgs)

            j_c_pk, h_c_pk = _backward_render_to_means3d(
                self._num_chs,
                tile_x,
                tile_y,
                tile_size_w,
                tile_size_h,
                img_w,
                img_h,
                view_mats,
                cam_mats,
                param[visibility],
                opacities,
                sh_coeffs,
                self._sh_deriv_funcs,
                alphas,
                blend_alphas,
                sh_colors,
                gaussians2d,
                means2d,
                conics2d,
                cam_covars3d,
                pin_jacob,
                tile_pixels,
                masks,
            )

            # jacob
            #   = j_l2_c @ j_c_pk
            #   -> [mN, tH, tW, mC] @ [mN, tH, tW, tK, mC, 3]
            #   -> [mN, tH, tW, tK, 3] -> [tK, 3]
            jacob = torch.einsum("nhwc,nhwkcx->nhwkx", j_l2_c, j_c_pk)
            jacob = jacob.sum(dim=[0, 1, 2])

            # hess = j_c_pk.T @ h_l2_c @ j_c_pk + j_l2_c @ h_c_pk
            # hess_0
            #   = j_c_pk.T @ h_l2_c @ j_c_pk
            #   -> [mN, tH, tW, tK, mC, 3].T @ [1] @ [mN, tH, tW, tK, mC, 3]
            hess_0 = h_l2_c * torch.einsum(
                "...ab,...bc->...ac", j_c_pk.transpose(-1, -2), j_c_pk
            )
            # hess_1
            #   = j_l2_c @ h_c_pk
            #   -> [mN, tH, tW, mC] @ [mN, tH, tW, tK, mC, 3, 3]
            hess_1 = torch.einsum("nhwc,nhwkcxy->nhwkxy", j_l2_c, h_c_pk)
            hess = hess_0 + hess_1
            # Dim = [tK, 3, 3]
            hess = hess.sum(dim=[0, 1, 2])

            hess_inv = torch.linalg.pinv(hess)

            update = -torch.einsum("kab,kb->ka", hess_inv, jacob)
            param[visibility] += update
