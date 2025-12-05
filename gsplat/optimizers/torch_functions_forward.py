import math
import torch
import torch.nn.functional as F
from typing import Optional

EPS = 1e-12


def project_points_3d_to_2d(
    pnts3d: torch.FloatTensor,
    view_mats: torch.FloatTensor,
    cam_mats: torch.FloatTensor,
) -> torch.FloatTensor:
    """Project means from 3d world to 2d image plane.

    Args:
        [1] pnts3d: 3D positions at World Coordinate system, i.e. [tK, 3]
        [2] view_mat: view matrix, i.e. [mN, 4, 4]
        [3] cam_mat: camera matrix, i.e. [mN, 3, 3]

    Return:
        [1] cam_pnts3d: 3D positions at Camera coordinate system.
        [2] img_pnts2d: 2D pixels, i.e. [tK, 2]
    """
    # Dim = [mN, 1, 3, 3] * [1, tK, 1, 3] + [mN, 1, 3] -> [mN, tK, 3]
    cam_pnts3d = (
        torch.sum(view_mats[:, None, :3, :3] * pnts3d[None, :, None, :], dim=-1)
        + view_mats[:, :3, 3][:, None, :]
    )

    # Dim = [mN, 1, 3, 3] * [mN, tK, 1, 3] -> [mN, tK, 3]
    img_pnts2d = torch.sum(cam_mats[:, None, :, :] * cam_pnts3d[:, :, None, :], dim=-1)

    # Dim = [mN, tK, 2]
    img_pnts2d = img_pnts2d[:, :, :2] / img_pnts2d[:, :, 2][..., None]
    return cam_pnts3d, img_pnts2d


def get_tile_size(
    tile_x: int, tile_y: int, tile_w: int, tile_h: int, img_w: int, img_h: int
):
    """Get valid tile size within image."""
    tile_xmin = tile_x * tile_w
    tile_ymin = tile_y * tile_h
    tile_xmax = min(img_w, tile_xmin + tile_w)
    tile_ymax = min(img_h, tile_ymin + tile_h)

    crop_w = tile_xmax - tile_xmin
    crop_h = tile_ymax - tile_ymin
    return tile_xmin, tile_ymin, crop_w, crop_h


def quaternion_to_rotation_matrix(
    quats: torch.FloatTensor,
) -> torch.FloatTensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    rot_mats = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return rot_mats.reshape(quats.shape[:-1] + (3, 3))


def compute_covariance_3d(
    quats: torch.FloatTensor,  # Dim = [tK, 4]
    scales: torch.FloatTensor,  # Dim = [tK, 3]
):
    """Compute 3D covariance from scale and quaternion."""
    rot_mats = quaternion_to_rotation_matrix(quats)

    covars3d = rot_mats * scales[..., None, :]  # [tK, 3, 3]
    covars3d = torch.bmm(covars3d, covars3d.transpose(-1, -2))
    return covars3d


def compute_gaussian_weights_2d_tile(
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    img_w: int,
    img_h: int,
    means2d: torch.FloatTensor,
    conics2d: torch.FloatTensor,
):
    """Compute 2D gaussian weights for each pixel within tile.

    Args:
        [1] tile_x: int.
        [2] tile_y: int.
        [3] tile_w: int.
        [4] tile_h: int.
        [5] img_w:  int.
        [6] img_h:  int.
        [7] means2d: Dim=[tK, 2].
        [8] conics2d: Dim=[tK, 2, 2].

    Returns:
        [1] gauss_weights: Dim = [tH, tW, tK]
    """
    tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
        tile_x, tile_y, tile_w, tile_h, img_w, img_h
    )

    tile_ys, tile_xs = torch.meshgrid(
        [
            torch.arange(crop_h, dtype=means2d.dtype, device=means2d.device),
            torch.arange(crop_w, dtype=means2d.dtype, device=means2d.device),
        ],
        indexing="ij",
    )

    tile_xs = tile_xs + tile_xmin + 0.5
    tile_ys = tile_ys + tile_ymin + 0.5

    # Dim = [tH, tW, 2]
    tile_pixels = torch.stack([tile_xs, tile_ys], dim=-1)

    # Dim = [1, 1, tK, 2] - [tH, tW, 1, 2] -> [tH, tW, tK, 2]
    tile_pixels = means2d[None, None, :, :] - tile_pixels[:, :, None, :]

    # Dim = [tK, 2, 2] * [tH, tW, tK, 1, 2]
    tile_sigmas = 0.5 * torch.sum(conics2d * tile_pixels[:, :, :, None, :], dim=-1)

    tile_gausses = torch.exp(-tile_sigmas)
    return tile_gausses


def compute_pinhole_jacobian(
    cam_mats: torch.FloatTensor,
    cam_means3d: torch.FloatTensor,
    img_w: int,
    img_h: int,
):
    """Compute pinhole projection jacobian.

    Args:
        [1] cam_mats: Dim = [mN, 3, 3]
        [2] cam_means3d: Dim = [mN, tK, 3]

    Returns:
        [1] pin_jacob: Dim = [mN, tK, 2, 3]

    PINHOLE:
        u = fx * x / z + cx
        v = fy * y / z + cy

        J = [[fx / z,    0,    -fx * x / z^2],
             [0,      fy / z,  -fy * y / z^2]]
    """
    # Dim = [mN, 1]
    fx = cam_mats[:, 0, 0][..., None]
    fy = cam_mats[:, 1, 1][..., None]
    cx = cam_mats[:, 0, 2][..., None]
    cy = cam_mats[:, 1, 2][..., None]

    # Dim = [mN, tK]
    tx = cam_means3d[:, :, 0]
    ty = cam_means3d[:, :, 1]
    tz = cam_means3d[:, :, 2]

    # Dim = [mN, 1]
    tan_fovx = 0.5 * img_w / fx
    tan_fovy = 0.5 * img_h / fy

    lim_x_pos = (img_w - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (img_h - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    # Dim = [mN, tK, 2, 3]
    pin_jacob = torch.zeros(
        size=[cam_mats.shape[0], cam_means3d.shape[1], 2, 3],
        dtype=cam_means3d.dtype,
        device=cam_means3d.device,
    )
    pin_jacob[:, :, 0, 0] = fx / (tz + EPS)
    # pin_jacob[:, :, 0, 1] = 0.0
    pin_jacob[:, :, 0, 2] = -fx * tx / (tz**2 + EPS)
    # pin_jacob[:, :, 1, 0] = 0.0
    pin_jacob[:, :, 1, 1] = fy / (tz + EPS)
    pin_jacob[:, :, 1, 2] = -fy * ty / (tz**2 + EPS)
    return pin_jacob


def project_gaussians_3d_to_2d(
    view_mats: torch.FloatTensor,
    cam_mats: torch.FloatTensor,
    means3d: torch.FloatTensor,
    covars3d: torch.FloatTensor,
    img_w: int,
    img_h: int,
    near_plane: float = 0.1,
    far_plane: float = 1e10,
):
    """Project 3D means and covariance matrix to 2D.

    Args:
        [1] view_mats: Dim = [mN, 4, 4]
        [2] cam_mats: Dim = [mN, 3, 3]
        [3] means3d:  Dim = [tK, 3]
        [4] covars3d: Dim = [tK, 3, 3]
        [5] img_w: int
        [6] img_h: int

    Return:
        [1] covars2d: Dim = [mN, tK, 2, 2]
        [2] conics2d: Dim = [mN, tK, 2, 2]

        covars2d = J @ W @ covars3d @ W.T @ J.T
    """
    cam_means3d, means2d = project_points_3d_to_2d(means3d, view_mats, cam_mats)

    # Dim = [mN, 3, 3]
    rot_mats = view_mats[:, :3, :3]
    # Dim = [mN, 3, 3] @ [tK, 3, 3] @ [mN, 3, 3].T -> [mN, tK, 3, 3]
    cam_covars3d = torch.einsum(
        "ijk,lkm,imn->iljn", rot_mats, covars3d, rot_mats.transpose(-1, -2)
    )

    # Dim = [mN, tK, 2, 3]
    pin_jacob = compute_pinhole_jacobian(cam_mats, cam_means3d, img_w, img_h)
    # Dim = [mN, tK, 2, 3] @ [mN, tK, 3, 3] @ [mN, tK, 2, 3].T
    covars2d = torch.einsum(
        "ijkl,ijlm,ijmn->ijkn", pin_jacob, cam_covars3d, pin_jacob.transpose(-1, -2)
    )

    # covars2d = [a, b, c, d]
    # inverse covars2d = 1.0 / (a * d - b * c) * [d, -b, -c, a]
    conics2d = torch.zeros_like(covars2d)

    a = covars2d[:, :, 0, 0] + 0.3
    b = covars2d[:, :, 0, 1]
    c = covars2d[:, :, 1, 0]
    d = covars2d[:, :, 1, 1] + 0.3

    # Dim = [mN, tK]
    det = 1.0 / (a * d - b * c + EPS)

    conics2d[:, :, 0, 0] = d
    conics2d[:, :, 0, 1] = -b
    conics2d[:, :, 1, 0] = -c
    conics2d[:, :, 1, 1] = a

    # Dim = [mN, tK, 2, 2] * [mN, tK, 1, 1]
    conics2d = conics2d * det[:, :, None, None]

    # Dim = [mN, tK]
    depths = cam_means3d[..., 2]

    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))
    # Dim = [mN, tK, 2]
    radius = torch.stack([radius_x, radius_y], dim=-1)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < img_w)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < img_h)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return means2d, conics2d, depths, radii


def combine_sh_colors_from_coefficients(
    view_dirs: torch.FloatTensor,
    sh_coeffs: torch.FloatTensor,
    sh_consts: Optional[torch.FloatTensor] = None,
):
    """Combine SH colors from (L + 1)^2 coefficients.

    Args:
        [1] view_dirs: Dim = [mN, tK, 3]
        [2] sh_coeffs: Dim = [tK, (L + 1)^2, mC]
        [3] sh_consts: Dim = [(L + 1)^2]

    Returns:
        [1] sh_colors: Dim = [mN, tK, mC]
    """
    n_ords = sh_coeffs.shape[-2]

    sh_deg = int(math.sqrt(n_ords)) - 1

    assert sh_deg <= 3, "Only support rotation degree <= 3!"

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

    if sh_consts is None:
        sh_consts = torch.tensor(
            SPHERICAL_HARMONICS_CONSTANTS[:n_ords],
            dtype=sh_coeffs.dtype,
            device=sh_coeffs.device,
        )

    x = view_dirs[..., 0]
    y = view_dirs[..., 1]
    z = view_dirs[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # SPHERICAL_HARMONICS_CARTESIAN_LAMBDAS = [
    #     1.0,  # Y(0, 0)
    #     y / r,  # Y(1, -1)
    #     z / r,  # Y(1, 0)
    #     x / r,  # Y(1, 1)
    #     x * y / r**2,  # Y(2, -2)
    #     y * z / r**2,  # Y(2, -1)
    #     (3 * z**2 - r**2) / r**2,  # Y(2, 0)
    #     x * z / r**2,  # Y(2, 1)
    #     (x**2 - y**2) / r**2,  # Y(2, 2)
    #     y * (3 * x**2 - y**2) / r**3,  # Y(3, -3)
    #     x * y * z / r**3,  # Y(3, -2)
    #     y * (5 * z**2 - r**2) / r**3,  # Y(3, -1)
    #     (5 * z**3 - 3 * z * r**2) / r**3,  # Y(3, 0)
    #     x * (5 * z**2 - r**2) / r**3,  # Y(3, 1)
    #     (x**2 - y**2) * z / r**3,  # Y(3, 2)
    #     x * (x**2 - 3 * y**2) / r**3,  # Y(3, 3)
    # ]

    n_views, n_splats, _ = view_dirs.shape

    # Dim = [(L + 1)^2, mN, tK]
    sh_vals = torch.zeros(
        size=[n_ords, n_views, n_splats], dtype=sh_coeffs.dtype, device=sh_coeffs.device
    )
    if sh_deg >= 0:
        sh_vals[0] = 1.0  # Y(0, 0)

    if sh_deg >= 1:
        sh_vals[1] = y / (r + EPS)  # Y(1, -1)
        sh_vals[2] = z / (r + EPS)  # Y(1, 0)
        sh_vals[3] = x / (r + EPS)  # Y(1, 1)

    r2 = None
    if sh_deg >= 2:
        r2 = r**2

        sh_vals[4] = x * y / (r2 + EPS)  # Y(2, -2)
        sh_vals[5] = y * z / (r2 + EPS)  # Y(2, -1)
        sh_vals[6] = (3 * z**2 - r2) / (r2 + EPS)  # Y(2, 0)
        sh_vals[7] = x * z / (r2 + EPS)  # Y(2, 1)
        sh_vals[8] = (x**2 - y**2) / (r2 + EPS)  # Y(2, 2)

    r3 = None
    if sh_deg >= 3:
        r3 = r**3

        sh_vals[9] = y * (3 * x**2 - y**2) / (r3 + EPS)  # Y(3, -3)
        sh_vals[10] = x * y * z / (r3 + EPS)  # Y(3, -2)
        sh_vals[11] = y * (5 * z**2 - r2) / (r3 + EPS)  # Y(3, -1)
        sh_vals[12] = (5 * z**3 - 3 * z * r2) / (r3 + EPS)  # Y(3, 0)
        sh_vals[13] = x * (5 * z**2 - r2) / (r3 + EPS)  # Y(3, 1)
        sh_vals[14] = (x**2 - y**2) * z / (r3 + EPS)  # Y(3, 2)
        sh_vals[15] = x * (x**2 - 3 * y**2) / (r3 + EPS)  # Y(3, 3)

    # Dim = [(L + 1)^2] * [mN, tK, (L + 1)^2] -> [mN, tK, (L + 1)^2]
    sh_vals = sh_consts * sh_vals.permute([1, 2, 0])

    # Dim = [mN, tK, (L + 1)^2, 1] * [1, tK, (L + 1)^2, mC]
    #    -> [mN, tK, mC]
    sh_colors = torch.sum(sh_vals[:, :, :, None] * sh_coeffs[None, :, :, :], dim=-2)
    return sh_colors


def rasterize_to_pixels_tile_forward(
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    img_w: int,
    img_h: int,
    means3d: torch.FloatTensor,  # Dim = [tK, 3]
    quats: torch.FloatTensor,  # Dim = [tK, 4]
    scales: torch.FloatTensor,  # Dim = [tK, 3]
    opacities: torch.FloatTensor,  # Dim = [tK]
    sh_coeffs: torch.FloatTensor,  # Dim = [tK, (L + 1)^2, 3]
    cam_mats: torch.FloatTensor,  # Dim = [mN, 3, 3]
    view_mats: torch.FloatTensor,  # Dim =[mN, 4, 4]
):
    """Torch tile-based rasterization implementation."""
    # Dim = [tK, 3, 3]
    covars3d = compute_covariance_3d(quats, scales)

    # Dim = [mN, tK, 2]
    means2d = project_means_3d_to_2d(means3d, view_mats, cam_mats)

    # Dim = [mN, tK, 2, 2]
    conics2d = project_covariances_3d_to_2d(view_mats, cam_mats, means3d, covars3d)

    # Dim = [mN, tH, tW, tK]
    tile_gausses2d = compute_gaussian_weights_2d_tile(
        tile_x, tile_y, tile_w, tile_h, img_w, img_h, means2d, conics2d
    )

    view_dirs = means3d[None, :, :] - torch.linalg.inv(view_mats)[:, :3, 3][:, None, :]
    # Dim = [mN, tK, mC]
    sh_colors = combine_sh_colors_from_coefficients(view_dirs, sh_coeffs)

    # Dim = [mN, tH, tW, tK]
    tile_alphas = tile_gausses2d * opacities
    tile_alphas = torch.clamp_max(tile_alphas, 0.9999)

    # Dim = [mN, tH, tW, tK]
    blend_alphas = 1.0 - tile_alphas
    blend_alphas = torch.cumprod(1.0 - tile_alphas, dim=-1)
    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
    blend_alphas[:, :, 0] = 1.0

    tile_rgb = torch.sum(
        tile_alphas[:, :, :, :, None]  # Dim = [mN,  tH, tW, tK, 1]
        * blend_alphas[:, :, :, :, None]  # Dim = [mN,  tH, tW, tK, 1]
        * sh_colors[:, None, None, :, :],  # Dim = [mN, 1,  1,  tK, mC]
        dim=-2,
    )

    tile_a = 1.0 - blend_alphas[:, :, :, -1, None]
    return tile_rgb, tile_a
