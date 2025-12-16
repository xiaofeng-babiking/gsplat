import math
import torch
import torch.nn.functional as F
from typing import Optional, Literal
from torch.nn.modules.loss import _Loss

EPS = 1e-12


def get_tile_size(
    tile_x: int, tile_y: int, tile_size_w: int, tile_size_h: int, img_w: int, img_h: int
):
    """Get valid tile size within image."""
    tile_xmin = tile_x * tile_size_w
    tile_ymin = tile_y * tile_size_h
    tile_xmax = min(img_w, tile_xmin + tile_size_w)
    tile_ymax = min(img_h, tile_ymin + tile_size_h)

    crop_w = tile_xmax - tile_xmin
    crop_h = tile_ymax - tile_ymin
    return tile_xmin, tile_ymin, crop_w, crop_h


def quaternion_to_rotation_matrix(
    quats: torch.Tensor,
) -> torch.Tensor:
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
    quats: torch.Tensor,  # Dim = [tK, 4]
    scales: torch.Tensor,  # Dim = [tK, 3]
):
    """Compute 3D covariance from scale and quaternion."""
    rot_mats = quaternion_to_rotation_matrix(quats)

    covars3d = rot_mats * scales[..., None, :]  # [tK, 3, 3]
    covars3d = torch.bmm(covars3d, covars3d.transpose(-1, -2))
    return covars3d


def compute_gaussian_weights_2d_tile(
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
    means2d: torch.Tensor,
    conics2d: torch.Tensor,
):
    """Compute 2D gaussian weights for each pixel within tile.

    Args:
        [1] tile_x: int.
        [2] tile_y: int.
        [3] tile_size_w: int.
        [4] tile_size_h: int.
        [5] img_w:  int.
        [6] img_h:  int.
        [7] means2d: Dim=[mN, tK, 2].
        [8] conics2d: Dim=[mN, tK, 2, 2].

    Returns:
        [1] gauss_weights: Dim = [mN, tH, tW, tK]
    """
    tile_xmin, tile_ymin, crop_w, crop_h = get_tile_size(
        tile_x, tile_y, tile_size_w, tile_size_h, img_w, img_h
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

    # Dim = [mN, 1, 1, tK] - [1, tH, tW, 1] -> [mN, tH, tW, tK]
    tile_xs = means2d[:, :, 0][:, None, None, :] - tile_xs[None, :, :, None]
    tile_ys = means2d[:, :, 1][:, None, None, :] - tile_ys[None, :, :, None]
    tile_pixels = torch.stack([tile_xs, tile_ys], dim=-1)

    # Dim = [mN, tH, tW, tK] * [mN, 1, 1, tK] -> [mN, tH, tW, tK]
    sigmas = 0.5 * (
        tile_xs * tile_xs * conics2d[:, :, 0][:, None, None, :]
        + 2.0 * tile_xs * tile_ys * conics2d[:, :, 1][:, None, None, :]
        + tile_ys * tile_ys * conics2d[:, :, 2][:, None, None, :]
    )

    # Dim = [mN, tH, tW, tK]
    gausses2d = torch.exp(-sigmas)
    tile_bbox = (tile_xmin, tile_ymin, crop_w, crop_h)
    return gausses2d, tile_pixels, tile_bbox


def compute_pinhole_jacobian(
    cam_mats: torch.Tensor,
    cam_means3d: torch.Tensor,
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


def compute_covariance_2d_inverse(
    covars2d: torch.Tensor,  # Dim = [mN, tK, 2, 2]
):
    """Compute inverse of the 2D covariance matrix."""
    # covars2d = [a, b, c, d]
    # inverse covars2d = 1.0 / (a * d - b * c) * [d, -b, -c, a]
    conics2d = torch.zeros(
        size=list(covars2d.shape[:-2]) + [2, 2],
        dtype=covars2d.dtype,
        device=covars2d.device,
    )

    a = covars2d[:, :, 0, 0] + 0.3
    b = covars2d[:, :, 0, 1]
    c = covars2d[:, :, 1, 0]
    d = covars2d[:, :, 1, 1] + 0.3
    # Dim = [mN, tK]
    det = a * d - b * c

    conics2d[:, :, 0, 0] = d
    conics2d[:, :, 0, 1] = -(b + c) / 2.0
    conics2d[:, :, 1, 0] = -(b + c) / 2.0
    conics2d[:, :, 1, 1] = a

    # Dim = [mN, tK, 2, 2] * [mN, tK, 1, 1]
    conics2d = conics2d / (det[:, :, None, None] + EPS)
    return conics2d, det


def transform_means_3d(means3d: torch.Tensor, view_mats: torch.Tensor):
    """Apply SE3 transform to mean3d vectors."""
    # Dim = [mN, 1, 3, 3] * [1, tK, 1, 3] + [mN, 1, 3] -> [mN, tK, 3]
    tf_means3d = (
        torch.sum(view_mats[:, None, :3, :3] * means3d[None, :, None, :], dim=-1)
        + view_mats[:, :3, 3][:, None, :]
    )
    return tf_means3d


def transform_covariance_3d(covars3d: torch.Tensor, view_mats: torch.Tensor):
    """Apply SE3 transform to covariance matrices."""
    # Dim = [mN, 3, 3]
    rot_mats = view_mats[:, :3, :3]
    # Dim = [mN, 3, 3] @ [tK, 3, 3] @ [mN, 3, 3].T -> [mN, tK, 3, 3]
    tf_covars3d = torch.einsum(
        "ijk,lkm,imn->iljn", rot_mats, covars3d, rot_mats.transpose(-1, -2)
    )
    return tf_covars3d


def project_means_3d(means3d: torch.Tensor, cam_mats: torch.Tensor):
    """Project points from world space into camera space."""
    # Dim = [mN, 1, 3, 3] * [mN, tK, 1, 3] -> [mN, tK, 3]
    means2d = torch.sum(cam_mats[:, None, :, :] * means3d[:, :, None, :], dim=-1)

    # Dim = [mN, tK, 2]
    means2d = means2d[:, :, :2] / means2d[:, :, 2][..., None]
    return means2d


def project_covariance_3d(
    means3d: torch.Tensor, covars3d: torch.Tensor, cam_mats: torch.Tensor
):
    """Project covariances from world space into camera space."""
    img_w = cam_mats[:, 0, 2] * 2.0
    img_h = cam_mats[:, 1, 2] * 2.0

    # Dim = [mN, tK, 2, 3]
    pin_jacob = compute_pinhole_jacobian(cam_mats, means3d, img_w, img_h)
    # Dim = [mN, tK, 2, 3] @ [mN, tK, 3, 3] @ [mN, tK, 2, 3].T
    covars2d = torch.einsum(
        "ijkl,ijlm,ijmn->ijkn", pin_jacob, covars3d, pin_jacob.transpose(-1, -2)
    )
    return covars2d, pin_jacob


def project_3d_to_2d_fused(
    view_mats: torch.Tensor,
    cam_mats: torch.Tensor,
    means3d: torch.Tensor,
    covars3d: torch.Tensor,
    img_w: int,
    img_h: int,
    near_plane: float = 0.1,
    far_plane: float = 1e10,
    opacities: Optional[torch.Tensor] = None,
    alpha_threshold: float = 1.0 / 255.0,
):
    """Project 3D means and covariance matrix to 2D.

    Args:
        [1] view_mats: Dim = [mN, 4, 4]
        [2] cam_mats: Dim = [mN, 3, 3]
        [3] means3d:  Dim = [tK, 3]
        [4] covars3d: Dim = [tK, 3, 3]
        [7] opacities: [tK]
        [8] alpha_threshold: float

    Return:
        [1] covars2d: Dim = [mN, tK, 2, 2]
        [2] conics2d: Dim = [mN, tK, 2, 2]

        covars2d = J @ W @ covars3d @ W.T @ J.T
    """
    cam_means3d = transform_means_3d(means3d, view_mats)
    cam_covars3d = transform_covariance_3d(covars3d, view_mats)

    means2d = project_means_3d(cam_means3d, cam_mats)
    covars2d, pin_jacob = project_covariance_3d(cam_means3d, cam_covars3d, cam_mats)

    # Dim = [mN, tK]
    depths = cam_means3d[..., 2]

    conics2d, det = compute_covariance_2d_inverse(covars2d)
    conics2d = conics2d[..., [0, 0, 1], [0, 1, 1]]

    extend = 3.3
    if opacities is not None:
        extend = extend * torch.ones_like(opacities)

        # Dim = [1, tK]
        extend = torch.minimum(
            extend, torch.sqrt(2.0 * torch.log(opacities / alpha_threshold))
        )[None]

    radius_x = torch.ceil(extend * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(extend * torch.sqrt(covars2d[..., 1, 1]))
    # Dim = [mN, tK, 2]
    radius = torch.stack([radius_x, radius_y], dim=-1)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    if opacities is not None:
        valid = opacities.reshape([1, -1]) >= alpha_threshold
        radius[~valid, :] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < img_w)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < img_h)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return (
        means2d,
        conics2d,
        depths,
        radii,
        cam_covars3d,
        pin_jacob,
    )


def combine_sh_colors_from_coefficients(
    view_dirs: torch.Tensor,
    sh_coeffs: torch.Tensor,
    sh_consts: Optional[torch.Tensor] = None,
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


def compute_sh_colors(
    means3d: torch.Tensor,
    view_mats: torch.Tensor,
    sh_coeffs: torch.Tensor,
    clamped: bool = True,
):
    """Combine spherical harmonic colors."""
    view_dirs = means3d[None, :, :] - torch.linalg.inv(view_mats)[:, :3, 3][:, None, :]
    # Dim = [mN, tK, mC]
    sh_colors = combine_sh_colors_from_coefficients(view_dirs, sh_coeffs)
    if clamped:
        sh_colors = torch.clamp_min(sh_colors + 0.5, 0.0)
    return sh_colors


def compute_blend_alphas(alphas: torch.Tensor, clamped: bool = True):
    """Compute blended alphas by cumprod."""
    if clamped:
        alphas = torch.clamp_max(alphas, 0.9999)

    # Dim = [mN, tH, tW, tK]
    blend_alphas = torch.cumprod(1.0 - alphas, dim=-1)
    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
    blend_alphas[:, :, :, 0] = 1.0
    return alphas, blend_alphas


def blend_sh_colors_with_alphas(
    sh_colors: torch.Tensor,
    alphas: torch.Tensor,
    blend_alphas: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
):
    """Render spherical harmonics colors with alphas."""
    # Dim = [mN, tH, tW, mC]
    rd_colors = torch.sum(
        alphas[:, :, :, :, None]  # Dim = [mN,  tH, tW, tK, 1]
        * blend_alphas[:, :, :, :, None]  # Dim = [mN,  tH, tW, tK, 1]
        * sh_colors[:, None, None, :, :]  # Dim = [mN, 1,  1,  tK, mC]
        * (
            masks[:, None, None, :, None].float() if masks is not None else 1.0
        ),  # Dim = [mN, 1, 1, tK, 1]
        dim=-2,
    )

    # Dim = [mN, tH, tW, tK]
    rd_alphas = 1.0 - blend_alphas[:, :, :, -1][..., None]
    return rd_colors, rd_alphas


def rasterize_to_pixels_tile_forward(
    tile_x: int,
    tile_y: int,
    tile_size_w: int,
    tile_size_h: int,
    img_w: int,
    img_h: int,
    means3d: torch.Tensor,  # Dim = [tK, 3]
    quats: torch.Tensor,  # Dim = [tK, 4]
    scales: torch.Tensor,  # Dim = [tK, 3]
    opacities: torch.Tensor,  # Dim = [tK]
    sh_coeffs: torch.Tensor,  # Dim = [tK, (L + 1)^2, 3]
    cam_mats: torch.Tensor,  # Dim = [mN, 3, 3]
    view_mats: torch.Tensor,  # Dim =[mN, 4, 4]
    masks: Optional[torch.Tensor] = None,
):
    """Torch tile-based rasterization implementation."""
    # Dim = [tK, 3, 3]
    covars3d = compute_covariance_3d(quats, scales)

    means2d, conics2d, depths, radii, cam_covars3d, pin_jacob = project_3d_to_2d_fused(
        view_mats,
        cam_mats,
        means3d,
        covars3d,
        img_w,
        img_h,
        opacities=opacities,
    )

    assert torch.all(
        depths[:-1] <= depths[1:]
    ), f"Depth should be sorted within a tile!"

    # Dim = [mN, tK]
    if masks is None:
        masks = (radii > 0).all(dim=-1)

    # Dim = [mN, tH, tW, tK]
    gausses2d, tile_pixels, tile_bbox = compute_gaussian_weights_2d_tile(
        tile_x, tile_y, tile_size_w, tile_size_h, img_w, img_h, means2d, conics2d
    )

    sh_colors = compute_sh_colors(means3d, view_mats, sh_coeffs)

    # Dim = [mN, tH, tW, tK]
    alphas = gausses2d * opacities

    alphas, blend_alphas = compute_blend_alphas(alphas)

    rd_colors, rd_alphas = blend_sh_colors_with_alphas(
        sh_colors, alphas, blend_alphas, masks
    )

    # tile_rgb = torch.clamp(tile_rgb, min=0.0, max=1.0)
    rd_colors = rd_colors.permute([0, 3, 1, 2])

    # Dim = [mN, tH, tW, 1]
    rd_alphas = rd_alphas.permute([0, 3, 1, 2])

    rd_meta = {
        "tile_pixels": tile_pixels,
        "tile_bbox": tile_bbox,
        "alphas": alphas,
        "blend_alphas": blend_alphas,
        "gaussians2d": gausses2d,
        "radii": radii,
        "masks": masks,
        "means2d": means2d,
        "conics2d": conics2d,
        "covars3d": covars3d,
        "sh_colors": sh_colors,
        "pinhole": pin_jacob,
        "cam_covars3d": cam_covars3d,
    }
    return rd_colors, rd_alphas, rd_meta


def compute_kernel_mean_2d(img: torch.Tensor, kernel: torch.Tensor):
    """Compute 2D kernel-weighted mean of an image."""
    c = img.shape[1]

    mean = torch.conv2d(
        img,
        torch.tile(kernel, (c, 1, 1, 1)),
        stride=1,
        padding="same",
        groups=c,
    )
    return mean


def compute_kernel_covariance_2d(
    img0: torch.Tensor,
    mean_0: Optional[torch.Tensor] = None,
    img1: Optional[torch.Tensor] = None,
    mean_1: Optional[torch.Tensor] = None,
    kernel: Optional[torch.Tensor] = None,
):
    """Compute covariance between two images."""
    if img1 is not None:
        assert (
            img0.shape == img1.shape
        ), "Source and target image must have the same shape!"

    if mean_0 is None:
        mean_0 = compute_kernel_mean_2d(img0, kernel)

    if img1 is not None and mean_1 is None:
        mean_1 = compute_kernel_mean_2d(img1, kernel)

    # No need to compute target mean when targe image is None
    mean_1 = None if img1 is None else mean_1

    if mean_1 is not None:
        assert mean_0.shape == mean_1.shape, "Mean tensors must have the same shape!"

    covar = (
        compute_kernel_mean_2d(img0 * img1 if img1 is not None else img0**2, kernel)
    ) - (mean_0 * mean_1 if mean_1 is not None else mean_0**2)
    return covar


class GroupSSIMLoss(_Loss):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        padding: Literal["same", "valid"] = "valid",
        kernel: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        *args,
        **kwargs,
    ):
        self._c1 = c1
        self._c2 = c2
        self._padding = padding
        self._reduction = reduction
        super().__init__(size_average, reduce, reduction, *args, **kwargs)

        self._kernel_size = kernel_size
        self._mu = kernel_size // 2
        self._sigma = sigma

        if kernel is not None:
            self._kernel = kernel
        else:
            self._filter = (
                1.0
                / (torch.sqrt(2 * torch.pi * sigma**2))
                * torch.exp(
                    -((torch.arange(self._kernel_size) - self._mu) ** 2)
                    / (2 * self._sigma**2)
                )
            )
            self._kernel = torch.outer(self._filter, self._filter)[None, None, ...]

        self._kernel = torch.nn.Parameter(self._kernel, requires_grad=False)

    def __str__(self):
        """Return string representation of SSIM kernel."""
        return (
            "SSIMLoss with Gaussian kernel "
            + f"size={self._kernel_size}, mean={self._kernel_size // 2}, variance={self._sigma:.6f}."
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss between source and target image."""

        assert (
            input.shape == target.shape
        ), "source and target image must have the same shape!"

        src_mean = compute_kernel_mean_2d(input, kernel=self._kernel)
        dst_mean = compute_kernel_mean_2d(target, kernel=self._kernel)

        src_var = compute_kernel_covariance_2d(input, src_mean, kernel=self._kernel)
        dst_var = compute_kernel_covariance_2d(target, dst_mean, kernel=self._kernel)
        src_dst_covar = compute_kernel_covariance_2d(
            input, src_mean, target, dst_mean, kernel=self._kernel
        )

        f0 = self._c1 + 2.0 * src_mean * dst_mean
        f1 = self._c2 + 2.0 * src_dst_covar
        f2 = self._c1 + src_mean**2 + dst_mean**2
        f3 = self._c2 + src_var + dst_var
        ssim_map = (f0 * f1) / (f2 * f3)

        half_ksize = self._kernel_size // 2
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

        ssim_loss = 1.0 - ssim_score
        return ssim_loss
