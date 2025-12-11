import torch
from typing import Optional


def _backward_render_to_sh_colors(
    num_chs: int,
    alphas: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
):
    """Backward pass of rendered pixels w.r.t spherical harmonics colors.

    Args:
        [1] num_chs: Number of color channels, int.
        [2] alphas: Pixelwise alphas within a tile, Dim=[mN, tH, tW, tK].
        [3] masks: Valid splats maks, Dim = [tK].

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

    # Dim = [mN, tH, tW, tK]
    jacob = (
        alphas
        * blend_alphas
        * (masks[:, None, None, :].float() if masks is not None else 1.0)
    )
    jacob = jacob[..., None].repeat([1, 1, 1, 1, num_chs])

    # Dim = [mN, tH, tW, tK, mC, mC]
    hess = torch.zeros(
        size=list(alphas.shape) + [num_chs, num_chs],
        dtype=alphas.dtype,
        device=alphas.device,
    )
    return jacob, hess
