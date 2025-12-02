import torch
import numpy as np
from scipy.optimize import curve_fit
from torch.nn.modules.loss import _Loss
from enum import Enum
from collections.abc import Iterable
from typing import Dict, Any, Optional, Literal, Tuple
from gsplat.rendering import spherical_harmonics


class GroupSSIMLoss(_Loss):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        padding: Literal["same", "valid"] = "valid",
        filter: Optional[torch.FloatTensor] = None,
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

        _kernel_size, _sigma, _filter, _kernel = GroupSSIMLoss.create_gauss_kernel_2d(
            kernel_size=kernel_size,
            sigma=sigma,
            filter=filter,
        )
        self._kernel_size = _kernel_size
        self._sigma = _sigma
        self._filter = torch.nn.Parameter(_filter, requires_grad=False)
        self._kernel = torch.nn.Parameter(_kernel, requires_grad=False)

    @staticmethod
    def create_gauss_kernel_2d(
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None,
        filter: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Create 2D Gaussian Kernel for SSIM convolution."""
        if filter is not None:
            _, sigma = GroupSSIMLoss._fit_gauss_filter_1d(
                filter.cpu().detach().numpy(),
            )

            kernel_size = len(filter)
        else:
            filter = GroupSSIMLoss._evaluate_gauss_filter_1d(
                x=np.arange(kernel_size),
                mu=kernel_size // 2,
                sigma=sigma,
            )
            filter = torch.from_numpy(filter).float()

        # [out_channel, in_channel, kernel_size, kernel_size] -> [1, 1, K, K]
        kernel = torch.outer(filter, filter)[None, None, ...]
        return kernel_size, sigma, filter, kernel

    @staticmethod
    def _evaluate_gauss_filter_1d(x: np.ndarray, mu: float, sigma: float):
        """Evaluate a 1D Gaussian distribution at given points `x`."""
        y = (
            1.0
            / (np.sqrt(2 * np.pi * sigma**2))
            * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        )
        return y

    @staticmethod
    def _fit_gauss_filter_1d(filter: np.ndarray) -> Tuple[float, float]:
        """Given a Gaussian filter in 1D, find its mean and variance."""
        n_items = len(filter)

        opt_mu = n_items // 2  # Central point of the Gaussian filter

        fit_func = lambda x, sigma: GroupSSIMLoss._evaluate_gauss_filter_1d(
            x, mu=opt_mu, sigma=sigma
        )

        popt, _ = curve_fit(
            fit_func,
            xdata=np.arange(len(filter)),
            ydata=filter,
        )

        opt_sigma = popt[0]
        return opt_mu, opt_sigma

    def __str__(self):
        """Return string representation of SSIM kernel."""
        return (
            "SSIMLoss with Gaussian kernel "
            + f"size={self._kernel_size}, mean={self._kernel_size // 2}, variance={self._sigma:.6f}."
        )

    @staticmethod
    def compute_mean(img: torch.FloatTensor, kernel: torch.FloatTensor):
        """Compute mean of an image."""
        c = img.shape[1]

        mean = torch.conv2d(
            img,
            torch.tile(kernel, (c, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=c,
        )
        return mean

    @staticmethod
    def compute_covariance(
        img0: torch.FloatTensor,
        mean_0: Optional[torch.FloatTensor] = None,
        img1: Optional[torch.FloatTensor] = None,
        mean_1: Optional[torch.FloatTensor] = None,
        kernel: Optional[torch.FloatTensor] = None,
    ):
        """Compute covariance between two images."""
        if img1 is not None:
            assert (
                img0.shape == img1.shape
            ), "Source and target image must have the same shape!"

        if mean_0 is None:
            mean_0 = GroupSSIMLoss.compute_mean(img0, kernel)

        if img1 is not None and mean_1 is None:
            mean_1 = GroupSSIMLoss.compute_mean(img1, kernel)

        # No need to compute target mean when targe image is None
        mean_1 = None if img1 is None else mean_1

        if mean_1 is not None:
            assert (
                mean_0.shape == mean_1.shape
            ), "Mean tensors must have the same shape!"

        covar = (
            GroupSSIMLoss.compute_mean(
                img0 * img1 if img1 is not None else img0**2, kernel
            )
        ) - (mean_0 * mean_1 if mean_1 is not None else mean_0**2)
        return covar

    def forward(
        self, input: torch.FloatTensor, target: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute SSIM loss between source and target image."""

        assert (
            input.shape == target.shape
        ), "source and target image must have the same shape!"

        src_mean = self.compute_mean(input, kernel=self._kernel)
        dst_mean = self.compute_mean(target, kernel=self._kernel)

        src_var = self.compute_covariance(input, src_mean, kernel=self._kernel)
        dst_var = self.compute_covariance(target, dst_mean, kernel=self._kernel)
        src_dst_covar = self.compute_covariance(
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


class GSParameterGroup(Enum):
    POSITION = 0  # 3DOFs position (x, y, z)
    ROTATION = 1  # 3DOFs rotation axis (nx, ny, nz) with angle θ
    SCALE = 2  # 3DOFs scale (sx, sy, sz)
    OPACITY = 3  # 1DOF opacity
    COLOR = 4  # Spherical Harmonic coefficients Dimension = (L + 1)^2


class GSGroupNewtonOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Dict[str, Any]],
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the optimizer.

        Args:
            [1] params: Iterable of dicts defining parameter groups.
                e.g.
                    [
                      {"POSITION": [torch.FloatTensor], "dimension": 3},
                      {"ROTATION": [torch.FloatTensor], "dimension": 3},
                      {"SCALE": [torch.FloatTensor], "dimension": 3},
                      {"OPACITY": [torch.FloatTensor], "dimension": 1},
                      {"COLOR": [torch.FloatTensor], "dimension": 16},
                    ]
        """
        super().__init__(params, defaults)

    @staticmethod
    def _backward_l2_to_render(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
        with_hessian: bool = True,
    ):
        """Computes Jacobian matrix from L2 to rendering RGB pixels.

        Args:
            [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
            [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.
            [3] with_hessian: Whether to compute Hessian matrix, bool.

        Returns:
            [1] jacob: Jacobian matrix from L2 to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.
                    L2 = 0.5 * MEAN(||rd_imgs - gt_imgs||^2)
                    RGB =  { pixel=(r, g, b) | for any pixel in rd_imgs }
                    RGB' = { pixel=(r, g, b) | for any pixel in gt_imgs }
                    ∂L2 / ∂(RGB) -> partial derivative of L2 (scalar) with respect to C (matrix)

                    ∂L2 / ∂(RGB)
                        = [∂L2 / ∂R, ∂L2 / ∂G, ∂L2 / ∂B]
                        = (rd_imgs - gt_imgs) / (N * C * H * W)
            [2] hess: Hessian matrix from L2 to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.
                    ∂²L2 / ∂(RGB)²
                    = [[∂²L2 / ∂R²,  ∂L2 / ∂R∂G, ∂L2 / ∂R∂B],
                       [∂²L2 / ∂R∂G, ∂²L2 / ∂G², ∂L2 / ∂G∂B],
                       [∂²L2 / ∂R∂B, ∂L2 / ∂G∂B, ∂²L2 / ∂B²]]
                    Since R, G and B are independent with each other,
                    = diagonal([1, 1, 1]).view(N, C, C, H, W)

                    NOTE!!!
                    R, G, B channel is irrelevant to each other,
                    therefore hessian_L2_to_RGB is diagonal,
                    dimension [N, C, C, H, W] could be compressed as [N, C, H, W].
                    no need to allocate memory for every pixel.
        """
        n, c, h, w = rd_imgs.shape

        factor = 1.0 / (n * c * h * w)

        jacob = factor * (rd_imgs - gt_imgs)

        hess = None
        if with_hessian:
            # NOTE!!!
            # hessian_L2_to_RGB dimension=[N, C, C, H, W]
            # but hessian_L2_to_RGB is repeated constant,
            #   i.e. eye(3).view(1, C, C, 1, 1).repeated(N, 1, 1, H, W)
            #
            hess = torch.ones(
                size=[1, c, 1, 1], dtype=rd_imgs.dtype, device=rd_imgs.device
            )
        return jacob, hess

    @staticmethod
    def _backward_ssim_to_render(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        kernel_size: int = 11,
        sigma: float = 1.5,
        padding: Literal["valid", "same"] = "valid",
        filter: Optional[torch.FloatTensor] = None,
        with_hessian: bool = True,
        eps: float = 1e-12,
    ):
        """Computes Jacobian matrix from SSIM to rendering RGB pixels.

        Args:
            [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
            [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.
            [3] c1: SSIM constant term 1, float.
            [4] c2: SSIM constant term 2, float.
            [5] kernel_size: Size of Gaussian kernel, int.
            [6] sigma: Standard deviation of Gaussian kernel, float.
            [7] padding: Padding mode, Literal["valid", "same"].
            [8] filter: 1D Gaussian filter, torch.FloatTensor.
            [9] with_hessian: Whether to compute Hessian matrix, bool.

        Returns:
            [1] jacob: Jacobian matrix from SSIM to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.
                    ∂Ls / ∂(RGB) = [∂Ls / ∂R, ∂Ls / ∂G, ∂Ls / ∂B]

                    For a given pixel P=(m, n), R(m, n) denotes its Red value,

                    ∂Ls / ∂R = { ∂Ls / ∂R(m, n) | for all pixels P=(m, n) in rendering image }
                    Let,
                        a = m + i - kW // 2
                        b = n + j - kH // 2
                    Then,
                        ∂Ls / ∂R(m, n)
                            = SUM_i_j( ∂fs(m + i - kW // 2, n + j - kH // 2, R) / ∂R(m, n) )
                            = SUM_a_b(
                                ∂Ls / ∂fs(a, b, R) 
                                * ∂fs(a, b, R) / ∂R(m, n)
                              )
                    
                    fs(a, b, R) 
                        = (c1 + 2.0 * μ(a, b, R) * μ'(a, b, R)) * (c2 + 2.0 * Σ(a, b, R)) \
                            / (c1 + μ(a, b, R) ** 2 + μ'(a, b, R) ** 2) * (c2 + σ(a, b, R) + σ'(a, b, R))
                    where,
                        μ, μ' = mean of Rendering/Groundtruth kernel Red values
                        σ, σ' = variance of Rendering/Groundtruth kernel Red values
                        Σ = covariance between Rendering and Groundtruth kernel Red values
                        c1, c2 = constant terms
                    
                    Let,
                        f0(a, b, R) = (c1 + 2.0 * μ(a, b, R) * μ'(a, b, R)))
                        f1(a, b, R) = (c2 + 2.0 * Σ(a, b, R))
                        f2(a, b, R) = (c1 + μ(a, b, R) ** 2 + μ'(a, b, R) ** 2)
                        f3(a, b, R) = (c2 + σ(a, b, R) + σ'(a, b, R))
                    Then,
                        fs(a, b, R) = (f0 * f1) / (f2 * f3)

                    μ(a, b, R) 
                        <- W(m, n, a, b) * R(m, n) 
                            # W(m, n, a, b) weight of R(m, n) contributing to R(a, b)'s mean
                        <- W(m - a + kW // 2, n - b + kW // 2) * R(m, n)
                        <- W(i, j) * R(m, n)
                    ∂μ(a, b, R) / ∂R(m, n) = W(i, j)
                    g0 = ∂f0(a, b, R) / ∂R(m, n)
                       = 2.0 * μ'(a, b, R) * (∂μ / ∂R(m, n))
                       = W(i, j) * 2.0 * μ'(a, b, R)
                    
                    And,
                        SUM_i_j(W(i, j) * 2.0 * μ'(a, b, R) * ?(a, b, R))
                            = W ⊗ (2.0 * μ'(a, b, R) * ?(a, b, R)))
                    Thus,
                        g0 = lambda x: 2.0 * W ⊗ (μ'(a, b, R) * x) 
                            i.e. in (a, b) coordinate space
                                (W ⊗) is a mapping from (a, b) to (m, n) space

                    
                    Σ(a, b, R)
                        <- W(i, j) * (R(m, n) * R'(m, n)) - μ'(a, b, R) * μ(a, b, R)
                        <- R'(m, n) * W(i, j) * R(m, n) - μ'(a, b, R) * μ(a, b, R)
                    ∂Σ(a, b, R) / ∂R(m, n)
                        = R'(m, n) * W(i, j) - μ'(a, b, R) * ∂μ(a, b, R) / ∂R(m, n)
                        = W(i, j) * R'(m, n) - W(i, j) * μ'(a, b, R))
                    SUM_i_j(W(i, j) * R'(m, n) * ?(a, b, R) - W(i, j) * μ'(a, b, R) * ?(a, b, R))
                        = R'(m, n) * (W ⊗ ?(a, b, R)) - W ⊗ (μ'(a, b, R) * ?(a, b, R))
                    Then,
                        g1 = lambda x: 2.0 * R'(m, n) * W ⊗ x - 2.0 * W ⊗ (μ'(a, b, R) * x)
                    
                    g2 = lambda x: 2.0 * W ⊗ (μ(a, b, R) * x)
                    g3 = lambda x: 2.0 * R(m, n) * W ⊗ x - 2.0 * W ⊗ (μ(a, b, R) * x)
                    
                    jacob = g0(f1 / f2f3) + g1(f0 / f2f3) - g2(f0f1 / f2²f3) - g3(f0f1 / f2f3²)
            
            [2] hess: Hessian matrix from SSIM to rendering RGB pixels, Dim=[N, C, C, H, W], torch.FloatTensor.

                    NOTE!!! 
                    R, G, B channel is irrelevant to each other,
                    therefore hessian_SSIM_to_RGB is diagonal,
                    dimension [N, C, C, H, W] could be compressed as [N, C, H, W].
                    no need to allocate memory for every pixel.

                    h0 = ∂g0(m, n, R) / ∂R(m, n) = 0
                    h1 = ∂g1(m, n, R) / ∂R(m, n) = 0
                    h2 = ∂g2(m, n, R) / ∂R(m, n) 
                       = lambda x: 2.0 * W^2 ⊗ x
                    h3 = ∂g3(m, n, R) / ∂R(m, n)
                       = lambda x: 2.0 * (W - W^2) ⊗ x
        """
        nb, nc, h, w = rd_imgs.shape

        device = rd_imgs.device

        group_ssim = GroupSSIMLoss(
            kernel_size=kernel_size,
            sigma=sigma,
            c1=c1,
            c2=c2,
            padding=padding,
            filter=filter,
        ).to(device)

        kernel = group_ssim._kernel.clone().detach()
        ksize = group_ssim._kernel_size
        half_ksize = ksize // 2

        rd_means = group_ssim.compute_mean(rd_imgs, kernel=kernel)
        gt_means = group_ssim.compute_mean(gt_imgs, kernel=kernel)
        rd_vars = group_ssim.compute_covariance(rd_imgs, rd_means, kernel=kernel)
        gt_vars = group_ssim.compute_covariance(gt_imgs, gt_means, kernel=kernel)
        rd_gt_covars = group_ssim.compute_covariance(
            rd_imgs, rd_means, gt_imgs, gt_means, kernel=kernel
        )

        if padding == "valid":
            mask = torch.zeros(
                size=(1, 1, h, w),
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            mask[:, :, half_ksize:-half_ksize, half_ksize:-half_ksize] = 1.0
            factor = 1.0 / (nb * nc * (h - half_ksize * 2) * (w - half_ksize * 2))
        else:
            mask = 1.0
            factor = 1.0 / (nb * nc * h * w)

        # f0 = (c1 + 2.0 * μ * μ')
        f0 = (c1 + 2.0 * rd_means * gt_means) * mask
        # f1 = (c2 + 2.0 * Σ)
        f1 = (c2 + 2.0 * rd_gt_covars) * mask
        # f2 = (c1 + μ ** 2 + μ' ** 2)
        f2 = (c1 + rd_means**2 + gt_means**2) * mask
        # f3 = (c2 + σ + σ')
        f3 = (c2 + rd_vars + gt_vars) * mask

        # g0 = lambda x: 2.0 * W ⊗ (μ'(a, b, R) * x)
        g0 = lambda x: 2.0 * group_ssim.compute_mean(gt_means * x, kernel=kernel)

        # g1 = lambda x: 2.0 * R'(m, n) * W ⊗ x - 2.0 * W ⊗ (μ'(a, b, R) * x)
        g1 = lambda x: 2.0 * gt_imgs * group_ssim.compute_mean(
            x, kernel=kernel
        ) - 2.0 * group_ssim.compute_mean(gt_means * x, kernel=kernel)

        # g2 = lambda x: 2.0 * W ⊗ (μ(a, b, R) * x)
        g2 = lambda x: 2.0 * group_ssim.compute_mean(rd_means * x, kernel=kernel)

        # g3 = lambda x: 2.0 * R(m, n) * W ⊗ x - 2.0 * W ⊗ (μ(a, b, R) * x)
        g3 = lambda x: 2.0 * rd_imgs * group_ssim.compute_mean(
            x, kernel=kernel
        ) - 2.0 * group_ssim.compute_mean(rd_means * x, kernel=kernel)

        # To avoid duplicated computation
        f0f1 = f0 * f1
        f2f3 = f2 * f3 + eps
        f2sqf3 = (f2**2) * f3 + eps
        f2f3sq = f2 * (f3**2) + eps

        # TODO： CUDA pixelwise parallel computation inside a gaussian kernel
        jacob = g0(f1 / f2f3) + g1(f0 / f2f3) - g2(f0f1 / f2sqf3) - g3(f0f1 / f2f3sq)

        hess = None
        if with_hessian:
            h2 = lambda x: 2.0 * group_ssim.compute_mean(x, kernel=kernel**2)
            h3 = lambda x: 2.0 * group_ssim.compute_mean(
                x, kernel=kernel * (1.0 - kernel)
            )

            hess = (
                g0(g1(1.0 / f2f3) + g3(-f1 * f2 / (f2f3**2)) + g2(-f1 * f3 / (f2f3**2)))
                + g1(
                    g0(1.0 / f2f3) + g3(-f0 * f2 / (f2f3**2)) + g2(-f0 * f3 / (f2f3**2))
                )
                + g2(
                    g1(-f0 / f2sqf3)
                    + g0(-f1 / f2sqf3)
                    + g2(2.0 * f2f3 * f0f1 / (f2sqf3**2))
                    + g3((f2**2) * f0f1 / (f2sqf3**2))
                )
                + h2(-f0f1 / f2sqf3)
                + g3(
                    g1(-f0 / f2f3sq)
                    + g0(-f1 / f2f3sq)
                    + g3(2.0 * f2f3 * f0f1 / (f2f3sq**2))
                    + g2((f3**2) * f0f1 / (f2f3sq**2))
                )
                + h3(-f0f1 / f2f3sq)
            )

        # SSIMLoss = (1.0 - SSIMScore)
        jacob *= -factor
        if hess is not None:
            hess *= -factor
        return jacob, hess

    @staticmethod
    def _cache_tile_to_splat_alphas(
        gauss_ids: torch.LongTensor,
        means2d: torch.FloatTensor,
        conics: torch.FloatTensor,
        opacities: torch.FloatTensor,
        img_height: int,
        img_width: int,
        tile_size: int,
        isect_offsets: torch.LongTensor,
        flatten_ids: torch.LongTensor,
    ):
        """Cache splats-to-pixels alpha-blending intermediate results.

        Args:
            [1] gauss_ids: global splat indices, Dim=[K], torch.LongTensor.
            [2] means2d: 2D means of rasterized splats, Dim=[K, 2], torch.FloatTensor.
            [3] conics: inverse of 2D position covariance of rasterized splats, Dim=[K, 3], torch.FloatTensor;
                        i.e. conics[i] = [1.0 / COV(X, X), 1.0 / COV(X, Y), 1.0 / COV(Y, Y)].
            [4] opacities: opacity of rasterized splats, Dim=[K], torch.FloatTensor.
            [5] img_height: height of input image, int.
            [6] image_width: width of input image, int.
            [7] tile_size: tile size, int.
            [8] isect_offsets: offsets of intersection between splats and tiles, Dim=[H/tile_size, W/tile_size], torch.LongTensor.
                        i.e. tile (i, j) intersects with splats,
                             isect_start = insect_offsets[i, j], isect_end = isect_offsets[i, j + 1]
                             thus, intersected splats indices = flatten_ids[isect_start:isect_end]
            [9] flatten_ids: flattened indices of splats, Dim=[M], torch.LongTensor.

        Returns:
            [1] tile_to_alphas_cache: tile index to alpha-blending cache, Dict.
                tile_to_alphas_cache[image_index][tile_y * tile_width + tile_x] = {
                    "image_index": image index,
                    "tile_x": tile column index,
                    "tile_y": tile row index,
                    "tile_width": tile width,
                    "tile_height": tile height,
                    "tile_size": tile size,
                    "splat_ids": intersected splats' global ids, Dim=[tK], torch.LongTensor,
                    "tile_splat_indices": intersected splats' within-tile indices, Dim=[tK], torch.LongTensor,
                    "tile_alphas": pixelwise alphas i.e. gaussian-weighted opacities of each splat, Dim=[tile_size, tile_size, tK], torch.LongTensor,
                    "blend_alphas": depth-sorted cumulative product alphas of each splat, Dim=[tile_size, tile_size, tK], torch.FloatTensor,
                    "tile_sigmas": 2D gaussian kernel weights of each pixel, Dim=[tile_size, tile_size, tK], torch.FloatTensor,
                }
        """
        tile_height = int(np.ceil(img_height / tile_size))
        tile_width = int(np.ceil(img_width / tile_size))
        assert isect_offsets.shape[1:] == (tile_height, tile_width)

        device = means2d.device

        xs, ys = torch.meshgrid(
            [torch.arange(tile_size), torch.arange(tile_size)], indexing="xy"
        )
        xs, ys = xs.to(device).float(), ys.to(device).float()

        num_imgs = isect_offsets.shape[0]

        isect_offsets = isect_offsets.flatten()

        for img_idx in range(num_imgs):
            tile_to_alphas_cache = {}
            for tile_y in range(tile_height):
                for tile_x in range(tile_width):
                    tile_idx = (img_idx * tile_height * tile_width) + (
                        tile_y * tile_width + tile_x
                    )

                    isect_start = isect_offsets[tile_idx]
                    isect_end = (
                        isect_offsets[tile_idx + 1]
                        if tile_idx < len(isect_offsets) - 1
                        else len(flatten_ids)
                    )

                    if isect_start >= isect_end:
                        continue

                    flat_idxs = flatten_ids[isect_start:isect_end]

                    splat_ids = gauss_ids[flat_idxs]

                    tile_xs = (
                        xs + tile_size * tile_x + 0.5
                    )  # Dim = [tile_size, tile_size]
                    tile_ys = (
                        ys + tile_size * tile_y + 0.5
                    )  # Dim = [tile_size, tile_size]

                    tile_means2d = means2d[flat_idxs]  # Dim = [tK, 2]
                    tile_conics = conics[flat_idxs]  # Dim = [tK, 3]
                    tile_opacities = opacities[flat_idxs]  # Dim = [tK]

                    tile_xs = (
                        tile_means2d[:, 0][None, None, :] - tile_xs[:, :, None]
                    )  # Dim = [tile_size, tile_size, tK]
                    tile_ys = (
                        tile_means2d[:, 1][None, None, :] - tile_ys[:, :, None]
                    )  # Dim = [tile_size, tile_size, tK]

                    tile_sigmas = 0.5 * (
                        tile_xs * tile_xs * tile_conics[:, 0][None, None, :]
                        + tile_ys * tile_ys * tile_conics[:, 2][None, None, :]
                        + 2.0 * tile_xs * tile_ys * tile_conics[:, 1][None, None, :]
                    )  # Dim = [tile_size, tile_size, tK]

                    tile_alphas = torch.clamp_max(
                        tile_opacities[None, None, :] * torch.exp(-tile_sigmas), 0.9999
                    )  # Dim = [tile_size, tile_size, tK]

                    blend_alphas = torch.cumprod(1.0 - tile_alphas, dim=-1)
                    blend_alphas = torch.roll(blend_alphas, shifts=1, dims=-1)
                    blend_alphas[:, :, 0] = 1.0

                    tile_to_alphas_cache[tile_idx] = {
                        "image_index": img_idx,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_width": tile_width,
                        "tile_height": tile_height,
                        "tile_size": tile_size,
                        "splat_ids": splat_ids,  # Dim = [tK]
                        "tile_splat_indices": flat_idxs,  # Dim = [tK]
                        "tile_alphas": tile_alphas,  # Dim = [tile_size, tile_size, tK]
                        "blend_alphas": blend_alphas,  # Dim = [tile_size, tile_size, tK]
                        "tile_sigmas": tile_sigmas,  # Dim = [tile_size, tile_size, tK]
                    }
        return tile_to_alphas_cache

    @staticmethod
    def _cache_sh_colors(
        means3d: torch.FloatTensor,
        view_mats: torch.FloatTensor,
        sh_coeffs: torch.FloatTensor,
        radii: torch.FloatTensor,
    ):
        """Cache view-dependent spherical harmonics colors.

        Args:
            [1] means3d: 3D means of rasterized splats, Dim=[mK, 3], torch.FloatTensor.
            [2] view_mats: view matrices, Dim=[mN, 4, 4], torch.FloatTensor.
            [3] sh_coeffs: spherical harmonics coefficients of rasterized splats, Dim=[mK, (L + 1)^2, 3], torch.FloatTensor.
            [4] radii: radiuses of rasterized 2D splats, Dim=[mK], torch.FloatTensor.

        Returns:
            [1] sh_colors_cache: spherical integrated colors of each splat, Dim=[mN, mK, 3], torch.FloatTensor.
        """
        n = len(view_mats)

        view_dirs = (
            means3d[None, :, :] - torch.linalg.inv(view_mats)[:, :3, 3][:, None, :]
        )  # Dim = [N, K, 3]
        view_masks = (radii > 0).all(dim=-1)[None].repeat([n, 1])  # Dim = [N, K]
        sh_colors_cache = spherical_harmonics(
            degrees_to_use=int(np.sqrt(sh_coeffs.shape[-2])) - 1,
            dirs=view_dirs,  # Dim = [N, K, 3]
            coeffs=sh_coeffs[None].repeat([n, 1, 1, 1]),  # Dim = [N, K, (L + 1)^2, 3]
            masks=view_masks,  # Dim = [N, K]
        )  # [N, K, 3]
        sh_colors_cache = torch.clamp_min(sh_colors_cache + 0.5, 0.0)
        return sh_colors_cache

    @staticmethod
    def _backward_render_to_position_tile():
        """Backward pass of rendered pixels w.r.t splats' positions."""
        raise NotImplementedError

    @staticmethod
    def _backward_render_to_sh_color_tile(
        n_channels: int,
        tile_x: int,
        tile_y: int,
        tile_alphas: int,
        blend_alphas: int,
        with_hessian: bool = False,
    ):
        """Backward pass of rendered pixels w.r.t spherical harmonics colors.

        Args:
            [1] n_channels: Number of color channels, int.
            [2] tile_x: int, Tile column index, int.
            [3] tile_y: int, Tile row index, int.
            [4] tile_size: Tile size, int.
            [5] tile_alphas: Pixelwise alphas within a tile, Dim=[tile_size, tile_size, tK].
            [6] blend_alphas: Pixelwise blended alphas within a tile, Dim=[tile_size, tile_size, tK], torch.FloatTensor.
            [7] with_hessian: Whether to compute the Hessian matrix, bool.

        Returns:
            [1] tile_jacob: Jacobian matrix, Dim=[tile_size, tile_size, tK, 3], torch.FloatTensor.
            [2] tile_hess: Hessian matrix, Dim=[tile_size, tile_size, tK, tK, 3], torch.FloatTensor.

        For each pixel (m, n) within this tile,
            RGB = SUM_k^{tK}(G(k) * σ(k) * SH(k) * CUMPROD_j^{k - 1}(1.0 - G(j) * σ(j)))

            ∂(RGB) / ∂(SH(k)) = G(k) * σ(k) * CUMPROD_j^{k - 1}(1.0 - G(j) * σ(j)))
                where,
                    a. tile_alphas = G(k) * σ(k)
                    b. blend_alphas = CUMPROD_j^{k - 1}(1.0 - G(j) * σ(j))

            ∂²(RGB) / ∂(SH(k))² = 0.0
        """
        tile_jacob = (tile_alphas * blend_alphas)[:, :, :, None].repeat(
            [1, 1, 1, n_channels]
        )  # Dim = [tile_size, tile_size, tK, 3]

        tile_size, _, n_splats = tile_alphas.shape

        tile_hess = (
            torch.zeros(
                size=[tile_size, tile_size, n_splats, n_splats, n_channels],
                dtype=tile_alphas.dtype,
                device=tile_alphas.device,
            )
            if with_hessian
            else None
        )
        return tile_jacob, tile_hess
