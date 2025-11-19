import torch
import numpy as np
from scipy.optimize import curve_fit
from torch.nn.modules.loss import _Loss
from enum import Enum
from collections.abc import Iterable
from typing import Dict, Any, Optional, Literal, Tuple


class GroupSSIMLoss(_Loss):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        padding: Literal["same", "valid"] = "valid",
        gauss_filter_1d: Optional[torch.FloatTensor] = None,
        size_average=None,
        reduce=None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        *args,
        **kwargs,
    ):
        self._kernel_size = kernel_size
        self._sigma = sigma
        self._c1 = c1
        self._c2 = c2
        self._padding = padding
        self._filter = gauss_filter_1d
        self._reduction = reduction
        super().__init__(size_average, reduce, reduction, *args, **kwargs)

        self._kernel = self.create_gauss_kernel_2d()

    def create_gauss_kernel_2d(self) -> torch.FloatTensor:
        """Create 2D Gaussian Kernel for SSIM convolution."""
        if self._filter is not None:
            _, opt_sigma = GroupSSIMLoss._fit_gauss_filter_1d(
                self._filter.cpu().detach().numpy(),
            )

            self._kernel_size = len(self._filter)
            self._sigma = opt_sigma
        else:
            filter_1d = GroupSSIMLoss._evaluate_gauss_filter_1d(
                x=np.arange(self._kernel_size),
                mu=self._kernel_size // 2,
                sigma=self._sigma,
            )
            self._filter = torch.from_numpy(filter_1d).float()

        # outer product
        kernel_2d = torch.outer(self._filter, self._filter)
        # [out_channel, in_channel, kernel_size, kernel_size] -> [1, 1, K, K]
        return kernel_2d[None, None, ...]

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
    def _fit_gauss_filter_1d(
        gauss_filter_1d: np.ndarray,
    ) -> Tuple[float, float]:
        """Given a Gaussian filter in 1D, find its mean and variance."""
        n_items = len(gauss_filter_1d)

        opt_mu = n_items // 2  # Central point of the Gaussian filter

        fit_func = lambda x, sigma: GroupSSIMLoss._evaluate_gauss_filter_1d(
            x, mu=opt_mu, sigma=sigma
        )

        popt, _ = curve_fit(
            fit_func,
            xdata=np.arange(len(gauss_filter_1d)),
            ydata=gauss_filter_1d,
        )

        opt_sigma = popt[0]
        return opt_mu, opt_sigma

    def __str__(self):
        """Return string representation of SSIM kernel."""
        return (
            "SSIMLoss with Gaussian kernel "
            + f"size={self._kernel_size}, mean={self._kernel_size // 2}, variance={self._sigma:.6f}."
        )

    def compute_mean(self, img: torch.FloatTensor):
        """Compute mean of an image."""
        c = img.shape[1]

        mean = torch.conv2d(
            img,
            torch.tile(self._kernel, (c, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=c,
        )
        return mean

    def compute_covariance(
        self,
        img_0: torch.FloatTensor,
        mean_0: Optional[torch.FloatTensor] = None,
        img_1: Optional[torch.FloatTensor] = None,
        mean_1: Optional[torch.FloatTensor] = None,
    ):
        """Compute covariance between two images."""
        if img_1 is not None:
            assert (
                img_0.shape == img_1.shape
            ), "Source and target image must have the same shape!"

        c = img_0.shape[1]

        if mean_0 is None:
            mean_0 = self.compute_mean(img_0)

        if img_1 is not None and mean_1 is None:
            mean_1 = self.compute_mean(img_1)

        # No need to compute target mean when targe image is None
        mean_1 = None if img_1 is None else mean_1

        if mean_1 is not None:
            assert (
                mean_0.shape == mean_1.shape
            ), "Mean tensors must have the same shape!"

        covar = torch.conv2d(
            (img_0 * img_1) if img_1 is not None else (img_0**2),
            torch.tile(self._kernel, (c, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=c,
        ) - ((mean_0 * mean_1) if img_1 is not None else (mean_0**2))
        return covar

    def forward(
        self, input: torch.FloatTensor, target: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute SSIM loss between source and target image."""

        assert (
            input.shape == target.shape
        ), "source and target image must have the same shape!"

        src_mean = self.compute_mean(input)
        dst_mean = self.compute_mean(target)

        src_var = self.compute_covariance(input, src_mean)
        dst_var = self.compute_covariance(target, dst_mean)
        src_dst_covar = self.compute_covariance(input, src_mean, target, dst_mean)

        a = self._c1 + src_mean**2 + dst_mean**2
        b = self._c2 + src_var + dst_var
        c = self._c1 + 2.0 * src_mean * dst_mean
        d = self._c2 + 2.0 * src_dst_covar
        ssim_map = (c * d) / (a * b)

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
    def _jacobian_l2_to_rgb(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
    ):
        """Computes Jacobian matrix from L2 to rendering RGB pixels.

        Args:
            [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
            [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.

        Returns:
            [1] jacob: Jacobian matrix from L2 to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.

                    L2 = 0.5 * MEAN(||rd_imgs - gt_imgs||^2)
                    RGB =  { pixel=(r, g, b) | for any pixel in rd_imgs }
                    RGB' = { pixel=(r, g, b) | for any pixel in gt_imgs }
                    ∂L2 / ∂(RGB) -> partial derivative of L2 (scalar) with respect to C (matrix)

                    ∂L2 / ∂(RGB)
                        = [∂L2 / ∂R, ∂L2 / ∂G, ∂L2 / ∂B]
                        = (rd_imgs - gt_imgs) / (N * C * H * W)
        """
        n, c, h, w = rd_imgs.shape

        factor = 1.0 / (n * c * h * w)

        jacob = factor * (rd_imgs - gt_imgs)
        return jacob

    @staticmethod
    def _hessian_l2_to_rgb(
        rd_imgs: torch.FloatTensor,
        gt_imgs: Optional[torch.FloatTensor] = None,
    ):
        """Computes Hessian matrix from L2 to rendering RGB pixels.

        Args:
            [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
            [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.

        Returns:
            [1] hess: Hessian matrix from L2 to rendering RGB pixels, Dim=[N, C, C, H, W], torch.FloatTensor.

                    L2 = 0.5 * MEAN(||rd_imgs - gt_imgs||^2)

                    RGB =  { pixel=(r, g, b) | for any pixel in rd_imgs }
                    ∂²L2 / ∂(RGB)²
                    = [[∂²L2 / ∂R²,  ∂L2 / ∂R∂G, ∂L2 / ∂R∂B],
                       [∂²L2 / ∂R∂G, ∂²L2 / ∂G², ∂L2 / ∂G∂B],
                       [∂²L2 / ∂R∂B, ∂L2 / ∂G∂B, ∂²L2 / ∂B²]]
                    Since R, G and B are independent with each other,
                    ∂²L2 / ∂(RGB)²
                    = [[∂²L2 / ∂R², 0, 0],
                       [0, ∂²L2 / ∂G², 0],
                       [0, 0, ∂²L2 / ∂B²]]
                    = [[∂(J_R) / ∂R, 0, 0],
                       [0, ∂(J_G) / ∂G, 0],
                       [0, 0, ∂(J_B) / ∂B]] where J_R = ∂L2 / ∂R
                    = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]] / (N * C * H * W)
        """
        n, c, h, w = rd_imgs.shape

        factor = 1.0 / (n * c * h * w)

        hess = factor * torch.eye(c, dtype=rd_imgs.dtype)
        hess = hess.view(1, c, c, 1, 1).repeat(n, 1, 1, h, w)
        return hess

    @staticmethod
    def _jacobian_ssim_to_rgb(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
        kernel_size: int = 11,
        sigma: float = 1.5,
        padding: Literal["valid", "same"] = "valid",
        gauss_filter_1d: Optional[torch.FloatTensor] = None,
        eps: float = 1e-12,
    ):
        """Computes Jacobian matrix from SSIM to rendering RGB pixels.

        Args:
            [1] rd_imgs: Rendering images, Dim=[N, C, H, W], torch.FloatTensor.
            [2] gt_imgs: Groundtruth images, Dim=[N, C, H, W], torch.FloatTensor.

        Returns:
            [1] jacob: Jacobian matrix from SSIM to rendering RGB pixels, Dim=[N, C, H, W], torch.FloatTensor.

                    ∂Ls / ∂(RGB) = [∂Ls / ∂R, ∂Ls / ∂G, ∂Ls / ∂B]

                    For a given pixel P=(m, n), R(m, n) denotes its Red value,

                    ∂Ls / ∂R = { ∂Ls / ∂R(m, n) | for all pixels P=(m, n) in rendering image }

                    ∂Ls / ∂R(m, n)
                        = SUM_i_j( ∂fs(m + i - kW // 2, n + j - kH // 2, R) / ∂R(m, n) )
                    where,
                        (kW, kH) denotes the 2D Gaussian kernel size;
                        fs(a, b, R) denotes SSIM error (Red part) center at pixel Q=(a, b);
                    Here fs(a, b, R) actually computes:
                        weighted contribution of P=(m, n) 
                            to different "gaussian-2D-convolution-kernels",
                    For example,
                        i=0,       j=0,       P(m, n) -> right-bottom corner of the kernel
                        i=kW - 1,  j=0,       P(m, n) -> left-bottom corner of the kernel
                        i=0,       j=kH - 1,  P(m, n) -> right-top corner of the kernel
                        i=kW - 1,  j=kH - 1,  P(m, n) -> left-top corner of the kernel
                        i=kW // 2, j=kH // 2, P(m, n) -> center of the kernel
                    for example,
                    Let (a, b) = (m + i - kW // 2, n + j - kH // 2)
                    then,
                        fs(a, b, R) 
                            = (c1 + 2.0 * μ * μ') * (c2 + 2.0 * Σ) \
                                / (c1 + μ ** 2 + μ' ** 2) * (c2 + σ + σ')
                        where,
                            μ, μ' = mean of Rendering/Groundtruth kernel Red values
                            σ, σ' = variance of Rendering/Groundtruth kernel Red values
                            Σ = covariance between Rendering and Groundtruth kernel Red values
                            c1, c2 = constant terms
                    
                    Let,
                        f0 = (c1 + 2.0 * μ * μ')
                        f1 = (c2 + 2.0 * Σ)
                        f2 = (c1 + μ ** 2 + μ' ** 2)
                        f3 = (c2 + σ + σ')
                        g0 = ∂f0 / ∂R(m, n)
                        g1 = ∂f1 / ∂R(m, n)
                        g2 = ∂f2 / ∂R(m, n)
                        g3 = ∂f3 / ∂R(m, n)
                    Then,
                        fs(a, b, R) = (f0 * f1) / (f2 * f3)
                    
                    ∂fs(a, b, R) / ∂R(m, n)
                        = ∂(f0 * f1) / ∂R(m, n) / (f2 * f3) - (f0 * f1) / ((f2 * f3) ** 2) * ∂ (f2 * f3) / ∂R(m, n)
                        = (g0 * f1 + f0 * g1) / (f2 * f3) - (f0 * f1) / ((f2 * f3) ** 2) * (g2 * f3 + f2 * g3)
                        = (f1/f2/f3)*g0 + (f0/f2/f3)*g1 - (f0*f1/(f2**2)/f3)*g2 - (f0*f1/f2/(f3**2))*g3
                    
                    μ(a, b) <- W(m, n) * R(m, n)
                    Σ(a, b) <- W(m, n) * R(m, n) * R'(m, n) - μ(a, b) * μ'(a, b)
                    σ(a, b) <- W(m, n) * (R(m, n) ** 2) - μ(a, b) ** 2

                    W(m, n) 
                        = W(m - a + kW // 2, n - b + kH // 2)
                        = W(kW - 1 - i, kH - 1 - j)
                        = W(i, j) # gaussian distribution sysmetric

                    g0 = 2.0 * μ' * ∂μ(a, b) / R(m, n) 
                       = 2.0 * μ' * W(m, n)                     
                    g1 = 2.0 * ∂Σ(a, b) / R(m, n) 
                       = 2.0 * W(m, n) * R'(m, n) - 2.0 * μ'* ∂μ(a, b) / R(m, n)
                       = 2.0 * W(m, n) * (R'(m, n) - μ'(a, b))
                    g2 = 2.0 * μ * ∂μ(a, b) / R(m, n)
                       = 2.0 * μ * W(m, n)
                    g3 = ∂σ(a, b) / R(m, n)
                       = 2.0 * W(m, n) * R(m, n) - 2.0 * μ * ∂μ(a, b) / R(m, n)
                       = 2.0 * W(m, n) * R(m, n) - 2.0 * μ * ∂μ(a, b) / R(m, n)
                       = 2.0 * W(m, n) * (R(m, n) - μ)
        """
        n, c, h, w = rd_imgs.shape

        device = rd_imgs.device

        factor = 1.0 / (n * c * h * w)

        group_ssim = GroupSSIMLoss(
            kernel_size=kernel_size,
            sigma=sigma,
            c1=c1,
            c2=c2,
            padding=padding,
            gauss_filter_1d=gauss_filter_1d,
        ).to(device)

        rd_means = group_ssim.compute_mean(rd_imgs)
        rd_vars = group_ssim.compute_covariance(rd_imgs, rd_means)

        gt_means = group_ssim.compute_mean(gt_imgs)
        gt_vars = group_ssim.compute_covariance(gt_imgs, gt_means)

        rd_gt_covars = group_ssim.compute_covariance(
            rd_imgs, rd_means, gt_imgs, gt_means
        )

        ksize = group_ssim._kernel_size
        half_ksize = ksize // 2
        kernel = group_ssim._kernel.data.clone().detach().squeeze()

        # m, n dimension = [H, W]
        m, n = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        m, n = m.to(device), n.to(device)

        # f0 = (c1 + 2.0 * μ * μ')
        f_0 = c1 + 2.0 * rd_means * gt_means
        # f1 = (c2 + 2.0 * Σ)
        f_1 = c2 + 2.0 * rd_gt_covars
        # f2 = (c1 + μ ** 2 + μ' ** 2)
        f_2 = c1 + rd_means**2 + gt_means**2
        # f3 = (c2 + σ + σ')
        f_3 = c2 + rd_vars + gt_vars

        # TODO： CUDA pixelwise parallel computation inside a gaussian kernel
        jacob = torch.zeros_like(rd_imgs)
        for i in range(ksize):
            for j in range(ksize):
                a = m + i - half_ksize
                b = n + j - half_ksize
                mask = (a < 0) | (a >= w) | (b < 0) | (b >= h)

                a = torch.clamp(a, min=0, max=w - 1)
                b = torch.clamp(b, min=0, max=h - 1)

                rd_means_a_b = rd_means[..., b, a]
                gt_means_a_b = gt_means[..., b, a]

                # g0 = 2.0 * μ'(a, b) * W(i, j)
                g_0 = 2.0 * gt_means_a_b * kernel[i, j]

                # g1 = 2.0 * W(i, j) * (R'(m, n) - μ'(a, b))
                g_1 = 2.0 * (gt_imgs - gt_means_a_b) * kernel[i, j]

                # g2 = 2.0 * μ(a, b) * W(i, j)
                g_2 = 2.0 * rd_means_a_b * kernel[i, j]

                # g3 = 2.0 * W(i, j) * (R(m, n) - μ(a, b))
                g_3 = 2.0 * (rd_imgs - rd_means_a_b) * kernel[i, j]

                jacob_i_j = (
                    (f_1 / f_2 / f_3) * g_0
                    + (f_0 / f_2 / f_3) * g_1
                    - (f_0 * f_1 / (f_2**2) / f_3) * g_2
                    - (f_0 * f_1 / f_2 / (f_3**2)) * g_3
                )
                jacob_i_j[:, :, mask] = 0.0

                jacob += jacob_i_j
                # torch.cuda.empty_cache()

        # SSIMLoss = (1.0 - SSIMScore)
        jacob *= -factor

        if padding == "valid":
            jacob[:, :, half_ksize:-half_ksize, half_ksize:-half_ksize] = 0.0
        return jacob

    @staticmethod
    def _hessian_ssim_to_rgb(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
    ):
        """Computes Hessian matrix from SSIM to rendering RGB pixels."""
        n, c, h, w = rd_imgs.shape

    @staticmethod
    def _jacobian_rgb_to_position(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
    ):
        """Computes Jacobian matrix from RGB to splats' positions."""
        n, c, h, w = rd_imgs.shape

    @staticmethod
    def _hessian_rgb_to_position(
        rd_imgs: torch.FloatTensor,
        gt_imgs: torch.FloatTensor,
    ):
        """Computes Hessian matrix from RGB to splats' positions."""
        n, c, h, w = rd_imgs.shape
