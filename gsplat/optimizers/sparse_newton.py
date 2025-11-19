import torch
import numpy as np
from scipy.optimize import curve_fit
from torch.nn.modules.loss import _Loss
from enum import Enum
from collections.abc import Iterable
from typing import Dict, Any, Optional, Literal, Tuple


class GSParameterGroup(Enum):
    POSITION = 0  # 3DOFs position (x, y, z)
    ROTATION = 1  # 3DOFs rotation axis (nx, ny, nz) with angle θ
    SCALE = 2  # 3DOFs scale (sx, sy, sz)
    OPACITY = 3  # 1DOF opacity
    COLOR = 4  # Spherical Harmonic coefficients Dimension = (L + 1)^2


class GSGroupSSIMLoss(_Loss):
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
        super().__init__(size_average, reduce, reduction, *args, **kwargs)

        self._kernel = torch.nn.Parameter(
            self.create_gauss_kernel_2d(), requires_grad=False
        )

    def create_gauss_kernel_2d(self) -> torch.FloatTensor:
        """Create 2D Gaussian Kernel for SSIM convolution."""
        if self._filter is not None:
            _, opt_sigma = GSGroupSSIMLoss._fit_gauss_filter_1d(
                self._filter.cpu().detach().numpy(),
            )

            self._kernel_size = len(self._filter)
            self._sigma = opt_sigma
        else:
            filter_1d = GSGroupSSIMLoss._evaluate_gauss_filter_1d(
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

        fit_func = lambda x, sigma: GSGroupSSIMLoss._evaluate_gauss_filter_1d(
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

    def forward(
        self, input: torch.FloatTensor, target: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute SSIM loss between source and target image."""

        assert (
            input.shape == target.shape
        ), "source and target image must have the same shape!"

        groups = input.shape[1]

        src_mean = torch.conv2d(
            input,
            torch.tile(self._kernel, (groups, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=groups,
        )
        dst_mean = torch.conv2d(
            target,
            torch.tile(self._kernel, (groups, 1, 1, 1)),
            stride=1,
            padding="same",
            groups=groups,
        )

        src_var = (
            torch.conv2d(
                input**2,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - src_mean**2
        )
        dst_var = (
            torch.conv2d(
                target**2,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - dst_mean**2
        )
        src_dst_covar = (
            torch.conv2d(
                input * target,
                torch.tile(self._kernel, (groups, 1, 1, 1)),
                stride=1,
                padding="same",
                groups=groups,
            )
            - src_mean * dst_mean
        )

        a = self._c1 + src_mean**2 + dst_mean**2
        b = self._c2 + src_var + dst_var
        c = self._c1 + 2.0 * src_mean * dst_mean
        d = self._c2 + 2.0 * src_dst_covar
        ssim_map = (c * d) / (a * b)

        half_ksize = self._kernel_size // 2
        if self._padding == "valid":
            ssim_map = ssim_map[:, :, half_ksize:-half_ksize, half_ksize:-half_ksize]

        if self.reduction == "mean":
            ssim_score = torch.mean(ssim_map, dim=[1, 2, 3])
        elif self.reduction == "sum":
            ssim_score = torch.sum(ssim_map, dim=[1, 2, 3])
        elif self.reduction == "none":
            ssim_score = ssim_map
        else:
            raise ValueError(f"Invalid reduction mode: {self._reduction}!")

        ssim_loss = 1.0 - ssim_score
        return ssim_loss


class GSGroupNewton(torch.optim.Optimizer):
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
