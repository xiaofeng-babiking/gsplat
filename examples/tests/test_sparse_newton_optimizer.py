import pytest
import os
import time
import torch

from gsplat.optimizers.torch_functions_forward import blend_sh_colors_with_alphas
from gsplat.optimizers.sparse_newton import _backward_render_to_sh_color
from gsplat.logger import create_logger

LOGGER = create_logger(name=os.path.basename(__file__), level="INFO")

mN = 8  # number of camera views
tH = 16  # tile size along height dimension
tW = 16  # tile size along width dimension
tK = 1000  # number of gaussian splats
mC = 3  # number of RGB color channels
KWARGS = {
    "dtype": torch.float32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


def test_backward_render_to_sh_color():
    """Test backward from render color to SH color."""
    name = "From_RENDER_To_SH_COLOR"

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
    jacob_ours, _ = _backward_render_to_sh_color(
        num_chs=mC,
        alphas=alphas,
        masks=masks,
    )
    end = time.time()
    assert torch.allclose(jacob_auto, jacob_ours, atol=1e-4, rtol=1e-3)
    LOGGER.info(
        f"Backward={name}, "
        + f"Jacobian=[{mN}, {tH}, {tW}, {mC}], Output=[{mN}, {tH}, {tW}, {mC}], Input=[{mN}, {tK}, {mC}], "
        + f"Elapsed={float(end - start):.6f} seconds."
    )


if __name__ == "__main__":
    test_backward_render_to_sh_color()
