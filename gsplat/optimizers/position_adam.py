import torch
from typing import List, Dict, Tuple, Optional


class PositionAdam(torch.optim.Adam):
    """A customized Adam optimizer for 3DGS position update."""

    def __init__(
        self,
        params: List[Dict[str, Optional[torch.Tensor]]],
        eps: float = 1e-8,
        betas: Tuple[float, float] = (0.9, 0.999),
        lr: float = 0.01,
    ):
        """Initialize torch.optim.Adam."""
        super().__init__(params=params, eps=eps, betas=betas, lr=lr)

        self._exp_avg = torch.zeros_like(
            params[0]["params"][0], memory_format=torch.preserve_format
        )
        self._exp_avg_sq = torch.zeros_like(
            params[0]["params"][0], memory_format=torch.preserve_format
        )

    @torch.no_grad()
    def step(
        self,
        visibility: Optional[torch.BoolTensor] = None,
        view: Optional[torch.FloatTensor] = None,
    ) -> None:
        """Update optimizable parameters step.

        Args:
            [1] visibility: Visible mask, Dim=[tK], tK denotes the number of splats.
                    i.e. visible_mask[i] = True denotes i-th splat is to update at this step.
            [2] view: View matrix, Dim=[1, 4, 4].
        """
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            name = group["name"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                raise AttributeError(f"NO gradients for Parameter={name}!")

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)

            step = self.state.get(param, None)["step"]
            step += 1.0

            grad = param.grad[visibility]
            exp_avg = self._exp_avg[visibility]
            exp_avg_sq = self._exp_avg_sq[visibility]

            exp_avg.lerp_(grad, 1 - beta1)
            self._exp_avg[visibility] = exp_avg

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            self._exp_avg_sq[visibility] = exp_avg_sq

            update = (
                -lr
                * (exp_avg / (1.0 - beta1**step))
                / (torch.sqrt(exp_avg_sq / (1.0 - beta2**step)) + eps)
            )

            param[visibility] += update
