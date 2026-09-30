"""Cautious optimizer comparison baselines.

Adapted from C-Optim (MIT, Kaizhao Liang 2024); see C_OPTIM_LICENSE.
These readable scalar paths use C-Optim's masking and
normalization, including its epsilon convention for CautiousAdamW.
"""

import math
import torch


class CautiousLion(torch.optim.Optimizer):
    """Lion with gradient-aligned coordinates and the upstream N/(K+1) mask."""

    def __init__(self, params, lr=1e-4, betas=(0.95, 0.98), weight_decay=0.0):
        if lr < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid learning rate, decay, or betas")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse or param.is_complex():
                    raise RuntimeError("CautiousLion requires dense real parameters")
                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param)
                momentum = state["exp_avg"]
                direction = momentum.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                mask = (direction * grad > 0).to(grad.dtype)
                mask.mul_(mask.numel() / (mask.sum() + 1))
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(direction * mask, alpha=-group["lr"])
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW with a gradient-aligned first moment and mean-normalized mask.

    Matches C-Optim's non-foreach baseline: epsilon is added to the raw
    second-moment square root before the scalar bias-correction factor.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid learning rate, epsilon, decay, or betas")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, correct_bias=correct_bias))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse or param.is_complex():
                    raise RuntimeError("CautiousAdamW requires dense real parameters")
                state = self.state[param]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(param),
                                 exp_avg_sq=torch.zeros_like(param))
                state["step"] += 1
                momentum, variance = state["exp_avg"], state["exp_avg_sq"]
                param.add_(param, alpha=-group["lr"] * group["weight_decay"])
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = variance.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    step_size *= math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                mask = (momentum * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                param.add_((momentum * mask) / denominator, alpha=-step_size)
        return loss
