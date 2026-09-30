import torch
from torch.optim.optimizer import Optimizer


# CIFAR-10 defaults reported in the VRAdam paper.
_ALPHA0 = 0.0846
_ALPHA1 = 29.0
_BETA1 = 0.9
_BETA2 = 0.999
_BETA3 = 1.015
_EPS = 1e-8
_WEIGHT_DECAY = 1e-5


class VRAdam(Optimizer):
	"""
	Velocity-Regularized Adam.

	This follows Algorithm 1 from the VRAdam paper:

		eta_t = alpha0 / (1 + min(beta3 * ||v_t||^2, alpha1))

	Here lr is alpha0, weight_decay is lambda, and beta3 is the velocity
	penalizer. The remaining VR-specific alpha1 cap stays fixed at the paper's
	CIFAR-10 value.
	The closest rho-like scale is rho_v = 1 / sqrt(beta3); the paper's
	CIFAR-10 beta3=1.015 gives rho_v ~= 0.993.
	"""

	def __init__(
			self,
			params,
			lr=_ALPHA0,
			betas=(_BETA1, _BETA2),
			beta3=_BETA3,
			eps=_EPS,
			weight_decay=_WEIGHT_DECAY,
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta1: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta2: {betas[1]}")
		if eps < 0.0:
			raise ValueError(f"Invalid epsilon: {eps}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay: {weight_decay}")
		if beta3 < 0.0:
			raise ValueError(f"Invalid beta3: {beta3}")

		defaults = dict(
			lr=lr,
			betas=betas,
			beta3=beta3,
			alpha1=_ALPHA1,
			eps=eps,
			weight_decay=weight_decay,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		velocity_sq = None
		params_with_grad = []

		for group in self.param_groups:
			beta1, beta2 = group['betas']

			for p in group['params']:
				if p.grad is None:
					continue

				grad = p.grad
				if grad.is_sparse:
					raise RuntimeError("VRAdam does not support sparse gradients")

				state = self.state[p]
				if len(state) == 0:
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(
						p, memory_format=torch.preserve_format)
					state['exp_avg_sq'] = torch.zeros_like(
						p, memory_format=torch.preserve_format)

				state['step'] += 1
				exp_avg = state['exp_avg']
				exp_avg_sq = state['exp_avg_sq']

				exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

				local_velocity_sq = exp_avg.detach().square().sum()
				velocity_sq = (
					local_velocity_sq if velocity_sq is None
					else velocity_sq + local_velocity_sq.to(velocity_sq.device)
				)
				params_with_grad.append((group, p, state, exp_avg, exp_avg_sq))

		if velocity_sq is None:
			return loss

		for group, p, state, exp_avg, exp_avg_sq in params_with_grad:
			lr = group['lr']
			beta1, beta2 = group['betas']
			eps = group['eps']
			wd = group['weight_decay']
			beta3 = group['beta3']
			alpha1 = group['alpha1']

			velocity_sq_t = velocity_sq.to(p.device)
			penalty = torch.clamp(beta3 * velocity_sq_t, max=alpha1)
			dynamic_lr = p.new_tensor(lr) / (1.0 + penalty)

			bias_correction1 = 1.0 - beta1 ** state['step']
			bias_correction2 = 1.0 - beta2 ** state['step']
			exp_avg_hat = exp_avg / bias_correction1
			denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

			if wd > 0:
				p.mul_(1.0 - dynamic_lr * wd)
			p.add_(dynamic_lr * (exp_avg_hat / denom), alpha=-1.0)

		return loss
