import math
import torch


DEFAULT_MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DEFAULT_MUON_NS_STEPS = 5
DEFAULT_MUON_NS_EPS = 1e-7


def _validate_real_parameter(param, optimizer_name, grad=None):
	if param.is_complex() or (grad is not None and grad.is_complex()):
		raise ValueError(
			f"{optimizer_name} does not support complex parameters or gradients")


def _zeropower_via_newton_schulz(
		matrix, ns_coefficients, ns_steps, eps):
	"""Muon's Newton-Schulz matrix zeroth-power approximation."""
	if matrix.ndim != 2:
		raise ValueError("Input tensor gradient must be a 2D matrix")
	a, b, c = ns_coefficients
	orthogonal = matrix.to(dtype=torch.bfloat16, copy=True)
	transposed = matrix.size(0) > matrix.size(1)
	if transposed:
		orthogonal = orthogonal.T
	orthogonal.div_(orthogonal.norm().clamp(min=eps))
	for _ in range(ns_steps):
		gram = orthogonal @ orthogonal.T
		gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
		orthogonal = torch.addmm(
			orthogonal, gram_update, orthogonal, beta=a)
	if transposed:
		orthogonal = orthogonal.T
	return orthogonal


def _muon_lr_ratio(matrix_shape, adjust_lr_fn):
	rows, cols = matrix_shape[:2]
	if adjust_lr_fn is None or adjust_lr_fn == 'original':
		return math.sqrt(max(1, rows / cols))
	if adjust_lr_fn == 'match_rms_adamw':
		return 0.2 * math.sqrt(max(rows, cols))
	raise ValueError(
		f"Adjust learning rate function {adjust_lr_fn} is not supported")


def _adamw_bias_corrections(
		beta1, beta2, beta3, step, bias_correction):
	if not bias_correction:
		return 1.0, 1.0
	return (
		1.0 - beta1 * beta2 ** (step - 1),
		1.0 - beta3 ** step,
	)


class CurvatureAwareSGD(torch.optim.Optimizer):
	"""
	SGD with Lion's curvature-aware gradient injection.

	The update is the linear, unclipped Lion rule:

		d_t = beta1 * m_{t-1} + (1 - beta1) * g_t
		theta_t = (1 - lr * weight_decay) * theta_{t-1} - lr * d_t
		m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

	The momentum state is initialized to zero and is not bias-corrected.
	"""

	def __init__(
			self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0,
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta1: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta2: {betas[1]}")
		if not 0.0 <= weight_decay:
			raise ValueError(f"Invalid weight_decay: {weight_decay}")

		defaults = dict(
			lr=lr,
			betas=betas,
			weight_decay=weight_decay,
		)
		super().__init__(params, defaults)
		for group in self.param_groups:
			for param in group['params']:
				_validate_real_parameter(param, type(self).__name__)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr = group['lr']
			beta1, beta2 = group['betas']
			weight_decay = group['weight_decay']

			params = []
			grads = []
			exp_avgs = []
			for param in group['params']:
				if param.grad is None:
					continue
				grad = param.grad
				_validate_real_parameter(
					param, type(self).__name__, grad=grad)

				state = self.state[param]
				if len(state) == 0:
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(
						param, memory_format=torch.preserve_format)

				state['step'] += 1
				params.append(param)
				grads.append(grad)
				exp_avgs.append(state['exp_avg'])

			if not params:
				continue

			updates = torch._foreach_mul(exp_avgs, beta1)
			torch._foreach_add_(updates, grads, alpha=1 - beta1)

			if weight_decay > 0:
				torch._foreach_mul_(params, 1 - lr * weight_decay)
			torch._foreach_add_(params, updates, alpha=-lr)

			torch._foreach_mul_(exp_avgs, beta2)
			torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2)

		return loss


class CurvatureAwareAdamW(torch.optim.Optimizer):
	"""
	AdamW with Lion's curvature-aware gradient injection.

	The update uses separate decays for the injected direction, stored first
	moment, and squared-gradient second moment:

		d_t = beta1 * m_{t-1} + (1 - beta1) * g_t
		m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
		v_t = beta3 * v_{t-1} + (1 - beta3) * g_t^2

	By default, both the direction and second moment are bias-corrected. Set
	``bias_correction=False`` to use the raw direction and second moment. When
	bias correction is enabled and beta1 equals beta2, this reduces to AdamW with
	betas=(beta2, beta3).
	"""

	def __init__(
			self,
			params,
			lr=1e-3,
			betas=(0.9, 0.99),
			beta3=0.999,
			eps=1e-8,
			weight_decay=1e-2,
			bias_correction=True,
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta1: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta2: {betas[1]}")
		if not 0.0 <= beta3 < 1.0:
			raise ValueError(f"Invalid beta3: {beta3}")
		if not 0.0 <= eps:
			raise ValueError(f"Invalid epsilon: {eps}")
		if not 0.0 <= weight_decay:
			raise ValueError(f"Invalid weight_decay: {weight_decay}")

		defaults = dict(
			lr=lr,
			betas=betas,
			beta3=beta3,
			eps=eps,
			weight_decay=weight_decay,
			bias_correction=bias_correction,
		)
		super().__init__(params, defaults)
		for group in self.param_groups:
			for param in group['params']:
				_validate_real_parameter(param, type(self).__name__)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr = group['lr']
			beta1, beta2 = group['betas']
			beta3 = group['beta3']
			eps = group['eps']
			weight_decay = group['weight_decay']
			bias_correction = group['bias_correction']

			for param in group['params']:
				if param.grad is None:
					continue

				grad = param.grad
				_validate_real_parameter(
					param, type(self).__name__, grad=grad)
				if grad.is_sparse:
					raise RuntimeError(
						"CurvatureAwareAdamW does not support sparse gradients")

				state = self.state[param]
				if len(state) == 0:
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(
						param, memory_format=torch.preserve_format)
					state['exp_avg_sq'] = torch.zeros_like(
						param, memory_format=torch.preserve_format)

				exp_avg = state['exp_avg']
				exp_avg_sq = state['exp_avg_sq']
				state['step'] += 1
				step = state['step']

				direction = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
				exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
				exp_avg_sq.mul_(beta3).addcmul_(
					grad, grad, value=1 - beta3)

				(
					direction_correction,
					second_moment_correction,
				) = _adamw_bias_corrections(
					beta1, beta2, beta3, step, bias_correction)
				denom = (exp_avg_sq / second_moment_correction).sqrt().add_(eps)
				if weight_decay > 0:
					param.mul_(1 - lr * weight_decay)
				param.addcdiv_(
					direction, denom,
					value=-lr / direction_correction,
				)

		return loss


class CurvatureAwareMuon(torch.optim.Optimizer):
	"""
	Muon with Lion's curvature-aware gradient injection.

	Hidden matrix and convolutional parameters assigned to the Muon path use

		d_t = beta1 * m_{t-1} + (1 - beta1) * g_t
		m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
		theta_t = (1 - lr * weight_decay) * theta_{t-1}
		          - adjusted_lr * zeropower(d_t)

	Parameters in groups with ``use_muon=False`` use the curvature-aware AdamW
	recurrence with the same ``beta1`` injection, ``beta2`` stored-momentum
	decay, and ``beta3`` squared-gradient decay. This supplies the fallback for
	the input layer, output head, embeddings, normalization parameters, and
	biases without changing the beta semantics across parameter groups.
	"""

	def __init__(
			self,
			params,
			lr=1e-3,
			betas=(0.9, 0.99),
			beta3=0.999,
			eps=1e-8,
			weight_decay=0.1,
			ns_coefficients=DEFAULT_MUON_NS_COEFFICIENTS,
			ns_steps=DEFAULT_MUON_NS_STEPS,
			ns_eps=DEFAULT_MUON_NS_EPS,
			adjust_lr_fn=None,
	):
		if not 0.0 <= lr:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta1: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta2: {betas[1]}")
		if not 0.0 <= beta3 < 1.0:
			raise ValueError(f"Invalid beta3: {beta3}")
		if not 0.0 <= eps:
			raise ValueError(f"Invalid epsilon: {eps}")
		if not 0.0 <= weight_decay:
			raise ValueError(f"Invalid weight_decay: {weight_decay}")
		if len(ns_coefficients) != 3:
			raise ValueError("ns_coefficients must contain exactly three values")
		if not 0 <= ns_steps < 100:
			raise ValueError("ns_steps must be between 0 and 99")
		if not 0.0 <= ns_eps:
			raise ValueError(f"Invalid Newton-Schulz epsilon: {ns_eps}")
		if adjust_lr_fn not in (None, 'original', 'match_rms_adamw'):
			raise ValueError(
				f"Adjust learning rate function {adjust_lr_fn} is not supported")

		defaults = dict(
			lr=lr,
			betas=betas,
			beta3=beta3,
			eps=eps,
			weight_decay=weight_decay,
			ns_coefficients=ns_coefficients,
			ns_steps=ns_steps,
			ns_eps=ns_eps,
			adjust_lr_fn=adjust_lr_fn,
			use_muon=True,
		)
		super().__init__(params, defaults)
		for group in self.param_groups:
			for param in group['params']:
				_validate_real_parameter(param, type(self).__name__)
				if group['use_muon'] and param.ndim < 2:
					raise ValueError(
						"CurvatureAwareMuon requires use_muon parameters "
						f"to have at least two dimensions, found {param.size()}")

	@torch.no_grad()
	def step(self, closure=None):
		for group in self.param_groups:
			if not group['use_muon']:
				continue
			for param in group['params']:
				if param.ndim < 2:
					raise ValueError(
						"CurvatureAwareMuon requires use_muon parameters "
						f"to have at least two dimensions, found {param.size()}")

		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			if group['use_muon']:
				self._step_muon_group(group)
			else:
				self._step_adamw_group(group)

		return loss

	def _step_muon_group(self, group):
		lr = group['lr']
		beta1, beta2 = group['betas']
		weight_decay = group['weight_decay']

		for param in group['params']:
			if param.grad is None:
				continue
			grad = param.grad
			_validate_real_parameter(
				param, type(self).__name__, grad=grad)
			if grad.is_sparse:
				raise RuntimeError(
					"CurvatureAwareMuon does not support sparse gradients")

			state = self.state[param]
			if 'momentum_buffer' not in state:
				state['momentum_buffer'] = torch.zeros_like(
					grad, memory_format=torch.preserve_format)
			momentum_buffer = state['momentum_buffer']
			direction = momentum_buffer.lerp(grad, 1 - beta1)
			matrix_direction = direction.reshape(direction.shape[0], -1)
			orthogonal_update = _zeropower_via_newton_schulz(
				matrix_direction,
				group['ns_coefficients'],
				group['ns_steps'],
				group['ns_eps'],
			).reshape_as(param)
			lr_ratio = _muon_lr_ratio(
				matrix_direction.shape, group['adjust_lr_fn'])


			if weight_decay > 0:
				param.mul_(1 - lr * weight_decay)
			param.add_(orthogonal_update, alpha=-lr * lr_ratio)
			momentum_buffer.lerp_(grad, 1 - beta2)

	def _step_adamw_group(self, group):
		lr = group['lr']
		beta1, beta2 = group['betas']
		beta3 = group['beta3']
		eps = group['eps']
		weight_decay = group['weight_decay']

		for param in group['params']:
			if param.grad is None:
				continue
			grad = param.grad
			_validate_real_parameter(
				param, type(self).__name__, grad=grad)
			if grad.is_sparse:
				raise RuntimeError(
					"CurvatureAwareMuon does not support sparse gradients")

			state = self.state[param]
			if len(state) == 0:
				state['step'] = 0
				state['exp_avg'] = torch.zeros_like(
					param, memory_format=torch.preserve_format)
				state['exp_avg_sq'] = torch.zeros_like(
					param, memory_format=torch.preserve_format)

			exp_avg = state['exp_avg']
			exp_avg_sq = state['exp_avg_sq']
			state['step'] += 1
			step = state['step']
			direction = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
			exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
			exp_avg_sq.mul_(beta3).addcmul_(
				grad, grad, value=1 - beta3)

			(
				direction_correction,
				second_moment_correction,
			) = _adamw_bias_corrections(
				beta1, beta2, beta3, step, True)
			denom = (exp_avg_sq / second_moment_correction).sqrt().add_(eps)
			if weight_decay > 0:
				param.mul_(1 - lr * weight_decay)
			param.addcdiv_(
				direction, denom,
				value=-lr / direction_correction,
			)
