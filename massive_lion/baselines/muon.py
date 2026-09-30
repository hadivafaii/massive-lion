import torch


def _materialize_param_groups(param_groups, use_muon):
	groups = []
	for group in param_groups:
		params = list(group['params'])
		if not params:
			continue
		materialized = dict(group)
		materialized['params'] = params
		materialized['use_muon'] = use_muon
		groups.append(materialized)
	return groups


class MuonWithAdamW(torch.optim.Optimizer):
	"""One optimizer interface for official PyTorch Muon plus AdamW fallback."""

	def __init__(
			self,
			muon_param_groups,
			adamw_param_groups,
			lr=1e-3,
			momentum=0.95,
			adamw_betas=(0.9, 0.999),
			weight_decay=0.1,
			adamw_eps=1e-8,
			adjust_lr_fn='match_rms_adamw',
			adamw_fused=False,
	):
		if not hasattr(torch.optim, 'Muon'):
			raise RuntimeError(
				"The 'muon' optimizer requires a PyTorch version with "
				"torch.optim.Muon")

		muon_groups = _materialize_param_groups(
			muon_param_groups, use_muon=True)
		adamw_groups = _materialize_param_groups(
			adamw_param_groups, use_muon=False)
		if not muon_groups and not adamw_groups:
			raise ValueError("optimizer got an empty parameter list")

		self.muon_optimizer = None
		if muon_groups:
			self.muon_optimizer = torch.optim.Muon(
				muon_groups,
				lr=lr,
				weight_decay=weight_decay,
				momentum=momentum,
				nesterov=False,
				adjust_lr_fn=adjust_lr_fn,
			)

		self.adamw_optimizer = None
		if adamw_groups:
			self.adamw_optimizer = torch.optim.AdamW(
				adamw_groups,
				lr=lr,
				weight_decay=weight_decay,
				betas=adamw_betas,
				eps=adamw_eps,
				fused=adamw_fused,
			)

		combined_groups = []
		for optimizer in (self.muon_optimizer, self.adamw_optimizer):
			if optimizer is not None:
				combined_groups.extend(optimizer.param_groups)
		super().__init__(combined_groups, defaults={})
		self._bind_child_optimizers()
		self._composite_initialized = True

	def _bind_child_optimizers(self):
		if self.muon_optimizer is not None:
			self.muon_optimizer.param_groups = [
				group for group in self.param_groups if group['use_muon']]
			self.muon_optimizer.state = self.state
		if self.adamw_optimizer is not None:
			self.adamw_optimizer.param_groups = [
				group for group in self.param_groups if not group['use_muon']]
			self.adamw_optimizer.state = self.state

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()
		if self.muon_optimizer is not None:
			self.muon_optimizer.step()
		if self.adamw_optimizer is not None:
			self.adamw_optimizer.step()
		return loss

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self._bind_child_optimizers()

	def add_param_group(self, param_group):
		if getattr(self, '_composite_initialized', False):
			raise RuntimeError(
				"MuonWithAdamW does not support adding parameter groups "
				"after construction")
		super().add_param_group(param_group)
