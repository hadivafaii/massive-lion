"""Gradient-noise processes used by simulations and visualizations."""

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class NoiseConfig:
	enabled_flag: bool = False
	mode: str = "additive"
	distribution: str = "gaussian"
	std: float = 0.0
	seed: int = 0
	xi_distribution: str = "student_t"
	sigma0: float = 0.03
	sigma1: float = 0.6
	tail_p: float = 1.5
	tail_margin: float = 0.3
	batch_size: int = 1
	hessian_scale: float = 0.1
	hessian_power: float = 1.0

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "NoiseConfig":
		if data is None:
			return cls()
		mode = str(data.get("mode", "additive"))
		std = max(0.0, float(data.get("std", 0.0)))
		sigma0 = max(0.0, float(data.get("sigma0", 0.03)))
		sigma1 = max(0.0, float(data.get("sigma1", 0.6)))
		hessian_scale = max(0.0, float(data.get("hessian_scale", 0.1)))
		hessian_power = max(0.0, float(data.get("hessian_power", 1.0)))
		if mode == "yu":
			enabled_default = sigma0 > 0.0 or sigma1 > 0.0
		elif mode == "hessian":
			enabled_default = hessian_scale > 0.0
		else:
			enabled_default = std > 0.0
		return cls(
			enabled_flag=_bool_value(data.get("enabled", enabled_default)),
			mode=mode,
			distribution=str(data.get("distribution", "gaussian")),
			std=std,
			seed=int(data.get("seed", 0)),
			xi_distribution=str(data.get("xi_distribution", "student_t")),
			sigma0=sigma0,
			sigma1=sigma1,
			tail_p=min(2.0, max(1.01, float(data.get("tail_p", 1.5)))),
			tail_margin=max(1e-3, float(data.get("tail_margin", 0.3))),
			batch_size=max(1, int(data.get("batch_size", 1))),
			hessian_scale=hessian_scale,
			hessian_power=hessian_power,
		)

	def enabled(self) -> bool:
		if not self.enabled_flag:
			return False
		if self.mode == "yu":
			return self.sigma0 > 0.0 or self.sigma1 > 0.0
		if self.mode == "hessian":
			return self.hessian_scale > 0.0
		return self.std > 0.0

	def as_dict(self) -> dict[str, Any]:
		data = self.__dict__.copy()
		data["enabled"] = data.pop("enabled_flag")
		return data


def sample_noise(
		like: torch.Tensor,
		config: NoiseConfig,
		unit_noise: torch.Tensor | None = None,
		*,
		hessian: torch.Tensor | None = None,
) -> torch.Tensor:
	"""Sample gradient noise, optionally from a precomputed unit-noise draw."""
	unit = sample_noise_unit(like, config) if unit_noise is None else unit_noise
	if config.mode == "yu":
		scale = config.sigma0 + config.sigma1 * like.detach().abs()
		return scale * unit
	if config.mode == "hessian":
		if hessian is None:
			raise ValueError("hessian noise requires a Hessian matrix")
		return _transform_hessian_noise(unit, hessian, config)
	return config.std * unit


def sample_noise_unit(like: torch.Tensor, config: NoiseConfig) -> torch.Tensor:
	"""Sample the standardized noise before amplitude or gradient-dependent scale."""
	if config.mode == "hessian":
		return torch.randn(like.shape, device=like.device, dtype=like.dtype)
	if config.mode == "yu":
		xi = torch.zeros_like(like)
		for _ in range(config.batch_size):
			xi = xi + _sample_yu_xi(like, config)
		return xi / config.batch_size

	name = config.distribution
	if name == "laplace":
		dist = torch.distributions.Laplace(
			loc=like.new_tensor(0.0),
			scale=like.new_tensor(1.0 / math.sqrt(2.0)),
		)
		return dist.sample(like.shape)
	if name == "student_t3":
		dist = torch.distributions.StudentT(df=like.new_tensor(3.0))
		return math.sqrt(1.0 / 3.0) * dist.sample(like.shape)
	if name == "student_t5":
		dist = torch.distributions.StudentT(df=like.new_tensor(5.0))
		return math.sqrt(3.0 / 5.0) * dist.sample(like.shape)
	if name == "rademacher":
		return 2.0 * torch.randint(
			0, 2, like.shape, device=like.device, dtype=like.dtype) - 1.0
	return torch.randn(
		like.shape, device=like.device, dtype=like.dtype)


def _transform_hessian_noise(
		unit_noise: torch.Tensor,
		hessian: torch.Tensor,
		config: NoiseConfig,
) -> torch.Tensor:
	flat_unit = unit_noise.reshape(-1)
	matrix = hessian.detach().to(
		device=flat_unit.device, dtype=flat_unit.dtype)
	if matrix.ndim != 2 or matrix.shape != (flat_unit.numel(), flat_unit.numel()):
		raise ValueError(
			"hessian shape must match the flattened gradient dimension")
	matrix = 0.5 * (matrix + matrix.T)
	eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
	spectral_scale = eigenvalues.abs().pow(0.5 * config.hessian_power)
	transformed = eigenvectors @ (
		spectral_scale * (eigenvectors.T @ flat_unit))
	return (
		math.sqrt(config.hessian_scale) * transformed
	).reshape_as(unit_noise)


def _sample_yu_xi(like: torch.Tensor, config: NoiseConfig) -> torch.Tensor:
	name = config.xi_distribution
	p = config.tail_p
	margin = config.tail_margin
	if name == "symmetric_pareto":
		alpha = p + margin
		u = torch.rand(like.shape, device=like.device, dtype=like.dtype).clamp_min(1e-12)
		magnitude = u.pow(-1.0 / alpha) - 1.0
		sign = _random_sign(like)
		moment = _lomax_abs_moment(alpha, p)
		return sign * magnitude / moment ** (1.0 / p)
	if name == "gaussian":
		moment = _gaussian_abs_moment(p)
		return torch.randn(like.shape, device=like.device, dtype=like.dtype) / moment ** (1.0 / p)
	if name == "student_t":
		nu = p + margin
		dist = torch.distributions.StudentT(df=like.new_tensor(nu))
		moment = _student_t_abs_moment(nu, p)
		return dist.sample(like.shape) / moment ** (1.0 / p)
	raise ValueError(f"Unknown Yu xi distribution: {name}")


def _random_sign(like: torch.Tensor) -> torch.Tensor:
	return 2.0 * torch.randint(
		0, 2, like.shape, device=like.device, dtype=like.dtype) - 1.0


def _bool_value(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() not in {"", "0", "false", "no", "off"}
	return bool(value)


def _student_t_abs_moment(nu: float, p: float) -> float:
	log_moment = (
		0.5 * p * math.log(nu)
		+ math.lgamma((p + 1.0) / 2.0)
		+ math.lgamma((nu - p) / 2.0)
		- 0.5 * math.log(math.pi)
		- math.lgamma(nu / 2.0)
	)
	return math.exp(log_moment)


def _gaussian_abs_moment(p: float) -> float:
	log_moment = (
		0.5 * p * math.log(2.0)
		+ math.lgamma((p + 1.0) / 2.0)
		- 0.5 * math.log(math.pi)
	)
	return math.exp(log_moment)


def _lomax_abs_moment(alpha: float, p: float) -> float:
	log_moment = (
		math.lgamma(p + 1.0)
		+ math.lgamma(alpha - p)
		- math.lgamma(alpha)
	)
	return math.exp(log_moment)
