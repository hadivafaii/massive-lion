"""Analytic 2D loss landscapes for optimizer kinematics demos."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch


@dataclass
class SharpValleyConfig:
	kind: str = "sloped_ravine"
	x_min: float = -3.4
	x_max: float = 3.4
	y_min: float = -2.4
	y_max: float = 2.4
	target_x: float = 2.2
	target_y: float = 0.0
	sharpness: float = 10.0
	valley_amp: float = 0.5
	valley_freq: float = 0.0
	valley_tilt: float = -0.55
	along_curvature: float = 0.08
	ripple_amp: float = 0.03
	ripple_freq: float = 6.0
	boundary: float = 0.02


DEFAULT_LANDSCAPE = SharpValleyConfig()


def coerce_config(data: dict[str, Any] | None = None) -> SharpValleyConfig:
	if data is None:
		return SharpValleyConfig()

	allowed = {field.name for field in fields(SharpValleyConfig)}
	values = {}
	for key, value in data.items():
		if key not in allowed or value is None:
			continue
		if key == "kind":
			values[key] = str(value)
		else:
			values[key] = float(value)
	return SharpValleyConfig(**values)


def landscape_loss(
		theta: torch.Tensor,
		config: SharpValleyConfig | dict[str, Any] | None = None,
) -> torch.Tensor:
	"""Return a sharp, curved valley loss with nontrivial theta_1 dynamics."""
	if not isinstance(config, SharpValleyConfig):
		config = coerce_config(config)
	if config.kind == "sloped_ravine":
		return _torch_sloped_ravine_loss(theta, config)

	x = theta[0]
	y = theta[1]
	center = _torch_valley_center(x, config)
	cross = y - center
	along = x - config.target_x

	loss = 0.5 * config.sharpness * cross.square()
	loss = loss + 0.5 * config.along_curvature * along.square()
	loss = loss + config.ripple_amp * (1.0 - torch.cos(config.ripple_freq * along))
	loss = loss + config.boundary * (x.pow(4) + y.pow(4))
	return loss


def landscape_hessian(
		theta: torch.Tensor,
		config: SharpValleyConfig | dict[str, Any] | None = None,
) -> torch.Tensor:
	"""Return the exact Hessian of the analytic landscape at ``theta``."""
	if not isinstance(config, SharpValleyConfig):
		config = coerce_config(config)
	coordinates = theta.detach().reshape(-1)
	if coordinates.numel() != 2:
		raise ValueError("landscape theta must contain exactly two coordinates")
	x = coordinates[0]
	y = coordinates[1]
	along = x - config.target_x

	if config.kind == "sloped_ravine":
		tanh_term = torch.tanh(config.valley_freq * (x - 0.2))
		sech_squared = 1.0 - tanh_term.square()
		center = (
			config.target_y
			+ config.valley_tilt * x
			+ config.valley_amp
			+ 0.12 * tanh_term
		)
		center_prime = (
			config.valley_tilt
			+ 0.12 * config.valley_freq * sech_squared
		)
		center_second = (
			-0.24
			* config.valley_freq ** 2
			* tanh_term
			* sech_squared
		)
		outer_x = 2.0 * config.boundary
		outer_y = x.new_tensor(2.0 * config.boundary)
	else:
		phase = config.valley_freq * x
		center = (
			config.target_y
			+ config.valley_amp * torch.sin(phase)
			+ config.valley_tilt * x
		)
		center_prime = (
			config.valley_amp * config.valley_freq * torch.cos(phase)
			+ config.valley_tilt
		)
		center_second = (
			-config.valley_amp * config.valley_freq ** 2 * torch.sin(phase)
		)
		outer_x = 12.0 * config.boundary * x.square()
		outer_y = 12.0 * config.boundary * y.square()

	cross = y - center
	hessian_xx = (
		config.sharpness * (center_prime.square() - cross * center_second)
		+ config.along_curvature
		+ config.ripple_amp
		* config.ripple_freq ** 2
		* torch.cos(config.ripple_freq * along)
		+ outer_x
	)
	hessian_xy = -config.sharpness * center_prime
	hessian_yy = config.sharpness + outer_y
	return torch.stack((
		torch.stack((hessian_xx, hessian_xy)),
		torch.stack((hessian_xy, hessian_yy)),
	))


def sample_landscape(
		config: SharpValleyConfig | dict[str, Any] | None = None,
		n: int = 85,
) -> dict[str, Any]:
	"""Sample the landscape and contour segments for the browser."""
	if not isinstance(config, SharpValleyConfig):
		config = coerce_config(config)

	with np.errstate(over="ignore", invalid="ignore"):
		xs = np.linspace(config.x_min, config.x_max, n)
		ys = np.linspace(config.y_min, config.y_max, n)
		x_grid, y_grid = np.meshgrid(xs, ys)
		z = _numpy_loss(x_grid, y_grid, config)
	if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(z).all()):
		raise ValueError("Landscape coordinates and loss must be finite across the plotting bounds")
	z_clip = float(np.percentile(z, 97.0))
	levels = np.linspace(float(np.min(z)), z_clip, 12)[1:]
	segments = _contour_segments(xs, ys, z, levels)
	min_y, min_x = np.unravel_index(np.argmin(z), z.shape)
	opt_x = float(xs[min_x])
	opt_y = float(ys[min_y])
	opt_loss = float(z[min_y, min_x])
	curvature = _local_curvature_payload(config, opt_x, opt_y)

	return {
		"x": xs.tolist(),
		"y": ys.tolist(),
		"z": z.tolist(),
		"z_min": float(np.min(z)),
		"z_clip": z_clip,
		"levels": levels.tolist(),
		"contours": segments,
		"optimum": {
			"x": opt_x,
			"y": opt_y,
			"loss": opt_loss,
		},
		**curvature,
	}


def _torch_valley_center(x: torch.Tensor, config: SharpValleyConfig) -> torch.Tensor:
	return (
		config.target_y
		+ config.valley_amp * torch.sin(config.valley_freq * x)
		+ config.valley_tilt * x
	)


def _numpy_valley_center(x: np.ndarray, config: SharpValleyConfig) -> np.ndarray:
	return (
		config.target_y
		+ config.valley_amp * np.sin(config.valley_freq * x)
		+ config.valley_tilt * x
	)


def _numpy_loss(
		x: np.ndarray,
		y: np.ndarray,
		config: SharpValleyConfig,
) -> np.ndarray:
	if config.kind == "sloped_ravine":
		return _numpy_sloped_ravine_loss(x, y, config)

	center = _numpy_valley_center(x, config)
	cross = y - center
	along = x - config.target_x
	return (
		0.5 * config.sharpness * cross ** 2
		+ 0.5 * config.along_curvature * along ** 2
		+ config.ripple_amp * (1.0 - np.cos(config.ripple_freq * along))
		+ config.boundary * (x ** 4 + y ** 4)
	)


def _torch_sloped_ravine_loss(
		theta: torch.Tensor,
		config: SharpValleyConfig,
) -> torch.Tensor:
	x = theta[0]
	y = theta[1]
	center = _torch_sloped_center(x, config)
	cross = y - center
	along = x - config.target_x
	return (
		0.5 * config.sharpness * cross.square()
		+ 0.5 * config.along_curvature * along.square()
		+ config.boundary * (x.square() + y.square())
		+ config.ripple_amp * (1.0 - torch.cos(config.ripple_freq * along))
	)


def _numpy_sloped_ravine_loss(
		x: np.ndarray,
		y: np.ndarray,
		config: SharpValleyConfig,
) -> np.ndarray:
	center = _numpy_sloped_center(x, config)
	cross = y - center
	along = x - config.target_x
	return (
		0.5 * config.sharpness * cross ** 2
		+ 0.5 * config.along_curvature * along ** 2
		+ config.boundary * (x ** 2 + y ** 2)
		+ config.ripple_amp * (1.0 - np.cos(config.ripple_freq * along))
	)


def _torch_sloped_center(
		x: torch.Tensor,
		config: SharpValleyConfig,
) -> torch.Tensor:
	center = config.target_y + config.valley_tilt * x + config.valley_amp
	if config.valley_freq != 0:
		center = center + 0.12 * torch.tanh(config.valley_freq * (x - 0.2))
	return center


def _numpy_sloped_center(
		x: np.ndarray,
		config: SharpValleyConfig,
) -> np.ndarray:
	center = config.target_y + config.valley_tilt * x + config.valley_amp
	if config.valley_freq != 0:
		center = center + 0.12 * np.tanh(config.valley_freq * (x - 0.2))
	return center


def _local_curvature_payload(
		config: SharpValleyConfig,
		x: float,
		y: float,
) -> dict[str, Any]:
	theta = torch.tensor([x, y], dtype=torch.float64)
	hessian = landscape_hessian(theta, config)
	hessian_np = hessian.detach().cpu().numpy()
	eigenvalues = np.linalg.eigvalsh(hessian_np)
	positive = eigenvalues[eigenvalues > 1e-12]
	curvature_ref = None
	if positive.size > 0:
		curvature_ref = float(np.exp(np.mean(np.log(positive))))
		if not np.isfinite(curvature_ref):
			curvature_ref = None

	return {
		"hessian": hessian_np.tolist(),
		"hessian_eigenvalues": eigenvalues.tolist(),
		"curvature_ref": curvature_ref,
	}


# noinspection PyTypeChecker
def _contour_segments(
		xs: np.ndarray,
		ys: np.ndarray,
		z: np.ndarray,
		levels: np.ndarray,
) -> list[dict[str, Any]]:
	segments: list[dict[str, Any]] = []
	ny, nx = z.shape

	for level_index, level in enumerate(levels):
		for yi in range(ny - 1):
			for xi in range(nx - 1):
				corners = [
					(xs[xi], ys[yi], z[yi, xi]),
					(xs[xi + 1], ys[yi], z[yi, xi + 1]),
					(xs[xi + 1], ys[yi + 1], z[yi + 1, xi + 1]),
					(xs[xi], ys[yi + 1], z[yi + 1, xi]),
				]
				points = []
				for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
					point = _edge_intersection(corners[a], corners[b], level)
					if point is not None:
						points.append(point)

				if len(points) == 2:
					segments.append({
						"level": level_index,
						"points": [points[0], points[1]],
					})
				elif len(points) == 4:
					segments.append({
						"level": level_index,
						"points": [points[0], points[1]],
					})
					segments.append({
						"level": level_index,
						"points": [points[2], points[3]],
					})

	return segments


def _edge_intersection(
		a: tuple[float, float, float],
		b: tuple[float, float, float],
		level: float,
) -> list[float] | None:
	x0, y0, z0 = a
	x1, y1, z1 = b
	if (z0 < level and z1 < level) or (z0 > level and z1 > level):
		return None
	if z0 == z1:
		return None

	t = (level - z0) / (z1 - z0)
	if t < 0.0 or t > 1.0:
		return None
	return [float(x0 + t * (x1 - x0)), float(y0 + t * (y1 - y0))]
