"""Shared 2D/3D landscape drawing helpers for video export."""

from typing import Any
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from experiments.plotting import create_figure

def _draw_order(learners: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return list(reversed(learners))


def _display_label(learner: dict[str, Any]) -> str:
	name = str(learner.get("name", learner.get("optimizer", "")))
	optimizer = str(learner.get("optimizer", name))
	mass = float(learner.get("mass", 0.0))
	if optimizer == "MassiveLion" and mass > 0:
		return "M-Lion"
	if optimizer == "MassiveSignum" and mass > 0:
		return "M-Signum"
	if name == "MassiveLion":
		return "M-Lion"
	if name == "MassiveSignum":
		return "M-Signum"
	return name


def _learner_color(learner: dict[str, Any]) -> str:
	return str(learner.get("color", "#111111"))


def _convergence_epsilon(data: dict[str, Any]) -> float:
	config = data.get("config", {})
	controls = data.get("controls", {})
	convergence = config.get("convergence", {})
	return float(
		convergence.get(
			"epsilon",
			controls.get("convergence_epsilon", 0.0),
		)
	)


def _loss_color_unit(
		loss: np.ndarray | float,
		z_min: float,
		z_clip: float,
		scale: str,
) -> np.ndarray | float:
	clipped = np.clip(loss, z_min, z_clip)
	if scale == "log":
		loss_range = max(1e-12, z_clip - z_min)
		offset = max(1e-9, 0.01 * loss_range)
		return np.clip(
			np.log(clipped - z_min + offset) / np.log(loss_range + offset),
			0.0,
			1.0,
		)
	return np.clip((clipped - z_min) / max(1e-12, z_clip - z_min), 0.0, 1.0)


def _loss_unit(loss: float, z_min: float, z_clip: float) -> float:
	return float(np.clip((np.clip(loss, z_min, z_clip) - z_min) / max(1e-12, z_clip - z_min), 0.0, 1.0))


def _contour_levels(
		landscape: dict[str, Any],
		count: int,
		scale: str,
) -> np.ndarray:
	z_min = float(landscape["z_min"])
	z_clip = float(landscape["z_clip"])
	if scale == "log":
		loss_range = max(1e-12, z_clip - z_min)
		offset = max(1e-9, 0.01 * loss_range)
		return z_min + np.exp(
			np.linspace(np.log(offset), np.log(loss_range + offset), count + 2)[1:-1]
		) - offset
	return np.linspace(z_min, z_clip, count + 2)[1:-1]


def _draw_2d_landscape(
		ax: plt.Axes,
		landscape: dict[str, Any],
		cmap: str,
		loss_color_scale: str,
		alpha: float,
		contour_count: int,
		contour_color: str,
		contour_linewidth: float,
		contour_alpha_min: float,
		contour_alpha_max: float,
) -> None:
	xs = np.asarray(landscape["x"], dtype=float)
	ys = np.asarray(landscape["y"], dtype=float)
	z = np.asarray(landscape["z"], dtype=float)
	z_min = float(landscape["z_min"])
	z_clip = float(landscape["z_clip"])
	color_values = _loss_color_unit(z, z_min, z_clip, loss_color_scale)
	ax.pcolormesh(
		xs,
		ys,
		color_values,
		cmap=cmap,
		vmin=0.0,
		vmax=1.0,
		shading="auto",
		alpha=alpha,
		rasterized=True,
		zorder=0,
	)
	levels = _contour_levels(landscape, contour_count, loss_color_scale)
	for index, level in enumerate(levels):
		fraction = index / max(1, len(levels) - 1)
		ax.contour(
			xs,
			ys,
			z,
			levels=[level],
			colors=[contour_color],
			linewidths=contour_linewidth,
			alpha=contour_alpha_min + fraction * (contour_alpha_max - contour_alpha_min),
			zorder=1,
		)


def _draw_2d_basin(
		ax: plt.Axes,
		landscape: dict[str, Any],
		epsilon: float,
		color: str,
		linewidth: float,
		dash: tuple[float, float],
) -> None:
	if epsilon <= 0:
		return
	optimum = landscape.get("optimum")
	if not optimum:
		return
	xs = np.asarray(landscape["x"], dtype=float)
	ys = np.asarray(landscape["y"], dtype=float)
	z = np.asarray(landscape["z"], dtype=float)
	contour = ax.contour(
		xs,
		ys,
		z,
		levels=[float(optimum["loss"]) + epsilon],
		colors=[color],
		linewidths=linewidth,
		zorder=3,
	)
	contour.set_linestyle((0.0, dash))


def _draw_2d_trajectory(
		ax: plt.Axes,
		learner: dict[str, Any],
		linewidth: float,
		taper: bool,
		fade: bool,
		alpha: float,
) -> None:
	points = np.asarray([sample["theta"] for sample in learner.get("trace", [])], dtype=float)
	if len(points) < 2:
		return
	_plot_path(
		ax=ax,
		x=points[:, 0],
		y=points[:, 1],
		color=_learner_color(learner),
		linewidth=linewidth,
		taper=taper,
		fade=fade,
		alpha=alpha,
		zorder=5,
	)


def _draw_2d_endpoint(
		ax: plt.Axes,
		learner: dict[str, Any],
		size: float,
		edgewidth: float,
) -> None:
	trace = learner.get("trace", [])
	if not trace:
		return
	x, y = trace[-1]["theta"]
	ax.scatter(
		[x],
		[y],
		s=size,
		color=_learner_color(learner),
		edgecolor="white",
		linewidth=edgewidth,
		zorder=9,
	)


def _draw_target_star(
		ax: plt.Axes,
		landscape: dict[str, Any],
		size: float,
		facecolor: str,
		edgecolor: str,
		edgewidth: float,
		zorder: float,
) -> None:
	optimum = landscape.get("optimum")
	if not optimum:
		return
	ax.scatter(
		[optimum["x"]],
		[optimum["y"]],
		s=size,
		marker="*",
		facecolor=facecolor,
		edgecolor=edgecolor,
		linewidth=edgewidth,
		zorder=zorder,
	)


def _style_2d_axes(
		ax: plt.Axes,
		show_axis_labels: bool,
		axis_label_size: float,
) -> None:
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_linewidth(0.55)
		spine.set_color("#333333")
	if show_axis_labels:
		ax.set_xlabel(r"$\theta_1$", fontsize=axis_label_size, labelpad=1.5)
		ax.set_ylabel(r"$\theta_2$", fontsize=axis_label_size, labelpad=1.5)
		ax.xaxis.set_label_coords(1.0, -0.035)
		ax.yaxis.set_label_coords(-0.04, 1.0)
	else:
		ax.set_xlabel("")
		ax.set_ylabel("")


def _make_projector(
		landscape: dict[str, Any],
		yaw: float,
		pitch: float,
		zoom: float,
		height_scale: float,
):
	xs = np.asarray(landscape["x"], dtype=float)
	ys = np.asarray(landscape["y"], dtype=float)
	x_min, x_max = float(xs[0]), float(xs[-1])
	y_min, y_max = float(ys[0]), float(ys[-1])
	x_mid = 0.5 * (x_min + x_max)
	y_mid = 0.5 * (y_min + y_max)
	z_min = float(landscape["z_min"])
	z_clip = float(landscape["z_clip"])
	pitch = float(np.clip(pitch, 0.25, 1.2))
	cy, sy = np.cos(yaw), np.sin(yaw)
	ground_tilt = np.sin(pitch)
	depth_tilt = np.cos(pitch)
	scale = 1.0 * zoom

	def project(x: float, y: float, loss: float) -> tuple[float, float, float]:
		x0 = 2.0 * (x - x_mid) / max(1e-12, x_max - x_min)
		y0 = 2.0 * (y - y_mid) / max(1e-12, y_max - y_min)
		z0 = _loss_unit(loss, z_min, z_clip)
		xr = cy * x0 - sy * y0
		yr = sy * x0 + cy * y0
		return (
			xr * scale,
			z0 * height_scale * scale - yr * ground_tilt * scale,
			yr * depth_tilt + z0 * 0.08,
		)

	return project


def _draw_3d_surface(
		ax: plt.Axes,
		landscape: dict[str, Any],
		project,
		cmap: str,
		loss_color_scale: str,
		stride: int,
		alpha: float,
		mesh_linewidth: float,
		mesh_alpha: float,
) -> None:
	xs = np.asarray(landscape["x"], dtype=float)
	ys = np.asarray(landscape["y"], dtype=float)
	z = np.asarray(landscape["z"], dtype=float)
	z_min = float(landscape["z_min"])
	z_clip = float(landscape["z_clip"])
	cmap_obj = plt.get_cmap(cmap)
	stride = max(1, int(stride))
	polygons = []
	facecolors = []
	depths = []
	for yi in range(0, len(ys) - stride, stride):
		for xi in range(0, len(xs) - stride, stride):
			face_losses = [
				z[yi, xi],
				z[yi, xi + stride],
				z[yi + stride, xi + stride],
				z[yi + stride, xi],
			]
			points = [
				project(xs[xi], ys[yi], face_losses[0]),
				project(xs[xi + stride], ys[yi], face_losses[1]),
				project(xs[xi + stride], ys[yi + stride], face_losses[2]),
				project(xs[xi], ys[yi + stride], face_losses[3]),
			]
			polygons.append([(point[0], point[1]) for point in points])
			depths.append(np.mean([point[2] for point in points]))
			color_unit = _loss_color_unit(
				float(np.mean(face_losses)),
				z_min,
				z_clip,
				loss_color_scale,
			)
			facecolors.append((*cmap_obj(color_unit)[:3], alpha))
	order = np.argsort(depths)
	collection = PolyCollection(
		[polygons[index] for index in order],
		facecolors=[facecolors[index] for index in order],
		edgecolors=(0.08, 0.09, 0.1, mesh_alpha),
		linewidths=mesh_linewidth,
		zorder=1,
	)
	ax.add_collection(collection)


def _draw_3d_axes(
		ax: plt.Axes,
		landscape: dict[str, Any],
		project,
		linewidth: float,
		alpha: float,
		label_size: float,
) -> None:
	x0, x1 = float(landscape["x"][0]), float(landscape["x"][-1])
	y0, y1 = float(landscape["y"][0]), float(landscape["y"][-1])
	base = float(landscape["z_min"])
	origin = project(x0, y0, base)
	axis_ends = (
		(project(x1, y0, base), r"$\theta_1$"),
		(project(x0, y1, base), r"$\theta_2$"),
		(project(x0, y0, float(landscape["z_clip"])), "loss"),
	)
	for end, label in axis_ends:
		ax.plot(
			[origin[0], end[0]],
			[origin[1], end[1]],
			color=(0.08, 0.09, 0.1, alpha),
			lw=linewidth,
			zorder=4,
		)
		ax.text(
			end[0],
			end[1],
			label,
			fontsize=label_size,
			color="#24272d",
			ha="left",
			va="center",
			clip_on=True,
			zorder=6,
		)


def _draw_3d_basin(
		ax: plt.Axes,
		landscape: dict[str, Any],
		project,
		epsilon: float,
		linewidth: float,
		dash: tuple[float, float],
) -> None:
	if epsilon <= 0:
		return
	optimum = landscape.get("optimum")
	if not optimum:
		return
	level = float(optimum["loss"]) + epsilon
	for a, b in _iter_contour_segments(landscape, level):
		p0 = project(a[0], a[1], level)
		p1 = project(b[0], b[1], level)
		line, = ax.plot(
			[p0[0], p1[0]],
			[p0[1], p1[1]],
			color=(0.0, 0.0, 0.0, 0.78),
			lw=linewidth,
			zorder=7,
		)
		line.set_dashes(dash)


def _draw_3d_trajectory(
		ax: plt.Axes,
		learner: dict[str, Any],
		project,
		landscape_config: Any,
		linewidth: float,
		taper: bool,
		fade: bool,
		alpha: float,
) -> None:
	points = _project_trace(learner.get("trace", []), project, landscape_config)
	if len(points) < 2:
		return
	_plot_path(
		ax=ax,
		x=np.asarray([point[0] for point in points]),
		y=np.asarray([point[1] for point in points]),
		color=_learner_color(learner),
		linewidth=linewidth,
		taper=taper,
		fade=fade,
		alpha=alpha,
		zorder=8,
	)


def _draw_3d_endpoint(
		ax: plt.Axes,
		learner: dict[str, Any],
		project,
		landscape_config: Any,
		size: float,
		edgewidth: float,
) -> None:
	trace = learner.get("trace", [])
	if not trace:
		return
	x, y = trace[-1]["theta"]
	loss = float(trace[-1].get("loss", _loss_at_theta(x, y, landscape_config)))
	point = project(x, y, loss)
	ax.scatter(
		[point[0]],
		[point[1]],
		s=size,
		color=_learner_color(learner),
		edgecolor="white",
		linewidth=edgewidth,
		zorder=10,
	)


def _draw_3d_target_star(
		ax: plt.Axes,
		landscape: dict[str, Any],
		project,
		size: float,
		facecolor: str,
		edgecolor: str,
		edgewidth: float,
) -> None:
	optimum = landscape.get("optimum")
	if not optimum:
		return
	point = project(optimum["x"], optimum["y"], optimum["loss"])
	ax.scatter(
		[point[0]],
		[point[1]],
		s=size,
		marker="*",
		facecolor=facecolor,
		edgecolor=edgecolor,
		linewidth=edgewidth,
		zorder=9,
	)


def _project_trace(
		trace: list[dict[str, Any]],
		project,
		landscape_config: Any,
) -> list[tuple[float, float, float]]:
	points = []
	for sample in trace:
		x, y = sample["theta"]
		loss = float(sample.get("loss", _loss_at_theta(x, y, landscape_config)))
		points.append(project(x, y, loss))
	return points


def _loss_at_theta(x: float, y: float, landscape_config: Any) -> float:
	import torch
	from dynamics_lab.landscapes import landscape_loss

	theta = torch.tensor([x, y], dtype=torch.float64)
	return float(landscape_loss(theta, landscape_config).detach())


def _plot_path(
		ax: plt.Axes,
		x: np.ndarray,
		y: np.ndarray,
		color: str,
		linewidth: float,
		taper: bool,
		fade: bool,
		alpha: float,
		zorder: float,
) -> None:
	if len(x) < 2:
		return
	if not taper and not fade:
		ax.plot(
			x,
			y,
			color=_with_alpha(color, alpha),
			lw=linewidth,
			solid_capstyle="round",
			solid_joinstyle="round",
			zorder=zorder,
		)
		return
	for index in range(len(x) - 1):
		progress = (index + 1) / max(1, len(x) - 1)
		width = linewidth * (0.35 + 0.65 * progress) if taper else linewidth
		line_alpha = alpha * (0.12 + 0.88 * progress) if fade else alpha
		ax.plot(
			x[index:index + 2],
			y[index:index + 2],
			color=_with_alpha(color, line_alpha),
			lw=width,
			solid_capstyle="round",
			zorder=zorder,
		)


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
	r, g, b = mpl.colors.to_rgb(color)
	return r, g, b, alpha


def _iter_contour_segments(
		landscape: dict[str, Any],
		level: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
	xs = np.asarray(landscape["x"], dtype=float)
	ys = np.asarray(landscape["y"], dtype=float)
	z = np.asarray(landscape["z"], dtype=float)
	segments = []
	for yi in range(len(ys) - 1):
		for xi in range(len(xs) - 1):
			corners = [
				(xs[xi], ys[yi], z[yi, xi]),
				(xs[xi + 1], ys[yi], z[yi, xi + 1]),
				(xs[xi + 1], ys[yi + 1], z[yi + 1, xi + 1]),
				(xs[xi], ys[yi + 1], z[yi + 1, xi]),
			]
			points = []
			for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
				point = _contour_intersection(corners[first], corners[second], level)
				if point is not None:
					points.append(point)
			if len(points) == 2:
				segments.append((points[0], points[1]))
			elif len(points) == 4:
				segments.append((points[0], points[1]))
				segments.append((points[2], points[3]))
	return segments


def _contour_intersection(
		a: tuple[float, float, float],
		b: tuple[float, float, float],
		level: float,
) -> tuple[float, float] | None:
	z0 = a[2]
	z1 = b[2]
	if (z0 < level and z1 < level) or (z0 > level and z1 > level) or z0 == z1:
		return None
	t = (level - z0) / (z1 - z0)
	return (
		a[0] + t * (b[0] - a[0]),
		a[1] + t * (b[1] - a[1]),
	)


def _set_projected_limits(
		ax: plt.Axes,
		landscape: dict[str, Any],
		project,
		learners: list[dict[str, Any]] | None,
		pad_frac: float,
) -> None:
	points = []
	for x in (float(landscape["x"][0]), float(landscape["x"][-1])):
		for y in (float(landscape["y"][0]), float(landscape["y"][-1])):
			for z in (float(landscape["z_min"]), float(landscape["z_clip"])):
				points.append(project(x, y, z))
	if learners is not None:
		for learner in learners:
			trace = learner.get("trace", [])
			for sample in trace[::max(1, len(trace) // 80)]:
				x, y = sample["theta"]
				points.append(project(x, y, sample["loss"]))
	xs = np.asarray([point[0] for point in points], dtype=float)
	ys = np.asarray([point[1] for point in points], dtype=float)
	x_pad = pad_frac * max(1e-12, float(xs.max() - xs.min()))
	y_pad = pad_frac * max(1e-12, float(ys.max() - ys.min()))
	ax.set_xlim(float(xs.min() - x_pad), float(xs.max() + x_pad))
	ax.set_ylim(float(ys.min() - y_pad), float(ys.max() + y_pad))


def _add_unit_colorbar(
		fig: plt.Figure,
		ax: plt.Axes,
		landscape: dict[str, Any],
		cmap: str,
		bounds: tuple[float, float, float, float],
		tick_size: float,
) -> None:
	mappable = mpl.cm.ScalarMappable(
		norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
		cmap=plt.get_cmap(cmap),
	)
	cax = ax.inset_axes(bounds)
	cbar = fig.colorbar(mappable, cax=cax)
	cbar.set_ticks([0.0, 1.0])
	cbar.set_ticklabels([
		_format_loss_tick(float(landscape["z_min"])),
		_format_loss_tick(float(landscape["z_clip"])),
	])
	cbar.outline.set_linewidth(0.4)
	cbar.ax.tick_params(length=1.5, width=0.35, pad=1.2, labelsize=tick_size)


def _format_loss_tick(value: float) -> str:
	if abs(value) >= 100.0 or (0 < abs(value) < 1e-2):
		return f"{value:.1e}".replace("e+0", "e+").replace("e-0", "e-")
	return f"{value:.2g}"


def _add_3d_legend(
		ax: plt.Axes,
		learners: list[dict[str, Any]],
		loc: str,
		bbox: tuple[float, float],
		fontsize: float,
		handlelength: float,
) -> None:
	by_label = {_display_label(learner): learner for learner in learners}
	handles = []
	for label in LEGEND_ORDER_3D:
		learner = by_label.get(label)
		if learner is None:
			continue
		handles.append(Line2D(
			[0],
			[0],
			color=_learner_color(learner),
			lw=1.8,
			label=label,
		))
	if not handles:
		return
	legend = ax.legend(
		handles=handles,
		loc=loc,
		bbox_to_anchor=bbox,
		frameon=True,
		framealpha=0.86,
		facecolor="white",
		edgecolor="#d0d0d0",
		fontsize=fontsize,
		handlelength=handlelength,
		borderpad=0.35,
		labelspacing=0.26,
	)
	legend.get_frame().set_linewidth(0.45)

