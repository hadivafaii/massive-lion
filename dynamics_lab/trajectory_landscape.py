"""Plot optimizer trajectories on analytic 2D loss landscapes."""

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from dynamics_lab.engine import SimulationState
from dynamics_lab.landscapes import coerce_config, landscape_loss
from experiments.plotting import create_figure


@dataclass
class TrajectoryResult:
	label: str
	optimizer: str
	spec: dict[str, Any]
	color: Any
	linestyle: str
	points: np.ndarray
	diverged: bool


def plot_optimizer_trajectories_on_landscape(
		landscape: dict[str, Any] | Any,
		theta0: list[float],
		optimizer_specs: list[dict[str, Any]],
		max_steps: int,
		path: str | Path | None = None,
		ax: plt.Axes | None = None,
		figsize: tuple[float, float] = (6.2, 2.15),
		xlim: tuple[float, float] | None = None,
		ylim: tuple[float, float] | None = None,
		grid_size: int = 260,
		colors: list[str] | None = None,
		labels: list[str] | None = None,
		linestyles: list[str] | None = None,
		background_cmap: str = "Greys",
		background_alpha: float = 0.36,
		background_quantiles: tuple[float, float] = (1.0, 99.0),
		background_levels: int = 80,
		contour_levels: int = 34,
		contour_color: str = "#777777",
		contour_linewidth: float = 0.35,
		contour_alpha: float = 0.28,
		rasterize_background: bool = False,
		raster_background_dpi: int = 300,
		show_axes: bool = False,
		show_legend: bool = False,
		linewidth_start: float = 0.8,
		linewidth_end: float = 2.5,
		alpha_start: float = 0.5,
		alpha_end: float = 1.0,
		equal_aspect: bool = False,
		fade: bool = True,
		taper: bool = True,
		start_marker_size: float = 34.0,
		target_marker_size: float = 70.0,
		end_marker_size: float = 12.0,
) -> tuple[plt.Figure, plt.Axes, list[TrajectoryResult]]:
	"""Run optimizer specs and plot their trajectories on a grayscale landscape."""
	landscape_config = coerce_config(
		asdict(landscape) if is_dataclass(landscape) else landscape)
	xlim = xlim or (landscape_config.x_min, landscape_config.x_max)
	ylim = ylim or (landscape_config.y_min, landscape_config.y_max)
	trajectories = run_optimizer_trajectories(
		landscape=landscape_config,
		theta0=theta0,
		optimizer_specs=optimizer_specs,
		max_steps=max_steps,
		colors=colors,
		labels=labels,
		linestyles=linestyles,
	)

	created_axes = ax is None
	if ax is None:
		fig, ax = create_figure(
			nrows=1,
			ncols=1,
			figsize=figsize,
			cnst=False,
		)
	else:
		fig = ax.figure
	if rasterize_background:
		draw_rasterized_landscape_background(
			ax=ax,
			landscape=landscape_config,
			xlim=xlim,
			ylim=ylim,
			grid_size=grid_size,
			cmap=background_cmap,
			alpha=background_alpha,
			quantiles=background_quantiles,
			background_levels=background_levels,
			contour_levels=contour_levels,
			contour_color=contour_color,
			contour_linewidth=contour_linewidth,
			contour_alpha=contour_alpha,
			dpi=raster_background_dpi,
		)
	else:
		draw_landscape_background(
			ax=ax,
			landscape=landscape_config,
			xlim=xlim,
			ylim=ylim,
			grid_size=grid_size,
			cmap=background_cmap,
			alpha=background_alpha,
			quantiles=background_quantiles,
			background_levels=background_levels,
			contour_levels=contour_levels,
			contour_color=contour_color,
			contour_linewidth=contour_linewidth,
			contour_alpha=contour_alpha,
		)
	for trajectory in trajectories:
		if trajectory.points.size == 0:
			continue
		plot_time_path(
			ax,
			trajectory.points[:, 0],
			trajectory.points[:, 1],
			color=trajectory.color,
			linestyle=trajectory.linestyle,
			linewidth_start=linewidth_start,
			linewidth_end=linewidth_end,
			label=trajectory.label,
			fade=fade,
			taper=taper,
			alpha_start=alpha_start,
			alpha_end=alpha_end,
		)
		ax.scatter(
			trajectory.points[-1, 0],
			trajectory.points[-1, 1],
			color=trajectory.color,
			s=end_marker_size,
			zorder=5,
			edgecolor="white",
			linewidth=0.4,
		)
	ax.scatter(
		[theta0[0]],
		[theta0[1]],
		facecolor="white",
		edgecolor="#222222",
		s=start_marker_size,
		marker="o",
		zorder=6,
		linewidth=0.9,
	)
	ax.scatter(
		[landscape_config.target_x],
		[landscape_config.target_y],
		facecolor="#f2c94c",
		edgecolor="#222222",
		s=target_marker_size,
		marker="*",
		zorder=6,
		linewidth=0.7,
	)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	if equal_aspect:
		ax.set_aspect("equal", adjustable="box")
	if not show_axes:
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel("")
		ax.set_ylabel("")
	if show_legend:
		ax.legend(frameon=True, fontsize=6)
	if created_axes:
		fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
	if path is not None:
		fig.savefig(path, bbox_inches="tight")
	return fig, ax, trajectories


def plot_time_path(
		ax: plt.Axes,
		xs: np.ndarray,
		ys: np.ndarray,
		color: Any,
		linestyle: str = "-",
		linewidth_start: float = 0.8,
		linewidth_end: float = 2.5,
		label: str | None = None,
		fade: bool = True,
		taper: bool = True,
		alpha_start: float = 0.5,
		alpha_end: float = 1.0,
) -> None:
	if len(xs) < 2:
		return
	rgba = mcolors.to_rgba(color)
	for index in range(len(xs) - 1):
		frac = (index + 1) / (len(xs) - 1)
		alpha = alpha_start + frac * (alpha_end - alpha_start) if fade else alpha_end
		width = linewidth_start + frac * (
			linewidth_end - linewidth_start) if taper else linewidth_end
		segment_color = (rgba[0], rgba[1], rgba[2], alpha * rgba[3])
		ax.plot(
			xs[index:index + 2],
			ys[index:index + 2],
			color=segment_color,
			linestyle=linestyle,
			linewidth=width,
			label=label if index == len(xs) - 2 else None,
			solid_capstyle="round",
			zorder=4,
		)


def run_optimizer_trajectories(
		landscape: dict[str, Any] | Any,
		theta0: list[float],
		optimizer_specs: list[dict[str, Any]],
		max_steps: int,
		colors: list[str] | None = None,
		labels: list[str] | None = None,
		linestyles: list[str] | None = None,
) -> list[TrajectoryResult]:
	landscape_config = coerce_config(
		asdict(landscape) if is_dataclass(landscape) else landscape)
	state = SimulationState()
	state.reset({
		"mode": "parallel",
		"max_steps": int(max_steps),
		"theta0": [float(theta0[0]), float(theta0[1])],
		"landscape": asdict(landscape_config),
		"noise": {"enabled": False},
		"optimizers": optimizer_specs,
	})
	while not state.done:
		state.step()
	snapshot = state.snapshot()
	results = []
	for index, learner in enumerate(snapshot["learners"]):
		trace = learner["trace"]
		points = np.asarray([sample["theta"] for sample in trace], dtype=float)
		spec = dict(optimizer_specs[index])
		label = labels[index] if labels is not None else spec.get("label", learner["name"])
		color = colors[index] if colors is not None else learner["color"]
		linestyle = linestyles[index] if linestyles is not None else "-"
		results.append(TrajectoryResult(
			label=str(label),
			optimizer=str(learner["optimizer"]),
			spec=spec,
			color=color,
			linestyle=str(linestyle),
			points=points,
			diverged=bool(learner["diverged"]),
		))
	return results


def draw_landscape_background(
		ax: plt.Axes,
		landscape: Any,
		xlim: tuple[float, float],
		ylim: tuple[float, float],
		grid_size: int,
		cmap: str,
		alpha: float,
		quantiles: tuple[float, float],
		background_levels: int,
		contour_levels: int,
		contour_color: str,
		contour_linewidth: float,
		contour_alpha: float,
) -> None:
	xs = np.linspace(xlim[0], xlim[1], grid_size)
	ys = np.linspace(ylim[0], ylim[1], grid_size)
	x_grid, y_grid = np.meshgrid(xs, ys)
	theta = torch.tensor(
		np.stack([x_grid, y_grid], axis=0),
		dtype=torch.float64,
	)
	with torch.no_grad():
		loss = landscape_loss(theta, landscape).detach().cpu().numpy()
	loss_display = np.log10(np.clip(loss, 1e-7, None))
	vmin, vmax = np.percentile(
		loss_display[np.isfinite(loss_display)],
		quantiles,
	)
	levels = np.linspace(vmin, vmax, contour_levels)
	ax.contourf(
		x_grid,
		y_grid,
		loss_display,
		levels=np.linspace(vmin, vmax, background_levels),
		cmap=cmap,
		alpha=alpha,
		extend="both",
	)
	ax.contour(
		x_grid,
		y_grid,
		loss_display,
		levels=levels,
		colors=contour_color,
		linewidths=contour_linewidth,
		alpha=contour_alpha,
	)


def draw_rasterized_landscape_background(
		ax: plt.Axes,
		landscape: Any,
		xlim: tuple[float, float],
		ylim: tuple[float, float],
		grid_size: int,
		cmap: str,
		alpha: float,
		quantiles: tuple[float, float],
		background_levels: int,
		contour_levels: int,
		contour_color: str,
		contour_linewidth: float,
		contour_alpha: float,
		dpi: int,
) -> None:
	width, height = axes_size_inches(ax)
	fig, raster_ax = plt.subplots(figsize=(width, height), dpi=dpi)
	fig.patch.set_alpha(0.0)
	raster_ax.set_position([0, 0, 1, 1])
	draw_landscape_background(
		ax=raster_ax,
		landscape=landscape,
		xlim=xlim,
		ylim=ylim,
		grid_size=grid_size,
		cmap=cmap,
		alpha=alpha,
		quantiles=quantiles,
		background_levels=background_levels,
		contour_levels=contour_levels,
		contour_color=contour_color,
		contour_linewidth=contour_linewidth,
		contour_alpha=contour_alpha,
	)
	raster_ax.set_xlim(xlim)
	raster_ax.set_ylim(ylim)
	raster_ax.set_axis_off()
	fig.canvas.draw()
	image = np.asarray(fig.canvas.buffer_rgba()).copy()
	plt.close(fig)
	ax.imshow(
		image,
		extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
		origin="upper",
		aspect="auto",
		zorder=0,
	)


def axes_size_inches(ax: plt.Axes) -> tuple[float, float]:
	position = ax.get_position()
	fig_width, fig_height = ax.figure.get_size_inches()
	width = max(position.width * fig_width, 0.1)
	height = max(position.height * fig_height, 0.1)
	return width, height
