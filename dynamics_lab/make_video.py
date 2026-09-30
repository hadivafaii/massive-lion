"""Render exported dynamics-lab configs as MP4 trajectory videos."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Literal

os.environ.setdefault(
	"MPLCONFIGDIR",
	str(Path("/tmp") / "massive_lion_matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from dynamics_lab.rendering import (
	_draw_2d_basin,
	_draw_2d_endpoint,
	_draw_2d_landscape,
	_draw_3d_basin,
	_draw_3d_endpoint,
	_draw_3d_surface,
	_draw_3d_target_star,
	_draw_order,
	_draw_target_star,
	_make_projector,
	_set_projected_limits,
	create_figure,
)
from dynamics_lab.engine import SimulationState
from dynamics_lab.landscapes import coerce_config, sample_landscape


CONFIG_DIR = Path(__file__).with_name("presets")
VIDEO_DIR = Path("outputs") / "dynamics_lab"
VALID_MODES = ("parallel", "serial", "ensemble")
VALID_SERIAL_HISTORY = ("keep", "replace")
VALID_VIEWS = ("auto", "2d", "3d")


def save_dynamics_lab_video(
		config_name: str | Path,
		mode: Literal["parallel", "serial", "ensemble"] | None = None,
		output_path: str | Path | None = None,
		serial_history: Literal["keep", "replace"] = "keep",
		ensemble_optimizer: str | None = None,
		ensemble_count: int | None = None,
		view: Literal["auto", "2d", "3d"] = "auto",
		fps: int | None = None,
		frame_stride: int | None = None,
		max_frames: int | None = None,
		dpi: int = 150,
		surface_n: int = 105,
		figsize: tuple[float, float] = (6.4, 4.8),
		macro_block_size: int = 16,
		progress: bool = True,
		ensemble_instance_id: str | None = None,
) -> Path:
	"""Run an exported dynamics-lab config and save the landscape panel as MP4."""
	payload_path = resolve_config_path(config_name)
	payload = load_json(payload_path)
	config = config_from_payload(
		payload,
		mode=mode,
		ensemble_optimizer=ensemble_optimizer,
		ensemble_count=ensemble_count,
		ensemble_instance_id=ensemble_instance_id,
	)
	controls = dict(payload.get("controls", {}))
	view = resolve_view(view, controls)
	output_path = default_output_path(payload_path, config, mode) if output_path is None else Path(output_path)

	final_snapshot = run_to_endpoint(config)
	frame_stride = resolve_frame_stride(final_snapshot, frame_stride, max_frames)
	fps = resolve_fps(fps, controls)
	landscape_config = coerce_config(config.get("landscape"))
	landscape = sample_landscape(landscape_config, n=surface_n)
	renderer = LandscapeRenderer(
		config=config,
		controls=controls,
		landscape=landscape,
		landscape_config=landscape_config,
		view=view,
		dpi=dpi,
		figsize=figsize,
		final_snapshot=final_snapshot,
	)
	steps = frame_steps(final_snapshot, frame_stride)
	frames = (
		renderer.render(snapshot_at_step(final_snapshot, step, serial_history=serial_history))
		for step in steps
	)
	frames = progress_frames(frames, total_frames=len(steps), enabled=progress)
	write_mp4(
		frames,
		output_path=output_path,
		fps=fps,
		macro_block_size=macro_block_size,
	)
	return output_path


def resolve_config_path(
		config_name: str | Path,
		config_dir: str | Path = CONFIG_DIR,
) -> Path:
	"""Resolve a config by path, filename, stem, or exported JSON name."""
	config_path = Path(config_name)
	if config_path.exists():
		return config_path

	config_dir = Path(config_dir)
	local_path = config_dir / config_path
	if local_path.exists():
		return local_path
	if local_path.suffix == "":
		local_json_path = local_path.with_suffix(".json")
		if local_json_path.exists():
			return local_json_path

	query = normalize_name(config_name)
	matches = []
	for path in sorted(config_dir.glob("*.json")):
		try:
			payload = load_json(path)
		except json.JSONDecodeError:
			continue
		names = {path.name, path.stem}
		exported_name = payload.get("name")
		if exported_name:
			exported_path = Path(str(exported_name))
			names.update({str(exported_name), exported_path.name, exported_path.stem})
		if query in {normalize_name(name) for name in names}:
			matches.append(path)

	if len(matches) == 1:
		return matches[0]
	if len(matches) > 1:
		joined = ", ".join(path.name for path in matches)
		raise ValueError(f"Config name {config_name!r} is ambiguous: {joined}")
	raise FileNotFoundError(f"Could not find dynamics-lab config {config_name!r} in {config_dir}")


def load_json(path: str | Path) -> dict[str, Any]:
	return json.loads(Path(path).read_text(encoding="utf-8"))


def config_from_payload(
		payload: dict[str, Any],
		mode: Literal["parallel", "serial", "ensemble"] | None = None,
		ensemble_optimizer: str | None = None,
		ensemble_count: int | None = None,
		ensemble_instance_id: str | None = None,
) -> dict[str, Any]:
	config = copy.deepcopy(payload.get("config", payload))
	if payload.get("schema_version", 1) == 1:
		for spec in config.get("optimizers", []):
			name = spec.get("name", "Lion")
			if name in {"MassiveLion", "m_lion", "MassiveSignum", "m_signum", "VectorMassiveLion", "vm_lion"}:
				spec.setdefault("adaptive_mass", False)
			elif name in {"GRLion", "gr_lion", "gr-lion", "GR-Lion", "grlion", "GeneralRelativisticLion"}:
				spec.setdefault("adaptive_mass", True)
				spec.setdefault("tie_mass", False)
				spec.setdefault("mass_mode", "gradient_diff")
				spec.setdefault("kappa", .1)
				spec.setdefault("beta_gravity", spec.get("beta3", .99))
	if mode is not None:
		if mode not in VALID_MODES:
			raise ValueError(f"mode must be one of {VALID_MODES}")
		config["mode"] = mode
	if ensemble_optimizer is not None:
		config["ensemble_optimizer"] = str(ensemble_optimizer)
		# A legacy type override must replace a saved instance selection.
		config.pop("ensemble_instance_id", None)
	if ensemble_instance_id is not None:
		config["ensemble_instance_id"] = str(ensemble_instance_id)
	if ensemble_count is not None:
		ensemble_count = int(ensemble_count)
		if not 1 <= ensemble_count <= 250:
			raise ValueError("ensemble_count must be between 1 and 250")
		config["ensemble_count"] = ensemble_count
	return config


def run_to_endpoint(config: dict[str, Any]) -> dict[str, Any]:
	state = SimulationState()
	snapshot = state.reset(config)
	while not snapshot["done"]:
		snapshot = state.step()
	return state.snapshot(include_landscape=False)


def snapshot_at_step(
		source: dict[str, Any],
		step: int,
		serial_history: Literal["keep", "replace"] = "keep",
) -> dict[str, Any]:
	if serial_history not in VALID_SERIAL_HISTORY:
		raise ValueError(f"serial_history must be one of {VALID_SERIAL_HISTORY}")
	step = int(step)
	active_serial_start = None
	if source.get("mode") == "serial" and serial_history == "replace":
		start_steps = [
			int(learner.get("start_step", 0))
			for learner in source.get("learners", [])
			if int(learner.get("start_step", 0)) <= step
		]
		if start_steps:
			active_serial_start = max(start_steps)

	learners = []
	for learner in source.get("learners", []):
		trace = [
			sample for sample in learner.get("trace", [])
			if int(sample.get("step", 0)) <= step
		]
		if not trace:
			continue
		if active_serial_start is not None and int(learner.get("start_step", 0)) != active_serial_start:
			continue

		visible_steps = max(0, len(trace) - 1)
		learner_copy = dict(learner)
		learner_copy["trace"] = trace
		learner_copy["local_step"] = visible_steps
		learner_copy["speed_samples"] = learner.get("speed_samples", [])[:visible_steps * 2]
		learner_copy["kinetic_samples"] = learner.get("kinetic_samples", [])[:visible_steps]
		divergence_step = learner.get("divergence_step")
		if divergence_step is None:
			learner_copy["diverged"] = (
				bool(learner.get("diverged"))
				and len(trace) == len(learner.get("trace", []))
			)
		else:
			learner_copy["diverged"] = bool(learner.get("diverged")) and step >= int(divergence_step)
		learners.append(learner_copy)

	return {
		**source,
		"global_step": step,
		"done": bool(source.get("done")) and step >= int(source.get("global_step", 0)),
		"learners": learners,
	}


def frame_steps(source: dict[str, Any], frame_stride: int) -> list[int]:
	frame_stride = max(1, int(frame_stride))
	final_step = max(0, int(source.get("global_step", 0)))
	steps = list(range(0, final_step + 1, frame_stride))
	if not steps or steps[-1] != final_step:
		steps.append(final_step)
	return steps


def resolve_frame_stride(
		source: dict[str, Any],
		frame_stride: int | None,
		max_frames: int | None,
) -> int:
	if frame_stride is not None:
		return max(1, int(frame_stride))
	if max_frames is None:
		return 1
	max_frames = max(1, int(max_frames))
	final_step = max(0, int(source.get("global_step", 0)))
	if final_step == 0 or max_frames <= 1:
		return max(1, final_step)
	return max(1, math.ceil(final_step / max(1, max_frames - 1)))


class LandscapeRenderer:
	def __init__(
			self,
			config: dict[str, Any],
			controls: dict[str, Any],
			landscape: dict[str, Any],
			landscape_config: Any,
			view: Literal["2d", "3d"],
			dpi: int,
			figsize: tuple[float, float],
			final_snapshot: dict[str, Any],
	):
		self.config = config
		self.controls = controls
		self.landscape = landscape
		self.landscape_config = landscape_config
		self.view = view
		self.dpi = int(dpi)
		self.figsize = figsize
		self.final_snapshot = apply_color_overrides(final_snapshot, controls)
		self.epsilon = convergence_epsilon(config, controls)

	def render(self, snapshot: dict[str, Any]) -> np.ndarray:
		snapshot = apply_color_overrides(snapshot, self.controls)
		if self.view == "3d":
			fig, ax = self._draw_3d(snapshot)
		else:
			fig, ax = self._draw_2d(snapshot)
		del ax
		fig.canvas.draw()
		frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
		plt.close(fig)
		return np.ascontiguousarray(frame)

	def _draw_2d(self, snapshot: dict[str, Any]):
		fig, ax = create_figure(nrows=1, ncols=1, figsize=self.figsize, dpi=self.dpi, cnst=False)
		ax.set_facecolor("white")
		_draw_2d_landscape(
			ax=ax,
			landscape=self.landscape,
			cmap="Greys",
			loss_color_scale=str(self.controls.get("loss_color_scale", "log")),
			alpha=0.72,
			contour_count=13,
			contour_color="#29313a",
			contour_linewidth=0.38,
			contour_alpha_min=0.13,
			contour_alpha_max=0.31,
		)
		_draw_target_star(
			ax=ax,
			landscape=self.landscape,
			size=520.0,
			facecolor="#ffbf32",
			edgecolor="#333333",
			edgewidth=0.55,
			zorder=4,
		)
		_draw_2d_basin(
			ax=ax,
			landscape=self.landscape,
			epsilon=self.epsilon,
			color="#111111",
			linewidth=0.72,
			dash=(5.0, 3.2),
		)
		for learner in _draw_order(snapshot["learners"]):
			draw_2d_trajectory(
				ax=ax,
				learner=learner,
				linewidth=3.0,
				taper=bool(self.controls.get("trajectory_taper", True)),
				fade=bool(self.controls.get("trajectory_fade", True)),
				alpha=1.0,
			)
		for learner in _draw_order(snapshot["learners"]):
			_draw_2d_endpoint(ax=ax, learner=learner, size=95.0, edgewidth=0.75)
		ax.set_xlim(self.landscape_config.x_min, self.landscape_config.x_max)
		ax.set_ylim(self.landscape_config.y_min, self.landscape_config.y_max)
		_apply_saved_camera(
			ax,
			self.controls.get("landscape_camera", {}),
			width_px=self.figsize[0] * self.dpi,
			height_px=self.figsize[1] * self.dpi,
		)
		ax.set_aspect("equal", adjustable="box")
		ax.set_axis_off()
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
		return fig, ax

	def _draw_3d(self, snapshot: dict[str, Any]):
		camera = self.controls.get("landscape_camera", {})
		project = _make_projector(
			landscape=self.landscape,
			yaw=float(camera.get("yaw", -0.72)),
			pitch=float(camera.get("pitch", 0.72)),
			zoom=1.0,
			height_scale=0.95,
		)
		fig, ax = create_figure(nrows=1, ncols=1, figsize=self.figsize, dpi=self.dpi, cnst=False)
		ax.set_facecolor("white")
		ax.set_aspect("equal")
		ax.axis("off")
		_draw_3d_surface(
			ax=ax,
			landscape=self.landscape,
			project=project,
			cmap="Greys",
			loss_color_scale=str(self.controls.get("loss_color_scale", "log")),
			stride=2,
			alpha=0.82,
			mesh_linewidth=0.22,
			mesh_alpha=0.16,
		)
		_draw_3d_basin(
			ax=ax,
			landscape=self.landscape,
			project=project,
			epsilon=self.epsilon,
			linewidth=0.72,
			dash=(5.0, 3.2),
		)
		for learner in _draw_order(snapshot["learners"]):
			draw_3d_trajectory(
				ax=ax,
				learner=learner,
				project=project,
				landscape_config=self.landscape_config,
				linewidth=3.0,
				taper=bool(self.controls.get("trajectory_taper", True)),
				fade=bool(self.controls.get("trajectory_fade", True)),
				alpha=1.0,
			)
		for learner in _draw_order(snapshot["learners"]):
			_draw_3d_endpoint(
				ax=ax,
				learner=learner,
				project=project,
				landscape_config=self.landscape_config,
				size=95.0,
				edgewidth=0.75,
			)
		_draw_3d_target_star(
			ax=ax,
			landscape=self.landscape,
			project=project,
			size=260.0,
			facecolor="#ffbf32",
			edgecolor="#333333",
			edgewidth=0.55,
		)
		_set_projected_limits(
			ax=ax,
			landscape=self.landscape,
			project=project,
			learners=self.final_snapshot["learners"],
			pad_frac=0.035,
		)
		_apply_saved_camera(
			ax,
			camera,
			width_px=self.figsize[0] * self.dpi,
			height_px=self.figsize[1] * self.dpi,
		)
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
		return fig, ax


def _apply_saved_camera(
		ax: plt.Axes,
		camera: dict[str, Any],
		width_px: float,
		height_px: float,
) -> None:
	"""Apply the browser's zoom and pixel pan to fixed data limits."""
	zoom = max(1e-12, float(camera.get("zoom", 1.0)))
	pan_x = float(camera.get("panX", 0.0))
	pan_y = float(camera.get("panY", 0.0))
	x_min, x_max = ax.get_xlim()
	y_min, y_max = ax.get_ylim()
	x_span = x_max - x_min
	y_span = y_max - y_min
	x_center = 0.5 * (x_min + x_max) - pan_x * x_span / max(1.0, width_px * zoom)
	y_center = 0.5 * (y_min + y_max) + pan_y * y_span / max(1.0, height_px * zoom)
	ax.set_xlim(x_center - 0.5 * x_span / zoom, x_center + 0.5 * x_span / zoom)
	ax.set_ylim(y_center - 0.5 * y_span / zoom, y_center + 0.5 * y_span / zoom)


def draw_2d_trajectory(
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
	draw_path_collection(
		ax=ax,
		points=points,
		color=str(learner.get("color", "#111111")),
		linewidth=linewidth,
		taper=taper,
		fade=fade,
		alpha=alpha,
		zorder=5,
	)


def draw_3d_trajectory(
		ax: plt.Axes,
		learner: dict[str, Any],
		project,
		landscape_config: Any,
		linewidth: float,
		taper: bool,
		fade: bool,
		alpha: float,
) -> None:
	del landscape_config
	points = []
	for sample in learner.get("trace", []):
		x, y = sample["theta"]
		projected = project(x, y, sample["loss"])
		points.append([projected[0], projected[1]])
	points = np.asarray(points, dtype=float)
	if len(points) < 2:
		return
	draw_path_collection(
		ax=ax,
		points=points,
		color=str(learner.get("color", "#111111")),
		linewidth=linewidth,
		taper=taper,
		fade=fade,
		alpha=alpha,
		zorder=8,
	)


def draw_path_collection(
		ax: plt.Axes,
		points: np.ndarray,
		color: str,
		linewidth: float,
		taper: bool,
		fade: bool,
		alpha: float,
		zorder: float,
) -> None:
	segments = np.stack([points[:-1], points[1:]], axis=1)
	progress = np.linspace(1.0 / len(segments), 1.0, len(segments))
	widths = linewidth * (0.35 + 0.65 * progress) if taper else linewidth
	base = np.asarray(mcolors.to_rgba(color), dtype=float)
	colors = np.tile(base, (len(segments), 1))
	if fade:
		colors[:, 3] = alpha * (0.12 + 0.88 * progress)
	else:
		colors[:, 3] = alpha * base[3]
	collection = LineCollection(
		segments,
		colors=colors,
		linewidths=widths,
		capstyle="round",
		joinstyle="round",
		zorder=zorder,
	)
	ax.add_collection(collection)


def write_mp4(
		frames: Iterable[np.ndarray],
		output_path: str | Path,
		fps: int,
		macro_block_size: int = 16,
) -> None:
	try:
		import imageio_ffmpeg
	except ImportError as error:
		raise RuntimeError(
			"MP4 export requires imageio-ffmpeg. Install it with "
			"`python -m pip install imageio-ffmpeg`."
		) from error

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	frame_iter = iter(frames)
	try:
		first_frame = next(frame_iter)
	except StopIteration as error:
		raise ValueError("No frames were produced for the video.") from error

	height, width = first_frame.shape[:2]
	writer = imageio_ffmpeg.write_frames(
		str(output_path),
		size=(width, height),
		fps=int(fps),
		quality=8,
		pix_fmt_in="rgb24",
		pix_fmt_out="yuv420p",
		macro_block_size=max(1, int(macro_block_size)),
		ffmpeg_log_level="warning",
	)
	writer.send(None)
	try:
		writer.send(np.ascontiguousarray(first_frame).tobytes())
		for frame in frame_iter:
			writer.send(np.ascontiguousarray(frame).tobytes())
	finally:
		writer.close()


def progress_frames(
		frames: Iterable[np.ndarray],
		total_frames: int,
		enabled: bool,
) -> Iterable[np.ndarray]:
	if not enabled:
		yield from frames
		return
	report_every = max(1, total_frames // 20)
	for index, frame in enumerate(frames, start=1):
		if index == 1 or index == total_frames or index % report_every == 0:
			print(f"rendered {index}/{total_frames} frames", file=sys.stderr, flush=True)
		yield frame


def apply_color_overrides(snapshot: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
	overrides = controls.get("color_overrides", {}) or {}
	if not overrides:
		return snapshot
	snapshot = dict(snapshot)
	row_indices = {
		spec.get("instance_id"): index
		for index, spec in enumerate(snapshot.get("optimizers", []))
	}
	learners = []
	for index, learner in enumerate(snapshot.get("learners", [])):
		learner = dict(learner)
		instance_id = learner.get("instance_id")
		if snapshot.get("mode") == "ensemble":
			legacy_keys = (learner.get("name"), f"learner-{index}", learner.get("optimizer"))
		else:
			row_index = row_indices.get(instance_id, index)
			legacy_keys = (f"learner-{row_index}",)
		# Current row identities take precedence; older exports used learner-N
		# in parallel/serial mode and the optimizer title in ensemble mode.
		for key in (instance_id, learner.get("id"), *legacy_keys):
			if key in overrides:
				learner["color"] = overrides[key]
				break
		learners.append(learner)
	snapshot["learners"] = learners
	return snapshot


def convergence_epsilon(config: dict[str, Any], controls: dict[str, Any]) -> float:
	convergence = config.get("convergence", {}) or {}
	value = convergence.get(
		"epsilon",
		controls.get("convergence_epsilon", controls.get("ensemble_loss_tolerance", 0.0)),
	)
	return float(value or 0.0)


def resolve_view(view: str, controls: dict[str, Any]) -> Literal["2d", "3d"]:
	if view not in VALID_VIEWS:
		raise ValueError(f"view must be one of {VALID_VIEWS}")
	if view != "auto":
		return view
	return "3d" if controls.get("landscape_view") == "3d" else "2d"


def resolve_fps(
		fps: int | None,
		controls: dict[str, Any],
) -> int:
	if fps is not None:
		return max(1, int(fps))
	delay_ms = float(controls.get("delay_ms", 33) or 33)
	return max(1, min(60, round(1000.0 / max(1.0, delay_ms))))


def default_output_path(
		payload_path: Path,
		config: dict[str, Any],
		mode: str | None,
) -> Path:
	resolved_mode = mode or str(config.get("mode", "parallel"))
	stem = f"{payload_path.stem}_{resolved_mode}"
	if resolved_mode == "ensemble":
		selection = config.get("ensemble_instance_id") or config.get("ensemble_optimizer", "optimizer")
		stem = f"{stem}_{safe_file_part(selection)}"
	return VIDEO_DIR / f"{stem}.mp4"


def normalize_name(value: str | Path) -> str:
	path = Path(str(value))
	text = path.name if path.suffix else str(value)
	if text.endswith(".json"):
		text = text[:-5]
	return text.lower()


def safe_file_part(value: Any) -> str:
	return "".join(
		char if char.isalnum() or char in "._-" else "_"
		for char in str(value).strip()
	) or "optimizer"


def main() -> None:
	args = parse_args()
	path = save_dynamics_lab_video(
		config_name=args.config,
		mode=args.mode,
		output_path=args.output,
		serial_history=args.serial_history,
		ensemble_optimizer=args.ensemble_optimizer,
		ensemble_count=args.ensemble_count,
		ensemble_instance_id=args.ensemble_instance_id,
		view=args.view,
		fps=args.fps,
		frame_stride=args.frame_stride,
		max_frames=args.max_frames,
		dpi=args.dpi,
		surface_n=args.surface_n,
		figsize=tuple(args.figsize),
		macro_block_size=args.macro_block_size,
		progress=not args.no_progress,
	)
	print(path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("config", help="Config path, filename, stem, or exported config name.")
	parser.add_argument("--mode", choices=VALID_MODES, default=None)
	parser.add_argument("--output", default=None, help="MP4 output path.")
	parser.add_argument(
		"--serial-history",
		choices=VALID_SERIAL_HISTORY,
		default="keep",
		help="Whether previous serial trajectories remain visible.")
	parser.add_argument(
		"--ensemble-optimizer",
		default=None,
		help="Select by optimizer type (legacy); replaces a saved instance selection.")
	parser.add_argument(
		"--ensemble-instance-id",
		default=None,
		help="Select an exported optimizer row by instance_id; takes precedence over type.")
	parser.add_argument(
		"--ensemble-count",
		type=int,
		default=None,
		help="Override config.ensemble_count for ensemble mode.")
	parser.add_argument("--view", choices=VALID_VIEWS, default="auto")
	parser.add_argument("--fps", type=int, default=None)
	parser.add_argument(
		"--frame-stride",
		type=int,
		default=None,
		help="Render every Nth simulation step.")
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Pick a frame stride to keep the render under this many frames.")
	parser.add_argument("--dpi", type=int, default=150)
	parser.add_argument("--surface-n", type=int, default=105)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		metavar=("WIDTH", "HEIGHT"),
		default=(6.4, 4.8),
		help="Matplotlib figure size in inches, e.g. 12.8 7.2 for 16:9.")
	parser.add_argument(
		"--macro-block-size",
		type=int,
		default=16,
		help="FFmpeg macroblock size; use 1 to preserve exact frame dimensions.")
	parser.add_argument("--no-progress", action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	main()
