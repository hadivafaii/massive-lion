"""Bounded, stateless simulations for the public website and scene publisher.

All numerical work goes through the existing engine and repository optimizers.
The process-wide lock covers reset as well as stepping: the engine temporarily
uses PyTorch's process-wide RNG when generating reproducible noise.
"""

from copy import deepcopy
from dataclasses import fields
import math
import re
import threading
from typing import Any

from dynamics_lab.engine import OptimizerSpec, SimulationState, canonical_optimizer_name
from dynamics_lab.landscapes import SharpValleyConfig
from dynamics_lab.noise import NoiseConfig


LIMITS = {
    "max_steps": 2000,
    "optimizer_rows": 12,
    "ensemble_count": 50,
    "optimizer_steps": 20000,
    "noise_batch_size": 32,
    "noise_draws": 100000,
    "landscape_grid": 85,
    "request_bytes": 65536,
    "concurrent_requests": 2,
}
SIMULATION_LOCK = threading.Lock()

_DEFAULT_NAMES = (
    "MassiveLion", "Lion", "CurvatureAwareSGD", "CurvatureAwareAdamW",
    "Signum", "MassiveSignum", "Adam", "SGD",
)
_BOOLEAN_FIELDS = {
    "adaptive_mass", "tie_mass", "foreach", "adaptive_geometry",
    "rms_speed_scale", "bias_correction",
}
_NUMERIC_FIELDS = {
    "lr", "momentum", "beta1", "beta2", "beta3", "mass", "weight_decay",
    "eps", "kappa", "beta_gravity",
}


def _number(value: Any, path: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{path} must be finite and between {minimum:g} and {maximum:g}")
    return value


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    _number(value, path, minimum, maximum)
    return value


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _choice(value: Any, path: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} must be one of: {', '.join(choices)}")
    return value


def _check_json(value: Any, path: str = "request", depth: int = 0) -> None:
    """Reject non-finite values even in optional exported display metadata."""
    if depth > 12:
        raise ValueError("Request nesting is too deep")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
    elif isinstance(value, str):
        if len(value) > 512:
            raise ValueError(f"{path} is too long (maximum 512 characters)")
    elif isinstance(value, dict):
        if len(value) > 128:
            raise ValueError(f"{path} has too many fields")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _check_json(item, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        if len(value) > 256:
            raise ValueError(f"{path} has too many entries")
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]", depth + 1)
    else:
        raise ValueError(f"{path} is not a JSON value")


def validate_config(payload: Any) -> dict[str, Any]:
    """Validate raw engine config or the lab's exported ``{config: ...}`` file.

    Return a detached config, preserving the engine's defaults and aliases.
    Browser display metadata in exports is accepted but never executed.
    """
    _object(payload, "request")
    _check_json(payload)
    config = deepcopy(_object(payload.get("config", payload), "config"))
    mode = _choice(config.get("mode", "parallel"), "mode", ("parallel", "serial", "ensemble"))
    steps = _integer(config.get("max_steps", 300), "max_steps", 1, LIMITS["max_steps"])
    ensemble_count = _integer(config.get("ensemble_count", 25), "ensemble_count", 1, LIMITS["ensemble_count"])
    theta = config.get("theta0", [-1.8, 2.3])
    if not isinstance(theta, list) or len(theta) != 2:
        raise ValueError("theta0 must contain exactly two coordinates")
    for index, value in enumerate(theta):
        _number(value, f"theta0[{index}]", -10000, 10000)

    landscape = _object(config.get("landscape", {}), "landscape")
    _choice(landscape.get("kind", "sloped_ravine"), "landscape.kind", ("sloped_ravine", "curved_valley"))
    defaults = SharpValleyConfig()
    for field in fields(SharpValleyConfig):
        if field.name == "kind":
            continue
        minimum = 0 if field.name in {"sharpness", "along_curvature", "ripple_amp", "boundary"} else -10000
        _number(landscape.get(field.name, getattr(defaults, field.name)), f"landscape.{field.name}", minimum, 10000)
    for axis in ("x", "y"):
        low = landscape.get(f"{axis}_min", getattr(defaults, f"{axis}_min"))
        high = landscape.get(f"{axis}_max", getattr(defaults, f"{axis}_max"))
        if high - low < 1e-6:
            raise ValueError(f"landscape.{axis}_max must exceed {axis}_min by at least 0.000001")

    rows = config.get("optimizers", [])
    if not isinstance(rows, list) or len(rows) > LIMITS["optimizer_rows"]:
        raise ValueError(f"optimizers must be a list with at most {LIMITS['optimizer_rows']} rows")
    specs = []
    identities = []
    for index, row in enumerate(rows or [{"name": name} for name in _DEFAULT_NAMES]):
        _object(row, f"optimizers[{index}]")
        for key in _BOOLEAN_FIELDS & row.keys():
            if not isinstance(row[key], bool):
                raise ValueError(f"optimizers[{index}].{key} must be a boolean")
        for key in _NUMERIC_FIELDS & row.keys():
            maximum = 0.999999999 if key in {"beta1", "beta2", "momentum", "beta_gravity"} else 10000
            _number(row[key], f"optimizers[{index}].{key}", 0, maximum)
        for key in ("noise_slot", "noise_stride"):
            if row.get(key) is not None:
                _integer(row[key], f"optimizers[{index}].{key}", 0 if key == "noise_slot" else 1, LIMITS["ensemble_count"])
        identity = row.get("instance_id")
        if identity is not None:
            if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", identity):
                raise ValueError(f"optimizers[{index}].instance_id must be an identifier")
            identities.append(identity)
        for key in ("label", "display_label"):
            if row.get(key) is not None and (not isinstance(row[key], str) or any(ord(char) < 32 for char in row[key])):
                raise ValueError(f"optimizers[{index}].{key} must be plain single-line text")
        if row.get("color") is not None and (not isinstance(row["color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", row["color"])):
            raise ValueError(f"optimizers[{index}].color must use #RGB or #RRGGBB")
        specs.append(OptimizerSpec.from_dict(row))
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate optimizer instance_id")
    ensemble_name = config.get("ensemble_optimizer", "Lion")
    if not isinstance(ensemble_name, str):
        raise ValueError("ensemble_optimizer must be a string")
    OptimizerSpec.from_dict({"name": canonical_optimizer_name(ensemble_name)})
    ensemble_id = config.get("ensemble_instance_id")
    if ensemble_id is not None and (not isinstance(ensemble_id, str) or ensemble_id not in identities):
        raise ValueError("ensemble_instance_id must identify an optimizer row")
    learners = ensemble_count if mode == "ensemble" else len(specs)
    if steps * learners > LIMITS["optimizer_steps"]:
        raise ValueError(f"Run exceeds {LIMITS['optimizer_steps']} optimizer steps; reduce steps or learners")

    noise_data = _object(config.get("noise", {}), "noise")
    if "enabled" in noise_data and not isinstance(noise_data["enabled"], bool):
        raise ValueError("noise.enabled must be a boolean")
    _choice(noise_data.get("mode", "additive"), "noise.mode", ("additive", "yu", "hessian"))
    _choice(noise_data.get("distribution", "gaussian"), "noise.distribution", ("gaussian", "laplace", "student_t3", "student_t5", "rademacher"))
    _choice(noise_data.get("xi_distribution", "student_t"), "noise.xi_distribution", ("student_t", "symmetric_pareto", "gaussian"))
    _integer(noise_data.get("seed", 0), "noise.seed", 0, 2**32 - 1)
    _integer(noise_data.get("batch_size", 1), "noise.batch_size", 1, LIMITS["noise_batch_size"])
    for key in ("std", "sigma0", "sigma1", "hessian_scale", "hessian_power"):
        if key in noise_data:
            _number(noise_data[key], f"noise.{key}", 0, 10 if key == "hessian_power" else 1000)
    if "tail_p" in noise_data:
        _number(noise_data["tail_p"], "noise.tail_p", 1.01, 2)
    if "tail_margin" in noise_data:
        _number(noise_data["tail_margin"], "noise.tail_margin", 0.001, 100)
    noise = NoiseConfig.from_dict(noise_data)
    if noise.enabled():
        draws = _noise_draw_count(specs, mode, ensemble_count, steps)
        if noise.mode == "yu":
            draws *= noise.batch_size
        if draws > LIMITS["noise_draws"]:
            raise ValueError(f"Run exceeds {LIMITS['noise_draws']} noise draws; reduce steps, noise batch size, or noise stride")
    return config


def _noise_draw_count(specs: list[OptimizerSpec], mode: str, count: int, steps: int) -> int:
    """Account for the engine's precomputed tables and serial fallback draws."""
    if mode == "ensemble":
        return count * steps
    identities = [
        (spec.noise_slot if spec.noise_slot is not None else index,
         max(1, spec.noise_stride if spec.noise_stride is not None else len(specs),
             (spec.noise_slot if spec.noise_slot is not None else index) + 1))
        for index, spec in enumerate(specs)
    ]
    tables: dict[int, int] = {}
    for slot, stride in identities[:1] if mode == "serial" else identities:
        tables[stride] = max(tables.get(stride, 0), (steps - 1) * stride + slot + 1)
    draws = sum(tables.values())
    if mode == "serial":
        for slot, stride in identities[1:]:
            size = tables.get(stride, 0)
            for step in range(steps):
                offset = step * stride + slot
                if offset >= size:
                    draws += offset + 1
    return draws


def simulate(payload: Any) -> dict[str, Any]:
    """Run one independent simulation and return the full existing snapshot."""
    config = validate_config(payload)
    with SIMULATION_LOCK:
        state = SimulationState()
        state.reset(config)
        while not state.done:
            state.advance()
        return state.snapshot(include_landscape=True)
