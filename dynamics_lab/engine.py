"""Shared optimizer simulation engine for dynamics visualizations and sweeps."""

from dataclasses import asdict, dataclass, field, replace
import math
import re
from typing import Any

import torch

from massive_lion import MassiveLion, Lion, Signum, MassiveSignum, SecretSauceAdamW
from massive_lion.baselines import (
    ArctanLion, CautiousAdamW, CautiousLion, CurvatureAwareAdamW,
    CurvatureAwareMuon, CurvatureAwareSGD, MuonWithAdamW, VRAdam,
)

MASS_MODES = ("momentum_diff", "gradient_diff")
UPDATE_MODES = ("coordinate", "vector", "vector_rms")
KINEMATICS = ("minkowski", "arctan", "tanh")

from dynamics_lab.landscapes import (
	DEFAULT_LANDSCAPE, coerce_config,
	landscape_hessian, landscape_loss, sample_landscape,
)
from dynamics_lab.noise import (
	NoiseConfig,
	sample_noise as _sample_noise,
	sample_noise_unit as _sample_noise_unit,
)


OPTIMIZER_ALIASES = {
	"SignSGD": "Signum",
	"sign_sgd": "Signum",
	"r_lion": "RLion",
	"r-lion": "RLion",
	"Rlion": "RLion",
	"RefineLion": "RLion",
	"RefinedLion": "RLion",
	"ArctanLion": "RLion",
	"lion": "Lion",
	"ss_adamw": "SecretSauceAdamW",
	"ss-adamw": "SecretSauceAdamW",
	"c_adamw": "CautiousAdamW",
	"c-adamw": "CautiousAdamW",
	"C-AdamW": "CautiousAdamW",
	"CAdamW": "CautiousAdamW",
	"cautious_adamw": "CautiousAdamW",
	"cautious-adamw": "CautiousAdamW",
	"c_lion": "CautiousLion",
	"c-lion": "CautiousLion",
	"C-Lion": "CautiousLion",
	"CLion": "CautiousLion",
	"cautious_lion": "CautiousLion",
	"cautious-lion": "CautiousLion",
	"m_lion": "MassiveLion",
	"vm_lion": "VectorMassiveLion",
	"vm-lion": "VectorMassiveLion",
	"VM-Lion": "VectorMassiveLion",
	"VMLion": "VectorMassiveLion",
	"vector_massive_lion": "VectorMassiveLion",
	"vector-massive-lion": "VectorMassiveLion",
	"ca_sgd": "CurvatureAwareSGD",
	"ca_adamw": "CurvatureAwareAdamW",
	"muon": "Muon",
	"MuonWithAdamW": "Muon",
	"ca_muon": "CurvatureAwareMuon",
	"ca-muon": "CurvatureAwareMuon",
	"CA-Muon": "CurvatureAwareMuon",
	"CAMuon": "CurvatureAwareMuon",
	"gr_lion": "MassiveLion",
	"gr-lion": "MassiveLion",
	"GR-Lion": "MassiveLion",
	"grlion": "MassiveLion",
	"GeneralRelativisticLion": "MassiveLion",
    "GRLion": "MassiveLion",
    "m_signum": "MassiveSignum",
}


MASSLESS_OPTIMIZER_BACKENDS = {
	"Lion": "Lion",
	"Signum": "Signum",
}


MUON_OPTIMIZERS = {"Muon", "CurvatureAwareMuon"}


# Dynamics Lab never inherits nonzero optimizer-class decay defaults. Decay is
# opt-in through each optimizer spec.
OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
	"GD": {"lr": 0.05, "momentum": 0.0, "beta1": 0.9, "beta2": 0.99, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0},
	"SGD": {"lr": 0.05, "momentum": 0.9, "beta1": 0.9, "beta2": 0.99, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0},
	"CurvatureAwareSGD": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0},
	"CurvatureAwareAdamW": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.999, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"Muon": {"lr": 0.03, "momentum": 0.0, "beta1": 0.95, "beta2": 0.99, "beta3": 0.999, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"CurvatureAwareMuon": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.999, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"Adam": {"lr": 0.045, "momentum": 0.0, "beta1": 0.9, "beta2": 0.999, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"AdamW": {"lr": 0.03, "momentum": 0.0, "beta1": 0.9, "beta2": 0.999, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"CautiousAdamW": {"lr": 0.03, "momentum": 0.0, "beta1": 0.9, "beta2": 0.999, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"SecretSauceAdamW": {"lr": 0.03, "momentum": 0.0, "beta1": 0.99, "beta2": 0.99, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0, "eps": 0.0},
	"VRAdam": {"lr": 0.03, "momentum": 0.0, "beta1": 0.9, "beta2": 0.999, "beta3": 1.015, "mass": 0.0, "weight_decay": 0.0, "eps": 1e-8},
	"CautiousLion": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.0, "mass": 0.0, "weight_decay": 0.0},
	"RLion": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.0, "mass": 0.2, "weight_decay": 0.0},
	"MassiveLion": {"lr": 0.03, "momentum": 0.0, "beta1": 0.5, "beta2": 0.99, "beta3": 0.0, "mass": 0.2, "weight_decay": 0.0},
	"MassiveSignum": {"lr": 0.03, "momentum": 0.99, "beta1": 0.9, "beta2": 0.99, "beta3": 0.0, "mass": 0.2, "weight_decay": 0.0},
}

# Public MassiveLion defaults use the paper recurrence. The specialized names
# below are parameter presets of the same implementation.
OPTIMIZER_DEFAULTS["MassiveLion"].update(
    adaptive_mass=True, tie_mass=True, mass_mode="momentum_diff",
    kinematics="minkowski", update_mode="coordinate", foreach=True)
OPTIMIZER_DEFAULTS["Lion"] = dict(OPTIMIZER_DEFAULTS["MassiveLion"], mass=0., adaptive_mass=False)
OPTIMIZER_DEFAULTS["Signum"] = dict(OPTIMIZER_DEFAULTS["MassiveSignum"], mass=0., adaptive_mass=False)
OPTIMIZER_DEFAULTS["MassiveSignum"].update(adaptive_mass=False, tie_mass=True)
OPTIMIZER_DEFAULTS["VectorMassiveLion"] = dict(
    OPTIMIZER_DEFAULTS["MassiveLion"], adaptive_mass=False, update_mode="vector_rms")


def canonical_optimizer_name(name: str) -> str:
	return OPTIMIZER_ALIASES.get(name, name)


def backend_optimizer_name(name: str) -> str:
	name = canonical_optimizer_name(name)
	return MASSLESS_OPTIMIZER_BACKENDS.get(name, name)


def _massless_display_name(name: str) -> str | None:
	name = canonical_optimizer_name(name)
	if name in MASSLESS_OPTIMIZER_BACKENDS:
		return name
	return None


OPTIMIZER_COLORS = {'GD': '#e377c2', 'SGD': '#888888', 'CurvatureAwareSGD': '#9467bd', 'CurvatureAwareAdamW': '#6f42c1', 'Muon': '#0072b2', 'CurvatureAwareMuon': '#cc79a7', 'Adam': '#17becf', 'AdamW': '#2d2d2d', 'CautiousAdamW': '#009e73', 'SecretSauceAdamW': '#a05195', 'VRAdam': '#bcbd22', 'Lion': '#1f77b4', 'CautiousLion': '#56b4e9', 'RLion': '#8c564b', 'Signum': '#ff7f0e', 'SignSGD': '#ff7f0e', 'MassiveLion': '#d62728', 'VectorMassiveLion': '#e83e8c', 'MassiveSignum': '#2ca02c'}


def optimizer_color(name: str, fallback: str = "#111111") -> str:
	return OPTIMIZER_COLORS.get(name, fallback)


COLORS = list(OPTIMIZER_COLORS.values())
MASSLESS_NORMALIZED_MOMENTUM = 1e12
DIVERGENCE_MIN_COORD = 25.0
DIVERGENCE_SPAN_FACTOR = 8.0
DIVERGENCE_MIN_LOSS = 1e6
DIVERGENCE_LOSS_FACTOR = 1e5


@dataclass
class OptimizerSpec:
    """A complete, serializable optimizer row in the lab.

    ``tie_mass`` is a UI convenience: it resolves kappa and mass memory to
    beta2. Disabling adaptation always resolves both to zero.
    """
    name: str
    lr: float
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    beta3: float = 0.0
    mass: float = 0.0
    weight_decay: float = 0.0
    eps: float = 1e-8
    kappa: float = 0.0
    beta_gravity: float = 0.0
    update_mode: str = "coordinate"
    mass_mode: str = "momentum_diff"
    kinematics: str = "minkowski"
    adaptive_mass: bool = True
    tie_mass: bool = True
    foreach: bool = True
    label: str | None = None
    display_label: str | None = None
    instance_id: str | None = None
    color: str | None = None
    noise_slot: int | None = None
    noise_stride: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerSpec":
        name = canonical_optimizer_name(str(data.get("name", "Lion")))
        if name not in OPTIMIZER_DEFAULTS:
            raise ValueError(f"Unknown optimizer {name!r}")
        defaults = OPTIMIZER_DEFAULTS[name]
        values = {**defaults, **data, "name": name}
        beta1 = float(values["beta1"])
        beta2 = float(values["beta2"])
        mass = float(values["mass"])
        adaptive = _bool_value(values.get("adaptive_mass", name in {"MassiveLion", "SecretSauceAdamW"}))
        mode = str(values.get("update_mode", "coordinate"))
        mass_mode = str(values.get("mass_mode", "momentum_diff"))
        kinematics = str(values.get("kinematics", "minkowski"))
        tie = _bool_value(data.get("tie_mass", not any(k in data for k in ("kappa", "beta_gravity", "beta3"))))
        if name == "VectorMassiveLion":
            if _bool_value(data.get("adaptive_geometry", False)):
                raise ValueError("Legacy VectorMassiveLion adaptive_geometry is unsupported; use MassiveLion vector mass adaptation")
            mode = data.get("update_mode", "vector_rms" if _bool_value(data.get("rms_speed_scale", True)) else "vector")
        if name in {"Lion", "Signum"}:
            mass, adaptive, mode, kinematics = 0., False, "coordinate", "minkowski"
        if name in {"Signum", "MassiveSignum"}:
            beta1 = beta2 = float(data.get("momentum", data.get("beta2", defaults["momentum"])))
            mode = "coordinate"
            mass_mode, tie = "momentum_diff", True
        if name == "SecretSauceAdamW":
            if "beta1" in data and float(data["beta1"]) != beta2:
                raise ValueError("SecretSauceAdamW requires equal beta1 and beta2")
            if float(data.get("mass", 0)) != 0 or float(data.get("eps", 0)) != 0 or _bool_value(data.get("bias_correction", False)):
                raise ValueError("SecretSauceAdamW requires zero mass/epsilon and no bias correction")
            expected = {"update_mode": "coordinate", "mass_mode": "momentum_diff", "kinematics": "minkowski", "adaptive_mass": True}
            if any(key in data and data[key] != value for key, value in expected.items()):
                raise ValueError("SecretSauceAdamW requires coordinate Minkowski momentum-difference adaptation")
            if any(key in data and float(data[key]) != beta2 for key in ("kappa", "beta_gravity")):
                raise ValueError("SecretSauceAdamW mass parameters must equal beta2")
            beta1, mass, adaptive, tie = beta2, 0., True, True
            mode, mass_mode, kinematics = "coordinate", "momentum_diff", "minkowski"
        kappa = float(values.get("kappa", beta2)) if adaptive and not tie else (beta2 if adaptive else 0.)
        gravity = float(data.get("beta_gravity", data.get("beta3", beta2))) if adaptive and not tie else (beta2 if adaptive else 0.)
        if name in {"MassiveLion", "MassiveSignum", "VectorMassiveLion", "Lion", "Signum", "SecretSauceAdamW"}:
            if mode not in UPDATE_MODES or mass_mode not in MASS_MODES or kinematics not in KINEMATICS:
                raise ValueError("Unsupported MassiveLion update, mass, or kinematics mode")
            if name == "MassiveSignum" and adaptive:
                mass_mode, kappa, gravity = "momentum_diff", beta2, beta2
        return cls(
            name=name, lr=float(values["lr"]), momentum=beta2 if name in {"Signum", "MassiveSignum"} else float(values["momentum"]),
            beta1=beta1, beta2=beta2, beta3=(gravity if name in {"MassiveLion", "VectorMassiveLion", "MassiveSignum"} else float(values.get("beta3", 0))),
            mass=mass, weight_decay=float(values.get("weight_decay", 0)),
            eps=float(values.get("eps", 1e-8)), kappa=kappa, beta_gravity=gravity,
            update_mode=mode, mass_mode=mass_mode, kinematics=kinematics,
            adaptive_mass=adaptive, tie_mass=tie, foreach=_bool_value(values.get("foreach", True)),
            label=_optional_label(data.get("label"), "label"),
            display_label=_optional_label(data.get("display_label"), "display_label"),
            instance_id=_instance_id(data.get("instance_id")), color=_custom_color(data.get("color")),
            noise_slot=_optional_int(data.get("noise_slot")),
            noise_stride=_optional_int(data.get("noise_stride")),
        )

    def display_name(self) -> str:
        if self.label or self.display_label:
            return self.label or self.display_label
        if self.name == "MassiveLion":
            details = []
            if self.update_mode != "coordinate":
                details.append(self.update_mode.replace("_rms", " RMS"))
            if self.kinematics != "minkowski":
                details.append(self.kinematics)
            if details:
                return f"MassiveLion ({', '.join(details)})"
        return self.display_family()

    def display_family(self) -> str:
        return self.name

    def color_name(self) -> str:
        return self.display_family()


@dataclass
class StepSample:
	"""A post-step state sample.

	For non-initial samples, ``grad`` is the possibly noisy gradient evaluated at
	the preceding parameter value and used to produce this transition. ``theta``
	and ``loss`` both describe the resulting post-step state.
	"""

	step: int
	theta: list[float]
	loss: float
	grad: list[float]
	velocity: list[float]
	speed: list[float]
	theta1_light: float
	kinetic_momentum: float


@dataclass
class Learner:
	id: str
	spec: OptimizerSpec
	color: str
	noise_slot: int
	noise_stride: int
	theta: torch.nn.Parameter
	optimizer: torch.optim.Optimizer
	start_theta1: float
	start_step: int = 0
	local_step: int = 0
	trace: list[StepSample] = field(default_factory=list)
	speed_samples: list[float] = field(default_factory=list)
	kinetic_samples: list[float] = field(default_factory=list)
	diverged: bool = False
	divergence_step: int | None = None
	divergence_reason: str = ""

	def snapshot(self) -> dict[str, Any]:
		kinetic_regimes_supported = _supports_relativistic_kinetic(self.spec)
		return {
			"id": self.id,
			"instance_id": self.spec.instance_id,
			"label": self.spec.label,
			"display_label": self.spec.display_label,
			"name": self.spec.display_name(),
			"optimizer": self.spec.name,
			"color": self.color,
			"lr": self.spec.lr,
			"momentum": self.spec.momentum,
			"beta1": self.spec.beta1,
			"beta2": self.spec.beta2,
			"mass": self.spec.mass,
			"beta3": self.spec.beta3,
			"weight_decay": self.spec.weight_decay,
			"eps": self.spec.eps,
			"kappa": self.spec.kappa,
			"beta_gravity": self.spec.beta_gravity,
			"update_mode": self.spec.update_mode,
			"mass_mode": self.spec.mass_mode,
			"kinematics": self.spec.kinematics,
			"adaptive_mass": self.spec.adaptive_mass,
			"tie_mass": self.spec.tie_mass,
			"foreach": self.spec.foreach,
			"local_step": self.local_step,
			"start_step": self.start_step,
			"trace": [sample.__dict__ for sample in self.trace],
			"speed_samples": list(self.speed_samples),
			"kinetic_samples": list(self.kinetic_samples),
			"kinetic_regimes_supported": kinetic_regimes_supported,
			"regime_counts": (
				_regime_counts(self.kinetic_samples)
				if kinetic_regimes_supported else {}
			),
			"diverged": self.diverged,
			"divergence_step": self.divergence_step,
			"divergence_reason": self.divergence_reason,
		}


class SimulationState:
	def __init__(self):
		self.landscape = DEFAULT_LANDSCAPE
		self.mode = "parallel"
		self.max_steps = 300
		self.theta0 = [-1.8, 2.3]
		self.noise = NoiseConfig()
		self.global_step = 0
		self.serial_index = 0
		self.ensemble_optimizer = "Lion"
		self.ensemble_instance_id: str | None = None
		self.ensemble_count = 25
		self.optimizer_specs: list[OptimizerSpec] = []
		self.learners: list[Learner] = []
		self.noise_draw_tables: dict[int, list[torch.Tensor]] = {}
		self.noise_rng_state: torch.Tensor | None = None
		self.rho_ref = 1.0
		self.landscape_payload = sample_landscape(self.landscape)

	def reset(self, config: dict[str, Any]) -> dict[str, Any]:
		noise = NoiseConfig.from_dict(config.get("noise"))
		with torch.random.fork_rng(devices=[]):
			torch.manual_seed(noise.seed)
			return self._reset_with_isolated_rng(config, noise)

	def _reset_with_isolated_rng(
			self,
			config: dict[str, Any],
			noise: NoiseConfig,
	) -> dict[str, Any]:
		self.landscape = coerce_config(config.get("landscape"))
		self.mode = str(config.get("mode", "parallel"))
		self.max_steps = int(config.get("max_steps", 300))
		self.theta0 = [
			float(config.get("theta0", [-1.8, 2.3])[0]),
			float(config.get("theta0", [-1.8, 2.3])[1]),
		]
		self.noise = noise
		self.global_step = 0
		self.serial_index = 0
		self.ensemble_optimizer = canonical_optimizer_name(str(
			config.get("ensemble_optimizer", "Lion")))
		self.ensemble_instance_id = _instance_id(config.get("ensemble_instance_id"))
		self.ensemble_count = max(1, min(250, int(config.get("ensemble_count", 25))))
		self.optimizer_specs = [
			OptimizerSpec.from_dict(item)
			for item in config.get("optimizers", [])
		]
		if not self.optimizer_specs:
			self.optimizer_specs = [
				OptimizerSpec.from_dict({"name": "MassiveLion"}),
				OptimizerSpec.from_dict({"name": "Lion"}),
				OptimizerSpec.from_dict({"name": "CurvatureAwareSGD"}),
				OptimizerSpec.from_dict({"name": "CurvatureAwareAdamW"}),
				OptimizerSpec.from_dict({"name": "Signum"}),
				OptimizerSpec.from_dict({"name": "MassiveSignum"}),
				OptimizerSpec.from_dict({"name": "Adam"}),
				OptimizerSpec.from_dict({"name": "SGD"}),
			]
		_assign_instance_identities(self.optimizer_specs)
		_assign_noise_identities(self.optimizer_specs)
		ensemble_spec = self._ensemble_spec(self.ensemble_optimizer, self.ensemble_instance_id)
		if ensemble_spec.instance_id is not None:
			self.ensemble_instance_id = ensemble_spec.instance_id
			self.ensemble_optimizer = ensemble_spec.display_family()
		else:
			ensemble_spec.instance_id = f"ensemble-{ensemble_spec.name}"
		rho_specs = self.optimizer_specs
		if self.mode == "ensemble":
			rho_specs = self.optimizer_specs + [ensemble_spec]
		self.rho_ref = _reference_rho(rho_specs)
		self.noise_draw_tables = {}

		self.landscape_payload = sample_landscape(self.landscape)
		self.learners = []
		if self.mode == "serial":
			self._start_serial_learner(self.theta0)
		elif self.mode == "ensemble":
			for index in range(self.ensemble_count):
				learner = self._new_learner(
					index,
					ensemble_spec,
					self._random_theta(),
					0,
					noise_slot=index,
					noise_stride=self.ensemble_count,
				)
				self.learners.append(learner)
		else:
			for index, spec in enumerate(self.optimizer_specs):
				self.learners.append(self._new_learner(index, spec, self.theta0, 0))
		self.noise_rng_state = torch.random.get_rng_state()
		if self.noise.enabled():
			self._prepare_noise_draw_tables()

		return self.snapshot(include_landscape=True)

	def step(self) -> dict[str, Any]:
		self.advance()
		return self.snapshot()

	def advance(self) -> None:
		"""Advance once without copying growing traces into a snapshot."""
		if self.done:
			return

		if self.mode == "serial":
			self._step_serial()
		else:
			for learner in self.learners:
				if not learner.diverged and learner.local_step < self.max_steps:
					self._step_learner(learner)
			self.global_step += 1

	@property
	def done(self) -> bool:
		if self.mode == "serial":
			return self.serial_index >= len(self.optimizer_specs)
		return bool(self.learners) and all(
			learner.diverged or learner.local_step >= self.max_steps
			for learner in self.learners)

	def snapshot(self, include_landscape: bool = False) -> dict[str, Any]:
		total_steps = self.max_steps
		if self.mode == "serial":
			total_steps = self.max_steps * len(self.optimizer_specs)
		data = {
			"mode": self.mode,
			"global_step": self.global_step,
			"max_steps": self.max_steps,
			"total_steps": total_steps,
			"done": self.done,
			"theta0": self.theta0,
			"noise": self.noise.as_dict(),
			"rho_ref": self.rho_ref,
			"ensemble_optimizer": self.ensemble_optimizer,
			"ensemble_instance_id": self.ensemble_instance_id,
			"optimizers": [asdict(spec) for spec in self.optimizer_specs],
			"ensemble_count": self.ensemble_count,
			"learners": [learner.snapshot() for learner in self.learners],
		}
		if include_landscape:
			data["landscape"] = self.landscape_payload
		return data

	def _step_serial(self) -> None:
		if self.serial_index >= len(self.optimizer_specs):
			return

		learner = self.learners[-1]
		if learner.local_step < self.max_steps:
			self._step_learner(learner)
			self.global_step += 1
			if learner.local_step >= self.max_steps:
				self.serial_index += 1
				if self.serial_index < len(self.optimizer_specs):
					self._start_serial_learner(self.theta0)
			return

		self.serial_index += 1
		if self.serial_index >= len(self.optimizer_specs):
			return
		self._start_serial_learner(self.theta0)

	def _start_serial_learner(self, theta: list[float]) -> None:
		if self.serial_index >= len(self.optimizer_specs):
			return
		learner = self._new_learner(
			self.serial_index,
			self.optimizer_specs[self.serial_index],
			theta,
			self.global_step,
		)
		self.learners.append(learner)

	def _ensemble_spec(self, name: str, instance_id: str | None = None) -> OptimizerSpec:
		if instance_id is not None:
			for spec in self.optimizer_specs:
				if spec.instance_id == instance_id:
					return spec
			raise ValueError(f"Unknown ensemble_instance_id: {instance_id!r}")
		name = canonical_optimizer_name(name)
		for spec in self.optimizer_specs:
			if _matches_optimizer_request(spec, name):
				return spec
		return OptimizerSpec.from_dict({"name": name})

	def _random_theta(self) -> list[float]:
		sample = torch.rand(2, dtype=torch.float64)
		x = self.landscape.x_min + sample[0] * (self.landscape.x_max - self.landscape.x_min)
		y = self.landscape.y_min + sample[1] * (self.landscape.y_max - self.landscape.y_min)
		return [float(x), float(y)]

	def _new_learner(
			self,
			index: int,
			spec: OptimizerSpec,
			theta: list[float],
			start_step: int,
			noise_slot: int | None = None,
			noise_stride: int | None = None,
	) -> Learner:
		parameter_data = torch.tensor(theta, dtype=torch.float64)
		if spec.name in MUON_OPTIMIZERS:
			# Muon is genuinely matrix-only. Represent the two lab coordinates as a
			# column matrix, then flatten only at the simulator boundary.
			parameter_data = parameter_data.reshape(2, 1)
		param = torch.nn.Parameter(parameter_data)
		optimizer = build_optimizer(spec, param)
		slot = spec.noise_slot if noise_slot is None else noise_slot
		stride = spec.noise_stride if noise_stride is None else noise_stride
		slot = index if slot is None else slot
		stride = max(slot + 1, 1) if stride is None else stride
		learner = Learner(
			id=(f"{spec.instance_id}:member-{index + 1}" if self.mode == "ensemble" else spec.instance_id),
			spec=spec,
			color=spec.color or optimizer_color(spec.color_name(), COLORS[index % len(COLORS)]),
			noise_slot=slot,
			noise_stride=max(1, stride, slot + 1),
			theta=param,
			optimizer=optimizer,
			start_theta1=float(theta[0]),
			start_step=start_step,
		)
		coordinates = _flat_coordinates(param)
		loss = float(landscape_loss(coordinates, self.landscape).detach())
		learner.trace.append(StepSample(
			step=start_step,
			theta=[float(coordinates[0].detach()), float(coordinates[1].detach())],
			loss=loss,
			grad=[0.0, 0.0],
			velocity=[0.0, 0.0],
			speed=[0.0, 0.0],
			theta1_light=0.0,
			kinetic_momentum=0.0,
		))
		return learner

	def _step_learner(self, learner: Learner) -> None:
		if learner.diverged:
			return
		optimizer = learner.optimizer
		param = learner.theta
		try:
			optimizer.zero_grad(set_to_none=True)
			pre_step_loss = landscape_loss(
				_flat_coordinates(param), self.landscape)
			if not torch.isfinite(pre_step_loss):
				self._mark_diverged(learner, "non-finite loss")
				return
			if self._loss_is_too_large(float(pre_step_loss.detach())):
				self._mark_diverged(learner, "loss guardrail")
				return
			pre_step_loss.backward()

			previous = param.detach().clone()
			if self.noise.enabled():
				param.grad.add_(self._sample_noise(learner, param.grad))
			grad = param.grad.detach().clone()
			if not torch.isfinite(grad).all():
				self._mark_diverged(learner, "non-finite gradient")
				return
			kinetic_momentum = _kinetic_momentum(learner, grad)
			optimizer.step()
		except (RuntimeError, ValueError, OverflowError) as error:
			self._mark_diverged(learner, type(error).__name__)
			return
		normalized_kinetic = _normalized_kinetic_momentum(
			learner, kinetic_momentum)
		if not _uses_vector_kinetic(learner.spec):
			normalized_kinetic = -normalized_kinetic
		if not math.isfinite(normalized_kinetic):
			self._mark_diverged(learner, "non-finite momentum")
			return
		new_theta = param.detach().clone()
		if not torch.isfinite(new_theta).all():
			self._mark_diverged(learner, "non-finite parameter")
			return
		if self._theta_is_outside_guardrail(new_theta):
			self._mark_diverged(learner, "parameter guardrail")
			return
		with torch.no_grad():
			post_step_loss = landscape_loss(
				_flat_coordinates(param), self.landscape)
		if not torch.isfinite(post_step_loss):
			self._mark_diverged(learner, "non-finite loss")
			return
		if self._loss_is_too_large(float(post_step_loss)):
			self._mark_diverged(learner, "loss guardrail")
			return

		lr = max(float(learner.spec.lr), 1e-12)
		velocity = _flat_coordinates((new_theta - previous) / lr)
		speed = velocity.abs()
		learner.speed_samples.extend(float(value) for value in speed.tolist())
		learner.kinetic_samples.append(normalized_kinetic)
		learner.local_step += 1

		new_coordinates = _flat_coordinates(new_theta)
		grad_coordinates = _flat_coordinates(grad)
		theta1_light = (
			float(new_coordinates[0]) - learner.start_theta1) / lr
		learner.trace.append(StepSample(
			step=learner.start_step + learner.local_step,
			theta=[float(new_coordinates[0]), float(new_coordinates[1])],
			loss=float(post_step_loss),
			grad=[float(grad_coordinates[0]), float(grad_coordinates[1])],
			velocity=[float(velocity[0]), float(velocity[1])],
			speed=[float(speed[0]), float(speed[1])],
			theta1_light=theta1_light,
			kinetic_momentum=normalized_kinetic,
		))

	def _mark_diverged(self, learner: Learner, reason: str) -> None:
		if learner.divergence_step is None:
			learner.divergence_step = (
				learner.start_step + learner.local_step + 1)
		learner.diverged = True
		learner.divergence_reason = reason
		learner.local_step = self.max_steps

	def _theta_is_outside_guardrail(self, theta: torch.Tensor) -> bool:
		span = max(
			self.landscape.x_max - self.landscape.x_min,
			self.landscape.y_max - self.landscape.y_min,
			1.0,
		)
		extent = max(
			abs(self.landscape.x_min),
			abs(self.landscape.x_max),
			abs(self.landscape.y_min),
			abs(self.landscape.y_max),
		)
		limit = max(DIVERGENCE_MIN_COORD, extent, DIVERGENCE_SPAN_FACTOR * span)
		return bool(theta.detach().abs().max() > limit)

	def _loss_is_too_large(self, loss: float) -> bool:
		z_clip = float(self.landscape_payload.get("z_clip", 1.0))
		limit = max(DIVERGENCE_MIN_LOSS, DIVERGENCE_LOSS_FACTOR * abs(z_clip))
		return abs(loss) > limit

	def _prepare_noise_draw_tables(self) -> None:
		dummy = torch.zeros(2, dtype=torch.float64)
		base_rng_state = (
			self.noise_rng_state
			if self.noise_rng_state is not None
			else torch.random.get_rng_state()
		)
		required_offsets: dict[int, int] = {}
		for learner in self.learners:
			last_offset = (self.max_steps - 1) * learner.noise_stride + learner.noise_slot
			required_offsets[learner.noise_stride] = max(
				required_offsets.get(learner.noise_stride, -1),
				last_offset,
			)
		for stride, last_offset in required_offsets.items():
			with torch.random.fork_rng(devices=[]):
				torch.random.set_rng_state(base_rng_state)
				self.noise_draw_tables[stride] = [
					_sample_noise_unit(dummy, self.noise).detach().clone()
					for _ in range(last_offset + 1)
				]

	def _sample_noise(
			self,
			learner: Learner,
			grad: torch.Tensor,
	) -> torch.Tensor:
		unit_noise = self._sample_noise_unit(
			learner.noise_slot, learner.noise_stride, learner.local_step, grad)
		hessian = None
		if self.noise.mode == "hessian":
			hessian = landscape_hessian(
				_flat_coordinates(learner.theta), self.landscape)
		return _sample_noise(
			grad, self.noise, unit_noise, hessian=hessian)

	def _sample_noise_unit(
			self,
			noise_slot: int,
			noise_stride: int,
			local_step: int,
			like: torch.Tensor,
	) -> torch.Tensor:
		offset = local_step * noise_stride + noise_slot
		draws = self.noise_draw_tables.get(noise_stride)
		if draws is not None and offset < len(draws):
			return draws[offset].to(
				device=like.device, dtype=like.dtype).reshape_as(like)
		base_rng_state = (
			self.noise_rng_state
			if self.noise_rng_state is not None
			else torch.random.get_rng_state()
		)
		with torch.random.fork_rng(devices=[]):
			torch.random.set_rng_state(base_rng_state)
			for _ in range(offset):
				_sample_noise_unit(like, self.noise)
			return _sample_noise_unit(like, self.noise)


def build_optimizer(spec: OptimizerSpec, param: torch.nn.Parameter) -> torch.optim.Optimizer:
    """Instantiate the same optimizer implementation used for model training."""
    common = dict(lr=spec.lr, weight_decay=spec.weight_decay)
    betas = (spec.beta1, spec.beta2)
    if spec.name in {"GD", "SGD"}:
        return torch.optim.SGD([param], momentum=spec.momentum if spec.name == "SGD" else 0., **common)
    if spec.name in {"Adam", "AdamW", "CautiousAdamW", "CurvatureAwareAdamW", "VRAdam"}:
        classes = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW,
                   "CautiousAdamW": CautiousAdamW, "CurvatureAwareAdamW": CurvatureAwareAdamW, "VRAdam": VRAdam}
        extra = {"beta3": spec.beta3} if spec.name in {"CurvatureAwareAdamW", "VRAdam"} else {}
        return classes[spec.name]([param], betas=betas, eps=spec.eps, **common, **extra)
    if spec.name in {"CurvatureAwareSGD", "CautiousLion", "RLion"}:
        classes = {"CurvatureAwareSGD": CurvatureAwareSGD, "CautiousLion": CautiousLion, "RLion": ArctanLion}
        extra = {"mass": spec.mass} if spec.name == "RLion" else {}
        return classes[spec.name]([param], betas=betas, **common, **extra)
    if spec.name == "Muon":
        return MuonWithAdamW([{"params": [param]}], [], momentum=spec.beta1,
                            adamw_betas=(spec.beta2, spec.beta3), adamw_eps=spec.eps,
                            adjust_lr_fn="match_rms_adamw", **common)
    if spec.name == "CurvatureAwareMuon":
        return CurvatureAwareMuon([{"params": [param], "use_muon": True}], betas=betas,
                                  beta3=spec.beta3, eps=spec.eps, adjust_lr_fn="match_rms_adamw", **common)
    if spec.name in {"Lion", "Signum", "SecretSauceAdamW"}:
        classes = {"Lion": Lion, "Signum": Signum, "SecretSauceAdamW": SecretSauceAdamW}
        return classes[spec.name]([param], betas=betas, foreach=spec.foreach, **common)
    if spec.name in {"MassiveLion", "MassiveSignum", "VectorMassiveLion"}:
        # The vector and equal-beta names are convenience presets, not separate algorithms.
        optimizer_class = MassiveSignum if spec.name == "MassiveSignum" else MassiveLion
        return optimizer_class([param], betas=betas, mass=spec.mass,
                           adaptive_mass=spec.adaptive_mass, foreach=spec.foreach,
                           kappa=spec.kappa, beta_gravity=spec.beta_gravity,
                           update_mode=spec.update_mode, mass_mode=spec.mass_mode,
                           kinematics=spec.kinematics, **common)
    raise ValueError(f"Unknown optimizer: {spec.name}")


def defaults_payload() -> dict[str, Any]:
	return {
		"optimizers": OPTIMIZER_DEFAULTS,
		"optimizer_colors": OPTIMIZER_COLORS,
		"landscape": DEFAULT_LANDSCAPE.__dict__,
		"noise": NoiseConfig().as_dict(),
		"theta0": [-1.8, 2.3],
		"max_steps": 300,
	}


def _optional_label(value: Any, field_name: str) -> str | None:
	if value is None:
		return None
	if not isinstance(value, str) or any(ord(char) < 32 for char in value):
		raise ValueError(f"{field_name} must be plain single-line text")
	return value.strip() or None


def _instance_id(value: Any) -> str | None:
	if value is None or value == "":
		return None
	if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
		raise ValueError("instance_id must be a nonempty identifier using letters, numbers, ., _, :, or -")
	return value


def _custom_color(value: Any) -> str | None:
	if value is None or value == "":
		return None
	if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", value):
		raise ValueError("color must use #RGB or #RRGGBB hexadecimal notation")
	value = value.lower()
	return "#" + "".join(char * 2 for char in value[1:]) if len(value) == 4 else value


def _assign_instance_identities(specs: list[OptimizerSpec]) -> None:
	used = set()
	for spec in specs:
		if spec.instance_id is not None:
			if spec.instance_id in used:
				raise ValueError(f"Duplicate optimizer instance_id: {spec.instance_id!r}")
			used.add(spec.instance_id)
	for index, spec in enumerate(specs):
		if spec.instance_id is None:
			number = index + 1
			while f"optimizer-{number}" in used:
				number += 1
			spec.instance_id = f"optimizer-{number}"
			used.add(spec.instance_id)


def _optional_int(value: Any) -> int | None:
	if value is None:
		return None
	return int(value)


def _bool_value(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() not in {"", "0", "false", "no", "off"}
	return bool(value)


def _flat_coordinates(value: torch.Tensor) -> torch.Tensor:
	return value.reshape(-1)


def normalize_optimizer_spec(spec: OptimizerSpec) -> OptimizerSpec:
	display_name = _massless_display_name(spec.name)
	if display_name is None:
		return spec
	return replace(
		spec,
		name=MASSLESS_OPTIMIZER_BACKENDS[display_name],
		mass=0.0,
		label=spec.label,
	)


def _matches_optimizer_request(spec: OptimizerSpec, requested_name: str) -> bool:
	requested_name = canonical_optimizer_name(requested_name)
	if requested_name in MASSLESS_OPTIMIZER_BACKENDS:
		return spec.display_family() == requested_name
	return spec.name == requested_name and spec.display_family() == requested_name


def _assign_noise_identities(specs: list[OptimizerSpec]) -> None:
	default_stride = max(1, len(specs))
	for index, spec in enumerate(specs):
		if spec.noise_slot is None:
			spec.noise_slot = index
		if spec.noise_stride is None:
			spec.noise_stride = default_stride
		spec.noise_stride = max(1, spec.noise_stride, spec.noise_slot + 1)


def _regime_counts(values: list[float]) -> dict[str, int]:
	counts = {
		"classical": 0,
		"relativistic": 0,
		"ultra": 0,
		"outside": 0,
	}
	for value in values:
		magnitude = abs(value)
		if magnitude <= 0.10:
			counts["classical"] += 1
		elif magnitude < 1.0:
			counts["relativistic"] += 1
		else:
			counts["ultra"] += 1
	return counts


def _reference_rho(specs: list[OptimizerSpec]) -> float:
	for spec in specs:
		if spec.name == "MassiveLion" and spec.mass > 0:
			return spec.mass
	for spec in specs:
		if spec.mass > 0:
			return spec.mass
	return 1.0


def _normalized_kinetic_momentum(learner: Learner, kinetic_momentum: float) -> float:
    spec = learner.spec
    if not _supports_relativistic_kinetic(spec):
        return float(kinetic_momentum)
    state = learner.optimizer.state.get(learner.theta, {})
    metric = None if spec.name == "RLion" else state.get("metric_diag")
    rho = math.sqrt(max(float(_flat_coordinates(metric)[0]), 0.)) if metric is not None else spec.mass
    if rho > 0:
        return float(kinetic_momentum / rho)
    return math.copysign(MASSLESS_NORMALIZED_MOMENTUM, kinetic_momentum) if kinetic_momentum else 0.


def _supports_relativistic_kinetic(spec: OptimizerSpec) -> bool:
    return spec.name in {"RLion", "MassiveLion", "MassiveSignum", "VectorMassiveLion", "Lion", "Signum", "SecretSauceAdamW"}


def _is_massless_sign_optimizer(spec: OptimizerSpec) -> bool:
    return (_supports_relativistic_kinetic(spec) and spec.update_mode == "coordinate"
            and spec.mass == 0 and (not spec.adaptive_mass or spec.kappa == 0))


def _uses_vector_kinetic(spec: OptimizerSpec) -> bool:
    return spec.name in {"MassiveLion", "VectorMassiveLion"} and spec.update_mode != "coordinate"


def _kinetic_momentum(learner: Learner, grad: torch.Tensor) -> float:
    spec = learner.spec
    state = learner.optimizer.state.get(learner.theta, {})
    momentum = state.get("exp_avg", torch.zeros_like(grad))
    direction = spec.beta1 * momentum + (1. - spec.beta1) * grad
    if _uses_vector_kinetic(spec):
        energy = direction.square().mean() if spec.update_mode == "vector_rms" else direction.square().sum()
        return float(energy.sqrt())
    return float(_flat_coordinates(direction)[0])
