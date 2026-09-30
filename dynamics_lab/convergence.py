"""Reusable convergence criteria for synthetic optimizer simulations."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConvergenceCriterion:
	window: int = 25
	frac_within: float = 0.8
	epsilon: float = 0.1

	def normalized(self) -> "ConvergenceCriterion":
		return ConvergenceCriterion(
			window=max(1, int(self.window)),
			frac_within=min(1.0, max(0.0, float(self.frac_within))),
			epsilon=max(0.0, float(self.epsilon)),
		)


@dataclass(frozen=True)
class ConvergenceResult:
	converged: int
	total: int
	fraction: float
	criterion: ConvergenceCriterion


def learner_converged(
		learner: Any,
		criterion: ConvergenceCriterion | None = None,
		optimum_loss: float = 0.0,
) -> bool:
	"""Return whether a learner spends enough late-window samples near optimum."""
	criterion = (criterion or ConvergenceCriterion()).normalized()
	if _field(learner, "diverged", False):
		return False

	trace = _field(learner, "trace", [])
	if len(trace) < criterion.window:
		return False

	threshold = float(optimum_loss) + criterion.epsilon
	recent = trace[-criterion.window:]
	within = sum(
		1 for sample in recent
		if _sample_loss(sample) <= threshold
	)
	return within / criterion.window >= criterion.frac_within


def ensemble_convergence(
		learners: list[Any],
		criterion: ConvergenceCriterion | None = None,
		optimum_loss: float = 0.0,
) -> ConvergenceResult:
	"""Count late-window convergence over an ensemble of learners."""
	criterion = (criterion or ConvergenceCriterion()).normalized()
	total = len(learners)
	converged = sum(
		1 for learner in learners
		if learner_converged(learner, criterion, optimum_loss)
	)
	return ConvergenceResult(
		converged=converged,
		total=total,
		fraction=converged / total if total else 0.0,
		criterion=criterion,
	)


def _field(item: Any, key: str, default: Any = None) -> Any:
	if isinstance(item, Mapping):
		return item.get(key, default)
	return getattr(item, key, default)


def _sample_loss(sample: Any) -> float:
	return float(_field(sample, "loss", 0.0))
