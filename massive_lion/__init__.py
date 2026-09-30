"""Massive Lion: one optimizer, with fixed/adaptive mass and named reductions."""

from .optimizer import MassiveLion
from .aliases import Lion, Signum, MassiveSignum, SecretSauceAdamW, create_optimizer

__all__ = ["MassiveLion", "Lion", "Signum", "MassiveSignum", "SecretSauceAdamW",
           "create_optimizer"]
