"""Comparison optimizers used by the experiments and Dynamics Lab."""

from .cautious import CautiousAdamW, CautiousLion
from .curvature_aware import CurvatureAwareSGD, CurvatureAwareAdamW, CurvatureAwareMuon
from .muon import MuonWithAdamW
from .vr_adam import VRAdam
from .arctan_lion import ArctanLion

QHM = CurvatureAwareSGD

__all__ = ["QHM", "CurvatureAwareSGD", "CurvatureAwareAdamW", "CurvatureAwareMuon",
           "CautiousAdamW", "CautiousLion", "MuonWithAdamW", "VRAdam", "ArctanLion"]
