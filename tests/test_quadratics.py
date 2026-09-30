import json
from pathlib import Path

import numpy as np
import pytest

from experiments.quadratics.heterogeneous import dataset, run, validate
from experiments.quadratics.summarize import metrics
from experiments.quadratics.two_dimensional import trajectory


def test_quadratic_construction_and_ss_adamw_identity():
    checks = validate()
    assert checks["max_update_error"] < 1e-14
    h1, _, batches1 = dataset([17], "heterogeneous", 4, 3)
    h2, _, batches2 = dataset([17], "homogeneous", 4, 3)
    np.testing.assert_allclose(np.linalg.eigvalsh(h1), np.linalg.eigvalsh(h2))
    np.testing.assert_array_equal(batches1, batches2)


def test_runner_is_reproducible_and_keeps_paired_initialization(tmp_path):
    job = dict(name="check", seed_start=1000, n_seeds=2, steps=4,
               landscape="heterogeneous", batch_size=3, initialization="gaussian",
               configs=[dict(beta=.95, zeta=0., lr=.01), dict(beta=.95, zeta=.2, lr=.01)])
    run(job, tmp_path / "first")
    run(job, tmp_path / "second")
    with np.load(tmp_path / "first/check_results.npz") as first, np.load(tmp_path / "second/check_results.npz") as second:
        np.testing.assert_array_equal(first["losses"], second["losses"])
        np.testing.assert_array_equal(first["losses"][0, 0], first["losses"][0, 1])
        assert not first["failed"].any()


def test_score_excludes_initial_loss_and_applies_floor():
    loss = np.array([1., 1e-2, 1e-30])[:, None, None]
    score = metrics(loss)
    np.testing.assert_allclose(score["auc"], -7.)
    np.testing.assert_allclose(score["endpoint"], -12.)


def test_fixed_mass_first_step_matches_hand_calculation():
    row = trajectory([1., 10.], "m_signum", steps=1, lr=.15, mass=.5,
                     betas=(.9, .9), adaptive_mass=False, weight_decay=0.)[0]
    direction = np.array([-.1, 3.])
    expected = np.array([-1., 3.]) - .15 * direction / np.sqrt(.25 + direction**2)
    np.testing.assert_allclose([row["theta1"], row["theta2"]], expected, atol=1e-14)


def test_plans_separate_tuning_and_confirmation_seeds():
    root = Path(__file__).resolve().parents[1] / "configs/quadratics"
    for prefix in ("random", "beta"):
        tuning = json.loads((root / f"{prefix}_tune_plan.json").read_text())
        confirmation = json.loads((root / f"{prefix}_confirm_plan.json").read_text())
        assert all(job["seed_start"] == 0 and job["n_seeds"] == 32 for job in tuning)
        assert all(job["seed_start"] == 1000 and job["n_seeds"] == 128 for job in confirmation)


def test_fixed_inputs_preserve_objective_and_initialization(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/quadratics/confirmation_inputs.json"
    fixture = json.loads(path.read_text())
    h, x = np.asarray(fixture["h"]), np.asarray(fixture["x"])
    np.testing.assert_allclose(x @ x.transpose(0, 2, 1) / 9, h, atol=2e-11)
    np.testing.assert_allclose(np.linalg.norm(fixture["w0"], axis=1), 3.)
    assert fixture["seeds"] == list(range(1000, 1128))
    job = dict(name="frozen", seed_start=1000, n_seeds=2, steps=3,
               landscape="heterogeneous", batch_size=3, initialization="gaussian",
               configs=[dict(beta=.95, zeta=0., lr=.01)])
    run(job, tmp_path, path)
    with np.load(tmp_path / "frozen_data.npz") as data:
        np.testing.assert_array_equal(data["h"], h[:2])
        np.testing.assert_array_equal(data["x"], x[:2])
        np.testing.assert_array_equal(data["w0"], fixture["w0"][:2])
    with pytest.raises(ValueError, match="eigenvector factor"):
        run(dict(job, factor="symmetric"), tmp_path / "invalid", path)
