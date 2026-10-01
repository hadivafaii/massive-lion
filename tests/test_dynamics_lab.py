import copy
import unittest

import torch

from dynamics_lab.engine import SimulationState
from dynamics_lab.landscapes import (
	coerce_config, landscape_hessian, landscape_loss, sample_landscape,
)
from dynamics_lab.noise import NoiseConfig, sample_noise


def run_trace(config):
	state = SimulationState()
	snapshot = state.reset(config)
	while not snapshot["done"]:
		snapshot = state.step()
	return {
		learner["name"]: learner["trace"]
		for learner in snapshot["learners"]
	}


def freeze_noise_identities(config):
	for index, spec in enumerate(config["optimizers"]):
		spec["noise_slot"] = index
		spec["noise_stride"] = len(config["optimizers"])


class SimulationEngineTests(unittest.TestCase):
	def test_analytic_landscape_hessian_matches_autograd(self):
		cases = [
			({
				"kind": "sloped_ravine",
				"target_x": 0.4,
				"target_y": -0.2,
				"sharpness": 17.0,
				"valley_amp": 0.7,
				"valley_freq": 2.3,
				"valley_tilt": -0.35,
				"along_curvature": 0.11,
				"ripple_amp": 0.07,
				"ripple_freq": 4.2,
				"boundary": 0.09,
			}, [-1.1, 0.8]),
			({
				"kind": "curved_valley",
				"target_x": 0.4,
				"target_y": -0.2,
				"sharpness": 17.0,
				"valley_amp": 0.7,
				"valley_freq": 2.3,
				"valley_tilt": -0.35,
				"along_curvature": 0.11,
				"ripple_amp": 0.07,
				"ripple_freq": 4.2,
				"boundary": 0.09,
			}, [0.6, -1.3]),
		]
		for config_data, coordinates in cases:
			with self.subTest(kind=config_data["kind"]):
				config = coerce_config(config_data)
				point = torch.tensor(coordinates, dtype=torch.float64)
				expected = torch.autograd.functional.hessian(
					lambda theta: landscape_loss(theta, config), point)
				actual = landscape_hessian(point, config)

				torch.testing.assert_close(
					actual, expected, rtol=1e-12, atol=1e-12)

	def test_hessian_noise_transform_has_requested_covariance(self):
		config = NoiseConfig.from_dict({
			"mode": "hessian",
			"hessian_scale": 0.2,
			"hessian_power": 1.5,
		})
		hessian = torch.tensor([
			[1.0, 3.0],
			[3.0, 1.0],
		], dtype=torch.float64)
		basis = torch.eye(2, dtype=torch.float64)
		factor = torch.stack([
			sample_noise(
				torch.zeros(2, dtype=torch.float64),
				config,
				basis[:, index],
				hessian=hessian,
			)
			for index in range(2)
		], dim=1)
		eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
		expected_covariance = config.hessian_scale * (
			eigenvectors
			@ torch.diag(eigenvalues.abs().pow(config.hessian_power))
			@ eigenvectors.T
		)

		torch.testing.assert_close(
			factor @ factor.T, expected_covariance)

	def test_trace_loss_matches_post_step_theta_and_gradient_is_transition_gradient(self):
		state = SimulationState()
		state.reset({
			"max_steps": 1,
			"theta0": [-1.8, 2.3],
			"noise": {"enabled": False},
			"optimizers": [{
				"name": "AdamW",
				"lr": 0.03,
				"beta1": 0.9,
				"beta2": 0.999,
				"weight_decay": 0.1,
			}],
		})
		learner = state.learners[0]
		pre_step_loss = landscape_loss(learner.theta, state.landscape)
		pre_step_grad = torch.autograd.grad(pre_step_loss, learner.theta)[0]

		snapshot = state.step()
		sample = snapshot["learners"][0]["trace"][-1]
		post_step_theta = torch.tensor(sample["theta"], dtype=torch.float64)
		expected_post_loss = float(
			landscape_loss(post_step_theta, state.landscape))

		self.assertAlmostEqual(sample["loss"], expected_post_loss, places=12)
		self.assertNotAlmostEqual(
			sample["loss"], float(pre_step_loss.detach()), places=10)
		for actual, expected in zip(sample["grad"], pre_step_grad.tolist()):
			self.assertAlmostEqual(actual, expected, places=12)

	def test_appending_optimizer_does_not_perturb_existing_parallel_noise(self):
		config = {
			"mode": "parallel",
			"max_steps": 8,
			"theta0": [-2.6, -1.5],
			"landscape": {
				"kind": "curved_valley",
				"x_min": -4,
				"x_max": 1.1,
				"y_min": -1.8,
				"y_max": 2.3,
				"sharpness": 50,
				"valley_amp": 0.65,
				"valley_freq": 2.3,
				"valley_tilt": -0.18,
				"along_curvature": 0.09,
				"ripple_amp": 0.055,
				"ripple_freq": 7.5,
				"boundary": 0.2,
			},
			"noise": {
				"mode": "yu",
				"seed": 0,
				"xi_distribution": "symmetric_pareto",
				"sigma0": 0.05,
				"sigma1": 1,
				"tail_p": 1.1,
				"tail_margin": 0.1,
				"batch_size": 1,
			},
			"optimizers": [
				{
					"name": "MassiveLion",
					"lr": 0.03,
					"beta1": 0.9,
					"beta2": 0.99,
					"mass": 1,
				},
				{
					"name": "Lion",
					"lr": 0.03,
					"beta1": 0.9,
					"beta2": 0.99,
				},
				{
					"name": "AdamW",
					"lr": 0.03,
					"beta1": 0.99,
					"beta2": 0.999,
				},
			],
		}
		freeze_noise_identities(config)
		with_extra = copy.deepcopy(config)
		with_extra["optimizers"].append({
			"name": "RLion",
			"lr": 0.03,
			"beta1": 0.9,
			"beta2": 0.99,
			"mass": 0.2,
			"noise_slot": len(with_extra["optimizers"]),
			"noise_stride": len(with_extra["optimizers"]) + 1,
		})

		baseline = run_trace(config)
		extended = run_trace(with_extra)

		for name, trace in baseline.items():
			self.assertEqual(trace, extended[name])


if __name__ == "__main__":
	unittest.main()


# Public-repository wiring: every UI choice and saved preset must use a real,
# current optimizer and preserve the configuration through serialization.
import json
from dataclasses import asdict
from pathlib import Path
import pytest
from dynamics_lab.engine import OPTIMIZER_DEFAULTS, OptimizerSpec, build_optimizer
from massive_lion import MassiveLion, MassiveSignum


@pytest.mark.parametrize("name", list(OPTIMIZER_DEFAULTS))
def test_every_optimizer_runs(name):
    state = SimulationState()
    state.reset({"max_steps": 3, "optimizers": [{"name": name}]})
    for _ in range(3):
        snapshot = state.step()
    learner = snapshot["learners"][0]
    assert not learner["diverged"]
    assert learner["local_step"] == 3


@pytest.mark.parametrize("mode", ["coordinate", "vector", "vector_rms"])
@pytest.mark.parametrize("kinematics", ["minkowski", "arctan", "tanh"])
@pytest.mark.parametrize("mass_mode", ["momentum_diff", "gradient_diff"])
def test_advanced_config_roundtrip_and_backend(mode, kinematics, mass_mode):
    data = dict(name="MassiveLion", lr=.04, beta1=.8, beta2=.95, mass=.3,
                adaptive_mass=True, tie_mass=False, kappa=.2, beta_gravity=.7,
                update_mode=mode, kinematics=kinematics, mass_mode=mass_mode, foreach=False)
    spec = OptimizerSpec.from_dict(data)
    assert OptimizerSpec.from_dict(json.loads(json.dumps(asdict(spec)))) == spec
    param = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    optimizer = build_optimizer(spec, param)
    assert isinstance(optimizer, MassiveLion)
    group = optimizer.param_groups[0]
    for key in ["update_mode", "kinematics", "mass_mode", "adaptive_mass", "kappa", "beta_gravity", "foreach"]:
        assert group[key] == data[key]
    param.grad = torch.tensor([.4, -.2], dtype=torch.float64)
    optimizer.step()
    assert torch.isfinite(param).all()


@pytest.mark.parametrize("name", ["Lion", "Signum", "MassiveSignum", "SecretSauceAdamW", "VectorMassiveLion"])
def test_named_reductions_use_one_optimizer(name):
    spec = OptimizerSpec.from_dict({"name": name})
    assert OptimizerSpec.from_dict(asdict(spec)) == spec
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = build_optimizer(spec, param)
    assert isinstance(optimizer, MassiveLion)
    if name == "MassiveSignum":
        assert isinstance(optimizer, MassiveSignum)
    if name in {"Signum", "MassiveSignum", "SecretSauceAdamW"}:
        assert spec.beta1 == spec.beta2
    if name == "SecretSauceAdamW":
        assert spec.mass == 0 and spec.kappa == spec.beta_gravity == spec.beta2


@pytest.mark.parametrize("changes", [dict(beta1=.8, beta2=.9), dict(mass=.1), dict(eps=1e-8),
    dict(bias_correction=True), dict(update_mode="vector"), dict(kinematics="tanh"), dict(kappa=.1)])
def test_secret_sauce_rejects_incompatible_settings(changes):
    with pytest.raises(ValueError, match="SecretSauceAdamW"):
        OptimizerSpec.from_dict({"name": "SecretSauceAdamW", **changes})


def test_all_saved_presets_preserve_fixed_mass_and_run():
    paths = sorted((Path(__file__).parents[1] / "dynamics_lab" / "presets").glob("*.json"))
    assert len(paths) == 17
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 2
        config = payload["config"]
        for row in config["optimizers"]:
            if row["name"] in {"MassiveLion", "MassiveSignum", "VectorMassiveLion"}:
                assert row["adaptive_mass"] is False
        config.update(max_steps=3, ensemble_count=2)
        state = SimulationState()
        snapshot = state.reset(config)
        while not snapshot["done"]:
            snapshot = state.step()
        assert all(not learner["diverged"] for learner in snapshot["learners"]), path.name


@pytest.mark.parametrize("mode", ["parallel", "serial", "ensemble"])
def test_simulation_modes_complete_reproducibly(mode):
    config = {"mode": mode, "max_steps": 5, "ensemble_count": 3,
              "ensemble_optimizer": "MassiveLion", "noise": {"seed": 17, "std": .01},
              "optimizers": [{"name": "MassiveLion"}, {"name": "Lion"}]}
    assert run_trace(config) == run_trace(config)


@pytest.mark.parametrize("mode", ["parallel", "serial", "ensemble"])
def test_failed_reset_preserves_active_run_and_rng(mode):
    config = {"mode": mode, "max_steps": 5, "ensemble_count": 3,
              "ensemble_optimizer": "MassiveLion", "noise": {"seed": 17, "std": .01},
              "optimizers": [{"name": "MassiveLion"}, {"name": "AdamW"}]}
    active, expected = SimulationState(), SimulationState()
    for state in (active, expected):
        state.reset(config)
        state.advance()
    previous = active.snapshot(include_landscape=True)
    rng_before = torch.random.get_rng_state().clone()
    invalid = copy.deepcopy(config)
    invalid.update(mode="parallel", max_steps=2, noise={"seed": 97, "std": .02})
    invalid["optimizers"][1]["lr"] = -1

    with pytest.raises(ValueError, match="learning rate"):
        active.reset(invalid)

    assert active.snapshot(include_landscape=True) == previous
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    while not expected.done:
        assert active.step() == expected.step()


@pytest.mark.parametrize("theta", [[float("nan"), 0.], [float("inf"), 0.], [1e100, 0.]])
def test_nonfinite_initial_trace_is_rejected_without_losing_active_run(theta):
    state = SimulationState()
    config = {"max_steps": 2, "landscape": {"kind": "curved_valley"},
              "optimizers": [{"name": "AdamW"}]}
    state.reset(config)
    previous = state.snapshot(include_landscape=True)

    with pytest.raises(ValueError, match="finite"):
        state.reset({**config, "theta0": theta})

    assert state.snapshot(include_landscape=True) == previous
    json.dumps(state.step(), allow_nan=False)


@pytest.mark.parametrize("landscape", [
    {"kind": "curved_valley", "x_max": 1e100},
    {"target_x": float("nan")},
    {"y_max": float("inf")},
])
def test_sample_landscape_rejects_nonfinite_grid(landscape):
    with pytest.raises(ValueError, match="finite"):
        sample_landscape(landscape, n=7)


@pytest.mark.parametrize("invalid", [
    None, [], "not a config",
    {"theta0": [0]}, {"theta0": [0, 0, 0]}, {"theta0": None},
    {"theta0": "12"}, {"theta0": {"x": 0, "y": 0}},
    {"optimizers": None}, {"optimizers": {"name": "Lion"}},
    {"optimizers": [None]}, {"optimizers": ["Lion"]},
    {"landscape": []}, {"noise": "gaussian"},
    {"landscape": {"x_min": 1, "x_max": 1}},
    {"landscape": {"y_min": 1, "y_max": -1}},
    {"mode": "unknown"}, {"mode": None},
    {"max_steps": -1}, {"max_steps": 1.5}, {"max_steps": True}, {"max_steps": None},
])
def test_malformed_reset_raises_value_error_and_preserves_active_run(invalid):
    state = SimulationState()
    state.reset({"max_steps": 2, "optimizers": [{"name": "Lion"}]})
    state.advance()
    previous = state.snapshot(include_landscape=True)
    with pytest.raises(ValueError):
        state.reset(invalid)
    assert state.snapshot(include_landscape=True) == previous
    assert state.step()["done"]


@pytest.mark.parametrize("mode", ["parallel", "serial", "ensemble"])
def test_zero_steps_is_an_immediately_complete_run(mode):
    state = SimulationState()
    initial = state.reset({"mode": mode, "max_steps": 0, "ensemble_count": 3,
                           "noise": {"std": .01, "seed": 17},
                           "optimizers": [{"name": "Lion"}, {"name": "AdamW"}]})
    assert initial["done"] and initial["global_step"] == initial["total_steps"] == 0
    assert len(initial["learners"]) == (3 if mode == "ensemble" else 2)
    assert all(row["local_step"] == 0 and len(row["trace"]) == 1 for row in initial["learners"])
    state.advance()
    assert state.snapshot(include_landscape=True) == initial


def test_landscape_grid_requires_two_points():
    from dynamics_lab.landscapes import sample_landscape
    with pytest.raises(ValueError, match="at least two"):
        sample_landscape({}, n=1)


@pytest.mark.parametrize("steps", ["2", 2.0])
def test_reset_preserves_integral_coercion_and_optional_defaults(steps):
    state = SimulationState()
    snapshot = state.reset({"max_steps": steps, "theta0": ("-1.8", 2.3),
                            "landscape": None, "noise": None,
                            "optimizers": ({"name": "Lion"},)})
    assert snapshot["max_steps"] == 2
    assert snapshot["theta0"] == [-1.8, 2.3]
    state.advance()
    assert state.step()["done"]


def test_cautious_masks_match_c_optim_conventions():
    from massive_lion.baselines import CautiousLion, CautiousAdamW
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = CautiousLion([p], lr=.1, betas=(.5,.9))
    p.grad = torch.tensor([1., 0.], dtype=torch.float64)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([.9, 1.], dtype=torch.float64))
    q = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = CautiousAdamW([q], lr=.1, betas=(.5,.5), eps=0)
    q.grad = torch.tensor([1., 2.], dtype=torch.float64)
    opt.step()
    torch.testing.assert_close(q, torch.full((2,), .9, dtype=torch.float64))


def test_legacy_vector_geometry_is_not_silently_changed():
    with pytest.raises(ValueError, match="adaptive_geometry"):
        OptimizerSpec.from_dict({"name": "VectorMassiveLion", "adaptive_geometry": True})


@pytest.mark.parametrize("mass", [0., .3])
def test_rlion_wrapper_preserves_native_mass_response(mass):
    from massive_lion.baselines import ArctanLion
    import math
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = ArctanLion([{"params": [p], "mass": mass}], lr=.1, betas=(.8, .9))
    assert isinstance(opt, MassiveLion)
    momentum = torch.zeros_like(p)
    expected = p.detach().clone()
    for values in ([1., -2.], [-3., 1.], [.2, .4]):
        gradient = torch.tensor(values, dtype=torch.float64)
        direction = .8 * momentum + .2 * gradient
        update = direction.sign() if mass == 0 else (2 / math.pi) * torch.atan(direction / mass)
        expected -= .1 * update
        momentum = .9 * momentum + .1 * gradient
        p.grad = gradient
        opt.step()
        torch.testing.assert_close(p, expected, rtol=1e-13, atol=1e-13)
