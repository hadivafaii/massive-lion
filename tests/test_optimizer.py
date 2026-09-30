import copy
import itertools
import math

import pytest
import torch

from massive_lion import (
    MassiveLion, Lion, Signum, MassiveSignum, SecretSauceAdamW, create_optimizer,
)


MODES = ("coordinate", "vector", "vector_rms")
MAPS = ("minkowski", "arctan", "tanh")
MASS_MODES = ("momentum_diff", "gradient_diff")


@pytest.mark.parametrize("foreach", (False, True))
@pytest.mark.parametrize("adaptive", (False, True))
def test_paper_equations(foreach, adaptive):
    parameter = torch.nn.Parameter(torch.tensor([1., -2., 0.], dtype=torch.float64))
    expected = parameter.detach().clone()
    momentum = torch.zeros_like(parameter)
    variance = torch.zeros_like(parameter)
    beta_mix, beta_momentum, mass = 0.4, 0.8, 0.2
    optimizer = MassiveLion([parameter], lr=0.1, betas=(beta_mix, beta_momentum),
                           mass=mass, weight_decay=0.2, adaptive_mass=adaptive,
                           foreach=foreach)
    for values in ([3., -4., 0.], [-2., 1., 0.], [0., 0., 0.]):
        gradient = torch.tensor(values, dtype=parameter.dtype)
        if adaptive:
            variance = beta_momentum * variance + beta_momentum * (1 - beta_momentum) * (gradient - momentum).square()
        direction = beta_mix * momentum + (1 - beta_mix) * gradient
        expected = 0.98 * expected - 0.1 * direction / (mass ** 2 + variance + direction.square()).sqrt()
        momentum = beta_momentum * momentum + (1 - beta_momentum) * gradient
        parameter.grad = gradient
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], momentum)
        torch.testing.assert_close(optimizer.state[parameter]["metric_diag"], mass ** 2 + variance)


@pytest.mark.parametrize("mode,map_name,mass_mode", list(itertools.product(MODES, MAPS, MASS_MODES)))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.float16, torch.bfloat16))
def test_foreach_matches_scalar(mode, map_name, mass_mode, dtype):
    generator = torch.Generator().manual_seed(19)
    scalar_params = [torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype))
                     for shape in ((3, 4), (2,), ())]
    batch_params = [torch.nn.Parameter(p.detach().clone()) for p in scalar_params]
    kwargs = dict(lr=0.03, betas=(0.7, 0.8), mass=0.1, weight_decay=0.1,
                  kappa=0.3, beta_gravity=0.6, update_mode=mode,
                  kinematics=map_name, mass_mode=mass_mode)
    scalar = MassiveLion(scalar_params, foreach=False, **kwargs)
    batch = MassiveLion(batch_params, foreach=True, **kwargs)
    for step in range(5):
        for index, (left, right) in enumerate(zip(scalar_params, batch_params)):
            gradient = torch.randn(left.shape, generator=generator, dtype=dtype)
            left.grad = None if step == 2 and index == 1 else gradient
            right.grad = None if left.grad is None else gradient.clone()
        scalar.step()
        batch.step()
        for left, right in zip(scalar_params, batch_params):
            torch.testing.assert_close(left, right)
            for key, value in scalar.state[left].items():
                torch.testing.assert_close(value, batch.state[right][key])


@pytest.mark.parametrize("constructor", (Lion, Signum, SecretSauceAdamW))
@pytest.mark.parametrize("foreach", (False, True))
def test_named_massless_reductions(constructor, foreach):
    parameter = torch.nn.Parameter(torch.tensor([1., -2., 0.], dtype=torch.float64))
    expected = parameter.detach().clone()
    momentum, second_moment = torch.zeros_like(parameter), torch.zeros_like(parameter)
    betas = (0.4, 0.8) if constructor is Lion else (0.8, 0.8)
    optimizer = constructor([parameter], lr=0.1, betas=betas, weight_decay=0.2, foreach=foreach)
    for values in ([3., -4., 0.], [-2., 1., 0.], [0., 0., 0.]):
        gradient = torch.tensor(values, dtype=parameter.dtype)
        direction = betas[0] * momentum + (1 - betas[0]) * gradient
        momentum = betas[1] * momentum + (1 - betas[1]) * gradient
        second_moment = betas[1] * second_moment + (1 - betas[1]) * gradient.square()
        if constructor is SecretSauceAdamW:
            denominator = second_moment.sqrt()
            update = momentum / denominator.masked_fill(denominator == 0, 1.)
        else:
            update = direction.sign()
        expected = 0.98 * expected - 0.1 * update
        parameter.grad = gradient
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode,map_name", list(itertools.product(MODES, MAPS)))
@pytest.mark.parametrize("foreach", (False, True))
def test_radial_speed_limits_and_zero_directions(mode, map_name, foreach):
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = MassiveLion([parameter], lr=1, betas=(0, 0), mass=0,
                           adaptive_mass=False, update_mode=mode,
                           kinematics=map_name, foreach=foreach, maximize=True)
    parameter.grad = torch.tensor([3., 4., 0.], dtype=parameter.dtype)
    optimizer.step()
    expected = torch.tensor([1., 1., 0.], dtype=parameter.dtype)
    if mode != "coordinate":
        expected = parameter.grad / 5 * (math.sqrt(3) if mode == "vector_rms" else 1)
    torch.testing.assert_close(parameter, expected)
    before = parameter.detach().clone()
    parameter.grad.zero_()
    optimizer.step()
    torch.testing.assert_close(parameter, before)


@pytest.mark.parametrize("foreach", (False, True))
@pytest.mark.parametrize("mass_mode", MASS_MODES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_resume_preserves_state_precision(foreach, mass_mode, dtype):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    optimizer = MassiveLion([parameter], mass=1e-5, foreach=foreach, mass_mode=mass_mode)
    parameter.grad = torch.tensor([0., 300., -400.], dtype=dtype)
    optimizer.step()
    restored = torch.nn.Parameter(parameter.detach().clone())
    other = MassiveLion([restored])
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for key, value in optimizer.state[parameter].items():
        torch.testing.assert_close(value, other.state[restored][key], rtol=0, atol=0)
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float32
    for values in ([0., -200., 300.], [0., 0., 0.]):
        parameter.grad = torch.tensor(values, dtype=dtype)
        restored.grad = parameter.grad.clone()
        optimizer.step()
        other.step()
        torch.testing.assert_close(parameter, restored, rtol=0, atol=0)


@pytest.mark.parametrize("foreach", (False, True))
def test_none_gradient_skips_decay_and_state(foreach):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = MassiveLion([parameter], weight_decay=0.5, foreach=foreach)
    optimizer.step()
    assert parameter not in optimizer.state
    torch.testing.assert_close(parameter, torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = parameter.detach().clone()
    state = copy.deepcopy(optimizer.state[parameter])
    parameter.grad = None
    optimizer.step()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(state[key], optimizer.state[parameter][key], rtol=0, atol=0)


@pytest.mark.parametrize("map_name", MAPS)
@pytest.mark.parametrize("foreach", (False, True))
def test_underflowed_mass_with_zero_gradient_is_finite(map_name, foreach):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = MassiveLion([parameter], mass=1e-24, kinematics=map_name, foreach=foreach)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)


@pytest.mark.parametrize("foreach", (False, True))
def test_groups_and_mixed_dtypes(foreach):
    parameters = [torch.nn.Parameter(torch.ones(2, dtype=dtype))
                  for dtype in (torch.float32, torch.float64, torch.float16)]
    optimizer = MassiveLion([{"params": parameters[:2], "betas": (0.4, 0.6)},
                             {"params": parameters[2:], "betas": (0.2, 0.8)}], foreach=foreach)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter, beta in zip(parameters, (0.6, 0.6, 0.8)):
        expected = torch.full_like(optimizer.state[parameter]["metric_diag"], 0.01 ** 2 + beta * (1 - beta))
        torch.testing.assert_close(optimizer.state[parameter]["metric_diag"], expected)


@pytest.mark.parametrize("constructor,conflict", [
    (Lion, {"mass": 0.1}), (Lion, {"adaptive_mass": True}),
    (Signum, {"betas": (0.8, 0.9)}),
    (MassiveSignum, {"update_mode": "vector"}),
    (MassiveSignum, {"mass_mode": "gradient_diff"}),
    (SecretSauceAdamW, {"adaptive_mass": False}),
    (SecretSauceAdamW, {"mass": 0.1}),
    (SecretSauceAdamW, {"beta_gravity": 0.8}),
    (SecretSauceAdamW, {"kappa": 0.8}),
    (SecretSauceAdamW, {"kinematics": "tanh"}),
])
def test_alias_constraints_cover_arguments_groups_and_added_groups(constructor, conflict):
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError):
        constructor([parameter], **conflict)
    with pytest.raises(ValueError):
        constructor([{"params": [parameter], **conflict}])
    optimizer = constructor([parameter])
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))], **conflict})


def test_alias_constraints_cover_later_mutation_and_checkpoint_load():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = SecretSauceAdamW([parameter])
    bad_checkpoint = copy.deepcopy(optimizer.state_dict())
    bad_checkpoint["param_groups"][0]["mass"] = 0.5
    with pytest.raises(ValueError, match="requires mass"):
        optimizer.load_state_dict(bad_checkpoint)
    optimizer.param_groups[0]["betas"] = (0.5, 0.9)
    with pytest.raises(ValueError, match="equal betas"):
        optimizer.step()


def test_adaptive_massive_signum_and_group_specific_ties():
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    optimizer = MassiveSignum([{"params": [parameters[0]], "betas": (0.8, 0.8)},
                               {"params": [parameters[1]], "betas": (0.9, 0.9)}], adaptive_mass=True)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter, beta in zip(parameters, (0.8, 0.9)):
        expected = torch.full_like(parameter, 0.01 ** 2 + beta * (1 - beta))
        torch.testing.assert_close(optimizer.state[parameter]["metric_diag"], expected)


@pytest.mark.parametrize("kwargs", [
    {"lr": -1}, {"weight_decay": -1}, {"mass": float("nan")},
    {"betas": (0.9, 1)}, {"kappa": -1}, {"beta_gravity": 1},
    {"adaptive_mass": False, "kappa": 0.9}, {"update_mode": "bad"},
    {"mass_mode": "bad"}, {"kinematics": "bad"}, {"foreach": "yes"},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        MassiveLion([torch.nn.Parameter(torch.ones(2))], **kwargs)


def test_sparse_and_complex_are_rejected():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = MassiveLion([parameter])
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.]), (2,), check_invariants=True)
    with pytest.raises(RuntimeError, match="dense real"):
        optimizer.step()
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.complex64))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="dense real"):
        MassiveLion([parameter]).step()


def test_closure_and_factory():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = create_optimizer("m_lion", [parameter])
    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss
    assert optimizer.step(closure).item() == 2
    assert isinstance(create_optimizer("ss_adamw", [parameter]), MassiveLion)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer("obsolete_optimizer", [parameter])
