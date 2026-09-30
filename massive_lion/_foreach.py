"""Batched equivalent of MassiveLion._step_single, without diagnostics."""

import torch

from .optimizer import _mass_coefficients, _response


def _squared_magnitudes(tensors, update_mode):
    if update_mode == "coordinate":
        return torch._foreach_mul(tensors, tensors)
    norms = torch._foreach_norm(tensors, 2)
    energies = torch._foreach_mul(norms, norms)
    if update_mode == "vector_rms":
        torch._foreach_div_(energies, [max(1, tensor.numel()) for tensor in tensors])
    return energies


def step_foreach(params, grads, momenta, masses, previous_grads, first_steps, group):
    beta_mix, beta_momentum = group["betas"]
    coupling, decay = _mass_coefficients(group)
    step_size = group["lr"] if group["maximize"] else -group["lr"]

    if group["weight_decay"]:
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])

    if group["mass_mode"] == "momentum_diff":
        innovations = torch._foreach_sub(grads, momenta)
    else:
        innovations = torch._foreach_sub(grads, previous_grads)
        initial = [value for value, first in zip(innovations, first_steps) if first]
        if initial:
            torch._foreach_zero_(initial)
    target_masses = _squared_magnitudes(innovations, group["update_mode"])
    torch._foreach_mul_(target_masses, coupling)
    torch._foreach_add_(target_masses, group["mass"] ** 2)
    torch._foreach_lerp_(masses, target_masses, 1 - decay)
    directions = torch._foreach_lerp(grads, momenta, beta_mix)

    if group["update_mode"] == "coordinate" and group["mass"] == 0 and coupling == 0:
        torch._foreach_sign_(directions)
        torch._foreach_add_(params, directions, alpha=step_size)
    elif group["kinematics"] == "minkowski":
        if group["update_mode"] == "coordinate":
            denominators = torch._foreach_addcmul(masses, directions, directions)
        else:
            energies = _squared_magnitudes(directions, group["update_mode"])
            denominators = torch._foreach_add(masses, energies)
        torch._foreach_sqrt_(denominators)
        # Zero direction with zero/underflowed squared mass must give zero.
        torch._foreach_add_(denominators, [denominator == 0 for denominator in denominators])
        torch._foreach_addcdiv_(params, directions, denominators, value=step_size)
    else:
        velocities = [_response(direction, mass, group["update_mode"], group["kinematics"])
                      for direction, mass in zip(directions, masses)]
        torch._foreach_add_(params, velocities, alpha=step_size)

    torch._foreach_lerp_(momenta, grads, 1 - beta_momentum)
    if group["mass_mode"] == "gradient_diff":
        torch._foreach_copy_(previous_grads, grads)
