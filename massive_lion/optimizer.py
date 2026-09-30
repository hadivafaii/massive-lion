"""Readable PyTorch implementation of Massive Lion.

The scalar update below follows the equations in ``docs/optimizer.md``.
The equivalent batched implementation lives in ``_foreach.py``.
"""

import math

import torch


UPDATE_MODES = ("coordinate", "vector", "vector_rms")
MASS_MODES = ("momentum_diff", "gradient_diff")
KINEMATICS = ("minkowski", "arctan", "tanh")


def _mass_coefficients(group):
    if not group["adaptive_mass"]:
        return 0.0, 0.0
    beta_momentum = group["betas"][1]
    coupling = beta_momentum if group["kappa"] is None else group["kappa"]
    decay = beta_momentum if group["beta_gravity"] is None else group["beta_gravity"]
    return coupling, decay


def _squared_magnitude(tensor, update_mode):
    if update_mode == "coordinate":
        return tensor.square()
    energy = tensor.norm().square()
    return energy / max(1, tensor.numel()) if update_mode == "vector_rms" else energy


def _response(direction, mass_squared, update_mode, kinematics):
    """Bound the direction coordinatewise, or by its tensor L2/RMS norm."""
    magnitude = direction if update_mode == "coordinate" else direction.norm()
    if update_mode == "vector_rms":
        magnitude = magnitude / math.sqrt(max(1, direction.numel()))
    if kinematics == "minkowski":
        denominator = (mass_squared + magnitude.square()).sqrt()
        return direction / denominator.masked_fill(denominator == 0, 1.0)

    mass = mass_squared.sqrt()
    if kinematics == "arctan":
        # atan2 includes the zero-mass sign limit and defines (0, 0) as zero.
        response = torch.atan2(magnitude, mass * (2 / math.pi)) * (2 / math.pi)
    else:  # tanh
        response = torch.tanh(magnitude / mass.masked_fill(mass == 0, 1.0))
        response = torch.where(mass == 0, magnitude.sign(), response)
    if update_mode == "coordinate":
        return response
    update = direction / magnitude.masked_fill(magnitude == 0, 1.0) * response
    # A floating-point norm may underflow while the linear response is nonzero.
    linear_response = direction / mass.masked_fill(mass == 0, 1.0)
    return torch.where((magnitude == 0) & (mass > 0), linear_response, update)


class MassiveLion(torch.optim.Optimizer):
    """Massive Lion with optional adaptive mass and decoupled weight decay.

    Args:
        params: Parameters, or dictionaries defining parameter groups.
        lr: Learning rate (nonnegative).
        betas: ``(beta_mix, beta_momentum)`` in [0, 1). Both the direction
            and the innovation use the momentum from the previous step.
        weight_decay: Decoupled weight decay (nonnegative).
        mass: Fixed saturation scale rho_0 (nonnegative), in gradient units.
        adaptive_mass: Enable mass adaptation. False fixes mass at rho_0.
        foreach: Use batched tensor operations. False selects the readable
            per-parameter update. Both implement the same algorithm.
        maximize: Ascend the objective; weight decay still shrinks parameters.
        update_mode: ``coordinate``, ``vector`` (per-tensor L2), or
            ``vector_rms`` (per-tensor RMS).
        mass_mode: ``momentum_diff`` for the paper's variance recurrence;
            ``gradient_diff`` for the optional gradient-change extension.
        kinematics: ``minkowski`` (paper), ``arctan``, or ``tanh``.
        kappa: Adaptive-mass coupling. None ties it to beta_momentum in each
            group; an explicit nonnegative value enables an extended rule.
        beta_gravity: Adaptive-mass decay in [0, 1). None ties it to each
            group's beta_momentum. With adaptation off, both controls must
            be None or zero.

    No bias correction or epsilon is used. A zero direction and denominator
    produce a zero update. FP16/BF16 parameters use FP32 state/arithmetic.
    Sparse gradients and complex parameters are unsupported. See
    ``docs/optimizer.md`` for equations, reductions, and all options.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 mass=0.01, adaptive_mass=True, foreach=True, maximize=False,
                 update_mode="coordinate", mass_mode="momentum_diff",
                 kinematics="minkowski", kappa=None, beta_gravity=None):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, mass=mass,
                        adaptive_mass=adaptive_mass, foreach=foreach,
                        maximize=maximize, update_mode=update_mode,
                        mass_mode=mass_mode, kinematics=kinematics,
                        kappa=kappa, beta_gravity=beta_gravity)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    def _validate_group(self, group):
        for name in ("lr", "weight_decay", "mass"):
            if not math.isfinite(group[name]) or group[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(group["betas"]) != 2 or not all(0 <= b < 1 for b in group["betas"]):
            raise ValueError("betas must contain two values in [0, 1)")
        for name in ("adaptive_mass", "foreach", "maximize"):
            if not isinstance(group[name], bool):
                raise ValueError(f"{name} must be a bool")
        for name, choices in (("update_mode", UPDATE_MODES),
                              ("mass_mode", MASS_MODES), ("kinematics", KINEMATICS)):
            if group[name] not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        kappa, decay = group["kappa"], group["beta_gravity"]
        if kappa is not None and (not math.isfinite(kappa) or kappa < 0):
            raise ValueError("kappa must be None or finite and nonnegative")
        if decay is not None and not 0 <= decay < 1:
            raise ValueError("beta_gravity must be None or in [0, 1)")
        if not group["adaptive_mass"] and (kappa not in (None, 0) or decay not in (None, 0)):
            raise ValueError("adaptive_mass=False requires kappa and beta_gravity to be None or zero")

    def add_param_group(self, param_group):
        # Validate overrides as well as constructor defaults, including groups
        # added later (for example, when unfreezing a layer).
        self._validate_group({**self.defaults, **param_group})
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            self._validate_group(group)
        super().load_state_dict(state_dict)
        # PyTorch casts floating state to the parameter dtype. Recover saved
        # FP32 buffers directly so low-precision checkpoint resume is lossless.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, param in zip(saved_group["params"], group["params"]):
                saved = state_dict["state"].get(saved_id, {})
                if param.dtype in (torch.float16, torch.bfloat16):
                    for key in ("exp_avg", "metric_diag", "prev_grad"):
                        if key in saved:
                            self.state[param][key] = saved[key].to(
                                device=param.device, dtype=torch.float32).clone()

    def _prepare_group(self, group):
        params, grads, momenta, masses, previous_grads, first_steps = [], [], [], [], [], []
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse or param.is_complex():
                raise RuntimeError("MassiveLion supports only dense real gradients and parameters")
            state = self.state[param]
            if not state:
                dtype = torch.float32 if param.dtype in (torch.float16, torch.bfloat16) else param.dtype
                state["exp_avg"] = torch.zeros_like(param, dtype=dtype)
                shape = param.shape if group["update_mode"] == "coordinate" else ()
                state["metric_diag"] = torch.full(shape, group["mass"] ** 2,
                                                   dtype=dtype, device=param.device)
            if group["mass_mode"] == "gradient_diff":
                if "prev_grad" not in state:
                    state["prev_grad"] = torch.zeros_like(state["exp_avg"])
                    state["step"] = 0
                state["step"] += 1
                previous_grads.append(state["prev_grad"])
                first_steps.append(state["step"] == 1)
            params.append(param)
            grads.append(param.grad.to(dtype=state["exp_avg"].dtype))
            momenta.append(state["exp_avg"])
            masses.append(state["metric_diag"])
        return params, grads, momenta, masses, previous_grads, first_steps

    @staticmethod
    def _step_single(params, grads, momenta, masses, previous_grads, first_steps, group):
        """The reference update: innovation, mass, direction, position, momentum."""
        beta_mix, beta_momentum = group["betas"]
        coupling, decay = _mass_coefficients(group)
        step_size = group["lr"] if group["maximize"] else -group["lr"]
        for index, (param, grad, momentum, mass_squared) in enumerate(zip(params, grads, momenta, masses)):
            if group["weight_decay"]:
                # Match foreach's scalar conversion for FP16/BF16 parameters.
                decay_factor = param.new_tensor(1 - group["lr"] * group["weight_decay"])
                param.mul_(decay_factor)

            # Form both innovation and direction before modifying the momentum.
            if group["mass_mode"] == "momentum_diff":
                innovation = grad - momentum
            else:
                innovation = grad - previous_grads[index]
                if first_steps[index]:
                    innovation.zero_()
            target_mass = _squared_magnitude(innovation, group["update_mode"])
            target_mass.mul_(coupling).add_(group["mass"] ** 2)
            mass_squared.lerp_(target_mass, 1 - decay)
            direction = torch.lerp(grad, momentum, beta_mix)

            if group["update_mode"] == "coordinate" and group["mass"] == 0 and coupling == 0:
                param.add_(direction.sign(), alpha=step_size)
            elif group["kinematics"] == "minkowski":
                if group["update_mode"] == "coordinate":
                    denominator = torch.addcmul(mass_squared, direction, direction).sqrt_()
                else:
                    denominator = (mass_squared + _squared_magnitude(direction, group["update_mode"])).sqrt_()
                denominator.masked_fill_(denominator == 0, 1.0)
                param.addcdiv_(direction, denominator, value=step_size)
            else:
                velocity = _response(direction, mass_squared, group["update_mode"], group["kinematics"])
                param.add_(velocity, alpha=step_size)

            momentum.lerp_(grad, 1 - beta_momentum)
            if group["mass_mode"] == "gradient_diff":
                previous_grads[index].copy_(grad)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            tensors = self._prepare_group(group)
            if not tensors[0]:
                continue
            if group["foreach"]:
                from ._foreach import step_foreach
                step_foreach(*tensors, group)
            else:
                self._step_single(*tensors, group)
        return loss
