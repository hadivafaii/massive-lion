"""Named reductions with constraints checked for every parameter group."""

from .optimizer import MassiveLion


class _PaperAlias(MassiveLion):
    _alias = ""

    def __init__(self, params, **kwargs):
        if self._alias != "lion":
            kwargs.setdefault("betas", (0.99, 0.99))
        kwargs.setdefault("mass", 0.01 if self._alias == "m_signum" else 0.0)
        kwargs.setdefault("adaptive_mass", self._alias == "ss_adamw")
        super().__init__(params, **kwargs)

    def _validate_group(self, group):
        super()._validate_group(group)
        requirements = {"update_mode": "coordinate"}
        if self._alias in ("lion", "signum"):
            requirements.update(mass=0.0, adaptive_mass=False)
        elif self._alias == "ss_adamw":
            requirements.update(mass=0.0, adaptive_mass=True, kinematics="minkowski")
        if self._alias in ("m_signum", "ss_adamw"):
            requirements["mass_mode"] = "momentum_diff"
        for key, expected in requirements.items():
            if group[key] != expected:
                raise ValueError(f"{self._alias} requires {key}={expected!r}")
        beta_mix, beta_momentum = group["betas"]
        if self._alias != "lion" and beta_mix != beta_momentum:
            raise ValueError(f"{self._alias} requires equal betas")
        if group["adaptive_mass"]:
            for key in ("kappa", "beta_gravity"):
                if group[key] not in (None, beta_momentum):
                    raise ValueError(f"{self._alias} requires {key}=beta_momentum (or None)")


class Lion(_PaperAlias):
    """Massless, fixed-mass Massive Lion: the Lion update."""
    _alias = "lion"


class Signum(_PaperAlias):
    """Lion with equal direction and momentum coefficients."""
    _alias = "signum"


class MassiveSignum(_PaperAlias):
    """Equal-beta Massive Lion; fixed mass by default, adaptation optional."""
    _alias = "m_signum"


class SecretSauceAdamW(_PaperAlias):
    """Equal-beta AdamW, without bias correction and with epsilon zero."""
    _alias = "ss_adamw"


def create_optimizer(name, params, **kwargs):
    """Construct Massive Lion or one of its checked paper reductions."""
    constructors = {
        "m_lion": MassiveLion, "massive_lion": MassiveLion,
        "lion": Lion, "signum": Signum,
        "m_signum": MassiveSignum, "massive_signum": MassiveSignum,
        "ss_adamw": SecretSauceAdamW, "secret_sauce_adamw": SecretSauceAdamW,
    }
    key = name.lower()
    if key not in constructors:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {tuple(constructors)}")
    return constructors[key](params, **kwargs)
