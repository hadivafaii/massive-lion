"""RLion's native mass convention expressed through the shared optimizer."""

import math
from massive_lion.optimizer import MassiveLion


class ArctanLion(MassiveLion):
    """Fixed-mass coordinate arctan response, without bias correction.

    RLion uses ``(2/pi) * atan(direction / mass)``. MassiveLion's arctan
    response has unit slope at the origin in reference-mass units, so this
    wrapper converts native RLion mass to ``mass * pi/2``. Optimizer parameter
    groups and checkpoints store the converted reference mass.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0.0, mass=0.0, foreach=True):
        parameters = list(params)
        if parameters and isinstance(parameters[0], dict):
            parameters = [dict(group, mass=group.get("mass", mass) * math.pi / 2)
                          for group in parameters]
        super().__init__(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                         mass=mass * math.pi / 2, adaptive_mass=False,
                         update_mode="coordinate", kinematics="arctan", foreach=foreach)

    def _validate_group(self, group):
        super()._validate_group(group)
        if group["adaptive_mass"] or group["update_mode"] != "coordinate" or group["kinematics"] != "arctan":
            raise ValueError("RLion requires fixed-mass coordinate arctan kinematics")
