# Massive Lion

PyTorch code for **Sign-based optimizers are massless relativistic learners**.
This repository contains the optimizer, an interactive Dynamics Lab, and the
experiments for the ICLR submission.

```python
from massive_lion import MassiveLion

optimizer = MassiveLion(
    model.parameters(),
    lr=1e-3,
    betas=(0.7, 0.95),        # direction mixing, stored-momentum decay
    mass=1.0,                # fixed rest-mass floor
    adaptive_mass=True,
    weight_decay=0.1,
    foreach=True,            # batched operations; False uses the readable path
)
```

Massive Lion smoothly limits update magnitude. Its direction combines the
current gradient with stored momentum; its effective mass combines a fixed
floor with an optional estimate of gradient innovation. The paper setting uses
momentum-difference adaptation with its decay and coupling tied to `beta2`.
There is no bias correction or additive epsilon.

## Install

Use Python 3.10 or newer and install a PyTorch build suitable for your machine.
From a clone of this repository:

```bash
python -m pip install -e .
```

Only PyTorch is required for the optimizer. Install optional components as needed:

```bash
python -m pip install -e '.[lab]'                       # interactive simulator and videos
python -m pip install -e '.[lab,vision,language,analysis,test]'  # all components and tests
```

The Lab extra requires PyTorch 2.9 or newer for its Muon comparison.
`requirements-tested.txt` records the environment used to check this release;
it is not a claim that all listed versions match the original training machines.

## One implementation, named reductions

```python
from massive_lion import create_optimizer

signum = create_optimizer("m_signum", model.parameters(),
                          lr=1e-3, betas=(0.9, 0.9), mass=0.1)
ss_adamw = create_optimizer("ss_adamw", model.parameters(),
                            lr=8e-3, betas=(0.95, 0.95))
```

| Name | Enforced reduction |
| --- | --- |
| `m_lion` | Full Massive Lion implementation |
| `lion` | Zero rest mass, adaptation off, coordinate updates |
| `signum` | Lion with equal betas |
| `m_signum` | Equal betas and coordinate updates; fixed mass by default, adaptation optional |
| `ss_adamw` | Zero rest mass, equal betas, tied adaptive mass, coordinate Minkowski response |

These wrappers inherit the same optimizer step and reject incompatible settings,
including parameter-group overrides. `ss_adamw` means the paper's uncorrected,
equal-beta reduction, not standard default AdamW. General Massive Lion also
supports vector and vector-RMS updates, arctan/tanh responses, and the documented
advanced mass controls.

Start with [the equations and option guide](docs/optimizer.md), then read
[`optimizer.py`](massive_lion/optimizer.py) for the per-tensor step and
[`_foreach.py`](massive_lion/_foreach.py) for the batched implementation.
[`aliases.py`](massive_lion/aliases.py) defines the reductions.

## Dynamics Lab

```bash
python -m dynamics_lab.interactive --port 8011
```

The Lab opens in your default browser and prints its local URL. Add
`--no-browser` to skip the automatic launch. Compare trajectories on 2D/3D
landscapes, vary noise and initial conditions, inspect velocities and momentum distributions,
run ensembles, and import/export presets. The Lab runs the actual PyTorch
optimizers through a local server. [Lab guide](docs/dynamics_lab.md).

## Reproduce the paper

The [reproduction guide](docs/reproduction.md) covers CIFAR-10/ResNet-18,
the 160M-class SlimPajama language model, selection rules, and supplied grids.
The [quadratic guide](docs/quadratics.md) covers the stability examples and
heterogeneous 9D study. The release includes the current ICLR experiments;
later 411M and 1.06B scaling experiments are outside its scope.

Run small checks without downloading training data:

```bash
python -m pytest -q
python -m experiments.cifar --smoke --output runs/smoke-cifar
python -m experiments.language --smoke --output runs/smoke-language
python -m experiments.quadratics.two_dimensional --output outputs/quadratics/2d
python -m experiments.quadratics.heterogeneous configs/quadratics/random_confirm_plan.json --smoke
```

Smoke runs use generated data and do not reproduce reported neural-network
scores. Full experiments require the documented datasets and compute. Metrics
are saved locally; no experiment-tracking account is needed.
See [release verification](docs/verification.md) for numerical checks and limits.

| Directory | Contents |
| --- | --- |
| [massive_lion/](massive_lion/README.md) | Optimizer, aliases, and comparison baselines |
| [dynamics_lab/](dynamics_lab/README.md) | Browser UI, simulation engine, landscapes, noise, and video export |
| [experiments/](experiments/README.md) | Data preparation, models, trainers, selection, and plotting |
| [configs/](configs/README.md) | Reproduction grids and seed plans |
| [docs/](docs/README.md) | Equations, recipes, and verification limits |
| [tests/](tests/README.md) | Numerical and experiment-contract checks |

Generated datasets, runs, outputs, and checkpoints are ignored by Git. Dataset
preparation, experiment configurations, training, and analysis are included here.

MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md) for adapted code.
