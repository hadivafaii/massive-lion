# Dynamics Lab

An interactive sandbox for watching optimizers move across a loss landscape.
Compare trajectories in 2D or 3D, run ensembles from random initial positions,
and explore how curvature, momentum, rest mass, and gradient noise change them.
The Python backend uses the same optimizer classes as model training.

## Try it

Run these commands from the repository root:

```bash
python -m pip install -e ".[lab,test]"
python -m dynamics_lab.interactive --port 8011
```

The Lab opens in your default browser at [the local URL](http://127.0.0.1:8011),
which is also printed in the terminal. Add `--no-browser` to skip opening the
browser automatically. Use Run, Pause, Step, and Reset to explore.
Use Import to load a JSON file from [presets/](presets/README.md); Export saves
settings for another session or video rendering. Stop the server with Ctrl-C.
The HTML page needs the Python server, so opening it directly is insufficient.

Add multiple instances of the same optimizer to compare settings. Each row
has its own color and optional custom name; automatic titles highlight differing
settings. Export/import preserves these instances, and ensemble mode selects
one specific row. Video export keeps their trajectories and colors separate.

`shared learning rate`, `shared mass/rho`, and `shared betas` are independent
switches. Turn one off to disable and dim its global inputs and edit the
corresponding row values. Mass sharing affects rest mass only; kappa and
mass-memory decay keep their own adaptation and tying controls.

New MassiveLion rows enable the paper's momentum-difference adaptive mass,
with mass parameters tied to `beta_mom`. Untie them to edit the mass decay and
kappa independently. The saved demonstration presets explicitly use fixed mass
to preserve their original trajectories. Standard Adam/AdamW label their moment
decays `beta_mom` and `beta_var`; mixing variants use `beta_mix` and `beta_mom`.
These display labels leave exported parameter keys unchanged.

## Where to look

| File | Purpose |
| --- | --- |
| [interactive.py](interactive.py), [interactive.html](interactive.html) | Local HTTP server and browser controls/plots. |
| [engine.py](engine.py) | Optimizer configuration, stepping, ensembles, and trajectory snapshots. |
| [landscapes.py](landscapes.py), [noise.py](noise.py) | Analytic losses and gradient-noise models. |
| [convergence.py](convergence.py) | Criteria for deciding whether a trajectory has converged. |
| [trajectory_landscape.py](trajectory_landscape.py) | Helpers for running and plotting trajectories from Python. |
| [make_video.py](make_video.py), [rendering.py](rendering.py) | MP4 export and shared landscape drawing helpers. |
| [presets/](presets/README.md) | Saved configurations to explore or render. |

## Check it

```bash
python -m pytest tests/test_dynamics_lab.py tests/test_dynamics_video.py -q
python -m dynamics_lab.make_video --help
```

The tests cover optimizer wiring, saved presets, reproducible noise, simulation
modes, configuration round trips, and video instance/color preservation. See the [complete guide](../docs/dynamics_lab.md)
for every control, compatibility details, and MP4 export examples, or return to
the [repository overview](../README.md).
