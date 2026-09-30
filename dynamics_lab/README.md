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

Open [the local lab](http://127.0.0.1:8011), then use Run, Pause, Step, and Reset.
Use Import to load a JSON file from [presets/](presets/README.md); Export saves
settings for another session or video rendering. Stop the server with Ctrl-C.
The HTML page needs the Python server, so opening it directly is insufficient.

New MassiveLion rows enable the paper's momentum-difference adaptive mass,
with mass parameters tied to beta2. The saved demonstration presets explicitly
use fixed mass to preserve their original trajectories.

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
python -m pytest tests/test_dynamics_lab.py -q
python -m dynamics_lab.make_video --help
```

The tests cover optimizer wiring, saved presets, reproducible noise, simulation
modes, and configuration round trips. See the [complete guide](../docs/dynamics_lab.md)
for every control, compatibility details, and MP4 export examples, or return to
the [repository overview](../README.md).
