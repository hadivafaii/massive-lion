# Dynamics Lab presets

These JSON files save optimizer settings, landscapes, noise, run modes, and
view controls for the [Dynamics Lab](../README.md). Import one in the browser
or pass its filename/stem to the video renderer. They are editable starting
points for exploring dynamics; training recipes are described under
[experiments/](../../experiments/).

## Good starting points

| Preset | What to explore |
| --- | --- |
| [curved_valley_2d.json](curved_valley_2d.json) | Optimizer trajectories through a curved valley. |
| [curved_valley_3d.json](curved_valley_3d.json) | The same landscape with a saved 3D view. |
| [sloped_ravine_2d.json](sloped_ravine_2d.json) | Motion along a narrow, tilted valley. |
| [curved_valley_ensemble_massive_lion.json](curved_valley_ensemble_massive_lion.json) | An ensemble of MassiveLion trajectories. |
| [noise_chatter_ensemble_2d.json](noise_chatter_ensemble_2d.json) | A saved noisy ensemble configuration. |
| [zeta_stability_quadratic_ravine_2d.json](zeta_stability_quadratic_ravine_2d.json) | Curvature and stability on a quadratic ravine. |

The graphical-abstract and date-stamped files preserve additional saved views
and comparisons. All 17 migrated presets explicitly disable adaptive mass for
the Massive variants, preserving their fixed-mass demonstrations. Change that
control deliberately when exploring the paper's adaptive-mass setting.

## Try or check a preset

Run these commands from the repository root:

```bash
python -m pip install -e ".[lab,test]"
python -m dynamics_lab.interactive --port 8011
```

The Lab opens in your default browser. Click Import and choose one of these
files. Add `--no-browser` to skip the automatic launch; the local URL is still
printed in the terminal.
To render a compact video or check that every saved preset runs:

```bash
python -m dynamics_lab.make_video curved_valley_3d --max-frames 40 --output outputs/dynamics_lab/curved_valley_3d.mp4
python -m pytest tests/test_dynamics_lab.py -q -k all_saved_presets
```

The video command runs the saved simulation and limits rendered frames. For
configuration fields, import compatibility, and export options, see the
[complete guide](../../docs/dynamics_lab.md).
