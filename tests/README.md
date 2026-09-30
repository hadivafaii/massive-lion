# Tests

These CPU tests check optimizer mathematics and the behavior needed to run the
experiments and Dynamics Lab. They use small tensors and generated data, with no
dataset downloads, tracking account, or GPU required.

## Run them

From the **repository root**, install all components and run the suite:

```bash
python -m pip install -e '.[lab,vision,language,analysis,test]'
python -m pytest -q
```

For just the optimizer, the smaller `python -m pip install -e '.[test]'`
installation is sufficient:

```bash
python -m pytest -q tests/test_optimizer.py
```

## What each file covers

| File | Coverage |
| --- | --- |
| [test_optimizer.py](test_optimizer.py) | Equations, reductions, scalar/foreach agreement, and checkpoint resume |
| [test_dynamics_lab.py](test_dynamics_lab.py) | Comparisons, advanced options, noise, simulation modes, and saved presets |
| [test_experiments.py](test_experiments.py) | Data transforms, schedules, grids, validation selection, and statistics |
| [test_quadratics.py](test_quadratics.py) | Objective construction, paired inputs, score definitions, and reduction identities |

Pass any file above to `python -m pytest -q` to check that component alone.
The [experiment README](../experiments/README.md) also provides short trainer
smoke commands. Tests and smoke runs do not replace full experimental replication;
see [verification limits](../docs/verification.md).
