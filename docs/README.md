# Guides

Choose a guide based on what you want to do:

| Guide | What it explains |
| --- | --- |
| [Optimizer](optimizer.md) | Equations, defaults, every option, and named reductions |
| [Dynamics Lab](dynamics_lab.md) | Browser controls, landscapes, noise, presets, and videos |
| [Neural experiments](reproduction.md) | Datasets, CIFAR and language recipes, selection, and statistics |
| [Quadratics](quadratics.md) | The 2D examples and 9D tuning/confirmation workflow |
| [Verification](verification.md) | Checks performed for the release and their limits |

## Start with a small check

From the **repository root**, verify the optimizer on CPU:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_optimizer.py
```

For a first hands-on example, follow the
[optimizer package README](../massive_lion/README.md). For generated-data training
checks, use the [experiment package README](../experiments/README.md).
These short examples help confirm the installation before running a full study.
