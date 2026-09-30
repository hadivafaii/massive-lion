# Reproduction configurations

These JSON files define the paper's training settings and seeds. The sweep
runner expands each grid into numbered jobs, making it possible to preview a
study or assign individual jobs to available machines.

| File | Study | Jobs |
| --- | --- | ---: |
| [cifar_selected.json](cifar_selected.json) | Frozen main-table configurations across seven seeds | 70 |
| [cifar_adaptive_mass.json](cifar_adaptive_mass.json) | Adaptive rest mass | 576 |
| [cifar_fixed_mass.json](cifar_fixed_mass.json) | Fixed mass versus learning rate | 1,248 |
| [cifar_kinematics.json](cifar_kinematics.json) | Response-function comparisons | 2,736 |
| [cifar_cutoffs.json](cifar_cutoffs.json) | Fixed-mass training cutoffs | 16 |
| [language_confirmation.json](language_confirmation.json) | SlimPajama confirmation study | 112 |

[quadratics/](quadratics/README.md) contains separate plans and fixed benchmark
inputs for the 9D study.

## Preview a study

Run from the **repository root**:

```bash
python -m pip install -e .
python -m experiments.sweep configs/cifar_selected.json --output runs/cifar-selected
python -m experiments.sweep configs/language_confirmation.json --output runs/language --index 0
```

These commands print the job count and training commands. They do not train,
download data, or create run directories. `--index` selects one zero-based job;
training starts only when `--execute` is supplied.

For a quick execution check with generated data, start with the
[experiment package](../experiments/README.md). For dataset preparation,
full-run commands, and selection rules, follow the
[reproduction guide](../docs/reproduction.md).
