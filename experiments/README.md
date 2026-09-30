# Experiments

Training and analysis code for the paper's CIFAR-10/ResNet-18 and 160M-class
SlimPajama experiments, plus [controlled quadratics](quadratics/README.md).
Runs save their metrics locally; no experiment-tracking account is needed.

## Try the training paths

Run all commands from the **repository root**, with Python 3.10 or newer:

```bash
python -m pip install -e '.[vision,language,analysis,test]'
python -m experiments.cifar --smoke --output runs/smoke-cifar --no-save-checkpoint
python -m experiments.language --smoke --output runs/smoke-language --no-save-checkpoint
```

Both smoke commands run two CPU training steps on generated data, without
fetching datasets. The language smoke uses a small version of the same model.
These check the training paths; they do not reproduce the paper's scores.

Each output directory contains `config.json`, `model.json`, `environment.json`,
`history.jsonl`, and `final.json`. CIFAR also writes `best.json`, selected by
validation accuracy. Use a fresh directory when rerunning: trainers refuse to
overwrite an existing run. Omit `--no-save-checkpoint` to save model checkpoints.

## Find the code

| File or folder | Purpose |
| --- | --- |
| [`cifar.py`](cifar.py), [`language.py`](language.py) | Training loops and command-line options |
| `prepare_*.py`, `*_data.py` | Dataset preparation, batching, and augmentation |
| `*_model.py`, `*_groups.py` | Models and weight-decay parameter groups |
| [`common.py`](common.py) | Shared optimizer setup, schedules, and local logging |
| [`sweep.py`](sweep.py) | Preview or execute grids from [`configs/`](../configs/) |
| [`summarize.py`](summarize.py), [`figures.py`](figures.py) | Select results, calculate statistics, and render PDFs |
| [`quadratics/`](quadratics/README.md) | Small CPU examples and the heterogeneous 9D study |

Preview a paper grid without launching training:

```bash
python -m experiments.sweep configs/cifar_selected.json --output runs/cifar-selected
```

Run the experiment-contract tests:

```bash
python -m pytest -q tests/test_experiments.py
```

See the [reproduction guide](../docs/reproduction.md) for dataset preparation,
full training, all supplied grids, validation-only selection, and plotting.
The [verification notes](../docs/verification.md) describe what was checked.
