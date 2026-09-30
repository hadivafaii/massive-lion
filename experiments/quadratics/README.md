# Controlled quadratics

Small, inspectable optimizer experiments on quadratic losses. The 2D examples
show stability and shrinking updates; the heterogeneous 9D study compares
curvature-aware Massive Lion with its SS-AdamW reduction. These use the same
optimizer implementation as model training.

## Try it on CPU

Run all commands from the **repository root**, with Python 3.10 or newer:

```bash
python -m pip install -e '.[analysis,test]'
python -m experiments.quadratics.two_dimensional --output outputs/quadratics/2d
```

This runs the two 100-step examples and writes `stability.csv`,
`shrinking_updates.csv`, and a PDF with each matching name. Add `--no-plots`
for numerical CSVs only. Figures use the project's `create_figure` theme.

Check the 9D runner with two seeds, two settings, and ten steps:

```bash
python -m experiments.quadratics.heterogeneous configs/quadratics/random_confirm_plan.json \
  --smoke --output outputs/quadratics/smoke
```

This writes `validation.json`, `smoke_9d_data.npz`, `smoke_9d_results.npz`, and
`smoke_9d_receipt.json`. The arrays contain inputs and trajectories; the receipt
records the settings and environment. Smoke output is an execution check,
not a full confirmation result. Neither example downloads training data.

## Find the code

| File | Purpose |
| --- | --- |
| [`two_dimensional.py`](two_dimensional.py) | Stability sweep, Signum/M-Signum trajectories, CSVs, and PDFs |
| [`heterogeneous.py`](heterogeneous.py) | 9D data construction, schedules, optimizer runs, and receipts |
| [`summarize.py`](summarize.py) | Freeze tuning selections and summarize paired confirmation seeds |
| [`plot.py`](plot.py) | Render the 9D numerical panels from completed runs |
| [`configs/quadratics/`](../../configs/quadratics/) | Seed plans and historical confirmation input fixture |

Run the focused numerical tests:

```bash
python -m pytest -q tests/test_quadratics.py
```

For the complete tuning → selection → confirmation workflow, follow the
[quadratic guide](../../docs/quadratics.md). It explains the score, seed pairing,
and why the included historical inputs matter across numerical backends.
See also [verification](../../docs/verification.md) and the
[parent experiment guide](../README.md).
