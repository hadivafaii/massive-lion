# Plans and inputs for the 9D study

These files specify quadratic objectives, optimizer settings, and paired seeds.
Each objective has three rotated blocks with different curvature scales.
Configurations within a seed share the objective, starting point, and batches.

- `random_tune_plan.json` explores learning rates and curvature strength at momentum 0.95.
- `beta_tune_plan.json` extends tuning to momentum 0.9 and 0.975.
- `random_confirm_plan.json` and `beta_confirm_plan.json` specify the paper's selected configurations for 128 fresh seeds.
- `confirmation_inputs.json` contains those seeds' matrices and initial points, with no saved scores or trajectories.

The fixed inputs make confirmation runs independent of eigensolver
sign conventions. The confirmation plans load them automatically.

## Check a plan

From the **repository root**, run a tiny CPU check:

```bash
python -m pip install -e '.[analysis,test]'
python -m experiments.quadratics.heterogeneous \
  configs/quadratics/random_confirm_plan.json --smoke \
  --output outputs/quadratics/smoke
python -m pytest -q tests/test_quadratics.py
```

The smoke run uses two seeds, two settings, and ten steps. It writes `smoke_9d`
data, results, and a JSON receipt to the output directory. It checks execution;
it is not a full confirmation result.

See the [quadratic package](../../experiments/quadratics/README.md) for the code
map and [study guide](../../docs/quadratics.md) for full tuning, selection,
confirmation, and plotting commands.
