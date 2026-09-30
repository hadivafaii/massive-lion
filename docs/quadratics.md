# Controlled quadratic experiments

These CPU experiments reproduce the mathematical setups in the ICLR submission.
Install `pip install -e '.[analysis]'`. Each optimizer update uses the public
`MassiveLion` implementation or its checked reductions. The code, seed plans,
and fixed confirmation inputs are included in this repository.

## Stability and shrinking updates

```bash
python -m experiments.quadratics.two_dimensional --output outputs/quadratics/2d
```

This writes two CSVs and two PDFs using the shared [plotting theme](../experiments/plotting.py):

- Stability: Hessian `diag(50, 1)`, initial position `(-1, 3)`, 100 steps,
  learning rate 0.3, fixed mass 5, momentum decay 0.9, and the twelve mixing
  coefficients specified in the appendix. AdamW uses `(0.9, 0.999)`.
- Shrinking updates: Hessian `diag(1, 10)`, the same initial position, 100 steps,
  learning rate 0.15, equal betas 0.9, and fixed mass 0 or 0.5.

The stability CSV records the curvature along the actual velocity
`v = (theta_after - theta_before) / lr`, namely `v.T @ H @ v`. No optimizer
diagnostics instrumentation is needed for this exact quadratic calculation.
The CSV positions and losses are measured after the corresponding update.

## Heterogeneous 9D study

The [runner](../experiments/quadratics/heterogeneous.py) and
[plans](../configs/quadratics/README.md) specify the paper's heterogeneous study.
The Hessian has three independently rotated blocks with spectra `{1,2,3}`,
`{99,100,101}`, and `{4998,4999,5000}`. Data columns are the
`3 U sqrt(Lambda)` factor, giving nine examples with zero targets. Batch size 3
samples without replacement within each step; batch size 9 uses exact gradients.
Gaussian initial vectors are normalized to norm 3. The optimizer uses float64,
zero rest mass, no weight decay, momentum-difference adaptation, and tied
`beta3 = kappa = beta2`.

Each run lasts 500 steps, with 5% linear warmup and a cosine schedule.
Configurations within a seed share the Hessian, initialization, and batch
sequence. Tuning uses seeds 0–31; confirmation uses seeds 1000–1127.

Run the tuning stages and freeze their selections:

```bash
python -m experiments.quadratics.heterogeneous configs/quadratics/random_tune_plan.json
python -m experiments.quadratics.heterogeneous configs/quadratics/beta_tune_plan.json
python -m experiments.quadratics.summarize select --prefix random
python -m experiments.quadratics.summarize select --prefix beta
```

Then run the **generated** confirmation plans and summarize:

```bash
python -m experiments.quadratics.heterogeneous outputs/quadratics/9d/random_confirm_plan.json --inputs configs/quadratics/confirmation_inputs.json
python -m experiments.quadratics.heterogeneous outputs/quadratics/9d/beta_confirm_plan.json --inputs configs/quadratics/confirmation_inputs.json
python -m experiments.quadratics.summarize confirm --prefix random
python -m experiments.quadratics.summarize confirm --prefix beta
python -m experiments.quadratics.summarize joint
python -m experiments.quadratics.plot
```

The runner accepts `--output DIRECTORY`; summary/plot commands accept
`--input DIRECTORY`. Plotting writes a PDF only. Do not mix results from a
previous plan with newly selected configurations: the summary checks receipt
configurations against the plan.

The supplied `configs/quadratics/*_confirm_plan.json` files specify the paper's
selected configurations and automatically load `confirmation_inputs.json`.
They can be executed directly to evaluate those exact settings without retuning:

```bash
python -m experiments.quadratics.heterogeneous configs/quadratics/random_confirm_plan.json
python -m experiments.quadratics.heterogeneous configs/quadratics/beta_confirm_plan.json
python -m experiments.quadratics.plot
```

The plotting command reads the actual receipts.
For hypothesis selection and paired comparisons, use the complete tuning →
selection → confirmation workflow above.

For a tiny execution check:

```bash
python -m experiments.quadratics.heterogeneous configs/quadratics/random_confirm_plan.json --smoke
```

This uses two seeds, two settings, and ten steps, and writes separately named
`smoke_9d` outputs. It is not a paper result.

## Selection and interpretation

The primary score is the mean over 500 post-update steps of
`log10(max(loss / initial_loss, 1e-12))`, averaged over tuning seeds. Learning
rate is selected independently at each curvature strength. Endpoint-tuned
and matched-learning-rate controls remain identified separately. Bootstrap
intervals use paired seeds; the plotted mean-score bands use a
pointwise normal approximation. Trajectory bands show interquartile ranges.

In the construction `H_block = U.T @ diag(eigenvalues) @ U`, changing eigenvector
column signs changes the objective. LAPACK does not prescribe those signs.
Thus library or operating-system changes can alter the sampled Hessian despite
the same seed. Pinning NumPy alone does not pin the system's eigensolver.

`confirmation_inputs.json` contains only the 128 confirmation Hessians, sample
factors, initial points, and seeds (250 KiB). It contains no loss trajectories
or experimental scores. Both batch sizes share these inputs, and their
batch sequences regenerate from the supplied seed rule. The runner records the
fixture's SHA-256 in each receipt. The fixture makes confirmation runs use the
same objectives and initial points across numerical backends. The 32 tuning-seed
matrices are generated at runtime rather than supplied as fixed inputs, so
retuning on another numerical backend can select different settings. Use the
supplied confirmation plans to evaluate the paper's selected comparison.

The fixed-input checks are described in [verification.md](verification.md).
The regenerated PDFs show the numerical
panels rather than reproducing the manuscript's composite artwork layout byte
for byte.
