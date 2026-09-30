# Reproduce the ICLR experiments

The public recipes follow the current ICLR manuscript. They cover ResNet-18/CIFAR-10, the 160M-class SlimPajama confirmation study, response-map ablations, fixed-mass training cutoffs, and controlled quadratics. Newer 411M/1.06B exploratory runs are outside this release.

The controlled quadratic recipes are documented in [quadratics.md](quadratics.md).

All commands below run from the repository root. Install the appropriate extras first:

```bash
python -m pip install -e '.[lab,vision,language,analysis,test]'
```

Install a PyTorch/torchvision build suitable for your GPU. The small checks below use CPU and generated data; they never download a dataset and are not experimental results:

```bash
python -m experiments.cifar --smoke --output runs/smoke-cifar --no-save-checkpoint
python -m experiments.language --smoke --output runs/smoke-language --no-save-checkpoint
python -m pytest -q
```

Use a fresh output directory for each run. A trainer refuses to overwrite an existing `config.json`. Local outputs include the effective configuration, model configuration, environment versions, `history.jsonl`, and `final.json`. CIFAR additionally writes `best.json`; checkpoint saving writes `best.pt` and `final.pt` for CIFAR, and `final.pt` for language. `--no-save-checkpoint` saves only the numerical results. No tracking account or cluster launcher is required.

## CIFAR-10

Prepare the public dataset:

```bash
python -m experiments.prepare_cifar --root data/cifar10
```

Preparation uses the original seed-0 permutation: `torch.randperm(50000, generator=torch.Generator().manual_seed(0))`, with its first 45,000 examples used for training and remaining 5,000 for validation. The official 10,000-example test split stays in source order. It writes normalized float32 arrays and split indices to `data/cifar10/processed`. Existing research arrays can also be used by passing their directory through `--data`.

The normalization is mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2023, 0.1994, 0.2010)`. The training loader preserves raw-black four-pixel padding followed by online 32-pixel crops and horizontal flips, using a device-local generator seeded with `training_seed + epoch`. Validation and test images are never augmented. The entire dataset and padded training images reside on the selected device.

The seed/split/normalization recipe was recovered from the research repository's `gra/utils/dataset.py` at commit `c5123cad0f6d01932a05fb7f0a0b056e809139a4`. The current training loader was extracted from the current research source; it is tested against raw-image augmentation independently.

Every full CIFAR run uses ResNet-18, batch size 256, 80,000 updates, cosine decay to `1e-5`, no warmup, no clipping, and evaluation every 5,000 updates. It retains the original ResNet construction order, sorted decay groups, and mixed-precision policy (bfloat16 where supported, otherwise float16). A learning rate is assigned before each update; the first update uses the peak learning rate.

| Grid | Purpose | Runs |
|---|---|---:|
| `configs/cifar_selected.json` | Ten frozen main-table configurations, seeds 0–6 | 70 |
| `configs/cifar_adaptive_mass.json` | Adaptive rest-mass panel | 576 |
| `configs/cifar_fixed_mass.json` | Four displayed fixed masses versus learning rate | 1,248 |
| `configs/cifar_kinematics.json` | Two broad response-map grids and seed extension | 2,736 |
| `configs/cifar_cutoffs.json` | Eight rest masses for each of Lion and Signum | 16 |

Preview a grid, then explicitly execute a job:

```bash
python -m experiments.sweep configs/cifar_selected.json --output runs/cifar-selected
python -m experiments.sweep configs/cifar_selected.json --output runs/cifar-selected --index 0 --execute
```

Omit `--index` to execute the entire grid sequentially. An external scheduler can assign distinct indices; the runner does not allocate machines or start hidden background jobs. Extra trainer options follow `--`, for example `-- --device cuda:1 --data /path/to/processed`. Training defaults to CUDA, with compilation enabled. Reducing the batch size or step count changes the recipe; such outputs will not pass the original grid's summary checks.

The main table's ten configurations come from `tables/cifar_main.tex` and the first of each pair in the research `2026_09_23__final_seed_sweep.json`. These recipes rerun the frozen winners. They do not claim to recreate every intermediate decision in the historical tuning campaign. Each JSON records its source documents.

## SlimPajama language modeling

Prepare the source document pools and tokenize them with GPT-NeoX:

```bash
python -m experiments.prepare_language --out-path data/slimpajama \
  --dataset-split train --output-split train --nrows 15000000 --target-chunks 1587200
python -m experiments.prepare_language --out-path data/slimpajama \
  --dataset-split validation --output-split valid --nrows 100000 --target-chunks 48804
```

The source defaults to `gmongaras/SlimPajama-627B_Reupload` and tokenizer `EleutherAI/gpt-neox-20b`. This preprocessing is expensive. It materializes the streamed document pool, shuffles with seed 1996, tokenizes nonempty documents with one trailing end-of-text token, truncates each 1,024-document mapping batch into 2,049-token rows, then shuffles chunks with seed 96 and selects the requested rows. Keep `--num-proc 8` and the map batch size for the supplied recipe. Changing map batching/process partitioning can change the resulting token stream.

Training uses the preserved plainLM transformer: 12 layers, width 768, 12 heads, GLU expansion `8/3`, rotary positions, untied embeddings, QK and embedding normalization. Each optimizer update accumulates eight microbatches of 32 sequences of length 2,048. The 6,200 updates consume 3,250,585,600 prediction tokens. All configurations use bfloat16, weight decay 0.1, zero rest mass, coordinatewise momentum-difference adaptive mass tied to momentum decay, no bias correction, and no optimizer epsilon. Gradient clipping is specified per configuration: 13 configurations use norm 1 and three use no clipping.

The data sampler traverses the prepared sequences in order. The original schedule starts with an update at learning rate zero, advances after each optimizer update, warms up for 620 steps, and then follows cosine decay to `1e-5`. The public trainer retains this order. Validation selects 48,804 rows from the 100M-token budget and drops the last incomplete microbatch, actually evaluating 48,800 sequences / 99,942,400 prediction tokens.

```bash
python -m experiments.sweep configs/language_confirmation.json --output runs/language
python -m experiments.sweep configs/language_confirmation.json --output runs/language --index 0 --execute
```

The grid contains exactly the 16 configurations in the research `2026_09_24__secret_sauce_confirmation_96.json`, expanded to seeds **100–106** to include the exploratory references: 112 runs. In the manuscript, seed 100 selected configurations; seeds 101–106 are fresh initialization replicates. The data order is fixed across seeds. The grid includes the selected SS-AdamW baseline at momentum 0.95/LR 0.008 and matched high-momentum controls. It does not substitute the older single-seed language sweep.

Full language training needs substantial GPU memory. To use a smaller microbatch, increase `--accumulation` proportionally and retain `--target-batch-size 256`, while recording this deviation. This public trainer uses one device per run. It does not reproduce multi-device reduction ordering.

## Result selection and statistical summaries

Run the corresponding summary after a grid completes:

```bash
python -m experiments.summarize selected --runs runs/cifar-selected \
  --config configs/cifar_selected.json --output outputs/cifar-selected
python -m experiments.summarize adaptive-mass --runs runs/cifar-adaptive \
  --config configs/cifar_adaptive_mass.json --output outputs/cifar-adaptive
python -m experiments.summarize fixed-mass --runs runs/cifar-fixed \
  --config configs/cifar_fixed_mass.json --output outputs/cifar-fixed
python -m experiments.summarize kinematics --runs runs/cifar-kinematics \
  --config configs/cifar_kinematics.json --output outputs/cifar-kinematics
python -m experiments.summarize cutoffs --runs runs/cifar-cutoffs \
  --config configs/cifar_cutoffs.json --output outputs/cifar-cutoffs
python -m experiments.summarize language --runs runs/language \
  --config configs/language_confirmation.json --output outputs/language
```

The summary verifies the requested grid against each local configuration and final step. It rejects smoke results, missing runs, and conflicting settings. `--allow-incomplete` explicitly enables descriptive partial summaries without paired tests (except kinematics, which requires the complete grid). The paper's adaptive-mass snapshot contained 574 completed runs from 576 planned; running the complete supplied grid may change validation-selected winners.

Selection rules:

- CIFAR selects each run's checkpoint by validation correct count; an exact tie keeps the earliest checkpoint. Test metrics come from that checkpoint and never affect selection.
- Adaptive-mass points select learning rate and weight decay within each mass/momentum pair. Fixed-mass learning-rate points select beta pairs and weight decay independently within each family/mass/LR cell.
- Main-table means and sample standard deviations use seeds 0–6. Pairwise test-accuracy comparisons exclude selection seed 0 and apply two-sided paired tests to seeds 1–6; Holm correction covers all 13 predeclared within-section comparisons.
- Kinematics selects by mean validation accuracy across five seeds per configuration. Matched map differences are averaged over the 12 configurations within each seed before computing a five-seed Student-t interval. Configurations are not independent seed replicates.
- Cutoffs use checkpoint prefixes of the original 80,000-step schedule. Exact checkpoint ties keep the earlier step; across masses, ties prefer the earlier selected checkpoint and then the smaller mass. No shorter training schedules are substituted.
- Language uses the final checkpoint. It averages **per-run perplexities**, not the exponential of mean loss. Seven-seed summaries include sample standard deviations and pointwise Student-t intervals. Separate fresh-seed means/SDs use 101–106. Ten paired tests use validation-loss differences on those six fresh seeds and share one Holm correction. These intervals do not adjust for selecting configurations.

Output schemas:

- `per_seed.csv` contains each configuration, seed, selected metrics, and run index.
- `summary.csv` contains selected single-seed rows for mass/LR panels; aggregated CIFAR rows use `mean_*`, `sd_*`, and `n_seeds` for validation/test metrics.
- Language rows contain `mean_validation_perplexity`, `sd_validation_perplexity`, `ci95_low`, `ci95_high`, and corresponding loss fields. `fresh_` fields describe seeds 101–106; `zeta = 1 - beta1/beta2`.
- Cutoff rows carry `family`, `mass`, `cutoff`, `epochs`, `selected_step`, and `selected_mass` alongside checkpoint metrics.
- `comparisons.csv` holds the planned paired tests or kinematics intervals; `coverage.json` records expected/completed runs and missing indices. Kinematics additionally writes `all_configurations.csv`.

## Render the experiment panels

Once the corresponding summaries exist, these commands render PDF panels using the project's `create_figure` theme:

```bash
python -m experiments.figures adaptive-mass --summary outputs/cifar-adaptive/summary.csv \
  --output outputs/cifar-adaptive/rest_mass.pdf
python -m experiments.figures fixed-mass --summary outputs/cifar-fixed/summary.csv \
  --output outputs/cifar-fixed/learning_rate_response.pdf
python -m experiments.figures cutoffs --summary outputs/cifar-cutoffs/summary.csv \
  --output outputs/cifar-cutoffs/training_cutoffs.pdf
python -m experiments.figures language --summary outputs/language/summary.csv \
  --output outputs/language/curvature.pdf
```

These regenerate numerical panels from the local runs; they do not embed the manuscript's artwork or cached scores. The language panel uses the matched momentum-0.975, clipping-on comparisons and seven-seed Student-t intervals.

## Provenance and limits

The training/model/data implementations were cleaned directly from the research repository's vision and Secret Sauce trainers, preserving model initialization, parameter order, data order, arithmetic, decay groups and schedule placement. Research diagnostics, remote orchestration, account identifiers, and output caches are omitted. The independent public optimizer tests establish agreement with the research update; experiment tests cover data transformations, grid sizes, schedules, reduction settings, validation-only selection, and statistical units.

The original experiments did not record immutable dataset/tokenizer revisions. The current download sources may evolve; reusing the original prepared splits is preferable for exact replay. GPU kernels, software versions, compilation, precision, and hardware can change trajectories. The release includes runnable recipes and source; it does not include checkpoints, prepared datasets, cached W&B histories, or hard-coded paper scores. CPU smoke tests establish executable paths, not full-scale replication.
