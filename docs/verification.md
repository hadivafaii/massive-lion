# Release verification

Checks were run on macOS ARM64 with Python 3.12 and CPU PyTorch 2.14.1.
[requirements-tested.txt](../requirements-tested.txt) records the directly used
package versions. The GitHub workflow also defines Linux checks on Python 3.10
and 3.12; those hosted checks have not been run as part of this local release.

## Optimizer checks

The self-contained checks in [test_optimizer.py](../tests/test_optimizer.py)
compare the optimizer with explicit update equations and compare scalar and
foreach implementations. Coverage includes:

- all coordinate, vector, and vector-RMS modes;
- Minkowski, arctan, and tanh responses;
- momentum-difference and gradient-difference mass adaptation;
- fixed, tied, and independently configured adaptive mass;
- float64, float32, float16, and bfloat16 parameters;
- scalar and foreach implementations, multiple groups, and missing gradients.

The tests also cover mathematical limiting cases, aliases and their
parameter-group constraints, zero directions, weight decay, and checkpoint
resume. Numerical comparisons use dtype-appropriate tolerances; exact checks
are used for state preservation and skipped updates.

Run these checks from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_optimizer.py
```

## Experiments

Both trainers completed CPU smoke runs with generated data. These execute the
actual model and training code, with a small language-model configuration.
The [experiment tests](../tests/test_experiments.py) check CIFAR normalization,
the seeded split, raw-image augmentation, language chunking, training schedules,
grid coverage, validation-only selection, and statistical summaries.
Named-reduction checks verify that saved configurations describe the effective
optimizer settings. See the [experiment README](../experiments/README.md) for
smoke commands that use generated data.

The [quadratic tests](../tests/test_quadratics.py) check objective construction,
the SS-AdamW reduction, repeatable runs with paired initial conditions, score
definitions, and separate tuning/confirmation seeds. They also verify that the
supplied fixed inputs preserve the Hessians, sample factors, and initial points
when loaded by the runner.

The fixed-input JSON has SHA-256
`abc5889170ee1cf58a0aa673c3966ca22661204840710d373422c653316ae27c`.
It contains benchmark inputs only. See [quadratics.md](quadratics.md) for why
these matrices are needed for reproducible confirmation runs and the remaining
limitation on retuning with newly generated inputs.

The [quadratic guide](quadratics.md) and [neural reproduction guide](reproduction.md)
provide commands to generate numerical panels from locally produced results.
Generated-data smoke outputs are not paper results.

## Dynamics Lab

Checks exercised all 18 menu choices, the full Massive Lion option combinations,
all 17 shipped presets, and deterministic parallel, serial, and ensemble runs.
Named reductions use the checked public wrappers. RLion also shares the core
update, with an explicit conversion from its native arctan mass units.

Browser checks covered startup, advanced optimizer controls, and 3D playback.
Both 2D and 3D MP4 smoke exports completed. Preset data and backend serialization
were tested; the browser's native file-import interaction was not fully verified.

## Scope of these checks

Full CIFAR and language-model training was not rerun. CUDA kernels, distributed
training, accelerator performance, and every supported dependency version were
not tested. The foreach implementation is supplied, but this release makes no
new measured speedup claim. Data downloads and tokenization of the full corpus
were not repeated; original immutable dataset revisions were not recorded.

The tests and recipes establish numerical agreement and executable paths.
They do not guarantee identical neural-network trajectories across hardware,
software versions, or newly downloaded dataset revisions.
