# Release verification

Checks were run on macOS ARM64 with Python 3.12 and CPU PyTorch 2.14.1.
[requirements-tested.txt](../requirements-tested.txt) records the directly used
package versions. The GitHub workflow also defines Linux checks on Python 3.10
and 3.12; those hosted checks have not been run as part of this local release.

## Optimizer equivalence

An independent comparison extracted the original `GeneralRelativisticLion`
class and its mathematical helpers from the research source, without importing
the research package. It compared 432 cases for 20 updates each:

- all coordinate, vector, and vector-RMS modes;
- Minkowski, arctan, and tanh responses;
- momentum-difference and gradient-difference mass adaptation;
- fixed, tied, and independently configured adaptive mass;
- float64, float32, float16, and bfloat16 parameters;
- scalar and foreach implementations, multiple groups, and missing gradients.

All parameter and optimizer-state values matched bitwise in this CPU check.
The public tests separately cover mathematical limiting cases, aliases and
their parameter-group constraints, zero directions, weight decay, checkpoint
resume, and scalar/foreach agreement. The obsolete research `MassiveLion` and
`MassiveSignum` implementations are not part of the release.

## Experiments

Both trainers completed CPU smoke runs with generated data. These execute the
actual model and training code, with a small language-model configuration.
Independent comparisons checked model initialization, Transformer forward
values, CIFAR decay-group ordering, and the recovered CIFAR split preparation
against their research counterparts. Named-reduction smoke runs also checked
that saved configurations describe the effective optimizer settings.

The supplied 9D confirmation plans were run for all 500 steps using the fixed
benchmark inputs. All **18,432 per-seed records** matched the archived study:
92,160 values across mean log loss, endpoint log loss, final loss, and the two
threshold times were exactly equal, with no failed runs. Configuration matching
used each new run's receipt and seed array. This includes both batch sizes and
all three momentum settings.

The fixed-input JSON has SHA-256
`abc5889170ee1cf58a0aa673c3966ca22661204840710d373422c653316ae27c`.
It contains benchmark inputs only. See [quadratics.md](quadratics.md) for why
these matrices are needed for reproducible historical replay and the remaining
limitation on retuning with newly generated inputs.

Both 2D quadratic studies and the 9D plot were executed and their PDFs inspected.
The four neural-experiment PDF renderers were checked with explicitly synthetic
summary fixtures kept outside this repository. Synthetic scores are not included
in the release or presented as experimental results.

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
