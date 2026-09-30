# Dynamics Lab

This simulator is an interactive 2D sandbox for comparing optimizer kinematics.
It is meant for quickly finding landscapes and hyperparameters where the
difference between massless sign dynamics, massive sign dynamics, and classical
optimizers is visible by eye.

The simulator is not a static HTML file. It uses a small local Python server so
the browser can call the repo's simulation code and optimizer implementations
one step at a time.

The server has one shared simulation state. Multiple browser tabs connected to
the same server therefore control and display the same run.

## Launch Locally

Install the lab dependencies, then run from the repository root:

```bash
python -m pip install -e ".[lab]"
```

```bash
python -m dynamics_lab.interactive --port 8011
```

The Lab opens automatically in your default browser at
[http://127.0.0.1:8011](http://127.0.0.1:8011). The tab is titled **Dynamics Lab**
and has a red optimization icon. The server also prints its URL in the terminal.

To start the server without opening a browser, add `--no-browser`:

```bash
python -m dynamics_lab.interactive --port 8011 --no-browser
```

If that port is already busy, choose another one:

```bash
python -m dynamics_lab.interactive --port 8012
```

Do not open `dynamics_lab/interactive.html` directly from Finder or with a
`file://` URL. The page will load, but the simulation API will not be available.

To stop the server, press `Ctrl-C` in the terminal where it is running.

After updating the code, export any settings you want to keep, stop the server
with `Ctrl-C`, and run the launch command again. Reload the page and import your
settings so the browser and Python server use the same version.

## Export MP4 Videos

Exported JSON configs can also be rendered directly to MP4. The command accepts
a config path, filename, stem, or exported config name:

```bash
python -m dynamics_lab.make_video curved_valley_2d --mode parallel
```

By default, the MP4 is written under `outputs/dynamics_lab/`.
Pass `--output` to choose a path. The renderer uses the config's saved
`landscape_view` by default, so both 2D and 3D configs work:

```bash
python -m dynamics_lab.make_video curved_valley_3d --output /tmp/curved_valley_3d.mp4
```

By default, the renderer writes every optimizer step for smooth playback. Use
`--frame-stride` to render every Nth step, or `--max-frames` to pick a stride
from a target frame budget.

Serial videos can either keep earlier trajectories on screen or replace them as
the next optimizer starts:

```bash
python -m dynamics_lab.make_video curved_valley_2d --mode serial --serial-history replace
```

Ensemble videos use the saved optimizer instance and its parameters. To select
another row, pass its `instance_id` from the exported JSON:

```bash
python -m dynamics_lab.make_video comparison.json --mode ensemble --ensemble-instance-id optimizer-2 --ensemble-count 50
```

The legacy optimizer-type override also remains available. It selects the first
matching row and replaces a saved instance selection:

```bash
python -m dynamics_lab.make_video curved_valley_2d --mode ensemble --ensemble-optimizer MassiveLion --ensemble-count 50
```

## What You Are Looking At

The page has five live views of the same run:

`loss landscape`
: The 2D or 3D loss surface. Learners are drawn as colored balls with visible
trajectory history. The star marks the sampled global minimum. The dashed black
curve marks a normalized excess-loss contour around that minimum.

`loss over time`
: Loss versus iteration. The y-axis defaults to log scale so early descent and
late small differences are visible together.

`theta1 over time`
: Absolute `theta1` value on the x-axis and iteration on the y-axis. This is the
spacetime plot. The vertical marker shows the global-minimum `theta1` location.
For massless sign optimizers, dashed pieces show the null-direction segments
between sign changes.

`theta-dot1`
: Coordinate velocity `theta-dot1 / c`. The plot always shows reference rails
at `-1` and `+1`, and expands beyond them when an unbounded optimizer requires
more range. Massless sign methods should jump between the rails. Massive sign
methods can be smooth when they are not fully saturated.

`kinetic momentum histogram`
: Normalized kinetic momentum `p1 / rho` for relativistic and massless sign
optimizers. Counts accumulate as the run proceeds. In ensemble mode, counts
aggregate over both time and all ensemble members. Optimizers without a
meaningful `rho`, such as SGD and Adam-family methods, are not assigned
relativistic regimes. For MassiveLion, `rho` is the current effective scale
`sqrt(metric_diag)` rather than its fixed baseline `mass`.
For MassiveLion's vector modes, it instead shows nonnegative momentum norm / effective
mass or momentum RMS / effective mass, matching the selected speed rule.

## Main Workflow

1. Pick a run mode and optimizer set.
2. Tune the landscape and initial point.
3. Press `Reset` to apply changed simulation settings.
4. Press `Run`, or use `Step` to inspect individual sign flips.
5. Use `Export` once you find a setting worth keeping.

Most controls are read when the simulation is reset. Display-only controls, such
as trajectory fading, loss-axis scale, color spacing, and view angle, update
without changing optimizer dynamics.

## Header Buttons

`Export`
: Opens the normal browser save dialog for a JSON snapshot of the current
settings. The suggested filename is based on the landscape type and timestamp.
The JSON includes optimizer order and parameters, stable instance IDs, custom
names and colors, landscape and noise settings, run mode, ensemble selection,
convergence controls, display controls, and camera state. The browser chooses
the save location; if you want configs in the repo, save them under a repo-local
config folder such as
`dynamics_lab/presets`.

`Import`
: Opens the normal browser file picker, reads an exported JSON config, applies
the saved controls, and resets the simulator.

Top legend
: Click an optimizer name in the top legend to cycle its color. In ensemble
mode, the legend also reports how many learners converged to the global-minimum
basin. If learners diverge, the legend and status line report that too.

## Run Controls

`Run`
: Starts live stepping. Points are generated one optimizer step at a time.

`Pause`
: Stops live stepping without resetting the current trajectories.

`Step`
: Advances one simulator tick. This is useful for inspecting sign changes,
velocity jumps, and sudden loss spikes.

`Reset`
: Rebuilds the landscape sample and restarts learners from the current settings.

`mode`
: `parallel` runs all listed optimizers side by side from the same initial
point. `serial` runs the listed optimizers one after another, but each optimizer
still starts from the same configured initial point. `ensemble` runs many random
initializations of one selected optimizer.

`ensemble optimizer`
: The optimizer instance used for every random start. Entries use the same
titles as their rows, so two instances of one optimizer can be selected
independently. The chosen row supplies its parameters and color. Older exports
that select an optimizer type still use the first matching row or its defaults.

`random starts`
: Number of ensemble learners. Initial points are drawn uniformly from the
current landscape bounds. The gradient-noise seed also seeds these starts.

`window`
: Number of most recent trace points used to decide whether an ensemble learner
has converged.

`frac within`
: Required fraction of the late-window trace points whose loss lies inside the
target energy basin.

`epsilon`
: Target suboptimality for ensemble convergence. A learner is counted as
converged when at least `frac within` of its last `window` samples satisfy
`V(theta) - V* <= epsilon`. The dashed target contour uses the same `epsilon`,
and the display `energy radius` control is kept synchronized with it.

`steps`
: Number of optimizer steps. This also fixes the vertical range of the `theta1`
and `theta-dot1` plots, so the light-cone geometry does not rescale while the run
unfolds.

`delay ms`
: Delay between live updates. Smaller values run faster; larger values are
easier to watch.

`theta1 start`, `theta2 start`
: Initial point for `parallel` and `serial` mode. Moving this near a ravine wall
usually makes oscillations more visible.

## Display Controls

`taper trajectories`
: Older trajectory segments are thinner; newer segments are thicker.

`fade trajectory history`
: Older trajectory segments are more transparent; the latest segment remains
fully opaque.

These two controls are display-only. They do not change the simulated updates.

`star size`
: Display radius for the target star.

`energy radius`
: Geometric radius for the dashed target contour. This is redundant with
`epsilon`: the simulator synchronizes the two controls using
`epsilon = 0.5 * k_ref * radius^2`, where `k_ref` is the geometric mean of the
positive local Hessian eigenvalues at the sampled optimum.

## Optimizer controls

Each row is an independent optimizer instance. Use `Add optimizer` or a row's
`Copy` button to compare copies of the same algorithm with different settings. Automatic titles
show the settings that distinguish same-family rows; identical copies receive
numbered titles. The suggestion is editable text: change any part to keep a
custom title, or clear it and leave the field to restore automatic naming.
Optimizer choices are listed alphabetically. Row colors, names, order, and instance identity survive
export/import and video rendering. Videos show the landscape trajectories;
they do not add the browser's legends or control panels.

The beta labels describe the Lab controls; JSON keys and training APIs are
unchanged. Standard `Adam`/`AdamW` show `beta_mom` for first-moment decay (`beta1`)
and `beta_var` for second-moment decay (`beta2`). Other two-beta rows show
`beta_mix` (`beta1`) and `beta_mom` (`beta2`); their algorithm-specific roles still
apply, including the moment decays of CautiousAdamW and VRAdam. A direct
`momentum` control, such as SGD's, is displayed as `beta_mom`.

The lab uses the installed PyTorch optimizers. It never reimplements their
update rules in JavaScript. Display traces and momentum histograms are computed
by the simulation engine; the optimizer itself has no research diagnostic hooks.

`MassiveLion` is the current implementation used by the paper (formerly called
`gr_lion` in research runs). `Lion`, `Signum`, `MassiveSignum`, and
`SecretSauceAdamW` are parameter presets of this one implementation.
`VectorMassiveLion` is a convenience preset for its tensorwise vector modes.

| Control | Meaning |
| --- | --- |
| learning rate | Step scale; one learning-rate unit per iteration is speed 1 in the plots. |
| beta_mix | Gradient mixing: direction = beta_mix × previous momentum + (1 − beta_mix) × gradient. |
| beta_mom | Stored momentum EMA decay for the Lion family. |
| mass/rho | Nonnegative baseline rest mass. Zero rest mass can still have adaptive mass. |
| adaptive mass | Enable mass adaptation; off gives a constant rest mass. |
| tie mass parameters to beta_mom | Paper setting: kappa = mass-memory decay = beta_mom. Untie to edit each independently. |
| kappa | Advanced nonnegative scale for the squared innovation, editable when untied. |
| beta_gravity | Advanced mass-memory EMA decay in [0, 1), editable when untied. |
| mass recurrence | `momentum_diff` uses gradient − previous momentum (paper); `gradient_diff` uses consecutive gradient differences, with first innovation zero. |
| update mode | `coordinate` uses coordinatewise masses/speed bounds; `vector` uses a scalar mass and L2 speed ≤ 1; `vector_rms` uses a scalar mass and RMS speed ≤ 1. |
| kinematics | Minkowski, arctan, or tanh; maps share unit slope at the origin and saturation at ±1. |
| foreach implementation | Use batched PyTorch tensor operations; disabling selects the readable scalar implementation. |
| wd | Decoupled weight decay for MassiveLion; defaults to zero in the lab. |

The speed bounds apply before weight decay. Vector modes use the two landscape
coordinates as one tensor. Vector histograms therefore use the nonnegative
momentum norm/RMS divided by effective mass; coordinate modes use signed p1/rho.

`Lion` fixes mass to zero and disables adaptation. `Signum` additionally ties
both betas to the momentum control. `MassiveSignum` keeps that equal-beta
constraint and exposes rest mass and the paper adaptive-mass toggle.
`SecretSauceAdamW` fixes both betas to beta_mom, rest mass and epsilon to zero,
and coordinate Minkowski momentum-difference adaptation with tied decay and
coupling, without bias correction. An imported config claiming this name but
requesting an incompatible reduction is rejected rather than mislabeled.

The comparison menu also includes GD, PyTorch SGD/Adam/AdamW, cautious Lion and
AdamW, curvature-aware SGD (QHM), curvature-aware AdamW/Muon, Muon, RLion, and
VRAdam. Their row controls only show settings they use. Cautious variants retain
the research baselines' exact scalar masking conventions; the C-Optim license
is included under `massive_lion/baselines`. Muon uses PyTorch's real matrix path
on a 2 × 1 parameter tensor and requires a PyTorch release with `torch.optim.Muon`.
RLion is a thin wrapper of MassiveLion with fixed coordinate arctan kinematics.
It converts its original native mass to the shared reference mass by multiplying
by pi/2; the MassiveLion arctan option uses the common unit-slope convention.

Three independent checkboxes control sharing: `shared learning rate`,
`shared mass/rho`, and `shared betas`. Each synchronizes only its corresponding
row controls. Turning one off disables and dims its global inputs and unlocks
the applicable row inputs at their current values. The other sharing switches
are unaffected.

`shared mass/rho` shares only baseline rest mass. Kappa and `beta_gravity`
remain governed by each row's adaptive-mass and mass-tying controls.
`shared betas` follows parameter roles: shared `beta_mom` writes JSON `beta2`
for Lion-family optimizers, `beta1` for Adam-family optimizers/Muon, and
`momentum` for SGD/Signum. Shared `beta_mix` affects `beta1` for mixed-momentum
variants; shared variance controls Adam-family `beta2` and curvature-aware
AdamW `beta3`. Drag rows to change their serial order.

## Saved presets and compatibility

All 17 original demonstration presets are in `dynamics_lab/presets`. They
explicitly disable adaptive mass to preserve the original demonstrations. The
two old vector presets use no adaptive geometry and map to the same fixed-mass
`vector_rms` dynamics. These examples do not silently adopt the new adaptive
default when imported.

Exports use schema version 2 and preserve adaptive mass, tying, recurrence,
kinematics, update mode, foreach, noise identity, camera, and display options.
Each optimizer row also retains its stable `instance_id`, custom `label` (when
set), and `color`; `ensemble_instance_id` identifies the selected row. Legacy
rows without IDs receive distinct IDs when loaded. Automatic titles are
recomputed from the imported settings, while custom names remain unchanged.
`controls.shared_mass` records rest-mass sharing; `controls.shared_beta` records
beta sharing only. Older exports without `shared_mass` use their saved
`shared_beta` state for mass sharing too, preserving their previous behavior.
Schema version 1 imports preserve fixed-mass legacy MassiveLion/Signum runs and
recognize `GRLion`/`gr_lion` as the current `MassiveLion`. Legacy vector configs
with `adaptive_geometry=true` are rejected: the old diagonal preconditioner is
not part of the unified implementation. Choose scalar adaptive vector mass
explicitly for a new experiment instead.

## Landscape Controls

`type`
: `sloped ravine` is the default landscape for the current demo. It has a sharp
diagonal valley with a target basin and is tuned to make sign oscillations
visible. `curved valley` restores the earlier sinusoidal valley style with a
more winding centerline. Changing this selector loads that landscape's preset
values into the shape controls.

`target x`, `target y`
: Coordinates of the quadratic target used by the landscape. In the pure
quadratic ravine setting, these are the exact optimum coordinates.

`x curvature`
: Pull along the `theta1` direction toward the target. In the axis-aligned
quadratic ravine, this is `lambda_x`.

`wall curvature`
: Cross-ravine wall curvature. In the axis-aligned quadratic ravine, this is
`lambda_y`; when the centerline is tilted or bent, it is the curvature
perpendicular to the centerline rather than a coordinate eigenvalue.

`centerline slope`
: Diagonal tilt of the ravine. Negative values make the valley descend across
the panel.

`centerline offset amp`
: Vertical offset of the ravine center in `sloped ravine`, and sinusoidal
amplitude in `curved valley`.

`centerline bend`
: Bend of the ravine center. In `sloped ravine`, nonzero values introduce a
smooth kink. In `curved valley`, this is the sinusoidal frequency.

`ripple amplitude`
: Height of small periodic bumps along the ravine.

`ripple frequency`
: Frequency of the periodic bumps.

`outer confinement`
: Stabilizing pull that keeps the landscape bounded. Increase it if learners
wander too far; decrease it if the basin overwhelms the ravine structure.

## Gradient Noise Controls

Noise is added directly to the gradient before each optimizer step.

`noise on`
: Enables or disables gradient noise without clearing the selected noise
parameters. When it is off, the run is deterministic apart from ensemble start
sampling, which still uses the seed.

`mode`
: `additive` keeps the original noise model. `Yu heavy-tail` uses
`grad_i + (sigma0 + sigma1 * |grad_i|) * xi_i`, matching the synthetic
additive-floor plus signal-dependent noise model. `Hessian-powered Gaussian`
draws centered Gaussian noise with covariance
`hessian_scale * |H(theta)|^hessian_power`, using the closed-form Hessian at
the learner's current position. The simulator remains noiseless by default
until `noise on` is enabled.

| Control | Applies to | Intuition |
| --- | --- | --- |
| `distribution` | additive | Shape of the old fixed-scale noise. |
| `std dev` | additive | One global noise amplitude, independent of gradient size. |
| `xi distribution` | Yu | Shape of the random multiplier that creates rare storms. |
| `sigma0` | Yu | Background noise floor, present even near a flat optimum. |
| `sigma1` | Yu | Extra noise that grows where gradients are large. |
| `tail p` | Yu | Tail-heaviness target; lower values mean wilder rare events. |
| `tail margin` | Yu | Distance from the infinite-moment boundary; smaller means more extreme outliers. |
| `batch size` | Yu | Number of noisy draws averaged per step; larger smooths the storm. |
| `covariance scale` | Hessian | Proportionality constant multiplying the covariance; defaults to `0.1`. |
| `Hessian power` | Hessian | Spectral power applied to the absolute Hessian; defaults to `1`. |
| `seed` | all | Reproducible random draws. |

`distribution`
: Additive-noise distribution. `gaussian` is the default. `laplace`,
`student t, df=5`, and `student t, df=3` are increasingly heavy-tailed.
`rademacher` adds random signs with fixed magnitude.

`std dev`
: Gradient-noise standard deviation. `0` gives deterministic gradients.

`xi distribution`
: Yu-mode distribution for `xi_i`, the random multiplier in each coordinate.
Student-t is a smooth heavy-tailed choice. Symmetric Pareto-like gives more
abrupt rare shocks. Gaussian is included as a light-tailed comparison. Each
option is centered and normalized so the requested finite `tail p` moment is
one.

`sigma0`
: Additive noise floor in Yu mode. This is the amount of randomness that
remains even when the true gradient is tiny. Increasing it makes the endpoint
more jittery and can keep learners wandering around a minimum instead of
settling quietly.

`sigma1`
: Signal-dependent noise strength in Yu mode. This multiplies `|grad_i|`, so
the noise gets stronger on steep walls and weaker near flat regions. Increasing
it creates the visual effect of storms in sharp or high-gradient parts of the
landscape.

`tail p`
: Moment order used to normalize `xi_i`. Think of it as the knob for how
aggressively heavy-tailed the storm is. Values closer to `2` behave more like
ordinary finite-variance noise. Values closer to `1` produce rarer but much
larger shocks.

`tail margin`
: How far the distribution tail parameter is kept above the requested `tail p`.
Small margins put the distribution close to the edge where that moment would
stop being finite, so outliers become more dramatic. Larger margins make the
noise less explosive while preserving the same normalization.

`batch size`
: Number of independent Yu-noise samples averaged into each stochastic
gradient. Larger values mimic mini-batch averaging: the trajectory becomes
smoother, but heavy-tailed shocks average away more slowly than Gaussian noise.

`covariance scale`
: Proportionality constant for Hessian-powered noise. It multiplies the
**covariance**, not the standard deviation: with the default power `1`, the
covariance is `0.1 * |H(theta)|` by default. Setting it to `0` removes this
noise even when the mode is enabled.

`Hessian power`
: Nonnegative spectral power applied to the absolute Hessian. The default `1`
makes covariance directly proportional to local curvature magnitude. `0`
gives isotropic covariance (scaled by `covariance scale`), while larger values
emphasize high-curvature eigendirections more strongly.

`seed`
: Random seed for gradient noise and ensemble initializations.

## Landscape View Controls

The landscape panel can be viewed as either `2D` or `3D`.

`2D`
: Top-down heatmap and contour plot. Drag to pan. Use the mouse wheel or `+` and
`-` buttons to zoom.

`3D`
: Projected loss surface. Left-drag rotates. Middle-drag, or
Alt/Option-left-drag, pans. Use the mouse wheel or `+` and `-` buttons to zoom.

Reset view
: Restores the default pan, zoom, and 3D camera angle.

Coordinate bounds
: The compact `theta1` and `theta2` number boxes in the landscape header set the
sampled plotting bounds for the two coordinates. Changing them requires
`Reset`, because the landscape grid is rebuilt.

`lin/log`
: Controls whether landscape colors and contour levels are spaced linearly or
logarithmically over the sampled loss range. The default is `log`.

None of these view controls change optimizer dynamics.

## Regime Definitions

For relativistic and massless sign optimizers, the histogram classifies
normalized kinetic momentum `p1 / rho`:

- classical: `|p1 / rho| <= 0.1`;
- relativistic: `0.1 < |p1 / rho| < 1.0`;
- ultra-relativistic: `|p1 / rho| >= 1.0`.

For coordinate MassiveLion, the denominator is its evolving effective scale
`rho_eff = sqrt(metric_diag)` for the plotted coordinate. Vector MassiveLion uses
`||p|| / rho_eff`, and Vector RMS uses `RMS(p) / rho_eff`, with the scalar
effective mass. These nonnegative magnitudes use the same thresholds above,
so the histogram reports whole-vector saturation even when the first
coordinate is stationary. The velocity panel still shows the signed first
coordinate: the `±1` rails bound individual coordinates in Vector mode, while
Vector RMS can reach `±sqrt(2)` along one axis in this two-dimensional lab.

For SecretSauceAdamW, the denominator is its evolving adaptive mass. The
direction and mass use the raw, uncorrected recurrence: this exact reduction
has no bias correction.

For coordinate massive optimizers,

```text
theta-dot1 / c = p1 / sqrt(p1^2 + rho^2)
```

so `|p1 / rho| = 1` corresponds to
`|theta-dot1 / c| = 1 / sqrt(2)`, not to exactly `1`. The velocity curve only
approaches the `+1` and `-1` boundaries when `|p1 / rho|` is very large.

Massless sign optimizers such as Lion and Signum have no finite `rho`. Any
nonzero sign step is treated as ultra-relativistic and placed at the histogram
tails.
The zero-mass limit of vector MassiveLion is a normalized vector step. Its projected
velocity need not be `±1`, so the lab does not draw coordinate sign/null guides
for either vector mode. With a nonzero direction and zero scalar mass, its
whole-vector regime is ultra-relativistic.

Unbounded and adaptive optimizers such as SGD, CurvatureAwareSGD, AdamW,
CautiousAdamW, and CurvatureAwareAdamW are excluded from this histogram because
normalizing their momentum by another optimizer's `rho` would not define a
physical regime.
CautiousLion is also excluded because its masked, tensor-rescaled step is not a
fixed lightlike update.

## Divergence Guardrails

If one learner produces non-finite loss, gradient, momentum, or parameters, or
if it reaches an extremely large loss or leaves the landscape by a large margin,
only that learner is marked `diverged` and frozen. Other learners keep running.

This is mainly to keep unstable SGD or Adam settings from stopping the whole
interactive session.

## Practical Tuning Tips

To make massless oscillation obvious, increase `wall curvature`, start near a
ravine wall, and watch `theta-dot1` jump between `-1` and `+1`.

To make MassiveLion look smoother than Lion, increase `mass/rho` until its
velocity spends visible time away from the saturated rails.

To make Signum fall into a trap, use high `momentum`, keep the ravine sharp,
and tune the starting point so the trajectory repeatedly crosses the valley
wall.

To isolate curvature-aware dissipation from clipping, compare
`CurvatureAwareSGD` against SGD and Lion with `shared learning rate` and
`shared betas` enabled. Comparing Lion/MassiveLion
against Signum still shows the same injection inside the bounded family. The
default sloped-ravine setup is tuned to make the resulting oscillatory
differences visible.
