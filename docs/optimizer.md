# Massive Lion optimizer

```python
from massive_lion import MassiveLion

optimizer = MassiveLion(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.99),  # (direction mixing, momentum decay)
    mass=0.01,
    adaptive_mass=True,
    weight_decay=0.1,
    foreach=True,
)

optimizer.zero_grad()
loss = loss_function(model(inputs), targets)
loss.backward()
optimizer.step()
```

`MassiveLion` is the optimizer called `gr_lion` in the research implementation.
It contains no optimizer diagnostic collectors or logging dependencies. The
named reductions below all use this same implementation.

## Read the update

The readable implementation is [`MassiveLion._step_single`](../massive_lion/optimizer.py).
The batched implementation is [`step_foreach`](../massive_lion/_foreach.py).
Select them with `foreach=False` or `foreach=True`, respectively.

With the default options, let `g` be the current gradient, `m` the previous
momentum, and `a2` the adaptive contribution to squared mass. Both state buffers
start at zero. One step is:

```python
innovation = g - m
a2 = beta_momentum * a2 + beta_momentum * (1 - beta_momentum) * innovation**2
direction = beta_mix * m + (1 - beta_mix) * g
velocity = direction / sqrt(direction**2 + mass**2 + a2)
parameter = (1 - lr * weight_decay) * parameter - lr * velocity
m = beta_momentum * m + (1 - beta_momentum) * g
```

All these operations are coordinatewise. The implementation defines `0/0 = 0`.
Setting `adaptive_mass=False` sets `a2=0`. `mass` is the fixed scale rho_0 in
the paper, in gradient/direction units; it is not converted from a mechanical
rest mass or rescaled when the learning rate changes. There is no epsilon or
bias correction. Both the innovation and direction use the **old** momentum.

For numerical agreement with the research implementation, the stored
`metric_diag` buffer is the **total** squared mass `mass**2 + a2`, initialized
at `mass**2`. Its recurrence is implemented using `lerp`; it is mathematically
equivalent to the formula above when mass is fixed. `exp_avg` stores momentum.

## Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `params` | required | Parameters or PyTorch parameter-group dictionaries. |
| `lr` | `1e-4` | Nonnegative step size. |
| `betas` | `(0.9, 0.99)` | `(beta_mix, beta_momentum)`, each in `[0,1)`. The first mixes the update direction; the second stores momentum. |
| `weight_decay` | `0.0` | Nonnegative decoupled decay, applied as `p *= 1 - lr * weight_decay`. |
| `mass` | `0.01` | Nonnegative fixed saturation scale rho_0. |
| `adaptive_mass` | `True` | Enable mass adaptation; `False` uses fixed mass. |
| `foreach` | `True` | Batch tensor operations to reduce launch/dispatch overhead. `False` uses the scalar reference path. |
| `maximize` | `False` | Reverse the optimization update for ascent; weight decay still shrinks parameters. |
| `update_mode` | `"coordinate"` | Coordinatewise response, or the per-tensor `"vector"` / `"vector_rms"` extensions below. |
| `mass_mode` | `"momentum_diff"` | Use deviations from stored momentum; `"gradient_diff"` enables the optional gradient-change rule. |
| `kinematics` | `"minkowski"` | Response map; also supports `"arctan"` and `"tanh"`. |
| `kappa` | `None` | Adaptive coupling. `None` ties it to that parameter group's `beta_momentum`. An explicit nonnegative value enables an independent coupling. |
| `beta_gravity` | `None` | Adaptive-mass decay. `None` ties it to that group's `beta_momentum`; an explicit value must be in `[0,1)`. |

The defaults specify the paper's adaptive recurrence. To reproduce fixed-mass
results, use `adaptive_mass=False`; explicit `kappa` and `beta_gravity` must then
be zero or omitted. Both are treated as zero internally. `beta_mix` may exceed
`beta_momentum` for ablations, although the dissipative regime discussed in the
paper has `beta_mix <= beta_momentum`. The discrete curvature parameter is
`zeta = 1 - beta_mix / beta_momentum` when `beta_momentum > 0`.

## Named reductions

```python
from massive_lion import Lion, Signum, MassiveSignum, SecretSauceAdamW
from massive_lion import create_optimizer

optimizer = SecretSauceAdamW(model.parameters(), lr=1e-3, betas=(0.95, 0.95))
# Equivalent constructor by name:
optimizer = create_optimizer("ss_adamw", model.parameters(), lr=1e-3,
                             betas=(0.95, 0.95))
```

| Constructor / factory name | Enforced reduction |
| --- | --- |
| `MassiveLion` / `m_lion`, `massive_lion` | Full optimizer and optional extensions. |
| `Lion` / `lion` | Coordinatewise, zero mass, adaptation off. Betas default to `(0.9, 0.99)`. |
| `Signum` / `signum` | Lion with equal betas, default `(0.99, 0.99)`. |
| `MassiveSignum` / `m_signum`, `massive_signum` | Coordinatewise with equal betas, default `(0.99, 0.99)` and mass `0.01`. Adaptation defaults off, but may be enabled with the tied momentum-difference recurrence. |
| `SecretSauceAdamW` / `ss_adamw`, `secret_sauce_adamw` | Equal betas, zero mass, tied momentum-difference adaptation, coordinatewise Minkowski response. Betas default to `(0.99, 0.99)`. |

Secret Sauce AdamW is equal-beta AdamW **without bias correction and with
epsilon zero**. It is not PyTorch AdamW with its usual defaults. The identity
underlying this reduction is `m**2 + a2 = EMA(g**2)` when both EMAs use the same
decay and start at zero.

Aliases reject conflicting arguments and parameter-group overrides. For
example, `SecretSauceAdamW(..., mass=0.1)` or a Signum parameter group with
unequal betas raises `ValueError`. Validation also covers groups added later,
loaded checkpoints, and changed group options. Changing the class name cannot
silently change the advertised reduction. These aliases are small subclasses
of `MassiveLion`; there is no separate legacy M-Signum update implementation.

## Optional extensions

The full `MassiveLion` class retains the GR implementation's exploratory
controls. These differ from the settings used for the paper's principal
adaptive optimizer.

For a reduction operator `R`, innovation `e`, coupling `kappa` and decay `b`,
the general mass recurrence is:

```text
M[0] = mass**2
M[t] = b * M[t-1] + (1-b) * (mass**2 + kappa * R(e[t]**2))
```

- `momentum_diff`: `e[t] = g[t] - m[t-1]`, including the first gradient.
- `gradient_diff`: `e[t] = g[t] - g[t-1]`, with the first innovation set to zero.
- `coordinate`: `R` is the identity; each coordinate has its own mass and
  speed bound of one.
- `vector`: `R` sums over each parameter tensor. Each tensor has one mass and
  an L2 speed bound of one.
- `vector_rms`: `R` averages over each parameter tensor. Each tensor has one
  mass and an RMS speed bound of one.

Vector modes operate on each parameter tensor, not on a flattened model or
the whole parameter group. Changing tensor partitioning can change the
algorithm in these modes. The same reduction is used for the innovation and
the direction's squared magnitude.

For scalar magnitude-to-mass ratio `z`, the response maps are:

| Map | Response phi(z) |
| --- | --- |
| `minkowski` | `z / sqrt(1 + z**2)` |
| `arctan` | `(2/pi) * atan((pi/2) * z)` |
| `tanh` | `tanh(z)` |

All maps have unit slope at the origin, so the same mass gives the same
small-direction linear response. All saturate at unit speed. Vector modes
preserve the direction and apply the response to its L2/RMS magnitude. At
zero mass, coordinate mode takes the sign and vector modes take the radial
unit-speed limit. Zero direction gives zero velocity.

## Parameter groups and numerical behavior

```python
optimizer = MassiveLion([
    {"params": model.encoder.parameters(), "betas": (0.8, 0.9)},
    {"params": model.head.parameters(), "betas": (0.9, 0.95), "weight_decay": 0.0},
], lr=1e-3, weight_decay=0.1)
```

Omitted adaptive controls follow each group's momentum decay, including
group overrides. Parameters with `grad=None` are skipped completely: no
weight decay, momentum decay or mass update. A zero gradient is an actual
step and can relax existing momentum and adaptive mass.

FP32 and FP64 parameters use their own dtype for optimizer state. FP16/BF16
parameters use FP32 state and intermediate gradients, then cast updates to
the parameter dtype. Checkpoint loading preserves the saved FP32 buffers.
The gradient-change extension additionally stores `prev_grad` and `step`;
the paper configuration does not need those buffers.

The optimizer supports closures and PyTorch `state_dict`/`load_state_dict`.
Sparse gradients and complex parameters are rejected. The two implementations
can have ordinary floating-point rounding differences. `foreach` uses
additional temporary tensor lists and may trade memory for reduced overhead;
the non-Minkowski maps still evaluate the response per tensor.

Optimizer checkpoints belong to this public implementation. Importing older
research checkpoints with different optimizer names or schemas requires an
explicit conversion; no silent legacy migration is performed.
