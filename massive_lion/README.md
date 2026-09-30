# Massive Lion optimizer

This is the PyTorch optimizer package. `MassiveLion` combines momentum with a
bounded update whose scale is set by fixed or adaptive mass. The named reductions
(`lion`, `signum`, `m_signum`, and `ss_adamw`) share the same implementation and
check that their hyperparameters define the requested algorithm.

## Try it

Run these commands from the **repository root**. This small CPU example minimizes
a quadratic, without downloading data:

```bash
python -m pip install -e '.[test]'
python - <<'PY'
import torch
from massive_lion import MassiveLion

x = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
optimizer = MassiveLion([x], lr=0.1, betas=(0.7, 0.95), mass=0.1)
initial_loss = x.square().sum().item()
for _ in range(50):
    optimizer.zero_grad()
    x.square().sum().backward()
    optimizer.step()
print(f"loss: {initial_loss:.4f} -> {x.square().sum().item():.4f}")
PY
```

The printed loss should decrease. Set `foreach=False` to use the readable
per-tensor path; the default `foreach=True` batches tensor operations.

Check the equations, reductions, checkpoint state, and agreement between both paths:

```bash
python -m pytest -q tests/test_optimizer.py
```

## Where to read

- [optimizer.py](optimizer.py): state initialization and the update, in equation order.
- [_foreach.py](_foreach.py): the batched version of the same update.
- [aliases.py](aliases.py): named reductions and their enforced settings.
- [baselines/](baselines/README.md): comparison optimizers for the experiments and Lab.

See the [option and equation guide](../docs/optimizer.md) for defaults, mass
adaptation, vector modes, and response functions, or the [Lab](../dynamics_lab/README.md)
to explore their behavior interactively.
