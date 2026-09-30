# Comparison optimizers

These optimizers provide comparisons in Dynamics Lab and the paper experiments:

- [cautious.py](cautious.py): cautious Lion and AdamW, which mask updates using gradient agreement.
- [curvature_aware.py](curvature_aware.py): SGD/QHM, AdamW, and Muon variants with separate direction and momentum coefficients.
- [muon.py](muon.py): PyTorch Muon for matrix parameters, with AdamW for other parameters.
- [vr_adam.py](vr_adam.py): the VRAdam comparison.
- [arctan_lion.py](arctan_lion.py): RLion expressed as a `MassiveLion` wrapper, converting its native arctan mass units.

The paper's `ss_adamw` and `m_signum` reductions live in
[aliases.py](../aliases.py). They inherit the main `MassiveLion` update.

## Try the comparisons

From the **repository root**, install the Lab dependencies and run the focused
checks. These take a few optimizer steps on CPU and need no dataset:

```bash
python -m pip install -e '.[lab,test]'
python -m pytest -q tests/test_dynamics_lab.py \
  -k 'every_optimizer_runs or cautious_masks or rlion_wrapper'
```

To compare trajectories visually:

```bash
python -m dynamics_lab.interactive --port 8011
```

Open [localhost:8011](http://127.0.0.1:8011) and choose optimizers from the menu.
Muon requires PyTorch 2.9 or newer, included in the Lab dependency requirements.

See the [Lab guide](../../docs/dynamics_lab.md) for controls and
[third-party notices](../../THIRD_PARTY_NOTICES.md) for source attribution and licenses.
