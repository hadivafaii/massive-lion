# Tests

These CPU tests check optimizer mathematics and the behavior needed to run the
experiments and Dynamics Lab. They use small tensors and generated data, with no
dataset downloads, tracking account, or GPU required.

## Run them

From the **repository root**, install all components and run the suite:

```bash
python -m pip install -e '.[lab,vision,language,analysis,test]'
python -m pytest -q
```

For just the optimizer, the smaller `python -m pip install -e '.[test]'`
installation is sufficient:

```bash
python -m pytest -q tests/test_optimizer.py
```

## What each file covers

| File | Coverage |
| --- | --- |
| [test_optimizer.py](test_optimizer.py) | Equations, reductions, scalar/foreach agreement, and checkpoint resume |
| [test_dynamics_lab.py](test_dynamics_lab.py) | Comparisons, advanced options, noise, simulation modes, and saved presets |
| [test_dynamics_http.py](test_dynamics_http.py) | Malformed local requests and preservation of the active run |
| [test_dynamics_web.py](test_dynamics_web.py) | Public API validation, queue bounds, uploads, cancellation, CORS, and parity with the engine |
| [test_dynamics_web_build.py](test_dynamics_web_build.py) | Frozen figure preservation, rendering data, API URLs, and source metadata |
| [test_dynamics_deployment.py](test_dynamics_deployment.py) | Backend/source mismatch and incomplete simulation deployment guards |
| [dynamics_ui.test.mjs](dynamics_ui.test.mjs) | Actual browser scripts: scrubbing, async races, imports, shared visibility, configuration defaults, and playback |
| [test_experiments.py](test_experiments.py) | Data transforms, schedules, grids, validation selection, and statistics |
| [test_quadratics.py](test_quadratics.py) | Objective construction, paired inputs, score definitions, and reduction identities |

Pass any Python file above to `python -m pytest -q` to check that component alone.
Run the JavaScript suite separately with `npm ci --ignore-scripts && npm test`
(Node.js 22; no browser or Python server required).
The [experiment README](../experiments/README.md) also provides short trainer
smoke commands. Tests and smoke runs do not replace full experimental replication;
see [verification limits](../docs/verification.md).
