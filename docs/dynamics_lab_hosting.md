# Optimizer Dynamics Lab on the web

The public playground is published at
[hadivafaii.github.io/optimizer-dynamics](https://hadivafaii.github.io/optimizer-dynamics/).
The separate `optimizer-dynamics` repository holds the Pages deployment workflow
and frozen published scenes. The numerical code and web interface live here.

Saved examples work on static hosting, including in a blog iframe. Custom
settings call the Python API, which runs the same optimizer classes used by
the local lab. JavaScript only displays the returned trajectories.

## Local preview

Install a CPU PyTorch version with Muon support, then the web dependencies:

```bash
python -m pip install -e '.[web,test]'
python -m dynamics_lab.build_web --api-url . --output dist/dynamics-lab
python -m dynamics_lab.web_api --static-dir dist/dynamics-lab --port 8013
```

Open `http://127.0.0.1:8013/`. An embedded example is available at
`http://127.0.0.1:8013/?embed=1&scene=curved-valley`.
To test completely static hosting, build without `--api-url` and serve the
output with `python -m http.server --directory dist/dynamics-lab 8014`.
Saved examples still play, while new simulation requests show an unavailable
message instead of silently replaying old results.

## Enable free custom simulations

The public playground uses [optimizer-dynamics-api.onrender.com](https://optimizer-dynamics-api.onrender.com/api/health),
deployed from this repository's `main` branch on Render's Free plan.
To deploy your own copy, use the preconfigured
[Deploy to Render link](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2Fhadivafaii%2Fmassive-lion%2Ftree%2Fmain)
or follow the manual steps below.

1. Visit [Render](https://dashboard.render.com/register), sign up, and connect
   the GitHub account that can access `hadivafaii/massive-lion`.
2. Choose **New → Blueprint**, select that repository and the
   `main` branch, and use its `render.yaml` file.
3. Review the service: **Free** plan, CPU, one `optimizer-dynamics-api` service,
   no database or paid additions. Deploy it. The Dockerfile explicitly installs
   CPU PyTorch wheels rather than CUDA dependencies.
4. Copy the resulting `https://…onrender.com` address. Visiting
   `/api/health` should return `status: ok`.
5. In `hadivafaii/optimizer-dynamics`, open **Settings → Secrets and variables →
   Actions → Variables**. Set repository variable `DYNAMICS_API_URL` to that
   Render address (no `/api` suffix). This URL is public, not a secret.
6. Open **Actions → Publish playground → Run workflow**. After Pages deploys,
   new settings run through the backend. Published examples remain independent
   of backend availability.

Render's free web services sleep after 15 minutes without traffic and can take
about a minute to wake. The page warms the service in the background, displays
progress for a custom run, and keeps saved examples usable. Free quotas apply;
no keepalive service is needed. See [Render's current free-service limits](https://render.com/docs/free).

The API permits two admitted requests (including uploads), computes one at a
time, and bounds steps, optimizer count, ensemble size, noise draws, and request
size. Uploads time out after 15 seconds. Disconnecting cannot bypass the queue
limit while a simulation is still running. Each run uses a
fresh simulation. The lock covers PyTorch's process-global RNG, so concurrent
visitors cannot change another run's seed or state. CPU thread counts are fixed
to one in the container. Measure real Linux memory/latency after deployment;
the free plan's memory limit may require smaller ensembles.

## Blog and project-page figures

Choose a saved scene and use **Copy embed**. The resulting iframe contains
only the landscape, legend, playback, and view controls. Multiple iframes can
use different scenes and play independently. **Open in playground** opens the
same example with the full controls.

For a permanent figure, finish a custom run and choose **Download scene**.
The bundle includes configuration, full trajectories, sampled landscape, and
source/version metadata. Give it a unique lowercase id (for example
`post-2026-curved-valley`) and place the JSON in `published/` in the
`optimizer-dynamics` repository. Push it and let Pages rebuild. The scene
appears in the picker and can be embedded with `?embed=1&scene=post-2026-curved-valley`.
The builder preserves frozen trajectory data; it does not rerun it against
newer optimizer code. Preserve published filenames and ids; use a new id when
revising a figure.

Alternatively generate a frozen bundle from a local lab export:

```bash
python -m dynamics_lab.build_web \
  --publish-config my-config.json \
  --scene-id post-2026-curved-valley \
  --title 'Momentum in a curved valley' \
  --description 'Compare the same initial point and learning rate.'
```

This saves a bundle under `dynamics_lab/web/published/`. Copy it into the
website repository's `published/` directory. The command refuses to overwrite
an existing id. A custom configuration link without a published bundle needs
the live backend; the share dialog makes that distinction explicit.

## Deployment and verification

The Pages workflow checks out `main` from `massive-lion` by default (or the
explicit `source_ref` supplied when running it), records the resolved commit,
installs pinned CPU dependencies, builds the renderer plus example trajectories,
adds the website repository's frozen bundles, and deploys the static artifact.
After updating the lab on `main`, wait for **Dynamics web checks**, deploy that
commit in Render, and run **Publish playground** with that same commit as
`source_ref`. The publishing workflow runs Python and browser regression tests,
then verifies the live backend's exact commit, numerical source digest, CORS,
and a real two-optimizer simulation. A mismatch or unavailable backend stops
publication and leaves the previous Pages deployment intact. This also applies
to website-repository pushes: deploy the current source `main` to Render first.
Render automatic deploys are disabled so these updates can be coordinated.
No optimizer implementation is maintained in the website repository.

Source CI also builds the actual Docker image and runs its health/simulation
checks with a 512 MiB memory limit. The container forwards termination signals
to Uvicorn so deployment shutdowns can finish admitted work.

Every generated scene records the source repository, commit, source SHA-256,
PyTorch version, dtype, and whether the source tree had uncommitted changes.
The four built-in examples are regenerated on deployment; freeze a bundle under
a unique id for a published scientific argument.

```bash
python -m pip install httpx
python -m pytest tests/test_dynamics_web.py tests/test_dynamics_web_build.py -q
npm ci --ignore-scripts
npm test
```

The browser regression suite uses jsdom and the actual local/hosted scripts,
with controlled API responses to exercise races, imports, scrubbing, and sharing.
It complements visual browser checks; it does not test canvas pixels.

Before publishing, check both the full and embedded pages, a custom run,
play/pause/scrub, mobile layout, and saved examples with the API disabled.
