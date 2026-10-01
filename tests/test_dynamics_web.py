"""Public simulations preserve the engine and cannot grow without bounds."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import threading
import time

import pytest
import torch

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dynamics_lab.engine import SimulationState
from dynamics_lab.web_api import create_app
from dynamics_lab.web_simulation import LIMITS, runtime_provenance, simulate, validate_config


def config(seed=19, mode="parallel"):
    return {
        "mode": mode, "max_steps": 6, "ensemble_count": 3,
        "ensemble_optimizer": "MassiveLion", "ensemble_instance_id": "massive",
        "theta0": [-1.8, 2.3],
        "noise": {"enabled": True, "mode": "yu", "xi_distribution": "student_t", "seed": seed},
        "optimizers": [
            {"name": "MassiveLion", "instance_id": "massive", "lr": 0.03},
            {"name": "AdamW", "instance_id": "adam", "lr": 0.03},
        ],
    }


@pytest.mark.parametrize("mode", ["parallel", "serial", "ensemble"])
def test_complete_web_run_matches_every_engine_step(mode):
    original = config(mode=mode)
    saved = deepcopy(original)
    state = SimulationState()
    state.reset(original)
    while not state.done:
        state.step()
    expected = state.snapshot(include_landscape=True)
    actual = simulate({"config": original, "controls": {"landscape_view": "3d"}})
    assert actual == expected
    assert original == saved
    assert len(actual["landscape"]["x"]) == 85
    assert actual["done"]


def test_concurrent_runs_are_reproducible_and_preserve_global_rng():
    torch.manual_seed(481)
    initial_rng = torch.random.get_rng_state().clone()
    expected = {seed: simulate(config(seed, "ensemble")) for seed in [3, 97]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        seeds = [3, 97, 97, 3]
        actual = list(pool.map(lambda seed: simulate(config(seed, "ensemble")), seeds))
    assert actual == [expected[seed] for seed in seeds]
    assert torch.equal(initial_rng, torch.random.get_rng_state())
    assert expected[3]["learners"][0]["trace"] != expected[97]["learners"][0]["trace"]


@pytest.mark.parametrize("change,match", [
    ({"max_steps": 2001}, "max_steps"),
    ({"max_steps": -1}, "max_steps"),
    ({"max_steps": "30"}, "integer"),
    ({"max_steps": True}, "integer"),
    ({"optimizers": [{"name": "Lion"}] * 13}, "at most"),
    ({"optimizers": {}}, "list"),
    ({"optimizers": [{"name": "Lion", "lr": "0.1"}]}, "number"),
    ({"optimizers": [{"name": "Lion", "lr": float("nan")}]}, "finite"),
    ({"optimizers": [{"name": "MassiveLion", "adaptive_mass": "false"}]}, "boolean"),
    ({"optimizers": [{"name": "Lion", "noise_stride": 1000000}]}, "noise_stride"),
    ({"noise": {"seed": 0.5}}, "integer"),
    ({"noise": {"batch_size": 33}}, "batch_size"),
    ({"noise": {"enabled": "true"}}, "boolean"),
    ({"noise": {"mode": "unknown"}}, "noise.mode"),
    ({"theta0": [0]}, "two coordinates"),
    ({"theta0": [False, 0]}, "number"),
    ({"landscape": {"kind": "unknown"}}, "landscape.kind"),
    ({"landscape": {"x_min": 4, "x_max": 3}}, "must exceed"),
    ({"landscape": {"sharpness": float("inf")}}, "finite"),
    ({"mode": "unknown"}, "mode"),
    ({"ensemble_count": 51}, "ensemble_count"),
    ({"mode": "ensemble", "ensemble_count": 50, "max_steps": 401}, "optimizer steps"),
    ({"noise": {"mode": "yu", "batch_size": 32}, "max_steps": 2000}, "noise draws"),
])
def test_invalid_and_excessive_requests_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        validate_config({**config(), **change})


def test_serial_noise_work_includes_uncached_stride_fallbacks():
    value = config(mode="serial")
    value["max_steps"] = 1000
    value["optimizers"][1]["noise_stride"] = 3
    with pytest.raises(ValueError, match="noise draws"):
        validate_config(value)


def test_generated_instance_selection_and_empty_optional_ids_match_engine():
    value = {
        "mode": "ensemble", "max_steps": 2, "ensemble_count": 1,
        "ensemble_instance_id": "optimizer-2",
        "optimizers": [
            {"name": "MassiveLion", "instance_id": "optimizer-1", "lr": 0.01},
            {"name": "MassiveLion", "instance_id": "", "color": "", "lr": 0.07},
        ],
    }
    result = simulate(value)
    assert result["ensemble_instance_id"] == "optimizer-2"
    assert result["learners"][0]["lr"] == 0.07
    value["ensemble_instance_id"] = ""
    assert validate_config(value) == value


def test_valid_beta_boundary_and_sgd_momentum_are_preserved():
    value = {"max_steps": 2, "optimizers": [
        {"name": "MassiveLion", "beta1": math.nextafter(1.0, 0.0), "beta2": 0.9999999999},
        {"name": "SGD", "momentum": 1.0},
    ]}
    result = simulate(value)
    assert result["optimizers"][0]["beta1"] == value["optimizers"][0]["beta1"]
    assert result["optimizers"][1]["momentum"] == 1.0
    for row in ({"name": "MassiveLion", "beta1": 1.0}, {"name": "Signum", "momentum": 1.0},
                {"name": "MassiveLion", "tie_mass": False, "beta3": 1.0}):
        with pytest.raises(ValueError):
            validate_config({"optimizers": [row]})


def test_extreme_allowed_geometry_produces_bounded_finite_snapshot():
    value = {
        "max_steps": 2, "theta0": [10000, 10000],
        "optimizers": [{"name": "MassiveLion"}, {"name": "SGD"}],
        "landscape": {
            "kind": "curved_valley", "x_min": -10000, "x_max": 10000,
            "y_min": -10000, "y_max": 10000, "target_x": 10000, "target_y": 10000,
            "sharpness": 10000, "valley_amp": 10000, "valley_freq": 10000,
            "valley_tilt": 10000, "ripple_amp": 10000, "ripple_freq": 10000,
            "boundary": 10000, "along_curvature": 10000,
        },
        "noise": {"enabled": True, "mode": "hessian", "hessian_scale": 1000, "hessian_power": 10},
    }
    result = simulate(value)
    json.dumps(result, allow_nan=False)
    assert result["done"]
    assert len(result["landscape"]["x"]) == LIMITS["landscape_grid"]
    assert all(len(row["trace"]) <= 3 for row in result["learners"])


def test_api_returns_real_snapshot_with_backend_provenance_and_json_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "a" * 40)
    (tmp_path / "index.html").write_text("<h1>Dynamics Lab preview</h1>")
    with TestClient(create_app(tmp_path, allowed_origins=["https://example.github.io"])) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/defaults").json()["limits"] == LIMITS
        response = client.post("/api/simulate", json=config(), headers={"Origin": "https://example.github.io"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://example.github.io"
        result = response.json()
        assert result["snapshot"] == simulate(config())
        provenance = result["provenance"]
        assert provenance["repository"] == "https://github.com/hadivafaii/massive-lion"
        assert provenance["commit"] == "a" * 40
        assert len(provenance["source_sha256"]) == 64
        int(provenance["source_sha256"], 16)
        assert provenance["uncommitted_changes"] in (True, False, None)
        assert provenance["torch_version"] == torch.__version__
        assert provenance["device"] == "cpu"
        assert provenance["dtype"] == "float64"
        for content, status in [("{", 400), ('{"max_steps":NaN}', 400), ("[]", 400), ("x" * 65537, 413)]:
            response = client.post("/api/simulate", content=content, headers={"Content-Type": "application/json"})
            assert response.status_code == status
            assert "error" in response.json()
        assert client.post("/api/simulate", content="{}").status_code == 415
        assert client.post("/api/simulate", json=config()).status_code == 200
        assert client.get("/api/missing").json() == {"error": "Unknown API endpoint"}
        denied = client.get("/api/health", headers={"Origin": "https://unrelated.example"})
        assert "access-control-allow-origin" not in denied.headers
        preflight = client.options("/api/simulate", headers={
            "Origin": "http://localhost:8013", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert preflight.status_code == 200


def test_api_rejects_streamed_oversized_body_without_content_length():
    async def exercise():
        async def chunks():
            yield b"x" * 40000
            yield b"x" * 40000

        application = create_app()
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                response = await client.post("/api/simulate", content=chunks(), headers={"Content-Type": "application/json"})
                assert response.status_code == 413

    asyncio.run(exercise())


def test_stalled_upload_times_out_and_canceled_uploads_release_slots(monkeypatch):
    monkeypatch.setitem(LIMITS, "request_timeout_seconds", 0.03)

    async def exercise():
        started = asyncio.Event()

        async def stalled_body():
            started.set()
            yield b"{"
            await asyncio.Event().wait()

        application = create_app()
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                timed_out = await client.post("/api/simulate", content=stalled_body(), headers={"Content-Type": "application/json"})
                assert timed_out.status_code == 408
                assert "timed out" in timed_out.json()["error"]
                for _ in range(LIMITS["concurrent_requests"]):
                    started.clear()
                    request = asyncio.create_task(client.post("/api/simulate", content=stalled_body(), headers={"Content-Type": "application/json"}))
                    await started.wait()
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                assert (await client.post("/api/simulate", json=config())).status_code == 200

    asyncio.run(exercise())


def test_api_bounds_queue_and_keeps_health_responsive(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_simulation(_config):
        started.set()
        assert release.wait(5)
        return {"done": True}

    monkeypatch.setattr("dynamics_lab.web_api.simulate", slow_simulation)

    async def exercise():
        application = create_app()
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/api/simulate", json=config()))
                assert await asyncio.to_thread(started.wait, 2)
                second = asyncio.create_task(client.post("/api/simulate", json=config()))
                await asyncio.sleep(0.02)
                try:
                    body_was_read = False

                    async def body():
                        nonlocal body_was_read
                        body_was_read = True
                        yield b"{}"

                    busy = await client.post("/api/simulate", content=body(), headers={"Content-Type": "application/json"})
                    assert busy.status_code == 503
                    assert busy.headers["retry-after"] == "5"
                    assert not body_was_read
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    # Canceling the HTTP waiter cannot release its CPU slot.
                    assert (await client.post("/api/simulate", json=config())).status_code == 503
                    assert (await client.get("/api/health")).status_code == 200
                finally:
                    release.set()
                    responses = await asyncio.gather(first, second, return_exceptions=True)
                assert isinstance(responses[0], asyncio.CancelledError)
                assert responses[1].status_code == 200
                assert (await client.post("/api/simulate", json=config())).status_code == 200

    asyncio.run(exercise())


def test_response_serialization_does_not_block_health(monkeypatch):
    from dynamics_lab import web_api
    started = threading.Event()
    release = threading.Event()
    response_type = web_api.JSONResponse

    def slow_response(content, *args, **kwargs):
        if isinstance(content, dict) and "snapshot" in content:
            started.set()
            assert release.wait(3)
        return response_type(content, *args, **kwargs)

    monkeypatch.setattr(web_api, "JSONResponse", slow_response)
    monkeypatch.setattr(web_api, "simulate", lambda _config: {"done": True})

    async def exercise():
        application = create_app()
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
                request = asyncio.create_task(client.post("/api/simulate", json=config()))
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    before = time.monotonic()
                    assert (await client.get("/api/health")).status_code == 200
                    assert time.monotonic() - before < 0.5
                finally:
                    release.set()
                    response = await request
                assert response.status_code == 200

    asyncio.run(exercise())


@pytest.mark.parametrize("error,status", [(ValueError("Invalid optimizer settings"), 400),
                                         (RuntimeError("Internal implementation detail"), 500)])
def test_simulation_errors_are_json_with_cors_and_capacity_recovers(monkeypatch, error, status):
    def failing_simulation(_config):
        raise error

    monkeypatch.setattr("dynamics_lab.web_api.simulate", failing_simulation)
    with TestClient(create_app(allowed_origins=["https://example.github.io"])) as client:
        for _ in range(3):
            response = client.post("/api/simulate", json=config(), headers={"Origin": "https://example.github.io"})
            assert response.status_code == status
            assert "error" in response.json()
            assert response.headers["access-control-allow-origin"] == "https://example.github.io"
            assert "Internal implementation detail" not in response.text
        monkeypatch.setattr("dynamics_lab.web_api.simulate", lambda _config: {"done": True})
        assert client.post("/api/simulate", json=config()).json()["snapshot"] == {"done": True}


def test_source_provenance_without_git_preserves_host_commit_and_unknown_dirty_state(monkeypatch):
    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("dynamics_lab.web_simulation.subprocess.run", no_git)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    source = runtime_provenance()
    assert source["commit"] is None
    assert source["uncommitted_changes"] is None
    assert len(source["source_sha256"]) == 64
    monkeypatch.setenv("RENDER_GIT_COMMIT", "b" * 40)
    hosted_source = runtime_provenance()
    assert hosted_source["commit"] == "b" * 40
    assert hosted_source["source_sha256"] == source["source_sha256"]
