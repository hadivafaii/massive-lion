"""Public simulations preserve the engine and cannot grow without bounds."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading

import pytest
import torch

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dynamics_lab.engine import SimulationState
from dynamics_lab.web_api import create_app
from dynamics_lab.web_simulation import LIMITS, simulate, validate_config


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
                    busy = await client.post("/api/simulate", json=config())
                    assert busy.status_code == 503
                    assert busy.headers["retry-after"] == "5"
                    assert (await client.get("/api/health")).status_code == 200
                finally:
                    release.set()
                    responses = await asyncio.gather(first, second)
                assert all(response.status_code == 200 for response in responses)

    asyncio.run(exercise())
