"""ASGI service for Dynamics Lab; run with ``python -m dynamics_lab.web_api``."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from dynamics_lab.engine import defaults_payload
from dynamics_lab.web_simulation import LIMITS, runtime_provenance, simulate, validate_config


LOGGER = logging.getLogger(__name__)


def create_app(static_dir: str | Path | None = None, *, allowed_origins: list[str] | None = None) -> FastAPI:
    """Create a service with a bounded queue and optional static site preview.

    ALLOWED_ORIGINS is a comma-separated list of website origins, without paths.
    Localhost is also accepted for development. No cookies or credentials are
    required, and the public endpoint does not retain visitor configurations.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dynamics")
    pending = 0
    provenance = runtime_provenance()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="Dynamics Lab", docs_url=None, redoc_url=None, lifespan=lifespan)
    origins = allowed_origins if allowed_origins is not None else [
        item.strip() for item in os.environ.get("ALLOWED_ORIGINS", "").split(",") if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "engine": "repository-python", "revision": provenance["commit"] or "local"}

    @app.get("/api/defaults")
    async def defaults():
        return {**defaults_payload(), "limits": LIMITS}

    @app.post("/api/simulate")
    async def run(request: Request):
        nonlocal pending
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return JSONResponse({"error": "Content-Type must be application/json"}, status_code=415)
        try:
            declared = int(request.headers.get("content-length", "0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return JSONResponse({"error": "Invalid Content-Length"}, status_code=400)
        if declared > LIMITS["request_bytes"]:
            return JSONResponse({"error": "Request is too large (maximum 64 KiB)"}, status_code=413)
        if pending >= LIMITS["concurrent_requests"]:
            return JSONResponse({"error": "The simulator is busy. Try again in a few seconds."}, status_code=503, headers={"Retry-After": "5"})
        pending += 1
        submitted = False

        async def read_body():
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > LIMITS["request_bytes"]:
                    raise HTTPException(413, "Request is too large (maximum 64 KiB)")
                body.extend(chunk)
            return body

        def simulate_response(config):
            # Encoding a large trace is CPU work too. Keep it off the event loop
            # so health checks and other visitors can still receive responses.
            result = simulate(config)
            return JSONResponse({"snapshot": result, "provenance": provenance})

        def finished(_future):
            nonlocal pending
            pending -= 1

        try:
            try:
                body = await asyncio.wait_for(read_body(), timeout=LIMITS["request_timeout_seconds"])
            except asyncio.TimeoutError:
                return JSONResponse({"error": "Request body timed out. Please try again."}, status_code=408)
            except ClientDisconnect:
                return JSONResponse({"error": "Request body was interrupted"}, status_code=400)
            config = validate_config(json.loads(body))
            future = asyncio.get_running_loop().run_in_executor(executor, simulate_response, config)
            submitted = True
            # A visitor disconnect must not free a slot while its CPU job still runs.
            future.add_done_callback(finished)
            return await asyncio.shield(future)
        except HTTPException as error:
            return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        except Exception:
            LOGGER.exception("Simulation failed")
            return JSONResponse({"error": "Simulation failed. Try a smaller run or different settings."}, status_code=500)
        finally:
            if not submitted:
                pending -= 1

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def missing_api(path: str):
        return JSONResponse({"error": "Unknown API endpoint"}, status_code=404)

    directory = static_dir or os.environ.get("DYNAMICS_LAB_STATIC_DIR")
    if directory:
        app.mount("/", StaticFiles(directory=directory, html=True), name="website")
    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8012")))
    parser.add_argument("--static-dir", help="Built website directory to serve at /")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.static_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
