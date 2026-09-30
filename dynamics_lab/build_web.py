"""Build the static playground using the lab's renderer and real PyTorch engine.

Run ``python -m dynamics_lab.build_web --output dist/dynamics-lab``.
Put exported scene bundles in ``dynamics_lab/web/published`` to publish frozen
interactive figures without recomputing their trajectories on future builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

import torch

from dynamics_lab.engine import defaults_payload
from dynamics_lab.web_simulation import simulate


LAB = Path(__file__).resolve().parent
SCENES = (
    ("curved-valley", "Curved valley", "Follow a winding valley. Compare how momentum and mass change the path.", "curved_valley_2d.json"),
    ("sloped-ravine", "Sloped ravine", "Watch the optimizers cross the walls of a narrow, tilted ravine.", "sloped_ravine_2d.json"),
    ("quadratic-ravine", "Quadratic ravine", "One direction is steeper than the other. See which methods settle and which keep oscillating.", "anisotropic_quadratic_ravine_2d.json"),
    ("noise-ensemble", "Noisy starts", "The same optimizer starts at 25 different points under heavy-tailed gradient noise.", "noise_chatter_ensemble_2d.json"),
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def provenance() -> dict[str, Any]:
    root = LAB.parent
    digest = hashlib.sha256()
    sources = sorted((root / "massive_lion").rglob("*.py")) + sorted(LAB.glob("*.py"))
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    return {
        "repository": "https://github.com/hadivafaii/massive-lion",
        "commit": git("rev-parse", "HEAD"),
        "source_sha256": digest.hexdigest(),
        "uncommitted_changes": bool(git("status", "--porcelain", "--", "massive_lion", "dynamics_lab")),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
    }


def scene_bundle(scene_id: str, title: str, description: str, payload: dict, source: dict) -> dict:
    config = payload if "config" in payload else {"schema_version": 2, "config": payload}
    if config.get("schema_version", 1) != 2:
        raise ValueError("Export this configuration from the current lab first (schema_version 2 required).")
    snapshot = simulate(config)
    return {"schema_version": 1, "id": scene_id, "title": title,
            "description": description, "config": config, "snapshot": snapshot,
            "provenance": source}


def validate_bundle(bundle: dict) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Published scene must be a version 1 scene bundle from Download scene.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", str(bundle.get("id", ""))):
        raise ValueError("Scene id must use lowercase letters, digits, and hyphens.")
    if not isinstance(bundle.get("title"), str) or not bundle["title"].strip():
        raise ValueError("Scene needs a title.")
    snapshot = bundle.get("snapshot", {})
    if not isinstance(snapshot, dict) or not snapshot.get("learners") or "landscape" not in snapshot:
        raise ValueError("Scene needs complete trajectories and its landscape.")
    if not snapshot.get("done"):
        raise ValueError("Finish the simulation before publishing the scene.")
    if not isinstance(bundle.get("config"), dict):
        raise ValueError("Scene needs its source configuration.")
    json.dumps(bundle, allow_nan=False)


def hosted_html() -> str:
    source = (LAB / "interactive.html").read_text()
    # Keep the local lab's controls and plotting code as the only renderer.
    # The hosted adapter boots from saved data instead of the local step API.
    source, count = re.subn(r"\(async function init\(\) \{.*?\}\)\(\)(?:\.catch\(reportSimulationError\))?;",
                            "/* Bootstrapped by web/site.js. */", source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("Could not locate local lab bootstrap; update the hosted adapter.")
    source = source.replace('href="/favicon', 'href="./favicon')
    source = re.sub(r"<title>.*?</title>", "<title>Optimizer Dynamics Lab</title>", source, count=1)
    source = source.replace("</head>", '<meta name="description" content="Explore optimizer trajectories on interactive loss landscapes. Powered by the actual Massive Lion PyTorch implementations.">\n<link rel="stylesheet" href="./site.css">\n</head>')
    return source.replace("</body>", '<script src="./site.js"></script>\n</body>')


def build(output: Path, api_base_url: str = "", published: Path | None = None) -> dict:
    if api_base_url and api_base_url != ".":
        url = urlparse(api_base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("API URL must be an HTTP(S) URL, '.' for same-origin preview, or empty.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(hosted_html())
    (output / ".nojekyll").touch()
    for filename in ("site.js", "site.css"):
        shutil.copyfile(LAB / "web" / filename, output / filename)
    for filename in ("favicon.svg", "favicon.ico"):
        if (LAB / filename).exists():
            shutil.copyfile(LAB / filename, output / filename)
    write_json(output / "defaults.json", defaults_payload())
    write_json(output / "site-config.json", {"api_base_url": api_base_url,
               "title": "Optimizer Dynamics Lab"})
    source = provenance()
    bundles = []
    for scene_id, title, description, filename in SCENES:
        print(f"Simulating {title}…", flush=True)
        payload = json.loads((LAB / "presets" / filename).read_text())
        bundles.append(scene_bundle(scene_id, title, description, payload, source))
    for path in sorted((published or LAB / "web" / "published").glob("*.json")):
        bundle = json.loads(path.read_text())
        validate_bundle(bundle)
        if any(item["id"] == bundle["id"] for item in bundles):
            raise ValueError(f"Duplicate scene id: {bundle['id']}; choose a unique published scene id.")
        bundles.append(bundle)
    manifest = {"schema_version": 1, "default_scene": "curved-valley", "scenes": []}
    for bundle in bundles:
        filename = f"scenes/{bundle['id']}.json"
        write_json(output / filename, bundle)
        manifest["scenes"].append({"id": bundle["id"], "title": bundle["title"],
                                    "description": bundle.get("description", ""), "file": filename})
    write_json(output / "scenes.json", manifest)
    write_json(output / "provenance.json", source)
    print(f"Built {len(bundles)} scenes in {output.resolve()}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/dynamics-lab"))
    parser.add_argument("--api-url", default="", help="Public backend URL, or '.' for a same-origin local preview")
    parser.add_argument("--published", type=Path, help="Directory of frozen scene bundles to include")
    parser.add_argument("--publish-config", type=Path, help="Generate a frozen scene bundle from a schema 2 lab export")
    parser.add_argument("--scene-id", help="Unique lowercase slug for --publish-config")
    parser.add_argument("--title", help="Display title for --publish-config")
    parser.add_argument("--description", default="")
    args = parser.parse_args()
    if args.publish_config:
        if not args.scene_id or not args.title:
            parser.error("--publish-config requires --scene-id and --title")
        payload = json.loads(args.publish_config.read_text())
        bundle = scene_bundle(args.scene_id, args.title, args.description, payload, provenance())
        validate_bundle(bundle)
        destination = (args.published or LAB / "web" / "published") / f"{args.scene_id}.json"
        if destination.exists():
            parser.error(f"{destination} already exists. Use a new id to preserve published figures.")
        write_json(destination, bundle)
        print(f"Published scene bundle: {destination}")
    build(args.output, args.api_url, args.published)


if __name__ == "__main__":
    main()
