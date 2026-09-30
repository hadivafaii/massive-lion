"""Saved Lab configurations keep optimizer instances distinct in video export."""

import copy
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from dynamics_lab.make_video import (
    LandscapeRenderer, apply_color_overrides, config_from_payload,
    default_output_path, run_to_endpoint, snapshot_at_step,
)
from dynamics_lab.landscapes import coerce_config, sample_landscape


def duplicate_payload():
    return {
        "schema_version": 2,
        "config": {
            "mode": "parallel", "max_steps": 3, "theta0": [-1.8, 2.3],
            "ensemble_count": 2, "ensemble_instance_id": "lion-fast",
            "ensemble_optimizer": "Lion", "noise": {"std": 0},
            "optimizers": [
                {"name": "Lion", "instance_id": "lion-slow", "lr": .02,
                 "label": "Slow custom", "color": "#c1121f"},
                {"name": "Lion", "instance_id": "lion-fast", "lr": .08,
                 "display_label": "Lion · lr=0.08", "color": "#003e90"},
            ],
        },
    }


def test_video_config_preserves_instances_without_mutating_export():
    payload = duplicate_payload()
    original = copy.deepcopy(payload)
    config = config_from_payload(payload, mode="ensemble")
    assert config["optimizers"] == payload["config"]["optimizers"]
    assert config["ensemble_instance_id"] == "lion-fast"
    config["optimizers"][0]["label"] = "Edited copy"
    assert payload == original


def test_explicit_ensemble_overrides_and_output_names():
    payload = duplicate_payload()
    by_type = config_from_payload(payload, mode="ensemble", ensemble_optimizer="SGD")
    assert "ensemble_instance_id" not in by_type
    assert by_type["ensemble_optimizer"] == "SGD"
    by_id = config_from_payload(payload, mode="ensemble", ensemble_optimizer="SGD",
                                ensemble_instance_id="lion-slow")
    assert by_id["ensemble_instance_id"] == "lion-slow"
    assert default_output_path(Path("comparison.json"), by_id, None).name == "comparison_ensemble_lion-slow.mp4"
    assert default_output_path(Path("comparison.json"), by_type, None).name == "comparison_ensemble_SGD.mp4"


@pytest.mark.parametrize("mode", ["parallel", "serial", "ensemble"])
def test_video_simulation_keeps_labels_colors_and_selected_instances(mode):
    config = config_from_payload(duplicate_payload(), mode=mode)
    final = run_to_endpoint(config)
    learners = final["learners"]
    assert len(learners) == 2
    assert len({item["id"] for item in learners}) == 2
    if mode == "ensemble":
        assert {item["instance_id"] for item in learners} == {"lion-fast"}
        assert all(item["lr"] == .08 and item["color"] == "#003e90" for item in learners)
    else:
        assert [item["instance_id"] for item in learners] == ["lion-slow", "lion-fast"]
        assert [item["color"] for item in learners] == ["#c1121f", "#003e90"]
        assert learners[0]["name"] == "Slow custom"
        assert learners[0]["label"] == "Slow custom"
        assert learners[1]["name"] == "Lion · lr=0.08"
        assert learners[1]["label"] is None
        assert learners[0]["trace"][-1]["theta"] != learners[1]["trace"][-1]["theta"]
    frame = snapshot_at_step(final, final["global_step"])
    assert [item["name"] for item in frame["learners"]] == [item["name"] for item in learners]


def test_instance_colors_and_legacy_colors_survive_serial_frame_filtering():
    config = config_from_payload(duplicate_payload(), mode="serial")
    final = run_to_endpoint(config)
    frame = snapshot_at_step(final, final["global_step"], serial_history="replace")
    assert len(frame["learners"]) == 1
    legacy = apply_color_overrides(frame, {"color_overrides": {"learner-1": "#abcdef"}})
    assert legacy["learners"][0]["color"] == "#abcdef"
    current = apply_color_overrides(frame, {"color_overrides": {
        "lion-fast": "#fedcba", "learner-1": "#abcdef"}})
    assert current["learners"][0]["color"] == "#fedcba"
    assert frame["learners"][0]["color"] == "#003e90"


@pytest.mark.parametrize("view", ["2d", "3d"])
def test_video_renders_both_same_family_trajectories_in_their_colors(view):
    config = config_from_payload(duplicate_payload())
    final = run_to_endpoint(config)
    landscape_config = coerce_config(config.get("landscape"))
    renderer = LandscapeRenderer(
        config=config, controls={}, landscape=sample_landscape(landscape_config, n=11),
        landscape_config=landscape_config, view=view, dpi=40, figsize=(3.2, 2.4),
        final_snapshot=final,
    )
    fig, ax = getattr(renderer, f"_draw_{view}")(final)
    paths = [item for item in ax.collections if isinstance(item, LineCollection)]
    assert len(paths) == 2
    colors = {tuple(item.get_colors()[0, :3]) for item in paths}
    assert colors == {to_rgb("#c1121f"), to_rgb("#003e90")}
    fig.canvas.draw()
    assert np.asarray(fig.canvas.buffer_rgba()).shape == (96, 128, 4)
    plt.close(fig)
