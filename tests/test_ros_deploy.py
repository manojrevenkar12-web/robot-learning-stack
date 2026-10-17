"""Deployment smoke test: an (untrained) exported policy drives the robot over ROS 2 in lockstep.

Checks the runtime contract, not task success: every tick gets a command, the
policy acts on fresh synchronized observations, and latency is reported."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.ros, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]


def test_policy_runtime_over_ros(tmp_path: Path) -> None:
    from rls.learn.export import export_onnx
    from rls.learn.policies import Normalizer, Policy, PolicyBundle
    from rls.learn.text import MAX_TOKENS  # noqa: F401
    from rls.ros.session import record_session
    from rls.sim.env import sample_episode
    from rls.sim.variants import LAB

    model = Policy.build("chunk_transformer", {"d_model": 32, "heads": 2, "enc_layers": 1, "dec_layers": 1, "chunk": 4, "cnn_width": 8})
    norm = Normalizer(np.zeros(8), np.ones(8), np.r_[[0, 0, 0, -2.2, 0, 2.2, 0.785], 1.0], np.full(8, 0.05))
    PolicyBundle(model, norm, {"run_name": "smoke"}).save(tmp_path / "bundle")
    export_onnx(tmp_path / "bundle", tmp_path / "policy.onnx", check_samples=2)
    specs = tmp_path / "specs.jsonl"
    specs.write_text(sample_episode(123, LAB, "pick_place").to_json() + "\n")
    out = tmp_path / "deploy"
    record_session(
        out,
        1,
        ROOT / "configs" / "faults" / "realistic.yaml",
        episode_specs=specs,
        operator="policy",
        record=False,
        policy_params={"bundle": str(tmp_path / "bundle"), "onnx": str(tmp_path / "policy.onnx"), "clock_offsets": json.dumps({"cam_wrist": 0.12})},
        driver_params={"max_steps": 60, "command_timeout_s": 2.0},
    )
    rt = json.loads((out / "policy_runtime.json").read_text())
    gt = json.loads((out / "ground_truth.json").read_text())
    assert gt["episodes"][0]["outcome"]["steps"] >= 59
    assert rt["acted"] >= 50 and rt["stale"] <= 5
    assert rt["infer_ms_p95"] < 200
