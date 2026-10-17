"""End-to-end over ROS 2: record a session with injected faults, ingest it, check every
fault the driver injected is found (and nothing else), then curate and release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ros, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stress_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rls.ros.session import record_session

    out = tmp_path_factory.mktemp("rec") / "stress"
    record_session(out, 4, ROOT / "configs" / "faults" / "stress.yaml", seed_start=7000, fault_seed=11)
    return out


def test_health_checks_match_injected_faults(stress_session: Path, tmp_path: Path) -> None:
    from rls.data.ingest import ingest_session

    rep = ingest_session(stress_session / "bag", tmp_path / "staging")
    truth = json.loads((stress_session / "ground_truth.json").read_text())
    assert len(rep["episodes"]) == len(truth["episodes"]) == 4
    for ep, gt in zip(rep["episodes"], truth["episodes"], strict=True):
        for cam in ("cam_front", "cam_wrist"):
            ev = gt["stream_events"][cam]
            true_drop = (ev["dropped"] + ev["outage_dropped"]) / max(ev["captured"], 1)
            got = ep["checks"][cam]["value"]
            assert abs(got["drop_rate"] - true_drop) < 0.02, (cam, got, ev)
            assert sum(got["corrupt"].values()) == ev["corrupted"], (cam, got, ev)
    # wrist camera clock: stress profile injects -180 ms and -1500 ppm
    clk = rep["clocks"]["cam_wrist"]
    true_off = -0.180 - 1500e-6 * clk["t_ref"]
    assert abs(clk["offset_s"] - true_off) < 0.025


def test_truncated_bag_recovers_complete_episodes(stress_session: Path, tmp_path: Path) -> None:
    from rls.data.ingest import ingest_session

    bag = tmp_path / "bag"
    bag.mkdir()
    data = next((stress_session / "bag").glob("*.mcap")).read_bytes()
    (bag / "cut.mcap").write_bytes(data[: int(len(data) * 0.6)])
    rep = ingest_session(bag, tmp_path / "staging", {"session": "cut"})
    statuses = [e["checks"].get("episode_complete", {}).get("status", "ok") for e in rep["episodes"]]
    assert statuses[-1] == "fail" and all(s == "ok" for s in statuses[:-1])
    assert any(e.get("path") for e in rep["episodes"][:-1])


def test_ingest_under_limited_compute(stress_session: Path, tmp_path: Path) -> None:
    """One thread and a 3 GB address-space cap: the ingest must still finish."""
    import resource
    import subprocess
    import sys

    code = (
        "import resource, sys; from pathlib import Path;"
        "resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30));"
        "from rls.data.ingest import ingest_session;"
        f"r = ingest_session(Path('{stress_session / 'bag'}'), Path('{tmp_path / 's'}'));"
        "print(len(r['episodes']))"
    )
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    import os

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, **env}, cwd=ROOT, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("4")
    del resource


def test_live_cpp_monitor_matches_injected_faults(stress_session: Path) -> None:
    """The C++ sync monitor (cpp/rls_sync_monitor) sees the same faults live, during recording."""
    from rls.ros.session import SYNC_MONITOR

    if not SYNC_MONITOR.exists():
        pytest.skip("sync_monitor not built (cpp/rls_sync_monitor)")
    mon = json.loads((stress_session / "sync_monitor.json").read_text())
    truth = json.loads((stress_session / "ground_truth.json").read_text())
    assert mon["episodes"] == len(truth["episodes"])
    for stream in ("joint_states", "wrist_ft", "cam_front", "cam_wrist"):
        ev = [e["stream_events"][stream] for e in truth["episodes"]]
        true_drop = sum(e["dropped"] + e["outage_dropped"] for e in ev) / sum(e["captured"] for e in ev)
        assert abs(mon["streams"][stream]["drop_rate"] - true_drop) < 0.03, (stream, mon["streams"][stream], true_drop)
    # stress profile: wrist camera clock 180 ms behind -> its frames look old on arrival and the cameras disagree
    assert mon["streams"]["cam_wrist"]["median_age_s"] > 0.15
    assert mon["streams"]["cam_wrist"]["level"] >= 1
    assert mon["median_wrist_minus_front_s"] < -0.1
