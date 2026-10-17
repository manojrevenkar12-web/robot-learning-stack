"""Collect every measured number into reports/RESULTS.json (the README tables are built from it).

    python scripts/collect_results.py --data data --runs runs --reports reports

Data-health numbers compare what ingest (offline) and the C++ sync monitor (online)
reported against the fault model's ground truth, which neither of them reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def data_health(data: Path) -> dict[str, Any]:
    raw, staging = data / "raw", data / "staging" / "teleop"
    drop_err: dict[str, list[float]] = {"cam_front": [], "cam_wrist": []}
    mon_err: dict[str, list[float]] = {k: [] for k in ("joint_states", "wrist_ft", "cam_front", "cam_wrist")}
    corrupt = {"injected": 0, "detected": 0}
    clock_err_ms, drift_err_ppm, clocks = [], [], []
    calib = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    status: dict[str, int] = {}
    episodes = sessions = 0
    for sess in sorted(p for p in raw.iterdir() if (p / "ground_truth.json").exists()):
        gt = _load(sess / "ground_truth.json")
        rep = _load(staging / sess.name / f"ingest_report_{sess.name}.json")
        if rep is None:
            continue
        sessions += 1
        prof = gt["profile"]["streams"]
        for cam in ("cam_front", "cam_wrist"):
            c = rep["clocks"].get(cam)
            if c:
                true_off = prof[cam].get("stamp_offset_s", 0.0) + prof[cam].get("stamp_drift_ppm", 0.0) * 1e-6 * c["t_ref"]
                clock_err_ms.append(abs(c["offset_s"] - true_off) * 1e3)
                if cam == "cam_wrist":
                    drift_err_ppm.append(abs(c["drift_ppm"] - prof[cam].get("stamp_drift_ppm", 0.0)))
                    clocks.append(
                        {
                            "session": sess.name,
                            "est_ms": round(c["offset_s"] * 1e3, 1),
                            "true_ms": round(true_off * 1e3, 1),
                            "est_ppm": round(c["drift_ppm"]),
                            "true_ppm": prof[cam].get("stamp_drift_ppm", 0.0),
                            "reliable": c["reliable"],
                        }
                    )
        for e, g in zip(rep["episodes"], gt["episodes"], strict=False):
            episodes += 1
            status[e["status"]] = status.get(e["status"], 0) + 1
            ch = e["checks"]
            for cam in ("cam_front", "cam_wrist"):
                if cam not in ch:
                    continue
                ev = g["stream_events"][cam]
                truth = (ev["dropped"] + ev["outage_dropped"]) / max(ev["captured"], 1)
                drop_err[cam].append(abs(ch[cam]["value"]["drop_rate"] - truth))
                corrupt["injected"] += ev["corrupted"]
                corrupt["detected"] += sum(ch[cam]["value"]["corrupt"].values())
            if "front_extrinsics" in ch:
                bad = g.get("calibration_error") is not None
                flagged = ch["front_extrinsics"]["status"] != "ok"
                calib["tp" if bad and flagged else "fn" if bad else "fp" if flagged else "tn"] += 1
        mon = _load(sess / "sync_monitor.json")
        if mon:
            for k in mon_err:
                ev = [g["stream_events"][k] for g in gt["episodes"]]
                truth = sum(e["dropped"] + e["outage_dropped"] for e in ev) / max(sum(e["captured"] for e in ev), 1)
                mon_err[k].append(abs(mon["streams"][k]["drop_rate"] - truth))
    pct = lambda xs, q: round(float(np.percentile(xs, q)), 4) if xs else None  # noqa: E731
    return {
        "sessions": sessions,
        "episodes": episodes,
        "health_status": status,
        "drop_rate_abs_error": {k: {"median": pct(v, 50), "max": pct(v, 100)} for k, v in drop_err.items()},
        "corrupt_frames": corrupt,
        "clock_offset_error_ms": {"median": pct(clock_err_ms, 50), "p90": pct(clock_err_ms, 90), "max": pct(clock_err_ms, 100)},
        "wrist_drift_error_ppm": {"median": pct(drift_err_ppm, 50), "max": pct(drift_err_ppm, 100)},
        "wrist_clocks": clocks,
        "extrinsics": calib,
        "live_monitor_drop_rate_abs_error": {k: {"median": pct(v, 50), "max": pct(v, 100)} for k, v in mon_err.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    a = ap.parse_args()
    out: dict[str, Any] = {"data_health": data_health(a.data)}
    out["curation"] = _load(a.data / "curated" / "curation_report.json")
    out["release"] = _load(a.data / "release" / "rls_panda" / "meta" / "rls_release.json")
    out["runs"] = {p.parent.name: _load(p) for p in sorted(a.runs.glob("*/summary.json"))}
    out["onnx"] = {p.parent.name: _load(p) for p in sorted(a.runs.glob("*/onnx_export.json"))}
    out["benchmarks"] = {}
    for p in sorted((a.reports / "bench").glob("*/benchmark.json")):
        b = _load(p)
        out["benchmarks"][p.parent.name] = {
            "conditions": b["conditions"],
            "splits": {k: {kk: v[kk] for kk in ("n", "success_rate", "ci95", "failures", "by_task")} for k, v in b["splits"].items()},
        }
    out["rl"] = _load(a.runs / "rl" / "rl_results.json")
    out["codec"] = _load(a.reports / "codec" / "codec_study.json")
    out["perception"] = _load(a.reports / "perception" / "perception_eval.json")
    out["finetune_data"] = _load(a.data / "release" / "rls_unseen_color" / "meta" / "rls_release.json")
    out["deploy"] = _load(a.reports / "deploy" / "deploy_eval.json")
    out["agent"] = _load(a.reports / "agent" / "agent_eval.json")
    out["release_check"] = _load(a.reports / "release" / "act_main" / "release_check.json")
    path = a.reports / "RESULTS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["data_health"], indent=1))
    print(path)


if __name__ == "__main__":
    main()
