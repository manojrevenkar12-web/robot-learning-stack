"""Release-readiness gate for a policy candidate (and the data path that produced it).

    # gate an existing checkpoint against the committed baseline
    python scripts/release_check.py --candidate runs/act_main/best --out reports/release/act_main
    # nightly: rebuild a small dataset over ROS 2, train briefly, gate the result
    python scripts/release_check.py --quick --out reports/nightly
    # accept the current candidate as the new baseline (after review)
    python scripts/release_check.py --candidate runs/act_main/best --out reports/release/act_main --update-baseline

Checks, each pass/fail with the measured value and threshold:
  data.*      a freshly recorded ROS 2 session (realistic faults): drop rates, corrupt
              frames and camera clock offset recovered by ingest vs the injected truth
  export.*    ONNX parity and inference latency
  closed_loop success on fixed benchmark seeds vs the baseline. A regression fails the
              gate only when it is both large (>= --max-drop) and significant (one-sided
              two-proportion z-test, p < 0.05), so benchmark noise alone does not block a release.
Writes RELEASE_CHECK.md and release_check.json; exit code 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / "configs" / "release" / "baseline.json"


def check(name: str, value: float, ok: bool, threshold: str) -> dict[str, Any]:
    return {"check": name, "value": value, "pass": bool(ok), "threshold": threshold}


def z_regression(k_base: int, n_base: int, k_new: int, n_new: int) -> float:
    """One-sided p-value for 'candidate success rate < baseline success rate'."""
    p = (k_base + k_new) / (n_base + n_new)
    se = math.sqrt(max(p * (1 - p) * (1 / n_base + 1 / n_new), 1e-12))
    z = (k_base / n_base - k_new / n_new) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


def data_checks(work: Path, episodes: int) -> list[dict[str, Any]]:
    from rls.data.ingest import ingest_session
    from rls.ros.session import record_session

    sess = work / "raw" / "check_session"
    if not (sess / "session.json").exists():
        record_session(sess, episodes, ROOT / "configs" / "faults" / "realistic.yaml", seed_start=8000, fault_seed=77)
    rep = ingest_session(sess / "bag", work / "staging" / "check_session")
    truth = json.loads((sess / "ground_truth.json").read_text())
    drop_err, corrupt_miss = 0.0, 0
    for ep, gt in zip(rep["episodes"], truth["episodes"], strict=True):
        for cam in ("cam_front", "cam_wrist"):
            ev = gt["stream_events"][cam]
            got = ep["checks"][cam]["value"]
            drop_err = max(drop_err, abs(got["drop_rate"] - (ev["dropped"] + ev["outage_dropped"]) / max(ev["captured"], 1)))
            corrupt_miss += abs(sum(got["corrupt"].values()) - ev["corrupted"])
    prof = truth["profile"]["streams"]["cam_wrist"]
    clk = rep["clocks"]["cam_wrist"]
    off_err = abs(clk["offset_s"] - (prof["stamp_offset_s"] + prof["stamp_drift_ppm"] * 1e-6 * clk["t_ref"]))
    healthy = sum(e["status"] != "fail" for e in rep["episodes"]) / max(len(rep["episodes"]), 1)
    return [
        check("data.drop_rate_max_abs_error", round(drop_err, 4), drop_err < 0.02, "< 0.02"),
        check("data.corrupt_frames_missed", corrupt_miss, corrupt_miss == 0, "== 0"),
        check("data.wrist_clock_offset_error_ms", round(off_err * 1e3, 1), off_err < 0.025, "< 25 ms"),
        # depends on the fault profile's own failure rate, not on the pipeline: reported, not gated
        check("data.healthy_episode_fraction", round(healthy, 3), True, "reported (realistic profile fails ~13% of episodes by design)"),
    ]


def export_checks(bundle: Path, onnx: Path) -> list[dict[str, Any]]:
    from rls.learn.export import export_onnx

    rep = export_onnx(bundle, onnx, check_samples=50)
    return [
        check("export.onnx_max_abs_diff", rep["max_abs_diff"], rep["max_abs_diff"] < 1e-4, "< 1e-4"),
        check("export.onnx_p95_ms", round(rep["onnx_p95_ms"], 1), rep["onnx_p95_ms"] < 50, "< 50 ms (15 Hz control budget is 66 ms)"),
    ]


def closed_loop_checks(
    bundle: Path, onnx: Path, out: Path, episodes: int, baseline: dict[str, Any] | None, max_drop: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rls.learn.benchmark import benchmark

    splits = list(baseline["splits"]) if baseline else ["in_distribution", "position_ood"]
    res = benchmark(bundle, splits, episodes, out / "bench", onnx=onnx, videos_per_mode=1)
    measured = {s: {"k": sum(e.success for e in r.episodes), "n": r.n} for s, r in res.items()}
    rows = []
    for s, m in measured.items():
        rate = m["k"] / m["n"]
        if not baseline:
            rows.append(check(f"closed_loop.{s}", round(rate, 3), True, "no baseline yet (recorded only)"))
            continue
        b = baseline["splits"][s]
        p = z_regression(b["k"], b["n"], m["k"], m["n"])
        drop = b["k"] / b["n"] - rate
        rows.append(
            check(
                f"closed_loop.{s}",
                round(rate, 3),
                not (drop >= max_drop and p < 0.05),
                f"baseline {b['k']}/{b['n']}; fail if drop >= {max_drop:.2f} and p < 0.05 (drop {drop:+.2f}, p {p:.3f})",
            )
        )
    return rows, measured


def quick_candidate(work: Path) -> Path:
    """Small end-to-end build: 2 ROS sessions + 60 synthetic episodes, 800 training steps."""
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    subprocess.run(
        [py, "scripts/build_dataset.py", "--root", str(work), "--sessions", "2", "--stress-sessions", "0", "--per-session", "8", "--synthetic", "60"],
        check=True,
        cwd=ROOT,
        env=env,
    )
    cfg = work / "quick.yaml"
    cfg.write_text(
        f"run_name: quick\ndataset: {work / 'release' / 'rls_panda'}\npolicy: chunk_transformer\npolicy_config: {{chunk: 20, action_space: ee, augment_shift: 2, aux_object_weight: 1.0, state_noise: 0.1, state_dropout: 0.3}}\n"
        f"steps: 800\nbatch_size: 32\neval_every: 200\nout_dir: {work / 'runs'}\n"
    )
    subprocess.run([py, "-m", "rls.learn.train", str(cfg)], check=True, cwd=ROOT, env=env)
    return work / "runs" / "quick" / "best"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, help="policy bundle to gate (default with --quick: train one)")
    ap.add_argument("--quick", action="store_true", help="nightly mode: build data + train a small candidate first")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--data-episodes", type=int, default=6)
    ap.add_argument("--max-drop", type=float, default=0.10)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    key = "quick" if a.quick else "full"
    baselines = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    baseline = baselines.get(key)
    candidate = a.candidate or (quick_candidate(a.out / "work") if a.quick else None)
    if candidate is None:
        ap.error("--candidate is required unless --quick")

    rows: list[dict[str, Any]] = []
    if not a.skip_data:
        rows += data_checks(a.out / "work", a.data_episodes)
    onnx = a.out / "policy.onnx"
    rows += export_checks(candidate, onnx)
    cl, measured = closed_loop_checks(candidate, onnx, a.out, a.episodes, baseline, a.max_drop)
    rows += cl
    ok = all(r["pass"] for r in rows)
    meta = json.loads((candidate / "bundle.json").read_text()).get("meta", {}) if (candidate / "bundle.json").exists() else {}
    result = {
        "candidate": str(candidate),
        "policy": meta,
        "mode": key,
        "pass": ok,
        "checks": rows,
        "measured": measured,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (a.out / "release_check.json").write_text(json.dumps(result, indent=1, default=str))
    md = [
        f"# Release check: {'PASS' if ok else 'FAIL'}",
        "",
        f"Candidate `{candidate}` (run `{meta.get('run_name')}`, dataset `{meta.get('dataset_version')}`), mode `{key}`.",
        "",
        "| check | value | threshold | result |",
        "|---|---|---|---|",
    ]
    md += [f"| {r['check']} | {r['value']} | {r['threshold']} | {'pass' if r['pass'] else '**FAIL**'} |" for r in rows]
    (a.out / "RELEASE_CHECK.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if a.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        baselines[key] = {"candidate": str(candidate), "run_name": meta.get("run_name"), "dataset_version": meta.get("dataset_version"), "splits": measured}
        BASELINE.write_text(json.dumps(baselines, indent=1) + "\n")
        print(f"baseline[{key}] updated -> {BASELINE}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
