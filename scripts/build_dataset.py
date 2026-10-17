"""Reference data workflow: record -> ingest -> synthesize -> curate -> release.

    python scripts/build_dataset.py --root data --sessions 18 --per-session 12 --synthetic 300

Writes, under --root:
    raw/session_XXX/          rosbag2 MCAP + session.json + ground_truth.json (injected faults)
    staging/                  one HDF5 per episode (teleop_ros and synthetic)
    curated/                  manifest.parquet, curation_report.json
    release/rls_panda/        LeRobot v3.0 dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data")
    ap.add_argument("--sessions", type=int, default=18)
    ap.add_argument("--stress-sessions", type=int, default=2)
    ap.add_argument("--per-session", type=int, default=12)
    ap.add_argument("--synthetic", type=int, default=300)
    ap.add_argument("--plant-duplicates", action="store_true", default=True)
    ap.add_argument("--skip-record", action="store_true")
    a = ap.parse_args()
    root = a.root
    raw, staging = root / "raw", root / "staging"
    t0 = time.time()

    from rls.data.curate import CurationConfig, curate, scan, write_outputs
    from rls.data.ingest import ingest_session
    from rls.data.synthetic import generate
    from rls.ros.session import record_session
    from rls.sim.variants import SYNTH_DR, with_tasks

    plan = [(f"session_{i:03d}", "realistic", 1000 + i * a.per_session, i) for i in range(a.sessions)]
    plan += [(f"stress_{i:03d}", "stress", 3000 + i * a.per_session, 100 + i) for i in range(a.stress_sessions)]
    if a.plant_duplicates:
        # an operator accidentally re-records session_000's scenes: near-duplicate episodes
        plan.append(("session_000_rerecorded", "realistic", 1000, 999))
    if not a.skip_record:
        for name, profile, seed_start, fault_seed in plan:
            out = raw / name
            if (out / "session.json").exists():
                continue
            shutil.rmtree(out, ignore_errors=True)
            log(f"recording {name} ({profile}, {a.per_session} episodes)")
            record_session(out, a.per_session, ROOT / "configs" / "faults" / f"{profile}.yaml", seed_start=seed_start, fault_seed=fault_seed)
    for name, *_ in plan:
        dst = staging / "teleop" / name
        if dst.exists() and any(dst.glob("*.h5")):
            continue
        log(f"ingesting {name}")
        rep = ingest_session(raw / name / "bag", dst)
        n_fail = sum(e["status"] == "fail" for e in rep["episodes"])
        log(
            f"  {len(rep['episodes'])} episodes, {n_fail} failed health, clocks {json.dumps({k: (round(v['offset_s'] * 1e3, 1), round(v['drift_ppm'])) for k, v in rep['clocks'].items()})}"
        )

    syn_dir = staging / "synthetic"
    have = len(list(syn_dir.glob("*.h5"))) if syn_dir.exists() else 0
    if have < a.synthetic:
        log(f"generating {a.synthetic - have} synthetic episodes")
        generate(syn_dir, with_tasks(SYNTH_DR, ("pick_place", "stack")), range(5000 + have, 5000 + a.synthetic), tasks=("pick_place", "stack"), log_every=50)
    if a.plant_duplicates:
        # the same synthetic files imported twice under new names: exact duplicates
        for p in sorted(syn_dir.glob("syn-synthetic_dr-*.h5"))[:5]:
            dup = syn_dir / p.name.replace("syn-", "syn-reimport-")
            if not dup.exists():
                shutil.copy(p, dup)
                import h5py

                with h5py.File(dup, "a") as f:
                    f.attrs["episode_id"] = dup.stem

    log("curating")
    man = scan([staging])
    df, report = curate(man, CurationConfig.load(ROOT / "configs" / "curation" / "default.yaml"))
    manifest = write_outputs(df, report, root / "curated")
    log(f"  {json.dumps({k: report[k] for k in ('version', 'scanned', 'kept', 'dropped_by_reason', 'kept_by_source', 'splits')})}")

    py = os.environ.get("RLS_LEROBOT_PYTHON", "/opt/lerobot-venv/bin/python")
    log(f"exporting LeRobot v3.0 with {py}")
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    subprocess.run(
        [py, "-m", "rls.data.export_lerobot", "--manifest", str(manifest), "--out", str(root / "release" / "rls_panda")], check=True, env=env, cwd=ROOT
    )
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
