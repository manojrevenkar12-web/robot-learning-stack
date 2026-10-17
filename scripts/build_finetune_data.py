"""Small adaptation dataset for the fine-tuning study: two object colours the main policy never saw.

    python scripts/build_finetune_data.py --root data --episodes 48

Writes staging/finetune_unseen_color/, curated_unseen_color/ and
release/rls_unseen_color/ (LeRobot v3.0). Seeds 7000+ are disjoint from the
main dataset (1000-5999) and from benchmark seeds (900000+).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data")
    ap.add_argument("--episodes", type=int, default=48)
    a = ap.parse_args()

    from rls.data.curate import CurationConfig, curate, scan, write_outputs
    from rls.data.synthetic import generate
    from rls.sim.variants import SPLITS, with_tasks

    tasks = ("pick_place", "stack")
    # lab appearance, unseen target colours (same factor the unseen_color benchmark split changes)
    variant = with_tasks(replace(SPLITS["unseen_color"], name="lab_unseen_color"), tasks)
    stage = a.root / "staging_finetune" / "unseen_color"
    have = len(list(stage.glob("*.h5"))) if stage.exists() else 0
    if have < a.episodes:
        generate(stage, variant, range(7000 + have, 7000 + a.episodes), tasks=tasks, log_every=10)
    df, rep = curate(scan([stage]), CurationConfig.load(ROOT / "configs" / "curation" / "default.yaml"))
    manifest = write_outputs(df, rep, a.root / "curated_unseen_color")
    print(json.dumps({k: rep[k] for k in ("version", "scanned", "kept", "dropped_by_reason", "splits")}))
    py = os.environ.get("RLS_LEROBOT_PYTHON", "/opt/lerobot-venv/bin/python")
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    subprocess.run(
        [py, "-m", "rls.data.export_lerobot", "--manifest", str(manifest), "--out", str(a.root / "release" / "rls_unseen_color")], check=True, env=env, cwd=ROOT
    )


if __name__ == "__main__":
    main()
