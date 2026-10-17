"""Release a curated manifest as a LeRobot v3.0 dataset, using LeRobot's own writer.

The format is owned upstream, so we do not re-implement it: this module runs
in an environment with ``lerobot`` installed (it pins an older torch than the
training stack, hence a separate interpreter; see ``RLS_LEROBOT_PYTHON``) and
calls ``LeRobotDataset.create / add_frame / save_episode / finalize``.

Episodes are written train-first, so ``info.json`` can carry contiguous
``train`` / ``val`` splits. Episode-level provenance that LeRobot has no column
for (source, staging id, health, outcome) goes into ``meta/rls_episodes.parquet``.

    $RLS_LEROBOT_PYTHON -m rls.data.export_lerobot --manifest data/curated/manifest.parquet --out data/release/rls_panda
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rls.data.schema import ACTION_NAMES, STATE_NAMES, WRENCH_NAMES, read_episode


def features(front_hw: tuple[int, int], wrist_hw: tuple[int, int]) -> dict:
    img = ["height", "width", "channels"]
    return {
        "observation.images.front": {"dtype": "video", "shape": (*front_hw, 3), "names": img},
        "observation.images.wrist": {"dtype": "video", "shape": (*wrist_hw, 3), "names": img},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "observation.wrench": {"dtype": "float32", "shape": (6,), "names": WRENCH_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
        "frame_flags": {"dtype": "int64", "shape": (1,), "names": None},
    }


def export(manifest_path: Path, out: Path, repo_id: str = "local/rls_panda", vcodec: str = "libsvtav1", crf: int = 30) -> dict:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    t0 = time.time()
    man = pd.read_parquet(manifest_path)
    man["split_order"] = man["split"].map({"train": 0, "val": 1})
    man = man.sort_values(["split_order", "source", "episode_id"]).reset_index(drop=True)
    first = read_episode(Path(man.loc[0, "path"]))
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=first.fps,
        features=features(first.front.shape[1:3], first.wrist.shape[1:3]),
        root=out,
        robot_type="franka_panda_sim",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec, crf=crf),
    )
    rows = []
    for i, r in man.iterrows():
        ep = read_episode(Path(r["path"]))
        flags = ep.frame_flags if ep.frame_flags is not None else np.zeros(ep.length, np.uint8)
        for k in range(ep.length):
            ds.add_frame(
                {
                    "observation.images.front": ep.front[k],
                    "observation.images.wrist": ep.wrist[k],
                    "observation.state": ep.state[k].astype(np.float32),
                    "observation.wrench": ep.wrench[k].astype(np.float32),
                    "action": ep.action[k].astype(np.float32),
                    "frame_flags": np.array([int(flags[k])], dtype=np.int64),
                    "task": ep.instruction,
                }
            )
        ds.save_episode()
        rows.append(
            {
                "episode_index": int(i),
                "episode_id": ep.episode_id,
                "source": ep.source,
                "split": r["split"],
                "task_family": ep.task,
                "health": r["health"],
                "calibration_suspect": bool(r["calibration_suspect"]),
                "failure_mode": r["failure_mode"],
                "content_hash": r["content_hash"],
                "staging_path": r["path"],
                "provenance": json.dumps(ep.provenance, default=str),
                "episode_spec": json.dumps(ep.episode_spec, default=str),  # scene layout: objects, target, support, bin
            }
        )
    ds.finalize()
    side = pd.DataFrame(rows)
    side.to_parquet(out / "meta" / "rls_episodes.parquet", index=False)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    n_train = int((side["split"] == "train").sum())
    info["splits"] = {"train": f"0:{n_train}", "val": f"{n_train}:{len(side)}"}
    info_path.write_text(json.dumps(info, indent=4))
    report = json.loads((manifest_path.parent / "curation_report.json").read_text()) if (manifest_path.parent / "curation_report.json").exists() else {}
    card = {
        "dataset_version": report.get("version", "unversioned"),
        "episodes": len(side),
        "frames": int(info["total_frames"]),
        "splits": info["splits"],
        "by_source": side["source"].value_counts().to_dict(),
        "video_codec": vcodec,
        "crf": crf,
        "export_seconds": round(time.time() - t0, 1),
        "lerobot_codebase_version": info.get("codebase_version"),
    }
    (out / "meta" / "rls_release.json").write_text(json.dumps(card, indent=1))
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/rls_panda")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--crf", type=int, default=30)
    a = ap.parse_args()
    print(json.dumps(export(a.manifest, a.out, a.repo_id, a.vcodec, a.crf), indent=1))


if __name__ == "__main__":
    main()
