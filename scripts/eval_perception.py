"""Accuracy of the RGB-D cube detector used for skill postconditions, against simulator truth.

    python scripts/eval_perception.py --scenes 200 --out reports/perception

Renders the front RGB-D camera for random scenes (held-out seeds), detects every
cube by colour and compares the back-projected position with the true cube
centre in x/y (the detector sees the top surface, so z is reported separately).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rls.perception.cubes import detect
from rls.sim.env import PandaEnv, sample_episode
from rls.sim.variants import SPLITS, with_tasks


def mj_cam_to_optical(mat: np.ndarray) -> np.ndarray:
    return mat @ np.diag([1.0, -1.0, -1.0])  # same convention as the ROS driver's /tf_static


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--splits", default="in_distribution,lighting_shift,two_distractors")
    ap.add_argument("--out", type=Path, default=Path("reports/perception"))
    a = ap.parse_args()
    env = PandaEnv(render=True, depth=True)
    rows = []
    for split in a.splits.split(","):
        v = with_tasks(SPLITS[split], ("pick_place", "stack"))
        for i in range(a.scenes // len(a.splits.split(","))):
            spec = sample_episode(800_000 + i, v, "pick_place" if i % 2 == 0 else "stack")
            obs = env.reset(spec)
            cam = env.camera_params("front")
            det = detect(
                obs["front_rgb"],
                np.round(obs["front_depth"] * 1000).astype(np.uint16),
                cam["K"],
                np.array(cam["cam_pos"]),
                mj_cam_to_optical(np.array(cam["cam_mat"])),
            )
            for o in spec.scene.objects:
                true = env.object_pos(o.name)
                d = det.get(o.color)
                rows.append(
                    {
                        "split": split,
                        "color": o.color,
                        "found": d is not None,
                        "xy_err_mm": float(np.linalg.norm(d.position[:2] - true[:2]) * 1000) if d else None,
                        "z_err_mm": float((d.position[2] - true[2]) * 1000) if d else None,
                    }
                )
    env.close_renderers()
    found = [r for r in rows if r["found"]]
    xy = np.array([r["xy_err_mm"] for r in found])
    res = {
        "objects": len(rows),
        "detected": len(found),
        "xy_err_mm": {"median": round(float(np.median(xy)), 1), "p95": round(float(np.percentile(xy, 95)), 1), "max": round(float(xy.max()), 1)},
        "by_split": {
            s: {
                "detected": sum(r["found"] for r in rows if r["split"] == s),
                "objects": sum(r["split"] == s for r in rows),
                "xy_median_mm": round(float(np.median([r["xy_err_mm"] for r in found if r["split"] == s])), 1),
            }
            for s in a.splits.split(",")
        },
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "perception_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
