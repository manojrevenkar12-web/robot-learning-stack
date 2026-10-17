"""Open-loop probes that explain closed-loop failures (see docs/training.md).

    python scripts/probe_policy.py runs/act_main/best --episodes 60

* grounding: at the first frame of an episode (arm at home, nothing has moved yet) the auxiliary
  head's estimate of the instructed cube's position, against the scene layout. If this is poor,
  the policy cannot know where to go until the arm is already there.
* reliance: grasp-position error 12 ticks before the grasp with the real inputs, with the camera
  images swapped for images from other episodes, and with noise on the joint state. A policy that
  barely changes without its images is extrapolating proprioception.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rls.learn.data import AUX_SCALE, ChunkDataset
from rls.learn.lerobot_reader import load
from rls.learn.policies.base import PolicyBundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("data/release/rls_panda"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    b = PolicyBundle.load(a.bundle)
    m = b.model.eval()
    space = str(getattr(m.cfg, "action_space", "joint"))
    d = load(a.dataset)
    rng = np.random.default_rng(0)
    res: dict[str, dict[str, float]] = {}
    for split in ("train", "val"):
        eps = d.split_episodes(split)[: a.episodes]
        ds = ChunkDataset(d, eps, m.cfg.chunk, b.normalizer, None, space)
        ground, real, swapped, noisy = [], [], [], []
        for e in eps:
            rows = d.episode_rows(e)
            first = ds.batch(np.array([np.where(ds.idx.rows == rows[0])[0][0]]))
            with torch.no_grad():
                if getattr(m.cfg, "aux_object_weight", 0) and ds.aux_xy is not None:
                    mem, mask = m.encode(first["front"], first["wrist"], first["state"], first["tokens"])
                    q = m.aux_queries.expand(1, -1, -1)
                    for blk in m.decoder:
                        q = blk(q, ctx=mem, ctx_mask=mask)
                    aux = m.aux_head(m.head_norm(q))[0].numpy()
                    ground.append(float(np.linalg.norm((aux[0] - first["aux_xy"][0, 0].numpy()) * AUX_SCALE) * 1000))
            if space != "ee":
                continue
            acts = ds.action[rows]
            k = int(np.argmax(acts[:, -1] < 0.5))
            k0 = max(0, k - 12)
            bt = ds.batch(np.array([np.where(ds.idx.rows == rows[k0])[0][0]]))
            other = ds.batch(rng.integers(0, len(ds), 1))
            tgt = acts[k, :2]
            j = min(k - k0, m.cfg.chunk - 1)
            for store, (f, w, s) in (
                (real, (bt["front"], bt["wrist"], bt["state"])),
                (swapped, (other["front"], other["wrist"], bt["state"])),
                (noisy, (bt["front"], bt["wrist"], bt["state"] + torch.randn_like(bt["state"]))),
            ):
                with torch.no_grad():
                    p = b.normalizer.denorm_action(m(f, w, s, bt["tokens"])[0].numpy())[j]
                store.append(float(np.linalg.norm(p[:2] - tgt) * 1000))
        res[split] = {
            "first_frame_target_error_mm_median": round(float(np.median(ground)), 1) if ground else float("nan"),
            "first_frame_target_within_30mm": round(float(np.mean(np.array(ground) < 30)), 3) if ground else float("nan"),
            "grasp_error_mm_real": round(float(np.median(real)), 1) if real else float("nan"),
            "grasp_error_mm_images_swapped": round(float(np.median(swapped)), 1) if swapped else float("nan"),
            "grasp_error_mm_state_noise": round(float(np.median(noisy)), 1) if noisy else float("nan"),
        }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"bundle": str(a.bundle), **res}, indent=1))


if __name__ == "__main__":
    main()
