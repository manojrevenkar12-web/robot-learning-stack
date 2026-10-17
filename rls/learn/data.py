"""Training samples from a LeRobot v3 dataset: (images, state, instruction) -> action chunk."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from rls.learn.lerobot_reader import LeRobotData
from rls.learn.policies.base import Normalizer
from rls.learn.text import tokenize

FRONT, WRIST = "observation.images.front", "observation.images.wrist"


@dataclass
class EpisodeIndex:
    rows: np.ndarray  # global frame rows used as samples
    ep_end: np.ndarray  # for each sample, global row one past its episode's last frame


def _converted_actions(data: LeRobotData, joint_actions: np.ndarray, space: str) -> np.ndarray:
    """Recorded joint commands in another action space, cached next to the dataset (FK is cheap but not free)."""
    from rls.learn.action_space import make

    cache = data.root / ".rls_cache" / f"action_{space}.npy"
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(joint_actions):
            return arr
    conv = make(space)
    assert conv is not None
    arr = conv.from_joint(joint_actions)
    cache.parent.mkdir(exist_ok=True)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, cache)
    return arr


AUX_CENTER = np.array([0.5, 0.0], np.float32)
AUX_SCALE = 0.15


def _object_labels(data: LeRobotData, actions: np.ndarray, space: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-frame (target xy, goal xy), normalised, from the episode's scene layout when the release has one.

    The target label is valid until the first gripper close (afterwards the cube moves with the hand);
    the goal (bin centre, or the support cube for stacking) stays put for the whole episode.
    """
    if data.extra is None or "episode_spec" not in data.extra:
        return None, None
    xy = np.zeros((len(data.frames), 2, 2), np.float32)
    valid = np.zeros((len(data.frames), 2), bool)
    grip = actions[:, -1]
    for _, r in data.extra.iterrows():
        spec = json.loads(r["episode_spec"])
        rows = data.episode_rows(int(r["episode_index"]))
        tgt = np.array(spec["init_poses"][spec["target"]][:2], np.float32)
        goal = np.array(spec["init_poses"][spec["support"]][:2] if spec.get("support") else spec["scene"]["bin_pos"][:2], np.float32)
        closed = np.nonzero(grip[rows] < 0.5)[0]
        first_close = int(closed[0]) if len(closed) else len(rows)
        xy[rows, 0] = (tgt - AUX_CENTER) / AUX_SCALE
        xy[rows, 1] = (goal - AUX_CENTER) / AUX_SCALE
        valid[rows[:first_close], 0] = True
        valid[rows, 1] = True
    return xy, valid


def build_index(data: LeRobotData, episodes: list[int]) -> EpisodeIndex:
    rows, ends = [], []
    for e in episodes:
        r = data.episode_rows(e)
        rows.append(r)
        ends.append(np.full(len(r), r[-1] + 1))
    return EpisodeIndex(np.concatenate(rows), np.concatenate(ends))


class ChunkDataset:
    """Random-access sampler over a frame range; batches are gathered with fancy indexing."""

    def __init__(
        self,
        data: LeRobotData,
        episodes: list[int],
        chunk: int,
        normalizer: Normalizer | None = None,
        sources: list[str] | None = None,
        action_space: str = "joint",
    ) -> None:
        if sources is not None and data.extra is not None:
            src = dict(zip(data.extra["episode_index"], data.extra["source"], strict=True))
            episodes = [e for e in episodes if src[e] in sources]
        self.data = data
        self.episodes = episodes
        self.idx = build_index(data, episodes)
        self.chunk = chunk
        fr = data.frames
        self.state = np.stack(fr["observation.state"].to_numpy()).astype(np.float32)
        self.action = np.stack(fr["action"].to_numpy()).astype(np.float32)
        if action_space != "joint":
            self.action = _converted_actions(data, self.action, action_space)
        self.tokens = np.array([tokenize(t) for t in data.tasks], dtype=np.int64)[fr["task_index"].to_numpy()]
        rows = self.idx.rows
        self.normalizer = normalizer or Normalizer.fit(self.state[rows], self.action[rows])
        self.front = data.images[FRONT]
        self.wrist = data.images[WRIST]
        self.aux_xy, self.aux_valid = _object_labels(data, self.action, action_space)

    def __len__(self) -> int:
        return len(self.idx.rows)

    def batch(self, sample_ids: np.ndarray) -> dict[str, torch.Tensor]:
        rows = self.idx.rows[sample_ids]
        ends = self.idx.ep_end[sample_ids]
        offs = rows[:, None] + np.arange(self.chunk)[None, :]
        valid = offs < ends[:, None]
        offs = np.minimum(offs, ends[:, None] - 1)
        order = np.argsort(rows)  # sorted reads are much faster on the memory-mapped image cache
        inv = np.argsort(order)
        front = np.asarray(self.front[rows[order]])[inv]
        wrist = np.asarray(self.wrist[rows[order]])[inv]
        n = self.normalizer
        return {
            "front": torch.from_numpy(front),
            "wrist": torch.from_numpy(wrist),
            "state": torch.from_numpy(n.norm_state(self.state[rows])),
            "tokens": torch.from_numpy(self.tokens[rows]),
            "action": torch.from_numpy(n.norm_action(self.action[offs])),
            "action_valid": torch.from_numpy(valid),
            **({"aux_xy": torch.from_numpy(self.aux_xy[rows]), "aux_valid": torch.from_numpy(self.aux_valid[rows])} if self.aux_xy is not None else {}),
        }

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        return self.batch(rng.integers(0, len(self), batch_size))
