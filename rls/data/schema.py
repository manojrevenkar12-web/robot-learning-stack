"""Staging schema: one aligned, fixed-rate episode per HDF5 file.

Both data sources land here before curation and release:

* ``teleop_ros``: ROS 2 recordings (MCAP bags) after health checks, clock
  correction and alignment (:mod:`rls.data.ingest`);
* ``synthetic``: episodes generated directly in simulation (:mod:`rls.data.synthetic`).

Layout (ALOHA/ACT-style HDF5, chunked per frame so a single frame is cheap to read)::

    /timestamp                (T,)        float64  seconds since episode start
    /observation/state        (T, 8)      float32  joint positions q1..q7, gripper opening [0, 1]
    /observation/wrench       (T, 6)      float32  wrist force (N) and torque (Nm)
    /observation/images/front (T, H, W, 3) uint8
    /observation/images/wrist (T, h, w, 3) uint8
    /observation/depth/front  (T, H, W)   uint16   millimetres, 0 = invalid
    /action                   (T, 8)      float32  commanded q1..q7, gripper [0, 1] (1 = open)
    /quality/frame_flags      (T,)        uint8    bit flags, see FLAG_*
    attrs: episode_id, source, fps, task, instruction, episode_spec (json),
           outcome (json), quality (json), provenance (json)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

STATE_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
WRENCH_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]
ACTION_NAMES = [f"joint{i}_target" for i in range(1, 8)] + ["gripper_target"]
FPS = 15

# Per-frame quality flags (bitmask). A flagged frame is still usable; the flag
# records that it was filled or repaired, so training can down-weight or drop it.
FLAG_FRONT_FILLED = 1  # front frame missing, previous frame repeated
FLAG_WRIST_FILLED = 2
FLAG_STATE_INTERP_GAP = 4  # joint state interpolated across a gap > 2 periods
FLAG_WRENCH_MISSING = 8
FLAG_ACTION_HELD = 16  # no new command; previous command held
FLAG_CORRUPT_REPLACED = 32  # corrupt frame dropped and filled


@dataclass
class Episode:
    episode_id: str
    source: str
    fps: int
    task: str
    instruction: str
    timestamp: np.ndarray
    state: np.ndarray
    wrench: np.ndarray
    action: np.ndarray
    front: np.ndarray
    wrist: np.ndarray
    depth: np.ndarray | None = None
    frame_flags: np.ndarray | None = None
    episode_spec: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.timestamp.shape[0])

    def validate(self) -> None:
        t = self.length
        for name, arr, shape in (
            ("state", self.state, (t, 8)),
            ("wrench", self.wrench, (t, 6)),
            ("action", self.action, (t, 8)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{self.episode_id}: {name} shape {arr.shape} != {shape}")
        if self.front.shape[0] != t or self.wrist.shape[0] != t:
            raise ValueError(f"{self.episode_id}: image count does not match timestamps")
        if self.front.dtype != np.uint8 or self.front.ndim != 4 or self.front.shape[-1] != 3:
            raise ValueError(f"{self.episode_id}: front images must be (T,H,W,3) uint8")
        if not np.all(np.diff(self.timestamp) > 0):
            raise ValueError(f"{self.episode_id}: timestamps not strictly increasing")
        if not (np.isfinite(self.state).all() and np.isfinite(self.action).all()):
            raise ValueError(f"{self.episode_id}: non-finite state or action")


def content_hash(ep: Episode) -> str:
    """Hash of the episode's signal content (not its metadata)."""
    h = hashlib.sha256()
    for a in (ep.timestamp, ep.state, ep.action, ep.front[:: max(1, ep.length // 8)]):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def write_episode(path: Path, ep: Episode, compression: str | None = "lzf") -> Path:
    ep.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    flags = ep.frame_flags if ep.frame_flags is not None else np.zeros(ep.length, np.uint8)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("timestamp", data=ep.timestamp.astype(np.float64))
        f.create_dataset("observation/state", data=ep.state.astype(np.float32))
        f.create_dataset("observation/wrench", data=ep.wrench.astype(np.float32))
        f.create_dataset("action", data=ep.action.astype(np.float32))
        f.create_dataset("quality/frame_flags", data=flags.astype(np.uint8))
        for key, arr in (("observation/images/front", ep.front), ("observation/images/wrist", ep.wrist)):
            f.create_dataset(key, data=arr, chunks=(1, *arr.shape[1:]), compression=compression)
        if ep.depth is not None:
            f.create_dataset("observation/depth/front", data=ep.depth.astype(np.uint16), chunks=(1, *ep.depth.shape[1:]), compression=compression)
        f.attrs["episode_id"] = ep.episode_id
        f.attrs["source"] = ep.source
        f.attrs["fps"] = ep.fps
        f.attrs["task"] = ep.task
        f.attrs["instruction"] = ep.instruction
        for k in ("episode_spec", "outcome", "quality", "provenance"):
            f.attrs[k] = json.dumps(getattr(ep, k), sort_keys=True, default=_json_default)
        f.attrs["content_hash"] = content_hash(ep)
    tmp.replace(path)
    return path


def read_episode(path: Path, images: bool = True) -> Episode:
    with h5py.File(path, "r") as f:
        a = f.attrs
        depth = f["observation/depth/front"][()] if images and "observation/depth/front" in f else None
        return Episode(
            episode_id=str(a["episode_id"]),
            source=str(a["source"]),
            fps=int(a["fps"]),
            task=str(a["task"]),
            instruction=str(a["instruction"]),
            timestamp=f["timestamp"][()],
            state=f["observation/state"][()],
            wrench=f["observation/wrench"][()],
            action=f["action"][()],
            front=f["observation/images/front"][()] if images else np.zeros((len(f["timestamp"]), 1, 1, 3), np.uint8),
            wrist=f["observation/images/wrist"][()] if images else np.zeros((len(f["timestamp"]), 1, 1, 3), np.uint8),
            depth=depth,
            frame_flags=f["quality/frame_flags"][()],
            episode_spec=json.loads(a["episode_spec"]),
            outcome=json.loads(a["outcome"]),
            quality=json.loads(a["quality"]),
            provenance=json.loads(a["provenance"]),
        )


def read_attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        out: dict[str, Any] = {k: (json.loads(v) if k in ("episode_spec", "outcome", "quality", "provenance") else v) for k, v in f.attrs.items()}
        out["length"] = int(f["timestamp"].shape[0])
        out["path"] = str(path)
        return out


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
