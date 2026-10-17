"""Minimal reader for LeRobot v3.0 datasets (parquet + MP4), with no LeRobot dependency.

Decodes every video file once, streaming frame by frame, into a uint8 cache
(memory-mapped .npy next to the dataset), so training epochs never touch the video
decoder and memory stays flat however large the dataset is: frames are written
straight to the memory-mapped file, never collected in RAM. The cache is written to a
temporary name and renamed, so concurrent readers never see a half-written cache.
Verified frame-for-frame against ``lerobot.datasets.LeRobotDataset``
(``tests/test_release.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd


@dataclass
class LeRobotData:
    root: Path
    info: dict
    frames: pd.DataFrame  # one row per frame (data/*.parquet)
    episodes: pd.DataFrame  # meta/episodes
    tasks: list[str]
    images: dict[str, np.ndarray]  # key -> (total_frames, H, W, 3) uint8, row-aligned with `frames`
    extra: pd.DataFrame | None  # meta/rls_episodes.parquet when present

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    def episode_rows(self, episode_index: int) -> np.ndarray:
        e = self.episodes.loc[self.episodes["episode_index"] == episode_index].iloc[0]
        return np.arange(int(e["dataset_from_index"]), int(e["dataset_to_index"]))

    def split_episodes(self, split: str) -> list[int]:
        rng = self.info.get("splits", {}).get(split)
        if not rng:
            return []
        a, b = (int(x) for x in rng.split(":"))
        return list(range(a, b))


def _decode_into(path: Path, out: np.ndarray, spans: list[tuple[int, int, int]]) -> None:
    """Stream-decode one video file; ``spans`` = (first frame in file, dataset row, n frames) per episode."""
    spans = sorted(spans)
    k = 0
    with av.open(str(path)) as c:
        for f_idx, frame in enumerate(c.decode(video=0)):
            while k < len(spans) and f_idx >= spans[k][0] + spans[k][2]:
                k += 1
            if k == len(spans):
                break
            start, row, _ = spans[k]
            if f_idx >= start:
                out[row + f_idx - start] = frame.to_ndarray(format="rgb24")


def load(root: Path, video_keys: list[str] | None = None, cache: bool = True) -> LeRobotData:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True).sort_values("index").reset_index(drop=True)
    episodes = (
        pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    tasks = [str(t) for t in tasks_df.sort_values("task_index").index]
    keys = video_keys or [k for k, f in info["features"].items() if f["dtype"] == "video"]
    fps = int(info["fps"])
    images: dict[str, np.ndarray] = {}
    for key in keys:
        cache_path = root / ".rls_cache" / f"{key}.npy"
        if cache and cache_path.exists():
            images[key] = np.load(cache_path, mmap_mode="r")
            continue
        h, w, _ = info["features"][key]["shape"]
        shape = (len(frames), h, w, 3)
        out: np.ndarray
        if cache:
            cache_path.parent.mkdir(exist_ok=True)
            tmp = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
        else:
            out = np.zeros(shape, np.uint8)
        files: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for _, e in episodes.iterrows():
            fk = (int(e[f"videos/{key}/chunk_index"]), int(e[f"videos/{key}/file_index"]))
            start = int(round(float(e[f"videos/{key}/from_timestamp"]) * fps))
            a, b = int(e["dataset_from_index"]), int(e["dataset_to_index"])
            files.setdefault(fk, []).append((start, a, b - a))
        for (ci, fi), spans in files.items():
            _decode_into(root / info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi), out, spans)
        if cache:
            if isinstance(out, np.memmap):
                out.flush()
            del out
            os.replace(tmp, cache_path)  # atomic: other processes see no cache or a complete one
            images[key] = np.load(cache_path, mmap_mode="r")
        else:
            images[key] = out
    extra_path = root / "meta" / "rls_episodes.parquet"
    extra = pd.read_parquet(extra_path) if extra_path.exists() else None
    return LeRobotData(root, info, frames, episodes, tasks, images, extra)
