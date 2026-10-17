"""Curation: filter, deduplicate and split staged episodes into a versioned dataset manifest.

Every episode that is dropped keeps a reason, so a curation run can be audited
("why is my episode not in v-3f2a?") and re-run with different rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rls.data.schema import read_attrs, read_episode

CURATION_VERSION = "curate-2"
SEVERITY = {"ok": 0, "pass": 0, "warn": 1, "fail": 2}


@dataclass
class CurationConfig:
    name: str = "default"
    sources: list[str] = field(default_factory=lambda: ["teleop_ros", "synthetic"])
    require_success: bool = True
    max_health: str = "warn"  # drop anything worse
    min_length: int = 40
    max_length: int = 320
    max_filled_fraction: float = 0.05  # frames repaired by the ingest step
    dedup: bool = True
    near_dup_threshold: float = 0.01  # mean abs action difference (rad) after resampling
    near_dup_scene_threshold: float = 15.0  # first front frame, 30x40 grey blocks: 99th pct abs difference (0-255)
    val_fraction: float = 0.1
    split_seed: int = 0
    max_episodes_per_source: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> CurationConfig:
        return CurationConfig(**yaml.safe_load(Path(path).read_text()))


def scan(staging_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in staging_dirs:
        for p in sorted(Path(d).rglob("*.h5")):
            a = read_attrs(p)
            q = a.get("quality", {})
            rows.append(
                {
                    "episode_id": a["episode_id"],
                    "path": str(p),
                    "source": a["source"],
                    "task": a["task"],
                    "instruction": a["instruction"],
                    "length": a["length"],
                    "content_hash": a["content_hash"],
                    "success": bool(a.get("outcome", {}).get("success", False)),
                    "failure_mode": a.get("outcome", {}).get("failure_mode", "unknown"),
                    "health": q.get("health", "ok"),
                    "calibration_suspect": q.get("checks", {}).get("front_extrinsics", {}).get("status") == "warn",
                    "variant": a.get("episode_spec", {}).get("variant", ""),
                    "spec_seed": a.get("episode_spec", {}).get("seed", -1),
                }
            )
    return pd.DataFrame(rows)


def _fingerprint(path: str, n: int = 48) -> np.ndarray:
    ep = read_episode(Path(path), images=False)
    t = np.linspace(0, 1, n)
    src = np.linspace(0, 1, ep.length)
    return np.stack([np.interp(t, src, ep.action[:, j]) for j in range(ep.action.shape[1])], axis=1)


def _scene(path: str) -> np.ndarray:
    """Coarse view of the initial scene: first front frame, grey, 30x40 block means (camera noise averages out)."""
    import h5py

    with h5py.File(path, "r") as f:
        img = f["observation/images/front"][0].astype(np.float32).mean(-1)
    gh, gw = min(30, img.shape[0]), min(40, img.shape[1])
    bh, bw = img.shape[0] // gh, img.shape[1] // gw
    return img[: bh * gh, : bw * gw].reshape(gh, bh, gw, bw).mean((1, 3))


def _filled_fraction(path: str) -> float:
    ep = read_episode(Path(path), images=False)
    flags = ep.frame_flags if ep.frame_flags is not None else np.zeros(1, np.uint8)
    return float(np.mean((flags & 0b11) > 0))


def curate(manifest: pd.DataFrame, cfg: CurationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = manifest.copy()
    df["drop_reason"] = ""

    def drop(mask: pd.Series, reason: str) -> None:
        sel = mask & (df["drop_reason"] == "")
        df.loc[sel, "drop_reason"] = reason

    drop(~df["source"].isin(cfg.sources), "source_excluded")
    drop(df["health"].map(SEVERITY).fillna(2) > SEVERITY[cfg.max_health], "health")
    if cfg.require_success:
        drop(~df["success"], "task_failed")
    drop(df["length"] < cfg.min_length, "too_short")
    drop(df["length"] > cfg.max_length, "too_long")
    alive = df["drop_reason"] == ""
    df["filled_fraction"] = 0.0
    df.loc[alive & (df["source"] == "teleop_ros"), "filled_fraction"] = [_filled_fraction(p) for p in df.loc[alive & (df["source"] == "teleop_ros"), "path"]]
    drop(df["filled_fraction"] > cfg.max_filled_fraction, "too_many_repaired_frames")

    if cfg.dedup:
        # the earliest file wins: a re-import or re-recording is the copy, not the original
        df["_mtime"] = [Path(p).stat().st_mtime if Path(p).exists() else 0.0 for p in df["path"]]
        order = df.sort_values(["_mtime", "episode_id"]).index
        exact = df.loc[order, "drop_reason"].eq("") & df.loc[order].duplicated("content_hash", keep="first")
        drop(exact.reindex(df.index, fill_value=False), "duplicate_exact")
        keep_idx = [i for i in order if df.at[i, "drop_reason"] == ""]
        fps = {i: _fingerprint(df.at[i, "path"]) for i in keep_idx}
        scenes: dict[int, np.ndarray] = {}

        def scene(i: int) -> np.ndarray:
            if i not in scenes:
                scenes[i] = _scene(df.at[i, "path"])
            return scenes[i]

        def near(i: int, j: int) -> bool:
            # same task, nearly the same actions AND the same initial scene: different scenes can share a
            # trajectory shape (every pick-and-place goes home -> object -> bin) without being duplicates
            if df.at[j, "task"] != df.at[i, "task"] or np.mean(np.abs(fps[i] - fps[j])) >= cfg.near_dup_threshold:
                return False
            return float(np.percentile(np.abs(scene(i) - scene(j)), 99)) < cfg.near_dup_scene_threshold

        kept: list[int] = []
        for i in keep_idx:  # greedy, in file-age order
            dup = next((j for j in kept if near(i, j)), None)
            if dup is not None:
                df.at[i, "drop_reason"] = f"duplicate_near:{df.at[dup, 'episode_id']}"
            else:
                kept.append(i)
        df = df.drop(columns="_mtime")

    for src, cap in cfg.max_episodes_per_source.items():
        idx = df.index[(df["drop_reason"] == "") & (df["source"] == src)]
        drop(df.index.isin(idx[cap:]), "source_cap")

    keep = df[df["drop_reason"] == ""].copy()
    # deterministic, stratified split: hash ranks within (task, source)
    keep["split"] = "train"
    for _, g in keep.groupby(["task", "source"]):
        h = g["episode_id"].map(lambda e: hashlib.sha256(f"{cfg.split_seed}:{e}".encode()).hexdigest())
        n_val = int(round(len(g) * cfg.val_fraction))
        keep.loc[h.sort_values().index[:n_val], "split"] = "val"
    keep["split_order"] = keep["split"].map({"train": 0, "val": 1})
    keep = keep.sort_values(["split_order", "source", "episode_id"]).drop(columns="split_order")
    df["split"] = ""
    df.loc[keep.index, "split"] = keep["split"]
    version = (
        "v-"
        + hashlib.sha256(
            (json.dumps(asdict(cfg), sort_keys=True) + CURATION_VERSION + "".join(keep["episode_id"] + keep["content_hash"] + keep["split"])).encode()
        ).hexdigest()[:10]
    )
    report = {
        "version": version,
        "config": asdict(cfg),
        "curation_version": CURATION_VERSION,
        "scanned": len(df),
        "kept": len(keep),
        "dropped_by_reason": df.loc[df["drop_reason"] != "", "drop_reason"].str.split(":").str[0].value_counts().to_dict(),
        "kept_by_source": keep["source"].value_counts().to_dict(),
        "kept_by_task": keep["task"].value_counts().to_dict(),
        "splits": keep["split"].value_counts().to_dict(),
        "frames": int(keep["length"].sum()),
        "calibration_suspect_kept": int(keep["calibration_suspect"].sum()),
    }
    return df.assign(version=version), report


def write_outputs(df: pd.DataFrame, report: dict[str, Any], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "manifest_all.parquet", index=False)
    kept = df[df["drop_reason"] == ""].copy()
    kept["split_order"] = kept["split"].map({"train": 0, "val": 1})
    kept = kept.sort_values(["split_order", "source", "episode_id"]).drop(columns="split_order")
    kept.to_parquet(out / "manifest.parquet", index=False)
    (out / "curation_report.json").write_text(json.dumps(report, indent=1, default=str))
    return out / "manifest.parquet"
