from pathlib import Path

import numpy as np
import pandas as pd

from rls.data.curate import CurationConfig, curate, scan
from rls.data.schema import Episode, read_episode, write_episode


def _ep(eid: str, seed: int, success: bool = True, health: str = "ok", source: str = "synthetic", task: str = "pick_place") -> Episode:
    rng = np.random.default_rng(seed)
    t = 80
    act = np.cumsum(rng.normal(0, 0.05, (t, 8)), 0).astype(np.float32)
    return Episode(
        episode_id=eid,
        source=source,
        fps=15,
        task=task,
        instruction="put the red cube in the bin",
        timestamp=np.arange(t) / 15,
        state=act.copy(),
        wrench=np.zeros((t, 6), np.float32),
        action=act,
        front=rng.integers(0, 255, (t, 12, 16, 3), dtype=np.uint8),
        wrist=rng.integers(0, 255, (t, 8, 8, 3), dtype=np.uint8),
        outcome={"success": success, "failure_mode": "success" if success else "dropped"},
        quality={"health": health, "checks": {}},
    )


def test_schema_roundtrip(tmp_path: Path) -> None:
    ep = _ep("a", 0)
    write_episode(tmp_path / "a.h5", ep)
    back = read_episode(tmp_path / "a.h5")
    assert np.array_equal(back.front, ep.front) and np.allclose(back.action, ep.action)
    assert back.outcome == ep.outcome


def test_filters_dedup_and_splits(tmp_path: Path) -> None:
    eps = [_ep(f"e{i:02d}", i) for i in range(20)]
    eps += [_ep("failed", 100, success=False), _ep("broken", 101, health="fail")]
    near = _ep("near_dup", 3)  # e03 re-recorded: same scene and trajectory, fresh sensor noise
    noise = np.random.default_rng(9).integers(-3, 4, near.front.shape)
    near.front = np.clip(near.front.astype(int) + noise, 0, 255).astype(np.uint8)
    near.action = near.action + np.random.default_rng(10).normal(0, 0.002, near.action.shape).astype(np.float32)
    eps.append(near)
    other = _ep("other_scene", 4)  # same trajectory as e04 but a different scene: not a duplicate
    other.front = np.random.default_rng(11).integers(0, 255, other.front.shape, dtype=np.uint8)
    eps.append(other)
    for e in eps:
        write_episode(tmp_path / f"{e.episode_id}.h5", e)
    exact = tmp_path / "exact.h5"
    exact.write_bytes((tmp_path / "e05.h5").read_bytes())
    df, rep = curate(scan([tmp_path]), CurationConfig(val_fraction=0.2))
    reasons = dict(zip(df["episode_id"], df["drop_reason"], strict=True))
    assert reasons["failed"] == "task_failed"
    assert reasons["broken"] == "health"
    assert reasons["near_dup"].startswith("duplicate_near:e03")
    assert sorted(r for r in reasons.values() if r.startswith("duplicate_exact")) == ["duplicate_exact"]
    assert reasons["other_scene"] == ""
    kept = df[df["drop_reason"] == ""]
    assert len(kept) == 21 and set(kept["split"]) == {"train", "val"}
    assert (kept["split"] == "val").sum() == 4
    # deterministic
    df2, rep2 = curate(scan([tmp_path]), CurationConfig(val_fraction=0.2))
    assert rep["version"] == rep2["version"]
    pd.testing.assert_series_equal(df["split"], df2["split"])
