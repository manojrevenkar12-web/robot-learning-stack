"""Release path: curated manifest -> LeRobot v3.0 (LeRobot's writer) -> our reader, checked
against LeRobot's own loader. Also smoke-tests the viewer API on the same data."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rls.data.curate import CurationConfig, curate, scan, write_outputs
from rls.data.schema import Episode, read_episode, write_episode

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_PY = os.environ.get("RLS_LEROBOT_PYTHON", "/opt/lerobot-venv/bin/python")
pytestmark = pytest.mark.skipif(not Path(LEROBOT_PY).exists(), reason="LeRobot environment not installed (RLS_LEROBOT_PYTHON)")


def _ep(i: int) -> Episode:
    rng = np.random.default_rng(i)
    t = 45
    act = np.cumsum(rng.normal(0, 0.05, (t, 8)), 0).astype(np.float32)
    # smooth images (video codecs are lossy on noise; the test is about indexing, not codec quality)
    yy, xx = np.mgrid[0:64, 0:64]
    base = ((xx * 3 + yy * 2 + i * 40) % 255).astype(np.uint8)
    front = np.stack([np.stack([np.roll(base, k, 1)] * 3, -1) for k in range(t)])
    return Episode(
        episode_id=f"rel{i:02d}",
        source="synthetic",
        fps=15,
        task="pick_place",
        instruction=f"put the {['red', 'blue'][i % 2]} cube in the bin",
        timestamp=np.arange(t) / 15,
        state=act.copy(),
        wrench=np.zeros((t, 6), np.float32),
        action=act,
        front=front,
        wrist=front[:, :, :, ::-1].copy(),
        outcome={"success": True, "failure_mode": "success"},
        quality={"health": "pass", "checks": {}},
    )


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("rel")
    for i in range(6):
        write_episode(tmp / "staging" / f"rel{i:02d}.h5", _ep(i))
    df, rep = curate(scan([tmp / "staging"]), CurationConfig(val_fraction=0.2))
    manifest = write_outputs(df, rep, tmp / "curated")
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    subprocess.run(
        [LEROBOT_PY, "-m", "rls.data.export_lerobot", "--manifest", str(manifest), "--out", str(tmp / "release")],
        check=True,
        env=env,
        cwd=ROOT,
        capture_output=True,
    )
    return tmp, tmp / "release"


def test_reader_matches_lerobot_loader(release: tuple[Path, Path]) -> None:
    from rls.learn.lerobot_reader import load

    _, root = release
    ours = load(root, cache=False)
    idx = [0, 7, len(ours.frames) // 2, len(ours.frames) - 1]
    code = (
        "import json, sys, numpy as np; from lerobot.datasets.lerobot_dataset import LeRobotDataset;"
        f"ds = LeRobotDataset('local/rls_panda', root='{root}');"
        f"idx = {idx}; out = {{}};"
        "[out.update({f'{k}_{i}': (ds[i][k].numpy() * (255 if 'images' in k else 1)) for k in ('observation.images.front', 'observation.images.wrist', 'action', 'episode_index')}) for i in idx];"
        f"np.savez('{root.parent / 'ref.npz'}', **out)"
    )
    subprocess.run([LEROBOT_PY, "-c", code], check=True, env=dict(os.environ, HF_HUB_OFFLINE="1"), capture_output=True)
    ref = np.load(root.parent / "ref.npz")
    for i in idx:
        for key in ("observation.images.front", "observation.images.wrist"):
            theirs = np.round(ref[f"{key}_{i}"]).astype(np.uint8).transpose(1, 2, 0)
            assert np.array_equal(ours.images[key][i], theirs), (key, i)
        assert np.allclose(np.stack(ours.frames.loc[i, "action"]), ref[f"action_{i}"])
        assert int(ours.frames.loc[i, "episode_index"]) == int(ref[f"episode_index_{i}"])


def test_release_preserves_trajectories_and_splits(release: tuple[Path, Path]) -> None:
    from rls.learn.lerobot_reader import load

    tmp, root = release
    data = load(root, cache=False)
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v3")
    assert len(data.split_episodes("train")) + len(data.split_episodes("val")) == 6
    for e in range(6):
        sid = data.extra.loc[data.extra["episode_index"] == e].iloc[0]["episode_id"]
        src = read_episode(tmp / "staging" / f"{sid}.h5")
        rows = data.episode_rows(e)
        assert np.allclose(np.stack(data.frames.loc[rows, "action"].to_numpy()), src.action)


def test_viewer_api(release: tuple[Path, Path]) -> None:
    from fastapi.testclient import TestClient

    from rls.tools.viewer import create_app

    tmp, _ = release
    c = TestClient(create_app(tmp, tmp / "runs", tmp / "reports"))
    assert c.get("/").status_code == 200
    eps = c.get("/api/episodes").json()
    assert len(eps) == 6
    d = c.get(f"/api/episode/{eps[0]['episode_id']}").json()
    assert d["length"] == 45 and len(d["state"]) == 45
    img = c.get(f"/api/episode/{eps[0]['episode_id']}/frame/3.jpg?cam=wrist")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert c.get("/reports/../../etc/passwd").status_code == 404
