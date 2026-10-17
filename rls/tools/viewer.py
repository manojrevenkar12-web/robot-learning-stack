"""Web viewer for datasets, training runs and benchmarks (FastAPI + one static page).

rls viewer --data data --runs runs --reports reports   # http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

from rls.data.schema import read_episode

STATIC = Path(__file__).parent / "static"


def _downsample(a: np.ndarray, n: int = 300) -> list[list[float]]:
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return np.round(a[idx], 4).tolist()


def create_app(data: Path, runs: Path, reports: Path) -> FastAPI:
    app = FastAPI(title="robot-learning-stack viewer")
    data, runs, reports = Path(data), Path(runs), Path(reports)

    def manifest() -> pd.DataFrame:
        p = data / "curated" / "manifest_all.parquet"
        if p.exists():
            return pd.read_parquet(p)
        from rls.data.curate import scan

        return scan([data / "staging"]) if (data / "staging").exists() else pd.DataFrame()

    @lru_cache(maxsize=8)
    def episode(path: str):
        return read_episode(Path(path))

    def path_of(eid: str) -> str:
        df = manifest()
        row = df.loc[df["episode_id"] == eid]
        if row.empty:
            raise HTTPException(404, f"unknown episode {eid}")
        return str(row.iloc[0]["path"])

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        cur = data / "curated" / "curation_report.json"
        rel = data / "release" / "rls_panda" / "meta" / "rls_release.json"
        return {
            "curation": json.loads(cur.read_text()) if cur.exists() else None,
            "release": json.loads(rel.read_text()) if rel.exists() else None,
            "runs": len(list(runs.glob("*/summary.json"))) if runs.exists() else 0,
            "benchmarks": len(list(reports.rglob("benchmark.json"))) if reports.exists() else 0,
        }

    @app.get("/api/episodes")
    def episodes() -> list[dict[str, Any]]:
        df = manifest()
        cols = [
            c
            for c in (
                "episode_id",
                "source",
                "task",
                "instruction",
                "length",
                "health",
                "success",
                "failure_mode",
                "drop_reason",
                "split",
                "calibration_suspect",
            )
            if c in df
        ]
        return json.loads(df[cols].to_json(orient="records"))

    @app.get("/api/episode/{eid}")
    def episode_detail(eid: str) -> dict[str, Any]:
        ep = episode(path_of(eid))
        flags = ep.frame_flags if ep.frame_flags is not None else np.zeros(ep.length, np.uint8)
        return {
            "episode_id": ep.episode_id,
            "source": ep.source,
            "instruction": ep.instruction,
            "length": ep.length,
            "fps": ep.fps,
            "outcome": ep.outcome,
            "quality": ep.quality,
            "provenance": {k: v for k, v in ep.provenance.items() if k != "clock_models"},
            "state": _downsample(ep.state),
            "action": _downsample(ep.action),
            "wrench": _downsample(ep.wrench),
            "flags": flags.tolist(),
        }

    @app.get("/api/episode/{eid}/frame/{k}.jpg")
    def frame(eid: str, k: int, cam: str = "front") -> Response:
        ep = episode(path_of(eid))
        arr = ep.front if cam == "front" else ep.wrist
        if not 0 <= k < len(arr):
            raise HTTPException(404, "frame out of range")
        buf = io.BytesIO()
        Image.fromarray(arr[k]).resize((arr.shape[2] * 2, arr.shape[1] * 2), Image.NEAREST).save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        out = []
        for s in sorted(runs.glob("*/summary.json")) if runs.exists() else []:
            d = json.loads(s.read_text())
            out.append(
                {
                    k: d.get(k)
                    for k in ("run_name", "dataset_version", "train_episodes", "finetune", "best_val_loss", "train_seconds", "steps", "params", "sources")
                }
            )
        return out

    @app.get("/api/runs/{name}/metrics")
    def run_metrics(name: str) -> list[dict[str, Any]]:
        p = runs / name / "metrics.jsonl"
        if not p.exists():
            raise HTTPException(404, name)
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    @app.get("/api/benchmarks")
    def list_bench() -> list[dict[str, Any]]:
        out = []
        for p in sorted(reports.rglob("benchmark.json")) if reports.exists() else []:
            d = json.loads(p.read_text())
            out.append(
                {
                    "id": str(p.parent.relative_to(reports)),
                    "policy": d["policy"].get("run_name"),
                    "conditions": d["conditions"],
                    "splits": {k: {kk: v[kk] for kk in ("n", "success_rate", "ci95", "failures", "by_task")} for k, v in d["splits"].items()},
                    "videos": sorted(str(v.relative_to(reports)) for v in p.parent.rglob("*.mp4"))[:40],
                }
            )
        return out

    @app.get("/reports/{path:path}")
    def report_file(path: str) -> FileResponse:
        p = (reports / path).resolve()
        if reports.resolve() not in p.parents or not p.exists():
            raise HTTPException(404, path)
        return FileResponse(p)

    return app
