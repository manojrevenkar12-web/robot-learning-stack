"""Model-agnostic training and fine-tuning with experiment tracking and checkpoint management.

    python -m rls.learn.train configs/train/act_main.yaml
    python -m rls.learn.train configs/train/act_main.yaml --resume   # continue from runs/<name>/last

Each run writes ``runs/<name>/``: ``best/`` and ``last/`` policy bundles
(weights + normaliser + config + lineage), ``state.pt`` (optimizer/RNG for
resume), ``metrics.jsonl``, and logs params/metrics/artifacts to MLflow
(file store under ``runs/mlruns``). Finished runs are appended to
``runs/registry.jsonl`` with their dataset version, so any checkpoint can be
traced to the exact data it saw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rls.learn.data import ChunkDataset
from rls.learn.lerobot_reader import load
from rls.learn.policies import Policy, PolicyBundle
from rls.learn.policies.base import count_parameters
from rls.learn.policies.layers import add_lora


@dataclass
class TrainConfig:
    run_name: str
    dataset: str
    policy: str = "chunk_transformer"
    policy_config: dict[str, Any] = field(default_factory=dict)
    sources: list[str] | None = None
    max_train_episodes: int | None = None
    steps: int = 3000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 150
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 8
    seed: int = 0
    threads: int = 2
    init_from: str | None = None
    finetune: str = "full"  # full | lora | head | scratch
    lora_rank: int = 8
    out_dir: str = "runs"
    tracking_uri: str | None = None
    notes: str = ""

    @staticmethod
    def load(path: Path, **overrides: Any) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text())
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return TrainConfig(**raw)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def configure_finetune(model: Policy, mode: str, lora_rank: int) -> dict[str, int]:
    """Freeze / wrap parameters for the fine-tuning mode. Returns parameter counts."""
    if mode in ("full", "scratch"):
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "head":
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(("decoder", "queries", "head", "tok")))
    elif mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        n = add_lora(model.encoder, rank=lora_rank) + add_lora(model.decoder, rank=lora_rank)  # type: ignore[arg-type]
        assert n > 0
        for name, p in model.named_parameters():
            if "lora_" in name or name.startswith(("tok", "head")):
                p.requires_grad_(True)
    else:
        raise ValueError(f"unknown finetune mode {mode!r}")
    return {"total": count_parameters(model), "trainable": count_parameters(model, trainable_only=True)}


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(cfg: TrainConfig, resume: bool = False) -> dict[str, Any]:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = load(Path(cfg.dataset))
    release = json.loads((Path(cfg.dataset) / "meta" / "rls_release.json").read_text()) if (Path(cfg.dataset) / "meta" / "rls_release.json").exists() else {}
    train_eps = data.split_episodes("train")
    if cfg.max_train_episodes:
        train_eps = train_eps[: cfg.max_train_episodes]
    val_eps = data.split_episodes("val")

    init_bundle = PolicyBundle.load(Path(cfg.init_from)) if cfg.init_from and cfg.finetune != "scratch" else None
    if init_bundle is not None:
        model = init_bundle.model
        model.train()
        normalizer = init_bundle.normalizer  # keep the pretrained action/state scaling
    else:
        model = Policy.build(cfg.policy, cfg.policy_config)
        normalizer = None
    chunk = int(model.cfg.chunk)  # type: ignore[attr-defined]
    space = str(getattr(model.cfg, "action_space", "joint"))
    train_ds = ChunkDataset(data, train_eps, chunk, normalizer, cfg.sources, space)
    normalizer = train_ds.normalizer
    val_ds = ChunkDataset(data, val_eps, chunk, normalizer, cfg.sources, space) if val_eps else None
    counts = configure_finetune(model, cfg.finetune, cfg.lora_rank)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step, best_val = 0, float("inf")
    if resume and (run_dir / "state.pt").exists():
        st = torch.load(run_dir / "state.pt", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step, best_val = st["step"], st["best_val"]
        rng = st["rng"]
        torch.set_rng_state(st["torch_rng"])

    cfg_hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:10]
    meta = {
        "run_name": cfg.run_name,
        "config_hash": cfg_hash,
        "dataset": cfg.dataset,
        "dataset_version": release.get("dataset_version", "unknown"),
        "git_sha": _git_sha(),
        "train_episodes": len(train_ds.episodes),
        "train_frames": len(train_ds),
        "sources": cfg.sources,
        "finetune": cfg.finetune,
        "init_from": cfg.init_from,
        "params": counts,
    }
    mlflow = None
    try:
        import mlflow as _mlflow

        _mlflow.set_tracking_uri(cfg.tracking_uri or (Path(cfg.out_dir).resolve() / "mlruns").as_uri())
        _mlflow.set_experiment("rls")
        _mlflow.start_run(run_name=cfg.run_name)
        _mlflow.log_params({k: str(v)[:250] for k, v in {**asdict(cfg), **{f"meta.{k}": v for k, v in meta.items()}}.items()})
        mlflow = _mlflow
    except Exception as exc:
        print(f"[train] MLflow disabled: {exc}")

    log_f = open(run_dir / "metrics.jsonl", "a")
    t0 = time.time()
    model.train()
    history: list[dict[str, float]] = []
    for step in range(start_step, cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        batch = train_ds.sample(cfg.batch_size, rng)
        out = model.compute_loss(batch)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if (step + 1) % 25 == 0:
            row = {
                "step": step + 1,
                "loss": float(out["loss"].detach()),
                "l1_first": float(out["l1_first_step"]),
                "grad_norm": float(gn),
                "lr": lr_at(step, cfg),
                "elapsed_s": round(time.time() - t0, 1),
            }
            if "aux_l1" in out:
                row["aux_l1"] = float(out["aux_l1"])
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                row["val_loss"] = evaluate_loss(model, val_ds, cfg) if val_ds else float("nan")
                bundle = PolicyBundle(model, normalizer, {**meta, "step": step + 1, "val_loss": row["val_loss"]})
                bundle.save(run_dir / "last")
                if row["val_loss"] < best_val or not np.isfinite(best_val):
                    best_val = row["val_loss"] if np.isfinite(row["val_loss"]) else best_val
                    bundle.save(run_dir / "best")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step + 1,
                        "best_val": best_val,
                        "rng": rng,
                        "torch_rng": torch.get_rng_state(),
                    },
                    run_dir / "state.pt",
                )
                print(
                    f"[train] {cfg.run_name} step {step + 1}/{cfg.steps} loss {row['loss']:.4f} val {row['val_loss']:.4f} ({row['elapsed_s']:.0f}s)", flush=True
                )
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            history.append(row)
            if mlflow:
                mlflow.log_metrics({k: v for k, v in row.items() if k != "step" and np.isfinite(v)}, step=step + 1)
    log_f.close()
    if not (run_dir / "best").exists():
        PolicyBundle(model, normalizer, {**meta, "step": cfg.steps}).save(run_dir / "best")
    summary = {**meta, "best_val_loss": best_val, "train_seconds": round(time.time() - t0, 1), "steps": cfg.steps, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(Path(cfg.out_dir) / "registry.jsonl", "a") as f:
        f.write(json.dumps({"bundle": str(run_dir / "best"), **summary}, default=str) + "\n")
    if mlflow:
        mlflow.log_metric("best_val_loss", best_val)
        mlflow.log_artifact(str(run_dir / "best" / "bundle.json"))
        mlflow.end_run()
    return summary


@torch.no_grad()
def evaluate_loss(model: Policy, ds: ChunkDataset, cfg: TrainConfig) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # fixed validation batches: comparable across steps
    losses = [float(model.compute_loss(ds.sample(cfg.batch_size, rng))["loss"]) for _ in range(cfg.eval_batches)]
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--run-name")
    ap.add_argument("--dataset")
    ap.add_argument("--max-train-episodes", type=int)
    ap.add_argument("--init-from")
    ap.add_argument("--finetune")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    cfg = TrainConfig.load(
        a.config, steps=a.steps, run_name=a.run_name, dataset=a.dataset, max_train_episodes=a.max_train_episodes, init_from=a.init_from, finetune=a.finetune, lr=a.lr
    )
    print(json.dumps(train(cfg, resume=a.resume), indent=1, default=str))


if __name__ == "__main__":
    main()
