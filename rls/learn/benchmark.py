"""Closed-loop benchmark in simulation: generalization splits, success, failure taxonomy.

    python -m rls.learn.benchmark runs/act_main/best --splits in_distribution,position_ood --episodes 30

Episodes are drawn from benchmark seeds (900000+) that never appear in any
training data. Each episode ends on sustained success or after ``max_steps``;
failures are classified by the simulator's outcome tracker, and a few
episodes per failure mode are saved as videos for the failure-analysis report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np

from rls.capture.faults import add_sensor_noise
from rls.learn.policies.base import PolicyBundle
from rls.learn.runner import ChunkRunner, OnnxModel
from rls.sim.env import PandaEnv, sample_episode
from rls.sim.variants import SPLITS, with_tasks

BENCH_SEED = 900_000
FAILURE_GUIDE = {
    "no_grasp": "never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.",
    "wrong_object": "manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.",
    "dropped": "lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.",
    "misplaced": "released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.",
    "timeout": "still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.",
}


@dataclass
class EnvConditions:
    """Deployment-side conditions the policy did not see in training (transfer studies)."""

    obs_latency_steps: int = 0  # observation is this many control ticks old
    image_noise_std: float = 2.0
    action_delay_steps: int = 0


@dataclass
class EpisodeResult:
    split: str
    seed: int
    task: str
    instruction: str
    success: bool
    failure_mode: str
    steps: int
    mean_infer_ms: float


@dataclass
class SplitResult:
    split: str
    n: int
    success_rate: float
    ci95: tuple[float, float]
    by_task: dict[str, float]
    failures: dict[str, int]
    episodes: list[EpisodeResult] = field(default_factory=list)


def bench_seed(split: str, i: int) -> int:
    """Benchmark episode seeds: fixed per split, disjoint from all data-generation seeds."""
    return BENCH_SEED + list(SPLITS).index(split) * 10_000 + i


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[: s.height, : s.width]), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def run_episode(
    env: PandaEnv,
    runner: ChunkRunner,
    spec: Any,
    cond: EnvConditions,
    max_steps: int = 300,
    record: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[EpisodeResult, list[np.ndarray]]:
    rng = rng or np.random.default_rng(spec.seed)
    obs = env.reset(spec)
    runner.reset(spec.instruction)
    hist: deque[dict[str, np.ndarray]] = deque(maxlen=cond.obs_latency_steps + 1)
    pending: deque[np.ndarray] = deque()
    frames: list[np.ndarray] = []
    ok_streak = 0
    hold = np.concatenate([obs["qpos"], [1.0]])
    for _ in range(max_steps):
        o = {
            "front": add_sensor_noise(obs["front_rgb"], cond.image_noise_std, rng),
            "wrist": add_sensor_noise(obs["wrist_rgb"], cond.image_noise_std, rng),
            "state": np.concatenate([obs["qpos"], obs["gripper"]]).astype(np.float32),
        }
        hist.append(o)
        seen = hist[0]
        a = runner.act(seen["front"], seen["wrist"], seen["state"])
        pending.append(a)
        cmd = pending.popleft() if len(pending) > cond.action_delay_steps else hold
        if record:
            wr = np.zeros_like(o["front"])
            wr[: o["wrist"].shape[0], : o["wrist"].shape[1]] = o["wrist"]
            frames.append(np.concatenate([o["front"], wr], axis=1))
        obs = env.step(cmd)
        ok_streak = ok_streak + 1 if env.task_success() else 0
        if ok_streak >= 8:
            break
    out = env.outcome()
    lat = runner.latencies_ms[-env.steps :] if runner.latencies_ms else [0.0]
    return (
        EpisodeResult(spec.variant, spec.seed, spec.task, spec.instruction, out.success, out.failure_mode, env.steps, float(np.mean(lat))),
        frames,
    )


def benchmark(
    bundle_path: Path,
    splits: list[str],
    episodes: int,
    out_dir: Path | None = None,
    tasks: tuple[str, ...] = ("pick_place", "stack"),
    cond: EnvConditions | None = None,
    onnx: Path | None = None,
    videos_per_mode: int = 2,
    max_steps: int = 300,
    mode: str = "ensemble",
) -> dict[str, SplitResult]:
    bundle = PolicyBundle.load(bundle_path)
    runner = ChunkRunner(bundle, OnnxModel(str(onnx)) if onnx else None, mode=mode)
    env = PandaEnv(render=True, depth=False)
    cond = cond or EnvConditions()
    results: dict[str, SplitResult] = {}
    for split in splits:
        variant = with_tasks(SPLITS[split], tasks)
        eps: list[EpisodeResult] = []
        saved: Counter[str] = Counter()
        t0 = time.time()
        for i in range(episodes):
            seed = bench_seed(split, i)
            spec = sample_episode(seed, variant, tasks[i % len(tasks)])
            want_video = out_dir is not None and videos_per_mode > 0
            res, frames = run_episode(env, runner, spec, cond, max_steps=max_steps, record=want_video)
            eps.append(res)
            if want_video and saved[res.failure_mode] < videos_per_mode and frames:
                assert out_dir is not None
                _write_video(out_dir / "videos" / split / f"{res.failure_mode}_{seed}.mp4", frames)
                saved[res.failure_mode] += 1
        k = sum(e.success for e in eps)
        by_task = {t: float(np.mean([e.success for e in eps if e.task == t])) for t in tasks if any(e.task == t for e in eps)}
        results[split] = SplitResult(
            split, len(eps), k / len(eps), wilson(k, len(eps)), by_task, dict(Counter(e.failure_mode for e in eps if not e.success)), eps
        )
        print(f"[bench] {split}: {k}/{len(eps)} success ({100 * k / len(eps):.0f}%), {time.time() - t0:.0f}s", flush=True)
    env.close_renderers()
    if out_dir is not None:
        write_report(out_dir, bundle, results, cond)
    return results


def write_report(out_dir: Path, bundle: PolicyBundle, results: dict[str, SplitResult], cond: EnvConditions) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "policy": bundle.meta,
        "conditions": asdict(cond),
        "splits": {
            k: {**{kk: vv for kk, vv in asdict(v).items() if kk != "episodes"}, "episodes": [asdict(e) for e in v.episodes]} for k, v in results.items()
        },
    }
    (out_dir / "benchmark.json").write_text(json.dumps(data, indent=1, default=str))
    lines = [
        f"# Benchmark: `{bundle.meta.get('run_name')}`",
        "",
        f"Dataset `{bundle.meta.get('dataset_version')}`, {bundle.meta.get('train_episodes')} training episodes, conditions {asdict(cond)}.",
        "",
        "| split | episodes | success | 95% CI | pick_place | stack | top failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        top = max(r.failures.items(), key=lambda kv: kv[1])[0] if r.failures else "-"
        lines.append(
            f"| {r.split} | {r.n} | {100 * r.success_rate:.0f}% | {100 * r.ci95[0]:.0f}-{100 * r.ci95[1]:.0f}% | "
            f"{100 * r.by_task.get('pick_place', float('nan')):.0f}% | {100 * r.by_task.get('stack', float('nan')):.0f}% | {top} |"
        )
    lines += ["", "## Failure analysis", ""]
    total: Counter[str] = Counter()
    for r in results.values():
        total.update(r.failures)
    if not total:
        lines.append("No failures.")
    for mode, n in total.most_common():
        where = ", ".join(f"{r.split} ({r.failures[mode]})" for r in results.values() if r.failures.get(mode))
        lines += [
            f"### {mode}: {n} episodes",
            "",
            f"- Where: {where}",
            f"- Likely cause: {FAILURE_GUIDE.get(mode, 'see videos')}",
            f"- Videos: `videos/<split>/{mode}_<seed>.mp4`",
            "- Owner / next step: _fill in_",
            "",
        ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--obs-latency", type=int, default=0)
    ap.add_argument("--action-delay", type=int, default=0)
    ap.add_argument("--tasks", default="pick_place,stack")
    ap.add_argument("--mode", default="ensemble")
    ap.add_argument("--videos-per-mode", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=300)
    a = ap.parse_args()
    cond = EnvConditions(obs_latency_steps=a.obs_latency, action_delay_steps=a.action_delay)
    res = benchmark(
        a.bundle,
        a.splits.split(","),
        a.episodes,
        a.out,
        tuple(a.tasks.split(",")),
        cond,
        a.onnx,
        videos_per_mode=a.videos_per_mode,
        max_steps=a.max_steps,
        mode=a.mode,
    )
    print(json.dumps({k: {"success": v.success_rate, "ci95": v.ci95, "failures": v.failures} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
