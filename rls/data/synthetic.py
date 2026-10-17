"""Synthetic episodes generated directly in simulation, in the staging schema.

Unlike the ROS 2 recordings these are perfectly synchronized, so they skip
ingest and health repair; they carry ``source="synthetic"`` and their full
generation provenance (variant, seed, expert style) so they can be traced and
re-generated bit-for-bit.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rls.capture.faults import add_sensor_noise
from rls.data.schema import FPS, Episode, write_episode
from rls.sim.env import EpisodeSpec, PandaEnv, Variant, sample_episode
from rls.sim.expert import ScriptedExpert

GENERATOR_VERSION = "synthetic-1"


def depth_to_mm(depth_m: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    d = np.where((depth_m > 0) & (depth_m < max_m), depth_m * 1000.0, 0.0)
    return np.round(d).astype(np.uint16)


def generate_episode(env: PandaEnv, spec: EpisodeSpec, seed: int, max_steps: int = 330, keep_failures: bool = True, noise_std: float = 2.0) -> Episode | None:
    rng = np.random.default_rng(seed + 104729)
    obs = env.reset(spec)
    expert = ScriptedExpert(spec, env.model, seed=seed)
    frames: dict[str, list[Any]] = {k: [] for k in ("t", "state", "wrench", "action", "front", "wrist", "depth")}
    settle = 0
    for _ in range(max_steps):
        a = expert.act(env.privileged())
        frames["t"].append(env.time)
        frames["state"].append(np.concatenate([obs["qpos"], obs["gripper"]]))
        frames["wrench"].append(obs["ft"])
        frames["action"].append(a)
        frames["front"].append(add_sensor_noise(obs["front_rgb"], noise_std, rng))
        frames["wrist"].append(add_sensor_noise(obs["wrist_rgb"], noise_std, rng))
        frames["depth"].append(depth_to_mm(obs["front_depth"]))
        obs = env.step(a)
        if expert.done:
            settle += 1
            if settle > 5:
                break
    out = env.outcome()
    if not keep_failures and not out.success:
        return None
    t = np.array(frames["t"])
    return Episode(
        episode_id=f"syn-{spec.variant}-{seed:06d}",
        source="synthetic",
        fps=FPS,
        task=spec.task,
        instruction=spec.instruction,
        timestamp=t - t[0],
        state=np.array(frames["state"], np.float32),
        wrench=np.array(frames["wrench"], np.float32),
        action=np.array(frames["action"], np.float32),
        front=np.stack(frames["front"]),
        wrist=np.stack(frames["wrist"]),
        depth=np.stack(frames["depth"]),
        episode_spec=json_safe(spec),
        outcome=asdict(out),
        quality={"health": "pass", "checks": {}},
        provenance={
            "generator": GENERATOR_VERSION,
            "seed": seed,
            "variant": spec.variant,
            "expert_log": expert.log,
            "expert_style": asdict(expert.style) if expert.style else None,
        },
    )


def json_safe(spec: EpisodeSpec) -> dict[str, Any]:
    import json

    return json.loads(spec.to_json())


def generate(out_dir: Path, variant: Variant, seeds: range | list[int], tasks: tuple[str, ...] | None = None, log_every: int = 25) -> list[Path]:
    env = PandaEnv(render=True, depth=True)
    paths = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        task = tasks[i % len(tasks)] if tasks else None
        spec = sample_episode(seed, variant, task)
        ep = generate_episode(env, spec, seed)
        assert ep is not None
        paths.append(write_episode(out_dir / f"{ep.episode_id}.h5", ep))
        if log_every and (i + 1) % log_every == 0:
            print(f"[synthetic] {i + 1}/{len(seeds)} episodes, {(time.time() - t0) / (i + 1):.2f} s/episode", flush=True)
    env.close_renderers()
    return paths
