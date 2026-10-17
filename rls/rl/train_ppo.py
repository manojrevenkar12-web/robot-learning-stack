"""PPO with and without domain randomization, evaluated on held-out "target" dynamics.

    python -m rls.rl.train_ppo --steps 400000 --out runs/rl

Trains two policies (Stable-Baselines3 PPO): one on the nominal model, one with
domain randomization. Both are evaluated on the nominal model and on the
held-out target dynamics, then the actor is exported to ONNX and checked
against PyTorch.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rls.learn.benchmark import wilson
from rls.rl.reach_env import NOMINAL, RANDOMIZED, TARGET, Dynamics, PandaReachEnv, make


def evaluate(policy_fn, dynamics: Dynamics, episodes: int = 100, seed: int = 10_000) -> dict[str, Any]:
    env = PandaReachEnv(dynamics)
    succ, perr, steps = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        info: dict = {}
        for _ in range(env.horizon):
            obs, _, done, trunc, info = env.step(policy_fn(obs))
            if done or trunc:
                break
        succ.append(bool(info.get("success")))
        perr.append(info.get("pos_err", np.nan))
        steps.append(env.t)
    k = int(np.sum(succ))
    return {
        "episodes": episodes,
        "successes": k,
        "ci95": wilson(k, episodes),
        "success_rate": float(np.mean(succ)),
        "final_pos_err_mm_median": float(np.median(perr) * 1000),
        "steps_mean": float(np.mean(steps)),
    }


class Actor(torch.nn.Module):
    """Deterministic actor extracted from an SB3 PPO policy, for ONNX export."""

    def __init__(self, model: PPO) -> None:
        super().__init__()
        p = model.policy
        self.features = p.features_extractor
        self.mlp = p.mlp_extractor.policy_net
        self.action_net = p.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.action_net(self.mlp(self.features(obs))), -1.0, 1.0)


def export_actor(model: PPO, path: Path) -> dict[str, float]:
    import onnxruntime as ort

    actor = Actor(model).eval()
    obs_shape = model.observation_space.shape
    assert obs_shape is not None
    x = torch.zeros(1, obs_shape[0])
    torch.onnx.export(
        actor, (x,), str(path), input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "b"}, "action": {0: "b"}}, opset_version=17, dynamo=False
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, x.shape[1])).astype(np.float32)
    ref = model.predict(obs, deterministic=True)[0]
    got = sess.run(["action"], {"obs": obs})[0]
    t0 = time.perf_counter()
    for i in range(200):
        sess.run(["action"], {"obs": obs[i % 64 : i % 64 + 1]})
    return {"max_abs_diff": float(np.abs(ref - got).max()), "onnx_mean_ms": (time.perf_counter() - t0) / 200 * 1e3}


def train_one(name: str, dyn: Dynamics, steps: int, out: Path, seed: int = 0, n_envs: int = 4) -> PPO:
    torch.set_num_threads(1)
    venv = VecMonitor(DummyVecEnv([make(dyn, seed * 100 + i) for i in range(n_envs)]))
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        learning_rate=3e-4,
        gamma=0.97,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        verbose=0,
    )
    t0 = time.time()
    model.learn(total_timesteps=steps)
    model.save(out / f"{name}.zip")
    print(f"[rl] {name}: {steps} steps in {time.time() - t0:.0f}s", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000, help="nominal policy")
    ap.add_argument("--dr-steps", type=int, default=1_000_000, help="domain-randomized policy (harder task, needs more samples)")
    ap.add_argument("--out", type=Path, default=Path("runs/rl"))
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name, dyn, steps in (("ppo_nominal", NOMINAL, a.steps), ("ppo_domain_randomized", RANDOMIZED, a.dr_steps)):
        model = train_one(name, dyn, steps, a.out)
        fn = lambda o, m=model: m.predict(o, deterministic=True)[0]  # noqa: E731
        results[name] = {
            "train_steps": steps,
            "eval_nominal": evaluate(fn, NOMINAL, a.episodes),
            "eval_randomized": evaluate(fn, RANDOMIZED, a.episodes),
            "eval_target": evaluate(fn, TARGET, a.episodes),
            "onnx": export_actor(model, a.out / f"{name}.onnx"),
        }
        print(json.dumps({name: results[name]}, indent=1), flush=True)
    (a.out / "rl_results.json").write_text(
        json.dumps({"dynamics": {"nominal": asdict(NOMINAL), "randomized": asdict(RANDOMIZED), "target": asdict(TARGET)}, "results": results}, indent=1)
    )


if __name__ == "__main__":
    main()
