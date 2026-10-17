"""Gymnasium environment for RL on the Panda: reach-and-hold a 6-DoF pose target.

State-based (no images), 15 Hz joint-position-delta control, matching the ROS 2
driver's control rate. Domain randomization covers what differs between a
simulator and a real arm: actuator gains, joint damping, an unknown payload,
observation noise and actuation latency. A separate *target* dynamics
distribution (never trained on) stands in for the real robot in the transfer
study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from rls.sim.ik import down_rotation, rotation_error
from rls.sim.scene import HOME_Q, N_ARM, SceneSpec, build_model

CONTROL_DT = 1.0 / 15.0
MAX_DELTA = 0.12  # rad per control step
# with up to 2 ticks of actuation latency the state is Markov only if the policy sees its in-flight commands
ACTION_HISTORY = 2


@dataclass(frozen=True)
class Dynamics:
    kp_scale: tuple[float, float] = (1.0, 1.0)
    damping_scale: tuple[float, float] = (1.0, 1.0)
    payload_kg: tuple[float, float] = (0.0, 0.0)
    obs_noise: float = 0.0
    latency_steps: tuple[int, int] = (0, 0)


NOMINAL = Dynamics()
RANDOMIZED = Dynamics(kp_scale=(0.6, 1.4), damping_scale=(0.5, 2.0), payload_kg=(0.0, 1.5), obs_noise=0.01, latency_steps=(0, 2))
# held-out "real robot": weaker actuators, heavier tool, more latency than the nominal model
TARGET = Dynamics(kp_scale=(0.5, 0.65), damping_scale=(1.5, 2.5), payload_kg=(1.0, 1.6), obs_noise=0.01, latency_steps=(2, 2))


class PandaReachEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012 (gymnasium API)

    def __init__(self, dynamics: Dynamics = NOMINAL, horizon: int = 60, seed: int | None = None) -> None:
        super().__init__()
        self.model = build_model(SceneSpec(objects=()))
        self.data = mujoco.MjData(self.model)
        self.hand = self.model.body("hand").id
        self.site = self.model.site("grip_site").id
        self.substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        self.dyn = dynamics
        self.horizon = horizon
        self.base_kp = self.model.actuator_gainprm[:N_ARM, 0].copy()
        self.base_bias = self.model.actuator_biasprm[:N_ARM, :3].copy()
        self.base_damp = self.model.dof_damping[:N_ARM].copy()
        self.base_mass = float(self.model.body_mass[self.hand])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (7 + 7 + 3 + 3 + 7 * ACTION_HISTORY,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.rng = np.random.default_rng(seed)

    def _apply_dynamics(self) -> None:
        d, r = self.dyn, self.rng
        kp = r.uniform(*d.kp_scale)
        m = self.model
        m.actuator_gainprm[:N_ARM, 0] = self.base_kp * kp
        m.actuator_biasprm[:N_ARM, 1] = self.base_bias[:, 1] * kp
        m.actuator_biasprm[:N_ARM, 2] = self.base_bias[:, 2] * kp
        m.dof_damping[:N_ARM] = self.base_damp * r.uniform(*d.damping_scale)
        m.body_mass[self.hand] = self.base_mass + r.uniform(*d.payload_kg)
        self.latency = int(r.integers(d.latency_steps[0], d.latency_steps[1] + 1))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._apply_dynamics()
        mujoco.mj_resetData(self.model, self.data)
        q0 = HOME_Q + self.rng.uniform(-0.15, 0.15, N_ARM)
        self.data.qpos[:N_ARM] = q0
        self.data.ctrl[:N_ARM] = q0
        self.data.ctrl[N_ARM] = 255
        mujoco.mj_forward(self.model, self.data)
        self.goal_pos = np.array([self.rng.uniform(0.35, 0.65), self.rng.uniform(-0.25, 0.25), self.rng.uniform(0.08, 0.35)])
        self.goal_rot = down_rotation(self.rng.uniform(-1.2, 1.2))
        self.q_cmd = q0.copy()
        self.prev_action = np.zeros(N_ARM)
        self.history = [np.zeros(N_ARM) for _ in range(ACTION_HISTORY)]
        self.pending: list[np.ndarray] = []
        self.t = 0
        self.hold = 0
        return self._obs(), {}

    def _errors(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.data.site_xpos[self.site]
        r = self.data.site_xmat[self.site].reshape(3, 3)
        return self.goal_pos - p, rotation_error(r, self.goal_rot)

    def _obs(self) -> np.ndarray:
        pe, re = self._errors()
        q = self.data.qpos[:N_ARM] + self.rng.normal(0, self.dyn.obs_noise, N_ARM)
        qd = self.data.qvel[:N_ARM] * 0.1
        return np.concatenate([q - HOME_Q, qd, pe, re, *self.history]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.pending.append(a)
        applied = self.pending.pop(0) if len(self.pending) > self.latency else np.zeros(N_ARM)
        self.q_cmd = np.clip(self.q_cmd + applied * MAX_DELTA, self.model.jnt_range[:N_ARM, 0], self.model.jnt_range[:N_ARM, 1])
        self.data.ctrl[:N_ARM] = self.q_cmd
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        pe, re = self._errors()
        dp, dr = float(np.linalg.norm(pe)), float(np.linalg.norm(re))
        reached = dp < 0.02 and dr < 0.1
        self.hold = self.hold + 1 if reached else 0
        reward = -dp - 0.3 * dr - 0.02 * float(np.square(a - self.prev_action).sum()) + (0.5 if reached else 0.0)
        self.prev_action = a
        self.history = [a, *self.history[:-1]]
        success = self.hold >= 5
        done = success
        trunc = self.t >= self.horizon
        return self._obs(), reward + (5.0 if success else 0.0), done, trunc, {"pos_err": dp, "rot_err": dr, "success": success}


def make(dynamics: Dynamics, seed: int):
    return lambda: PandaReachEnv(dynamics, seed=seed)


def replace_dynamics(d: Dynamics, **kw: Any) -> Dynamics:
    return replace(d, **kw)
