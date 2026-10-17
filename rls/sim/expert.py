"""Scripted teleoperator: produces human-like demonstrations from privileged state.

It plays the role of the human operator during data collection. Each episode
gets an operator "style" (speed, precision) so demonstrations are not all
identical, and occasionally a sloppy grasp that fails and is recovered from,
which is the kind of data imitation learning needs to learn recoveries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from rls.sim.env import EpisodeSpec
from rls.sim.ik import IKSolver, down_rotation
from rls.sim.scene import BIN_WALL_H, HOME_Q, N_ARM

CONTROL_DT = 1.0 / 15.0


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Style:
    speed: float = 0.22  # m/s for Cartesian moves
    noise: float = 0.003  # waypoint noise (m)
    grasp_noise: float = 0.004
    sloppy_prob: float = 0.08
    pause_prob: float = 0.1

    @staticmethod
    def sample(rng: np.random.Generator) -> Style:
        return Style(speed=float(rng.uniform(0.16, 0.28)), noise=float(rng.uniform(0.001, 0.004)))


@dataclass
class _Segment:
    start: np.ndarray
    end: np.ndarray
    yaw0: float
    yaw1: float
    grip: float
    steps: int
    t: int = 0


@dataclass
class ScriptedExpert:
    spec: EpisodeSpec
    model: mujoco.MjModel
    seed: int = 0
    style: Style | None = None
    max_retries: int = 2
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed + 7919)
        self.style = self.style or Style.sample(self.rng)
        self.ik = IKSolver(self.model)
        self.q_cmd = HOME_Q.copy()
        self.grip_cmd = 1.0
        self.phase = "start"
        self.queue: list[_Segment] = []
        self.retries = 0
        self.done = False
        self.hold = 0
        self.pos_cmd, _ = self.ik.fk(self.q_cmd)
        self.yaw_cmd = self.ik.hand_yaw(self.q_cmd)
        self.target_h = next(o.half_size for o in self.spec.scene.objects if o.name == self.spec.target)

    # --------------------------------------------------------------- planning
    def _move(self, end: np.ndarray, yaw: float, grip: float, speed_scale: float = 1.0, min_t: float = 0.4) -> None:
        assert self.style is not None
        start = self.queue[-1].end if self.queue else self.pos_cmd
        yaw0 = self.queue[-1].yaw1 if self.queue else self.yaw_cmd
        dist = float(np.linalg.norm(end - start))
        t = max(min_t, dist / (self.style.speed * speed_scale))
        self.queue.append(_Segment(start.copy(), end.copy(), yaw0, yaw0 + wrap(yaw - yaw0), grip, max(1, int(round(t / CONTROL_DT)))))

    def _wait(self, seconds: float, grip: float) -> None:
        end = self.queue[-1].end if self.queue else self.pos_cmd
        yaw = self.queue[-1].yaw1 if self.queue else self.yaw_cmd
        self.queue.append(_Segment(end.copy(), end.copy(), yaw, yaw, grip, max(1, int(round(seconds / CONTROL_DT)))))

    def _grasp_yaw(self, obj_yaw: float) -> float:
        cands = [obj_yaw + k * np.pi / 2 for k in range(-4, 5)]
        return float(min(cands, key=lambda c: abs(wrap(c - self.yaw_cmd))))

    def _noise(self, scale: float) -> np.ndarray:
        n = self.rng.normal(0, scale, 3)
        n[2] *= 0.3
        return n

    def _plan_pick(self, state: dict[str, Any]) -> None:
        assert self.style is not None
        x, y, z, yaw = state["objects"][self.spec.target]
        gyaw = self._grasp_yaw(yaw)
        offset = self._noise(self.style.grasp_noise)
        if self.rng.random() < self.style.sloppy_prob:
            perp = np.array([-np.sin(gyaw), np.cos(gyaw), 0.0])  # across the fingers: pads catch only the cube edge
            offset = offset + perp * self.rng.choice([-1, 1]) * self.rng.uniform(0.028, 0.04)
            self.log.append("sloppy_grasp")
        above = np.array([x, y, z + 0.10]) + self._noise(self.style.noise)
        grasp = np.array([x, y, z - 0.004]) + offset
        grasp[2] = max(grasp[2], 0.012)
        self._move(above, gyaw, 1.0)
        if self.rng.random() < self.style.pause_prob:
            self._wait(self.rng.uniform(0.2, 0.6), 1.0)
        self._move(grasp, gyaw, 1.0, speed_scale=0.5)
        self._wait(0.5, 0.0)
        self._move(np.array([grasp[0], grasp[1], 0.17]), gyaw, 0.0, speed_scale=0.7)
        self.phase = "picking"

    def _plan_place(self, state: dict[str, Any]) -> None:
        assert self.style is not None
        if self.spec.task == "pick_place":
            gx, gy = state["bin"]
            gz = BIN_WALL_H + self.target_h + 0.03
            goal_yaw = self.queue[-1].yaw1 if self.queue else self.yaw_cmd
        else:
            assert self.spec.support is not None
            sx, sy, sz, syaw = state["objects"][self.spec.support]
            hs = next(o.half_size for o in self.spec.scene.objects if o.name == self.spec.support)
            gx, gy = sx, sy
            gz = sz + hs + self.target_h + 0.006
            goal_yaw = self._grasp_yaw(syaw)
        g = np.array([gx, gy, gz]) + self._noise(self.style.noise * 0.6)
        self._move(np.array([g[0], g[1], 0.17]), goal_yaw, 0.0)
        self._move(g, goal_yaw, 0.0, speed_scale=0.5)
        self._wait(0.3, 0.0)
        self._wait(0.5, 1.0)
        self._move(np.array([g[0], g[1], 0.2]), goal_yaw, 1.0, speed_scale=0.7)
        self._wait(0.4, 1.0)
        self.phase = "placing"

    # ---------------------------------------------------------------- acting
    def act(self, state: dict[str, Any]) -> np.ndarray:
        """``state``: ``objects`` {name: [x, y, z, yaw]}, ``bin`` [x, y]. Returns 8-dim action."""
        if not self.queue and not self.done:
            if self.phase == "start":
                self._plan_pick(state)
            elif self.phase == "picking":
                lifted = state["objects"][self.spec.target][2] > self.target_h + 0.04
                if lifted:
                    self._plan_place(state)
                elif self.retries < self.max_retries:
                    self.retries += 1
                    self.log.append("retry_grasp")
                    self._move(self.pos_cmd + np.array([0, 0, 0.0]), self.yaw_cmd, 1.0, min_t=0.3)
                    self._plan_pick(state)
                else:
                    self.log.append("gave_up")
                    self.done = True
            elif self.phase == "placing":
                self.done = True
        if self.queue:
            seg = self.queue[0]
            seg.t += 1
            s = min_jerk(seg.t / seg.steps)
            self.pos_cmd = seg.start + s * (seg.end - seg.start)
            self.yaw_cmd = seg.yaw0 + s * (seg.yaw1 - seg.yaw0)
            self.grip_cmd = seg.grip
            if seg.t >= seg.steps:
                self.queue.pop(0)
            self.q_cmd, _ = self.ik.solve(self.q_cmd, self.pos_cmd, down_rotation(self.yaw_cmd))
        return np.concatenate([self.q_cmd, [self.grip_cmd]])


def run_expert_episode(env, spec: EpisodeSpec, seed: int, max_steps: int = 330, on_step=None):
    """Roll out the expert in ``env``; returns (actions, outcome, expert log)."""
    obs = env.reset(spec)
    expert = ScriptedExpert(spec, env.model, seed=seed)
    actions = []
    settle = 0
    for _ in range(max_steps):
        a = expert.act(env.privileged())
        if on_step is not None:
            on_step(obs, a)
        actions.append(a)
        obs = env.step(a)
        if expert.done:
            settle += 1
            if settle > 5:
                break
    return np.array(actions), env.outcome(), expert.log


assert N_ARM == 7
