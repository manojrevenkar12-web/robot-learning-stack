"""Differential inverse kinematics for the Panda tool centre point (damped least squares)."""

from __future__ import annotations

import mujoco
import numpy as np

from rls.sim.scene import HOME_Q, N_ARM


def down_rotation(yaw: float) -> np.ndarray:
    """Hand frame with the gripper pointing down and fingers sliding along ``yaw``."""
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum([np.cross(current[:, i], target[:, i]) for i in range(3)], axis=0)


class IKSolver:
    """Solves for arm joint targets on a private MjData, never touching the live sim."""

    def __init__(self, model: mujoco.MjModel, site: str = "grip_site", damping: float = 1e-3, nullspace_gain: float = 0.1) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.low = model.jnt_range[:N_ARM, 0].copy()
        self.high = model.jnt_range[:N_ARM, 1].copy()

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # mj_jacSite needs cdof
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def hand_yaw(self, q: np.ndarray) -> float:
        _, r = self.fk(q)
        return float(np.arctan2(r[1, 1], r[0, 1]))

    def solve(self, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 30, tol: float = 1e-4) -> tuple[np.ndarray, float]:
        q = np.array(q0, dtype=np.float64)
        err_norm = np.inf
        for _ in range(iters):
            p, r = self.fk(q)
            err = np.concatenate([pos - p, rotation_error(r, rot)])
            err_norm = float(np.linalg.norm(err[:3]))
            if err_norm < tol and np.linalg.norm(err[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
            j = np.vstack([self.jacp[:, :N_ARM], self.jacr[:, :N_ARM]])
            jjt = j @ j.T + self.damping * np.eye(6)
            dq = j.T @ np.linalg.solve(jjt, err)
            null = (np.eye(N_ARM) - np.linalg.pinv(j) @ j) @ (HOME_Q - q) * self.nullspace_gain
            q = np.clip(q + dq + null, self.low, self.high)
        return q, err_norm
