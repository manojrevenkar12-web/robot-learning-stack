"""Extrinsic sanity check for the front RGB-D camera.

Project a point on the robot hand (from joint states + kinematics) into the
front camera using the *published* intrinsics and extrinsics, and compare the
measured depth there with the point's expected depth. With a good calibration
the pixel lands on the hand and the residual is a few centimetres (the point
sits inside the hand housing); with a bumped camera it lands on the table or
the background and the residual jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rls.sim.scene import N_ARM, SceneSpec, build_model


def quat_xyzw_to_mat(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class HandKinematics:
    """Forward kinematics of points on the hand, using the robot model only."""

    POINT_IN_HAND = np.array([0.0, 0.0, 0.03])

    def __init__(self) -> None:
        self.model = build_model(SceneSpec(objects=()))
        self.data = mujoco.MjData(self.model)
        self.hand = self.model.body("hand").id

    def hand_point(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:N_ARM] = q[:N_ARM]
        mujoco.mj_kinematics(self.model, self.data)
        r = self.data.xmat[self.hand].reshape(3, 3)
        return self.data.xpos[self.hand] + r @ self.POINT_IN_HAND


@dataclass
class ExtrinsicCheck:
    median_residual_m: float
    p90_abs_residual_m: float
    frac_off_hand: float
    n_frames: int

    # The probe point sits ~5 cm inside the hand housing, so a correct calibration
    # reads about -0.048 m (measured over clean recordings, spread +-0.012 m).
    EXPECTED_RESIDUAL_M = -0.048
    TOLERANCE_M = 0.035
    MAX_OFF_HAND = 0.2

    @property
    def suspect(self) -> bool:
        if self.n_frames < 10:
            return False
        off_expected = abs(self.median_residual_m - self.EXPECTED_RESIDUAL_M) > self.TOLERANCE_M
        return bool(off_expected or self.frac_off_hand > self.MAX_OFF_HAND)


def check_extrinsics(
    hand_points_world: np.ndarray,
    depth_frames_mm: list[np.ndarray],
    k: list[float],
    t_world_cam: np.ndarray,
    r_world_cam: np.ndarray,
    window: int = 2,
    off_hand_m: float = 0.08,
) -> ExtrinsicCheck:
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    res = []
    for p_w, depth in zip(hand_points_world, depth_frames_mm, strict=True):
        p_c = r_world_cam.T @ (p_w - t_world_cam)
        if p_c[2] <= 0.05:
            continue
        u, v = fx * p_c[0] / p_c[2] + cx, fy * p_c[1] / p_c[2] + cy
        ui, vi = int(round(u)), int(round(v))
        h, w = depth.shape
        if not (window <= ui < w - window and window <= vi < h - window):
            continue
        patch = depth[vi - window : vi + window + 1, ui - window : ui + window + 1].astype(np.float64) / 1000.0
        patch = patch[patch > 0]
        if patch.size == 0:
            continue
        res.append(float(np.median(patch)) - float(p_c[2]))
    if not res:
        return ExtrinsicCheck(float("nan"), float("nan"), 1.0, 0)
    r = np.array(res)
    return ExtrinsicCheck(float(np.median(r)), float(np.percentile(np.abs(r), 90)), float(np.mean(np.abs(r) > off_hand_m)), len(r))
