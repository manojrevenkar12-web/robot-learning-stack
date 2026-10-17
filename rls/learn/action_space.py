"""Action representations: what the policy predicts versus what the robot is sent.

``joint``: the commanded joint positions and gripper exactly as recorded (8-D).

``ee``: the commanded tool pose in the robot base frame, ``[x, y, z, cos 2ψ, sin 2ψ, gripper]``.
Every demonstration (scripted teleop and synthetic expert) keeps the gripper pointing down
and steers in Cartesian space, so the pose is recovered exactly from the recorded joint
commands by forward kinematics. The yaw is encoded as 2ψ because the parallel gripper is
symmetric under a half turn. At run time the predicted pose goes through damped
least-squares IK, warm-started at the measured joint state, to joint targets.

Why it matters: in joint space a 0.05 rad error at the shoulder is several centimetres at
the fingers, and small regression errors spread over 7 coupled joints; in tool space the
error is the grasp error, and the target is a near-linear function of where the cube is in
the image.
"""

from __future__ import annotations

import numpy as np

ACTION_DIMS = {"joint": 8, "ee": 6}


class EEActionSpace:
    dim = 6

    def __init__(self) -> None:
        from rls.sim.ik import IKSolver
        from rls.sim.scene import SceneSpec, build_model

        self.ik = IKSolver(build_model(SceneSpec(objects=())))

    def from_joint(self, actions: np.ndarray) -> np.ndarray:
        """(T, 8) joint commands + gripper -> (T, 6) tool poses + gripper."""
        out = np.zeros((len(actions), 6), np.float32)
        for i, a in enumerate(actions):
            pos, _ = self.ik.fk(a[:7])
            yaw = self.ik.hand_yaw(a[:7])
            out[i] = [pos[0], pos[1], pos[2], np.cos(2 * yaw), np.sin(2 * yaw), a[7]]
        return out

    def to_joint(self, ee: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """(6,) tool pose + gripper -> (8,) joint targets + gripper, IK warm-started at ``q_now``."""
        from rls.sim.ik import down_rotation

        yaw = 0.5 * float(np.arctan2(ee[4], ee[3]))
        cur = self.ik.hand_yaw(q_now)
        # the gripper is symmetric under a half turn: take the equivalent yaw closest to the current one
        yaw = min((yaw, yaw + np.pi, yaw - np.pi), key=lambda y: abs(np.angle(np.exp(1j * (y - cur)))))
        q, _ = self.ik.solve(np.asarray(q_now, np.float64), np.asarray(ee[:3], np.float64), down_rotation(yaw))
        return np.concatenate([q, [ee[5]]]).astype(np.float32)


def make(name: str) -> EEActionSpace | None:
    if name == "joint":
        return None
    if name == "ee":
        return EEActionSpace()
    raise ValueError(f"unknown action space {name!r}")
