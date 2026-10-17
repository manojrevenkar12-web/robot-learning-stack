"""Skill executor: multi-step instructions -> planned skills -> learned policy + scripted moves.

For each episode it perceives the scene (front RGB-D), plans the instruction
into skills (:mod:`rls.agent.planner`), and runs them one by one: the learned
policy executes each skill (conditioned on that step's instruction), RGB-D
perception checks the postcondition, a scripted move returns the arm home
between skills, and a failed skill is retried once. The executor is the only
publisher on ``/joint_command``; it relays the policy's commands from
``/policy/joint_command`` while a skill runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

from rls.agent.planner import PlanError, Step, default_planner, plan_to_json
from rls.data.calib import quat_xyzw_to_mat
from rls.perception.cubes import detect, in_bin, stacked
from rls.ros.driver import ARM_JOINTS, to_time
from rls.sim.scene import HOME_Q

BIN_XY = (0.45, -0.28)  # workcell fixture position (cell configuration, not ground truth)


class ExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("skill_executor")
        self.declare_parameters("", [("skill_timeout_ticks", 240), ("confirm_ticks", 8), ("home_ticks", 20), ("max_retries", 1), ("report_path", "")])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.skill_timeout = int(g("skill_timeout_ticks"))
        self.confirm = int(g("confirm_ticks"))
        self.home_ticks = int(g("home_ticks"))
        self.max_retries = int(g("max_retries"))
        self.report_path = str(g("report_path"))
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(JointState, "/joint_command", reliable)
        self.instr_pub = self.create_publisher(String, "/policy/instruction", latched)
        self.enable_pub = self.create_publisher(Bool, "/policy/enable", latched)
        self.status_pub = self.create_publisher(String, "/operator/status", 10)
        self.create_subscription(JointState, "/policy/joint_command", self._relay, reliable)
        self.create_subscription(String, "/sim/privileged", self._on_tick, reliable)
        self.create_subscription(String, "/episode/events", self._on_event, latched)
        self.create_subscription(JointState, "/joint_states", self._on_joints, reliable)
        self.create_subscription(CompressedImage, "/cam_front/color/image_raw/compressed", self._on_rgb, reliable)
        self.create_subscription(Image, "/cam_front/depth/image_raw", self._on_depth, reliable)
        self.create_subscription(CameraInfo, "/cam_front/color/camera_info", lambda m: setattr(self, "K", list(m.k)), reliable)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf, latched)
        self.planner = default_planner()
        self.K: list[float] | None = None
        self.tf: tuple[np.ndarray, np.ndarray] | None = None
        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.q: np.ndarray | None = None
        self.reports: list[dict[str, Any]] = []
        self.phase = "idle"
        self.enable_pub.publish(Bool(data=False))

    # ------------------------------------------------------------- sensing
    def _on_rgb(self, m: CompressedImage) -> None:
        import io

        from PIL import Image as PILImage

        try:
            self.rgb = np.asarray(PILImage.open(io.BytesIO(bytes(m.data))).convert("RGB"))
        except Exception:
            pass

    def _on_depth(self, m: Image) -> None:
        self.depth = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.width)

    def _on_tf(self, m: TFMessage) -> None:
        for tr in m.transforms:
            if tr.child_frame_id == "cam_front_optical_frame":
                t, r = tr.transform.translation, tr.transform.rotation
                self.tf = (np.array([t.x, t.y, t.z]), quat_xyzw_to_mat([r.x, r.y, r.z, r.w]))

    def _on_joints(self, m: JointState) -> None:
        pos = dict(zip(m.name, m.position, strict=False))
        self.q = np.array([pos[j] for j in ARM_JOINTS] + [pos.get("finger_joint1", 0) + pos.get("finger_joint2", 0)])

    def perceive(self) -> dict[str, np.ndarray]:
        if self.rgb is None or self.depth is None or self.K is None or self.tf is None:
            return {}
        return {c: d.position for c, d in detect(self.rgb, self.depth, self.K, self.tf[0], self.tf[1]).items()}

    def postcondition(self, step: Step) -> bool:
        seen = self.perceive()
        obj = seen.get(step.args["object"])
        gripper_open = self.q is not None and self.q[7] > 0.06
        if obj is None or not gripper_open:
            return False
        if step.skill == "pick_place":
            return in_bin(obj, BIN_XY)
        sup = seen.get(step.args["support"])
        return sup is not None and stacked(obj, sup)

    # ----------------------------------------------------------- episodes
    def _on_event(self, m: String) -> None:
        ev = json.loads(m.data)
        if ev["event"] == "start":
            self.goal = ev["spec"]["instruction"]
            self.phase = "plan"
            self.report: dict[str, Any] = {"episode_index": ev["episode_index"], "instruction": self.goal, "steps": [], "planner": type(self.planner).__name__}
        elif ev["event"] == "end" and self.phase != "idle":
            self.report["final_state"] = ev.get("final_state")
            self.reports.append(self.report)
            self.phase = "idle"

    def _publish(self, t: float, a: np.ndarray) -> None:
        cmd = JointState()
        cmd.header.stamp = to_time(t)
        cmd.name = [*ARM_JOINTS, "gripper"]
        cmd.position = [float(v) for v in a]
        self.cmd_pub.publish(cmd)

    def _relay(self, m: JointState) -> None:
        if self.phase == "skill":
            self.last_cmd = np.array(m.position)
            self.cmd_pub.publish(m)

    def _start_skill(self) -> None:
        st = self.plan[self.step_i]
        self.instr_pub.publish(String(data=st.instruction))
        self.enable_pub.publish(Bool(data=True))
        self.phase, self.ticks, self.ok = "skill", 0, 0

    def _start_home(self, t: float) -> None:
        self.enable_pub.publish(Bool(data=False))
        self.phase, self.ticks = "home", 0
        self.home_from = self.last_cmd if getattr(self, "last_cmd", None) is not None else np.concatenate([self.q[:7], [1.0]])  # type: ignore[index]

    def _on_tick(self, m: String) -> None:
        tick = json.loads(m.data)
        t = float(tick["stamp"])
        hold = np.concatenate([self.q[:7], [1.0]]) if self.q is not None else np.concatenate([HOME_Q, [1.0]])
        if self.phase == "plan":
            seen = set(self.perceive())
            if not seen:
                self._publish(t, hold)
                return
            try:
                self.plan = self.planner.plan(self.goal, seen)
            except PlanError as exc:
                self.report["plan_error"] = str(exc)
                self._finish(t, hold)
                return
            self.report["plan"] = json.loads(plan_to_json(self.plan))
            self.step_i, self.retries, self.t_step0 = 0, 0, time.monotonic()
            self._publish(t, hold)
            self._start_skill()
        elif self.phase == "skill":
            self.ticks += 1
            self.ok = self.ok + 1 if self.postcondition(self.plan[self.step_i]) else 0
            if self.ok >= self.confirm:
                self._record_step("success")
                self.step_i += 1
                self.retries = 0
                self._start_home(t)
            elif self.ticks >= self.skill_timeout:
                if self.retries < self.max_retries:
                    self.retries += 1
                    self.report["steps"].append({"step": self.step_i, "instruction": self.plan[self.step_i].instruction, "result": "retry"})
                    self._start_home(t)
                else:
                    self._record_step("failed")
                    self.step_i = len(self.plan)  # abort: later steps depend on this one
                    self._start_home(t)
        elif self.phase == "home":
            self.ticks += 1
            s = min(1.0, self.ticks / self.home_ticks)
            target = np.concatenate([HOME_Q, [1.0]])
            a = self.home_from + (target - self.home_from) * (3 * s * s - 2 * s * s * s)
            self.last_cmd = a
            self._publish(t, a)
            if self.ticks >= self.home_ticks + 3:
                if self.step_i < len(self.plan):
                    self._start_skill()
                else:
                    self._finish(t, a)
        elif self.phase == "done":
            self._publish(t, hold)

    def _record_step(self, result: str) -> None:
        st = self.plan[self.step_i]
        self.report["steps"].append({"step": self.step_i, "instruction": st.instruction, "result": result, "ticks": self.ticks, "retries": self.retries})

    def _finish(self, t: float, hold: np.ndarray) -> None:
        self.enable_pub.publish(Bool(data=False))
        self.phase = "done"
        self._publish(t, hold)
        self.status_pub.publish(String(data=json.dumps({"done": True})))

    def write_report(self) -> None:
        if self.report_path:
            Path(self.report_path).write_text(json.dumps(self.reports, indent=1, default=str))


def main() -> None:
    rclpy.init()
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_report()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
