"""ROS 2 policy runtime: camera + joint-state topics in, joint commands out.

* runs an exported ONNX policy (or a PyTorch bundle) through the same
  :class:`~rls.learn.runner.ChunkRunner` as the simulation benchmarks;
* checks every observation for freshness and cross-camera sync before acting,
  using the per-camera clock offsets estimated by the data pipeline (camera
  stamps are on the camera's clock, not the robot's), and tracks how those
  offsets drift while running: the calibrated offset holds at start-up, and a
  running median of message age (receive time minus stamp) follows the drift,
  since transport latency stays roughly constant while the camera clock does not;
* holds position instead of acting on stale data, and reports inference and
  end-to-end latency, stale ticks and dropped ticks on ``/diagnostics``.

Ticks come from ``/sim/privileged`` in lockstep simulation (one message per
control step); on hardware the same callback would run on a 15 Hz timer.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from PIL import Image
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String

from rls.learn.policies.base import PolicyBundle
from rls.learn.runner import ChunkRunner, OnnxModel
from rls.ros.driver import ARM_JOINTS, from_time, to_time


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_runtime")
        self.declare_parameters(
            "",
            [
                ("bundle", ""),
                ("onnx", ""),
                ("mode", "ensemble"),
                ("max_obs_age_s", 0.25),
                ("max_cam_skew_s", 0.12),
                ("clock_offsets", "{}"),  # JSON {camera: offset_s}, from the ingest clock model
                ("track_clock_drift", True),
                ("summary_path", ""),
                ("start_enabled", True),
                ("command_topic", "/joint_command"),  # the skill executor remaps this to /policy/joint_command
            ],
        )
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        bundle = PolicyBundle.load(Path(g("bundle")))
        model = OnnxModel(g("onnx")) if g("onnx") else None
        self.runner = ChunkRunner(bundle, model, mode=g("mode"))
        self.max_age = float(g("max_obs_age_s"))
        self.max_skew = float(g("max_cam_skew_s"))
        self.offsets: dict[str, float] = json.loads(g("clock_offsets"))
        self.track_drift = bool(g("track_clock_drift"))
        self.ages: dict[str, list[float]] = {"front": [], "wrist": []}
        self.age0: dict[str, float] = {}
        self.summary_path = g("summary_path")
        self.enabled = bool(g("start_enabled"))
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.latest: dict[str, tuple[float, Any]] = {}
        self.create_subscription(CompressedImage, "/cam_front/color/image_raw/compressed", lambda m: self._on_img("front", m), reliable)
        self.create_subscription(CompressedImage, "/cam_wrist/color/image_raw/compressed", lambda m: self._on_img("wrist", m), reliable)
        self.create_subscription(JointState, "/joint_states", self._on_joints, reliable)
        self.create_subscription(String, "/sim/privileged", self._on_tick, reliable)
        self.create_subscription(String, "/episode/events", self._on_event, latched)
        self.create_subscription(String, "/policy/instruction", self._on_instruction, latched)
        self.create_subscription(Bool, "/policy/enable", lambda m: setattr(self, "enabled", bool(m.data)), latched)
        self.cmd_pub = self.create_publisher(JointState, str(g("command_topic")), reliable)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.stats: dict[str, Any] = {"ticks": 0, "acted": 0, "stale": 0, "skewed": 0, "decode_errors": 0, "e2e_ms": [], "infer_ms": []}
        self.instruction = ""
        self.last_cmd: np.ndarray | None = None

    # ------------------------------------------------------------- inputs
    def clock_offset(self, cam: str, raw_stamp: float) -> float:
        """Calibrated offset, corrected for drift since start-up (see module docstring)."""
        off = self.offsets.get(f"cam_{cam}", 0.0)
        if not self.track_drift:
            return off
        ages = self.ages[cam]
        ages.append(self.get_clock().now().nanoseconds * 1e-9 - raw_stamp)
        del ages[:-31]
        if cam not in self.age0:
            if len(ages) >= 15:
                self.age0[cam] = float(np.median(ages))
            return off
        return off + self.age0[cam] - float(np.median(ages))

    def _on_img(self, cam: str, msg: CompressedImage) -> None:
        raw = from_time(msg.header.stamp)
        stamp = raw - self.clock_offset(cam, raw)
        try:
            img = np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
        except Exception:
            self.stats["decode_errors"] += 1
            return
        if 8 < img.mean() < 247:
            self.latest[cam] = (stamp, img)

    def _on_joints(self, msg: JointState) -> None:
        pos = dict(zip(msg.name, msg.position, strict=False))
        q = np.array([pos[j] for j in ARM_JOINTS] + [np.clip((pos.get("finger_joint1", 0) + pos.get("finger_joint2", 0)) / 0.08, 0, 1)], np.float32)
        self.latest["joints"] = (from_time(msg.header.stamp), q)

    def _on_event(self, msg: String) -> None:
        ev = json.loads(msg.data)
        if ev["event"] == "start":
            self.runner.reset(self.instruction or ev["spec"]["instruction"])

    def _on_instruction(self, msg: String) -> None:
        self.instruction = msg.data
        self.runner.reset(msg.data)

    # --------------------------------------------------------------- tick
    def _on_tick(self, msg: String) -> None:
        tick = json.loads(msg.data)
        t = float(tick["stamp"])
        self.stats["ticks"] += 1
        if not self.enabled:
            return
        t_wall = time.perf_counter()
        need = ("front", "wrist", "joints")
        fresh = all(k in self.latest and t - self.latest[k][0] <= self.max_age for k in need)
        skew = abs(self.latest["front"][0] - self.latest["wrist"][0]) if fresh else 0.0
        if not fresh or skew > self.max_skew:
            self.stats["stale" if not fresh else "skewed"] += 1
            # hold: never act on stale or inconsistent data
            if self.last_cmd is not None:
                self._publish(t, self.last_cmd)
            elif "joints" in self.latest:
                self._publish(t, np.concatenate([self.latest["joints"][1][:7], [1.0]]))
            return
        a = self.runner.act(self.latest["front"][1], self.latest["wrist"][1], self.latest["joints"][1])
        self.last_cmd = a
        self._publish(t, a)
        self.stats["acted"] += 1
        self.stats["infer_ms"].append(self.runner.latencies_ms[-1])
        self.stats["e2e_ms"].append((time.perf_counter() - t_wall) * 1e3)
        if self.stats["acted"] % 50 == 0:
            self._diagnostics()

    def _publish(self, t: float, a: np.ndarray) -> None:
        cmd = JointState()
        cmd.header.stamp = to_time(t)
        cmd.name = [*ARM_JOINTS, "gripper"]
        cmd.position = [float(v) for v in a]
        self.cmd_pub.publish(cmd)

    def summary(self) -> dict[str, Any]:
        s = dict(self.stats)
        s["clock_offsets_now"] = {
            c: round(self.offsets.get(f"cam_{c}", 0.0) + self.age0[c] - float(np.median(a)), 4) for c, a in self.ages.items() if c in self.age0 and a
        }
        for k in ("e2e_ms", "infer_ms"):
            v = np.array(s.pop(k) or [0.0])
            s[f"{k}_p50"] = float(np.percentile(v, 50))
            s[f"{k}_p95"] = float(np.percentile(v, 95))
            s[f"{k}_max"] = float(v.max())
        return s

    def _diagnostics(self) -> None:
        s = self.summary()
        st = DiagnosticStatus(name="policy_runtime", hardware_id="sim_panda")
        st.level = DiagnosticStatus.WARN if s["stale"] > 0.05 * max(s["ticks"], 1) else DiagnosticStatus.OK
        st.message = f"acted {s['acted']}/{s['ticks']} ticks"
        st.values = [KeyValue(key=k, value=f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in s.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.diag_pub.publish(arr)

    def write_summary(self) -> None:
        if self.summary_path:
            Path(self.summary_path).write_text(json.dumps(self.summary(), indent=1))


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_summary()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
