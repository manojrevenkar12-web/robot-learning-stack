"""ROS 2 driver for the simulated Panda cell.

Publishes what a real cell publishes, with the timing and faults of
:mod:`rls.capture.faults`:

====================================  ====================================  =========
topic                                 type                                  rate
====================================  ====================================  =========
/clock                                rosgraph_msgs/Clock                   120 Hz
/joint_states                         sensor_msgs/JointState                120 Hz
/wrist_ft                             geometry_msgs/WrenchStamped           120 Hz
/cam_front/color/image_raw/compressed sensor_msgs/CompressedImage (jpeg)    15 Hz
/cam_front/depth/image_raw            sensor_msgs/Image (16UC1, mm)         15 Hz
/cam_front/color/camera_info          sensor_msgs/CameraInfo                15 Hz
/cam_wrist/color/image_raw/compressed sensor_msgs/CompressedImage (jpeg)    15 Hz
/cam_wrist/color/camera_info          sensor_msgs/CameraInfo                15 Hz
/tf_static                            tf2_msgs/TFMessage                    latched
/episode/events                       std_msgs/String (JSON)                per event
/sim/privileged                       std_msgs/String (JSON, operator only) 15 Hz
====================================  ====================================  =========

and executes ``/joint_command`` (sensor_msgs/JointState: 7 joint position
targets + ``gripper`` in [0, 1]). In lockstep mode the simulation waits for the
command answering each control tick, so recordings are reproducible regardless
of machine load; otherwise it paces against the wall clock.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, WrenchStamped
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from rls.capture.faults import FaultProfile, StreamSchedule, add_sensor_noise, corrupt_image
from rls.data.synthetic import depth_to_mm
from rls.sim.env import EpisodeSpec, PandaEnv, sample_episode
from rls.sim.variants import LAB, SPLITS, SYNTH_DR, with_tasks

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
VARIANTS = {"lab": LAB, "synthetic_dr": SYNTH_DR, **SPLITS}


def to_time(t: float) -> Time:
    # round once in integer nanoseconds, then split: splitting first and rounding the fraction
    # turns t = 1.9999999999 into (1 s, 0 ns), a stamp one second in the past (found by the
    # C++ sync monitor as a 1 s "gap" once per second on the 120 Hz streams)
    sec, ns = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=int(sec), nanosec=int(ns))


def from_time(t: Time) -> float:
    return t.sec + t.nanosec * 1e-9


def rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


def mj_cam_to_optical(mat: np.ndarray) -> np.ndarray:
    """MuJoCo camera frame (looks along -z, y up) -> ROS optical frame (z forward, y down)."""
    return mat @ np.diag([1.0, -1.0, -1.0])


def axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


class SimDriver(Node):
    def __init__(self) -> None:
        super().__init__("sim_driver")
        p = self.declare_parameters(
            "",
            [
                ("session_dir", "/tmp/rls_session"),
                ("variant", "lab"),
                ("tasks", "pick_place,stack"),
                ("seed_start", 0),
                ("n_episodes", 3),
                ("episode_specs", ""),  # optional JSON-lines file of EpisodeSpecs (benchmarks)
                ("fault_profile", "configs/faults/realistic.yaml"),
                ("fault_seed", 0),
                ("lockstep", True),
                ("realtime_factor", 0.0),
                ("max_steps", 330),
                ("settle_steps", 6),
                ("success_stop_steps", 0),  # >0: end the episode after this many consecutive successful ticks
                ("command_timeout_s", 5.0),
                ("start_delay_s", 0.5),
                ("wait_for_topics", "/joint_states,/cam_front/color/image_raw/compressed,/sim/privileged"),
                ("wait_timeout_s", 30.0),
            ],
        )
        del p
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.session_dir = Path(g("session_dir"))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.profile = FaultProfile.load(Path(g("fault_profile")))
        self.rng = np.random.default_rng(int(g("fault_seed")))
        self.lockstep = bool(g("lockstep"))
        self.rtf = float(g("realtime_factor"))
        self.max_steps = int(g("max_steps"))
        self.settle_steps = int(g("settle_steps"))
        self.success_stop = int(g("success_stop_steps"))
        self.command_timeout = float(g("command_timeout_s"))
        variant = with_tasks(VARIANTS[g("variant")], tuple(g("tasks").split(",")))
        if g("episode_specs"):
            self.specs = [EpisodeSpec.from_json(line) for line in Path(g("episode_specs")).read_text().splitlines() if line.strip()]
        else:
            self.specs = [sample_episode(int(g("seed_start")) + i, variant) for i in range(int(g("n_episodes")))]

        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = {
            "clock": self.create_publisher(Clock, "/clock", 10),
            "joint_states": self.create_publisher(JointState, "/joint_states", reliable),
            "wrist_ft": self.create_publisher(WrenchStamped, "/wrist_ft", reliable),
            "cam_front": self.create_publisher(CompressedImage, "/cam_front/color/image_raw/compressed", reliable),
            "cam_front_depth": self.create_publisher(Image, "/cam_front/depth/image_raw", reliable),
            "cam_front_info": self.create_publisher(CameraInfo, "/cam_front/color/camera_info", reliable),
            "cam_wrist": self.create_publisher(CompressedImage, "/cam_wrist/color/image_raw/compressed", reliable),
            "cam_wrist_info": self.create_publisher(CameraInfo, "/cam_wrist/color/camera_info", reliable),
            "tf_static": self.create_publisher(TFMessage, "/tf_static", latched),
            "events": self.create_publisher(String, "/episode/events", latched),
            "privileged": self.create_publisher(String, "/sim/privileged", reliable),
        }
        self.create_subscription(JointState, "/joint_command", self._on_command, reliable)
        self.commands: dict[int, np.ndarray] = {}
        self.last_command: np.ndarray | None = None
        self.env = PandaEnv(render=False)
        self.session_t = 1.0  # sim time 0 means "clock not started" to ROS
        self.truth: dict[str, Any] = {"profile": self.profile.to_dict(), "episodes": []}
        self.stats = {"command_timeouts": 0}

    # ----------------------------------------------------------------- inputs
    def _on_command(self, msg: JointState) -> None:
        key = int(round(from_time(msg.header.stamp) * 1e6))
        pos = dict(zip(msg.name, msg.position, strict=False))
        cmd = np.array([pos[j] for j in ARM_JOINTS] + [pos.get("gripper", 1.0)])
        self.commands[key] = cmd
        self.last_command = cmd

    # ---------------------------------------------------------------- helpers
    def _publish_tf(self, spec: EpisodeSpec, cal_error: dict[str, Any] | None) -> None:
        cam = self.env.camera_params("front")
        pos = np.array(cam["cam_pos"])
        rot = mj_cam_to_optical(np.array(cam["cam_mat"]))
        if cal_error:
            pos = pos + np.array(cal_error["dt"])
            rot = axis_angle(np.array(cal_error["axis"]), cal_error["deg"]) @ rot
        tf = TransformStamped()
        tf.header.stamp = to_time(self.session_t)
        tf.header.frame_id = "world"
        tf.child_frame_id = "cam_front_optical_frame"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(v) for v in pos)
        x, y, z, w = rot_to_quat(rot)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = x, y, z, w
        self.pub["tf_static"].publish(TFMessage(transforms=[tf]))

    def _camera_info(self, cam: str, stamp: float) -> CameraInfo:
        c = self.env.camera_params(cam)
        info = CameraInfo()
        info.header.stamp = to_time(stamp)
        info.header.frame_id = f"cam_{cam}_optical_frame"
        info.width, info.height = int(c["width"]), int(c["height"])
        info.k = [float(v) for v in c["K"]]
        info.p = [c["K"][0], 0.0, c["K"][2], 0.0, 0.0, c["K"][4], c["K"][5], 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        return info

    def _jpeg(self, rgb: np.ndarray) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="JPEG", quality=self.profile.jpeg_quality)
        return buf.getvalue()

    # ------------------------------------------------------------ main loop
    def wait_for_peers(self) -> None:
        """Block until every listed topic has enough subscribers (recorder, monitor, operator, policy).

        Entries are ``topic`` or ``topic:count``. Counts matter: with the live monitor running, "someone
        subscribes to the camera" is true before rosbag2 has subscribed, and the first frames of the
        session would be missing from the bag.
        """
        want: dict[str, int] = {}
        for entry in str(self.get_parameter("wait_for_topics").value).split(","):
            if entry:
                topic, _, n = entry.partition(":")
                want[topic] = int(n) if n else 1
        deadline = time.monotonic() + float(self.get_parameter("wait_timeout_s").value)
        while time.monotonic() < deadline:
            missing = [t for t, n in want.items() if self.count_subscribers(t) < n]
            if not missing:
                return
            # rosbag2 with --use-sim-time only starts subscribing once /clock ticks
            self.pub["clock"].publish(Clock(clock=to_time(self.session_t)))
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f"no subscribers on {missing} after waiting")

    def run(self) -> None:
        self.wait_for_peers()
        time.sleep(float(self.get_parameter("start_delay_s").value))
        for idx, spec in enumerate(self.specs):
            self._run_episode(idx, spec)
        (self.session_dir / "ground_truth.json").write_text(json.dumps(self.truth, indent=1, default=str))
        self.get_logger().info(f"session done: {len(self.specs)} episodes, {self.stats}")

    def _run_episode(self, idx: int, spec: EpisodeSpec) -> None:
        env = self.env
        env.render_enabled = False
        env.reset(spec)
        t0 = self.session_t
        ep_len_s = self.max_steps / 15.0
        scheds = {name: StreamSchedule(name, f, self.rng, t0, ep_len_s) for name, f in self.profile.streams.items() if name != "joint_command"}
        cal_error = None
        if self.profile.calibration.prob and self.rng.random() < self.profile.calibration.prob:
            axis = self.rng.normal(size=3)
            direction = self.rng.normal(size=3)
            cal_error = {
                "dt": (direction / np.linalg.norm(direction) * self.profile.calibration.translation_m).tolist(),
                "axis": axis.tolist(),
                "deg": float(self.profile.calibration.rotation_deg),
            }
        self._publish_tf(spec, cal_error)
        self.commands.clear()
        self._event({"event": "start", "episode_index": idx, "t": t0, "spec": json.loads(spec.to_json())})
        prev_img: dict[str, np.ndarray] = {}
        outbox: list[tuple[float, str, Any]] = []
        done_at: int | None = None
        succ_streak = 0
        status = {"done": False}

        def on_status(msg: String) -> None:
            status.update(json.loads(msg.data))

        sub = self.create_subscription(String, "/operator/status", on_status, 10)
        wall0 = time.monotonic()
        hold = np.concatenate([env.data.qpos[:7], [1.0]])  # type: ignore[union-attr]

        def substep(i: int) -> None:
            t = t0 + env.data.time  # type: ignore[union-attr]
            if i % 4 == 0:
                self.pub["clock"].publish(Clock(clock=to_time(t)))
            for name, s in scheds.items():
                while s.due(t):
                    cap = s.advance()
                    fate = s.fate(cap)
                    if fate in ("drop", "outage"):
                        continue
                    # sample the sensor NOW (capture time); publish later (transport latency)
                    msgs = self._capture(name, s.stamp(cap), fate, prev_img)
                    outbox.append((s.publish_time(cap), name, msgs))
            ready = [o for o in outbox if o[0] <= t]
            for o in ready:
                outbox.remove(o)
                for topic_key, m in o[2]:
                    self.pub[topic_key].publish(m)

        for step in range(self.max_steps + 1):
            t_step = t0 + env.data.time  # type: ignore[union-attr]
            priv = env.privileged()
            priv.update({"step": step, "stamp": t_step, "episode_index": idx})
            self.pub["privileged"].publish(String(data=json.dumps(priv)))
            key = int(round(t_step * 1e6))
            cmd = self._wait_command(key)
            if cmd is None:
                cmd = self.last_command if self.last_command is not None else hold
            rclpy.spin_once(self, timeout_sec=0.0)
            if status.get("done") and done_at is None:
                done_at = step
            succ_streak = succ_streak + 1 if self.success_stop and env.task_success() else 0
            if self.success_stop and succ_streak >= self.success_stop:
                break
            if step == self.max_steps or (done_at is not None and step - done_at >= self.settle_steps):
                break
            env.step(cmd, substep_callback=substep)
            if self.rtf > 0:
                lag = (env.data.time / self.rtf) - (time.monotonic() - wall0)  # type: ignore[union-attr]
                if lag > 0:
                    time.sleep(lag)
        # flush messages still in flight at the end of the episode
        t_end = t0 + env.data.time  # type: ignore[union-attr]
        for o in sorted(outbox, key=lambda o: o[0]):
            for topic_key, m in o[2]:
                self.pub[topic_key].publish(m)
        out = env.outcome()
        self.destroy_subscription(sub)
        self._event({"event": "end", "episode_index": idx, "t": t_end, "outcome": asdict(out), "final_state": env.privileged()})
        self.truth["episodes"].append(
            {
                "episode_index": idx,
                "t_start": t0,
                "t_end": t_end,
                "spec_seed": spec.seed,
                "task": spec.task,
                "outcome": asdict(out),
                "calibration_error": cal_error,
                "stream_events": {k: s.events for k, s in scheds.items()},
                "outages": {k: s.outage for k, s in scheds.items() if s.outage},
                "final_state": env.privileged(),
                "instruction": spec.instruction,
                "extra": spec.extra,
            }
        )
        self.session_t = t_end + 1.0  # one second between episodes, as an operator would reset
        for _ in range(15):  # keep /clock moving through the gap
            self.pub["clock"].publish(Clock(clock=to_time(t_end + _ / 15.0)))

    def _wait_command(self, key: int) -> np.ndarray | None:
        if not self.lockstep:
            rclpy.spin_once(self, timeout_sec=0.0)
            return self.commands.pop(key, None)
        deadline = time.monotonic() + self.command_timeout
        while time.monotonic() < deadline:
            if key in self.commands:
                return self.commands.pop(key)
            rclpy.spin_once(self, timeout_sec=0.01)
        self.stats["command_timeouts"] += 1
        self.get_logger().warning(f"no command for tick {key / 1e6:.3f}s; holding last command")
        return None

    def _event(self, payload: dict[str, Any]) -> None:
        self.pub["events"].publish(String(data=json.dumps(payload, default=str)))
        rclpy.spin_once(self, timeout_sec=0.05)

    def _capture(self, name: str, stamp: float, fate: str, prev_img: dict[str, np.ndarray]) -> list[tuple[str, Any]]:
        """Sample one sensor at the current sim state; returns (publisher key, message) pairs."""
        env = self.env
        d = env.data
        assert d is not None
        if name == "joint_states":
            m = JointState()
            m.header.stamp = to_time(stamp)
            m.name = [*ARM_JOINTS, "finger_joint1", "finger_joint2"]
            m.position = [float(v) for v in d.qpos[:9]]
            m.velocity = [float(v) for v in d.qvel[:9]]
            m.effort = [float(v) for v in d.qfrc_actuator[:9]]
            return [("joint_states", m)]
        if name == "wrist_ft":
            ft = env.proprio()["ft"]
            w = WrenchStamped()
            w.header.stamp = to_time(stamp)
            w.header.frame_id = "ft_sensor"
            w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = (float(v) for v in ft[:3])
            w.wrench.torque.x, w.wrench.torque.y, w.wrench.torque.z = (float(v) for v in ft[3:])
            return [("wrist_ft", w)]
        cam = name.split("_")[1]
        rgb = add_sensor_noise(env.render(cam), self.profile.image_noise_std, self.rng)
        payload: np.ndarray | bytes = rgb
        if fate == "corrupt":
            payload, _ = corrupt_image(rgb, prev_img.get(cam), self.rng)
        prev_img[cam] = rgb
        c = CompressedImage()
        c.header.stamp = to_time(stamp)
        c.header.frame_id = f"cam_{cam}_optical_frame"
        c.format = "rgb8; jpeg compressed bgr8"
        c.data = payload if isinstance(payload, bytes) else self._jpeg(payload)
        out: list[tuple[str, Any]] = [(name, c), (f"{name}_info", self._camera_info(cam, stamp))]
        if cam == "front":
            dm = depth_to_mm(env.render("front", depth=True))
            img = Image()
            img.header = c.header
            img.height, img.width = dm.shape
            img.encoding = "16UC1"
            img.step = dm.shape[1] * 2
            img.data = dm.tobytes()
            out.append(("cam_front_depth", img))
        return out


def main() -> None:
    rclpy.init()
    node = SimDriver()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
