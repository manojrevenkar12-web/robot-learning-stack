"""Read ROS 2 MCAP bags without a ROS installation (mcap + mcap-ros2-support).

Messages are reduced to plain arrays per topic. Images stay as encoded bytes
until a frame is actually selected, so reading a session is cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

TOPICS = {
    "joint_states": "/joint_states",
    "wrist_ft": "/wrist_ft",
    "cam_front": "/cam_front/color/image_raw/compressed",
    "cam_front_depth": "/cam_front/depth/image_raw",
    "cam_front_info": "/cam_front/color/camera_info",
    "cam_wrist": "/cam_wrist/color/image_raw/compressed",
    "cam_wrist_info": "/cam_wrist/color/camera_info",
    "tf_static": "/tf_static",
    "joint_command": "/joint_command",
    "events": "/episode/events",
}


def _t(stamp: Any) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


@dataclass
class Stream:
    stamp: list[float] = field(default_factory=list)  # header stamp (sensor clock)
    log: list[float] = field(default_factory=list)  # bag receive time (robot/sim clock)
    data: list[Any] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.stamp), np.asarray(self.log)


@dataclass
class Session:
    path: Path
    streams: dict[str, Stream]
    events: list[dict[str, Any]]
    camera_info: dict[str, dict[str, Any]]
    tf: dict[str, dict[str, Any]]
    tf_history: list[tuple[float, dict[str, Any]]]
    read_errors: list[str] = field(default_factory=list)  # e.g. a bag truncated by a crash or full disk

    def episodes(self) -> list[dict[str, Any]]:
        """Pair start/end events into episode windows."""
        starts = {e["episode_index"]: e for e in self.events if e["event"] == "start"}
        ends = {e["episode_index"]: e for e in self.events if e["event"] == "end"}
        last_t = max((s.log[-1] for s in self.streams.values() if s.log), default=0.0)
        out = []
        for idx, s in starts.items():
            e = ends.get(idx)
            out.append(
                {
                    "episode_index": idx,
                    "t_start": s["t"],
                    "t_end": e["t"] if e else last_t,
                    "spec": s["spec"],
                    "outcome": e.get("outcome", {}) if e else {},
                    "complete": e is not None,  # no end event: recording stopped mid-episode
                }
            )
        return sorted(out, key=lambda x: x["t_start"])


def _iter_tolerant(reader: Any):
    """Decoded messages in file order. Reading without the summary section works on
    bags whose writer died before finalising the file."""
    yield from reader.iter_decoded_messages(log_time_order=False)


def read_session(bag_dir: Path) -> Session:
    files = sorted(Path(bag_dir).glob("*.mcap")) if Path(bag_dir).is_dir() else [Path(bag_dir)]
    if not files:
        raise FileNotFoundError(f"no .mcap files in {bag_dir}")
    rev = {v: k for k, v in TOPICS.items()}
    streams: dict[str, Stream] = {k: Stream() for k in TOPICS}
    events: list[dict[str, Any]] = []
    cam_info: dict[str, dict[str, Any]] = {}
    tf: dict[str, dict[str, Any]] = {}
    tf_hist: list[tuple[float, dict[str, Any]]] = []
    errors: list[str] = []
    for f in files:
        with open(f, "rb") as fh:
            # stream the file front to back (no footer/summary needed): a truncated bag still yields everything before the damage
            reader = NonSeekingReader(fh, decoder_factories=[DecoderFactory()])
            try:
                for _schema, channel, message, msg in _iter_tolerant(reader):
                    key = rev.get(channel.topic)
                    if key is None:
                        continue
                    log_t = message.log_time * 1e-9
                    s = streams[key]
                    if key == "events":
                        events.append(json.loads(msg.data))
                        continue
                    if key == "tf_static":
                        for tr in msg.transforms:
                            t = tr.transform
                            entry = {
                                "parent": tr.header.frame_id,
                                "translation": [t.translation.x, t.translation.y, t.translation.z],
                                "rotation_xyzw": [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                            }
                            tf[tr.child_frame_id] = entry
                            tf_hist.append((log_t, {tr.child_frame_id: entry}))
                        continue
                    if key.endswith("_info"):
                        cam_info[key.replace("_info", "")] = {"K": list(msg.k), "width": msg.width, "height": msg.height}
                        continue
                    s.stamp.append(_t(msg.header.stamp))
                    s.log.append(log_t)
                    if key == "joint_states":
                        pos = dict(zip(msg.name, msg.position, strict=False))
                        vel = dict(zip(msg.name, msg.velocity, strict=False))
                        s.data.append(
                            (
                                np.array([pos[j] for j in ARM_JOINTS] + [pos.get("finger_joint1", 0.0) + pos.get("finger_joint2", 0.0)]),
                                np.array([vel.get(j, 0.0) for j in ARM_JOINTS]),
                            )
                        )
                    elif key == "wrist_ft":
                        w = msg.wrench
                        s.data.append(np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]))
                    elif key == "joint_command":
                        pos = dict(zip(msg.name, msg.position, strict=False))
                        s.data.append(np.array([pos[j] for j in ARM_JOINTS] + [pos.get("gripper", 1.0)]))
                    elif key in ("cam_front", "cam_wrist"):
                        s.data.append(bytes(msg.data))
                    elif key == "cam_front_depth":
                        s.data.append((msg.height, msg.width, bytes(msg.data)))
            except Exception as exc:
                errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
    for st in streams.values():  # file order is chunk order; restore receive-time order per stream
        if st.log and np.any(np.diff(st.log) < 0):
            order = np.argsort(np.asarray(st.log), kind="stable")
            st.stamp = [st.stamp[i] for i in order]
            st.log = [st.log[i] for i in order]
            st.data = [st.data[i] for i in order]
    events.sort(key=lambda e: e.get("t", 0.0))
    tf_hist.sort(key=lambda x: x[0])
    return Session(Path(bag_dir), streams, events, cam_info, tf, tf_hist, errors)
