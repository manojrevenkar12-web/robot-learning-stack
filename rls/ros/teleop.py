"""Teleoperation node: the scripted operator, driving the arm over ROS 2.

Stands in for a human with a spacemouse or leader arm. It sees the scene
through ``/sim/privileged`` (operator eyes, never recorded as robot data) and
publishes ``/joint_command`` at the control rate, stamped with the tick it
answers so the driver can run in lockstep.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from rls.ros.driver import ARM_JOINTS, to_time
from rls.sim.env import EpisodeSpec
from rls.sim.expert import ScriptedExpert
from rls.sim.scene import build_model


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("teleop")
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(JointState, "/joint_command", reliable)
        self.status_pub = self.create_publisher(String, "/operator/status", 10)
        self.create_subscription(String, "/episode/events", self._on_event, latched)
        self.create_subscription(String, "/sim/privileged", self._on_state, reliable)
        self.expert: ScriptedExpert | None = None
        self.episode_index = -1
        self.pending: list[dict] = []
        self.sent_done = False

    def _on_event(self, msg: String) -> None:
        ev = json.loads(msg.data)
        if ev["event"] == "start":
            spec = EpisodeSpec.from_json(json.dumps(ev["spec"]))
            self.expert = ScriptedExpert(spec, build_model(spec.scene), seed=spec.seed)
            self.episode_index = ev["episode_index"]
            self.sent_done = False
            self.status_pub.publish(String(data=json.dumps({"done": False})))
            backlog, self.pending = self.pending, []
            for s in backlog:
                self._act(s)

    def _on_state(self, msg: String) -> None:
        self._act(json.loads(msg.data))

    def _act(self, state: dict) -> None:
        if self.expert is None or state["episode_index"] != self.episode_index:
            self.pending.append(state)
            return
        action = self.expert.act(state)
        cmd = JointState()
        cmd.header.stamp = to_time(state["stamp"])
        cmd.name = [*ARM_JOINTS, "gripper"]
        cmd.position = [float(v) for v in action]
        self.cmd_pub.publish(cmd)
        if self.expert.done and not self.sent_done:
            self.status_pub.publish(String(data=json.dumps({"done": True, "log": self.expert.log})))
            self.sent_done = True


def main() -> None:
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
