"""Record a teleoperation session: driver + operator + rosbag2 (MCAP), one bag per session.

python -m rls.ros.session --out data/raw/session_000 --episodes 20 --profile configs/faults/realistic.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RECORDED_TOPICS = [
    "/joint_states",
    "/wrist_ft",
    "/cam_front/color/image_raw/compressed",
    "/cam_front/depth/image_raw",
    "/cam_front/color/camera_info",
    "/cam_wrist/color/image_raw/compressed",
    "/cam_wrist/color/camera_info",
    "/tf_static",
    "/joint_command",
    "/episode/events",
]


def _ros_args(params: dict[str, object]) -> list[str]:
    out = ["--ros-args"]
    for k, v in params.items():
        if isinstance(v, str) and v[:1] in "{[":
            v = "'" + v.replace("'", "''") + "'"  # JSON payload: pass as a YAML string, not a YAML mapping
        out += ["-p", f"{k}:={v}"]
    return out


SYNC_MONITOR = Path(
    os.environ.get(
        "RLS_SYNC_MONITOR", Path(__file__).resolve().parents[2] / "cpp" / "rls_sync_monitor" / "install" / "lib" / "rls_sync_monitor" / "sync_monitor"
    )
)


def _default_signals() -> None:
    """Children must handle SIGINT even when this process was started in the background
    (a non-interactive shell's `cmd &` ignores SIGINT, and the ignore is inherited). Without
    this, rosbag2 never sees the stop request, is killed, and leaves an unfinalized bag."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def record_session(
    out: Path,
    episodes: int,
    profile: Path,
    variant: str = "lab",
    tasks: str = "pick_place,stack",
    seed_start: int = 0,
    fault_seed: int = 0,
    episode_specs: Path | None = None,
    operator: str = "teleop",
    timeout_s: float = 3600.0,
    record: bool = True,
    policy_params: dict[str, object] | None = None,
    driver_params: dict[str, object] | None = None,
    monitor: bool = True,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    py = sys.executable
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []
    bag = out / "bag"
    if monitor and SYNC_MONITOR.exists():  # C++ live health monitor (cpp/rls_sync_monitor), optional
        procs.append(
            subprocess.Popen(
                [str(SYNC_MONITOR), *_ros_args({"use_sim_time": True, "summary_path": str(out / "sync_monitor.json")})],
                stdout=open(logs / "sync_monitor.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                preexec_fn=_default_signals,
            )
        )
    if record:
        rec = subprocess.Popen(
            ["ros2", "bag", "record", "-s", "mcap", "--use-sim-time", "-o", str(bag), "--topics", *RECORDED_TOPICS],
            stdout=open(logs / "recorder.log", "w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_default_signals,
        )
        procs.append(rec)
    if operator == "agent":
        pp = {
            "use_sim_time": True,
            "summary_path": str(out / "policy_runtime.json"),
            "command_topic": "/policy/joint_command",
            "start_enabled": False,
            **(policy_params or {}),
        }
        procs.append(
            subprocess.Popen(
                [py, "-m", "rls.ros.policy_node", *_ros_args(pp)],
                stdout=open(logs / "policy.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                preexec_fn=_default_signals,
            )
        )
        ep = {"use_sim_time": True, "report_path": str(out / "agent_report.json")}
        procs.append(
            subprocess.Popen(
                [py, "-m", "rls.ros.executor_node", *_ros_args(ep)],
                stdout=open(logs / "executor.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                preexec_fn=_default_signals,
            )
        )
    elif operator == "policy":
        pp = {"use_sim_time": True, "summary_path": str(out / "policy_runtime.json"), **(policy_params or {})}
        op = subprocess.Popen(
            [py, "-m", "rls.ros.policy_node", *_ros_args(pp)],
            stdout=open(logs / "policy.log", "w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_default_signals,
        )
        procs.append(op)
    elif operator == "teleop":
        op = subprocess.Popen(
            [py, "-m", "rls.ros.teleop", *_ros_args({"use_sim_time": True})],
            stdout=open(logs / "teleop.log", "w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_default_signals,
        )
        procs.append(op)
    params = {
        "session_dir": str(out),
        "n_episodes": episodes,
        "variant": variant,
        "tasks": tasks,
        "seed_start": seed_start,
        "fault_profile": str(profile),
        "fault_seed": fault_seed,
    }
    if episode_specs:
        params["episode_specs"] = str(episode_specs)
    monitored = monitor and SYNC_MONITOR.exists()
    if record:
        # wait until the recorder itself has subscribed: count every other subscriber we started
        base = 1 + int(monitored)
        pol, exe = int(operator in ("policy", "agent")), int(operator == "agent")
        counts = {
            "/cam_front/color/image_raw/compressed": base + pol + exe,
            "/cam_wrist/color/image_raw/compressed": base + pol,
            "/cam_front/depth/image_raw": base + exe,
            "/wrist_ft": base,
            "/sim/privileged": 1,
        }
        params["wait_for_topics"] = ",".join(f"{t}:{n}" for t, n in counts.items())
    else:
        params["wait_for_topics"] = "/sim/privileged"
    params.update(driver_params or {})
    t0 = time.time()
    drv = subprocess.run(
        [py, "-m", "rls.ros.driver", *_ros_args(params)], stdout=open(logs / "driver.log", "w"), stderr=subprocess.STDOUT, env=env, timeout=timeout_s
    )
    time.sleep(1.5)  # let the recorder drain its queues
    for p in procs:
        os.killpg(p.pid, signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    if drv.returncode != 0:
        raise RuntimeError(f"driver failed (exit {drv.returncode}); see {logs / 'driver.log'}")
    meta = {
        "session": out.name,
        "episodes": episodes,
        "variant": variant,
        "tasks": tasks,
        "seed_start": seed_start,
        "fault_profile": str(profile),
        "fault_seed": fault_seed,
        "operator": operator,
        "robot": "franka_panda (MuJoCo Menagerie), simulated",
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "session.json").write_text(json.dumps(meta, indent=1))
    return bag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--profile", type=Path, default=Path("configs/faults/realistic.yaml"))
    ap.add_argument("--variant", default="lab")
    ap.add_argument("--tasks", default="pick_place,stack")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--fault-seed", type=int, default=0)
    a = ap.parse_args()
    bag = record_session(a.out, a.episodes, a.profile, a.variant, a.tasks, a.seed_start, a.fault_seed)
    print(bag)


if __name__ == "__main__":
    main()
