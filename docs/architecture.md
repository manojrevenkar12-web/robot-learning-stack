# Architecture

```
             ┌──────────────────── ROS 2 Jazzy (sim time, lockstep) ────────────────────┐
             │                                                                          │
 MuJoCo ──►  │  sim_driver ──/joint_states 120 Hz, /wrist_ft 120 Hz──────────┐           │
 Panda       │      │      ──/cam_front/{color,depth} 15 Hz, /cam_wrist 15 Hz─┤           │
             │      │      ──/tf_static, camera_info, /episode/events──────────┤  rosbag2  │──► MCAP bag
             │      │                                                          │  (mcap)   │
             │      ◄──/joint_command── teleop (scripted operator)  or         │           │
             │                          policy_node (ONNX) or executor_node    ├──►sync_monitor (C++,
             │                                                                 │   rclcpp) ──/diagnostics
             │                                                                 ┘           │
             └──────────────────────────────────────────────────────────────────────────┘
                     fault model per stream: rate, phase, latency + jitter, clock offset + drift,
                     drops, burst drops, outages, corrupt frames, extrinsic miscalibration

 MCAP bag ──► ingest (no ROS needed) ──► staging HDF5 ──► curate ──► manifest ──► LeRobot v3.0 release
               health checks              ▲                 filter, dedup,         (LeRobot's own writer,
               clock estimation           │                 stratified splits,      AV1 video)
               causal alignment       synthetic             content-hashed version
               quality flags          (domain-randomized)
                                                                   │
                                                                   ▼
            train (ACT-style chunk transformer, LoRA/full/head fine-tuning, MLflow, bundles)
                                                                   │
                         ┌─────────────────────────────┬───────────┴──────────────┐
                         ▼                             ▼                          ▼
              sim benchmark (6 splits,        ONNX export (parity-     ROS 2 deployment: policy_node,
              Wilson CIs, failure taxonomy,   checked) ──────────────► executor_node (planner + skills
              videos, REPORT.md)                                       + RGB-D postcondition checks)
                                                                   │
                                          release_check.py: data health vs truth, export parity,
                                          latency, closed-loop success vs baseline (nightly CI)
```

## Packages

| package | responsibility |
|---|---|
| `rls.sim` | MuJoCo scene (Menagerie Franka Panda, patched at load time with a wrist camera and F/T site), episode sampling, IK, scripted expert, named variants and benchmark splits |
| `cpp/rls_sync_monitor` | C++ (rclcpp) node: live per-stream drop rate, longest gap, message age and front/wrist stamp skew while recording; `/diagnostics` at 1 Hz and a JSON summary per session |
| `rls.capture` | sensor fault model (`FaultProfile`, `StreamSchedule`), image corruption and noise |
| `rls.ros` | ROS 2 nodes: `driver` (simulator + sensors), `teleop`, `policy_node`, `executor_node`; `session` records a session with rosbag2; `evaluate` runs closed-loop evaluation over ROS |
| `rls.data` | bag reader (no ROS dependency, tolerant of truncated files), clock estimation, extrinsic check, ingest, staging schema, synthetic generation, curation, LeRobot export |
| `rls.learn` | LeRobot reader, chunk dataset, policies (registry, LoRA layers), training, runner (temporal ensembling), ONNX export, benchmark |
| `rls.perception` | RGB-D cube detection (HSV + depth back-projection), used for skill postconditions |
| `rls.agent` | instruction → skill plan: rule-based planner, optional LLM planner behind a validator |
| `rls.rl` | PPO reach task with dynamics randomization, ONNX actor export |
| `rls.tools` | web viewer for datasets, runs and benchmarks |

## Design decisions

**Bags are the source of truth.** Everything downstream of recording can be rebuilt from the
MCAP bags plus the ingest code version. Ingest runs without ROS (`mcap` + `mcap-ros2-support`), so
data engineers and CI do not need a ROS install to reprocess data.

**Sensor clocks are not trusted.** Header stamps come from each device's clock. Ingest estimates a
per-camera offset by correlating image motion energy with joint-velocity energy (high-pass
filtered), per episode, then fits offset + drift per session with a robust line fit. Frames are
re-timed onto the robot clock before alignment. When the fit is unreliable (few usable
episodes, high residual) it falls back to a constant offset and says so in the report.

**Online and offline checks.** The C++ monitor runs during recording and flags a bad session
while the operator can still re-record; ingest repeats the checks offline with the whole session
in view (clock fits need many episodes). Both are scored against the injected ground truth in
the tests. The monitor found a real bug: header stamps for times just below a whole second were
rounded to the start of that second, one second early, once per second on the 120 Hz streams.

**Causal alignment.** Each 15 Hz control tick takes the latest sample captured *before* the tick
on the corrected clock, never a future one: a policy at run time cannot see the future either.
Gaps are filled by holding the previous frame and the frame is flagged; episodes over the fill
budget are dropped at curation, not silently trained on.

**Release format owned upstream.** The release is written with LeRobot's own `LeRobotDataset`
writer in a separate pinned environment, rather than a re-implementation of the format. Our
training reader is verified pixel-exact against `LeRobotDataset` in the tests.

**Every checkpoint is traceable.** A policy bundle carries its config hash, dataset version
(content hash of the curated manifest), git SHA and training-episode count; `runs/registry.jsonl`
lists every finished run.

**Simulation only.** There is no physical robot here. The fault model, the clock problem and the
deployment checks are the parts that carry over; the success rates do not.
