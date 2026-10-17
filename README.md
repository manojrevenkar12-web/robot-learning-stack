# robot-learning-stack

From ROS 2 recordings to a deployed robot policy: multimodal capture with sensor-health checks,
synchronisation and curation, LeRobot v3 datasets that mix ROS 2 recordings with synthetic data,
policy training and closed-loop benchmarks, and an ONNX policy runtime and skill executor back in
the ROS 2 graph.

The robot is a simulated Franka Panda (MuJoCo, [Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
model) with a front RGB-D camera, a wrist camera, joint encoders and a wrist force-torque sensor,
doing language-conditioned pick-and-place and stacking ("stack the blue cube on the yellow cube").
It is driven through a real ROS 2 Jazzy graph. Each camera runs on its own drifting clock, and the
simulator injects dropped frames, burst drops, outages, corrupt frames and camera calibration
errors while recording exactly what it injected, so every data check in this repository is scored
against ground truth.

> **All numbers below were measured in this repository on a 2-core CPU machine, in simulation.**
> Nothing here has run on a physical robot. See [Limits](#limits).

<!-- RESULTS -->

## What is in here

| stage | code | what it does |
|---|---|---|
| capture | `rls/ros/driver.py`, `teleop.py`, `session.py` | ROS 2 nodes for the simulated robot and its sensors (120 Hz joints and F/T, 15 Hz RGB-D and wrist camera, `camera_info`, `/tf_static`), a scripted teleoperator, rosbag2 recording (MCAP, sim time, lockstep) |
| live health | `cpp/rls_sync_monitor` (C++, rclcpp) | per-stream drop rate, gaps, message age and camera stamp skew on `/diagnostics` while recording |
| ingest | `rls/data/bag.py`, `sync.py`, `calib.py`, `ingest.py` | reads MCAP without ROS (tolerates truncated bags), estimates camera clock offset and drift, checks extrinsics, aligns every stream causally onto 15 Hz control ticks, flags repaired frames, writes one HDF5 per episode |
| synthetic | `rls/data/synthetic.py` | domain-randomised episodes from the same simulator in the same schema |
| curation | `rls/data/curate.py` | health / outcome / length filters, exact and near-duplicate detection, deterministic stratified splits, content-hashed dataset version |
| release | `rls/data/export_lerobot.py` | LeRobot v3.0 via LeRobot's own writer (AV1 video), lineage back to the source bag |
| training | `rls/learn/` | model-agnostic trainer (policy registry), ACT-style chunking transformer, single-step BC, diffusion head, LoRA / head / full fine-tuning, MLflow tracking, resumable checkpoints, ONNX export with parity check |
| evaluation | `rls/learn/benchmark.py`, `scripts/run_studies.py` | closed-loop benchmark on 6 generalisation splits, Wilson intervals, failure taxonomy, videos, studies |
| deployment | `rls/ros/policy_node.py`, `executor_node.py`, `evaluate.py` | ONNX policy runtime in the ROS 2 graph (freshness and cross-camera sync checks, latency diagnostics); skill executor with a rule or LLM planner and RGB-D postcondition checks |
| RL | `rls/rl/` | PPO (Stable-Baselines3) with domain randomisation, evaluated on held-out dynamics, ONNX actor |
| tooling | `rls/cli.py`, `rls/tools/viewer.py`, `scripts/release_check.py` | one CLI for every stage, a web viewer for episodes, runs and benchmarks, a release gate used by the nightly workflow |

Docs: [architecture](docs/architecture.md) · [dataset schema](docs/dataset.md) · [training notes](docs/training.md) · [runbook](docs/runbook.md)

## Quick start

```bash
pip install -e ".[dev]"                                  # Python 3.12; CPU torch is enough
source /opt/ros/jazzy/setup.bash                         # only for recording and ROS deployment
cmake -S cpp/rls_sync_monitor -B build/mon && cmake --build build/mon && \
  cmake --install build/mon --prefix cpp/rls_sync_monitor/install   # optional C++ monitor

python scripts/build_dataset.py --root data              # record over ROS 2 -> ingest -> synth -> curate -> release
rls train configs/train/act_main.yaml
rls export runs/act_main/best --out runs/act_main/policy.onnx
rls bench runs/act_main/best --onnx runs/act_main/policy.onnx --splits in_distribution,position_ood --episodes 30 --out reports/bench/act_main
rls deploy runs/act_main/best --onnx runs/act_main/policy.onnx --episodes 10 --out reports/deploy
rls viewer                                               # http://127.0.0.1:8765
```

Releases need LeRobot in its own environment (`RLS_LEROBOT_PYTHON`, see the [runbook](docs/runbook.md)).
A `Dockerfile` on `ros:jazzy-ros-base` packages all of it.

## Limits

* **Simulation only.** No physical robot, no real cameras, no real teleoperation device. The fault
  model is built from how real capture rigs fail, but it is a model; the success rates are
  simulation success rates and say nothing about a real cell.
* **Scripted operator.** "Teleop" demonstrations come from a scripted operator with per-episode style
  noise and occasional sloppy grasps, not from a human.
* **Small compute.** Policies train for a few thousand steps on 2 CPU cores; published ACT and
  diffusion-policy results use one to two orders of magnitude more. Benchmarks use 20-40 episodes per
  cell, so confidence intervals are wide and stated everywhere.
* **No pretrained VLA.** The training framework is model-agnostic and fine-tunes with LoRA, but the
  models here are trained from scratch; no OpenVLA / π0 / SmolVLA checkpoint was fine-tuned.
* The Dockerfile and GitHub Actions workflows were written for this repository but have not been run
  on GitHub's infrastructure from this machine; the same steps were run locally.

## License

Apache-2.0. The Franka Panda model under `rls/sim/assets/franka_panda` is from MuJoCo Menagerie
(Apache-2.0, see its LICENSE).
