# Runbook

Operational notes: how to run each stage, what its report means, what to do when a check fails.

## Environment

* Python 3.12, `pip install -e ".[dev]"` (CPU torch is enough).
* ROS 2 Jazzy for recording and ROS deployment (`source /opt/ros/jazzy/setup.bash`); everything
  after recording runs without ROS.
* LeRobot in its own venv for releases: `python -m venv /opt/lerobot-venv && /opt/lerobot-venv/bin/pip install "lerobot[dataset]==0.6.1" pyyaml h5py`, then `export RLS_LEROBOT_PYTHON=/opt/lerobot-venv/bin/python`.
* Headless rendering: `MUJOCO_GL=egl` (GPU or Mesa EGL) or `MUJOCO_GL=osmesa`.
* Or use the container: `docker build -t rls . && docker run --rm -it -v $PWD/data:/opt/rls/data rls bash`.

## Record a session

```bash
rls record --out data/raw/s042 --episodes 12 --profile configs/faults/realistic.yaml --seed-start 42000
```

Starts the simulator driver, the operator and `ros2 bag record` (MCAP, sim time). The session is
done when `data/raw/s042/bag/metadata.yaml` exists and `logs/recorder.log` ends with
`Recording stopped`. A bag without `metadata.yaml` was not finalized (recorder killed, disk
full): it is still usable, ingest reads up to the last complete chunk and marks the
interrupted episode `episode_complete: fail`.

If a recorder started from a background shell never stops: a non-interactive shell's `cmd &`
makes children ignore SIGINT. `record_session` resets signal handlers in its children for
this reason; custom launchers need the same.

## Ingest and read the health report

```bash
rls inspect data/raw/s042/bag                      # episodes, message counts, read errors
rls ingest data/raw/s042/bag --out data/staging/teleop/s042
```

| check | warn / fail | first thing to look at |
|---|---|---|
| `cam_front`, `cam_wrist` | warn: drop rate > 1 % or any corrupt frame; fail: drop rate > 5 %, corrupt rate > 2 % or a gap > 0.5 s (outage) | USB bandwidth, camera fps/exposure, cable; driver logs at the gap time |
| `cam_*_clock` | warn: drift fit unreliable, constant offset applied | camera clock reset mid-session; PTP/NTP status |
| `front_extrinsics` | warn: projected hand position disagrees with measured depth | camera bumped or calibration stale: recalibrate before recording more |
| `joint_states`, `wrist_ft` | gaps or non-finite values | driver / controller load |
| `command_stream`, `commands` | operator commands missing | teleop device, network |
| `frame_fill` | fail: more than 6 consecutive filled camera frames | follows from drops or outages above |
| `episode_complete` | fail: no end event | recording stopped mid-episode |
| `bag_readable` | warn: bag damaged, episodes before the damage recovered | recorder crash, disk full |

A session whose wrist clock fit reports `reliable: false` uses a constant offset; check the
residual before training on it.

## Curate and release

```bash
rls curate data/staging --config configs/curation/default.yaml --out data/curated
rls release data/curated/manifest.parquet --out data/release/rls_panda
```

`curation_report.json`: scanned, kept, `dropped_by_reason`, kept per source and per split, and
the dataset `version`. A release is immutable by convention: a new curation config or new
data gives a new version and a new release directory.

`python scripts/build_dataset.py --root data` runs record → ingest → synth → curate → release
end to end; each stage is skipped when its output exists.

## Train

```bash
rls train configs/train/act_main.yaml              # runs/act_main/{best,last,state.pt,metrics.jsonl}
rls train configs/train/act_main.yaml --resume     # after an interruption
rls viewer                                         # http://127.0.0.1:8765 : data, runs, benchmarks
```

MLflow file store under `runs/mlruns` (`mlflow ui --backend-store-uri runs/mlruns`).
`runs/registry.jsonl` lists every finished run with its dataset version and config hash.

Fine-tune from a checkpoint: set `init_from` and `finetune: lora | head | full` in the config, or
`--init-from runs/act_main/best --finetune lora`. LoRA adapters are merged into the base weights
at export, so the deployed ONNX graph is the same size as the base model.

## Evaluate

```bash
rls export runs/act_main/best --out runs/act_main/policy.onnx     # fails if ONNX != torch
rls bench runs/act_main/best --onnx runs/act_main/policy.onnx --splits in_distribution,position_ood --episodes 40 --out reports/bench/act_main
rls deploy runs/act_main/best --onnx runs/act_main/policy.onnx --episodes 10 --out reports/deploy   # over ROS 2
rls agent  runs/act_main/best --onnx runs/act_main/policy.onnx --episodes 10 --out reports/agent    # multi-step instructions
```

`REPORT.md` lists success per split with 95% Wilson intervals and a failure section per mode
(`no_grasp`, `wrong_object`, `dropped`, `misplaced`, `timeout`) with likely cause and videos.
With 40 episodes per split an interval is roughly ±15 points: compare runs on the same seeds
and treat overlapping intervals as "no evidence of a difference".

## Release gate

```bash
python scripts/release_check.py --candidate runs/act_main/best --out reports/release/act_main
```

Fails on: data-health errors against injected truth, ONNX parity > 1e-4, p95 latency > 50 ms,
or a closed-loop regression that is both ≥ 10 points and significant (one-sided z-test,
p < 0.05) against `configs/release/baseline.json`. `--update-baseline` after review accepts the
candidate. The nightly workflow runs the `--quick` variant (small rebuilt dataset, short
training). Run the gate on an otherwise idle machine: with two training jobs sharing a 2-core box,
ONNX p95 latency rose from 34 ms to 69 ms and the gate (correctly) failed.

## Common failures

| symptom | cause | fix |
|---|---|---|
| `rosbag2` records nothing | `--use-sim-time` waits for `/clock`; sim time 0 counts as "not started" | driver starts sessions at t = 1 s and publishes `/clock` while waiting for peers |
| `ImportError: lerobot` during release | release runs in the LeRobot venv | set `RLS_LEROBOT_PYTHON` |
| pytest crashes on import of `launch_testing` | ROS pytest plugins | already disabled in `pyproject.toml` addopts |
| benchmark much slower than expected | software rendering | `MUJOCO_GL=egl`; shadows are off and collision meshes are rendered by design |
