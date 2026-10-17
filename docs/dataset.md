# Dataset

Three layers, each rebuildable from the one before it.

| layer | where | format | written by |
|---|---|---|---|
| raw | `data/raw/<session>/` | rosbag2 MCAP + `session.json` + `ground_truth.json` | `rls record` (ROS 2) |
| staging | `data/staging/{teleop,synthetic}/` | one HDF5 per episode, aligned at 15 Hz | `rls ingest`, `rls synth` |
| curated | `data/curated/` | `manifest.parquet`, `manifest_all.parquet`, `curation_report.json` | `rls curate` |
| release | `data/release/rls_panda/` | LeRobot v3.0 (parquet + AV1 video) | `rls release` (LeRobot's writer) |

## Raw: recorded topics

| topic | type | rate | clock |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | 120 Hz | robot |
| `/wrist_ft` | `geometry_msgs/WrenchStamped` | 120 Hz | robot |
| `/cam_front/color/image_raw/compressed` | `sensor_msgs/CompressedImage` (JPEG) | 15 Hz | front camera |
| `/cam_front/depth/image_raw` | `sensor_msgs/Image` (16UC1, mm) | 15 Hz | front camera |
| `/cam_wrist/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 15 Hz | wrist camera (own clock) |
| `/cam_*/color/camera_info` | `sensor_msgs/CameraInfo` | latched | |
| `/tf_static` | `tf2_msgs/TFMessage` | latched | |
| `/joint_command` | `sensor_msgs/JointState` | 15 Hz | robot |
| `/episode/events` | `std_msgs/String` (JSON: start / end, spec, outcome) | per episode | robot |

`ground_truth.json` holds what the fault model actually did (drops, corrupt frames, outages,
clock offsets, calibration errors) per episode. Ingest never reads it; tests and
`release_check.py` compare the ingest report against it.

## Staging: HDF5 episode

```
/timestamp                (T,)          float64  s since episode start, 15 Hz ticks
/observation/state        (T, 8)        float32  q1..q7 (rad), gripper opening [0, 1]
/observation/wrench       (T, 6)        float32  wrist force (N), torque (Nm)
/observation/images/front (T, 120, 160, 3) uint8
/observation/images/wrist (T, 96, 128, 3)  uint8
/observation/depth/front  (T, 120, 160)  uint16  mm, 0 = invalid
/action                   (T, 8)        float32  commanded q1..q7, gripper (1 = open)
/quality/frame_flags      (T,)          uint8    bit flags below
attrs: episode_id, source, fps, task, instruction, episode_spec, outcome, quality, provenance (JSON)
```

| flag | bit | meaning |
|---|---|---|
| `FRONT_FILLED` | 1 | no fresh front frame for this tick; previous frame held |
| `WRIST_FILLED` | 2 | same for the wrist camera |
| `STATE_INTERP_GAP` | 4 | joint state interpolated across a gap longer than 2 periods |
| `WRENCH_MISSING` | 8 | no F/T sample near the tick |
| `ACTION_HELD` | 16 | no new command; previous command held |
| `CORRUPT_REPLACED` | 32 | corrupt image detected, dropped and filled |

`quality.checks` holds one entry per check (`status` ok / warn / fail, measured `value`,
`detail`): per-camera drop rate, corrupt frames, outages, clock offset vs session fit,
extrinsic check, fill fraction, episode completeness. `quality.health` is the worst status.
`provenance` records the bag, the ingest version, the clock models used and (synthetic) the
generator version, seed and variant, so an episode can be regenerated exactly.

## Curation

Configured in `configs/curation/default.yaml`. In order:

1. **source / outcome / health filters**: unsuccessful episodes and `health == fail` are dropped.
2. **length and fill budget**: outside `[min_length, max_length]` or more than
   `max_filled_fraction` of frames filled → dropped.
3. **exact duplicates**: content hash over timestamps, state, actions and every 8th front frame.
4. **near duplicates**: same task, action trajectories within `near_dup_threshold`
   (mean absolute difference in rad after resampling both to 48 steps); the first
   episode is kept and the other records which one it duplicates.
5. **splits**: stratified by (task, source); within each group episodes are ranked by
   a hash of `split_seed` and the episode id, so the split is deterministic and does not
   depend on file order.

The dataset version is a content hash of the kept manifest (episode ids + content hashes +
splits + config), so a training run records exactly which data it saw.

## Release: LeRobot v3.0

Written with `LeRobotDataset.create / add_frame / save_episode / finalize` from
`lerobot==0.6.1`. Features: `observation.images.front`, `observation.images.wrist` (video,
AV1), `observation.state` (8), `observation.wrench` (6), `action` (8), task string.
`meta/info.json` carries contiguous `train` / `val` episode ranges. Episode-level fields
LeRobot has no column for (source, staging id, health, outcome) are in
`meta/rls_episodes.parquet`.

Load with LeRobot:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/rls_panda", root="data/release/rls_panda")
```

or with the training stack's reader (no LeRobot dependency; tested pixel-exact against the above):

```python
from rls.learn.lerobot_reader import load

data = load("data/release/rls_panda")
```
