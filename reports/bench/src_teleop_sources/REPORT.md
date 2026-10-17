# Benchmark: `src_teleop`

Dataset `v-8d8fc45d7b`, 162 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| in_distribution | 20 | 15% | 5-36% | 30% | 0% | no_grasp |
| lighting_shift | 20 | 10% | 3-30% | 20% | 0% | no_grasp |
| camera_shift | 20 | 20% | 8-42% | 40% | 0% | no_grasp |

## Failure analysis

### no_grasp: 41 episodes

- Where: in_distribution (14), lighting_shift (13), camera_shift (14)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 4 episodes

- Where: in_distribution (2), lighting_shift (2)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 3 episodes

- Where: in_distribution (1), lighting_shift (2)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 3 episodes

- Where: lighting_shift (1), camera_shift (2)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

