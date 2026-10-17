# Benchmark: `ft_lora_10`

Dataset `v-1994eee9ac`, 10 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| unseen_color | 20 | 10% | 3-30% | 20% | 0% | no_grasp |
| in_distribution | 20 | 35% | 18-57% | 70% | 0% | no_grasp |

## Failure analysis

### no_grasp: 24 episodes

- Where: unseen_color (11), in_distribution (13)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 4 episodes

- Where: unseen_color (4)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 2 episodes

- Where: unseen_color (2)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 1 episodes

- Where: unseen_color (1)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

