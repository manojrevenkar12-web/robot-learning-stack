# Benchmark: `ft_scratch_25`

Dataset `v-1994eee9ac`, 25 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| unseen_color | 20 | 10% | 3-30% | 20% | 0% | no_grasp |
| in_distribution | 20 | 0% | 0-16% | 0% | 0% | no_grasp |

## Failure analysis

### no_grasp: 29 episodes

- Where: unseen_color (15), in_distribution (14)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 3 episodes

- Where: unseen_color (1), in_distribution (2)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 3 episodes

- Where: unseen_color (1), in_distribution (2)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 2 episodes

- Where: in_distribution (2)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

### misplaced: 1 episodes

- Where: unseen_color (1)
- Likely cause: released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.
- Videos: `videos/<split>/misplaced_<seed>.mp4`
- Owner / next step: _fill in_

