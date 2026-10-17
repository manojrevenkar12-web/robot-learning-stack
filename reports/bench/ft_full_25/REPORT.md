# Benchmark: `ft_full_25`

Dataset `v-1994eee9ac`, 25 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| unseen_color | 20 | 20% | 8-42% | 40% | 0% | no_grasp |
| in_distribution | 20 | 30% | 15-52% | 60% | 0% | no_grasp |

## Failure analysis

### no_grasp: 17 episodes

- Where: unseen_color (8), in_distribution (9)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 6 episodes

- Where: unseen_color (6)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 3 episodes

- Where: unseen_color (1), in_distribution (2)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### misplaced: 2 episodes

- Where: unseen_color (1), in_distribution (1)
- Likely cause: released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.
- Videos: `videos/<split>/misplaced_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 2 episodes

- Where: in_distribution (2)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

