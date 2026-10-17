# Benchmark: `ft_full_10`

Dataset `v-1994eee9ac`, 10 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| unseen_color | 20 | 5% | 1-24% | 0% | 10% | no_grasp |
| in_distribution | 20 | 50% | 30-70% | 90% | 10% | no_grasp |

## Failure analysis

### no_grasp: 19 episodes

- Where: unseen_color (13), in_distribution (6)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### misplaced: 3 episodes

- Where: unseen_color (1), in_distribution (2)
- Likely cause: released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.
- Videos: `videos/<split>/misplaced_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 3 episodes

- Where: unseen_color (1), in_distribution (2)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 2 episodes

- Where: unseen_color (2)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 2 episodes

- Where: unseen_color (2)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

