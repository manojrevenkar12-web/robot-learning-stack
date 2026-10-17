# Benchmark: `act_main`

Dataset `v-8d8fc45d7b`, 409 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| in_distribution | 30 | 43% | 27-61% | 53% | 33% | no_grasp |
| position_ood | 30 | 50% | 33-67% | 73% | 27% | no_grasp |

## Failure analysis

### no_grasp: 16 episodes

- Where: in_distribution (10), position_ood (6)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 6 episodes

- Where: in_distribution (5), position_ood (1)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### misplaced: 5 episodes

- Where: in_distribution (2), position_ood (3)
- Likely cause: released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.
- Videos: `videos/<split>/misplaced_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 4 episodes

- Where: position_ood (4)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 1 episodes

- Where: position_ood (1)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

