# Benchmark: `act_main`

Dataset `v-8d8fc45d7b`, 409 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| in_distribution | 30 | 43% | 27-61% | 53% | 33% | no_grasp |
| position_ood | 30 | 50% | 33-67% | 73% | 27% | no_grasp |
| two_distractors | 30 | 70% | 52-83% | 93% | 47% | no_grasp |
| lighting_shift | 30 | 37% | 22-54% | 60% | 13% | no_grasp |
| camera_shift | 30 | 53% | 36-70% | 80% | 27% | no_grasp |
| unseen_color | 30 | 17% | 7-34% | 27% | 7% | no_grasp |

## Failure analysis

### no_grasp: 52 episodes

- Where: in_distribution (10), position_ood (6), two_distractors (5), lighting_shift (8), camera_shift (8), unseen_color (15)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### dropped: 17 episodes

- Where: in_distribution (5), position_ood (1), two_distractors (1), lighting_shift (6), camera_shift (1), unseen_color (3)
- Likely cause: lost the object in transit: grasp quality or gripper timing. Inspect gripper-close timing relative to contact.
- Videos: `videos/<split>/dropped_<seed>.mp4`
- Owner / next step: _fill in_

### misplaced: 16 episodes

- Where: in_distribution (2), position_ood (3), two_distractors (3), lighting_shift (4), camera_shift (4)
- Likely cause: released near but not at the goal: final alignment. Typical for stacking; more precise demos or a higher-resolution goal view help.
- Videos: `videos/<split>/misplaced_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 7 episodes

- Where: position_ood (4), lighting_shift (1), camera_shift (1), unseen_color (1)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

### wrong_object: 7 episodes

- Where: position_ood (1), unseen_color (6)
- Likely cause: manipulated a distractor: language grounding. Check colour balance per instruction and distractor counts in training.
- Videos: `videos/<split>/wrong_object_<seed>.mp4`
- Owner / next step: _fill in_

