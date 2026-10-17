# Benchmark: `bc_single_step`

Dataset `v-8d8fc45d7b`, 409 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| in_distribution | 20 | 0% | 0-16% | 0% | 0% | no_grasp |
| position_ood | 20 | 0% | 0-16% | 0% | 0% | no_grasp |

## Failure analysis

### no_grasp: 38 episodes

- Where: in_distribution (19), position_ood (19)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

### timeout: 2 episodes

- Where: in_distribution (1), position_ood (1)
- Likely cause: still holding the object at the time limit: hesitation or oscillation. Check temporal ensembling and chunk length.
- Videos: `videos/<split>/timeout_<seed>.mp4`
- Owner / next step: _fill in_

