# Benchmark: `diffusion_chunk`

Dataset `v-8d8fc45d7b`, 409 training episodes, conditions {'obs_latency_steps': 0, 'image_noise_std': 2.0, 'action_delay_steps': 0}.

| split | episodes | success | 95% CI | pick_place | stack | top failure |
|---|---|---|---|---|---|---|
| in_distribution | 20 | 0% | 0-16% | 0% | 0% | no_grasp |
| position_ood | 20 | 0% | 0-16% | 0% | 0% | no_grasp |

## Failure analysis

### no_grasp: 40 episodes

- Where: in_distribution (20), position_ood (20)
- Likely cause: never lifted the target: approach/grasp pose. Check wrist-camera coverage of the grasp phase and grasp-height variance in the data.
- Videos: `videos/<split>/no_grasp_<seed>.mp4`
- Owner / next step: _fill in_

