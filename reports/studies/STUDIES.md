# Studies

Closed-loop success in simulation on held-out benchmark seeds (the same seeds for every policy), with 95% Wilson
intervals. With 20-30 episodes per cell an interval spans roughly 30-40 points, so only differences outside
overlapping intervals are evidence. Everything trains on 2 CPU cores; step budgets are small by robot-learning
standards, which is the point of studies 1 and 4: what works on a small budget.

## 1. Main policy (? steps, ? training episodes)

| split | success (95% CI) | top failure |
|---|---|---|
| in_distribution | – | – |
| position_ood | – | – |
| two_distractors | – | – |
| lighting_shift | – | – |
| camera_shift | – | – |
| unseen_color | – | – |

Failure analysis and videos: `reports/bench/act_main/REPORT.md`.

## 2. Data source (same recipe, 1500 steps)

| training data | episodes | in_distribution | lighting_shift | camera_shift |
|---|---|---|---|---|
| ROS 2 teleop only | – | – | – | – |
| synthetic (domain-randomised) only | – | – | – | – |
| both | 409 | – | – | – |

## 3. Policy class (same encoder, recipe and data, 1500 steps)

| policy | in_distribution | position_ood |
|---|---|---|
| single-step behaviour cloning (chunk 1) | – | – |
| action chunking (chunk 20, temporal ensembling) | – | – |
| diffusion head (chunk 20, 10-step DDIM, temporal ensembling) | – | – |

## 4. Ablation: making closed-loop behaviour cloning work (in_distribution)

Each row changes the recipe of the row it names; all use all training data.

| run | change | val loss | first-frame grounding | success (95% CI) | top failure |
|---|---|---|---|---|---|
| `act_joint` | first recipe: joint-position actions, DrQ shift 4 px (3500 steps) | 0.145 | – | – | – |
| `pilot_ee` | tool-pose (EE) actions | 0.159 | – | – | – |
| `pilot_noaug` | EE, no image shift augmentation | 0.150 | – | – | – |
| `pilot_kp` | EE, spatial-softmax keypoint readout, shift 2 px | 0.152 | – | – | – |
| `pilot_nostate` | EE, no proprioception input | 0.249 | – | – | – |
| `pilot_sd` | EE, proprioception noise 0.1 and dropout 0.5 | 0.199 | – | – | – |
| `pilot_aux` | EE, auxiliary object localisation | 0.303 | – | – | – |
| `pilot_aux_sd` | EE, auxiliary localisation + proprioception noise 0.1 and dropout 0.3 | 0.356 | – | – | – |
| `pilot_film` | previous row + FiLM: the instruction modulates every CNN stage (final recipe) | 0.321 | – | – | – |

First-frame grounding: share of validation episodes where, before the arm has moved, the auxiliary head puts the
instructed cube within 30 mm (`scripts/probe_policy.py`; only for runs with the auxiliary head). Validation loss does
not predict closed-loop success here; grounding from the first frame does. See `docs/training.md`.

## 5. Transfer under deployment conditions (act_main, in_distribution)

| condition | success (95% CI) |
|---|---|
| as trained | – |
| observations 2 ticks old (133 ms) | – |
| observations 4 ticks old (267 ms) | – |
| actions applied 2 ticks late | – |

## 6. Fine-tuning to unseen object colours

Starting point: `act_main`, which never saw purple or orange cubes, scores – on unseen_color.

| demos | method | trainable params | unseen_color | in_distribution (forgetting check) |
|---|---|---|---|---|
| 10 | lora | – | – | – |
| 10 | full | – | – | – |
| 25 | lora | – | – | – |
| 25 | full | – | – | – |
| 25 | scratch | – | – | – |
