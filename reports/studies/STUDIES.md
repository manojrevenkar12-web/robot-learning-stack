# Studies

Closed-loop success in simulation on held-out benchmark seeds (the same seeds for every policy), with 95% Wilson
intervals. With 20-30 episodes per cell an interval spans roughly 30-40 points, so only differences outside
overlapping intervals are evidence. Everything trains on 2 CPU cores; step budgets are small by robot-learning
standards, which is the point of studies 1 and 4: what works on a small budget.

## 1. Main policy (6000 steps, 409 training episodes)

| split | success (95% CI) | top failure |
|---|---|---|
| in_distribution | 43% (27–61) | no_grasp |
| position_ood | 50% (33–67) | no_grasp |
| two_distractors | 70% (52–83) | no_grasp |
| lighting_shift | 37% (22–54) | no_grasp |
| camera_shift | 53% (36–70) | no_grasp |
| unseen_color | 17% (7–34) | no_grasp |

Failure analysis and videos: `reports/bench/act_main/REPORT.md`.

## 2. Data source (same recipe, 1500 steps)

| training data | episodes | in_distribution | lighting_shift | camera_shift |
|---|---|---|---|---|
| ROS 2 teleop only | 162 | 15% (5–36) | 10% (3–30) | 20% (8–42) |
| synthetic (domain-randomised) only | 247 | 20% (8–42) | 15% (5–36) | 15% (5–36) |
| both | 409 | 35% (18–57) | 30% (15–52) | 40% (22–61) |

## 3. Policy class (same encoder, recipe and data, 1500 steps)

| policy | in_distribution | position_ood |
|---|---|---|
| single-step behaviour cloning (chunk 1) | 0% (0–16) | 0% (0–16) |
| action chunking (chunk 20, temporal ensembling) | 35% (18–57) | 15% (5–36) |
| diffusion head (chunk 20, 10-step DDIM, temporal ensembling) | 0% (0–16) | 0% (0–16) |

## 4. Ablation: making closed-loop behaviour cloning work (in_distribution)

Each row changes the recipe of the row it names; all use all training data.

| run | change | val loss | first-frame grounding | success (95% CI) | top failure |
|---|---|---|---|---|---|
| `act_joint` | first recipe: joint-position actions, DrQ shift 4 px (3500 steps) | 0.145 | – | 5% (1–24) | no_grasp |
| `pilot_ee` | tool-pose (EE) actions | 0.159 | – | 0% (0–16) | no_grasp |
| `pilot_noaug` | EE, no image shift augmentation | 0.150 | – | 5% (1–24) | no_grasp |
| `pilot_kp` | EE, spatial-softmax keypoint readout, shift 2 px | 0.152 | – | 5% (1–24) | no_grasp |
| `pilot_nostate` | EE, no proprioception input | 0.249 | – | 0% (0–16) | no_grasp |
| `pilot_sd` | EE, proprioception noise 0.1 and dropout 0.5 | 0.199 | – | 10% (3–30) | no_grasp |
| `pilot_aux` | EE, auxiliary object localisation | 0.303 | 8% | 0% (0–16) | no_grasp |
| `pilot_aux_sd` | EE, auxiliary localisation + proprioception noise 0.1 and dropout 0.3 | 0.356 | 2% | 25% (11–47) | no_grasp |
| `pilot_film` | previous row + FiLM: the instruction modulates every CNN stage (final recipe) | 0.321 | 35% | 35% (18–57) | no_grasp |
| `act_main` | final recipe, 6000 steps (30 episodes) | 0.217 | 75% | 43% (27–61) | no_grasp |

Validation loss includes the auxiliary term for runs that have it, so it is only comparable within the first and
second halves of the table.

First-frame grounding: share of validation episodes where, before the arm has moved, the auxiliary head puts the
instructed cube within 30 mm (`scripts/probe_policy.py`; only for runs with the auxiliary head). Validation loss does
not predict closed-loop success here; grounding from the first frame does. See `docs/training.md`.

## 5. Transfer under deployment conditions (act_main, in_distribution)

| condition | success (95% CI) |
|---|---|
| as trained | 43% (27–61) |
| observations 2 ticks old (133 ms) | 47% (30–64) |
| observations 4 ticks old (267 ms) | 50% (33–67) |
| actions applied 2 ticks late | 50% (33–67) |

## 6. Fine-tuning to unseen object colours

Starting point: `act_main`, which never saw purple or orange cubes, scores 17% (7–34) on unseen_color.

| demos | method | trainable params | unseen_color | in_distribution (forgetting check) |
|---|---|---|---|---|
| 10 | lora | 0.14 M | 10% (3–30) | 35% (18–57) |
| 10 | full | 2.76 M | 5% (1–24) | 50% (30–70) |
| 25 | lora | 0.14 M | 5% (1–24) | 15% (5–36) |
| 25 | full | 2.76 M | 20% (8–42) | 30% (15–52) |
| 25 | scratch | 2.76 M | 10% (3–30) | 0% (0–16) |
