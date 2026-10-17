# Training notes: what it took to make closed-loop behaviour cloning work

The policy is a language-conditioned action-chunking transformer in the style of ACT: front and
wrist camera features, proprioception and instruction tokens go through a transformer encoder;
20 learned queries decode the next 20 actions. Everything trains on 2 CPU cores, so the budget is
a few thousand steps, not the 100k+ that published ACT runs use. This page records how the
first recipe failed and what fixed it; the numbers are in `reports/studies/STUDIES.md` (study 4).

## 1. The first recipe fit the data and failed the task

Joint-position actions, DrQ random-shift augmentation (4 px), 3500 steps: validation L1 kept
falling, and closed-loop success in distribution was 3 of 40 episodes on the first benchmark. Every failure looked the same in the
traces: the arm went to roughly the right part of the table, a few centimetres from the cube,
closed the gripper on nothing, and retried nearby.

Checks that ruled out the pipeline:

* staged frames from the ROS 2 recordings against a fresh render of the same scene: same
  channel order, mean pixel difference 1.5 (the injected sensor noise);
* predictions from a dataset frame and from a live frame of the same scene: identical;
* tokenizer, normaliser and state layout: identical between training and the runtime.

## 2. Why: the arm already knows where it is going

An open-loop check on held-out episodes, run on one of the tool-pose pilots (below), was the useful
one. Twelve ticks before the grasp the
predicted grasp position was within 7 mm of the demonstration; with the images replaced by images
from other episodes it was still within 31 mm; with unit Gaussian noise on the normalised joint state it was off by 60 mm.
The policy was mostly extrapolating the arm's own motion: in a demonstration the arm is already
heading to the cube, so the joint state predicts the next action. In closed loop the arm is
wherever the policy put it, and that shortcut has nothing to extrapolate from (the "copycat"
problem in behaviour cloning).

## 3. What fixed it

| change | why |
|---|---|
| **tool-pose actions** (`action_space: ee`) | predict `[x, y, z, cos 2ψ, sin 2ψ, gripper]` and convert with IK at run time; a grasp error is now a position error instead of the sum of seven coupled joint errors. Recovered from the recorded joint commands by forward kinematics (exact to 0.1 mm), IK takes 0.6 ms per tick. |
| **proprioception noise and dropout** | Gaussian noise (0.1 in normalised units) and dropping the state token for 30% of samples during training remove the shortcut without removing the information. Removing the state entirely was worse. |
| **auxiliary object localisation** | two extra decoder queries regress where the instructed cube and its goal (bin or support cube) are. Labels come from the scene layout recorded with every simulated episode. It gives the visual encoder a direct, dense signal for the one thing the task depends on; the target label stops at the first gripper close, after which the cube moves with the hand. |
| **binary gripper at run time** | demonstrations open and close fully; a regressed or ensembled 0.5 half-closes the fingers on the cube and never builds grip force. |
| **smaller image shift (2 px)** | with fixed cameras the image position of a cube is its world position, so a 4 px random shift (about 1 cm on the table) is label noise for localisation. |

Neither the auxiliary loss nor the state regularisation helped much alone; together they moved
in-distribution success from 0-10% to 25% at 1500 steps.

## 4. The remaining gap was grounding, and FiLM closed part of it

With images now mattering, the failures changed: the arm went confidently to *a* cube, often the
wrong one. `scripts/probe_policy.py` measures this directly: on the first frame of an episode,
before the arm has moved, does the auxiliary head put the instructed cube within 30 mm? Without
FiLM that happened in about 10% of validation episodes, even after 5000 steps: the policy found
the cube only once the wrist camera was over it. The instruction reached the images only through
transformer attention over finished CNN features, so "blue" could not change what the CNN looked
for.

**FiLM** (as in RT-1): the instruction tokens, in order, go through a small MLP that outputs a
per-channel scale and shift for every CNN stage of both cameras, zero-initialised so it starts as
a no-op. At 1500 steps first-frame grounding rose to 35% of validation episodes and closed-loop
success to 35%. `act_main` trains this recipe for 6000 steps.

## 5. Things that did not help (at this budget)

* spatial-softmax keypoints instead of one token per feature-map cell: similar validation loss and
  success, 20% faster per step. Kept as an option (`keypoints: 32`).
* no image augmentation at all.

## 6. Caveats

* The auxiliary labels are free in simulation. On real recordings they would come from the
  synthetic part of a mixed dataset, from a calibrated perception stack or from annotation.
* Pilot runs used an earlier implementation of the auxiliary head in which the two localisation
  queries shared the action queries' decoder pass; the final implementation runs them through the
  decoder separately (same parameters, same loss).
* Budgets are small and intervals wide; these are directions, not tuned optima.
