"""Named scene distributions: data sources and benchmark generalization splits."""

from __future__ import annotations

from dataclasses import replace

from rls.sim.env import Variant

TRAIN_COLORS = ("red", "green", "blue", "yellow")

# The "lab": fixed appearance, as a real cell would be. ROS 2 teleop recordings use it.
LAB = Variant(name="lab", light_jitter=0.0, table_jitter=0.0, cam_pos_jitter=0.0, bin_xy_jitter=0.01, colors=TRAIN_COLORS)

# Synthetic data: same layout distribution, randomized appearance and camera pose.
SYNTH_DR = Variant(name="synthetic_dr", light_jitter=0.35, table_jitter=0.12, cam_pos_jitter=0.015, colors=TRAIN_COLORS)

# Benchmark splits. Each changes one factor relative to LAB.
SPLITS: dict[str, Variant] = {
    "in_distribution": replace(LAB, name="in_distribution"),
    "position_ood": replace(LAB, name="position_ood", x_range=(0.36, 0.66), y_range=(-0.16, 0.24), yaw_range=(-0.78, 0.78)),
    "two_distractors": replace(LAB, name="two_distractors", n_distractors=(2, 2)),
    "lighting_shift": replace(LAB, name="lighting_shift", light_jitter=0.5, table_jitter=0.12),
    "camera_shift": replace(LAB, name="camera_shift", cam_pos_jitter=0.03),
    "unseen_color": replace(LAB, name="unseen_color", colors=("purple", "orange", "red", "green", "blue", "yellow"), target_colors=("purple", "orange")),
}


def with_tasks(v: Variant, tasks: tuple[str, ...]) -> Variant:
    return replace(v, tasks=tasks)
