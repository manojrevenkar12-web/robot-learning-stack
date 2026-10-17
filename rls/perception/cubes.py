"""Classical RGB-D perception for the skill executor: find coloured cubes in 3D.

Colour segmentation in HSV on the front camera, depth lookup, and back-projection
through the published intrinsics and extrinsics. It is deliberately simple and
fast (about 1 ms per frame); its job is to check skill postconditions ("is the
red cube in the bin?"), not to drive the arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# hue ranges (degrees) and minimum saturation/value per named colour
HUES: dict[str, tuple[tuple[float, float], ...]] = {
    "red": ((345.0, 360.0), (0.0, 12.0)),
    "orange": ((18.0, 38.0),),
    "yellow": ((42.0, 65.0),),
    "green": ((95.0, 150.0),),
    "blue": ((205.0, 240.0),),
    "purple": ((265.0, 310.0),),
}
MIN_SAT, MIN_VAL, MIN_PIXELS = 0.45, 0.25, 4


@dataclass
class Detection:
    color: str
    pixel: tuple[float, float]
    n_pixels: int
    position: np.ndarray  # world frame, metres (visible surface point)


def rgb_to_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    return h, s, mx


def detect(
    rgb: np.ndarray, depth_mm: np.ndarray, k: list[float], t_world_cam: np.ndarray, r_world_cam: np.ndarray, colors: list[str] | None = None
) -> dict[str, Detection]:
    h, s, v = rgb_to_hsv(rgb)
    chroma = (s > MIN_SAT) & (v > MIN_VAL)
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    out: dict[str, Detection] = {}
    for color in colors or list(HUES):
        mask = np.zeros_like(chroma)
        for lo, hi in HUES[color]:
            mask |= (h >= lo) & (h < hi)
        mask &= chroma & (depth_mm > 0)
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        vs, us = np.nonzero(mask)
        z = float(np.median(depth_mm[vs, us])) / 1000.0
        u, vv = float(us.mean()), float(vs.mean())
        p_cam = np.array([(u - cx) / fx * z, (vv - cy) / fy * z, z])
        out[color] = Detection(color, (u, vv), n, r_world_cam @ p_cam + t_world_cam)
    return out


def in_bin(p: np.ndarray, bin_xy: tuple[float, float], half: float = 0.08, max_z: float = 0.09) -> bool:
    return bool(abs(p[0] - bin_xy[0]) < half and abs(p[1] - bin_xy[1]) < half and p[2] < max_z)


def stacked(top: np.ndarray, bottom: np.ndarray, xy_tol: float = 0.03, min_dz: float = 0.025) -> bool:
    return bool(np.linalg.norm(top[:2] - bottom[:2]) < xy_tol and top[2] - bottom[2] > min_dz)
