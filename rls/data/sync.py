"""Temporal calibration: estimate each camera's clock offset and drift against the robot clock.

Camera header stamps come from the camera's own clock. We recover the mapping
to robot time from the data itself: image motion energy between consecutive
frames must line up with how fast the arm was moving at that moment (joint
speed from ``/joint_states``, robot clock). The lag that maximises their
correlation is the clock offset for that episode; a robust line through the
per-episode offsets over a session gives offset and drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OffsetEstimate:
    offset_s: float  # camera_stamp - robot_time
    peak_corr: float
    sharpness: float  # peak minus median correlation across the search window
    n_intervals: int

    @property
    def confident(self) -> bool:
        return self.peak_corr > 0.5 and self.sharpness > 0.25 and self.n_intervals >= 20


@dataclass
class ClockModel:
    offset_s: float  # at t_ref
    drift_ppm: float
    t_ref: float
    residual_ms: float
    n_episodes: int
    reliable: bool = True

    def offset_at(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.offset_s + self.drift_ppm * 1e-6 * (np.asarray(t) - self.t_ref)

    def to_robot_time(self, stamps: np.ndarray) -> np.ndarray:
        # stamp = t + offset(t)  ->  t = stamp - offset(stamp) (offset varies slowly; one fixed-point step is exact to ~1e-9 s)
        t = stamps - self.offset_at(stamps)
        return stamps - self.offset_at(t)


def motion_energy(gray: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames, (N-1,)."""
    g = gray.astype(np.float32)
    return np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))


def _interval_mean(t: np.ndarray, v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of piecewise-linear v(t) over [a, b], vectorised."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    ia = np.interp(a, t, cum)
    ib = np.interp(b, t, cum)
    return (ib - ia) / np.maximum(b - a, 1e-6)


def motion_signal(qvel: np.ndarray, gripper_width: np.ndarray, t: np.ndarray, gripper_gain: float = 5.0) -> np.ndarray:
    """Robot-side 'how much should the image change' signal: arm joint speed plus
    finger motion (the fingers are in the wrist camera's view)."""
    return np.linalg.norm(qvel, axis=1) + gripper_gain * np.abs(np.gradient(gripper_width, t))


def estimate_offset(
    cam_stamps: np.ndarray,
    gray: np.ndarray,
    robot_t: np.ndarray,
    robot_motion: np.ndarray,
    valid: np.ndarray | None = None,
    search_s: float = 0.4,
    step_s: float = 0.002,
    period_s: float = 1 / 15,
    highpass: bool = True,
) -> OffsetEstimate:
    """Lag between camera stamps and robot time that best aligns image motion with arm motion.

    With ``highpass`` the correlation is computed on first differences, which
    sharpens the peak considerably (onsets and stops of motion carry the timing).
    """
    e = motion_energy(gray)
    a, b = cam_stamps[:-1], cam_stamps[1:]
    ok = (b - a) < 1.6 * period_s
    if valid is not None:
        ok &= valid[:-1] & valid[1:]
    lags = np.arange(-search_s, search_s + step_s / 2, step_s)
    corr = np.full(lags.shape, -1.0)
    n_used = int(ok.sum())
    if n_used < 10 or np.std(e[ok]) == 0:
        return OffsetEstimate(0.0, 0.0, 0.0, n_used)
    for i, d in enumerate(lags):
        lo, hi = a - d, b - d
        inside = ok & (lo >= robot_t[0]) & (hi <= robot_t[-1])
        v = _interval_mean(robot_t, robot_motion, lo, hi)
        if highpass:
            m = inside[1:] & inside[:-1]
            x, y = np.diff(e)[m], np.diff(v)[m]
        else:
            x, y = e[inside], v[inside]
        if len(x) >= 10 and np.std(y) > 0 and np.std(x) > 0:
            corr[i] = float(np.corrcoef(x, y)[0, 1])
    k = int(np.argmax(corr))
    best = lags[k]
    if 0 < k < len(lags) - 1:  # parabolic refinement
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            best += 0.5 * (y0 - y2) / den * step_s
    valid_corr = corr[corr > -1]
    return OffsetEstimate(float(best), float(corr[k]), float(corr[k] - np.median(valid_corr)), n_used)


def fit_clock(
    t_mid: np.ndarray, offsets: np.ndarray, weights: np.ndarray | None = None, max_residual_ms: float = 15.0, max_drift_ppm: float = 5000.0
) -> ClockModel:
    """Robust line offset(t) = a + b (t - t_ref) through per-episode estimates."""
    t_mid, offsets = np.asarray(t_mid, float), np.asarray(offsets, float)
    w = np.ones_like(offsets) if weights is None else np.asarray(weights, float)
    t_ref = float(np.median(t_mid))
    keep = np.ones_like(offsets, dtype=bool)
    a, b = float(np.median(offsets)), 0.0
    for _ in range(3):
        if keep.sum() >= 2 and np.ptp(t_mid[keep]) > 1.0:
            x = t_mid[keep] - t_ref
            b, a = np.polyfit(x, offsets[keep], 1, w=w[keep])
        else:
            a = float(np.average(offsets[keep], weights=w[keep]))
            b = 0.0
        res = offsets - (a + b * (t_mid - t_ref))
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-4
        keep = np.abs(res) < max(5 * 1.4826 * mad, 0.004)
    res = offsets[keep] - (a + b * (t_mid[keep] - t_ref))
    rms_ms = float(np.sqrt(np.mean(res**2)) * 1e3) if len(res) else float("nan")
    if not np.isfinite(rms_ms) or rms_ms > max_residual_ms or abs(b) * 1e6 > max_drift_ppm:
        # estimates disagree too much for a drift fit: fall back to a constant offset
        med = float(np.median(offsets))
        return ClockModel(med, 0.0, t_ref, rms_ms, int(keep.sum()), reliable=False)
    return ClockModel(float(a), float(b * 1e6), t_ref, rms_ms, int(keep.sum()))
