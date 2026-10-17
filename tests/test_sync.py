import numpy as np

from rls.data.sync import estimate_offset, fit_clock, motion_signal


def _scene(offset: float, rng: np.random.Generator):
    """Robot moves in bursts; a camera on its own clock sees image change when the arm moves."""
    t = np.arange(0, 12, 1 / 120)
    speed = np.zeros_like(t)
    for start in rng.uniform(0.5, 11, 8):
        speed += np.exp(-0.5 * ((t - start) / 0.25) ** 2)
    cam_true = np.arange(0.05, 11.9, 1 / 15)
    # image = cumulative motion "texture": energy between frames ~ motion in the interval
    pos = np.cumsum(speed) / 120
    frames = np.stack([np.sin(np.linspace(0, 40, 64)[None, :] + 30 * np.interp(tc, t, pos)) * np.ones((8, 1)) for tc in cam_true])
    return t, speed, cam_true + offset, frames


def test_offset_recovered_within_a_few_ms() -> None:
    rng = np.random.default_rng(0)
    for off in (-0.15, 0.0, 0.08):
        t, speed, stamps, frames = _scene(off, rng)
        est = estimate_offset(stamps, frames, t, speed)
        assert est.confident
        assert abs(est.offset_s - off) < 0.012, (off, est)


def test_clock_fit_recovers_drift_and_rejects_outliers() -> None:
    t = np.linspace(10, 600, 40)
    off = 0.12 + 600e-6 * (t - t.mean())
    noisy = off + np.random.default_rng(1).normal(0, 0.004, t.size)
    noisy[5] += 0.2  # one bad episode
    m = fit_clock(t, noisy)
    assert m.reliable
    assert abs(m.drift_ppm - 600) < 60
    assert abs(m.offset_at(t.mean()) - 0.12) < 0.004
    back = m.to_robot_time(t + off)
    assert np.allclose(back, t, atol=0.005)


def test_motion_signal_includes_fingers() -> None:
    t = np.linspace(0, 1, 121)
    s = motion_signal(np.zeros((121, 7)), np.linspace(0, 0.08, 121), t)
    assert np.all(s > 0)


def test_header_stamp_conversion_never_jumps_back() -> None:
    """Regression: t just below a whole second used to become (sec, 0 ns), one second early."""
    import pytest

    driver = pytest.importorskip("rls.ros.driver")
    ts = np.cumsum(np.full(5000, 1 / 120))  # float accumulation as in the simulator loop
    back = np.array([driver.from_time(driver.to_time(t)) for t in ts])
    assert np.all(np.diff(back) > 0)
    assert np.max(np.abs(back - ts)) < 1e-8
