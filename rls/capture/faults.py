"""Sensor timing and fault model for the simulated robot cell.

Real capture rigs do not deliver perfect data: cameras run on their own clocks
(offset + drift against the robot controller), frames are dropped in bursts,
USB hiccups produce black or truncated frames, a driver can stall for a second,
and the published extrinsics can be stale after someone bumps a camera. The
driver applies this model to every stream, and the injected ground truth is
written next to the bag so the health checks can be scored against it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class StreamFaults:
    rate_hz: float
    phase_s: float = 0.0  # capture phase relative to the robot clock
    latency_s: float = 0.0  # mean capture-to-publish delay
    latency_jitter_s: float = 0.0
    stamp_offset_s: float = 0.0  # sensor clock minus robot clock
    stamp_drift_ppm: float = 0.0
    stamp_jitter_s: float = 0.0
    drop_prob: float = 0.0
    burst_prob: float = 0.0  # probability that a drop starts a burst
    burst_len: tuple[int, int] = (3, 10)
    corrupt_prob: float = 0.0  # images: black / frozen / truncated JPEG
    outage_prob: float = 0.0  # per episode: stream silent for outage_s
    outage_s: tuple[float, float] = (0.5, 1.5)


@dataclass
class CalibrationFault:
    """Published front-camera extrinsic error (the true camera pose is unchanged)."""

    prob: float = 0.0
    translation_m: float = 0.03
    rotation_deg: float = 2.0


@dataclass
class FaultProfile:
    name: str
    streams: dict[str, StreamFaults]
    calibration: CalibrationFault = field(default_factory=CalibrationFault)
    jpeg_quality: int = 90
    image_noise_std: float = 2.0  # sensor noise (8-bit levels); real frames are never bit-identical

    @staticmethod
    def load(path: Path) -> FaultProfile:
        raw = yaml.safe_load(Path(path).read_text())
        def fields(v: dict[str, Any]) -> dict[str, Any]:
            return {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}

        streams = {k: StreamFaults(**fields(v)) for k, v in raw["streams"].items()}
        cal = CalibrationFault(**raw.get("calibration", {}))
        return FaultProfile(
            name=raw["name"], streams=streams, calibration=cal, jpeg_quality=raw.get("jpeg_quality", 90), image_noise_std=raw.get("image_noise_std", 2.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamSchedule:
    """Decides, per stream, when a sample is captured, how it is stamped, and its fate."""

    def __init__(self, name: str, f: StreamFaults, rng: np.random.Generator, episode_start: float, episode_len_s: float) -> None:
        self.name = name
        self.f = f
        self.rng = rng
        self.period = 1.0 / f.rate_hz
        self.next_t = episode_start + f.phase_s
        self.burst_left = 0
        self.outage: tuple[float, float] | None = None
        if f.outage_prob and rng.random() < f.outage_prob:
            dur = rng.uniform(*f.outage_s)
            start = episode_start + rng.uniform(0.2, max(0.3, episode_len_s - dur - 0.2))
            self.outage = (start, start + dur)
        self.events = {"captured": 0, "dropped": 0, "corrupted": 0, "outage_dropped": 0}

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.next_t

    def advance(self) -> float:
        cap = self.next_t
        self.next_t += self.period
        return cap

    def stamp(self, capture_t: float) -> float:
        """Header stamp as the sensor's own clock reports the capture time."""
        f = self.f
        jitter = float(self.rng.normal(0, f.stamp_jitter_s)) if f.stamp_jitter_s else 0.0
        return capture_t + f.stamp_offset_s + f.stamp_drift_ppm * 1e-6 * capture_t + jitter

    def publish_time(self, capture_t: float) -> float:
        lat = self.f.latency_s + (abs(self.rng.normal(0, self.f.latency_jitter_s)) if self.f.latency_jitter_s else 0.0)
        return capture_t + lat

    def fate(self, capture_t: float) -> str:
        """'ok' | 'drop' | 'outage' | 'corrupt'."""
        self.events["captured"] += 1
        if self.outage and self.outage[0] <= capture_t < self.outage[1]:
            self.events["outage_dropped"] += 1
            return "outage"
        if self.burst_left > 0:
            self.burst_left -= 1
            self.events["dropped"] += 1
            return "drop"
        if self.f.drop_prob and self.rng.random() < self.f.drop_prob:
            if self.f.burst_prob and self.rng.random() < self.f.burst_prob:
                self.burst_left = int(self.rng.integers(*self.f.burst_len)) - 1
            self.events["dropped"] += 1
            return "drop"
        if self.f.corrupt_prob and self.rng.random() < self.f.corrupt_prob:
            self.events["corrupted"] += 1
            return "corrupt"
        return "ok"


def corrupt_image(rgb: np.ndarray, prev: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray | bytes, str]:
    """Returns a corrupted payload and its kind. Bytes = a truncated JPEG."""
    kind = str(rng.choice(["black", "frozen", "truncated", "saturated"]))
    if kind == "black":
        return np.zeros_like(rgb), kind
    if kind == "saturated":
        return np.full_like(rgb, 255), kind
    if kind == "frozen" and prev is not None:
        return prev.copy(), kind
    return b"truncated", "truncated"


def add_sensor_noise(rgb: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return rgb
    noisy = rgb.astype(np.int16) + rng.normal(0, std, rgb.shape).round().astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)
