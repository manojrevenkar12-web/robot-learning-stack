"""Bag -> staging episodes: health checks, clock correction, calibration check, alignment.

For each recorded session (one MCAP bag, many episodes):

1. decode camera frames per episode and classify corrupt ones (decode failure,
   black, saturated, frozen);
2. estimate each camera's clock offset per episode from motion correlation and
   fit offset + drift over the session (:mod:`rls.data.sync`);
3. run the health checks and the extrinsic check;
4. resample every modality onto the control ticks (15 Hz) with causal frame
   selection, filling short gaps and flagging every filled frame;
5. write one staging HDF5 per episode with its health report and provenance.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rls.data import schema as S
from rls.data.bag import Session, read_session
from rls.data.calib import HandKinematics, check_extrinsics, quat_xyzw_to_mat
from rls.data.sync import ClockModel, estimate_offset, fit_clock, motion_signal

PIPELINE_VERSION = "ingest-1"
CAMERAS = ("cam_front", "cam_wrist")


@dataclass
class HealthThresholds:
    drop_warn: float = 0.01
    drop_fail: float = 0.05
    corrupt_fail: float = 0.02
    camera_gap_fail_s: float = 0.5
    joint_gap_warn_s: float = 0.05
    joint_gap_fail_s: float = 0.5
    ft_gap_warn_s: float = 0.1
    clock_offset_note_s: float = 0.02
    max_fill_frames: int = 6  # longest run of filled camera frames tolerated


@dataclass
class Check:
    status: str  # ok | warn | fail
    value: Any
    detail: str = ""


@dataclass
class EpisodeReport:
    episode_id: str
    status: str
    checks: dict[str, Check] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"health": self.status, "checks": {k: asdict(v) for k, v in self.checks.items()}}


def _worst(statuses: list[str]) -> str:
    return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"


@dataclass
class DecodedCamera:
    stamps: np.ndarray
    frames: list[np.ndarray | None]
    corrupt: dict[str, int]
    valid: np.ndarray


def decode_camera(stamps: np.ndarray, payloads: list[bytes], shape: tuple[int, int]) -> DecodedCamera:
    frames: list[np.ndarray | None] = []
    corrupt = {"decode_error": 0, "black": 0, "saturated": 0, "frozen": 0}
    valid = np.ones(len(payloads), dtype=bool)
    prev: np.ndarray | None = None
    for i, b in enumerate(payloads):
        try:
            img = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            if img.shape[:2] != shape:
                raise ValueError("unexpected size")
        except Exception:
            corrupt["decode_error"] += 1
            valid[i] = False
            frames.append(None)
            continue
        m = float(img.mean())
        kind = "black" if m < 8 else "saturated" if m > 247 else "frozen" if prev is not None and np.array_equal(img, prev) else None
        if kind:
            corrupt[kind] += 1
            valid[i] = False
        frames.append(img)
        prev = img
    return DecodedCamera(stamps, frames, corrupt, valid)


def _gaps(stamps: np.ndarray, t0: float, t1: float, period: float) -> tuple[float, float, int]:
    """(drop rate, max gap s, expected count) over [t0, t1] from header stamps."""
    expected = max(1, int(round((t1 - t0) / period)))
    if len(stamps) < 2:
        return 1.0, t1 - t0, expected
    edges = np.concatenate([[t0], np.sort(stamps), [t1]])
    gaps = np.diff(edges)
    missing = np.sum(np.maximum(np.round(gaps / period) - 1, 0))
    return float(missing / expected), float(gaps.max()), expected


class SessionIngest:
    def __init__(self, bag_dir: Path, thresholds: HealthThresholds | None = None) -> None:
        self.session: Session = read_session(bag_dir)
        self.th = thresholds or HealthThresholds()
        js = self.session.streams["joint_states"]
        self.jt, _ = js.arrays()
        self.jq = np.array([d[0] for d in js.data])
        self.jv = np.array([d[1] for d in js.data])
        self.motion = motion_signal(self.jv, self.jq[:, 7], self.jt)
        self.kin = HandKinematics()
        self.clocks: dict[str, ClockModel] = {}
        self.offsets: dict[str, list[dict[str, float]]] = {c: [] for c in CAMERAS}
        self._decoded: dict[tuple[str, int], DecodedCamera] = {}

    # -------------------------------------------------------------- decoding
    def _window(self, key: str, t0: float, t1: float, pad: float = 0.6) -> np.ndarray:
        st, _ = self.session.streams[key].arrays()
        return np.where((st >= t0 - pad) & (st <= t1 + pad))[0]

    def decoded(self, cam: str, ep: dict[str, Any]) -> DecodedCamera:
        key = (cam, ep["episode_index"])
        if key not in self._decoded:
            idx = self._window(cam, ep["t_start"], ep["t_end"])
            info = self.session.camera_info.get(cam, {"height": 120, "width": 160})
            st, _ = self.session.streams[cam].arrays()
            data = self.session.streams[cam].data
            self._decoded[key] = decode_camera(st[idx], [data[i] for i in idx], (int(info["height"]), int(info["width"])))
        return self._decoded[key]

    # ----------------------------------------------------------- clock model
    def fit_clocks(self) -> dict[str, ClockModel]:
        for ep in self.session.episodes():
            for cam in CAMERAS:
                d = self.decoded(cam, ep)
                if len(d.frames) < 20:
                    continue
                ref = next((x for x in d.frames if x is not None), None)
                if ref is None:
                    continue
                gray = np.stack([(f if f is not None else np.zeros_like(ref)).mean(axis=2)[::2, ::2] for f in d.frames])
                est = estimate_offset(d.stamps, gray, self.jt, self.motion, d.valid)
                self.offsets[cam].append(
                    {
                        "episode_index": ep["episode_index"],
                        "t_mid": 0.5 * (ep["t_start"] + ep["t_end"]),
                        "offset_s": est.offset_s,
                        "corr": est.peak_corr,
                        "sharpness": est.sharpness,
                        "confident": est.confident,
                    }
                )
        for cam in CAMERAS:
            rows = [r for r in self.offsets[cam] if r["confident"]] or self.offsets[cam]
            if rows:
                self.clocks[cam] = fit_clock(
                    np.array([r["t_mid"] for r in rows]), np.array([r["offset_s"] for r in rows]), np.array([max(r["corr"], 0.05) for r in rows])
                )
        return self.clocks

    # ------------------------------------------------------------- episodes
    def _tf_at(self, t: float) -> dict[str, Any] | None:
        best = None
        for log_t, entry in self.session.tf_history:
            if "cam_front_optical_frame" in entry and log_t <= t + 1e-6:
                best = entry["cam_front_optical_frame"]
        return best or self.session.tf.get("cam_front_optical_frame")

    def build_episode(self, ep: dict[str, Any], session_meta: dict[str, Any]) -> tuple[S.Episode | None, EpisodeReport]:
        th = self.th
        t0, t1 = ep["t_start"], ep["t_end"]
        eid = f"rec-{session_meta.get('session', 'session')}-{ep['episode_index']:04d}"
        rep = EpisodeReport(eid, "ok")
        period = 1.0 / S.FPS
        if not ep.get("complete", True):
            rep.checks["episode_complete"] = Check("fail", False, "no end event: recording stopped mid-episode")
        if self.session.read_errors:
            rep.checks["bag_readable"] = Check("warn", self.session.read_errors[:3], "bag damaged; episodes before the damage were recovered")

        # --- control ticks from the operator's commands
        ct, _ = self.session.streams["joint_command"].arrays()
        cdata = self.session.streams["joint_command"].data
        cm = (ct >= t0 - 1e-6) & (ct <= t1 + 1e-6)
        cmd_t = ct[cm]
        cmd = np.array([cdata[i] for i in np.where(cm)[0]])
        if len(cmd_t) < 10:
            rep.checks["commands"] = Check("fail", len(cmd_t), "fewer than 10 operator commands")
            rep.status = "fail"
            return None, rep
        ticks = np.arange(cmd_t[0], cmd_t[-1] + period / 2, period)
        cmd_drop, cmd_gap, _ = _gaps(cmd_t, ticks[0], ticks[-1], period)
        rep.checks["command_stream"] = Check("warn" if cmd_drop > 0 else "ok", round(cmd_drop, 4), f"max gap {cmd_gap:.3f}s")

        # --- joint states (robot clock): gaps, sanity
        jm = (self.jt >= t0 - 0.2) & (self.jt <= t1 + 0.2)
        jt, jq = self.jt[jm], self.jq[jm]
        jdrop, jgap, _ = _gaps(jt[(jt >= t0) & (jt <= t1)], t0, t1, 1 / 120)
        finite = bool(np.isfinite(jq).all())
        jstat = "fail" if jgap > th.joint_gap_fail_s or not finite else "warn" if jgap > th.joint_gap_warn_s else "ok"
        rep.checks["joint_states"] = Check(jstat, {"drop_rate": round(jdrop, 4), "max_gap_s": round(jgap, 4), "finite": finite})

        # --- wrist F/T
        ft_t, _ = self.session.streams["wrist_ft"].arrays()
        fm = (ft_t >= t0 - 0.2) & (ft_t <= t1 + 0.2)
        ft_t_e = ft_t[fm]
        ft = np.array([self.session.streams["wrist_ft"].data[i] for i in np.where(fm)[0]]) if fm.any() else np.zeros((0, 6))
        fdrop, fgap, _ = _gaps(ft_t_e[(ft_t_e >= t0) & (ft_t_e <= t1)], t0, t1, 1 / 120)
        rep.checks["wrist_ft"] = Check(
            "warn" if fgap > th.ft_gap_warn_s else "ok",
            {"drop_rate": round(fdrop, 4), "max_gap_s": round(fgap, 4)},
            "missing samples are flagged per frame" if fgap > th.ft_gap_warn_s else "",
        )

        # --- cameras: drops, outages, corruption, clock correction
        cams: dict[str, tuple[np.ndarray, list[np.ndarray | None], np.ndarray]] = {}
        for cam in CAMERAS:
            d = self.decoded(cam, ep)
            clock = self.clocks.get(cam)
            robot_t = clock.to_robot_time(d.stamps) if clock else d.stamps
            inside = (robot_t >= t0) & (robot_t <= t1)
            drop, gap, expected = _gaps(robot_t[inside], t0, t1, period)
            n_corrupt = int(sum(d.corrupt.values()))
            corrupt_rate = n_corrupt / max(expected, 1)
            st = (
                "fail"
                if gap > th.camera_gap_fail_s or drop > th.drop_fail or corrupt_rate > th.corrupt_fail
                else "warn"
                if drop > th.drop_warn or n_corrupt
                else "ok"
            )
            rep.checks[cam] = Check(st, {"drop_rate": round(drop, 4), "max_gap_s": round(gap, 4), "corrupt": d.corrupt})
            if clock:
                off = float(clock.offset_at(0.5 * (t0 + t1)))
                rep.checks[f"{cam}_clock"] = Check(
                    "ok" if clock.reliable else "warn",
                    {
                        "offset_ms": round(off * 1e3, 1),
                        "drift_ppm": round(clock.drift_ppm, 1),
                        "fit_residual_ms": round(clock.residual_ms, 2),
                        "reliable": clock.reliable,
                    },
                    ("corrected" if abs(off) > th.clock_offset_note_s else "") + ("" if clock.reliable else "; drift fit unreliable, constant offset applied"),
                )
            cams[cam] = (robot_t, d.frames, d.valid)

        # --- extrinsics (front RGB-D)
        tf = self._tf_at(t0)
        info = self.session.camera_info.get("cam_front")
        dt_, _ = self.session.streams["cam_front_depth"].arrays()
        if tf and info and len(dt_):
            dm = np.where((dt_ >= t0) & (dt_ <= t1))[0]
            raw = self.session.streams["cam_front_depth"].data
            depths = [np.frombuffer(raw[i][2], np.uint16).reshape(raw[i][0], raw[i][1]) for i in dm]
            qs = np.array([np.interp(dt_[i], self.jt, self.jq[:, j]) for i in dm for j in range(7)]).reshape(len(dm), 7)
            pts = np.array([self.kin.hand_point(q) for q in qs])
            chk = check_extrinsics(pts, depths, info["K"], np.array(tf["translation"]), quat_xyzw_to_mat(tf["rotation_xyzw"]))
            rep.checks["front_extrinsics"] = Check(
                "warn" if chk.suspect else "ok",
                {"median_residual_m": round(chk.median_residual_m, 4), "frac_off_hand": round(chk.frac_off_hand, 3), "n": chk.n_frames},
                "published extrinsics disagree with depth; quarantine for 3D use" if chk.suspect else "",
            )

        # --- alignment onto ticks
        n = len(ticks)
        flags = np.zeros(n, np.uint8)
        action = np.zeros((n, 8), np.float32)
        ci = np.searchsorted(cmd_t, ticks + 1e-6, side="right") - 1
        exact = np.isclose(cmd_t[np.clip(ci, 0, None)], ticks, atol=1e-4)
        action[:] = cmd[np.clip(ci, 0, None)]
        flags[~exact] |= S.FLAG_ACTION_HELD
        state = np.zeros((n, 8), np.float32)
        for j in range(8):
            state[:, j] = np.interp(ticks, jt, jq[:, j])
        state[:, 7] = np.clip(state[:, 7] / 0.08, 0, 1)
        jgap_at = np.diff(np.searchsorted(jt, np.stack([ticks - 2 / 120, ticks + 2 / 120])), axis=0)[0] == 0
        flags[jgap_at] |= S.FLAG_STATE_INTERP_GAP
        wrench = np.zeros((n, 6), np.float32)
        if len(ft_t_e) >= 2:
            for j in range(6):
                wrench[:, j] = np.interp(ticks, ft_t_e, ft[:, j])
            miss = np.diff(np.searchsorted(ft_t_e, np.stack([ticks - 0.05, ticks + 1e-6])), axis=0)[0] == 0
            flags[miss] |= S.FLAG_WRENCH_MISSING
        else:
            flags[:] |= S.FLAG_WRENCH_MISSING

        images: dict[str, np.ndarray] = {}
        worst_run = 0
        for cam, bit in (("cam_front", S.FLAG_FRONT_FILLED), ("cam_wrist", S.FLAG_WRIST_FILLED)):
            robot_t, frames, valid = cams[cam]
            order = np.argsort(robot_t)
            robot_t, valid = robot_t[order], valid[order]
            frames = [frames[i] for i in order]
            good = np.where(valid)[0]
            out = []
            run = 0
            last: np.ndarray | None = None
            for k, t in enumerate(ticks):
                # causal: newest valid frame captured at or before the tick; fresh if
                # younger than 1.5 periods (cameras free-run, so phase and jitter vary)
                c = good[robot_t[good] <= t + 1e-6]
                if len(c) and t - robot_t[c[-1]] <= period * 1.5:
                    last = frames[c[-1]]
                    run = 0
                elif k == 0 and len(good) and robot_t[good[0]] - t <= period:
                    last = frames[good[0]]  # first tick: the camera's first frame lands just after it
                else:
                    run += 1
                    worst_run = max(worst_run, run)
                    flags[k] |= bit
                    if last is None and len(good):
                        last = frames[good[0]]
                        flags[k] |= S.FLAG_CORRUPT_REPLACED
                out.append(last if last is not None else np.zeros((120, 160, 3), np.uint8))
            images[cam] = np.stack(out)
        if worst_run > th.max_fill_frames:
            rep.checks["frame_fill"] = Check("fail", worst_run, f"{worst_run} consecutive filled frames")
        else:
            rep.checks["frame_fill"] = Check("ok" if worst_run == 0 else "warn", worst_run)

        # depth: same selection as the front camera frames (shares stamps)
        dt_, _ = self.session.streams["cam_front_depth"].arrays()
        draw = self.session.streams["cam_front_depth"].data
        depth = np.zeros(images["cam_front"].shape[:3], np.uint16)
        if len(dt_):
            clock = self.clocks.get("cam_front")
            drt = clock.to_robot_time(dt_) if clock else dt_
            for k, t in enumerate(ticks):
                i = int(np.searchsorted(drt, t + 1e-6) - 1)
                if i >= 0:
                    h, w, b = draw[i]
                    depth[k] = np.frombuffer(b, np.uint16).reshape(h, w)

        rep.status = _worst([c.status for c in rep.checks.values()])
        outcome = ep.get("outcome", {})
        epi = S.Episode(
            episode_id=eid,
            source="teleop_ros",
            fps=S.FPS,
            task=ep["spec"]["task"],
            instruction=ep["spec"]["instruction"],
            timestamp=ticks - ticks[0],
            state=state,
            wrench=wrench,
            action=action,
            front=images["cam_front"],
            wrist=images["cam_wrist"],
            depth=depth,
            frame_flags=flags,
            episode_spec=ep["spec"],
            outcome=outcome,
            quality=rep.to_dict(),
            provenance={
                "pipeline": PIPELINE_VERSION,
                "bag": str(self.session.path),
                "t_start": t0,
                "t_end": t1,
                "session": session_meta,
                "clock_models": {k: asdict(v) for k, v in self.clocks.items()},
            },
        )
        return epi, rep


def ingest_session(bag_dir: Path, out_dir: Path, session_meta: dict[str, Any] | None = None, write_failed: bool = True) -> dict[str, Any]:
    t_start = time.time()
    meta = session_meta or {}
    sess_json = Path(bag_dir).parent / "session.json"
    if not meta and sess_json.exists():
        meta = json.loads(sess_json.read_text())
    ing = SessionIngest(Path(bag_dir))
    clocks = ing.fit_clocks()
    summary: dict[str, Any] = {
        "session": meta.get("session"),
        "episodes": [],
        "clocks": {k: asdict(v) for k, v in clocks.items()},
        "per_episode_offsets": ing.offsets,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in ing.session.episodes():
        epi, rep = ing.build_episode(ep, meta)
        row = {
            "episode_id": rep.episode_id,
            "status": rep.status,
            "checks": {k: asdict(v) for k, v in rep.checks.items()},
            "outcome": ep.get("outcome", {}).get("failure_mode"),
        }
        if epi is not None and (write_failed or rep.status != "fail"):
            S.write_episode(out_dir / f"{epi.episode_id}.h5", epi)
            row["path"] = str(out_dir / f"{epi.episode_id}.h5")
        summary["episodes"].append(row)
    summary["wall_time_s"] = round(time.time() - t_start, 1)
    (out_dir / f"ingest_report_{meta.get('session', 'session')}.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
