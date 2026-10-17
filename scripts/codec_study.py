"""Video encoding for robot datasets: size, fidelity, speed and frame integrity per codec.

    python scripts/codec_study.py --staging data/staging/teleop --episodes 12 --out reports/codec

For each codec setting, every episode's camera stream is encoded as one video (as the
LeRobot release does), decoded again and compared frame by frame against the staged
frames: bytes per frame, PSNR, worst-frame PSNR, encode / decode throughput, and whether
the decoded stream has exactly the input frame count (a dropped or duplicated frame
would silently misalign images and actions).
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import av
import numpy as np

SETTINGS = [
    ("raw (uncompressed)", None, {}),
    ("JPEG quality 90 per frame (as in the bags)", "jpeg", {"quality": "90"}),
    ("FFV1 lossless", "ffv1", {}),
    ("H.264 CRF 23", "libx264", {"crf": "23", "preset": "medium"}),
    ("H.264 CRF 30", "libx264", {"crf": "30", "preset": "medium"}),
    ("H.264 CRF 23, GOP 2", "libx264", {"crf": "23", "preset": "medium", "g": "2"}),
    ("HEVC CRF 28", "libx265", {"crf": "28", "preset": "medium", "x265-params": "log-level=error"}),
    ("AV1 CRF 30, GOP 2 (release setting)", "libsvtav1", {"crf": "30"}),
    ("AV1 CRF 40, GOP 2", "libsvtav1", {"crf": "40"}),
]


def encode(frames: np.ndarray, codec: str, opts: dict[str, str], fps: int = 15) -> bytes | list[bytes]:
    if codec == "jpeg":  # what the ROS driver publishes: one JPEG per frame (PIL, like the driver)
        from PIL import Image

        out = []
        for f in frames:
            b = io.BytesIO()
            Image.fromarray(f).save(b, format="JPEG", quality=int(opts["quality"]))
            out.append(b.getvalue())
        return out
    buf = io.BytesIO()
    fmt = "matroska" if codec == "ffv1" else "mp4"
    with av.open(buf, "w", format=fmt) as c:
        s = c.add_stream(codec, rate=fps, options=opts)
        s.width, s.height = frames.shape[2], frames.shape[1]
        s.pix_fmt = "bgr0" if codec == "ffv1" else "yuv420p"
        if codec == "libsvtav1":
            s.options = {**opts, "g": "2"}  # short GOP: cheap random access for training
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def decode(data: bytes | list[bytes]) -> np.ndarray:
    if isinstance(data, list):
        from PIL import Image

        return np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in data])
    with av.open(io.BytesIO(data)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).reshape(len(a), -1).mean(1)
    return np.where(mse == 0, 99.0, 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12)))


def main() -> None:
    from rls.data.schema import read_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, default=Path("data/staging/teleop"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--camera", default="front", choices=["front", "wrist"])
    ap.add_argument("--out", type=Path, default=Path("reports/codec"))
    a = ap.parse_args()
    paths = sorted(a.staging.rglob("*.h5"))[:: max(1, len(sorted(a.staging.rglob("*.h5"))) // a.episodes)][: a.episodes]
    clips = [getattr(read_episode(p), a.camera) for p in paths]
    n = sum(len(c) for c in clips)
    rows = []
    for label, codec, opts in SETTINGS:
        size, enc_s, dec_s, ps, exact = 0, 0.0, 0.0, [], True
        for frames in clips:
            frames = np.ascontiguousarray(frames)
            if codec is None:
                size += frames.nbytes
                ps.append(np.full(len(frames), 99.0))
                continue
            t0 = time.perf_counter()
            data = encode(frames, codec, opts)
            enc_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            back = decode(data)
            dec_s += time.perf_counter() - t0
            size += sum(map(len, data)) if isinstance(data, list) else len(data)
            exact &= len(back) == len(frames)
            m = min(len(back), len(frames))
            ps.append(psnr(frames[:m], back[:m]))
        p = np.concatenate(ps)
        rows.append(
            {
                "setting": label,
                "bytes_per_frame": round(size / n),
                "ratio_vs_raw": round(clips[0][0].nbytes * n / size, 1),
                "psnr_mean_db": round(float(p.mean()), 2) if codec else None,
                "psnr_p1_db": round(float(np.percentile(p, 1)), 2) if codec else None,
                "encode_fps": round(n / enc_s) if enc_s else None,
                "decode_fps": round(n / dec_s) if dec_s else None,
                "frame_count_exact": bool(exact),
            }
        )
        print(rows[-1], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    h, w = clips[0].shape[1:3]
    (a.out / "codec_study.json").write_text(
        json.dumps({"camera": a.camera, "resolution": [w, h], "episodes": len(clips), "frames": n, "results": rows}, indent=1)
    )
    md = [
        f"# Video encoding study ({a.camera} camera, {w}x{h}, {len(clips)} episodes, {n} frames)",
        "",
        "| setting | bytes / frame | vs raw | PSNR mean (dB) | PSNR 1st pct (dB) | encode fps | decode fps | frame count exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['setting']} | {r['bytes_per_frame']:,} | {r['ratio_vs_raw']}x | {r['psnr_mean_db'] or 'lossless'} | {r['psnr_p1_db'] or '-'} | {r['encode_fps'] or '-'} | {r['decode_fps'] or '-'} | {'yes' if r['frame_count_exact'] else '**no**'} |"
        )
    md += [
        "",
        "Reference frames are the staged frames, which were decoded from the bag's per-frame JPEGs; re-encoding them as JPEG at the",
        "same quality is close to idempotent, so the JPEG row's PSNR is optimistic. GOP = keyframe interval: GOP 2 (LeRobot's",
        "default) costs size but lets a training loader seek to any frame by decoding at most one extra frame; long-GOP H.264",
        "is much smaller but random access has to decode from the previous keyframe.",
    ]
    (a.out / "CODEC_STUDY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
