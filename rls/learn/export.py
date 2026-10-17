"""Export a policy bundle to ONNX and verify parity with the PyTorch model.

    python -m rls.learn.export runs/act_main/best --out runs/act_main/policy.onnx

LoRA adapters are merged into the base weights before export, so a fine-tuned
model deploys as a plain network with no adapter code on the robot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rls.learn.policies.base import PolicyBundle
from rls.learn.policies.layers import merge_lora
from rls.learn.text import MAX_TOKENS


class _Wrapped(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(front, wrist, state, tokens)


def export_onnx(bundle_path: Path, out: Path, opset: int = 17, check_samples: int = 20) -> dict[str, float]:
    import onnxruntime as ort

    bundle = PolicyBundle.load(bundle_path)
    model = bundle.model.eval()
    merge_lora(model)
    cfg = model.cfg  # type: ignore[attr-defined]
    rng = np.random.default_rng(0)
    ex = (
        torch.from_numpy(rng.integers(0, 255, (1, *cfg.front_hw, 3), dtype=np.uint8)),
        torch.from_numpy(rng.integers(0, 255, (1, *cfg.wrist_hw, 3), dtype=np.uint8)),
        torch.randn(1, cfg.state_dim),
        torch.from_numpy(rng.integers(1, 20, (1, MAX_TOKENS))),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _Wrapped(model).eval()
    torch.onnx.export(
        wrapped,
        ex,
        str(out),
        input_names=["front", "wrist", "state", "tokens"],
        output_names=["actions"],
        dynamic_axes={k: {0: "batch"} for k in ("front", "wrist", "state", "tokens", "actions")},
        opset_version=opset,
        dynamo=False,
    )
    model.eval()  # export restores the wrapper's previous mode; make sure no dropout/augmentation runs
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    max_err, times = 0.0, []
    for _ in range(check_samples):
        f = rng.integers(0, 255, (1, *cfg.front_hw, 3), dtype=np.uint8)
        w = rng.integers(0, 255, (1, *cfg.wrist_hw, 3), dtype=np.uint8)
        s = rng.normal(size=(1, cfg.state_dim)).astype(np.float32)
        tk = rng.integers(1, 20, (1, MAX_TOKENS)).astype(np.int64)
        with torch.no_grad():
            ref = model(torch.from_numpy(f), torch.from_numpy(w), torch.from_numpy(s), torch.from_numpy(tk)).numpy()
        t0 = time.perf_counter()
        got = sess.run(["actions"], {"front": f, "wrist": w, "state": s, "tokens": tk})[0]
        times.append((time.perf_counter() - t0) * 1e3)
        max_err = max(max_err, float(np.abs(ref - got).max()))
    report = {"max_abs_diff": max_err, "onnx_p50_ms": float(np.median(times)), "onnx_p95_ms": float(np.percentile(times, 95)), "bytes": out.stat().st_size}
    (out.parent / "onnx_export.json").write_text(json.dumps(report, indent=1))
    if max_err > 1e-3:
        raise RuntimeError(f"ONNX parity check failed: max abs diff {max_err}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(export_onnx(a.bundle, a.out), indent=1))


if __name__ == "__main__":
    main()
