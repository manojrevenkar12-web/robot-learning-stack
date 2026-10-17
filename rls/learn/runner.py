"""Policy execution with action chunking: the same code runs in sim benchmarks,
the ONNX runtime and the ROS 2 policy node."""

from __future__ import annotations

import time
from collections import deque
from typing import Protocol

import numpy as np
import torch

from rls.learn.policies.base import PolicyBundle
from rls.learn.text import tokenize


class ChunkModel(Protocol):
    def __call__(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray, tokens: np.ndarray) -> np.ndarray: ...


class TorchModel:
    def __init__(self, bundle: PolicyBundle, threads: int = 2) -> None:
        torch.set_num_threads(threads)
        self.model = bundle.model.eval()

    @torch.no_grad()
    def __call__(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        out = self.model(torch.from_numpy(front), torch.from_numpy(wrist), torch.from_numpy(state), torch.from_numpy(tokens))
        return out.numpy()


class OnnxModel:
    def __init__(self, path: str, threads: int = 2) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        return self.sess.run(["actions"], {"front": front, "wrist": wrist, "state": state, "tokens": tokens})[0]


class ChunkRunner:
    """Turns chunk predictions into one action per control tick.

    ``mode="ensemble"``: predict every tick and average all predictions for the
    current tick with weights exp(-k * i), i = 0 for the oldest (ACT temporal ensembling).
    ``mode="chunk"``: predict every ``replan`` ticks and play the chunk open loop.
    """

    def __init__(
        self, bundle: PolicyBundle, model: ChunkModel | None = None, mode: str = "ensemble", k: float = 0.1, replan: int = 8, grip_threshold: float = 0.5
    ) -> None:
        self.bundle = bundle
        self.grip_threshold = grip_threshold
        from rls.learn.action_space import make

        self.space = make(str(getattr(bundle.model.cfg, "action_space", "joint")))
        self.n = bundle.normalizer
        self.model = model or TorchModel(bundle)
        self.chunk = bundle.chunk
        self.mode = mode if self.chunk > 1 else "chunk"
        self.k = k
        self.replan = replan if self.chunk > 1 else 1
        self.latencies_ms: list[float] = []
        self.reset("")

    def reset(self, instruction: str) -> None:
        self.tokens = np.array([tokenize(instruction)], dtype=np.int64)
        self.t = 0
        self.preds: deque[tuple[int, np.ndarray]] = deque(maxlen=self.chunk)
        self.plan: np.ndarray | None = None
        self.plan_t = -1

    def _predict(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.model(front[None], wrist[None], self.n.norm_state(state)[None].astype(np.float32), self.tokens)[0]
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)
        return self.n.denorm_action(out)

    def act(self, front: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.mode == "ensemble":
            self.preds.append((self.t, self._predict(front, wrist, state)))
            acts: list[np.ndarray] = []
            ws: list[float] = []
            for t_pred, chunk in self.preds:  # oldest first; as in ACT, older predictions weigh more
                age = self.t - t_pred
                if age < len(chunk):
                    acts.append(chunk[age])
                    ws.append(np.exp(-self.k * len(ws)))
            w = np.array(ws)[:, None]
            a = (np.array(acts) * w).sum(0) / w.sum()
        else:
            if self.plan is None or self.t - self.plan_t >= self.replan:
                self.plan, self.plan_t = self._predict(front, wrist, state), self.t
            a = self.plan[self.t - self.plan_t]
        self.t += 1
        a = a.copy() if self.space is None else self.space.to_joint(a, state[:7])
        # the gripper is a binary skill in every demonstration (open = 1, close = 0); a regressed or
        # ensembled value near 0.5 would half-close the fingers on the cube and never build grip force
        a[7] = 1.0 if a[7] > self.grip_threshold else 0.0
        return a
