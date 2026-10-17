"""Model-agnostic policy interface, registry and the deployable bundle.

Anything that implements :class:`Policy` (``forward`` + ``compute_loss`` with
the shared batch layout) can be trained, fine-tuned, evaluated, exported to
ONNX and served by the ROS 2 runtime without touching those components. A
wrapper around an external foundation model (e.g. a LeRobot SmolVLA policy)
plugs in the same way.

Batch layout (all tensors):
    front   (B, H, W, 3) uint8      wrist (B, h, w, 3) uint8
    state   (B, 8) float, normalised
    tokens  (B, L) int64             instruction token ids
    action  (B, chunk, 8) float, normalised;  action_valid (B, chunk) bool
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

REGISTRY: dict[str, type[Policy]] = {}


def register(name: str):
    def deco(cls: type[Policy]) -> type[Policy]:
        REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return deco


class Policy(nn.Module):
    registry_name: ClassVar[str] = ""
    Config: ClassVar[type]
    cfg: Any  # the policy's Config dataclass instance

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def build(cls, name: str, cfg: dict[str, Any]) -> Policy:
        klass = REGISTRY[name]
        names = {f.name for f in fields(klass.Config)}
        clean = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in names}
        return klass(klass.Config(**clean))


@dataclass
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def fit(states: np.ndarray, actions: np.ndarray, min_std: float = 1e-2) -> Normalizer:
        return Normalizer(states.mean(0), np.maximum(states.std(0), min_std), actions.mean(0), np.maximum(actions.std(0), min_std))

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        return ((s - self.state_mean) / self.state_std).astype(np.float32)

    def norm_action(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.action_mean) / self.action_std).astype(np.float32)

    def denorm_action(self, a: np.ndarray) -> np.ndarray:
        return a * self.action_std + self.action_mean

    def to_dict(self) -> dict[str, list[float]]:
        return {k: getattr(self, k).tolist() for k in ("state_mean", "state_std", "action_mean", "action_std")}

    @staticmethod
    def from_dict(d: dict[str, list[float]]) -> Normalizer:
        return Normalizer(*(np.asarray(d[k], np.float64) for k in ("state_mean", "state_std", "action_mean", "action_std")))


@dataclass
class PolicyBundle:
    """Everything needed to run a trained policy: weights, normaliser, config, lineage."""

    model: Policy
    normalizer: Normalizer
    meta: dict[str, Any]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "bundle.json").write_text(
            json.dumps(
                {"policy": self.model.registry_name, "config": self.model.describe(), "normalizer": self.normalizer.to_dict(), "meta": self.meta},
                indent=1,
                default=str,
            )
        )

    @staticmethod
    def load(path: Path, map_location: str = "cpu") -> PolicyBundle:
        b = json.loads((path / "bundle.json").read_text())
        model = Policy.build(b["policy"], b["config"])
        model.load_state_dict(torch.load(path / "model.pt", map_location=map_location))
        model.eval()
        return PolicyBundle(model, Normalizer.from_dict(b["normalizer"]), b["meta"])

    @property
    def chunk(self) -> int:
        return int(self.model.cfg.chunk)  # type: ignore[attr-defined]


def count_parameters(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
