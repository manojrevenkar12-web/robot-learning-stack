"""Diffusion policy head on the shared observation encoder.

Same observation encoder as :mod:`chunking` (front + wrist images, proprioception,
instruction). The decoder denoises a whole action chunk: its queries are the
noisy actions plus a diffusion-timestep embedding, cross-attending to the
encoded observation.

Choices that matter on a small CPU budget:

* the network predicts the clean chunk (``x0``) rather than the noise; noise
  prediction converges much more slowly at a few thousand steps;
* cosine noise schedule with ``train_steps`` levels, deterministic DDIM sampling
  with ``infer_steps`` steps at run time. The observation is encoded once and only
  the decoder runs per denoising step;
* the initial latent is a fixed buffer, so inference is deterministic: the ONNX
  graph (DDIM unrolled) reproduces PyTorch exactly and closed-loop runs are
  repeatable. Multimodality still comes from the observation-conditioned
  denoising path, not from resampling.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from rls.learn.policies.base import register
from rls.learn.policies.chunking import ChunkConfig, ChunkTransformer


@dataclass
class DiffusionConfig(ChunkConfig):
    train_steps: int = 100
    infer_steps: int = 10
    seed: int = 0


def _cosine_alpha_bar(n: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, n, n + 1, dtype=torch.float64) / n
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab[1:].float()  # alpha_bar for t = 1..n


def _t_embed(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float().unsqueeze(-1) * freqs
    return torch.cat([a.sin(), a.cos()], -1)


@register("diffusion_transformer")
class DiffusionTransformer(ChunkTransformer):
    Config = DiffusionConfig
    cfg: DiffusionConfig
    alpha_bar: torch.Tensor
    x_init: torch.Tensor
    infer_ts: torch.Tensor

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__(cfg)
        d = cfg.d_model
        self.act_in = nn.Linear(cfg.action_dim, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("alpha_bar", _cosine_alpha_bar(cfg.train_steps), persistent=False)
        g = torch.Generator().manual_seed(cfg.seed)
        self.register_buffer("x_init", torch.randn(1, cfg.chunk, cfg.action_dim, generator=g))
        ts = torch.linspace(cfg.train_steps - 1, 0, cfg.infer_steps).round().long()
        self.register_buffer("infer_ts", ts, persistent=False)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, mem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict the clean chunk from a noisy one at diffusion step(s) ``t`` (B,)."""
        temb = self.t_mlp(_t_embed(t, self.cfg.d_model)).unsqueeze(1)
        q = self.queries + self.act_in(x_t) + temb
        for blk in self.decoder:
            q = blk(q, ctx=mem, ctx_mask=mask)
        return self.head(self.head_norm(q))

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mem, mask = self.encode(front, wrist, state, tokens)
        b = state.shape[0]
        x = self.x_init.expand(b, -1, -1)
        ts = self.infer_ts.tolist()
        for i, t in enumerate(ts):  # deterministic DDIM (eta = 0), unrolled when exported
            ab = self.alpha_bar[t]
            x0 = self.denoise(x, torch.full((b,), t, dtype=torch.long, device=state.device), mem, mask)
            if i + 1 == len(ts):
                return x0
            ab_next = self.alpha_bar[ts[i + 1]]
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mem, mask = self.encode(batch["front"], batch["wrist"], batch["state"], batch["tokens"])
        x0 = batch["action"][:, : self.cfg.chunk]
        valid = batch["action_valid"][:, : self.cfg.chunk].unsqueeze(-1).float()
        b = x0.shape[0]
        t = torch.randint(0, self.cfg.train_steps, (b,), device=x0.device)
        ab = self.alpha_bar[t].view(b, 1, 1)
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.denoise(x_t, t, mem, mask)
        l1 = ((pred - x0).abs() * valid).sum() / (valid.sum() * x0.shape[-1]).clamp(min=1)
        with torch.no_grad():
            first = (pred[:, 0] - x0[:, 0]).abs().mean()
        return self.add_aux_loss({"loss": l1, "l1_first_step": first}, mem, mask, batch)

    def describe(self) -> dict[str, Any]:
        return {"type": "diffusion_transformer", **asdict(self.cfg)}
