"""Building blocks: image encoder, transformer layers with plain Linear projections (so
LoRA can wrap them), and LoRA itself."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen. B starts at zero, so wrapping is a no-op."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale

    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.lora_b @ self.lora_a * self.scale)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def add_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0, targets: tuple[str, ...] = ("q", "k", "v", "o", "ff1", "ff2")) -> int:
    """Wrap matching Linear layers in place. Returns the number wrapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha))
            n += 1
        else:
            n += add_lora(child, rank, alpha, targets)
    return n


def merge_lora(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged())
        else:
            merge_lora(child)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        m = ctx.shape[1]
        q = self.q(x).view(b, n, self.h, d // self.h).transpose(1, 2)
        k = self.k(ctx).view(b, m, self.h, d // self.h).transpose(1, 2)
        v = self.v(ctx).view(b, m, self.h, d // self.h).transpose(1, 2)
        attn_mask = None if mask is None else ~mask[:, None, None, :]  # mask: True = padding
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Pre-norm transformer block; cross-attends to ``ctx`` when given (decoder)."""

    def __init__(self, d: int, heads: int, cross: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.cross = Attention(d, heads, dropout) if cross else None
        self.nc = nn.LayerNorm(d) if cross else None
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor | None = None, ctx_mask: torch.Tensor | None = None, self_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, self_mask))
        if self.cross is not None and ctx is not None:
            x = x + self.drop(self.cross(self.nc(x), ctx, ctx_mask))  # type: ignore[misc]
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))


class ConvEncoder(nn.Module):
    """Small CNN -> spatial feature tokens (B, H'W', d) with learned 2D position embeddings.

    Sized for CPU training and inference (~20 MFLOPs per 120x160 frame): four
    stride-2 stages, extra residual convolutions only at the two low-resolution
    stages, GroupNorm throughout.
    """

    def __init__(self, d: int, in_hw: tuple[int, int], width: int = 24, keypoints: int = 0) -> None:
        super().__init__()
        self.keypoints = keypoints
        self.film_channels = [width, 2 * width, 4 * width, 4 * width]  # per-stage channels, for FiLM conditioning
        chans = [3, width, 2 * width, 4 * width, 4 * width]
        self.stages = nn.ModuleList()
        self.refine = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1), nn.GroupNorm(8, chans[i + 1]), nn.GELU()))
            self.refine.append(
                nn.Sequential(nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1), nn.GroupNorm(8, chans[i + 1]), nn.GELU()) if i >= 2 else nn.Identity()
            )
        self.proj = nn.Conv2d(chans[-1], d, 1)
        h, w = in_hw
        for _ in range(4):
            h, w = (h + 1) // 2, (w + 1) // 2
        self.out_hw = (h, w)
        self.pos = nn.Parameter(torch.randn(1, h * w, d) * 0.02)
        if keypoints:
            # spatial-softmax readout (as in robomimic / Diffusion Policy): each keypoint is the expected image
            # position of one learned feature map, so position is explicit instead of learned per token
            self.kp = nn.Conv2d(chans[-1], keypoints, 1)
            self.kp_temp = nn.Parameter(torch.zeros(1))
            self.kp_proj = nn.Linear(2, d)
            self.kp_emb = nn.Parameter(torch.randn(1, keypoints, d) * 0.02)
            self.pool_proj = nn.Linear(chans[-1], d)
            gy, gx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            self.register_buffer("grid", torch.stack([gx.flatten(), gy.flatten()], -1), persistent=False)

    @property
    def n_tokens(self) -> int:
        return self.keypoints + 1 if self.keypoints else self.out_hw[0] * self.out_hw[1]

    def forward(self, img: torch.Tensor, film: torch.Tensor | None = None) -> torch.Tensor:
        """``film``: (B, 2 * sum(film_channels)) per-channel scale and shift for every stage, or None."""
        x = img
        off = 0
        for stage, ref in zip(self.stages, self.refine, strict=True):
            x = stage(x)
            x = x + ref(x) if not isinstance(ref, nn.Identity) else x
            if film is not None:
                c = x.shape[1]
                gamma, beta = film[:, off : off + c], film[:, off + c : off + 2 * c]
                x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
                off += 2 * c
        if self.keypoints:
            att = torch.softmax(self.kp(x).flatten(2) * torch.exp(self.kp_temp), dim=-1)  # (B, K, HW)
            xy = att @ self.grid  # (B, K, 2) expected coordinates in [-1, 1]
            pooled = self.pool_proj(x.mean((2, 3))).unsqueeze(1)  # appearance summary token
            return torch.cat([self.kp_proj(xy) + self.kp_emb, pooled], 1)
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style augmentation: pad by replication and crop back at a random offset per sample."""
    b, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    ac = torch.linspace(-1.0 + eps, 1.0 - eps, w + 2 * pad, device=x.device)[:w]
    gy, gx = torch.meshgrid(ar, ac, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=x.device).float()
    shift[..., 0] *= 2.0 / (w + 2 * pad)
    shift[..., 1] *= 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)
