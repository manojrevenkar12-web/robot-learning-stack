"""Language-conditioned action-chunking transformer (ACT-style, without the CVAE).

Observation tokens (front + wrist image features, proprioception, instruction
words) go through a transformer encoder; a decoder with ``chunk`` learned
queries predicts the next ``chunk`` actions at once. ``chunk=1`` gives the
single-step behaviour-cloning baseline with the same backbone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rls.learn.policies.base import Policy, register
from rls.learn.policies.layers import Block, ConvEncoder, random_shift
from rls.learn.text import MAX_TOKENS, VOCAB


@dataclass
class ChunkConfig:
    d_model: int = 160
    heads: int = 4
    enc_layers: int = 3
    dec_layers: int = 2
    chunk: int = 20
    dropout: float = 0.1
    cnn_width: int = 24
    front_hw: tuple[int, int] = (120, 160)
    wrist_hw: tuple[int, int] = (96, 128)
    state_dim: int = 8
    action_dim: int = 8
    augment_shift: int = 4
    use_language: bool = True
    action_space: str = "joint"  # joint | ee (see rls.learn.action_space)
    keypoints: int = 0  # >0: spatial-softmax keypoint readout per camera instead of one token per feature-map cell
    # proprioception regularisation: with the joint state as input, behaviour cloning can predict the next action
    # from where the arm already is and ignore the cameras ("copycat" shortcut); noise and dropout on the state
    # token during training force the policy to localise objects visually
    use_state: bool = True
    state_noise: float = 0.0  # std of Gaussian noise on the normalised state (training only)
    state_dropout: float = 0.0  # probability of zeroing the state token per sample (training only)
    # auxiliary object localisation (training only): two extra decoder queries regress where the instructed
    # object and its goal (bin or support cube) are, from the same instruction-conditioned memory the action
    # queries read. Labels come from the scene layout recorded with simulated episodes.
    aux_object_weight: float = 0.0
    # FiLM: the instruction modulates every CNN stage (per-channel scale and shift, as in RT-1), so "the blue
    # cube" can light up blue regions before the transformer sees any tokens. Zero-initialised: a no-op at start.
    film: bool = False


@register("chunk_transformer")
class ChunkTransformer(Policy):
    Config = ChunkConfig

    def __init__(self, cfg: ChunkConfig) -> None:
        super().__init__()
        from rls.learn.action_space import ACTION_DIMS

        cfg.action_dim = ACTION_DIMS[cfg.action_space]
        self.cfg = cfg
        d = cfg.d_model
        self.front = ConvEncoder(d, cfg.front_hw, cfg.cnn_width, cfg.keypoints)
        self.wrist = ConvEncoder(d, cfg.wrist_hw, cfg.cnn_width, cfg.keypoints)
        if cfg.film:
            n_film = 2 * sum(self.front.film_channels)
            # word order matters ("stack the blue cube on the yellow cube"): read the whole token sequence, not a mean
            self.film_trunk = nn.Sequential(nn.Linear(MAX_TOKENS * d, 128), nn.GELU())
            self.film_front = nn.Linear(128, n_film)
            self.film_wrist = nn.Linear(128, n_film)
            for lin in (self.film_front, self.film_wrist):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.tok = nn.Embedding(len(VOCAB), d, padding_idx=0)
        self.tok_pos = nn.Parameter(torch.randn(1, MAX_TOKENS, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)  # front, wrist, state, language
        self.encoder = nn.ModuleList([Block(d, cfg.heads, dropout=cfg.dropout) for _ in range(cfg.enc_layers)])
        self.queries = nn.Parameter(torch.randn(1, cfg.chunk, d) * 0.02)
        self.decoder = nn.ModuleList([Block(d, cfg.heads, cross=True, dropout=cfg.dropout) for _ in range(cfg.dec_layers)])
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.action_dim)
        if cfg.aux_object_weight:
            self.aux_queries = nn.Parameter(torch.randn(1, 2, d) * 0.02)
            self.aux_head = nn.Linear(d, 2)

    # ------------------------------------------------------------------ model
    @staticmethod
    def _img(x: torch.Tensor) -> torch.Tensor:
        # uint8 (B, H, W, 3) -> float (B, 3, H, W), roughly zero-mean
        return (x.permute(0, 3, 1, 2).float() / 255.0 - 0.45) / 0.25

    def encode(self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f, w = self._img(front), self._img(wrist)
        if self.training and self.cfg.augment_shift:
            f, w = random_shift(f, self.cfg.augment_shift), random_shift(w, self.cfg.augment_shift)
        b = state.shape[0]
        if self.training and self.cfg.state_noise:
            state = state + torch.randn_like(state) * self.cfg.state_noise
        s_tok = self.state_proj(state).unsqueeze(1)
        if not self.cfg.use_state:
            s_tok = s_tok * 0.0
        elif self.training and self.cfg.state_dropout:
            s_tok = s_tok * (torch.rand(b, 1, 1, device=state.device) >= self.cfg.state_dropout).float()
        film_f = film_w = None
        if self.cfg.film:
            lang = self.film_trunk(self.tok(tokens).flatten(1))  # padding id 0 embeds to zeros
            film_f, film_w = self.film_front(lang), self.film_wrist(lang)
        parts = [self.front(f, film_f) + self.type_emb[0], self.wrist(w, film_w) + self.type_emb[1], s_tok + self.type_emb[2]]
        masks = [torch.zeros(b, p.shape[1], dtype=torch.bool, device=state.device) for p in parts]
        if self.cfg.use_language:
            parts.append(self.tok(tokens) + self.tok_pos[:, : tokens.shape[1]] + self.type_emb[3])
            masks.append(tokens == 0)
        x = torch.cat(parts, 1)
        mask = torch.cat(masks, 1)
        for blk in self.encoder:
            x = blk(x, self_mask=mask)
        return x, mask

    def forward(self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Normalised action chunk (B, chunk, action_dim) from normalised state."""
        mem, mask = self.encode(front, wrist, state, tokens)
        return self.decode(mem, mask)

    def decode(self, mem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.queries.expand(mem.shape[0], -1, -1)
        for blk in self.decoder:
            q = blk(q, ctx=mem, ctx_mask=mask)
        return self.head(self.head_norm(q))

    def add_aux_loss(self, out: dict[str, torch.Tensor], mem: torch.Tensor, mask: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Auxiliary object localisation: two queries through the same decoder, read by a separate head."""
        if not self.cfg.aux_object_weight or "aux_xy" not in batch:
            return out
        q = self.aux_queries.expand(mem.shape[0], -1, -1)
        for blk in self.decoder:
            q = blk(q, ctx=mem, ctx_mask=mask)
        aux = self.aux_head(self.head_norm(q))
        av = batch["aux_valid"].unsqueeze(-1).float()
        aux_l1 = ((aux - batch["aux_xy"]).abs() * av).sum() / (av.sum() * 2).clamp(min=1)
        return {**out, "loss": out["loss"] + self.cfg.aux_object_weight * aux_l1, "aux_l1": aux_l1.detach()}

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mem, mask = self.encode(batch["front"], batch["wrist"], batch["state"], batch["tokens"])
        pred = self.decode(mem, mask)
        tgt = batch["action"][:, : self.cfg.chunk]
        valid = batch["action_valid"][:, : self.cfg.chunk].unsqueeze(-1).float()
        l1 = (F.l1_loss(pred, tgt, reduction="none") * valid).sum() / (valid.sum() * pred.shape[-1]).clamp(min=1)
        with torch.no_grad():
            first = (pred[:, 0] - tgt[:, 0]).abs().mean()
        return self.add_aux_loss({"loss": l1, "l1_first_step": first}, mem, mask, batch)

    def describe(self) -> dict[str, Any]:
        return {"type": "chunk_transformer", **asdict(self.cfg)}
