"""``MoveDecoder``: a hand-written GPT decoder over the fixed UCI vocabulary.

Pure model code: it knows about tensors and nothing about the data pipeline, so it can be
imported by the training loop, the sampler and the ONNX exporter alike. Blocks are pre-norm
(``x = x + attn(ln1(x))``; ``x = x + mlp(ln2(x))``), attention is causal through
``F.scaled_dot_product_attention(..., is_causal=True)``, the MLP uses GELU and the language
modelling head is tied to the token embedding.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rukh.models.config import DecoderConfig

# ``<pad>`` is id 0 in every scheme; the loss must ignore it (see ``loader.IGNORE_INDEX``).
IGNORE_INDEX = 0


def rope_tables(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10_000.0
) -> tuple[Tensor, Tensor]:
    """``(cos, sin)`` of shape ``(seq_len, head_dim)`` for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    full = torch.cat([angles, angles], dim=-1)
    return full.cos().to(device), full.sin().to(device)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``(B, H, T, D)`` by the angles of the first ``T`` positions."""
    t = x.shape[-2]
    cos_t = cos[:t].to(dtype=x.dtype).view(1, 1, t, -1)
    sin_t = sin[:t].to(dtype=x.dtype).view(1, 1, t, -1)
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos_t + rotated * sin_t


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a single fused ``qkv`` projection."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=2)
        shape = (batch, seq, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.resid_drop(self.proj(out))


class Mlp(nn.Module):
    """Position-wise GELU feed-forward network."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.ff)
        self.proj = nn.Linear(cfg.ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = Mlp(cfg)

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class MoveDecoder(nn.Module):
    """GPT decoder over move tokens: ``forward`` returns ``(logits, loss)``.

    ``logits`` is always ``(B, T, vocab_size)``; ``loss`` is ``None`` unless ``targets`` is
    given, in which case it is the mean cross entropy with ``ignore_index=0`` (``<pad>``).
    """

    def __init__(self, cfg: DecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        self.tokens = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.positions = (
            nn.Embedding(self.cfg.block, self.cfg.d_model) if self.cfg.pos == "learned" else None
        )
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.tokens.weight
        if self.cfg.pos == "rope":
            cos, sin = rope_tables(self.cfg.block, self.cfg.head_dim, torch.device("cpu"))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; ``non_embedding`` drops the learned position table (nanoGPT rule)."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.positions is not None:
            total -= self.positions.weight.numel()
        return total

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Logits ``(B, T, V)`` for ``idx`` ``(B, T)`` and, with ``targets``, the scalar loss."""
        seq = idx.shape[-1]
        if seq > self.cfg.block:
            raise ValueError(f"sequence of {seq} tokens is longer than block {self.cfg.block}")
        x = self.tokens(idx)
        cos = sin = None
        if self.positions is not None:
            steps = torch.arange(seq, device=idx.device)
            x = x + self.positions(steps)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @torch.no_grad()
    def next_logits(self, idx: Tensor) -> Tensor:
        """Logits of the last step only, ``(B, vocab_size)``, cropped to the block size."""
        cropped = idx[:, -self.cfg.block :]
        logits, _ = self(cropped)
        return logits[:, -1, :]
