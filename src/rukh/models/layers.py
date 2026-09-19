"""Transformer building blocks shared by the causal decoder and the bidirectional encoder.

The only structural difference between ``MoveDecoder`` and ``PositionEncoder`` is the attention
mask: the decoder may look left, the encoder may look everywhere. Everything else (pre-norm
residual blocks, the fused ``qkv`` projection, the GELU MLP, rotary embeddings) is literally the
same code, so it lives here once and both models import it. The classes take plain numbers
rather than a config object precisely so neither model has to depend on the other's config.

Module attribute names (``ln1``, ``attn.qkv``, ``attn.proj``, ``ln2``, ``mlp.fc``, ``mlp.proj``)
and their creation order are part of the on-disk format: they are the keys of every checkpoint
written so far, and changing them would silently break ``load_state``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class SelfAttention(nn.Module):
    """Multi-head self-attention with a single fused ``qkv`` projection.

    With ``causal=True`` the attention is the decoder's: ``is_causal=True`` and no mask tensor,
    which is the fast SDPA path. With ``causal=False`` it is the encoder's: every position sees
    every other one, except the keys an ``attn_mask`` switches off.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
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
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.resid_drop(self.proj(out))


class Mlp(nn.Module):
    """Position-wise GELU feed-forward network."""

    def __init__(self, d_model: int, ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, ff)
        self.proj = nn.Linear(ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block: ``x + attn(ln1(x))`` then ``x + mlp(ln2(x))``."""

    def __init__(
        self, d_model: int, n_head: int, ff: int, dropout: float = 0.0, causal: bool = True
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, dropout, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, ff, dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        return x + self.mlp(self.ln2(x))


def init_weights(module: nn.Module) -> None:
    """Normal(0, 0.02) on every ``Linear`` and ``Embedding``; biases start at zero."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
