"""``MoveDecoder``: a hand-written GPT decoder over the fixed UCI vocabulary.

Pure model code: it knows about tensors and nothing about the data pipeline, so it can be
imported by the training loop, the sampler and the ONNX exporter alike. Blocks are pre-norm
(``x = x + attn(ln1(x))``; ``x = x + mlp(ln2(x))``), attention is causal through
``F.scaled_dot_product_attention(..., is_causal=True)``, the MLP uses GELU and the language
modelling head is tied to the token embedding.

The blocks themselves live in ``rukh.models.layers``, shared with the bidirectional
``PositionEncoder``; the classes below are the ``DecoderConfig`` adapters of those blocks, so
the module names in a checkpoint (``blocks.N.attn.qkv`` and friends) are unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rukh.models import layers
from rukh.models.config import DecoderConfig
from rukh.models.layers import apply_rope, rope_tables

__all__ = [
    "IGNORE_INDEX",
    "Block",
    "CausalSelfAttention",
    "Mlp",
    "MoveDecoder",
    "apply_rope",
    "rope_tables",
]

# ``<pad>`` is id 0 in every scheme; the loss must ignore it (see ``loader.IGNORE_INDEX``).
IGNORE_INDEX = 0


class CausalSelfAttention(layers.SelfAttention):
    """Multi-head causal self-attention with a single fused ``qkv`` projection."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__(cfg.d_model, cfg.n_head, cfg.dropout, causal=True)


class Mlp(layers.Mlp):
    """Position-wise GELU feed-forward network."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__(cfg.d_model, cfg.ff, cfg.dropout)


class Block(layers.Block):
    """One pre-norm transformer block."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__(cfg.d_model, cfg.n_head, cfg.ff, cfg.dropout, causal=True)


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
        layers.init_weights(module)

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
