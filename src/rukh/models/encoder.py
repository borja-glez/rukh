"""``PositionEncoder``: a hand-written bidirectional transformer over positions.

The brother of ``MoveDecoder``, and deliberately built from the same ``rukh.models.layers``
blocks: the whole difference is the attention mask. The decoder answers "what comes next?", so
a position may only look left; the encoder answers "what is this position?", so every token
sees every other one, including the ones after it. That is why the test that proves the decoder
right (a future token never changes a past logit) has to fail here, and ``test_encoder`` asserts
the opposite.

Two input schemes share the class (``EncoderConfig.input``):

``moves``
    the game so far in the P1 UCI vocabulary, the very tokens the decoder was trained on, so
    the masked-move head can be tied to the embedding and a pretrained decoder's intuition is
    directly comparable.
``squares``
    the 69 tokens of ``rukh.models.squares``, the position itself rather than its history.

Padding is expressed as a key mask, not as a special attention: ``attention_mask`` is ``True``
on real tokens and ``False`` on ``<pad>``, and the padded keys are switched off for every
query. The rows of padded queries are still computed (nobody reads them) and at least one real
token per row is required, otherwise softmax would see an entirely masked row and return NaN.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rukh.models import layers
from rukh.models.config import EncoderConfig
from rukh.models.layers import rope_tables

PAD_ID = 0
"""``<pad>`` is id 0 in both schemes."""

MMM_IGNORE_INDEX = -100
"""Label of a position the masked-move loss must skip.

Not the decoder's ``0``. ``0`` is ``<pad>`` in **both** schemes, so using it to mean "nothing to
predict" would overload one id with two jobs: the loss could no longer tell a position it must
skip from a position where the right answer happens to be ``<pad>``. The decoder gets away with
it because its target is the next token of a packed stream and ``<pad>`` is never a target there;
here the distinction has to be explicit, and ``-100`` is outside every vocabulary, so "not
predicted" and "predict ``<pad>``" stay different things.
"""


def _tracing() -> bool:
    """``True`` while ``torch.export`` or ``torch.compile`` is capturing a graph.

    The padding check below reads a tensor to decide whether to raise, and a data-dependent
    branch like that is precisely what a graph capture cannot represent. Under ``torch.export``
    it would send the exporter back to the deprecated TorchScript tracer (D-027); under
    ``torch.compile`` it is worse in a quieter way — Dynamo cannot prove the condition, so it
    graph-breaks around it on **every** masked step of the MMM loop, which is a synchronisation
    point plus two half-graphs per step for a check that has already passed. Eager behaviour is
    unchanged: outside a capture both calls are ``False`` and the ValueError is raised as before.
    """
    return torch.compiler.is_exporting() or torch.compiler.is_compiling()


class PositionEncoder(nn.Module):
    """Bidirectional transformer encoder: ``forward`` returns the hidden states ``(B, T, d)``."""

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.tokens = nn.Embedding(self.cfg.tokens, self.cfg.d_model)
        self.positions = (
            nn.Embedding(self.cfg.seq, self.cfg.d_model) if self.cfg.pos == "learned" else None
        )
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList(
            layers.Block(
                self.cfg.d_model, self.cfg.n_head, self.cfg.ff, self.cfg.dropout, causal=False
            )
            for _ in range(self.cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        self.mlm_head = nn.Linear(self.cfg.d_model, self.cfg.tokens, bias=False)
        if self.cfg.tie_embeddings and self.cfg.input == "moves":
            # Tied only for ``moves``: it is the P1 vocabulary, where the embedding and the head
            # describe the same 2 030 moves. The 47 square tokens are too few to be worth tying.
            self.mlm_head.weight = self.tokens.weight
        if self.cfg.pos == "rope":
            cos, sin = rope_tables(self.cfg.seq, self.cfg.head_dim, torch.device("cpu"))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(layers.init_weights)
        scale = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; ``non_embedding`` drops the learned position table (nanoGPT rule)."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.positions is not None:
            total -= self.positions.weight.numel()
        return total

    @staticmethod
    def padding_mask(idx: Tensor) -> Tensor:
        """``True`` where ``idx`` is a real token, ``False`` on ``<pad>``; shape ``(B, T)``."""
        return idx != PAD_ID

    @staticmethod
    def _key_mask(attention_mask: Tensor | None) -> Tensor | None:
        """``(B, T)`` into the ``(B, 1, 1, T)`` boolean key mask SDPA expects."""
        if attention_mask is None:
            return None
        mask = attention_mask.bool()
        if mask.dim() != 2:
            raise ValueError(f"attention_mask must be (B, T), got {tuple(attention_mask.shape)}")
        if not _tracing() and not bool(mask.any(dim=-1).all()):
            # An entirely masked row would make softmax return NaN, so it is refused here rather
            # than debugged three layers down. The check reads a tensor, which is exactly what
            # `torch.export` cannot trace (a data-dependent guard), and skipping it under the
            # exporter is what keeps the encoder on the modern exporter instead of the
            # deprecated tracer: the exported graph is a pure function of its input either way.
            raise ValueError("every sequence needs at least one unmasked token")
        return mask[:, None, None, :]

    def forward(self, idx: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Hidden states ``(B, T, d_model)``; ``attention_mask`` is ``True`` on real tokens."""
        seq = idx.shape[-1]
        if seq > self.cfg.seq:
            raise ValueError(f"sequence of {seq} tokens is longer than block {self.cfg.seq}")
        key_mask = self._key_mask(attention_mask)
        x = self.tokens(idx)
        cos = sin = None
        if self.positions is not None:
            steps = torch.arange(seq, device=idx.device)
            x = x + self.positions(steps)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, cos, sin, key_mask)
        return self.ln_f(x)

    def masked_lm(
        self, idx: Tensor, labels: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Masked-move logits ``(B, T, V)`` and, with ``labels``, the scalar loss.

        ``labels`` is ``MMM_IGNORE_INDEX`` wherever nothing was masked; see ``rukh.train.mmm``.
        """
        logits = self.mlm_head(self(idx, attention_mask))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=MMM_IGNORE_INDEX,
            )
        return logits, loss

    def pool(
        self,
        idx: Tensor,
        how: Literal["cls", "mean"] = "mean",
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """One vector per sequence, ``(B, d_model)``: the position's representation.

        ``cls`` takes position 0 (``<bos>`` for ``moves``, ``<cls>`` for ``squares``) and
        ``mean`` averages the real tokens only. Without an explicit ``attention_mask`` the
        padding is read off ``idx`` itself, so ``pool(idx, "mean")`` never averages ``<pad>``
        into the representation, which is the whole point of the operation.
        """
        if how not in ("cls", "mean"):
            raise ValueError(f"how must be 'cls' or 'mean', got {how!r}")
        mask = self.padding_mask(idx) if attention_mask is None else attention_mask.bool()
        hidden = self(idx, mask)
        if how == "cls":
            return hidden[:, 0]
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
