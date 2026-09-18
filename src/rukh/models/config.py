"""Configuration of the hand-written GPT decoder and the three course presets."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from rukh.config import BaseConfig


class DecoderConfig(BaseConfig):
    """Shape of a ``MoveDecoder``.

    ``d_ff`` defaults to ``4 * d_model`` and ``pos`` picks learned positional embeddings or
    rotary embeddings (RoPE) applied to the queries and keys of every head.
    """

    vocab_size: int = 2030
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    d_ff: int | None = None  # None -> 4 * d_model
    block: int = 200
    dropout: float = 0.0
    pos: Literal["learned", "rope"] = "learned"
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def _check(self) -> DecoderConfig:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        if min(self.vocab_size, self.n_layer, self.n_head, self.d_model, self.block) < 1:
            raise ValueError("vocab_size, n_layer, n_head, d_model and block must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.d_ff is not None and self.d_ff < 1:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.pos == "rope" and (self.d_model // self.n_head) % 2:
            raise ValueError("rope needs an even head dimension")
        return self

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.d_model // self.n_head

    @property
    def ff(self) -> int:
        """Hidden width of the MLP (``d_ff`` or ``4 * d_model``)."""
        return self.d_ff if self.d_ff is not None else 4 * self.d_model


PRESETS: dict[str, DecoderConfig] = {
    "tiny": DecoderConfig(n_layer=6, n_head=4, d_model=256),
    "small": DecoderConfig(),
    "medium": DecoderConfig(n_layer=16, n_head=12, d_model=768),
}


def preset(name: str) -> DecoderConfig:
    """A copy of a named preset, so callers may mutate it freely."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {', '.join(PRESETS)}")
    return PRESETS[name].model_copy(deep=True)
