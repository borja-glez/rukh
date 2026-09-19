"""Configuration of the hand-written models: the GPT decoder, its presets and the encoder."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from rukh.config import BaseConfig
from rukh.models.squares import SQUARE_TOKENS, SQUARE_VOCAB_SIZE


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


class EncoderConfig(BaseConfig):
    """Shape of a ``PositionEncoder``.

    ``input`` picks the representation the encoder reads: ``moves`` reuses the P1 UCI
    vocabulary and the ``block`` context of the decoder, ``squares`` reads the 69 fixed tokens
    of ``rukh.models.squares``. The rest mirrors ``DecoderConfig``, on purpose: the two models
    are the same blocks with a different attention mask.
    """

    input: Literal["moves", "squares"] = "moves"
    vocab_size: int = 2030  # "moves": the P1 vocabulary
    square_vocab: int = SQUARE_VOCAB_SIZE  # "squares": <pad>/<mask>/<cls>, pieces, turn, ...
    n_layer: int = 8
    n_head: int = 6
    d_model: int = 384
    d_ff: int | None = None  # None -> 4 * d_model
    block: int = 200  # "moves"; "squares" always uses its 69 fixed positions
    dropout: float = 0.1
    pos: Literal["learned", "rope"] = "learned"
    tie_embeddings: bool = True
    """Tie the masked-move head to the token embedding; only ever applied to ``moves``."""

    @model_validator(mode="after")
    def _check(self) -> EncoderConfig:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        if min(self.vocab_size, self.square_vocab, self.n_layer, self.n_head, self.d_model) < 1:
            raise ValueError("vocab_size, square_vocab, n_layer, n_head and d_model must be > 0")
        if self.block < 1:
            raise ValueError(f"block must be positive, got {self.block}")
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

    @property
    def tokens(self) -> int:
        """Size of the vocabulary this scheme reads."""
        return self.vocab_size if self.input == "moves" else self.square_vocab

    @property
    def seq(self) -> int:
        """Longest sequence this scheme produces: ``block`` for moves, 69 for squares."""
        return self.block if self.input == "moves" else SQUARE_TOKENS
