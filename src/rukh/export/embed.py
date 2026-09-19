"""Mean-pooled position embeddings: the encoder's secondary output, for phase 2.

``docs/spec/02`` §"Componente 2" asks for one vector per position so that similar positions can
be retrieved later (the A1 agent of phase 2). The vector is ``PositionEncoder.pool(idx, "mean")``
over the ``squares`` tokens of a FEN: the average of the real tokens, which is why padding never
enters it.

A matrix of floats is useless without knowing which row is which position, and a ``.npy`` file
has no room to say so. Every run therefore writes two files: the array, and a parquet sidecar
with ``row`` and ``fen4`` in the array's own order. The four-field FEN is the key P1 uses for
positions and evaluations, so the sidecar joins straight back to them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

FEN_COLUMN = "fen"
SIDECAR_SUFFIX = ".parquet"
POOLING = "mean"


class EmbedResult(BaseModel):
    """What one embedding run wrote."""

    model_config = ConfigDict(extra="forbid")

    positions: int
    dim: int
    array: str
    sidecar: str
    pooling: str = POOLING
    input: str = "squares"
    checkpoint: str


def fen4(fen: str) -> str:
    """The four-field FEN: placement, side to move, castling and en passant."""
    return " ".join(str(fen).split(" ")[:4])


def _encoder(ckpt: Path, device: str | None = None):  # type: ignore[no-untyped-def]
    """The ``PositionEncoder`` inside a checkpoint, whether or not it carries the heads."""
    from rukh.models.heads import MultiHead
    from rukh.train import load_any

    model, _payload, kind = load_any(Path(ckpt), map_location=device or "cpu")
    if kind != "encoder":
        raise ValueError(f"{ckpt} is a {kind} checkpoint; embeddings need an encoder")
    return model.encoder if isinstance(model, MultiHead) else model


def embed_positions(
    ckpt: Path,
    positions: Path,
    out: Path,
    batch_size: int = 256,
    device: str | None = None,
    column: str = FEN_COLUMN,
) -> EmbedResult:
    """Embed every position of a parquet and write the array plus its ``fen4`` sidecar."""
    import numpy as np
    import torch

    from rukh.models.squares import fen_to_tokens
    from rukh.train import pick_device

    source = Path(positions)
    frame = pl.read_parquet(source)
    if column not in frame.columns:
        raise ValueError(f"{source} has no '{column}' column (found {', '.join(frame.columns)})")
    fens = [str(value) for value in frame[column].to_list()]
    if not fens:
        raise ValueError(f"{source} holds no positions")

    where = torch.device(device or pick_device())
    encoder = _encoder(Path(ckpt), str(where)).to(where).eval()
    if encoder.cfg.input != "squares":
        raise ValueError(
            f"the checkpoint reads the {encoder.cfg.input!r} scheme; embeddings of a FEN table "
            "need the 'squares' scheme, which is the one the labelled positions carry"
        )
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fens), max(1, batch_size)):
            chunk = fens[start : start + max(1, batch_size)]
            idx = torch.tensor([fen_to_tokens(fen) for fen in chunk], dtype=torch.long)
            pooled = encoder.pool(idx.to(where), POOLING)
            vectors.append(pooled.detach().float().cpu().numpy())
    array = np.concatenate(vectors) if vectors else np.zeros((0, encoder.cfg.d_model), "float32")

    target = Path(out)
    if target.suffix != ".npy":
        target = target / "embeddings.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array.astype("float32"))
    sidecar = target.with_suffix(SIDECAR_SUFFIX)
    pl.DataFrame(
        {
            "row": pl.Series("row", range(len(fens)), dtype=pl.UInt32),
            "fen4": [fen4(fen) for fen in fens],
        }
    ).write_parquet(sidecar)
    log.info("wrote %d embeddings of %d dimensions to %s", array.shape[0], array.shape[1], target)
    return EmbedResult(
        positions=int(array.shape[0]),
        dim=int(array.shape[1]),
        array=target.as_posix(),
        sidecar=sidecar.as_posix(),
        input=encoder.cfg.input,
        checkpoint=Path(ckpt).as_posix(),
    )
