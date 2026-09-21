"""An exported ONNX graph standing in for the decoder, so an export can sit on the Elo ladder.

Parity (M2) says in how many positions the exported model picks the same move as the checkpoint.
It does not say what the other positions cost: an int8 export that agrees 95 % of the time is a
different player, and the only way to learn how different is to let it play the same ladder the
checkpoint played. ``OnnxDecoder`` is the smallest object that ``pick_move`` accepts in place of a
``MoveDecoder`` -- ``next_logits``, ``parameters`` (for the device) and ``cfg.block`` -- so the
export goes through ``DecoderPlayer`` and the very same masking, sampling and rescue as every
published number. A comparison whose two halves went through different code is not a comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rukh.export.onnx import INPUT_NAME

METADATA_BLOCK = "rukh_block"


def _session(onnx_path: Path) -> Any:
    """An ONNX Runtime session on the CPU provider (tests monkeypatch this)."""
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


class _Config:
    """The one field of ``DecoderConfig`` the sampler reads."""

    def __init__(self, block: int) -> None:
        self.block = block


class OnnxDecoder:
    """``model.onnx`` (or its fp16/int8 sibling) with the surface ``pick_move`` needs.

    ``block`` comes from the metadata ``rukh export`` writes; pass it when the file has none.
    """

    def __init__(self, onnx_path: Path, block: int | None = None) -> None:
        self.path = Path(onnx_path)
        self.session = _session(self.path)
        if block is None:
            meta = self.session.get_modelmeta().custom_metadata_map
            if METADATA_BLOCK not in meta:
                raise ValueError(f"{self.path.as_posix()} carries no {METADATA_BLOCK}; pass block=")
            block = int(meta[METADATA_BLOCK])
        self.cfg = _Config(block)
        self._anchor = torch.zeros(1)

    def parameters(self):  # type: ignore[no-untyped-def]
        """One CPU tensor: the sampler asks where the weights live, and ORT runs on the CPU."""
        yield self._anchor

    def next_logits(self, idx: Tensor) -> Tensor:
        """Logits of the last step only, ``(B, vocab_size)``, cropped to the block size."""
        cropped = idx[:, -self.cfg.block :].detach().cpu().numpy().astype(np.int64)
        logits = self.session.run(None, {INPUT_NAME: cropped})[0]
        return torch.from_numpy(np.asarray(logits, dtype=np.float32))

    def eval(self) -> OnnxDecoder:
        return self
