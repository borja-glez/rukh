"""An ONNX export plays through the same player as the checkpoint it came from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from rukh.export.onnx import INPUT_NAME, LastStepLogits
from rukh.infer import DecoderPlayer, SampleConfig
from rukh.infer import onnx_player as module
from rukh.infer.onnx_player import OnnxDecoder
from rukh.models.decoder import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession`` with the checkpoint's own logits."""

    def __init__(self, model: MoveDecoder, block: int | None = TOY.block) -> None:
        self.wrapper = LastStepLogits(model).eval()
        self.block = block
        self.seen: list[int] = []

    def get_modelmeta(self) -> Any:
        meta = {} if self.block is None else {module.METADATA_BLOCK: str(self.block)}
        return type("Meta", (), {"custom_metadata_map": meta})()

    def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        idx = feed[INPUT_NAME]
        assert idx.dtype == np.int64
        self.seen.append(idx.shape[1])
        with torch.no_grad():
            return [self.wrapper(torch.from_numpy(idx)).numpy()]


@pytest.fixture
def model() -> MoveDecoder:
    torch.manual_seed(0)
    return MoveDecoder(TOY).eval()


def test_the_export_picks_the_checkpoints_move_greedily(
    model: MoveDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = FakeSession(model)
    monkeypatch.setattr(module, "_session", lambda _path: session)
    tok = UciTokenizer()
    cfg = SampleConfig(temperature=0, top_k=None, mask_illegal=False, seed=0)

    reference = DecoderPlayer(model, tok, cfg)
    exported = DecoderPlayer(OnnxDecoder(Path("model.onnx")), tok, cfg)  # type: ignore[arg-type]
    assert exported.limit == reference.limit

    board = chess.Board()
    for player in (reference, exported):
        player.start(1800, 1800)
    for _ in range(6):
        a, legal_a = reference.choose(board)
        b, legal_b = exported.choose(board)
        assert a == b and legal_a == legal_b
        if a is None:
            break
        board.push(a)
        reference.observe(a)
        exported.observe(a)
    assert session.seen and max(session.seen) <= TOY.block


def test_the_block_comes_from_the_metadata_or_the_caller(
    model: MoveDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_session", lambda _path: FakeSession(model, block=None))
    with pytest.raises(ValueError, match="rukh_block"):
        OnnxDecoder(Path("model.onnx"))
    assert OnnxDecoder(Path("model.onnx"), block=32).cfg.block == 32
    monkeypatch.setattr(module, "_session", lambda _path: FakeSession(model, block=48))
    assert OnnxDecoder(Path("model.onnx")).cfg.block == 48
