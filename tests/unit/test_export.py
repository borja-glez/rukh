"""Tests for rukh.export: the last-step wrapper, the ONNX file, quantization and parity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from rukh.export import (
    INPUT_NAME,
    MODEL_NAME,
    LastStepLogits,
    export_onnx,
    parity,
    parity_positions,
    quantize_int8,
    random_prefixes,
    read_metadata,
    sequence_lengths,
    target_path,
    to_fp16,
    validation_prefixes,
)
from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

# ``rukh.export`` re-exports ``parity`` under the module's own name, so the module object for
# monkeypatching has to be asked for explicitly.
parity_module = importlib.import_module("rukh.export.parity")
onnx_module = importlib.import_module("rukh.export.onnx")

TOY = DecoderConfig(vocab_size=2030, n_layer=2, n_head=2, d_model=32, block=64)
SEQ_LEN = 16
GAME = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"
HAS_ONNX = all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime"))
requires_onnx = pytest.mark.skipif(not HAS_ONNX, reason="onnx and onnxruntime are not installed")


@pytest.fixture(scope="module")
def tok() -> UciTokenizer:
    return UciTokenizer()


@pytest.fixture(scope="module")
def model() -> MoveDecoder:
    torch.manual_seed(0)
    return MoveDecoder(TOY).eval()


@pytest.fixture(scope="module")
def prefixes(tok: UciTokenizer) -> list[list[int]]:
    return [ids[:SEQ_LEN] for ids in random_prefixes(tok, 100, seed=0, min_ply=4, max_ply=20)]


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``: whatever the callable returns."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer

    def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.answer(feed[INPUT_NAME])]


def faithful(model: MoveDecoder):  # type: ignore[no-untyped-def]
    wrapper = LastStepLogits(model).eval()

    def answer(idx: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return wrapper(torch.from_numpy(idx)).numpy()

    return answer


def test_the_wrapper_returns_only_the_last_step(model: MoveDecoder) -> None:
    idx = torch.zeros((3, SEQ_LEN), dtype=torch.long)
    with torch.no_grad():
        out = LastStepLogits(model)(idx)
        reference = model.next_logits(idx)
    assert out.shape == (3, TOY.vocab_size)
    assert torch.allclose(out, reference)


def test_target_path_accepts_a_directory_or_a_file(tmp_path: Path) -> None:
    assert target_path(tmp_path) == tmp_path / MODEL_NAME
    assert target_path(tmp_path / "small.onnx") == tmp_path / "small.onnx"


def test_exporting_longer_than_the_block_is_an_error(model: MoveDecoder, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="longer than the model's block"):
        export_onnx(model, tmp_path, seq_len=TOY.block + 1)


def test_random_prefixes_are_legal_games_and_reproducible(tok: UciTokenizer) -> None:
    first = random_prefixes(tok, 5, seed=7, min_ply=2, max_ply=6)
    assert first == random_prefixes(tok, 5, seed=7, min_ply=2, max_ply=6)
    for ids in first:
        assert ids[0] == tok.bos_id
        board = chess.Board()
        for token in tok.decode(ids[3:]):
            move = chess.Move.from_uci(token)
            assert board.is_legal(move)
            board.push(move)


def test_parity_is_perfect_when_the_session_agrees(
    model: MoveDecoder, prefixes: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_module, "_session", lambda _path: FakeSession(faithful(model)))
    result = parity(model, Path("unused.onnx"), prefixes)
    assert result.positions == len(prefixes)
    assert result.agreement == 1.0
    assert result.max_abs_logit_delta == pytest.approx(0.0)
    assert result.mismatches == []


def test_parity_detects_an_injected_discrepancy(
    model: MoveDecoder, prefixes: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = faithful(model)

    def broken(idx: np.ndarray) -> np.ndarray:
        logits = answer(idx).copy()
        logits[0, 7] += 1_000.0  # token 7 now wins every position
        return logits

    monkeypatch.setattr(parity_module, "_session", lambda _path: FakeSession(broken))
    result = parity(model, Path("unused.onnx"), prefixes)
    assert result.agreement < 1.0
    assert result.max_abs_logit_delta == pytest.approx(1_000.0)
    assert result.mismatches


def test_parity_honours_n(
    model: MoveDecoder, prefixes: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_module, "_session", lambda _path: FakeSession(faithful(model)))
    assert parity(model, Path("unused.onnx"), prefixes, n=10).positions == 10


def test_parity_without_positions_is_an_error(
    model: MoveDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_module, "_session", lambda _path: FakeSession(faithful(model)))
    with pytest.raises(ValueError, match="at least one position"):
        parity(model, Path("unused.onnx"), [])


@pytest.fixture
def games(tmp_path: Path) -> Path:
    import polars as pl

    path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "uci": [GAME, GAME, GAME],
            "white_elo": [1850, 2150, 1900],
            "black_elo": [1950, 2050, 1900],
        }
    ).write_parquet(path)
    return path


def test_parity_positions_come_from_the_validation_games(tok: UciTokenizer, games: Path) -> None:
    prefixes, source, warning = parity_positions(tok, n=3, block=TOY.block, seed=0, games=games)
    assert source == "validation" and warning is None
    assert prefixes == validation_prefixes(games, tok, n=3, block=TOY.block, seed=0)
    for ids in prefixes:
        assert ids[0] == tok.bos_id
        assert ids[1] in (tok.vocab["<w1800>"], tok.vocab["<w2100>"], tok.vocab["<w1900>"])
        # every prefix is a real continuation of the validation game
        assert [tok.ids[i] for i in ids[3:]] == GAME.split()[: len(ids) - 3]


def test_parity_falls_back_to_random_walks_with_a_warning(
    tok: UciTokenizer, tmp_path: Path
) -> None:
    prefixes, source, warning = parity_positions(
        tok, n=5, block=TOY.block, seed=0, games=tmp_path / "absent.parquet"
    )
    assert source == "random-walk"
    assert warning is not None and "random legal walks" in warning
    assert prefixes == random_prefixes(tok, 5, seed=0)


def test_the_verification_runs_the_file_at_two_different_lengths() -> None:
    assert sequence_lengths(16, 64) == [16, 17]
    assert sequence_lengths(64, 64) == [63, 64]


def _minimal_onnx_bytes() -> bytes:
    """Serialized identity model, small enough to stand in for a real export."""
    try:
        from onnx import TensorProto, helper
    except ModuleNotFoundError:  # pragma: no cover - the fixture is only used with onnx around
        return b"onnx"
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, ["batch", "seq"])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2030])
    node = helper.make_node("Cast", ["idx"], ["logits"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "fake", [idx], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return model.SerializeToString()


_MINIMAL_ONNX = _minimal_onnx_bytes()


@pytest.fixture
def fake_export(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Stand in for ``torch.onnx.export``: it needs ``onnx``, and these tests do not.

    What is under test here is what ``export_onnx`` does with the answer of the verification,
    not the exporter itself; the real file is exercised by the ``requires_onnx`` tests.
    """

    def export(_wrapper: Any, _args: Any, path: str, **_kwargs: Any) -> None:
        # A parseable, minimal model: `export_onnx` writes metadata into whatever the exporter
        # produced, so a placeholder of raw bytes would fail inside `onnx.load` once `onnx` is
        # installed (which it now is) and hide what the test is actually about.
        Path(path).write_bytes(_MINIMAL_ONNX)

    monkeypatch.setattr(torch.onnx, "export", export)


def test_a_baked_in_length_is_an_error_when_dynamic_was_asked_for(
    model: MoveDecoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_export: None
) -> None:
    monkeypatch.setattr(onnx_module, "verify_dynamic_seq", lambda *_args: False)
    with pytest.raises(ValueError, match="baked the length in"):
        export_onnx(model, tmp_path, seq_len=SEQ_LEN, dynamic_seq=True)


def test_a_fixed_export_reports_what_the_file_really_does(
    model: MoveDecoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_export: None
) -> None:
    monkeypatch.setattr(onnx_module, "verify_dynamic_seq", lambda *_args: False)
    result = export_onnx(model, tmp_path, seq_len=SEQ_LEN, dynamic_seq=False)
    assert result.dynamic_seq is False and result.dynamic_seq_verified is True
    assert result.block == TOY.block


def test_an_unverifiable_export_says_so(
    model: MoveDecoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_export: None
) -> None:
    monkeypatch.setattr(onnx_module, "verify_dynamic_seq", lambda *_args: None)
    result = export_onnx(model, tmp_path, seq_len=SEQ_LEN, dynamic_seq=True)
    assert result.dynamic_seq is True and result.dynamic_seq_verified is False
    assert result.metadata == {} or result.metadata["rukh_block"] == str(TOY.block)


def test_a_verified_dynamic_axis_is_recorded_as_such(
    model: MoveDecoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_export: None
) -> None:
    monkeypatch.setattr(onnx_module, "verify_dynamic_seq", lambda *_args: True)
    result = export_onnx(model, tmp_path, seq_len=SEQ_LEN, dynamic_seq=True)
    assert (result.dynamic_seq, result.dynamic_seq_verified) == (True, True)
    assert result.block == TOY.block and result.params == model.num_params(non_embedding=False)


@pytest.fixture
def exported(model: MoveDecoder, tmp_path: Path):  # type: ignore[no-untyped-def]
    return export_onnx(model, tmp_path, seq_len=SEQ_LEN)


@requires_onnx
def test_the_exported_file_runs_and_matches_pytorch(
    model: MoveDecoder, prefixes: list[list[int]], exported: Any
) -> None:
    # The modern exporter must actually be the one that ran. It used to lose to a
    # UnicodeEncodeError on Windows (it prints check marks and the console is cp1252), and the
    # silent fallback produced an fp16 graph that onnxruntime refuses to load.
    assert exported.exporter == "dynamo", exported.warning
    assert exported.warning is None
    assert Path(exported.path).is_file()
    result = parity(model, Path(exported.path), prefixes)
    assert result.agreement == 1.0
    assert result.max_abs_logit_delta < 1e-3


@requires_onnx
def test_the_batch_axis_is_dynamic(exported: Any, model: MoveDecoder) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(exported.path, providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        idx = np.zeros((batch, SEQ_LEN), dtype=np.int64)
        logits = session.run(None, {INPUT_NAME: idx})[0]
        assert logits.shape == (batch, TOY.vocab_size)
        with torch.no_grad():
            reference = LastStepLogits(model)(torch.from_numpy(idx)).numpy()
        assert np.allclose(logits, reference, atol=1e-3)


@requires_onnx
def test_the_sequence_axis_is_dynamic(exported: Any) -> None:
    import onnxruntime as ort

    assert exported.dynamic_seq is True
    assert exported.dynamic_seq_verified is True  # it was run before being claimed
    session = ort.InferenceSession(exported.path, providers=["CPUExecutionProvider"])
    for length in (SEQ_LEN, SEQ_LEN + 5):
        idx = np.zeros((1, length), dtype=np.int64)
        assert session.run(None, {INPUT_NAME: idx})[0].shape == (1, TOY.vocab_size)


@requires_onnx
def test_the_file_carries_the_context_length_in_its_metadata(exported: Any) -> None:
    metadata = read_metadata(Path(exported.path))
    assert metadata["rukh_block"] == str(TOY.block)
    assert metadata["rukh_vocab_size"] == str(TOY.vocab_size)
    assert metadata["rukh_dynamic_seq"] == "True"
    assert metadata["rukh_exporter"] == exported.exporter
    assert exported.metadata == metadata


@requires_onnx
def test_the_exported_file_is_self_contained(exported: Any) -> None:
    """No `<name>.onnx.data` sidecar: the browser and the Hub get one file, not two.

    The dynamo exporter writes the weights beside the graph and `set_metadata` then saves them
    back inline, so the sidecar it leaves behind is dead weight that used to ship with every
    export.
    """
    path = Path(exported.path)
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".data").exists()
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


@requires_onnx
def test_the_int8_file_is_smaller_than_the_fp32_one(exported: Any) -> None:
    quantized = quantize_int8(Path(exported.path))
    assert Path(quantized.path).is_file()
    assert quantized.bytes < exported.bytes


@requires_onnx
def test_the_fp16_file_is_smaller_than_the_fp32_one(exported: Any) -> None:
    half = to_fp16(Path(exported.path))
    assert Path(half.path).is_file()
    assert half.bytes < exported.bytes


@requires_onnx
def test_the_quantized_files_still_agree_with_pytorch(
    model: MoveDecoder, prefixes: list[list[int]], exported: Any
) -> None:
    half = to_fp16(Path(exported.path))
    result = parity(model, Path(half.path), prefixes[:20])
    assert result.agreement >= 0.9


@requires_onnx
def test_the_export_writes_the_parity_it_measured_next_to_the_files(tmp_path: Path) -> None:
    """The parity is evidence, so it is written down and not only printed.

    Without this file the publisher cannot quote an agreement without repeating a claim from a
    console log, which is exactly how a number stops being a measurement.
    """
    from rukh.export import PARITY_NAME, PARITY_VERSION, export_all
    from rukh.train import save_checkpoint

    torch.manual_seed(0)
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(
        ckpt,
        step=1,
        model=MoveDecoder(TOY),
        optimizer=None,
        cfg={},
        model_cfg=TOY.model_dump(),
        vocab_hash="vocab-sha",
        git_sha="git-sha",
    )
    bundle = export_all(
        ckpt,
        tmp_path / "onnx",
        seq_len=SEQ_LEN,
        fp16=True,
        int8=True,
        check_parity=True,
        positions=8,
        games=tmp_path / "no-such-games.parquet",
    )
    written = tmp_path / "onnx" / PARITY_NAME
    assert bundle.parity_path == written.as_posix()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["version"] == PARITY_VERSION
    assert payload["kind"] == "decoder"
    assert payload["measures"] == "the argmax move"
    assert payload["positions"] == 8
    assert payload["source"] == "random-walk"  # there is no validation parquet under tmp_path
    assert payload["warning"] and "random legal walks" in payload["warning"]
    assert payload["exporter"] == bundle.onnx.exporter
    assert set(payload["precisions"]) == {"fp32", "fp16", "int8"}
    assert payload["precisions"]["int8"]["file"] == "model-int8.onnx"
    for name, entry in payload["precisions"].items():
        assert entry["agreement"] == bundle.parity[name].agreement
        assert entry["max_abs_logit_delta"] == bundle.parity[name].max_abs_logit_delta


@requires_onnx
def test_an_export_that_checked_nothing_writes_no_parity_file(tmp_path: Path) -> None:
    """No file means "not checked", which a card must never read as "checked and perfect"."""
    from rukh.export import PARITY_NAME, export_all
    from rukh.train import save_checkpoint

    torch.manual_seed(0)
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(
        ckpt, step=1, model=MoveDecoder(TOY), optimizer=None, cfg={}, model_cfg=TOY.model_dump()
    )
    bundle = export_all(ckpt, tmp_path / "onnx", seq_len=SEQ_LEN, check_parity=False)
    assert bundle.parity_path is None
    assert not (tmp_path / "onnx" / PARITY_NAME).exists()
