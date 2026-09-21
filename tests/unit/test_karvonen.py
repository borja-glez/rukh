"""Tests for rukh.eval.karvonen: the public baseline goes through the harness unchanged.

Nothing here downloads anything. What is worth testing is the part that could silently make the
row wrong: that the checkpoint format loads (the ``_orig_mod.`` prefix of ``torch.compile``),
that the prompt is the author's spelling to the character, that what the model writes is read
the same way Qwen's answers are, and that a failed proposal is rescued *and* counted.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import chess
import pytest
import torch
from torch import Tensor, nn

from rukh.eval import nightly
from rukh.eval.karvonen import (
    GPT,
    STOI,
    VOCAB,
    GPTConfig,
    KarvonenPlayer,
    karvonen_source,
    load_karvonen,
    prefix_board,
    read_meta,
    strip_compiled,
    transcript,
)
from rukh.eval.karvonen_suite import KarvonenResult, render_markdown, row_of
from rukh.eval.qwen_source import QwenStats
from rukh.eval.report import WebRow, render_benchmarks
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token

pytestmark = pytest.mark.unit

TOY = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=64, bias=False, vocab_size=32)


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_the_vocabulary_is_the_32_characters_of_meta_pkl() -> None:
    assert len(VOCAB) == 32 and len(STOI) == 32
    assert STOI[" "] == 0 and STOI[";"] == 15 and STOI["x"] == 31
    # Every SAN python-chess can write spells with it: captures, promotions, checks, castling.
    for san in ("exd5", "Nbd2", "O-O-O", "e8=Q+", "Qxf7#", "R1a3"):
        assert all(char in STOI for char in san), san


def test_a_compiled_checkpoint_loads_with_its_prefix_stripped(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original = GPT(TOY)
    compiled = {f"_orig_mod.{key}": value for key, value in original.state_dict().items()}
    assert all(key.startswith("_orig_mod.") for key in compiled)
    torch.save(
        {"model_args": {**asdict(TOY), "dropout": 0.0}, "model": compiled}, tmp_path / "ckpt.pt"
    )

    loaded = load_karvonen(tmp_path / "ckpt.pt")
    assert set(strip_compiled(compiled)) == set(loaded.state_dict())
    weight = loaded.transformer.h[1].mlp.c_fc.weight
    assert torch.equal(weight, original.transformer.h[1].mlp.c_fc.weight)
    # The head is tied to the embedding, as nanoGPT ties it: one tensor, not two copies.
    assert loaded.lm_head.weight.data_ptr() == loaded.transformer.wte.weight.data_ptr()
    assert not loaded.training
    idx = torch.tensor([[STOI[c] for c in ";1.e4"]])
    assert loaded(idx).shape == (1, 5, 32)


def test_the_meta_file_is_refused_unless_it_is_the_pinned_one(tmp_path: Path) -> None:
    meta = tmp_path / "meta.pkl"
    meta.write_bytes(pickle.dumps({"stoi": STOI, "itos": dict(enumerate(VOCAB))}))
    with pytest.raises(ValueError, match="pinned"):
        read_meta(meta)


@pytest.mark.parametrize(
    ("sans", "expected"),
    [
        ((), ";1."),
        (("e4",), ";1.e4"),
        (("e4", "e5"), ";1.e4 e5 2."),
        (("e4", "e5", "Nf3"), ";1.e4 e5 2.Nf3"),
        (("e4", "e5", "Nf3", "Nc6", "Bb5"), ";1.e4 e5 2.Nf3 Nc6 3.Bb5"),
    ],
    ids=["start", "black-to-move", "white-to-move", "three-plies", "five-plies"],
)
def test_the_transcript_is_the_authors_format_to_the_character(
    sans: tuple[str, ...], expected: str
) -> None:
    """``;`` first, no space after the move number, no trailing space when Black is to move."""
    assert transcript(_board(*sans)) == expected


def test_the_transcript_writes_san_in_the_position_it_belongs_to() -> None:
    """``Nbd2`` only reads as such where both knights reach d2; a replay from the root knows."""
    board = _board("Nf3", "d5", "d4", "Nf6", "Nbd2")
    assert transcript(board) == ";1.Nf3 d5 2.d4 Nf6 3.Nbd2"


class _Scripted(nn.Module):
    """A stand-in for the network that writes a fixed string, one character per call."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.cfg = TOY
        self.script = list(text)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx: Tensor) -> Tensor:
        logits = torch.full((1, idx.shape[1], 32), -10.0)
        char = self.script.pop(0) if self.script else " "
        logits[0, -1, STOI[char]] = 10.0
        return logits


def test_writing_stops_at_the_space_that_ends_the_move() -> None:
    assert KarvonenPlayer(_Scripted("Nf3 Nc6 3.")).write(";1.e4 e5 2.") == "Nf3"
    # Black's move starts with the space the model writes itself; the second space ends it.
    assert KarvonenPlayer(_Scripted(" e5 2.Nf3")).write(";1.e4") == " e5"
    # A new game delimiter means the model thinks the game is over: nothing was written.
    assert KarvonenPlayer(_Scripted(";1.e4")).write(";1.e4 e5 2.") == ""
    # A cap, so a model that never writes a space cannot run for ever.
    assert KarvonenPlayer(_Scripted("a" * 40), max_new_chars=5).write(";1.") == "aaaaa"


def test_a_legal_move_is_parsed_from_what_it_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    player = KarvonenPlayer(_Scripted(""))
    monkeypatch.setattr(player, "write", lambda prompt: " e5" if prompt == ";1.e4" else "??")
    move, was_legal = player.choose(_board("e4"))
    assert was_legal is True
    assert move is not None and move.uci() == "e7e5"
    assert player.stats.legal == 1 and player.stats.asked == 1


def test_an_illegal_proposal_falls_back_and_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same rule as the decoder and Qwen: the game goes on, the count does not hide it.

    The queen is boxed in at the start, so ``Qh5`` is well formed and illegal; the rescue is the
    first legal move in the position's own order, and the model is not asked again.
    """
    player = KarvonenPlayer(_Scripted(""))
    monkeypatch.setattr(player, "write", lambda prompt: "Qh5")
    move, was_legal = player.choose(chess.Board())
    assert was_legal is False
    assert move in chess.Board().legal_moves
    assert player.stats.illegal == 1 and player.stats.asked == 1

    monkeypatch.setattr(player, "write", lambda prompt: "Nf9")
    _, was_legal = player.choose(chess.Board())
    assert was_legal is False and player.stats.unparseable == 1

    mate = chess.Board("7k/5KQ1/8/8/8/8/8/8 b - - 0 1")
    assert player.choose(mate) == (None, False)


def test_the_player_carries_the_decoders_ply_limit_and_a_shared_stats_object() -> None:
    stats = QwenStats()
    player = KarvonenPlayer(_Scripted(""), stats=stats)
    assert player.limit == 196
    assert player.stats is stats
    player.start(1800, 1800)  # no header to set; must not fail either
    player.observe(chess.Move.from_uci("e2e4"))


def test_a_puzzle_prompt_is_rebuilt_from_the_games_own_moves() -> None:
    """The model cannot read a FEN, so the puzzle gets the game in front of it when there is one."""
    tok = UciTokenizer()
    header = [tok.bos_id, tok.vocab[elo_token(1800, "w")], tok.vocab[elo_token(1800, "b")]]
    history = header + [tok.vocab[uci] for uci in ("e2e4", "e7e5", "g1f3")]
    position = chess.Board(_board("e4", "e5", "Nf3").fen())  # a FEN board: no move stack
    assert not position.move_stack

    rebuilt = prefix_board(position, history, tok)
    assert transcript(rebuilt) == ";1.e4 e5 2.Nf3"
    # A prefix that does not lead to this position is not this game: the board is kept as is.
    elsewhere = chess.Board(_board("d4", "d5").fen())
    assert prefix_board(elsewhere, history, tok) is elsewhere

    source = karvonen_source(KarvonenPlayer(_Scripted(" Nc6")), tok)
    assert source(position, history) == chess.Move.from_uci("b8c6")
    assert source.blind == 0
    source(elsewhere, history)
    assert source.blind == 1


def _result(**overrides: object) -> KarvonenResult:
    base = {
        "stage": "karvonen-8l",
        "source": "adamkarvonen/chess_llms/lichess_8layers_ckpt_no_optimizer.pt",
        "checkpoint": "checkpoints/karvonen-8l/lichess_8layers_ckpt_no_optimizer.pt",
        "model_sha": "abc",
        "params": 25_714_688,
        "date": "2026-09-21",
        "device": "cpu",
        "positions": 4,
        "written": QwenStats(asked=4, legal=3, illegal=1),
        "top1": 0.5,
    }
    return KarvonenResult(**{**base, **overrides})  # type: ignore[arg-type]


def test_the_row_is_marked_as_a_baseline_and_older_rows_are_not() -> None:
    row = row_of(_result())
    assert row.baseline is True
    assert row.stage == "karvonen-8l" and row.legality == pytest.approx(0.75)
    # Rows written before the flag existed carry no key and read as the course's own models.
    assert WebRow.model_validate({"stage": "tiny", "params": 1, "date": "2026-09-19"}).baseline is (
        False
    )


def test_the_benchmarks_table_marks_the_external_row_and_still_renders() -> None:
    rows = [
        row_of(_result()).model_dump(),
        {"stage": "tiny-greedy", "legality": 0.945, "date": "2026-09-20"},
    ]
    table = render_benchmarks(rows)
    assert "| `karvonen-8l` (externo) | 75.0 %" in table
    assert "| `tiny-greedy` | 94.5 %" in table
    assert "2 etapas medidas con la misma suite, 1 de ellas externas" in table


def test_the_report_names_its_source_and_the_blind_puzzles() -> None:
    text = render_markdown(_result(notes=["a note"]))
    assert "adamkarvonen/chess_llms" in text
    assert "| illegal here | 1 | 25.00 % |" in text
    assert "- a note" in text


def test_the_nightly_plans_the_baseline_without_putting_it_in_the_catalogue(
    rukh_home: Path,
) -> None:
    from rukh.hub import catalogue

    planned = nightly.plan({"karvonen-8l"})
    assert [a.stage for a in planned] == ["karvonen-8l"]
    assert planned[0].measure == "karvonen" and planned[0].repo_id == "adamkarvonen/chess_llms"
    assert "karvonen-8l" not in {a.name for a in catalogue()}
    # Not on disk and not allowed to download: the stage fails on its own, not the run.
    with pytest.raises(FileNotFoundError, match="karvonen"):
        nightly._checkpoint_of(nightly.KARVONEN, pull_missing=False)
    report = nightly.run_nightly(only={"karvonen-8l"}, pull_missing=False, dry_run=True)
    assert [r.measure for r in report.records] == ["karvonen"]


def test_the_cli_exposes_the_baseline_with_the_same_flags_as_qwen() -> None:
    import typer.main

    from rukh.cli import app

    root = typer.main.get_command(app)
    commands = root.commands["eval"].commands  # type: ignore[attr-defined]
    options = {opt for param in commands["karvonen"].params for opt in param.opts}
    assert options >= {"--suite", "--config", "--stage", "--no-cache", "--device", "--checkpoint"}
    qwen = {opt for param in commands["qwen"].params for opt in param.opts}
    assert qwen - options == {"--adapter"}
