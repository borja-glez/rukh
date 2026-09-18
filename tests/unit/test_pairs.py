"""Tests for rukh.data.pairs: mover-relative margins, mates, legality and phase balance."""

from __future__ import annotations

from pathlib import Path

import chess
import polars as pl
import pytest

from rukh.data.pairs import PairsConfig, balance, make_pair, run, score_mover

pytestmark = pytest.mark.unit

WHITE = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
BLACK = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"


def _pv(move: str, cp: int | None = None, mate: int | None = None) -> dict[str, object]:
    return {"move": move, "cp": cp, "mate": mate, "depth": 30}


def test_score_mover_sign_and_mate() -> None:
    assert score_mover(150, None, "w") == 150
    assert score_mover(150, None, "b") == -150
    assert score_mover(None, 2, "w") == 10_000
    assert score_mover(None, 2, "b") == -10_000
    assert score_mover(None, -1, "b") == 10_000
    assert score_mover(None, None, "w") is None


def test_make_pair_white_and_black_perspective() -> None:
    pair = make_pair(WHITE, [_pv("g1f3", 40), _pv("b1c3", 30), _pv("f2f4", -70)], 100)
    assert pair is not None
    assert (pair["chosen"], pair["rejected"]) == ("g1f3", "f2f4")
    assert (pair["cp_chosen"], pair["cp_rejected"]) == (40, -70)
    # For Black a lower cp is better: -20 beats 90 by 110.
    pair = make_pair(BLACK, [_pv("e7e5", -20), _pv("c7c5", -10), _pv("f7f5", 90)], 100)
    assert pair is not None
    assert (pair["chosen"], pair["rejected"]) == ("e7e5", "f7f5")
    assert (pair["cp_chosen"], pair["cp_rejected"]) == (-20, 90)


def test_make_pair_margin_and_mates() -> None:
    assert make_pair(WHITE, [_pv("g1f3", 40), _pv("b1c3", -50)], 100) is None
    pair = make_pair(WHITE, [_pv("g1f3", None, 2), _pv("b1c3", 300)], 100)
    assert pair is not None and pair["chosen"] == "g1f3" and pair["cp_chosen"] == 10_000
    pair = make_pair(BLACK, [_pv("e7e5", None, -3), _pv("c7c5", -400)], 100)
    assert pair is not None and pair["chosen"] == "e7e5" and pair["cp_chosen"] == -10_000
    assert pair["rejected"] == "c7c5"
    assert make_pair(WHITE, [_pv("g1f3", 40)], 100) is None


def test_make_pair_rejects_illegal_moves() -> None:
    assert make_pair(WHITE, [_pv("e1e5", 40), _pv("b1c3", -200)], 100) is None
    assert make_pair(WHITE, [_pv("g1f3", 40), _pv("e4e6", -200)], 100) is None
    assert make_pair("not a fen", [_pv("g1f3", 40), _pv("b1c3", -200)], 100) is None


def test_balance_uses_smallest_phase() -> None:
    frame = pl.DataFrame(
        {
            "fen": [f"f{i}" for i in range(9)],
            "chosen": ["a"] * 9,
            "rejected": ["b"] * 9,
            "cp_chosen": [0] * 9,
            "cp_rejected": [0] * 9,
            "phase": ["opening"] * 4 + ["middlegame"] * 3 + ["endgame"] * 2,
        }
    )
    out = balance(frame, max_per_phase=10, seed=1)
    assert out.group_by("phase").len()["len"].to_list() == [2, 2, 2]
    assert balance(frame, max_per_phase=1, seed=1).height == 3
    assert balance(frame.filter(pl.col("phase") != "endgame"), 10, 1).height == 0


def _evals_parquet(path: Path) -> None:
    board = chess.Board()
    fens, phases, pvs = [], [], []
    moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8"
    for ply, move in enumerate(moves.split(), start=1):
        board.push_uci(move)
        legal = [m.uci() for m in list(board.legal_moves)[:3]]
        sign = 1 if board.turn == chess.WHITE else -1
        fens.append(" ".join(board.fen().split()[:4]))
        phases.append("opening" if ply <= 10 else "middlegame" if ply <= 13 else "endgame")
        pvs.append([_pv(legal[0], sign * 60), _pv(legal[1], sign * 20), _pv(legal[2], -sign * 90)])
    pl.DataFrame({"fen": fens, "phase": phases, "pvs": pvs}).write_parquet(path.as_posix())


def test_run_writes_balanced_pairs(rukh_home: Path) -> None:
    evals = rukh_home / "data" / "evals" / "positions-eval.parquet"
    evals.parent.mkdir(parents=True)
    _evals_parquet(evals)
    manifest = run(PairsConfig(max_per_phase=2))
    frame = pl.read_parquet((rukh_home / "data" / "pairs" / "pairs.parquet").as_posix())
    assert frame.columns == ["fen", "chosen", "rejected", "cp_chosen", "cp_rejected", "phase"]
    assert manifest.counts["candidates"] == 16
    assert manifest.counts["opening"] == 2
    assert manifest.counts["middlegame"] == 2 and manifest.counts["endgame"] == 2
    assert frame.height == 6
    for fen, chosen, rejected, cp_c, cp_r, _ in frame.iter_rows():
        board = chess.Board(fen + " 0 1")
        legal = {m.uci() for m in board.legal_moves}
        assert chosen in legal and rejected in legal and chosen != rejected
        mover = 1 if board.turn == chess.WHITE else -1
        assert mover * (cp_c - cp_r) >= 100
