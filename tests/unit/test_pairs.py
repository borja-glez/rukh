"""Tests for rukh.data.pairs: mover-relative margins, mates, legality and phase balance."""

from __future__ import annotations

from pathlib import Path

import chess
import polars as pl
import pytest

from rukh.data import evals as evals_module
from rukh.data.evals import EvalsConfig
from rukh.data.pairs import PairsConfig, balance, build_pairs, make_pair, run, score_mover

pytestmark = pytest.mark.unit

WHITE = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
BLACK = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"


def _pv(
    move: str, cp: int | None = None, mate: int | None = None, depth: int = 30
) -> dict[str, object]:
    return {"move": move, "cp": cp, "mate": mate, "depth": depth}


def test_score_mover_sign_and_mate() -> None:
    assert score_mover(150, None, "w") == 150
    assert score_mover(150, None, "b") == -150
    assert score_mover(None, 2, "w") == 9_998
    assert score_mover(None, 2, "b") == -9_998
    assert score_mover(None, -1, "b") == 9_999
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
    assert pair is not None and pair["chosen"] == "g1f3" and pair["cp_chosen"] == 9_998
    pair = make_pair(BLACK, [_pv("e7e5", None, -3), _pv("c7c5", -400)], 100)
    assert pair is not None and pair["chosen"] == "e7e5" and pair["cp_chosen"] == -9_997
    assert pair["rejected"] == "c7c5"
    # A mate in 1 outranks a mate in 5, but the margin between them is only 4 centipawns.
    assert make_pair(WHITE, [_pv("g1f3", None, 5), _pv("b1c3", None, 1)], 100) is None
    # Depth outranks the score: the deepest line is chosen, a shallow better one is no pair.
    assert make_pair(WHITE, [_pv("g1f3", 900, depth=12), _pv("b1c3", 40, depth=30)], 100) is None
    pair = make_pair(WHITE, [_pv("g1f3", 40, depth=30), _pv("b1c3", -400, depth=12)], 100)
    assert pair is not None and pair["chosen"] == "g1f3" and pair["rejected"] == "b1c3"
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


GAME = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8"


def _evals_parquet(path: Path) -> None:
    board = chess.Board()
    fens, phases, pvs, game_ids, plies = [], [], [], [], []
    for ply, move in enumerate(GAME.split(), start=1):
        board.push_uci(move)
        legal = [m.uci() for m in list(board.legal_moves)[:3]]
        sign = 1 if board.turn == chess.WHITE else -1
        fens.append(" ".join(board.fen().split()[:4]))
        phases.append("opening" if ply <= 10 else "middlegame" if ply <= 13 else "endgame")
        pvs.append([_pv(legal[0], sign * 60), _pv(legal[1], sign * 20), _pv(legal[2], -sign * 90)])
        game_ids.append("1")
        plies.append(ply)
    pl.DataFrame(
        {"fen": fens, "phase": phases, "pvs": pvs, "game_id": game_ids, "ply": plies}
    ).write_parquet(path.as_posix())


def _games_parquet(games_dir: Path, game_id: int = 1, uci: str = GAME) -> None:
    """One converted game, in the month layout and with the integer id `rukh data uci` writes."""
    path = games_dir / "year=2025" / "month=01" / "games.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "game_id": pl.Series([game_id], dtype=pl.Int64),
            "uci": [uci],
            "n_plies": pl.Series([len(uci.split())], dtype=pl.Int16),
            "white_elo": pl.Series([2015], dtype=pl.Int16),
            "black_elo": pl.Series([2028], dtype=pl.Int16),
        }
    ).write_parquet(path.as_posix())


def test_run_writes_balanced_pairs(rukh_home: Path) -> None:
    evals = rukh_home / "data" / "evals" / "positions-eval.parquet"
    evals.parent.mkdir(parents=True)
    _evals_parquet(evals)
    _games_parquet(rukh_home / "data" / "uci")
    manifest = run(PairsConfig(max_per_phase=2))
    frame = pl.read_parquet((rukh_home / "data" / "pairs" / "pairs.parquet").as_posix())
    assert frame.columns == [
        "fen",
        "chosen",
        "rejected",
        "cp_chosen",
        "cp_rejected",
        "phase",
        "game_id",
        "ply",
    ]
    assert manifest.counts["candidates"] == 16
    assert manifest.counts["opening"] == 2
    assert manifest.counts["middlegame"] == 2 and manifest.counts["endgame"] == 2
    assert frame.height == 6
    for fen, chosen, rejected, cp_c, cp_r, _phase, _game, _ply in frame.iter_rows():
        board = chess.Board(fen + " 0 1")
        legal = {m.uci() for m in board.legal_moves}
        assert chosen in legal and rejected in legal and chosen != rejected
        mover = 1 if board.turn == chess.WHITE else -1
        assert mover * (cp_c - cp_r) >= 100


def test_run_writes_the_prompts_the_decoder_reads(rukh_home: Path) -> None:
    """``dpo-prompts.parquet``: each pair joined to its game, prefix and ratings included.

    The file every DPO, GRPO, reward-model and on-policy config points at. It used to come from
    a join nobody could rerun; now it is what ``rukh data pairs`` writes next to the pairs.
    """
    evals = rukh_home / "data" / "evals" / "positions-eval.parquet"
    evals.parent.mkdir(parents=True)
    _evals_parquet(evals)
    _games_parquet(rukh_home / "data" / "uci")
    manifest = run(PairsConfig(max_per_phase=2))
    prompts = pl.read_parquet((rukh_home / "data" / "pairs" / "dpo-prompts.parquet").as_posix())
    assert prompts.columns == [
        "game_id",
        "ply",
        "chosen",
        "rejected",
        "cp_chosen",
        "cp_rejected",
        "phase",
        "white_elo",
        "black_elo",
        "prefix",
    ]
    assert prompts.height == 6 == manifest.counts["prompts"]
    assert manifest.counts["prompts_dropped"] == 0
    assert [f.path for f in manifest.files] == ["pairs.parquet", "dpo-prompts.parquet"]
    moves = GAME.split()
    for row in prompts.iter_rows(named=True):
        assert row["prefix"] == " ".join(moves[: row["ply"]])
        assert (row["white_elo"], row["black_elo"]) == (2015, 2028)


def test_a_prompt_deeper_than_the_context_is_dropped_and_counted(rukh_home: Path) -> None:
    """Three header tokens plus ``ply`` moves must fit ``block``; the rest are dropped, not cut."""
    from rukh.data.pairs import build_prompts

    _games_parquet(rukh_home / "data" / "uci")
    pairs = pl.DataFrame(
        {
            "fen": ["-", "-"],
            "chosen": ["a", "a"],
            "rejected": ["b", "b"],
            "cp_chosen": pl.Series([1, 1], dtype=pl.Int32),
            "cp_rejected": pl.Series([0, 0], dtype=pl.Int32),
            "phase": ["opening", "endgame"],
            "game_id": ["1", "1"],
            "ply": pl.Series([5, 14], dtype=pl.Int32),
        }
    )
    kept = build_prompts(pairs, rukh_home / "data" / "uci", block=16)
    assert kept["ply"].to_list() == [5]
    assert kept["prefix"][0] == " ".join(GAME.split()[:5])
    missing_game = pairs.with_columns(pl.lit("2").alias("game_id"))
    assert build_prompts(missing_game, rukh_home / "data" / "uci").height == 0


def test_chosen_is_the_consolidated_best_move(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pairs`` and the ``evals`` consolidation must agree, mates included."""
    board = chess.Board()
    board.push_uci("f2f3")
    board.push_uci("e7e5")
    board.push_uci("g2g4")  # 2...Qh4# is mate in 1 for Black
    mate_fen = " ".join(board.fen().split()[:4])
    quiet = " ".join(chess.Board().fen().split()[:4])
    quiet_moves = [m.uci() for m in list(chess.Board().legal_moves)[:3]]
    rows = [
        # Black to move: the mate in 1 must win over the deeper, better-looking cp line.
        (mate_fen, "d8h4", 12, None, -1),
        (mate_fen, "d7d5", 40, -300, None),
        (mate_fen, "b8c6", 40, 200, None),
        # White to move: deepest line first, then the best cp.
        (quiet, quiet_moves[0], 40, 30, None),
        (quiet, quiet_moves[1], 40, -200, None),
        (quiet, quiet_moves[2], 12, 900, None),
    ]
    remote = rukh_home / "remote.parquet"
    pl.DataFrame(
        [
            {
                "fen": fen,
                "line": f"{move} a7a6",
                "depth": depth,
                "knodes": 10,
                "cp": cp,
                "mate": mate,
            }
            for fen, move, depth, cp, mate in rows
        ]
    ).write_parquet(remote.as_posix())
    positions = rukh_home / "data" / "positions" / "positions.parquet"
    positions.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "fen4": [mate_fen, quiet],
            "game_id": [1, 2],
            "ply": [3, 0],
            "last_move": ["g2g4", "e2e4"],
            "result": ["0-1", "1-0"],
            "phase": ["opening", "opening"],
            "n_seen": [1, 1],
        }
    ).write_parquet(positions.as_posix())
    monkeypatch.setattr(evals_module, "_source", lambda cfg, index: remote.as_posix())
    evals_module.run(EvalsConfig(n_files=1), files="0")

    consolidated = pl.read_parquet(
        (rukh_home / "data" / "evals" / "positions-eval.parquet").as_posix()
    )
    assert consolidated.height == 2
    for row in consolidated.to_dicts():
        pair = make_pair(row["fen"], row["pvs"], 100)
        assert pair is not None, row["fen"]
        assert pair["chosen"] == row["best_move"]
    best_by_fen = {r["fen"]: r["best_move"] for r in consolidated.to_dicts()}
    assert best_by_fen[mate_fen] == "d8h4"


def test_pairs_carry_the_game_they_came_from(tmp_path: Path) -> None:
    """Every pair keeps ``game_id`` and ``ply``.

    The decoder reads a move sequence, not a board, so a pair identified only by its FEN cannot
    be turned into a prompt for it. These two columns are what join a pair back to its game and
    recover the prefix that led to the position (D-070).
    """
    evals = tmp_path / "positions-eval.parquet"
    _evals_parquet(evals)
    frame = build_pairs(evals, min_delta_cp=50)
    assert frame.height > 0
    assert {"game_id", "ply"} <= set(frame.columns)
    assert frame["game_id"].to_list() == ["1"] * frame.height
    assert all(p >= 1 for p in frame["ply"].to_list())
