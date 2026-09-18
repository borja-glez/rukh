"""Tests for rukh.data.evals: file spec parsing, resumable parts and best-line consolidation."""

from __future__ import annotations

from pathlib import Path

import chess
import polars as pl
import pytest

from rukh.data import evals as evals_module
from rukh.data.evals import EvalsConfig, consolidate, fetch_part, parse_files, run

pytestmark = pytest.mark.unit


def test_parse_files() -> None:
    assert parse_files("0-19", 20) == list(range(20))
    assert parse_files("3", 20) == [3]
    assert parse_files("0,2,5-7", 20) == [0, 2, 5, 6, 7]
    with pytest.raises(ValueError):
        parse_files("20", 20)


def _fen(moves: str) -> str:
    board = chess.Board()
    for move in moves.split():
        board.push_uci(move)
    return " ".join(board.fen().split()[:4])


def _legal(fen: str, n: int) -> list[str]:
    board = chess.Board(fen + " 0 1")
    return [m.uci() for m in list(board.legal_moves)[:n]]


# Six positions: A, C, E White to move; B, D, F Black to move.
FEN_A = _fen("e2e4 e7e5")
FEN_B = _fen("e2e4")
FEN_C = _fen("e2e4 e7e5 g1f3 b8c6")
FEN_D = _fen("d2d4")
FEN_E = _fen("d2d4 d7d5")
FEN_F = _fen("c2c4")


def _lines() -> list[dict[str, object]]:
    a, b, c, d, e, f = (_legal(x, 3) for x in (FEN_A, FEN_B, FEN_C, FEN_D, FEN_E, FEN_F))
    rows = [
        # A (white): deepest lines first, then the highest cp -> a[1]
        (FEN_A, a[0], 30, 50, None),
        (FEN_A, a[1], 30, 120, None),
        (FEN_A, a[2], 20, 200, None),
        # B (black): lowest cp among the deepest -> b[1]
        (FEN_B, b[0], 30, 50, None),
        (FEN_B, b[1], 30, -80, None),
        (FEN_B, b[2], 20, -300, None),
        # C (white): mate for the mover beats a big cp -> c[0]
        (FEN_C, c[0], 30, None, 3),
        (FEN_C, c[1], 30, 500, None),
        # D (black): mate for the mover at lower depth still first -> d[0]
        (FEN_D, d[0], 25, None, -2),
        (FEN_D, d[1], 30, -900, None),
        # E (white): getting mated is worse than -700 -> e[1]
        (FEN_E, e[0], 30, None, -4),
        (FEN_E, e[1], 30, -700, None),
        # F: a single line
        (FEN_F, f[0], 18, 10, None),
    ]
    return [
        {"fen": fen, "line": f"{move} a7a6", "depth": depth, "knodes": 1000, "cp": cp, "mate": mate}
        for fen, move, depth, cp, mate in rows
    ]


def _positions_parquet(path: Path) -> None:
    fens = [FEN_A, FEN_B, FEN_C, FEN_D, FEN_E, FEN_F, _fen("g1f3")]
    pl.DataFrame(
        {
            "fen4": fens,
            "game_id": list(range(len(fens))),
            "ply": [2, 1, 4, 1, 2, 1, 1],
            "last_move": ["e7e5", "e2e4", "b8c6", "d2d4", "d7d5", "c2c4", "g1f3"],
            "result": ["1-0"] * len(fens),
            "phase": ["opening"] * len(fens),
            "n_seen": [5, 9, 2, 3, 1, 1, 4],
        }
    ).write_parquet(path.as_posix())


@pytest.fixture
def staged(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[EvalsConfig, Path]:
    positions = rukh_home / "data" / "positions" / "positions.parquet"
    positions.parent.mkdir(parents=True)
    _positions_parquet(positions)
    lines = _lines()
    remote = rukh_home / "remote"
    remote.mkdir()
    # Split the lines over two "remote" files plus a stranger FEN that must not survive.
    stranger = {"fen": _fen("a2a3"), "line": "a7a6", "depth": 9, "knodes": 1, "cp": 0, "mate": None}
    pl.DataFrame(lines[:7] + [stranger]).write_parquet((remote / "0.parquet").as_posix())
    pl.DataFrame(lines[7:]).write_parquet((remote / "1.parquet").as_posix())
    calls: list[int] = []

    def fake_source(cfg: EvalsConfig, index: int) -> str:
        calls.append(index)
        return (remote / f"{index}.parquet").as_posix()

    monkeypatch.setattr(evals_module, "_source", fake_source)
    monkeypatch.setattr(evals_module, "_calls", calls, raising=False)
    return EvalsConfig(n_files=2), rukh_home / "data" / "evals"


def test_fetch_part_is_resumable(staged: tuple[EvalsConfig, Path], rukh_home: Path) -> None:
    cfg, out_dir = staged
    out_dir.mkdir(parents=True)
    positions = rukh_home / "data" / "positions" / "positions.parquet"
    path, fetched = fetch_part(cfg, 0, positions, out_dir)
    assert fetched and path.name == "part-00.parquet"
    part = pl.read_parquet(path.as_posix())
    assert part.height == 7  # the stranger FEN is dropped by the semi join
    assert not list(out_dir.glob("*.tmp"))
    again, fetched = fetch_part(cfg, 0, positions, out_dir)
    assert not fetched and again == path
    assert evals_module._calls == [0]  # type: ignore[attr-defined]


def test_consolidate_picks_best_line_by_turn(staged: tuple[EvalsConfig, Path]) -> None:
    cfg, out_dir = staged
    manifest = run(cfg, files="0-1")
    assert manifest.counts == {"positions": 7, "evaluated": 6, "lines": 13}
    assert manifest.filters["files_present"] == [0, 1]
    assert manifest.filters["coverage"] == round(6 / 7, 4)
    frame = pl.read_parquet((out_dir / "positions-eval.parquet").as_posix())
    by_fen = {row["fen"]: row for row in frame.to_dicts()}
    a, b, c, d, e = (_legal(x, 3) for x in (FEN_A, FEN_B, FEN_C, FEN_D, FEN_E))
    assert by_fen[FEN_A]["best_move"] == a[1] and by_fen[FEN_A]["cp"] == 120
    assert by_fen[FEN_B]["best_move"] == b[1] and by_fen[FEN_B]["cp"] == -80
    assert by_fen[FEN_C]["best_move"] == c[0] and by_fen[FEN_C]["mate"] == 3
    assert by_fen[FEN_D]["best_move"] == d[0] and by_fen[FEN_D]["mate"] == -2
    assert by_fen[FEN_E]["best_move"] == e[1] and by_fen[FEN_E]["cp"] == -700
    assert len(by_fen[FEN_A]["pvs"]) == 3
    assert by_fen[FEN_A]["pvs"][0]["move"] == a[1]
    assert by_fen[FEN_A]["n_seen"] == 5 and by_fen[FEN_A]["phase"] == "opening"
    assert _fen("g1f3") not in by_fen


def test_run_skips_existing_parts(staged: tuple[EvalsConfig, Path]) -> None:
    cfg, out_dir = staged
    run(cfg, files="0")
    manifest = run(cfg, files="0-1")
    assert manifest.filters["files_fetched_now"] == [1]
    assert evals_module._calls == [0, 1]  # type: ignore[attr-defined]
    assert (
        consolidate(out_dir, out_dir.parent / "positions" / "positions.parquet")["evaluated"] == 6
    )
