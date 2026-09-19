"""Tests for rukh.data.puzzles: filters, bands, the deterministic split and the game prefix."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rukh.data import puzzles as puzzles_module
from rukh.data.puzzles import (
    PREFIX_COLUMNS,
    PUZZLE_COLUMNS,
    PuzzlesConfig,
    band,
    rebuild_prefix,
    run,
    split_frame,
    split_key,
)

pytestmark = pytest.mark.unit

PUZZLE_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
"""After 1. e4 e5 2. Nf3 Nc6: the position the puzzle of the fixture starts from."""
PREFIX = "e2e4 e7e5 g1f3 b8c6"
MOVETEXT = (
    "1. e4 { [%eval 0.2] [%clk 0:10:00] } 1... e5 { [%eval 0.3] [%clk 0:10:00] } "
    "2. Nf3 { [%clk 0:09:58] } 2... Nc6 { [%eval 0.25] } "
    "3. Bb5?! { (0.25 -> -0.10) Inaccuracy. Nc3 was best. } 3... a6 "
    "4. Ba4 $1 Nf6 5. O-O Be7 1-0"
)
UNREACHABLE_FEN = "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"
"""A King's Indian position: it never occurs in ``MOVETEXT``, so the prefix cannot be rebuilt."""


def test_band_edges() -> None:
    assert band(999) is None
    assert band(1000) == "1000-1500"
    assert band(1499) == "1000-1500"
    assert band(1500) == "1500-2000"
    assert band(2000) == "2000+"
    assert band(3100) == "2000+"


def test_split_key_is_stable() -> None:
    assert split_key("abc12", 42) == split_key("abc12", 42)
    assert split_key("abc12", 42) != split_key("abc12", 43)
    assert split_key("abc12", 42) != split_key("abc13", 42)


def test_rebuild_prefix_replays_the_game_up_to_the_puzzle_position() -> None:
    prefix = rebuild_prefix(MOVETEXT, PUZZLE_FEN)
    assert prefix.reason == "ok"
    assert prefix.uci == PREFIX
    assert prefix.plies == 4


def test_rebuild_prefix_ignores_the_move_counters_of_the_puzzle_fen() -> None:
    """The four-field FEN is what decides the match, so a different clock still matches."""
    shifted = PUZZLE_FEN.replace(" 2 3", " 9 40")
    assert rebuild_prefix(MOVETEXT, shifted).uci == PREFIX


def test_rebuild_prefix_reports_why_it_failed() -> None:
    assert rebuild_prefix(MOVETEXT, UNREACHABLE_FEN).reason == "no_match"
    assert rebuild_prefix(None, PUZZLE_FEN).reason == "missing_game"
    assert rebuild_prefix("1. e4 e5 2. Qxf7#", PUZZLE_FEN).reason == "illegal"


def _remote(path: Path, fen: str = PUZZLE_FEN, movetext: str | None = MOVETEXT) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "PuzzleId": f"p{i:05d}",
                "GameId": f"g{i}",
                "FEN": fen,
                "Moves": "f1b5 a7a6",
                "Rating": 900 + i * 40,
                "RatingDeviation": 150 if i % 10 == 0 else 80,
                "Popularity": 90,
                "NbPlays": 50 if i % 7 == 0 else 500,
                "Themes": ["fork", "short"],
                "OpeningTags": ["Ruy_Lopez"],
                "movetext": movetext,
                "WhiteElo": 1912,
                "BlackElo": 1755,
                "Result": "1-0",
            }
        )
    pl.DataFrame(rows).write_parquet(path.as_posix())


def _local(rukh_home: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> Path:
    remote = rukh_home / "remote.parquet"
    _remote(remote, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(puzzles_module, "_source", lambda cfg: remote.as_posix())
    return rukh_home / "data" / "puzzles" / "puzzles.parquet"


def test_run_filters_bands_and_splits(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _local(rukh_home, monkeypatch)
    manifest = run(PuzzlesConfig(n_test=3, n_train=5, workers=1))
    frame = pl.read_parquet(out.as_posix())
    assert frame.columns == PUZZLE_COLUMNS + PREFIX_COLUMNS
    assert frame["rating"].min() >= 1000
    assert frame["puzzle_id"].n_unique() == frame.height
    for b in ("1000-1500", "1500-2000", "2000+"):
        test = frame.filter((pl.col("band") == b) & (pl.col("split") == "test"))
        train = frame.filter((pl.col("band") == b) & (pl.col("split") == "train"))
        assert test.height == 3 and 0 < train.height <= 5
        assert not set(test["puzzle_id"]) & set(train["puzzle_id"])
    assert manifest.counts["2000+/test"] == 3
    assert manifest.dataset == "Lichess/chess-puzzles-with-games"
    # Rows dropped by quality filters never appear.
    ids = set(frame["puzzle_id"].to_list())
    assert "p00000" not in ids and "p00007" not in ids and "p00010" not in ids


def test_the_real_game_prefix_is_written_next_to_the_puzzle(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _local(rukh_home, monkeypatch)
    manifest = run(PuzzlesConfig(n_test=3, n_train=5, workers=1))
    frame = pl.read_parquet(out.as_posix())
    assert frame["prefix_uci"].unique().to_list() == [PREFIX]
    assert frame["prefix_plies"].unique().to_list() == [4]
    assert frame["white_elo"].unique().to_list() == [1912]
    assert frame["black_elo"].unique().to_list() == [1755]
    assert manifest.counts["dropped_no_prefix"] == 0
    assert manifest.filters["with_games"] is True


def test_a_puzzle_whose_position_never_appears_is_dropped_and_counted(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _local(rukh_home, monkeypatch, fen=UNREACHABLE_FEN)
    manifest = run(PuzzlesConfig(n_test=3, n_train=5, workers=1))
    assert pl.read_parquet(out.as_posix()).height == 0
    selected = manifest.counts["selected"]
    assert selected > 0
    assert manifest.counts["dropped_no_prefix"] == selected
    prefix = manifest.filters["prefix"]
    assert isinstance(prefix, dict)
    assert prefix["dropped"] == {"no_match": selected}
    assert prefix["dropped_fraction"] == 1.0


def test_a_puzzle_without_its_game_is_dropped_and_counted(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _local(rukh_home, monkeypatch, movetext=None)
    manifest = run(PuzzlesConfig(n_test=3, n_train=5, workers=1))
    prefix = manifest.filters["prefix"]
    assert isinstance(prefix, dict)
    assert set(prefix["dropped"]) == {"missing_game"}


def test_the_puzzles_only_source_still_builds_a_parquet(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``with_games = false`` is the P1 path: same filters and split, no prefix columns."""
    out = _local(rukh_home, monkeypatch)
    manifest = run(PuzzlesConfig(n_test=3, n_train=5, with_games=False))
    frame = pl.read_parquet(out.as_posix())
    assert frame.columns == PUZZLE_COLUMNS
    assert manifest.dataset == "Lichess/chess-puzzles"
    assert manifest.filters["with_games"] is False
    assert manifest.counts["dropped_no_prefix"] == 0


def test_the_parquet_it_writes_is_what_the_eval_harness_reads(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer and the puzzle suite have to agree on the names of the prefix columns."""
    from rukh.eval.puzzles import GAME_PREFIX, load_puzzles

    out = _local(rukh_home, monkeypatch)
    run(PuzzlesConfig(n_test=3, n_train=5, workers=1))
    items = load_puzzles(out, per_band=1, seed=0)
    assert items
    assert all(item.prompt_style == GAME_PREFIX for item in items)
    assert items[0].prefix == PREFIX.split()
    assert (items[0].white_elo, items[0].black_elo) == (1912, 1755)


def test_split_is_deterministic_and_seed_sensitive(rukh_home: Path) -> None:
    remote = rukh_home / "remote.parquet"
    _remote(remote)
    frame = (
        pl.read_parquet(remote.as_posix())
        .rename(
            {
                "PuzzleId": "puzzle_id",
                "FEN": "fen",
                "Moves": "moves",
                "Rating": "rating",
                "Themes": "themes",
            }
        )
        .select("puzzle_id", "fen", "moves", "rating", "themes")
    )
    cfg = PuzzlesConfig(n_test=4, n_train=6, seed=42)
    a = split_frame(frame, cfg)
    b = split_frame(frame, cfg)
    assert a.equals(b)
    c = split_frame(frame, cfg.model_copy(update={"seed": 7}))
    assert set(a.filter(pl.col("split") == "test")["puzzle_id"]) != set(
        c.filter(pl.col("split") == "test")["puzzle_id"]
    )
