"""Tests for rukh.eval.cache: keying by weights, isolation and skipping finished work."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from rukh.eval.cache import EvalCache, config_sha, file_sha
from rukh.eval.puzzles import PuzzleItem, run_puzzles
from rukh.tokenize.uci_vocab import UciTokenizer

pytestmark = pytest.mark.unit

SCHOLAR = PuzzleItem(
    puzzle_id="p1",
    fen="r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
    moves=["g8f6", "h5f7"],
    rating=1100,
    band="1000-1500",
)


@pytest.fixture
def tok() -> UciTokenizer:
    return UciTokenizer()


class CountingSource:
    """A scripted move source that records how many times it was asked."""

    def __init__(self, moves: list[str]) -> None:
        self.moves = moves
        self.calls = 0

    def __call__(self, board: chess.Board, history: list[int]) -> chess.Move | None:
        move = self.moves[min(self.calls, len(self.moves) - 1)]
        self.calls += 1
        return chess.Move.from_uci(move)


def test_put_and_get_round_trip(tmp_path: Path) -> None:
    with EvalCache(tmp_path / "cache.sqlite", "sha-a") as cache:
        cache.put("elo", "uci-1500:0", {"score": 1.0, "plies": 42})
        assert cache.get("elo", "uci-1500:0") == {"score": 1.0, "plies": 42}
        assert cache.get("elo", "uci-1500:1") is None
        assert cache.count("elo") == 1


def test_different_weights_do_not_share_results(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    with EvalCache(path, "sha-a") as first:
        first.put("puzzles", "p1", {"solved": True})
    with EvalCache(path, "sha-b") as second:
        assert second.get("puzzles", "p1") is None
    with EvalCache(path, "sha-a") as again:
        assert again.get("puzzles", "p1") == {"solved": True}


def test_a_disabled_cache_never_reads_or_writes(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    with EvalCache(path, "sha-a") as warm:
        warm.put("puzzles", "p1", {"solved": True})
    with EvalCache(path, "sha-a", enabled=False) as cold:
        assert cold.get("puzzles", "p1") is None
        cold.put("puzzles", "p2", {"solved": False})
        assert cold.count() == 0
    with EvalCache(path, "sha-a") as check:
        assert check.get("puzzles", "p2") is None


def test_a_cache_without_a_path_is_a_no_op() -> None:
    with EvalCache(None, "sha-a") as cache:
        cache.put("puzzles", "p1", {"solved": True})
        assert cache.get("puzzles", "p1") is None


def test_the_cache_stops_a_puzzle_from_being_replayed(tmp_path: Path, tok: UciTokenizer) -> None:
    source = CountingSource(["h5f7"])
    with EvalCache(tmp_path / "cache.sqlite", "sha-a") as cache:
        first = run_puzzles(source, tok, [SCHOLAR], cache=cache)
        assert source.calls == 1
        second = run_puzzles(source, tok, [SCHOLAR], cache=cache)
    assert source.calls == 1
    assert second.model_dump() == first.model_dump()


def test_without_a_cache_the_puzzle_is_replayed(tok: UciTokenizer) -> None:
    source = CountingSource(["h5f7"])
    run_puzzles(source, tok, [SCHOLAR])
    run_puzzles(source, tok, [SCHOLAR])
    assert source.calls == 2


def test_different_settings_do_not_share_results(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    slow = config_sha({"temperature": 0.6, "elo_move_time": 0.1})
    fast = config_sha({"temperature": 0.6, "elo_move_time": 0.05})
    assert slow != fast
    with EvalCache(path, "sha-a", config_sha=slow) as first:
        first.put("elo", "uci-1500:0", {"score": 1.0})
    with EvalCache(path, "sha-a", config_sha=fast) as second:
        assert second.get("elo", "uci-1500:0") is None  # the move time changed the games
        assert second.count("elo") == 0
    with EvalCache(path, "sha-a", config_sha=slow) as again:
        assert again.get("elo", "uci-1500:0") == {"score": 1.0}


def test_one_key_per_family_so_more_games_reuse_the_ones_already_played() -> None:
    """The P4 backlog item: raising ``elo_games`` must not throw away the games on disk.

    A game is ``rung:index`` with seed ``seed + index``, the same game whatever the total is, and
    the same game whatever the puzzle settings are. So the games key ignores the count and the
    puzzle flags, and the puzzles key ignores the clock and the rungs.
    """
    from rukh.eval.suite import EvalConfig

    base = EvalConfig(elo_games=20)
    games_key = config_sha(base.cache_fields("games"))
    puzzles_key = config_sha(base.cache_fields("puzzles"))
    assert config_sha(EvalConfig(elo_games=40).cache_fields("games")) == games_key
    assert config_sha(EvalConfig(elo_games=40).cache_fields("puzzles")) == puzzles_key
    # What changes a game: the sampler, the clock (or the node budget), the plies, the rungs.
    assert config_sha(EvalConfig(temperature=0.05).cache_fields("games")) != games_key
    assert config_sha(EvalConfig(elo_move_time=0.05).cache_fields("games")) != games_key
    assert config_sha(EvalConfig(elo_nodes=200_000).cache_fields("games")) != games_key
    assert config_sha(EvalConfig(elo_max_plies=100).cache_fields("games")) != games_key
    # What does not: the puzzle header flag.
    assert config_sha(EvalConfig(force_header=True).cache_fields("games")) == games_key
    # And the mirror image for puzzles.
    assert config_sha(EvalConfig(elo_move_time=0.05).cache_fields("puzzles")) == puzzles_key
    assert config_sha(EvalConfig(elo_nodes=200_000).cache_fields("puzzles")) == puzzles_key
    assert config_sha(EvalConfig(force_header=True).cache_fields("puzzles")) != puzzles_key
    assert config_sha(EvalConfig(temperature=0.05).cache_fields("puzzles")) != puzzles_key
    with pytest.raises(ValueError, match="unknown cache family"):
        base.cache_fields("weather")


def test_the_node_budget_replaces_the_clock_in_the_opponent() -> None:
    """With ``nodes`` the opponent's limit is a node count; without it, the clock."""
    from rukh.infer.game import StockfishOpponent

    on_clock = StockfishOpponent.__new__(StockfishOpponent)
    on_clock.move_time, on_clock.nodes = 0.1, None
    assert on_clock.limit.time == 0.1 and on_clock.limit.nodes is None
    on_nodes = StockfishOpponent.__new__(StockfishOpponent)
    on_nodes.move_time, on_nodes.nodes = 0.1, 200_000
    assert on_nodes.limit.nodes == 200_000 and on_nodes.limit.time is None


def test_the_config_hash_ignores_key_order_and_reads_every_field() -> None:
    fields = {"temperature": 0.6, "top_k": 20, "seed": 42}
    assert config_sha(fields) == config_sha(dict(reversed(list(fields.items()))))
    assert config_sha(fields) != config_sha({**fields, "top_k": 10})
    assert config_sha(fields) != config_sha({**fields, "seed": 7})


def test_the_weights_sha_is_still_what_the_report_shows(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache.sqlite", "sha-a", config_sha="deadbeef")
    assert cache.model_sha == "sha-a"
    assert cache.key.startswith("sha-a:")
    cache.close()


def test_file_sha_identifies_the_weights(tmp_path: Path) -> None:
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"weights")
    second.write_bytes(b"weights")
    assert file_sha(first) == file_sha(second)
    second.write_bytes(b"other")
    assert file_sha(first) != file_sha(second)
