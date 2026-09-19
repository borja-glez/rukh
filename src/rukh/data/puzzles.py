"""Puzzle splits by difficulty band, with the game each puzzle came from (deterministic by id).

``Lichess/chess-puzzles-with-games`` carries the puzzle columns **and** the ``movetext`` of the
game the position was taken from, so the moves that lead to the puzzle can be replayed: the game
is converted to UCI with the helpers of ``rukh.data.uci`` and stopped at the first ply whose
normalized four-field FEN equals the puzzle's. Those moves become ``prefix_uci``, the prompt a
move-sequence model can actually be given. Without them the only prompt available is the puzzle
line itself, which is not a game and does not start from the initial position.

``with_games = false`` falls back to the puzzle-only ``Lichess/chess-puzzles`` of P1: same
filters, same bands, same split, no prefix. Puzzles whose prefix cannot be rebuilt (the position
never appears, the movetext is unplayable, the game is missing) are dropped and counted in the
manifest.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import polars as pl
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest
from rukh.data.parallel import run_batches
from rukh.data.uci import resolve_workers, san_tokens, sha256_file
from rukh.paths import resolve

PUZZLES_FILE = "puzzles.parquet"
BANDS: list[tuple[str, int, int | None]] = [
    ("1000-1500", 1000, 1500),
    ("1500-2000", 1500, 2000),
    ("2000+", 2000, None),
]
PUZZLE_COLUMNS = ["puzzle_id", "fen", "moves", "rating", "themes", "band", "split"]
PREFIX_COLUMNS = ["prefix_uci", "prefix_plies", "white_elo", "black_elo"]
BATCH_PUZZLES = 2_000
OK = "ok"
"""Reason code of a prefix that was rebuilt; every other code is a dropped row."""


class PuzzlesConfig(BaseConfig):
    """Remote dataset, quality filters and split sizes."""

    with_games: bool = True
    """Read the with-games dataset and rebuild the real prefix of every puzzle."""
    dataset: str = "Lichess/chess-puzzles-with-games"
    """Source used when ``with_games``: puzzle columns plus the game's ``movetext``."""
    puzzles_only_dataset: str = "Lichess/chess-puzzles"
    """Source used when ``with_games`` is off: the P1 puzzle columns alone, no prefix."""
    remote_glob: str = "hf://datasets/{dataset}/**/*.parquet"
    out_dir: str = "data/puzzles"
    max_rating_deviation: int = Field(default=100, ge=0)
    min_plays: int = Field(default=100, ge=0)
    n_test: int = Field(default=2_000, ge=1)
    n_train: int = Field(default=50_000, ge=1)
    seed: int = 42
    workers: int = Field(default=0, ge=0)
    """Processes that replay the games (``0`` = ``cpu_count() - 1``)."""
    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)

    def source_dataset(self) -> str:
        """The Hub dataset the run reads, which ``with_games`` selects."""
        return self.dataset if self.with_games else self.puzzles_only_dataset


def band(rating: int) -> str | None:
    """Difficulty band for ``rating``; ``None`` below 1000."""
    for name, lo, hi in BANDS:
        if rating >= lo and (hi is None or rating < hi):
            return name
    return None


def _source(cfg: PuzzlesConfig) -> str:
    """Remote parquet glob (tests monkeypatch this to a local file)."""
    return cfg.remote_glob.format(dataset=cfg.source_dataset())


def load_filtered(cfg: PuzzlesConfig) -> pl.DataFrame:
    """Read the puzzles that pass the quality filters with DuckDB (predicate pushdown).

    ``movetext`` is deliberately not read here: the filters leave millions of rows and the games
    are a kilobyte each, so the prefixes are rebuilt after the split, for the selected ids only.
    """
    con = connect(cfg.duckdb)
    try:
        table = con.execute(
            f"""
            SELECT PuzzleId AS puzzle_id, FEN AS fen, Moves AS moves, Rating AS rating,
                   Themes AS themes
            FROM read_parquet('{_source(cfg)}')
            WHERE RatingDeviation <= {cfg.max_rating_deviation}
              AND NbPlays >= {cfg.min_plays}
              AND Rating >= 1000
            """
        ).arrow()
    finally:
        con.close()
    return pl.from_arrow(table)  # type: ignore[return-value]


def load_games(cfg: PuzzlesConfig, puzzle_ids: list[str]) -> pl.DataFrame:
    """Read ``movetext`` and both ratings for the selected puzzles only (a join, not a scan)."""
    wanted = pl.DataFrame({"puzzle_id": pl.Series(puzzle_ids, dtype=pl.String)}).to_arrow()
    con = connect(cfg.duckdb)
    try:
        con.register("wanted_puzzles", wanted)
        table = con.execute(
            f"""
            SELECT p.PuzzleId AS puzzle_id, p.movetext AS movetext,
                   p.WhiteElo AS white_elo, p.BlackElo AS black_elo
            FROM read_parquet('{_source(cfg)}') p
            JOIN wanted_puzzles w ON w.puzzle_id = p.PuzzleId
            """
        ).arrow()
    finally:
        con.close()
    frame: pl.DataFrame = pl.from_arrow(table)  # type: ignore[assignment]
    return frame.with_columns(
        pl.col("white_elo").cast(pl.Int32, strict=False),
        pl.col("black_elo").cast(pl.Int32, strict=False),
    )


class Prefix(NamedTuple):
    """The moves that lead to a puzzle, or why they could not be rebuilt."""

    uci: str
    plies: int
    reason: str


def normalize_fen(fen: str) -> str | None:
    """The four-field FEN of ``rukh.data.positions``; ``None`` when the FEN is unusable."""
    import chess

    from rukh.data.positions import fen4

    try:
        return fen4(chess.Board(fen))
    except ValueError:
        return None


def rebuild_prefix(movetext: str | None, fen: str) -> Prefix:
    """Replay the game's SAN until the board is the puzzle's position; the moves are the prompt.

    The comparison is on the normalized four-field FEN, so the move counters of the puzzle's FEN
    (which the game does not reproduce byte for byte) never decide the match.
    """
    import chess

    from rukh.data.positions import fen4

    target = normalize_fen(fen)
    if target is None:
        return Prefix("", 0, "bad_fen")
    if not movetext:
        return Prefix("", 0, "missing_game")
    board = chess.Board()
    moves: list[str] = []
    if fen4(board) == target:
        return Prefix("", 0, OK)
    for san in san_tokens(movetext):
        try:
            move = board.parse_san(san)
        except ValueError:
            return Prefix("", 0, "illegal")
        moves.append(move.uci())
        board.push(move)
        if fen4(board) == target:
            return Prefix(" ".join(moves), len(moves), OK)
    return Prefix("", 0, "no_match")


def rebuild_batch(rows: list[tuple[str, str | None, str]]) -> list[tuple[str, str, int, str]]:
    """Worker entry point: ``(puzzle_id, movetext, fen)`` rows into prefixes and reason codes."""
    out: list[tuple[str, str, int, str]] = []
    for puzzle_id, movetext, fen in rows:
        prefix = rebuild_prefix(movetext, fen)
        out.append((puzzle_id, prefix.uci, prefix.plies, prefix.reason))
    return out


def _batches(rows: list[tuple[str, str | None, str]]) -> list[list[tuple[str, str | None, str]]]:
    return [rows[start : start + BATCH_PUZZLES] for start in range(0, len(rows), BATCH_PUZZLES)]


def attach_prefixes(frame: pl.DataFrame, cfg: PuzzlesConfig) -> tuple[pl.DataFrame, dict[str, int]]:
    """Add ``prefix_uci``/``prefix_plies``/both ratings; drop and count what cannot be rebuilt."""
    games = load_games(cfg, frame["puzzle_id"].to_list())
    joined = frame.join(games, on="puzzle_id", how="left")
    rows = list(
        zip(
            joined["puzzle_id"].to_list(),
            joined["movetext"].to_list(),
            joined["fen"].to_list(),
            strict=True,
        )
    )
    rebuilt: list[tuple[str, str, int, str]] = []
    for part in run_batches(rebuild_batch, _batches(rows), resolve_workers(cfg.workers)):
        rebuilt.extend(part)
    prefixes = pl.DataFrame(
        rebuilt,
        schema={
            "puzzle_id": pl.String,
            "prefix_uci": pl.String,
            "prefix_plies": pl.Int32,
            "reason": pl.String,
        },
        orient="row",
    )
    dropped = {
        reason: int(n)
        for reason, n in prefixes.filter(pl.col("reason") != OK)
        .group_by("reason")
        .len()
        .sort("reason")
        .iter_rows()
    }
    out = (
        joined.drop("movetext")
        .join(prefixes, on="puzzle_id", how="left")
        .filter(pl.col("reason") == OK)
        .drop("reason")
        .select(*PUZZLE_COLUMNS, *PREFIX_COLUMNS)
    )
    return out, dropped


def split_key(puzzle_id: str, seed: int) -> int:
    """Stable 64-bit order key: ``blake2b(seed:puzzle_id)``."""
    digest = hashlib.blake2b(f"{seed}:{puzzle_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def split_frame(frame: pl.DataFrame, cfg: PuzzlesConfig) -> pl.DataFrame:
    """Assign ``band`` and ``split``: per band, first ``n_test`` by key then up to ``n_train``."""
    keyed = frame.with_columns(
        pl.col("rating").map_elements(band, return_dtype=pl.String).alias("band"),
        pl.col("puzzle_id")
        .map_elements(lambda p: split_key(p, cfg.seed), return_dtype=pl.UInt64)
        .alias("key"),
        pl.col("themes").cast(pl.List(pl.String)),
    ).filter(pl.col("band").is_not_null())
    keyed = keyed.sort(["band", "key"]).with_columns(
        pl.int_range(pl.len()).over("band").alias("rank")
    )
    return (
        keyed.with_columns(
            pl.when(pl.col("rank") < cfg.n_test)
            .then(pl.lit("test"))
            .when(pl.col("rank") < cfg.n_test + cfg.n_train)
            .then(pl.lit("train"))
            .otherwise(None)
            .alias("split")
        )
        .filter(pl.col("split").is_not_null())
        .select(*PUZZLE_COLUMNS)
    )


def run(cfg: PuzzlesConfig) -> Manifest:
    """Download, filter, split and write ``out_dir/puzzles.parquet`` plus the manifest."""
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = split_frame(load_filtered(cfg), cfg)
    selected = frame.height
    dropped: dict[str, int] = {}
    if cfg.with_games:
        frame, dropped = attach_prefixes(frame, cfg)
    target = out_dir / PUZZLES_FILE
    frame.write_parquet(target.as_posix(), compression="zstd")
    counts = {
        f"{b}/{s}": int(n)
        for b, s, n in frame.group_by("band", "split").len().sort("band", "split").iter_rows()
    }
    counts["selected"] = selected
    counts["dropped_no_prefix"] = sum(dropped.values())
    manifest = Manifest(
        dataset=cfg.source_dataset(),
        months=[],
        filters={
            "max_rating_deviation": cfg.max_rating_deviation,
            "min_plays": cfg.min_plays,
            "bands": [b for b, _, _ in BANDS],
            "n_test": cfg.n_test,
            "n_train": cfg.n_train,
            "seed": cfg.seed,
            "split_by": "blake2b(seed:PuzzleId)",
            "with_games": cfg.with_games,
            "prefix": {
                "source": "movetext replayed to the puzzle FEN" if cfg.with_games else None,
                "dropped": dropped,
                "dropped_fraction": (sum(dropped.values()) / selected) if selected else 0.0,
            },
        },
        counts=counts,
        files=[
            FileHash(path=PUZZLES_FILE, sha256=sha256_file(target), bytes=target.stat().st_size)
        ],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
