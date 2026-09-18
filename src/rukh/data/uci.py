"""SAN movetext to UCI: cleaning, legality check and ply filter, one parquet per month.

Every game is replayed with python-chess; a game with an unparsable or illegal move is
dropped (``illegal``), as are games shorter than ``min_plies`` (``short``) or longer than
``max_plies`` (``long``). Batches of ``BATCH_GAMES`` rows are converted in a ``spawn``
process pool (Windows-safe: every worker entry point is a module-level function).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.parallel import run_batches
from rukh.paths import resolve

BATCH_GAMES = 20_000
RAW_COLUMNS = [
    "Site",
    "movetext",
    "WhiteElo",
    "BlackElo",
    "Result",
    "TimeControl",
    "UTCDate",
    "ECO",
    "month",
]
OUT_SCHEMA = pa.schema(
    [
        ("game_id", pa.int64()),
        ("uci", pa.string()),
        ("n_plies", pa.int16()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("result", pa.string()),
        ("time_control", pa.string()),
        ("utc_date", pa.date32()),
        ("eco", pa.string()),
        ("month", pa.string()),
    ]
)
RESULTS = ("1-0", "0-1", "1/2-1/2")

_COMMENT_RE = re.compile(r"\{[^}]*\}")
_VARIATION_RE = re.compile(r"\([^()]*\)")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUMBER_RE = re.compile(r"\b\d+\.(\.\.)?")
_RESULT_RE = re.compile(r"(1-0|0-1|1/2-1/2|\*)\s*$")
_ANNOTATION_RE = re.compile(r"[!?]+$")


class UciConfig(BaseConfig):
    """Input/output directories, ply bounds and parallelism (``0`` = ``cpu_count() - 1``)."""

    raw_dir: str = "data/raw"
    out_dir: str = "data/uci"
    min_plies: int = Field(default=20, ge=0)
    max_plies: int = Field(default=300, ge=1)
    workers: int = Field(default=0, ge=0)


def clean_movetext(movetext: str) -> str:
    """Strip ``{...}`` comments (``%clk``, ``%eval``), variations, NAGs, move numbers, result."""
    text = _COMMENT_RE.sub(" ", movetext)
    while _VARIATION_RE.search(text):
        text = _VARIATION_RE.sub(" ", text)
    text = _NAG_RE.sub(" ", text)
    text = _RESULT_RE.sub(" ", text.strip())
    text = _MOVE_NUMBER_RE.sub(" ", text)
    return " ".join(text.split())


def san_to_uci(movetext: str) -> tuple[str, int] | None:
    """Replay the cleaned SAN with python-chess; ``None`` when any move is illegal."""
    import chess

    board = chess.Board()
    moves: list[str] = []
    for san in clean_movetext(movetext).split():
        san = _ANNOTATION_RE.sub("", san)
        if not san:
            continue
        try:
            move = board.parse_san(san)
        except ValueError:
            return None
        moves.append(move.uci())
        board.push(move)
    return " ".join(moves), len(moves)


def game_id(site: str) -> int:
    """Signed 64-bit id: the first 8 bytes (little-endian) of ``sha256(Site)``."""
    return int.from_bytes(hashlib.sha256(site.encode("utf-8")).digest()[:8], "little", signed=True)


def convert_rows(
    rows: list[dict[str, object]], min_plies: int, max_plies: int
) -> dict[str, object]:
    """Convert one batch of raw rows (dicts with ``RAW_COLUMNS``) into output columns + counts.

    Module-level so ``multiprocessing`` can pickle it under the ``spawn`` start method.
    """
    out: dict[str, list[object]] = {name: [] for name in OUT_SCHEMA.names}
    counts = {"rows": 0, "kept": 0, "illegal": 0, "short": 0, "long": 0}
    for row in rows:
        counts["rows"] += 1
        result = str(row["Result"])
        converted = san_to_uci(str(row["movetext"])) if result in RESULTS else None
        if converted is None:
            counts["illegal"] += 1
            continue
        uci, n_plies = converted
        if n_plies < min_plies:
            counts["short"] += 1
            continue
        if n_plies > max_plies:
            counts["long"] += 1
            continue
        counts["kept"] += 1
        out["game_id"].append(game_id(str(row["Site"])))
        out["uci"].append(uci)
        out["n_plies"].append(n_plies)
        out["white_elo"].append(int(row["WhiteElo"]))  # type: ignore[call-overload]
        out["black_elo"].append(int(row["BlackElo"]))  # type: ignore[call-overload]
        out["result"].append(result)
        out["time_control"].append(row["TimeControl"])
        out["utc_date"].append(_parse_date(row["UTCDate"]))
        out["eco"].append(row["ECO"])
        out["month"].append(row["month"])
    return {"columns": out, "counts": counts}


def _parse_date(value: object) -> object:
    from datetime import date

    if value is None or isinstance(value, date):
        return value
    text = str(value).replace(".", "-")
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def convert_batch(args: tuple[list[dict[str, object]], int, int]) -> dict[str, object]:
    """Worker entry point: ``(rows, min_plies, max_plies)`` into columns and counts."""
    rows, min_plies, max_plies = args
    return convert_rows(rows, min_plies, max_plies)


def iter_batches(src: Path) -> Iterator[list[dict[str, object]]]:
    reader = pq.ParquetFile(src)
    columns = [c for c in RAW_COLUMNS if c in reader.schema_arrow.names]
    for batch in reader.iter_batches(batch_size=BATCH_GAMES, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            for name in RAW_COLUMNS:
                row.setdefault(name, None)
        yield rows


def resolve_workers(workers: int) -> int:
    return workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)


def convert_month(src: Path, dst: Path, cfg: UciConfig) -> dict[str, int]:
    """Convert ``src`` (raw parquet) into ``dst`` (UCI parquet); returns the counts."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    totals = {"rows": 0, "kept": 0, "illegal": 0, "short": 0, "long": 0}
    workers = resolve_workers(cfg.workers)
    jobs = ((rows, cfg.min_plies, cfg.max_plies) for rows in iter_batches(Path(src)))
    with pq.ParquetWriter(dst, OUT_SCHEMA, compression="zstd") as writer:
        write_results(writer, run_batches(convert_batch, jobs, workers), totals)
    return totals


def write_results(
    writer: pq.ParquetWriter, results: Iterator[dict[str, object]], totals: dict[str, int]
) -> None:
    for result in results:
        counts: dict[str, int] = result["counts"]  # type: ignore[assignment]
        for key, value in counts.items():
            totals[key] += value
        columns: dict[str, list[object]] = result["columns"]  # type: ignore[assignment]
        if columns["game_id"]:
            writer.write_table(pa.table(columns, schema=OUT_SCHEMA))


def month_dirs(raw_dir: Path) -> list[tuple[str, Path]]:
    """``(YYYY-MM, games.parquet)`` for every ``year=YYYY/month=MM`` under ``raw_dir``."""
    found: list[tuple[str, Path]] = []
    for path in sorted(Path(raw_dir).glob("year=*/month=*/games.parquet")):
        year = path.parents[1].name.split("=", 1)[1]
        mm = path.parent.name.split("=", 1)[1]
        found.append((f"{year}-{mm}", path))
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg: UciConfig) -> Manifest:
    """Convert every month under ``raw_dir`` and write ``out_dir/manifest.json``."""
    raw_dir = resolve(cfg.raw_dir)
    out_dir = resolve(cfg.out_dir)
    months = month_dirs(raw_dir)
    if not months:
        raise FileNotFoundError(f"no year=*/month=*/games.parquet under {raw_dir}")
    raw_manifest_path = raw_dir / "manifest.json"
    raw_manifest = (
        Manifest.model_validate_json(raw_manifest_path.read_text("utf-8"))
        if raw_manifest_path.is_file()
        else None
    )
    counts: dict[str, int] = {}
    details: dict[str, dict[str, int]] = {}
    files: list[FileHash] = []
    for month, src in months:
        rel = src.relative_to(raw_dir)
        dst = out_dir / rel
        totals = convert_month(src, dst, cfg)
        counts[month] = totals["kept"]
        details[month] = totals
        files.append(
            FileHash(path=rel.as_posix(), sha256=sha256_file(dst), bytes=dst.stat().st_size)
        )
    filters: dict[str, object] = {
        "min_plies": cfg.min_plies,
        "max_plies": cfg.max_plies,
        "legal_only": True,
        "conversion": details,
    }
    if raw_manifest is not None:
        filters["raw"] = raw_manifest.filters
    manifest = Manifest(
        dataset=raw_manifest.dataset if raw_manifest else "Lichess/standard-chess-games",
        months=[m for m, _ in months],
        filters=filters,
        counts=counts,
        files=files,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
