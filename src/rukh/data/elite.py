"""Lichess Elite Database (2500+ vs 2300+, no bullet): download, extract, convert to UCI."""

from __future__ import annotations

import multiprocessing
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import (
    OUT_SCHEMA,
    RAW_COLUMNS,
    _convert_batch,
    _write_results,
    resolve_workers,
    sha256_file,
)
from rukh.paths import resolve

GAMES_FILE = "games.parquet"
BATCH_GAMES = 20_000
_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


class EliteConfig(BaseConfig):
    """Which monthly zips to fetch and the ply bounds of the conversion."""

    base_url: str = "https://database.nikonoel.fr"
    months: list[str] = Field(default=["2025-01", "2025-02"], min_length=1)
    out_dir: str = "data/elite"
    min_plies: int = Field(default=20, ge=0)
    max_plies: int = Field(default=300, ge=1)
    workers: int = Field(default=0, ge=0)


def zip_name(month: str) -> str:
    return f"lichess_elite_{month}.zip"


def zip_url(cfg: EliteConfig, month: str) -> str:
    return f"{cfg.base_url.rstrip('/')}/{zip_name(month)}"


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (tests monkeypatch this)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)


def extract_pgn(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the (single) ``.pgn`` member of the zip; returns its path."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".pgn")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected one .pgn member, found {members}")
        archive.extract(members[0], dest_dir)
    return dest_dir / members[0]


def iter_pgn_rows(pgn_path: Path, month: str) -> Iterator[dict[str, object]]:
    """Yield raw rows (``RAW_COLUMNS``) from a PGN file with a light line parser."""
    headers: dict[str, str] = {}
    movetext: list[str] = []
    in_moves = False

    def flush() -> dict[str, object] | None:
        if not headers or not movetext:
            return None
        return {
            "Site": headers.get("Site", ""),
            "movetext": " ".join(movetext),
            "WhiteElo": _int(headers.get("WhiteElo")),
            "BlackElo": _int(headers.get("BlackElo")),
            "Result": headers.get("Result", "*"),
            "TimeControl": headers.get("TimeControl"),
            "UTCDate": headers.get("UTCDate", headers.get("Date")),
            "ECO": headers.get("ECO"),
            "month": month,
        }

    with pgn_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                match = _HEADER_RE.match(line)
                if match is None:
                    continue
                if in_moves:
                    row = flush()
                    if row is not None:
                        yield row
                    headers, movetext, in_moves = {}, [], False
                headers[match.group(1)] = match.group(2)
            elif line:
                in_moves = True
                movetext.append(line)
    row = flush()
    if row is not None:
        yield row


def _int(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


def _batches(rows: Iterator[dict[str, object]]) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for row in rows:
        for name in RAW_COLUMNS:
            row.setdefault(name, None)
        batch.append(row)
        if len(batch) >= BATCH_GAMES:
            yield batch
            batch = []
    if batch:
        yield batch


def convert_pgns(pgns: list[tuple[str, Path]], dst: Path, cfg: EliteConfig) -> dict[str, int]:
    """Convert every ``(month, pgn)`` into one UCI parquet with the ``uci.py`` schema."""
    totals = {"rows": 0, "kept": 0, "illegal": 0, "short": 0, "long": 0}
    workers = resolve_workers(cfg.workers)

    def jobs() -> Iterator[tuple[list[dict[str, object]], int, int]]:
        for month, pgn in pgns:
            for batch in _batches(iter_pgn_rows(pgn, month)):
                yield batch, cfg.min_plies, cfg.max_plies

    dst.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dst, OUT_SCHEMA, compression="zstd") as writer:
        if workers == 1:
            _write_results(writer, map(_convert_batch, jobs()), totals)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(workers) as pool:
                _write_results(writer, pool.imap(_convert_batch, jobs()), totals)
    return totals


def run(cfg: EliteConfig) -> Manifest:
    """Download each month's zip (once), extract, convert and write the manifest."""
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[FileHash] = []
    sources: dict[str, str] = {}
    pgns: list[tuple[str, Path]] = []
    for month in cfg.months:
        zip_path = out_dir / zip_name(month)
        url = zip_url(cfg, month)
        if not zip_path.is_file():
            _download(url, zip_path)
        sources[month] = url
        files.append(
            FileHash(
                path=zip_path.name, sha256=sha256_file(zip_path), bytes=zip_path.stat().st_size
            )
        )
        pgns.append((month, extract_pgn(zip_path, out_dir)))
    target = out_dir / GAMES_FILE
    totals = convert_pgns(pgns, target, cfg)
    files.append(FileHash(path=GAMES_FILE, sha256=sha256_file(target), bytes=target.stat().st_size))
    manifest = Manifest(
        dataset="Lichess Elite Database (database.nikonoel.fr)",
        months=list(cfg.months),
        filters={
            "source_urls": sources,
            "min_plies": cfg.min_plies,
            "max_plies": cfg.max_plies,
            "legal_only": True,
            "conversion": totals,
        },
        counts={"games": totals["kept"]},
        files=files,
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
