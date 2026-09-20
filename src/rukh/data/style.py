"""Style slices of the corpus: the games one LoRA adapter is meant to sound like.

An adapter is only interesting if you can *see* it work, and the cheapest visible axis in this
data is the opening. A model whose adapter was trained on games that begin ``1. e4`` opens with
``1. e4``; swap the adapter and it opens ``1. d4``. That is two megabytes changing the behaviour
of a 115 M model, in front of the reader, with a number behind it (the share of first moves) and
a cost that can be measured (the Elo of the adapted model against the base).

The slice is defined by **typed fields**, not by a SQL string in the YAML. A predicate written in
a config file is an injection waiting to happen and, worse, it is undocumented: nobody reading
`configs/` would know that `eco` is a column or what shape it has. `first_move` and `eco_prefix`
say what they select, and the manifest records them, so a published adapter can state exactly
which games made it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, model_validator

from rukh.config import BaseConfig
from rukh.data.db import DuckDbConfig, connect
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import month_dirs, sha256_file
from rukh.paths import resolve

log = logging.getLogger(__name__)

__all__ = ["StyleConfig", "StyleSpec", "build_styles", "predicate_of"]


class StyleSpec(BaseConfig):
    """One adapter's worth of games."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9-]+$")
    first_move: str | None = None
    """UCI of White's first move, e.g. ``e2e4``. Matched on the prefix of the ``uci`` column."""
    eco_prefix: list[str] = []
    """ECO codes the game may have, by prefix: ``["B2", "B3"]`` is most of the Sicilian."""
    min_elo: int | None = None
    """Both players at least this strong, so a style adapter is not also a weakness adapter."""
    n_games: int = Field(default=200_000, ge=1)

    @model_validator(mode="after")
    def _check(self) -> StyleSpec:
        if self.first_move is None and not self.eco_prefix:
            raise ValueError(f"style {self.name!r} selects nothing: give first_move or eco_prefix")
        if self.first_move is not None and len(self.first_move) not in (4, 5):
            raise ValueError(f"first_move {self.first_move!r} is not a UCI move")
        return self


class StyleConfig(BaseConfig):
    """Where the games come from and which slices to cut out of them."""

    sources: list[str] = ["data/uci"]
    out_dir: str = "data/style"
    seed: int = 42
    styles: list[StyleSpec] = []
    duckdb: DuckDbConfig = Field(default_factory=DuckDbConfig)

    @model_validator(mode="after")
    def _check(self) -> StyleConfig:
        names = [style.name for style in self.styles]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate style names: {names}")
        return self


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def predicate_of(spec: StyleSpec) -> str:
    """The WHERE clause of one style, built from its typed fields and nothing else."""
    parts: list[str] = []
    if spec.first_move is not None:
        # `uci` is space-separated moves, so the first one is everything up to the first space.
        parts.append(f"split_part(uci, ' ', 1) = {_sql_string(spec.first_move)}")
    if spec.eco_prefix:
        matches = " OR ".join(
            f"eco LIKE {_sql_string(prefix + '%')}" for prefix in sorted(spec.eco_prefix)
        )
        parts.append(f"({matches})")
    if spec.min_elo is not None:
        parts.append(f"white_elo >= {int(spec.min_elo)} AND black_elo >= {int(spec.min_elo)}")
    return " AND ".join(parts)


def build_styles(cfg: StyleConfig) -> dict[str, Manifest]:
    """Write one parquet per style under ``out_dir`` and return their manifests."""
    if not cfg.styles:
        raise ValueError("no styles configured")
    months: list[tuple[str, Path]] = []
    for source in cfg.sources:
        found = month_dirs(resolve(source))
        if not found:
            raise FileNotFoundError(f"no games under {resolve(source)}")
        months.extend(found)
    files = ", ".join(f"'{path.as_posix()}'" for _, path in months)
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, Manifest] = {}
    con = connect(cfg.duckdb)
    try:
        for spec in cfg.styles:
            target = out_dir / f"{spec.name}.parquet"
            where = predicate_of(spec)
            log.info("style %s: %s", spec.name, where)
            con.execute(
                f"""
                COPY (
                  SELECT * FROM read_parquet([{files}], hive_partitioning = false,
                                             union_by_name = true)
                  WHERE {where}
                  ORDER BY hash(game_id) + {cfg.seed}
                  LIMIT {int(spec.n_games)}
                ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            row = con.execute(
                f"SELECT count(*) FROM read_parquet('{target.as_posix()}')"
            ).fetchone()
            count = int(row[0]) if row else 0
            manifests[spec.name] = Manifest(
                dataset="Lichess/standard-chess-games",
                months=sorted({month for month, _ in months}),
                filters={
                    "style": spec.model_dump(mode="json"),
                    "sources": list(cfg.sources),
                    "predicate": where,
                    "seed": cfg.seed,
                },
                counts={spec.name: count},
                files=[
                    FileHash(
                        path=target.name,
                        sha256=sha256_file(target),
                        bytes=target.stat().st_size,
                    )
                ],
            )
            (out_dir / f"{spec.name}.manifest.json").write_text(
                manifests[spec.name].model_dump_json(indent=2) + "\n", "utf-8"
            )
    finally:
        con.close()
    return manifests
