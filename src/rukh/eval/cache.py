"""SQLite cache of finished evaluation items, keyed by ``(model_sha, suite, item_id)``.

Games against Stockfish and puzzle lines are the expensive part of the harness: a full suite is
hundreds of games and thousands of puzzles. Every finished item is stored as JSON under the SHA
of the weights that produced it, so re-running a suite after a crash (or after adding one rung)
only pays for what is missing. ``--no-cache`` builds a disabled cache: it reads nothing and
writes nothing, which is what a benchmark of the harness itself wants.

The key is the weights **and** the settings that change what an item means: temperature, top-k,
the engine's move time, the ply limit, the rung definitions, the number of games and the seed.
Without them, lowering the temperature or raising the move time would silently reuse games
played under the old settings and report them as the new ones.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    model_sha TEXT NOT NULL,
    suite     TEXT NOT NULL,
    item_id   TEXT NOT NULL,
    payload   TEXT NOT NULL,
    PRIMARY KEY (model_sha, suite, item_id)
)
"""
CHUNK = 1 << 20
SHA_PREFIX = 16
"""Characters of each SHA kept in the key: enough to identify, short enough to read."""


def config_sha(fields: Mapping[str, Any]) -> str:
    """SHA-256 of the evaluation-relevant settings, as canonical JSON."""
    payload = json.dumps(dict(fields), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    """SHA-256 of a file, read in chunks (a checkpoint does not fit comfortably in memory)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class EvalCache:
    """Key-value store of evaluation items; ``enabled=False`` turns every call into a no-op."""

    def __init__(
        self,
        path: Path | None,
        model_sha: str,
        enabled: bool = True,
        config_sha: str | None = None,
    ) -> None:
        self.model_sha = model_sha
        self.config_sha = config_sha
        self.key = (
            model_sha if not config_sha else f"{model_sha[:SHA_PREFIX]}:{config_sha[:SHA_PREFIX]}"
        )
        self.enabled = enabled and path is not None
        self.path = Path(path) if path is not None else None
        self._conn: sqlite3.Connection | None = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.execute(SCHEMA)
            self._conn.commit()

    def get(self, suite: str, item_id: str) -> dict[str, Any] | None:
        """The stored payload of one item, or None when it has not been computed yet."""
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT payload FROM items WHERE model_sha = ? AND suite = ? AND item_id = ?",
            (self.key, suite, item_id),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None

    def put(self, suite: str, item_id: str, payload: Mapping[str, Any]) -> None:
        """Store (or replace) one finished item."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO items (model_sha, suite, item_id, payload) VALUES (?, ?, ?, ?)",
            (self.key, suite, item_id, json.dumps(dict(payload), default=str)),
        )
        self._conn.commit()

    def count(self, suite: str | None = None) -> int:
        """How many items are stored for this model (optionally for one suite only)."""
        if self._conn is None:
            return 0
        if suite is None:
            query = "SELECT COUNT(*) FROM items WHERE model_sha = ?"
            args: tuple[str, ...] = (self.key,)
        else:
            query = "SELECT COUNT(*) FROM items WHERE model_sha = ? AND suite = ?"
            args = (self.key, suite)
        return int(self._conn.execute(query, args).fetchone()[0])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> EvalCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
