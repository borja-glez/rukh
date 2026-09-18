"""Tests for rukh.data.manifest."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rukh import __version__
from rukh.data.manifest import FileHash, Manifest

pytestmark = pytest.mark.unit


def _manifest() -> Manifest:
    return Manifest(
        dataset="Lichess/standard-chess-games",
        months=["2025-01", "2025-02"],
        filters={"min_elo": 1800, "terminations": ["Normal", "Time forfeit"]},
        counts={"2025-01": 10, "2025-02": 12},
        files=[
            FileHash(path="data/raw/year=2025/month=01/games.parquet", sha256="ab" * 32, bytes=1)
        ],
        created_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )


def test_manifest_json_roundtrip_is_idempotent() -> None:
    m = _manifest()
    again = Manifest.model_validate_json(m.model_dump_json())
    assert again == m
    assert again.model_dump_json() == m.model_dump_json()


def test_manifest_defaults_rukh_version() -> None:
    assert _manifest().rukh_version == __version__


def test_manifest_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate({**_manifest().model_dump(), "extra": 1})


def test_file_hash_validates_sha256() -> None:
    with pytest.raises(ValidationError):
        FileHash(path="x", sha256="not-a-hash", bytes=0)
