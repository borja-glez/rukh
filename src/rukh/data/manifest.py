"""Manifest written next to fetched data: what was asked for, what came back."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from rukh import __version__


class FileHash(BaseModel):
    """One materialized file with its content hash and size."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class Manifest(BaseModel):
    """Provenance record for a fetched dataset slice."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    months: list[str]
    filters: dict[str, object]
    counts: dict[str, int]
    files: list[FileHash]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rukh_version: str = __version__
