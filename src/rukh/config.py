"""Strict pydantic configs loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base class for every config: unknown keys are an error, never silently ignored."""

    model_config = ConfigDict(extra="forbid")


def load_yaml[T: BaseModel](path: Path, model: type[T]) -> T:
    """Load ``path`` as YAML and validate it against ``model``.

    Raises ``ValueError`` when the document is not a mapping and
    ``pydantic.ValidationError`` when it does not match the model.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return model.model_validate(raw)
