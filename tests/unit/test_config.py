"""Tests for rukh.config: strict pydantic configs loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rukh.config import BaseConfig, load_yaml

pytestmark = pytest.mark.unit


class DemoConfig(BaseConfig):
    name: str
    steps: int = 10


def test_base_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        DemoConfig(name="x", unknown=1)  # type: ignore[call-arg]


def test_load_yaml_returns_model(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("name: demo\nsteps: 3\n", encoding="utf-8")
    cfg = load_yaml(path, DemoConfig)
    assert cfg == DemoConfig(name="demo", steps=3)


def test_load_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("name: demo\ntypo: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_yaml(path, DemoConfig)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path, DemoConfig)
