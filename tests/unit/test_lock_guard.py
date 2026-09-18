"""Guard the torch pin: both extras must resolve the same torch range and lock the same version."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TORCH_SPEC_PREFIX = "torch>=2.11,<2.12"


def _torch_spec(extras: dict[str, list[str]], name: str) -> str:
    specs = [dep for dep in extras[name] if re.match(r"^torch\b", dep)]
    assert len(specs) == 1, f"extra {name!r} must declare exactly one torch requirement"
    return specs[0]


def test_extras_pin_the_same_torch_range(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    cpu = _torch_spec(extras, "cpu")
    cu128 = _torch_spec(extras, "cu128")
    assert cpu == cu128
    assert cpu.startswith(TORCH_SPEC_PREFIX)


def test_extras_are_declared_as_conflicting(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    conflicts = pyproject["tool"]["uv"]["conflicts"]
    assert [{"extra": "cpu"}, {"extra": "cu128"}] in conflicts


def test_lock_has_torch_2_11_for_both_indexes(repo_root: Path) -> None:
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    torch_entries = [pkg for pkg in lock["package"] if pkg["name"] == "torch"]
    assert torch_entries, "uv.lock has no torch entry"
    by_index: dict[str, set[str]] = {"cpu": set(), "cu128": set()}
    for entry in torch_entries:
        index_url = entry.get("source", {}).get("registry", "")
        if index_url.endswith("/whl/cpu"):
            by_index["cpu"].add(entry["version"])
        elif index_url.endswith("/whl/cu128"):
            by_index["cu128"].add(entry["version"])
    assert by_index["cpu"], "no torch entry locked from the pytorch-cpu index"
    assert by_index["cu128"], "no torch entry locked from the pytorch-cu128 index"
    assert any(v.endswith("+cpu") for v in by_index["cpu"]), by_index
    assert any(v.endswith("+cu128") for v in by_index["cu128"]), by_index
    for index, versions in by_index.items():
        for version in versions:
            assert version.startswith("2.11."), f"torch from {index} locked at {version}"
