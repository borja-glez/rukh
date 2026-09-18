"""Tests for scripts/get_stockfish.py: archive member selection."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "get_stockfish.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("get_stockfish", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _tar(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_windows_zip_picks_the_exe_not_the_sources(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script.platform, "system", lambda: "Windows")
    payload = _zip(
        {
            "stockfish/src/stockfish.cpp": b"// source",
            "stockfish/src/stockfish.h": b"// header",
            "stockfish/stockfish-windows-x86-64-universal.exe": b"MZ-binary",
        }
    )
    target = tmp_path / "stockfish.exe"
    script.extract_executable("stockfish-windows-x86-64-universal.zip", payload, target)
    assert target.read_bytes() == b"MZ-binary"


def test_linux_tar_picks_the_plain_binary_not_the_sources(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script.platform, "system", lambda: "Linux")
    payload = _tar(
        {
            "stockfish/src/stockfish.cpp": b"// source",
            "stockfish/src/stockfish": b"// source dir file without extension",
            "stockfish/stockfish-ubuntu-x86-64-avx2": b"ELF-binary",
        }
    )
    target = tmp_path / "stockfish"
    script.extract_executable("stockfish-ubuntu-x86-64-avx2.tar", payload, target)
    assert target.read_bytes() == b"ELF-binary"


def test_archive_with_only_sources_fails(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script.platform, "system", lambda: "Windows")
    payload = _zip({"stockfish/src/stockfish.cpp": b"// source"})
    with pytest.raises(SystemExit, match="no stockfish executable"):
        script.extract_executable("stockfish.zip", payload, tmp_path / "stockfish.exe")
