"""Download Stockfish 19 into ``tools/stockfish/`` (gitignored). Idempotent.

Usage: ``uv run python scripts/get_stockfish.py [--force]``

Windows gets ``stockfish-windows-x86-64-universal.zip`` (sf_19 ships no separate avx2 build; the
universal binary picks the best code path at runtime). Linux gets
``stockfish-ubuntu-x86-64-avx2.tar``. The executable ends up as ``tools/stockfish/stockfish.exe``
or ``tools/stockfish/stockfish``.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "sf_19"
BASE_URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{RELEASE}"
ASSETS = {
    "Windows": "stockfish-windows-x86-64-universal.zip",
    "Linux": "stockfish-ubuntu-x86-64-avx2.tar",
}


def repo_root() -> Path:
    home = os.environ.get("RUKH_HOME")
    return Path(home).expanduser().resolve() if home else Path(__file__).resolve().parents[1]


def target_path(dest: Path) -> Path:
    return dest / ("stockfish.exe" if platform.system() == "Windows" else "stockfish")


def download(url: str) -> bytes:
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "rukh/get_stockfish"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def is_executable_member(name: str) -> bool:
    """True for the engine binary inside the release archive, never for a ``src/`` file.

    Windows archives ship ``stockfish-<variant>.exe``; the others ship a plain binary with no
    extension. Both archives also contain the sources under ``src/``, which must be skipped.
    """
    path = Path(name)
    if "src" in path.parts or not path.name.startswith("stockfish"):
        return False
    if platform.system() == "Windows":
        return path.suffix == ".exe"
    return path.suffix == ""


def extract_executable(asset: str, payload: bytes, target: Path) -> None:
    """Pull the single ``stockfish*`` executable out of the archive into ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [
                n for n in archive.namelist() if not n.endswith("/") and is_executable_member(n)
            ]
            if not names:
                raise SystemExit(f"no stockfish executable inside {asset}")
            with archive.open(names[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = [
                m for m in archive.getmembers() if m.isfile() and is_executable_member(m.name)
            ]
            if not members:
                raise SystemExit(f"no stockfish executable inside {asset}")
            src = archive.extractfile(members[0])
            if src is None:
                raise SystemExit(f"cannot read {members[0].name} from {asset}")
            with src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    if platform.system() != "Windows":
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root() / "tools" / "stockfish",
        help="Destination directory (default: <repo>/tools/stockfish).",
    )
    args = parser.parse_args(argv)

    system = platform.system()
    asset = ASSETS.get(system)
    if asset is None:
        print(f"unsupported platform {system!r}; download Stockfish manually", file=sys.stderr)
        return 2

    target = target_path(args.dest)
    if target.is_file() and not args.force:
        print(f"already present: {target}")
        return 0

    payload = download(f"{BASE_URL}/{asset}")
    print(f"received {len(payload) / 1e6:.1f} MB")
    extract_executable(asset, payload, target)
    print(f"installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
