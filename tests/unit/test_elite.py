"""Tests for rukh.data.elite: monkeypatched download, PGN parsing and UCI conversion."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from rukh.data import elite as elite_module
from rukh.data.elite import EliteConfig, extract_pgn, iter_pgn_rows, run, zip_url

pytestmark = pytest.mark.unit

PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc1"]
[Date "2025.01.03"]
[White "A"]
[Black "B"]
[Result "1-0"]
[UTCDate "2025.01.03"]
[WhiteElo "2612"]
[BlackElo "2380"]
[ECO "C50"]
[TimeControl "180+2"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4 8. O-O Bxc3
9. d5 Bf6 10. Re1 Ne7 11. Rxe4 d6 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/abc2"]
[Result "0-1"]
[UTCDate "2025.01.04"]
[WhiteElo "2540"]
[BlackElo "2610"]
[ECO "A00"]
[TimeControl "300+0"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "Rated Blitz game"]
[Site "https://lichess.org/abc3"]
[Result "1/2-1/2"]
[UTCDate "2025.01.04"]
[WhiteElo "2500"]
[BlackElo "2400"]
[TimeControl "300+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8
10. d4 Nbd7 11. Nbd2 Bb7 1/2-1/2

"""


def test_zip_url() -> None:
    cfg = EliteConfig()
    assert zip_url(cfg, "2025-01") == "https://database.nikonoel.fr/lichess_elite_2025-01.zip"


def test_iter_pgn_rows(tmp_path: Path) -> None:
    pgn = tmp_path / "x.pgn"
    pgn.write_text(PGN, encoding="utf-8")
    rows = list(iter_pgn_rows(pgn, "2025-01"))
    assert [r["Site"] for r in rows] == [f"https://lichess.org/abc{i}" for i in (1, 2, 3)]
    assert rows[0]["WhiteElo"] == 2612 and rows[0]["Result"] == "1-0"
    assert rows[0]["movetext"].startswith("1. e4 e5") and rows[0]["movetext"].endswith("d6 1-0")
    assert rows[1]["ECO"] == "A00" and rows[2]["ECO"] is None
    assert rows[2]["month"] == "2025-01"


def _fake_download(pgn_text: str, calls: list[str]):  # type: ignore[no-untyped-def]
    def download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        month = url.rsplit("_", 1)[1].removesuffix(".zip")
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr(f"lichess_elite_{month}.pgn", pgn_text)

    return download


def test_run_downloads_once_and_converts(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(elite_module, "_download", _fake_download(PGN, calls))
    cfg = EliteConfig(months=["2025-01", "2025-02"], workers=1)
    manifest = run(cfg)
    assert calls == [zip_url(cfg, "2025-01"), zip_url(cfg, "2025-02")]
    out = rukh_home / "data" / "elite"
    frame = pl.read_parquet((out / "games.parquet").as_posix())
    # The 4-ply game is too short; the other two survive, once per month.
    assert frame.height == 4
    assert frame["month"].to_list() == ["2025-01", "2025-01", "2025-02", "2025-02"]
    assert frame["uci"][0].startswith("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5")
    assert frame.schema["white_elo"] == pl.Int16
    assert str(frame["utc_date"][0]) == "2025-01-03"
    assert manifest.counts == {"games": 4}
    assert manifest.filters["conversion"]["short"] == 2
    assert [f.path for f in manifest.files] == [
        "lichess_elite_2025-01.zip",
        "lichess_elite_2025-02.zip",
        "games.parquet",
    ]
    assert manifest.filters["source_urls"]["2025-01"].endswith("lichess_elite_2025-01.zip")
    # A second run reuses the zips on disk.
    run(cfg)
    assert len(calls) == 2
    assert extract_pgn(out / "lichess_elite_2025-01.zip", out).name == "lichess_elite_2025-01.pgn"
