"""Tests for rukh.data.uci: movetext cleaning, SAN to UCI, ply filters and month conversion."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from rukh.cli import app
from rukh.data.manifest import Manifest
from rukh.data.uci import (
    UciConfig,
    clean_movetext,
    convert_month,
    convert_rows,
    game_id,
    run,
    san_to_uci,
)

pytestmark = pytest.mark.unit

LICHESS = (
    "1. e4 { [%eval 0.3] [%clk 0:05:00] } e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] } "
    "Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 "
    "1/2-1/2"
)
ILLEGAL = "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nxe4 4. Qh5 Qh4 1-0"  # 3...Nxe4: c6 knight cannot go there
PROMO = (
    "1. e4 d5 2. exd5 c6 3. dxc6 Nf6 4. cxb7 Bd7 5. bxa8=Q Nc6 6. Qxc6 Bxc6 7. d4 e5 "
    "8. dxe5 Qxd1+ 9. Kxd1 Nd7 10. f4 Nb6 1-0"
)
SHORT = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
CASTLES = (
    "1. d4 d5 2. Nf3 Nf6 3. Nc3 Nc6 4. Bf4 Bf5 5. Qd2 Qd7 6. O-O-O O-O-O 7. e3 e6 8. Bd3 Bd6 "
    "9. Ne5 Ne4 10. Bxe4 Bxe5 1-0"
)


def test_clean_movetext_strips_comments_numbers_and_result() -> None:
    cleaned = clean_movetext("1. e4 { [%clk 0:05:00] } e5 { [%eval 0.2] } 2. Nf3 $1 1... Nc6 1-0")
    assert cleaned == "e4 e5 Nf3 Nc6"
    assert clean_movetext("1. d4 (1. e4 e5) d5 *") == "d4 d5"
    assert clean_movetext("") == ""


def test_san_to_uci_castling_and_promotion() -> None:
    uci, n = san_to_uci(LICHESS)
    assert n == 20
    assert uci.split()[8] == "e1g1"  # 5. O-O
    assert uci.split()[15] == "e8g8"  # 8... O-O
    uci, _ = san_to_uci(PROMO)
    assert "b7a8q" in uci.split()
    uci, _ = san_to_uci(CASTLES)
    assert uci.split()[10] == "e1c1" and uci.split()[11] == "e8c8"
    annotated = san_to_uci("1. e4 e5 2. Nf3 Nc6 3. Bb5!? a6 1-0")
    assert annotated == ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", 6)


def test_san_to_uci_rejects_illegal_or_garbage() -> None:
    assert san_to_uci(ILLEGAL) is None
    assert san_to_uci("1. e4 e5 2. Zz9") is None
    assert san_to_uci("1. e5") is None


def test_game_id_is_stable_int64() -> None:
    value = game_id("https://lichess.org/abcd1234")
    assert value == game_id("https://lichess.org/abcd1234")
    assert -(2**63) <= value < 2**63
    assert value != game_id("https://lichess.org/abcd1235")


def _raw_rows() -> list[dict[str, object]]:
    def row(site: str, movetext: str, result: str) -> dict[str, object]:
        return {
            "Site": f"https://lichess.org/{site}",
            "movetext": movetext,
            "WhiteElo": 1900,
            "BlackElo": 1850,
            "Result": result,
            "TimeControl": "300+0",
            "UTCDate": "2025.01.01",
            "ECO": "C60",
            "month": "2025-01",
        }

    return [
        row("g1", LICHESS, "1/2-1/2"),
        row("g2", PROMO, "1-0"),
        row("g3", CASTLES, "1-0"),
        row("g4", ILLEGAL, "1-0"),
        row("g5", LICHESS.replace("1/2-1/2", "0-1"), "0-1"),
    ]


def test_convert_rows_counts() -> None:
    out = convert_rows(_raw_rows(), min_plies=20, max_plies=300)
    assert out["counts"] == {"rows": 5, "kept": 4, "illegal": 1, "short": 0, "long": 0}
    assert out["columns"]["n_plies"] == [20, 20, 20, 20]
    out = convert_rows(_raw_rows() + [_raw_rows()[0] | {"movetext": SHORT}], 20, 300)
    assert out["counts"]["short"] == 1
    out = convert_rows(_raw_rows(), min_plies=0, max_plies=10)
    assert out["counts"]["long"] == 4 and out["counts"]["kept"] == 0


def _write_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_raw_rows()).write_parquet(path.as_posix())


def test_convert_month_keeps_four_of_five(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    _write_raw(src)
    dst = tmp_path / "out" / "games.parquet"
    counts = convert_month(src, dst, UciConfig(workers=1))
    assert counts["rows"] == 5 and counts["kept"] == 4 and counts["illegal"] == 1
    frame = pl.read_parquet(dst.as_posix())
    assert frame.columns == [
        "game_id",
        "uci",
        "n_plies",
        "white_elo",
        "black_elo",
        "result",
        "time_control",
        "utc_date",
        "eco",
        "month",
    ]
    assert frame.schema["game_id"] == pl.Int64
    assert frame.schema["n_plies"] == pl.Int16
    assert frame.schema["white_elo"] == pl.Int16
    assert frame.schema["utc_date"] == pl.Date
    assert frame["result"].to_list() == ["1/2-1/2", "1-0", "1-0", "0-1"]
    assert frame["uci"][0].startswith("e2e4 e7e5 g1f3 b8c6 f1b5")
    assert str(frame["utc_date"][0]) == "2025-01-01"


def test_convert_month_with_two_workers(tmp_path: Path) -> None:
    """The spawn pool path: on Windows every worker re-imports the entry point module."""
    src = tmp_path / "raw.parquet"
    _write_raw(src)
    dst = tmp_path / "out" / "games.parquet"
    counts = convert_month(src, dst, UciConfig(workers=2))
    assert counts == {"rows": 5, "kept": 4, "illegal": 1, "short": 0, "long": 0}
    assert pl.read_parquet(dst.as_posix()).height == 4


def test_run_writes_months_and_manifest(rukh_home: Path) -> None:
    for month in ("01", "02"):
        _write_raw(rukh_home / "data" / "raw" / "year=2025" / f"month={month}" / "games.parquet")
    manifest = run(UciConfig(workers=1))
    assert manifest.months == ["2025-01", "2025-02"]
    assert manifest.counts == {"2025-01": 4, "2025-02": 4}
    assert manifest.filters["min_plies"] == 20 and manifest.filters["max_plies"] == 300
    assert manifest.filters["conversion"]["2025-01"]["illegal"] == 1
    assert [f.path for f in manifest.files] == [
        "year=2025/month=01/games.parquet",
        "year=2025/month=02/games.parquet",
    ]
    on_disk = Manifest.model_validate_json(
        (rukh_home / "data" / "uci" / "manifest.json").read_text("utf-8")
    )
    assert on_disk.counts == manifest.counts


def test_cli_data_uci(rukh_home: Path) -> None:
    _write_raw(rukh_home / "data" / "raw" / "year=2025" / "month=01" / "games.parquet")
    cfg = rukh_home / "pipeline.yaml"
    cfg.write_text("uci:\n  workers: 1\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["data", "uci", "--config", str(cfg), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["counts"] == {"2025-01": 4}
