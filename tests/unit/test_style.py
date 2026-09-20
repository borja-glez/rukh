"""Tests for rukh.data.style: the game slices a LoRA adapter is trained to sound like."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pydantic
import pytest

from rukh.data.style import StyleConfig, StyleSpec, build_styles, predicate_of

pytestmark = pytest.mark.unit


def _stage(rukh_home: Path) -> None:
    """One month with three openings and two rating levels, all replayable."""
    rows = [
        ("e2e4 e7e5 g1f3", "C40", 2000),
        ("e2e4 c7c5 g1f3", "B20", 2100),
        ("e2e4 e7e6 d2d4", "C00", 1850),
        ("d2d4 d7d5 c2c4", "D06", 2000),
        ("d2d4 g8f6 c2c4", "E00", 1900),
        ("c2c4 e7e5", "A20", 2200),
    ]
    frame = pl.DataFrame(
        {
            "game_id": list(range(len(rows))),
            "uci": [uci for uci, _, _ in rows],
            "n_plies": [len(uci.split()) for uci, _, _ in rows],
            "white_elo": [elo for _, _, elo in rows],
            "black_elo": [elo for _, _, elo in rows],
            "result": ["1-0"] * len(rows),
            "time_control": ["300+0"] * len(rows),
            "eco": [eco for _, eco, _ in rows],
            "month": ["2025-01"] * len(rows),
        }
    )
    target = rukh_home / "data" / "uci" / "year=2025" / "month=01" / "games.parquet"
    target.parent.mkdir(parents=True)
    frame.write_parquet(target.as_posix())


def _read(rukh_home: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet((rukh_home / "data" / "style" / f"{name}.parquet").as_posix())


def test_the_first_move_slice_only_holds_games_that_open_with_it(rukh_home: Path) -> None:
    """The axis the demo shows: swap the adapter, the model opens differently."""
    _stage(rukh_home)
    manifests = build_styles(
        StyleConfig(styles=[StyleSpec(name="e4", first_move="e2e4", n_games=10)])
    )
    frame = _read(rukh_home, "e4")
    assert frame.height == 3
    assert all(uci.startswith("e2e4 ") for uci in frame["uci"])
    assert manifests["e4"].counts == {"e4": 3}


def test_a_first_move_prefix_does_not_leak_into_another_move(rukh_home: Path) -> None:
    """`d2d4` must not also collect `d2d4` games that began `d2d3`; the match is on the token."""
    _stage(rukh_home)
    build_styles(StyleConfig(styles=[StyleSpec(name="d4", first_move="d2d4", n_games=10)]))
    frame = _read(rukh_home, "d4")
    assert sorted(frame["eco"].to_list()) == ["D06", "E00"]


def test_eco_prefixes_select_a_family_of_openings(rukh_home: Path) -> None:
    _stage(rukh_home)
    build_styles(StyleConfig(styles=[StyleSpec(name="sicilian", eco_prefix=["B2"], n_games=10)]))
    assert _read(rukh_home, "sicilian")["eco"].to_list() == ["B20"]


def test_a_rating_floor_keeps_a_style_adapter_from_being_a_weakness_adapter(
    rukh_home: Path,
) -> None:
    _stage(rukh_home)
    build_styles(
        StyleConfig(styles=[StyleSpec(name="e4", first_move="e2e4", min_elo=1900, n_games=10)])
    )
    assert _read(rukh_home, "e4").height == 2  # the 1850 game is out


def test_the_cap_limits_how_many_games_a_style_gets(rukh_home: Path) -> None:
    _stage(rukh_home)
    manifests = build_styles(
        StyleConfig(styles=[StyleSpec(name="e4", first_move="e2e4", n_games=2)])
    )
    assert manifests["e4"].counts == {"e4": 2}


def test_the_manifest_records_the_predicate_a_published_adapter_has_to_state(
    rukh_home: Path,
) -> None:
    _stage(rukh_home)
    manifests = build_styles(
        StyleConfig(styles=[StyleSpec(name="e4", first_move="e2e4", n_games=10)])
    )
    filters = manifests["e4"].filters
    assert "split_part(uci, ' ', 1) = 'e2e4'" in str(filters["predicate"])
    assert filters["style"]["first_move"] == "e2e4"  # type: ignore[index]


def test_several_styles_are_cut_in_one_pass(rukh_home: Path) -> None:
    _stage(rukh_home)
    manifests = build_styles(
        StyleConfig(
            styles=[
                StyleSpec(name="e4", first_move="e2e4", n_games=10),
                StyleSpec(name="d4", first_move="d2d4", n_games=10),
            ]
        )
    )
    assert set(manifests) == {"e4", "d4"}
    assert (rukh_home / "data" / "style" / "d4.manifest.json").is_file()


def test_the_slice_is_deterministic_for_a_seed(rukh_home: Path) -> None:
    _stage(rukh_home)
    spec = StyleSpec(name="e4", first_move="e2e4", n_games=2)
    build_styles(StyleConfig(styles=[spec], seed=1))
    first = _read(rukh_home, "e4")
    build_styles(StyleConfig(styles=[spec], seed=1))
    assert _read(rukh_home, "e4").equals(first)


def test_a_style_that_selects_nothing_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="selects nothing"):
        StyleSpec(name="empty")


def test_a_first_move_that_is_not_a_uci_move_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="not a UCI move"):
        StyleSpec(name="bad", first_move="e4")


def test_duplicate_style_names_are_refused() -> None:
    with pytest.raises(pydantic.ValidationError, match="duplicate style names"):
        StyleConfig(
            styles=[
                StyleSpec(name="e4", first_move="e2e4"),
                StyleSpec(name="e4", first_move="d2d4"),
            ]
        )


def test_a_quote_in_a_field_cannot_break_out_of_the_predicate() -> None:
    """Typed fields are the reason the config cannot carry SQL; the escaping is the backstop."""
    spec = StyleSpec(name="odd", eco_prefix=["B'2"])
    assert "'B''2%'" in predicate_of(spec)


def test_no_styles_is_refused(rukh_home: Path) -> None:
    with pytest.raises(ValueError, match="no styles configured"):
        build_styles(StyleConfig())
