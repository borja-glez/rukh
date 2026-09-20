"""Tests for rukh.data.publish: registry, card rendering and network-free dry runs."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rukh.cli import app
from rukh.data import publish as publish_module
from rukh.data.manifest import FileHash, Manifest
from rukh.data.publish import (
    DATASETS,
    PublishConfig,
    publish,
    render_card,
    size_category,
)

pytestmark = pytest.mark.unit

EXPECTED = {
    "rukh-games-1800",
    "rukh-games-elite",
    "rukh-elo-bins",
    "rukh-positions-eval",
    "rukh-puzzles-split",
    "rukh-pairs-dpo",
    "rukh-pairs-onpolicy",
    "rukh-tokenizer",
}


def test_registry_covers_seven_datasets_and_tokenizer() -> None:
    assert set(DATASETS) == EXPECTED
    assert DATASETS["rukh-tokenizer"].repo_type == "model"
    assert all(s.repo_type == "dataset" for n, s in DATASETS.items() if n != "rukh-tokenizer")
    for spec in DATASETS.values():
        assert spec.license == "cc0-1.0"
        assert "manifest.json" in spec.patterns or spec.repo_type == "model"
        # The Hub card spec only accepts `original` and `extended|<id>` here.
        assert spec.source_datasets == ["original"]
    assert {n for n, s in DATASETS.items() if s.large} == {"rukh-games-1800", "rukh-elo-bins"}


def test_size_category() -> None:
    assert size_category(0) == "n<1K"
    assert size_category(5_000) == "1K<n<10K"
    assert size_category(250_000) == "100K<n<1M"
    assert size_category(3_000_000) == "1M<n<10M"


def _manifest(counts: dict[str, int]) -> Manifest:
    return Manifest(
        dataset="Lichess/standard-chess-games",
        months=["2025-01", "2025-02"],
        filters={"min_elo": 1800, "terminations": ["Normal", "Time forfeit"]},
        counts=counts,
        files=[FileHash(path="games.parquet", sha256="ab" * 32, bytes=123)],
        created_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )


def _fail_api():  # type: ignore[no-untyped-def]
    raise AssertionError("HfApi must not be used in a dry run")


def test_render_card_has_mandatory_metadata() -> None:
    card = render_card(DATASETS["rukh-games-1800"], _manifest({"2025-01": 4, "2025-02": 6}), "x")
    head, body = card.split("---\n", 2)[1:]
    assert "license: cc0-1.0" in head
    assert "task_categories:\n  - text-generation" in head
    assert "size_categories:\n  - n<1K" in head
    assert "source_datasets:\n  - original" in head
    assert "pretty_name: Rukh games 1800+" in head
    assert "# x/rukh-games-1800" in body
    assert "Rows: **10**" in body
    assert "| `game_id` |" in body and "| `uci` |" in body
    assert '"min_elo": 1800' in body
    assert "months 2025-01, 2025-02" in body
    assert "2026-09-18" in body
    assert "Lichess" in body


def test_render_card_excludes_helper_counts() -> None:
    card = render_card(
        DATASETS["rukh-pairs-dpo"],
        _manifest({"opening": 2, "middlegame": 2, "endgame": 2, "candidates": 99}),
        "chorcat",
    )
    assert "Rows: **6**" in card


def test_dry_run_writes_card_without_network(
    rukh_home: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publish_module, "_api", _fail_api)
    out = rukh_home / "data" / "pairs"
    out.mkdir(parents=True)
    shutil.copy(repo_root / "tests" / "fixtures" / "games.parquet", out / "pairs.parquet")
    (out / "manifest.json").write_text(
        _manifest({"opening": 1, "middlegame": 1, "endgame": 1}).model_dump_json(), "utf-8"
    )
    result = publish("rukh-pairs-dpo", PublishConfig(), dry_run=True)
    assert result.dry_run and result.repo_id == "chorcat/rukh-pairs-dpo"
    assert result.files == ["pairs.parquet", "manifest.json"]
    assert result.rows == 3
    card = rukh_home / "data" / "publish" / "rukh-pairs-dpo" / "README.md"
    assert card.is_file() and card.as_posix() == result.card_path
    assert "chosen" in card.read_text("utf-8")


def test_dry_run_tokenizer_uses_artifact_readme(
    rukh_home: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publish_module, "_api", _fail_api)
    shutil.copytree(repo_root / "artifacts" / "tokenizer", rukh_home / "artifacts" / "tokenizer")
    result = publish("rukh-tokenizer", PublishConfig(), dry_run=True)
    assert result.repo_type == "model"
    assert set(result.files) == {"vocab.json", "bpe.json", "fixtures/games.json", "README.md"}
    card = Path(result.card_path).read_text("utf-8")
    assert card.startswith("---\nlicense: cc0-1.0")
    assert "## Scheme 1: fixed UCI vocabulary" in card


class FakeApi:
    """Records the calls ``publish`` makes; ``calls`` is shared by every instance."""

    calls: list[tuple[str, dict[str, object]]] = []
    has_large: bool = True

    def __init__(self) -> None:
        if not FakeApi.has_large:
            # A huggingface_hub without upload_large_folder must still work.
            self.upload_large_folder = None  # type: ignore[assignment]

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        FakeApi.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_file(self, **kwargs: object) -> None:
        FakeApi.calls.append(("upload_file", kwargs))

    def upload_folder(self, **kwargs: object) -> None:
        FakeApi.calls.append(("upload_folder", kwargs))

    def upload_large_folder(self, **kwargs: object) -> None:
        FakeApi.calls.append(("upload_large_folder", kwargs))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeApi]:
    FakeApi.calls = []
    FakeApi.has_large = True
    monkeypatch.setattr(publish_module, "_api", FakeApi)
    return FakeApi


def test_publish_uploads_the_folder_in_one_commit(
    rukh_home: Path, repo_root: Path, fake_api: type[FakeApi]
) -> None:
    shutil.copytree(repo_root / "artifacts" / "tokenizer", rukh_home / "artifacts" / "tokenizer")
    result = publish("rukh-tokenizer", PublishConfig(owner="someone"), dry_run=False)
    assert not result.dry_run
    assert fake_api.calls[0] == (
        "create_repo",
        {"repo_id": "someone/rukh-tokenizer", "repo_type": "model", "exist_ok": True},
    )
    assert [c[0] for c in fake_api.calls] == ["create_repo", "upload_folder", "upload_file"]
    folder = fake_api.calls[1][1]
    assert folder["folder_path"] == str(rukh_home / "artifacts" / "tokenizer")
    assert folder["allow_patterns"] == DATASETS["rukh-tokenizer"].patterns
    assert folder["ignore_patterns"] == ["README.md"]
    assert folder["repo_type"] == "model"
    assert fake_api.calls[2][1]["path_in_repo"] == "README.md"


def test_publish_uses_upload_large_folder_for_the_big_datasets(
    rukh_home: Path, repo_root: Path, fake_api: type[FakeApi]
) -> None:
    out = rukh_home / "data" / "elo-bins"
    out.mkdir(parents=True)
    shutil.copy(repo_root / "tests" / "fixtures" / "games.parquet", out / "games.parquet")
    (out / "manifest.json").write_text(_manifest({"1800": 4}).model_dump_json(), "utf-8")
    publish("rukh-elo-bins", PublishConfig(), dry_run=False)
    assert [c[0] for c in fake_api.calls] == ["create_repo", "upload_large_folder", "upload_file"]
    assert "commit_message" not in fake_api.calls[1][1]

    # Without upload_large_folder in huggingface_hub it falls back to upload_folder.
    FakeApi.calls = []
    FakeApi.has_large = False
    publish("rukh-elo-bins", PublishConfig(), dry_run=False)
    assert [c[0] for c in fake_api.calls] == ["create_repo", "upload_folder", "upload_file"]


def test_publish_unknown_or_missing(rukh_home: Path) -> None:
    with pytest.raises(KeyError):
        publish("nope", PublishConfig(), dry_run=True)
    with pytest.raises(FileNotFoundError):
        publish("rukh-elo-bins", PublishConfig(), dry_run=True)


def test_cli_publish_dry_run(
    rukh_home: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publish_module, "_api", _fail_api)
    shutil.copytree(repo_root / "artifacts" / "tokenizer", rukh_home / "artifacts" / "tokenizer")
    result = CliRunner().invoke(
        app, ["data", "publish", "--name", "rukh-tokenizer", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo_id"] == "chorcat/rukh-tokenizer" and payload["dry_run"] is True
    listed = CliRunner().invoke(app, ["data", "publish", "--list"])
    assert listed.exit_code == 0
    assert "rukh-pairs-dpo" in listed.output
