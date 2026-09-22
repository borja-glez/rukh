"""Tests for the course links on every card, `rukh publish cards` and the Hub collection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rukh.cli import app
from rukh.data.publish import CARDS_DIR
from rukh.publish import cards, collection
from rukh.publish.links import course_links

pytestmark = pytest.mark.unit


def test_every_card_template_carries_the_course_section() -> None:
    templates = [p for p in CARDS_DIR.glob("*.md.jinja") if not p.name.startswith("_")]
    assert len(templates) >= 7
    for template in templates:
        text = template.read_text(encoding="utf-8")
        assert "{% include '_course.md.jinja' %}" in text, template.name
        # Before the licence: the links are part of the card, not an afterthought under it.
        assert text.index("_course.md.jinja") < text.index("## Licen"), template.name


def test_the_links_name_the_lesson_the_table_and_the_demo_when_they_apply() -> None:
    medium = course_links("chorcat/rukh-medium")
    assert (
        medium["lesson_url"] == "https://lab.rukh.borjaglez.com/curso/m2/11-mas-datos-no-mas-red/"
    )
    assert medium["lesson_title"].startswith("M2")
    assert medium["table_url"] == "https://lab.rukh.borjaglez.com/proyecto/"
    assert medium["stage"] == "medium-v4-greedy"
    assert medium["reproduce"] == "uv run rukh pull medium-v4"
    pairs = course_links("chorcat/rukh-pairs-dpo")
    assert pairs["lesson_url"].endswith("/curso/m1/10-labs-del-pipeline/")
    assert pairs["stage"] is None and pairs["demo_url"] is None
    # A repository the catalogue does not know gets the table and nothing it cannot back.
    unknown = course_links("chorcat/rukh-nope", course_url="https://x", demo_url="https://d")
    assert unknown == {
        "table_url": "https://x/proyecto/",
        "lesson_url": None,
        "lesson_title": None,
        "stage": None,
        "demo_url": None,
        "reproduce": None,
    }


def test_refresh_stages_every_card_and_uploads_only_the_readme(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged: list[str] = []

    def fake_stage(artefact, cfg):  # noqa: ANN001
        if artefact.name == "rm":
            raise RuntimeError("no run.json")
        path = rukh_home / "artifacts" / "publish" / artefact.repo_id / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {artefact.repo_id}\n", encoding="utf-8")
        staged.append(artefact.name)
        return path

    uploads: list[dict[str, object]] = []
    api = SimpleNamespace(upload_file=lambda **kw: uploads.append(kw))
    monkeypatch.setattr(cards, "stage_card", fake_stage)

    dry = cards.refresh_cards(only={"tiny", "rukh-pairs-dpo", "rm"}, dry_run=True, api=api)
    assert [o.status for o in dry] == ["staged", "failed", "staged"]
    assert uploads == []
    assert "no run.json" in (dry[1].detail or "")

    wet = cards.refresh_cards(only={"tiny", "rukh-pairs-dpo"}, dry_run=False, api=api)
    assert [o.status for o in wet] == ["uploaded", "uploaded"]
    assert [u["repo_id"] for u in uploads] == ["chorcat/rukh-tiny", "chorcat/rukh-pairs-dpo"]
    assert [u["repo_type"] for u in uploads] == ["model", "dataset"]
    assert all(u["path_in_repo"] == "README.md" for u in uploads)
    assert all(str(u["path_or_fileobj"]).endswith("README.md") for u in uploads)


def test_the_collection_lists_the_catalogue_once_in_course_order_and_is_idempotent() -> None:
    items = collection.collection_items()
    ids = [repo for repo, _ in items]
    assert ids[0] == "chorcat/rukh-tiny"
    assert "chorcat/rukh-medium-grpo" in ids and "chorcat/rukh-pairs-onpolicy" in ids
    assert len(ids) == len(set(ids))
    assert dict(items)["chorcat/rukh-tokenizer"] == "model"
    assert dict(items)["chorcat/rukh-games-1800"] == "dataset"

    calls: list[tuple[str, ...]] = []

    class FakeApi:
        def __init__(self, existing: bool) -> None:
            self.existing = existing

        def list_collections(self, owner: str):  # noqa: ANN202
            calls.append(("list", owner))
            return [SimpleNamespace(title="Rukh", slug="chorcat/rukh-abc")] if self.existing else []

        def create_collection(self, title, **kw):  # noqa: ANN001, ANN003
            calls.append(("create", title, kw["namespace"]))
            return SimpleNamespace(title=title, slug="chorcat/rukh-new")

        def add_collection_item(self, slug, item_id, item_type, **kw):  # noqa: ANN001, ANN003
            calls.append(("add", slug, item_id, item_type, str(kw.get("exists_ok"))))

    first = collection.sync_collection(api=FakeApi(existing=False))
    assert first.created and first.slug == "chorcat/rukh-new"
    assert ("create", "Rukh", "chorcat") in calls
    assert sum(1 for c in calls if c[0] == "add") == len(items)
    assert all(c[-1] == "True" for c in calls if c[0] == "add")
    calls.clear()
    second = collection.sync_collection(api=FakeApi(existing=True))
    assert not second.created and second.slug == "chorcat/rukh-abc"
    assert not any(c[0] == "create" for c in calls)
    dry = collection.sync_collection(dry_run=True)
    assert dry.slug is None and dry.items == ids


def test_the_cli_commands_run_dry(rukh_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cards, "stage_card", lambda artefact, cfg: rukh_home / f"{artefact.name}.md"
    )
    result = CliRunner().invoke(app, ["publish", "cards", "--dry-run", "--only", "tiny"])
    assert result.exit_code == 0, result.output
    assert "chorcat/rukh-tiny" in result.output and "staged" in result.output
    listed = CliRunner().invoke(app, ["publish", "collection", "--dry-run"])
    assert listed.exit_code == 0, listed.output
    assert "(dry run)" in listed.output and "chorcat/rukh-rm" in listed.output
