"""``rukh publish collection``: the ``Rukh`` collection on the Hub, kept equal to the catalogue.

A collection is the one page that lists everything the course trained, in the order the course
built it. It is derived from ``rukh.hub`` so it cannot drift from what ``rukh pull`` can bring
back: adding a model to the catalogue and running this is what "publishing the collection" is.
Idempotent on purpose -- the Hub refuses duplicates when asked (``exists_ok``), and a second run
changes nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from rukh.hub import catalogue
from rukh.publish.model import ModelPublishConfig, _api

log = logging.getLogger(__name__)

TITLE = "Rukh"
DESCRIPTION = (
    "A chess language model built from scratch as a course on generative and agentic AI: every "
    "model, adapter and dataset, in course order."
)
"""The Hub caps a collection's description at 150 characters."""


class CollectionSync(BaseModel):
    """What the collection ended up holding."""

    model_config = ConfigDict(extra="forbid")

    slug: str | None
    created: bool
    items: list[str]
    dry_run: bool


def collection_items() -> list[tuple[str, str]]:
    """``(repo id, item type)`` for every catalogued artefact, models first, in course order."""
    items: list[tuple[str, str]] = []
    for artefact in catalogue():
        item_type = artefact.hub_repo_type
        if (artefact.repo_id, item_type) not in items:
            items.append((artefact.repo_id, item_type))
    return items


def sync_collection(
    cfg: ModelPublishConfig | None = None, dry_run: bool = False, api: Any | None = None
) -> CollectionSync:
    """Create the collection if it does not exist and add every catalogued repository to it."""
    cfg = cfg or ModelPublishConfig()
    items = collection_items()
    if dry_run:
        return CollectionSync(slug=None, created=False, items=[r for r, _ in items], dry_run=True)
    client = api or _api()
    existing = next((c for c in client.list_collections(owner=cfg.owner) if c.title == TITLE), None)
    created = existing is None
    collection = existing or client.create_collection(
        TITLE, namespace=cfg.owner, description=DESCRIPTION, private=False, exists_ok=True
    )
    for repo_id, item_type in items:
        client.add_collection_item(collection.slug, repo_id, item_type, exists_ok=True)
        log.info("collection %s: %s (%s)", collection.slug, repo_id, item_type)
    return CollectionSync(
        slug=collection.slug, created=created, items=[r for r, _ in items], dry_run=False
    )
