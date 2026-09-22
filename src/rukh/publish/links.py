"""The links every card carries back to the course: its lesson, its row in the table, the demo.

A model card that does not say where its number came from is a number without a method. The
single results table (`/proyecto/` on the course site) is where every stage is measured the same
way, and the lesson is where it was built; both are looked up in the catalogue of ``rukh.hub`` so
a repo id is enough and nothing is typed twice.
"""

from __future__ import annotations

from typing import Any

from rukh.hub import catalogue

TABLE_PATH = "/proyecto/"

LESSON_TITLES: dict[str, str] = {
    "/curso/m1/10-labs-del-pipeline/": "M1 · Los labs del pipeline",
    "/curso/m2/03-la-receta-de-entrenamiento/": "M2 · La receta de entrenamiento",
    "/curso/m2/11-mas-datos-no-mas-red/": "M2 · Más datos, no más red: de small a medium-v4",
    "/curso/m3/11-labs-del-encoder/": "M3 · El encoder: los labs",
    "/curso/m4/10-labs-de-afinado/": "M4 · Los labs del afinado",
    "/curso/m5/09-labs-de-alineamiento/": "M5 · Alineamiento: los labs",
}


def course_links(
    repo_id: str,
    course_url: str = "https://lab.rukh.borjaglez.com",
    demo_url: str = "https://rukh.borjaglez.com",
) -> dict[str, Any]:
    """What the card's "In the course" section needs for ``repo_id``.

    ``lesson_url`` and ``stage`` are ``None`` for a repository the catalogue does not know, and
    the section then names only the table: a card never claims a lesson it was not built in.
    """
    found = next((a for a in catalogue() if a.repo_id == repo_id), None)
    lesson = found.lesson if found else None
    from rukh.publish.model import DEMO_STAGES

    name = repo_id.split("/")[-1].removeprefix("rukh-")
    demo_stage = DEMO_STAGES.get(name)
    return {
        "table_url": f"{course_url}{TABLE_PATH}",
        "lesson_url": f"{course_url}{lesson}" if lesson else None,
        "lesson_title": LESSON_TITLES.get(lesson or "", lesson),
        "stage": found.stage if found else None,
        "demo_url": f"{demo_url}/?stage={demo_stage}" if demo_stage else None,
        "reproduce": f"uv run rukh pull {found.name}" if found else None,
    }
