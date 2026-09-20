"""Publish the derived datasets and the tokenizer to the Hugging Face Hub under ``chorcat``.

``DATASETS`` is the registry: repo name, local directory, files, description, columns and
license. ``publish`` renders the card (Jinja, English, HF YAML metadata), creates the repo and
uploads the matching files in one commit with ``upload_folder`` (``upload_large_folder`` for
the two multi-gigabyte datasets, which resumes and uploads in parallel), then the card. With
``dry_run`` nothing touches the network: the card is written under
``data/publish/<name>/README.md`` and the files are listed. All ``HfApi`` calls go through
``_api`` so tests can replace it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict

from rukh.config import BaseConfig
from rukh.data.manifest import Manifest
from rukh.paths import resolve

CARDS_DIR = Path(__file__).resolve().parent / "cards"
LICHESS_CC0 = (
    "CC0 1.0. The games, puzzles and evaluations come from the Lichess open database "
    "(https://database.lichess.org, https://huggingface.co/Lichess), published under CC0; "
    "the derived files keep that license. Please credit Lichess when you use them."
)
ELITE_LICENSE = (
    "CC0 1.0 de facto. The games come from Lichess (CC0) through the Lichess Elite Database "
    "curated by nikonoel (https://database.nikonoel.fr), which publishes no explicit license; "
    "credit both Lichess and the Elite Database."
)
UCI_COLUMNS: list[tuple[str, str]] = [
    ("game_id", "signed 64-bit hash of the Lichess `Site` URL (first 8 bytes of SHA-256)"),
    ("uci", "space-separated UCI moves of the whole game, verified legal with python-chess"),
    ("n_plies", "number of half-moves"),
    ("white_elo", "White's Glicko-2 rating at game time"),
    ("black_elo", "Black's Glicko-2 rating at game time"),
    ("result", "`1-0`, `0-1` or `1/2-1/2`"),
    ("time_control", "Lichess `base+increment` in seconds"),
    ("utc_date", "date the game was played (UTC)"),
    ("eco", "ECO opening code"),
    ("month", "`YYYY-MM` partition the game came from"),
]


class DatasetSpec(BaseModel):
    """One publishable artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str
    repo_type: Literal["dataset", "model"]
    local_dir: str
    patterns: list[str]
    command: str
    pretty_name: str
    description: str
    columns: list[tuple[str, str]]
    license: str = "cc0-1.0"
    license_text: str = LICHESS_CC0
    source: str
    source_datasets: list[str]
    task_categories: list[str]
    tags: list[str] = []
    exclude_counts: list[str] = []
    large: bool = False
    """Multi-gigabyte: upload with ``upload_large_folder`` when the Hub client has it."""


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        DatasetSpec(
            name="rukh-games-1800",
            repo_type="dataset",
            large=True,
            local_dir="data/uci",
            patterns=["year=*/month=*/games.parquet", "manifest.json"],
            command="fetch` and `rukh data uci",
            pretty_name="Rukh games 1800+",
            description=(
                "Rated standard Lichess games with both players at 1800+ Elo, base time of "
                "at least 180 seconds, normal or time-forfeit terminations, 20 to 300 plies, "
                "converted from SAN to legal UCI. Partitioned by month: train on 2025-01, "
                "validate on 2025-02."
            ),
            columns=UCI_COLUMNS,
            source="`Lichess/standard-chess-games` on the Hugging Face Hub",
            source_datasets=["original"],
            task_categories=["text-generation"],
            tags=["games", "uci"],
        ),
        DatasetSpec(
            name="rukh-games-elite",
            repo_type="dataset",
            local_dir="data/elite",
            patterns=["games.parquet", "manifest.json"],
            command="elite",
            pretty_name="Rukh Elite games",
            description=(
                "Games from the Lichess Elite Database (2500+ against 2300+, no bullet), "
                "converted to legal UCI with the same schema as `rukh-games-1800`. Used for "
                "supervised fine-tuning on strong play."
            ),
            columns=UCI_COLUMNS,
            license_text=ELITE_LICENSE,
            source="Lichess Elite Database monthly zips (database.nikonoel.fr)",
            source_datasets=["original"],
            task_categories=["text-generation"],
            tags=["games", "uci", "elite"],
        ),
        DatasetSpec(
            name="rukh-elo-bins",
            repo_type="dataset",
            large=True,
            local_dir="data/elo-bins",
            patterns=["games.parquet", "manifest.json"],
            command="elo-bins",
            pretty_name="Rukh Elo-balanced games",
            description=(
                "A balanced sample of `rukh-games-1800`: up to `n_per_bin` games per 100-Elo "
                "bin of the average rating of both players, for Elo conditioning experiments."
            ),
            columns=UCI_COLUMNS + [("bin", "lower edge of the 100-Elo bin of the average rating")],
            source="`rukh-games-1800` (from `Lichess/standard-chess-games`)",
            source_datasets=["original"],
            task_categories=["text-generation"],
            tags=["games", "uci", "elo"],
        ),
        DatasetSpec(
            name="rukh-positions-eval",
            repo_type="dataset",
            local_dir="data/evals",
            patterns=["positions-eval.parquet", "manifest.json"],
            command="positions` and `rukh data evals",
            pretty_name="Rukh positions with Stockfish evaluations",
            description=(
                "Distinct positions sampled from `rukh-games-1800`, keyed by the four-field FEN "
                "and joined with `Lichess/chess-position-evaluations`. One row per position "
                "with the best line and every analysed line, for value heads, reward models "
                "and preference pairs without running an engine."
            ),
            columns=[
                ("fen", "pieces, side to move, castling rights and en-passant square"),
                ("game_id", "game of the first occurrence"),
                ("ply", "half-move index of the first occurrence"),
                ("last_move", "UCI move that reached the position"),
                ("result", "result of that game"),
                ("phase", "`opening` (ply <= 10), `middlegame` (14+ pieces) or `endgame`"),
                ("n_seen", "occurrences in the sampled games"),
                ("best_move", "first move of the best line for the side to move"),
                ("cp", "centipawns of the best line, from White's point of view"),
                ("mate", "moves to mate of the best line (positive: White mates)"),
                ("depth", "search depth of the best line"),
                ("knodes", "kilonodes searched"),
                ("pvs", "every line as `{move, cp, mate, depth}`, best first"),
            ],
            source=(
                "`Lichess/standard-chess-games` (positions) and "
                "`Lichess/chess-position-evaluations` (evaluations)"
            ),
            source_datasets=["original"],
            task_categories=["tabular-regression", "text-classification"],
            tags=["positions", "stockfish"],
            exclude_counts=["lines", "positions"],
        ),
        DatasetSpec(
            name="rukh-puzzles-split",
            repo_type="dataset",
            local_dir="data/puzzles",
            patterns=["puzzles.parquet", "manifest.json"],
            command="puzzles",
            pretty_name="Rukh puzzle splits",
            description=(
                "Lichess puzzles with rating deviation <= 100 and at least 100 plays, banded by "
                "difficulty (1000-1500, 1500-2000, 2000+) and split into test and train by a "
                "seeded hash of the puzzle id, each with the moves of the game it came from, "
                "for tactical evaluation and fine-tuning."
            ),
            columns=[
                ("puzzle_id", "Lichess puzzle id"),
                ("fen", "position before the opponent's last move (Lichess convention)"),
                ("moves", "UCI moves: the opponent's move first, then the solution"),
                ("rating", "puzzle rating"),
                ("themes", "Lichess theme tags"),
                ("band", "`1000-1500`, `1500-2000` or `2000+`"),
                ("split", "`test` or `train`"),
                ("prefix_uci", "the real game's UCI moves up to `fen`, space-separated"),
                ("prefix_plies", "number of moves in `prefix_uci`"),
                ("white_elo", "rating of White in that game"),
                ("black_elo", "rating of Black in that game"),
            ],
            source="`Lichess/chess-puzzles-with-games` on the Hugging Face Hub",
            source_datasets=["original"],
            task_categories=["text-generation", "question-answering"],
            tags=["puzzles"],
        ),
        DatasetSpec(
            name="rukh-pairs-dpo",
            repo_type="dataset",
            local_dir="data/pairs",
            patterns=["pairs.parquet", "manifest.json"],
            command="pairs",
            pretty_name="Rukh DPO pairs",
            description=(
                "Preference pairs from the multi-PV Stockfish lines of `rukh-positions-eval`: "
                "the best first move against a legal move at least 100 centipawns worse for the "
                "side to move (mates count as 10000). One pair per position, balanced by phase."
            ),
            columns=[
                ("fen", "four-field FEN of the position"),
                ("chosen", "best move (UCI)"),
                ("rejected", "clearly worse legal move (UCI)"),
                ("cp_chosen", "score of `chosen` from White's point of view (mate = +/-10000)"),
                ("cp_rejected", "score of `rejected` from White's point of view"),
                ("phase", "`opening`, `middlegame` or `endgame`"),
            ],
            source="`rukh-positions-eval` (from `Lichess/chess-position-evaluations`)",
            source_datasets=["original"],
            task_categories=["text-generation"],
            tags=["dpo", "preferences"],
            exclude_counts=["candidates"],
        ),
        DatasetSpec(
            name="rukh-pairs-onpolicy",
            repo_type="dataset",
            local_dir="data/pairs-onpolicy",
            patterns=["pairs.parquet", "manifest.json"],
            command="onpolicy",
            pretty_name="Rukh on-policy pairs",
            description=(
                "Preference pairs built from the moves the model itself proposes. The positions "
                "are exactly those of `rukh-pairs-dpo`, so the only thing that differs between the "
                "two datasets is where the two moves came from -- which is what makes a DPO run on "
                "each of them a comparison. Four moves sampled per position at temperature 1.0, "
                "scored by Stockfish at a fixed depth of 10, keeping the best and the worst when "
                "they are at least 100 centipawns apart.\n\n"
                "Of the 13 838 positions, 6 386 produced a pair. The rest is the interesting half: "
                "3 743 (27 %) because the model proposed a single distinct legal move, and 3 709 "
                "(27 %) because its best and worst candidate were less than a pawn apart. And only "
                "2.9 % of these pairs involve a mate, against a third of the off-policy ones: a "
                "multi-PV search finds mates the model was never going to propose, so the two "
                "datasets are two distributions of difficulty and not two sources of one thing."
            ),
            columns=[
                ("game_id", "game the position came from, for splitting without leakage"),
                ("ply", "half-move index of the position"),
                ("prefix", "moves played up to the position (UCI, space separated)"),
                ("chosen", "best of the sampled moves (UCI)"),
                ("rejected", "worst of the sampled moves (UCI)"),
                ("cp_chosen", "score of `chosen` for the side to move (mate = +/-10000)"),
                ("cp_rejected", "score of `rejected` for the side to move"),
                ("phase", "`opening`, `middlegame` or `endgame`"),
                ("white_elo", "Elo header the model was prompted with"),
                ("black_elo", "Elo header the model was prompted with"),
                ("candidates", "distinct legal moves the model proposed here (2 to 4)"),
            ],
            source="`rukh-pairs-dpo` for the positions, `chorcat/rukh-medium` for the moves",
            source_datasets=["original"],
            task_categories=["text-generation"],
            tags=["dpo", "preferences", "on-policy"],
            # `positions` counts what was *visited*, not what was written: 13 838 against 6 386
            # rows. Summing it into the row count would double the size of the dataset on the card.
            exclude_counts=["positions"],
        ),
        DatasetSpec(
            name="rukh-tokenizer",
            repo_type="model",
            local_dir="artifacts/tokenizer",
            patterns=["vocab.json", "bpe.json", "fixtures/games.json", "README.md"],
            command="tokenize --export-fixture",
            pretty_name="Rukh tokenizer",
            description=(
                "Fixed UCI vocabulary, BPE and char-level SAN tokenizers with a parity fixture."
            ),
            columns=[],
            source="enumeration plus a BPE trained on `rukh-games-1800`",
            source_datasets=["original"],
            task_categories=[],
            tags=["tokenizer"],
        ),
    ]
}


class PublishConfig(BaseConfig):
    """Hub owner and where dry runs stage their output."""

    owner: str = "chorcat"
    publish_dir: str = "data/publish"


class PublishResult(BaseModel):
    """What was (or would be) uploaded."""

    model_config = ConfigDict(extra="forbid")

    name: str
    repo_id: str
    repo_type: str
    dry_run: bool
    card_path: str
    files: list[str]
    rows: int


def size_category(rows: int) -> str:
    for limit, label in (
        (1_000, "n<1K"),
        (10_000, "1K<n<10K"),
        (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"),
        (10_000_000, "1M<n<10M"),
    ):
        if rows < limit:
            return label
    return "10M<n<100M"


def local_files(spec: DatasetSpec) -> list[Path]:
    root = resolve(spec.local_dir)
    found: list[Path] = []
    for pattern in spec.patterns:
        found.extend(sorted(root.glob(pattern)))
    return found


def read_manifest(spec: DatasetSpec) -> Manifest | None:
    path = resolve(spec.local_dir) / "manifest.json"
    if not path.is_file():
        return None
    return Manifest.model_validate_json(path.read_text("utf-8"))


def count_rows(spec: DatasetSpec, manifest: Manifest | None) -> int:
    if manifest is None:
        return 0
    return sum(v for k, v in manifest.counts.items() if k not in spec.exclude_counts)


def render_card(spec: DatasetSpec, manifest: Manifest | None, owner: str) -> str:
    """Render the dataset card; the tokenizer uses its own README as the card."""
    if spec.repo_type == "model":
        return (resolve(spec.local_dir) / "README.md").read_text("utf-8")
    env = Environment(
        loader=FileSystemLoader(str(CARDS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    rows = count_rows(spec, manifest)
    context = {
        "repo_id": f"{owner}/{spec.name}",
        "license": spec.license,
        "license_text": spec.license_text,
        "pretty_name": spec.pretty_name,
        "task_categories": spec.task_categories,
        "size_category": size_category(rows),
        "source_datasets": spec.source_datasets,
        "tags": spec.tags,
        "description": spec.description,
        "command": spec.command,
        "columns": spec.columns,
        "source": spec.source,
        "rows": rows,
        "files": manifest.files if manifest else [],
        "months": manifest.months if manifest else [],
        "filters_json": json.dumps(manifest.filters if manifest else {}, indent=2, default=str),
        "counts_json": json.dumps(manifest.counts if manifest else {}, indent=2),
        "created_at": manifest.created_at.date().isoformat() if manifest else "unknown",
        "rukh_version": manifest.rukh_version if manifest else "unknown",
    }
    return env.get_template("dataset.md.jinja").render(**context)


def _api():  # type: ignore[no-untyped-def]
    """The ``HfApi`` client (tests monkeypatch this)."""
    from huggingface_hub import HfApi

    return HfApi()


def upload_folder(api: object, spec: DatasetSpec, repo_id: str) -> str:
    """Upload the whole local directory in one go; returns the method that was used.

    The card is always uploaded separately from the staged copy, so ``README.md`` is excluded
    here. A ``large`` dataset uses ``upload_large_folder`` (resumable, multi-threaded) when the
    installed ``huggingface_hub`` provides it.
    """
    large = callable(getattr(api, "upload_large_folder", None))
    if spec.large and large:
        api.upload_large_folder(  # type: ignore[attr-defined]
            repo_id=repo_id,
            folder_path=str(resolve(spec.local_dir)),
            repo_type=spec.repo_type,
            allow_patterns=spec.patterns,
            ignore_patterns=["README.md"],
        )
        return "upload_large_folder"
    api.upload_folder(  # type: ignore[attr-defined]
        repo_id=repo_id,
        folder_path=str(resolve(spec.local_dir)),
        repo_type=spec.repo_type,
        allow_patterns=spec.patterns,
        ignore_patterns=["README.md"],
        commit_message="Upload data",
    )
    return "upload_folder"


def publish(name: str, cfg: PublishConfig, dry_run: bool = False) -> PublishResult:
    """Render the card and upload (or, with ``dry_run``, stage) one registry entry."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    manifest = read_manifest(spec)
    files = local_files(spec)
    if not files:
        raise FileNotFoundError(f"nothing to publish under {resolve(spec.local_dir)}")
    repo_id = f"{cfg.owner}/{spec.name}"
    card = render_card(spec, manifest, cfg.owner)
    stage = resolve(cfg.publish_dir) / spec.name
    stage.mkdir(parents=True, exist_ok=True)
    card_path = stage / "README.md"
    card_path.write_text(card, encoding="utf-8", newline="\n")
    rel_files = [p.relative_to(resolve(spec.local_dir)).as_posix() for p in files]
    if not dry_run:
        api = _api()
        api.create_repo(repo_id, repo_type=spec.repo_type, exist_ok=True)
        upload_folder(api, spec, repo_id)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=spec.repo_type,
            commit_message="Update card",
        )
    return PublishResult(
        name=spec.name,
        repo_id=repo_id,
        repo_type=spec.repo_type,
        dry_run=dry_run,
        card_path=card_path.as_posix(),
        files=rel_files,
        rows=count_rows(spec, manifest),
    )
