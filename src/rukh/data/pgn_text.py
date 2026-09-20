"""Games as plain text, the way a general language model would be shown them.

The comparison M4 owes the reader is: a 115 M decoder built for this one job against a 0.6 B
general model fine-tuned on the same games. For that to mean anything the *data* has to be the
same games and the *metric* has to be the same harness; only the representation may differ, and
here it differs as much as it can. The decoder reads a fixed vocabulary where one token is one
legal move. Qwen reads the movetext a human would paste into a chess site -- `1. e4 e5 2. Nf3` --
tokenised into whatever pieces its BPE happens to cut it into, with nothing stopping it from
writing `Nf9` or a sentence.

Two consequences are the lesson, not an accident of the setup:

- The move numbers and the SAN spelling are tokens the model has to spend capacity predicting,
  and they carry no chess. The specialist never sees them.
- SAN is *contextual*: `Nf3` names a different move depending on where the knights are, and a
  legal-looking string can be illegal or ambiguous. That is why the evaluation counts moves it
  could not parse as its own number rather than folding them into "illegal".

The header keeps the two Elo tags because that is exactly what the decoder's `<wXXXX> <bXXXX>`
tokens are, so the conditioning question can be asked of both models in their own language.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chess
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.uci import sha256_file
from rukh.paths import resolve

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"


class PgnTextConfig(BaseConfig):
    """Which games become text, how many, and how long each one is allowed to be."""

    train_games: str = "data/uci/year=2025/month=01/games.parquet"
    val_games: str = "data/uci/year=2025/month=02/games.parquet"
    out_dir: str = "data/pgn-text"
    n_train: int = Field(default=100_000, ge=1)
    n_val: int = Field(default=2_000, ge=1)
    max_plies: int = Field(default=120, ge=2)
    """Sixty moves a side. Past that the text runs longer than the context is worth spending."""
    val_offset: int = Field(default=0, ge=0)
    """Rows to skip in the validation month, so the split can avoid the evaluation pool."""
    seed: int = 42


def movetext(board: chess.Board, max_plies: int | None = None) -> str:
    """The moves played on ``board`` as SAN movetext with numbers: ``1. e4 e5 2. Nf3``.

    Replayed from the board's root rather than read off the stack, because SAN depends on the
    position it is written in: the same move is ``Nf3`` or ``Ngf3`` depending on where the other
    knight is, and only the position at that moment knows which.
    """
    replay = board.root()
    parts: list[str] = []
    moves = list(board.move_stack)
    if max_plies is not None:
        moves = moves[:max_plies]
    for move in moves:
        if replay.turn == chess.WHITE:
            parts.append(f"{replay.fullmove_number}.")
        elif not parts:
            parts.append(f"{replay.fullmove_number}...")
        parts.append(replay.san(move))
        replay.push(move)
    return " ".join(parts)


def header(white_elo: int, black_elo: int, fen: str | None = None) -> str:
    """The PGN tags the prompt carries: the two ratings, and a FEN when the game is not a game.

    The ratings are here for the same reason the decoder has `<wXXXX> <bXXXX>`: so the
    conditioning question can be put to both models in the language each one reads.
    """
    tags = [f'[WhiteElo "{white_elo}"]', f'[BlackElo "{black_elo}"]']
    if fen is not None:
        tags.append(f'[FEN "{fen}"]')
    return " ".join(tags)


def prompt_for(board: chess.Board, white_elo: int = 1800, black_elo: int = 1800) -> str:
    """The text a model is asked to continue: the header, the movetext so far, and a space.

    A position reached from a FEN (a puzzle) carries the FEN tag, so the model is not asked to
    guess a history that never happened.
    """
    root = board.root()
    fen = None if root.fen() == chess.STARTING_FEN else root.fen()
    played = movetext(board)
    line = f"{header(white_elo, black_elo, fen)}\n{played}".rstrip()
    if board.turn == chess.WHITE:
        line = f"{line} {board.fullmove_number}."
    elif not played:
        # Black to move at the very start of a FEN position needs the "17..." marker; after a
        # white move the number is already in the text and the model only has to answer.
        line = f"{line} {board.fullmove_number}..."
    # Never end on a space. A BPE writes moves as " Nc6" with the space attached, so a prompt
    # that already carries it forces the model onto the spelling it almost never saw in training.
    return line.rstrip()


def game_text(uci: str, white_elo: int, black_elo: int, result: str, max_plies: int) -> str | None:
    """One whole game as the training sample: header, movetext, result. None if it will not replay.

    A game that does not replay is dropped rather than repaired: the UCI corpus was validated
    with python-chess in P1, so anything that fails here is a bug worth seeing in the counts and
    not a row worth guessing at.
    """
    board = chess.Board()
    try:
        for index, token in enumerate(uci.split()):
            if index >= max_plies:
                break
            board.push(chess.Move.from_uci(token))
    except (ValueError, AssertionError):
        return None
    if not board.move_stack:
        return None
    body = movetext(board)
    return f"{header(white_elo, black_elo)}\n{body} {result}"


def _rows(path: Path, limit: int, offset: int = 0) -> Iterator[dict[str, object]]:
    import polars as pl

    frame = (
        pl.scan_parquet(path.as_posix())
        .select(["uci", "white_elo", "black_elo", "result"])
        .slice(offset, limit)
        .collect()
    )
    yield from frame.rows(named=True)


def _write_split(
    source: Path, target: Path, limit: int, max_plies: int, offset: int = 0
) -> tuple[int, int]:
    """Write one JSONL split; return ``(written, dropped)``."""
    written = dropped = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in _rows(source, limit, offset):
            text = game_text(
                str(row["uci"]),
                int(row["white_elo"] or 0),
                int(row["black_elo"] or 0),
                str(row["result"]),
                max_plies,
            )
            if text is None:
                dropped += 1
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written, dropped


def build(cfg: PgnTextConfig) -> Manifest:
    """Write ``train.jsonl`` and ``val.jsonl`` plus the manifest that says what went in."""
    out_dir = resolve(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve(cfg.train_games)
    val_source = resolve(cfg.val_games)
    for path in (train_source, val_source):
        if not path.is_file():
            raise FileNotFoundError(f"games not found at {path}")
    train_written, train_dropped = _write_split(
        train_source, out_dir / TRAIN_FILE, cfg.n_train, cfg.max_plies
    )
    val_written, val_dropped = _write_split(
        val_source, out_dir / VAL_FILE, cfg.n_val, cfg.max_plies, cfg.val_offset
    )
    log.info("wrote %s train and %s val samples", train_written, val_written)
    files = [
        FileHash(
            path=name, sha256=sha256_file(out_dir / name), bytes=(out_dir / name).stat().st_size
        )
        for name in (TRAIN_FILE, VAL_FILE)
    ]
    manifest = Manifest(
        dataset="rukh-pgn-text",
        months=[],
        filters={
            "train_games": cfg.train_games,
            "val_games": cfg.val_games,
            "max_plies": cfg.max_plies,
            "val_offset": cfg.val_offset,
            "dropped": {"train": train_dropped, "val": val_dropped},
        },
        counts={"train": train_written, "val": val_written},
        files=files,
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    return manifest
