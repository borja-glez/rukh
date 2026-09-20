"""Preference pairs built from the moves the **model** would actually play.

The pairs the project already has are off-policy: they come from Lichess's multi-PV evaluations,
so they say "in this position ``e4`` is better than ``h4``" regardless of whether the model was
ever going to consider either. That is information, and it is not the information that fixes a
particular model. A model only ever loses games to the moves it plays.

An on-policy pair asks a narrower and more useful question: *of the moves you were going to
consider here, which was best and which was worst?* Sample ``candidates`` moves from the model,
score them with the engine, keep the best and the worst if they are far enough apart.

Three decisions worth defending, because each one is a way the comparison could have been unfair.

**The positions are the same ones the off-policy pairs use.** Same prefixes, same phase balance,
same games. The only thing that changes between the two datasets is where the two moves came
from, which is the whole point of running both.

**The engine is limited by depth, not by time.** P4 measured that a time-limited Stockfish makes
a measurement irreproducible (D-107): the same run twice gave 1498 and 1558. A depth limit gives
the same answer on a busy machine and on an idle one, and a dataset is a thing that should be
rebuildable.

**Sampling is at temperature 1.0.** At the near-deterministic setting every published number is
read at, a model proposes the same move four times and there is no pair to build. The candidates
have to come from the distribution, not from its mode -- which is the same lesson M4 ended on.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chess
import chess.engine
import polars as pl
import torch
from pydantic import Field

from rukh.config import BaseConfig
from rukh.data.manifest import FileHash, Manifest
from rukh.data.scoring import MATE_SCORE
from rukh.data.uci import sha256_file
from rukh.paths import resolve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rukh.models import MoveDecoder
    from rukh.tokenize.uci_vocab import UciTokenizer

log = logging.getLogger(__name__)

__all__ = ["OnPolicyConfig", "build_on_policy", "score_move", "sample_candidates"]


class OnPolicyConfig(BaseConfig):
    """Where the positions come from, which model proposes, and how deep the engine looks."""

    source: str = "data/pairs/dpo-prompts.parquet"
    """The off-policy pairs. Only their **positions** are reused; their moves are not."""
    checkpoint: str = ""
    out: str = "data/pairs-onpolicy/pairs.parquet"
    candidates: int = Field(default=4, ge=2)
    """Moves sampled per position. `docs/spec/02` says four; more is more engine time per pair."""
    temperature: float = Field(default=1.0, gt=0)
    top_k: int | None = 20
    depth: int = Field(default=10, ge=1)
    """Engine depth. A depth limit is reproducible; a time limit is not (D-107)."""
    min_delta_cp: int = Field(default=100, ge=1)
    max_positions: int | None = None
    seed: int = 42
    device: str | None = None


def score_move(
    engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move, depth: int
) -> int:
    """Centipawns **for the side that plays ``move``**, after playing it.

    The engine is asked about the position the move leads to, where the *opponent* is to move, so
    the score comes back from the opponent's point of view and is negated. Mates use the project's
    own ``MATE_SCORE`` convention so these numbers can sit in the same column as the off-policy
    ones without a footnote.
    """
    board.push(move)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(not board.turn)
        return int(score.score(mate_score=MATE_SCORE))
    finally:
        board.pop()


def sample_candidates(
    model: MoveDecoder,
    tok: UciTokenizer,
    board: chess.Board,
    history: list[int],
    count: int,
    temperature: float,
    top_k: int | None,
) -> list[chess.Move]:
    """Distinct legal moves the model proposes here, sampled from its own distribution.

    Illegal proposals are dropped rather than rescued: an on-policy pair is about ranking the
    moves the model would *play*, and the demo masks illegal moves before they reach the board.
    How often it proposes one is a different measurement, and the suite already makes it.
    """
    from rukh.infer.sampler import SampleConfig, model_generator, pick_move

    cfg = SampleConfig(temperature=temperature, top_k=top_k, mask_illegal=False)
    generator = model_generator(model, cfg)
    seen: dict[str, chess.Move] = {}
    for _ in range(count * 4):  # a few extra draws, since duplicates are the common case
        move, _ = pick_move(model, tok, board, history, cfg, generator)
        if move is not None and move in board.legal_moves:
            seen.setdefault(move.uci(), move)
        if len(seen) >= count:
            break
    return list(seen.values())


def _history_for(tok: UciTokenizer, prefix: str, white_elo: int, black_elo: int) -> list[int]:
    """The prompt the model sees: the same three header tokens the ladder uses, then the game."""
    from rukh.infer.game import _history

    ids = _history(tok, white_elo, black_elo)
    ids.extend(tok.vocab.get(uci, tok.unk_id) for uci in prefix.split())
    return ids


def build_on_policy(cfg: OnPolicyConfig) -> Manifest:
    """Sample, score and write the on-policy pairs, with a manifest that says how."""
    from rukh.engine import find_stockfish
    from rukh.eval.suite import resolve_model
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model, pick_device

    binary = find_stockfish()
    if binary is None:
        raise FileNotFoundError("Stockfish not found; set RUKH_STOCKFISH")
    torch.manual_seed(cfg.seed)
    where = cfg.device or pick_device()
    tok = UciTokenizer()
    model, _payload = load_model(resolve_model(cfg.checkpoint), map_location=where)
    model = model.to(where).eval()

    frame = pl.read_parquet(resolve(cfg.source))
    if cfg.max_positions is not None:
        # Shuffled first, because the source is phase-balanced *in blocks*: taking the head would
        # buy a dataset made entirely of openings and then compare it to one that is not.
        frame = frame.sample(n=min(cfg.max_positions, frame.height), shuffle=True, seed=cfg.seed)

    rows: list[dict[str, object]] = []
    skipped_same = skipped_close = 0
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:
        for index, row in enumerate(frame.iter_rows(named=True)):
            board = chess.Board()
            try:
                for uci in (row["prefix"] or "").split():
                    board.push_uci(uci)
            except ValueError:
                continue
            if board.is_game_over():
                continue
            history = _history_for(
                tok, row["prefix"] or "", int(row["white_elo"]), int(row["black_elo"])
            )
            moves = sample_candidates(
                model, tok, board, history, cfg.candidates, cfg.temperature, cfg.top_k
            )
            if len(moves) < 2:
                skipped_same += 1
                continue
            scored = [(score_move(engine, board, move, cfg.depth), move) for move in moves]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            (best_cp, best), (worst_cp, worst) = scored[0], scored[-1]
            if best_cp - worst_cp < cfg.min_delta_cp:
                skipped_close += 1
                continue
            rows.append(
                {
                    "game_id": str(row["game_id"]),
                    "ply": int(row["ply"]),
                    "prefix": row["prefix"],
                    "chosen": best.uci(),
                    "rejected": worst.uci(),
                    "cp_chosen": int(best_cp),
                    "cp_rejected": int(worst_cp),
                    "phase": row["phase"],
                    "white_elo": int(row["white_elo"]),
                    "black_elo": int(row["black_elo"]),
                    "candidates": len(moves),
                }
            )
            if index and index % 250 == 0:
                log.info(
                    "%d positions, %d pairs (%d all-same, %d too close)",
                    index,
                    len(rows),
                    skipped_same,
                    skipped_close,
                )

    if not rows:
        raise ValueError("no on-policy pair survived; loosen min_delta_cp or raise the temperature")
    out = resolve(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows)
    table.write_parquet(out)
    log.info("wrote %d on-policy pairs to %s", table.height, out)

    manifest = Manifest(
        dataset="rukh-pairs-onpolicy",
        months=[],
        filters={
            "source": cfg.source,
            "checkpoint": cfg.checkpoint,
            "candidates": cfg.candidates,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "engine_depth": cfg.depth,
            "min_delta_cp": cfg.min_delta_cp,
            "seed": cfg.seed,
            "skipped_all_same": skipped_same,
            "skipped_too_close": skipped_close,
            "note": (
                "the moves come from the model, the positions from the off-policy pairs; "
                "the engine is limited by depth so the dataset is rebuildable (D-107)"
            ),
        },
        counts={"pairs": table.height, "positions": frame.height},
        files=[FileHash(path=out.name, sha256=sha256_file(out), bytes=out.stat().st_size)],
    )
    text = manifest.model_dump_json(indent=2) + "\n"
    (out.parent / "manifest.json").write_text(text, "utf-8")
    return manifest
