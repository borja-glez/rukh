"""Training the reward model on preference pairs, and measuring it on pairs it never saw.

The data is the one the DPO pairs already use: a prefix of a real game, a move an engine scored
better and a move it scored worse, at least ``min_delta_cp`` apart. What this loop adds is the
translation from "(prefix, move)" to "the position the move leads to", which is what the encoder
of M3 reads, and a split that does not lie.

**The split goes by game, not by pair.** Two pairs from the same game are two moves a few plies
apart in the same position, so a model that memorised one has most of the other. `game_split` is
the same pure CRC-32 function M3 used, for the same reason: it gives the same answer in every
process and every future run.

**The metric is accuracy on held-out pairs**, which `docs/acceptance.md` puts a bar on, plus the
correlation
between the reward and the centipawn difference it was trained from -- Spearman as well as
Pearson, because M3 measured that a head can get the order right and the scale wrong, and the two
numbers are the diagnosis.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import chess
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ConfigDict, Field
from torch import Tensor

from rukh.config import BaseConfig
from rukh.models.config import EncoderConfig
from rukh.models.encoder import PositionEncoder
from rukh.models.reward import RewardModel
from rukh.models.squares import fen_to_tokens

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator, Sequence

log = logging.getLogger(__name__)

__all__ = [
    "RewardBand",
    "RewardConfig",
    "RewardExample",
    "RewardResult",
    "encode_pairs",
    "load_squares_encoder",
    "train_reward",
]


class RewardConfig(BaseConfig):
    """One reward-model run: where the pairs are, what to start from, and the two knobs."""

    pairs: str = "data/pairs/dpo-prompts.parquet"
    encoder_ckpt: str | None = None
    """The masked-move checkpoint to start from. ``None`` trains from scratch, which is the
    honest baseline the pretraining has to beat -- the same argument as M3's heads."""
    model: EncoderConfig | None = None
    pooling: str = "mean"
    out_dir: str = "checkpoints"
    run_name: str = "rm"
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    batch_size: int = Field(default=64, ge=1)
    epochs: int = Field(default=3, ge=1)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    seed: int = 42
    device: str | None = None
    max_pairs: int | None = None
    """Cap for a smoke run; ``None`` uses every usable pair."""
    point_of_view: Literal["mover", "white"] = "white"
    """Whose side the reward is good for.

    ``mover`` asks the model for "how good is this position for whoever just moved", which is what
    a reward *is* and which makes the sign flip every ply -- the model has to read the side to move
    out of the FEN and invert. ``white`` asks for the value head's question instead, "how good is
    this for White", and puts the flip in the **loss**, where it is a known constant rather than
    something to learn.

    They are the same information and they are not the same task, which is why this is a knob and
    not a decision: M5 measured both (D-112)."""


class RewardExample(BaseConfig):
    """One preference, already turned into the two positions the encoder reads."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    game_id: str
    chosen: list[int]
    rejected: list[int]
    white_moved: bool
    """Whose move the pair is about. The reward is read from that side's point of view."""
    delta_cp: float
    """How much better the engine said the chosen move was. Not used by the loss -- Bradley-Terry
    only sees the order -- but it is what the correlation is measured against."""


class RewardBand(BaseConfig):
    """Accuracy over the pairs whose engine gap falls in one band."""

    band: str
    pairs: int
    accuracy: float


class RewardResult(BaseConfig):
    """What the run produced and what it measured."""

    checkpoint: str
    pairs: int
    train_pairs: int
    val_pairs: int
    accuracy: float
    """Share of held-out pairs ordered correctly. The bar of ``docs/acceptance.md`` is 75 %."""
    loss: float
    pearson: float | None = None
    spearman: float | None = None
    pearson_without_mates: float | None = None
    """The same correlation over the pairs an evaluation can decide.

    Both are published because on this dataset they do not share a sign (D-119): the mate pairs
    carry the largest engine gap and the smallest learned margin, and a third of the points in that
    corner is enough to bend the line the other way."""
    spearman_without_mates: float | None = None
    epochs: int = 0
    from_scratch: bool = True
    point_of_view: str = "white"
    bands: list[RewardBand] = []
    """Accuracy split by how far apart the engine scored the two moves.

    The number that turns a missed bar into a diagnosis. Accuracy rises with the gap, as it
    should -- until the mate band, where it falls off a cliff: a mate is a tactical fact and a
    static evaluation of the resulting position cannot see it. A third of these pairs are
    mate-scored, so the overall figure is dragged down by the very rows a reward *model* is worst
    equipped for and a verifiable reward answers exactly."""
    accuracy_without_mates: float | None = None
    """Accuracy over the pairs an evaluation could decide, which is the fair read of the bar."""


def _position_after(prefix: str, move_uci: str) -> tuple[str, bool] | None:
    """``(fen after the move, the mover was white)``; ``None`` when the row does not replay.

    A row is dropped rather than repaired. The pairs came from a different pipeline and a prefix
    that does not replay means the row disagrees with python-chess about the game; guessing which
    of the two is right is how a dataset quietly acquires positions nobody played.
    """
    board = chess.Board()
    try:
        for uci in prefix.split():
            board.push_uci(uci)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None
        mover_was_white = board.turn == chess.WHITE
        board.push(move)
    except (ValueError, AssertionError):
        return None
    return board.fen(), mover_was_white


def encode_pairs(frame: Any, limit: int | None = None) -> list[RewardExample]:
    """Turn the pair rows into the two token sequences the encoder reads."""
    rows = frame.iter_rows(named=True) if hasattr(frame, "iter_rows") else frame
    examples: list[RewardExample] = []
    for row in rows:
        prefix = row.get("prefix") or ""
        after_chosen = _position_after(prefix, row["chosen"])
        after_rejected = _position_after(prefix, row["rejected"])
        if after_chosen is None or after_rejected is None:
            continue
        chosen_fen, white_moved = after_chosen
        rejected_fen, _ = after_rejected
        examples.append(
            RewardExample(
                game_id=str(row["game_id"]),
                chosen=fen_to_tokens(chosen_fen),
                rejected=fen_to_tokens(rejected_fen),
                white_moved=white_moved,
                delta_cp=abs(float(row["cp_rejected"]) - float(row["cp_chosen"])),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def split_examples(
    examples: Sequence[RewardExample], val_fraction: float, seed: int
) -> tuple[list[RewardExample], list[RewardExample]]:
    """Split by **game**, not by pair: two pairs of one game are almost the same pair."""
    from rukh.data.labels import game_split

    train: list[RewardExample] = []
    val: list[RewardExample] = []
    for example in examples:
        side = game_split(example.game_id, val_fraction, seed)
        (val if side == "val" else train).append(example)
    return train, val


def batches(
    examples: Sequence[RewardExample], size: int, shuffle: bool, seed: int
) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """``(chosen, rejected, sign, delta_cp)``, where ``sign`` is +1 when White made the move."""
    order = list(range(len(examples)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [examples[index] for index in order[start : start + size]]
        yield (
            torch.tensor([example.chosen for example in chunk], dtype=torch.long),
            torch.tensor([example.rejected for example in chunk], dtype=torch.long),
            torch.tensor(
                [1.0 if example.white_moved else -1.0 for example in chunk], dtype=torch.float
            ),
            torch.tensor([example.delta_cp for example in chunk], dtype=torch.float),
        )


def margins(
    model: RewardModel, chosen: Tensor, rejected: Tensor, sign: Tensor, point_of_view: str
) -> Tensor:
    """``r(chosen) - r(rejected)``, oriented so that positive means "the preferred one won".

    With ``point_of_view="white"`` the model answers White's question and the flip lives here, in
    a constant the loss knows. With ``"mover"`` the model has to work out from the board who just
    moved and answer for them, and the sign is always +1.
    """
    difference = model(chosen) - model(rejected)
    return difference * sign if point_of_view == "white" else difference


@torch.no_grad()
def evaluate(
    model: RewardModel,
    examples: Sequence[RewardExample],
    size: int,
    device: torch.device,
    point_of_view: str = "white",
) -> dict[str, float]:
    """Accuracy, loss and the two correlations against the engine's margin."""
    model.eval()
    correct = total = 0
    losses: list[float] = []
    scored: list[float] = []
    deltas: list[float] = []
    for chosen, rejected, sign, delta in batches(examples, size, shuffle=False, seed=0):
        chosen, rejected, sign = chosen.to(device), rejected.to(device), sign.to(device)
        margin = margins(model, chosen, rejected, sign, point_of_view)
        losses.append(float(-F.logsigmoid(margin).mean()))
        correct += int((margin > 0).sum())
        total += int(chosen.shape[0])
        scored.extend(margin.tolist())
        deltas.extend(delta.tolist())
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "loss": float(np.mean(losses)) if losses else math.nan,
        **_correlations(scored, deltas),
        "bands": band_accuracy(scored, deltas),
        **_correlations(
            [m for m, d in zip(scored, deltas, strict=True) if d < MATE_GAP],
            [d for d in deltas if d < MATE_GAP],
            suffix="_without_mates",
        ),
    }


MATE_GAP = 2000.0
"""Above this the pair involves a mate score, which is not a position an evaluation can read."""

BANDS: tuple[tuple[float, float, str], ...] = (
    (100, 200, "100-200"),
    (200, 400, "200-400"),
    (400, 800, "400-800"),
    (800, MATE_GAP, "800-2000"),
    (MATE_GAP, float("inf"), "mate"),
)


def band_accuracy(scored: Sequence[float], deltas: Sequence[float]) -> list[RewardBand]:
    """Accuracy by how far apart the engine put the two moves, plus a mate band of its own."""
    margin = np.asarray(scored, dtype=float)
    gap = np.asarray(deltas, dtype=float)
    rows: list[RewardBand] = []
    for low, high, label in BANDS:
        mask = (gap >= low) & (gap < high)
        if not mask.any():
            continue
        rows.append(
            RewardBand(band=label, pairs=int(mask.sum()), accuracy=float((margin[mask] > 0).mean()))
        )
    return rows


def _correlations(
    margins: Sequence[float], deltas: Sequence[float], suffix: str = ""
) -> dict[str, float]:
    """Pearson and Spearman between the learned margin and the engine's centipawn gap.

    Both, because M3 measured a head that got the order right and the scale wrong (D-076). When
    they disagree, the disagreement *is* the diagnosis.

    Reported twice, over all pairs and over the pairs that are not mates, because on this dataset
    the two do not even share a **sign**: -0.128 against +0.058 (D-119). Mate pairs carry the
    largest engine gap by construction and the smallest learned margin, so a third of the points
    sits in the corner that bends the line. One correlation over two regimes describes neither.
    """
    keys = (f"pearson{suffix}", f"spearman{suffix}")
    if len(margins) < 3:
        return dict.fromkeys(keys, math.nan)
    from scipy.stats import pearsonr, spearmanr

    margin = np.asarray(margins, dtype=float)
    delta = np.asarray(deltas, dtype=float)
    if margin.std() == 0 or delta.std() == 0:
        return dict.fromkeys(keys, math.nan)
    return {
        keys[0]: float(pearsonr(margin, delta).statistic),
        keys[1]: float(spearmanr(margin, delta).statistic),
    }


def load_squares_encoder(path: Path, map_location: str = "cpu") -> PositionEncoder:
    """The encoder inside a checkpoint, refusing anything that does not read positions.

    The guard is the whole point. A reward model is fed the 69 square tokens of
    ``rukh.models.squares``, and an encoder trained on the ``moves`` scheme accepts those ids
    without complaining -- they are all inside its 2 030-token vocabulary -- while meaning
    something else entirely. Nothing crashes, nothing warns, and the run produces a plausible
    number from a scrambled representation. It happened: the first reward model of this milestone
    reached 72.98 % that way.

    A heads checkpoint is accepted too, and is in fact the better start: its encoder was already
    fine-tuned to answer "how good is this position", which is what a reward is.
    """
    from rukh.train.checkpoint import load_checkpoint, load_state

    payload = load_checkpoint(path, map_location=map_location)
    shape = EncoderConfig.model_validate(payload["model_cfg"])
    if shape.input != "squares":
        raise ValueError(
            f"{path} was trained on the `{shape.input}` scheme; a reward model reads positions "
            "and needs an encoder trained on `squares`"
        )
    encoder = PositionEncoder(shape)
    state = payload["model_state"]
    inner = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    load_state(encoder, inner or state)
    return encoder


def build_model(cfg: RewardConfig, device: torch.device) -> tuple[RewardModel, bool]:
    """The reward model, from a pretrained encoder when there is one."""
    if cfg.encoder_ckpt:
        encoder = load_squares_encoder(Path(cfg.encoder_ckpt), map_location=str(device))
        return RewardModel(encoder, cfg.pooling).to(device), False
    shape = cfg.model or EncoderConfig(input="squares")
    if shape.input != "squares":
        raise ValueError("the reward model reads positions; `model.input` must be `squares`")
    return RewardModel(PositionEncoder(shape), cfg.pooling).to(device), True


def train_reward(cfg: RewardConfig) -> RewardResult:
    """Train the reward model and return what it measured on pairs it never saw."""
    from datetime import UTC, datetime

    import polars as pl

    from rukh.train import pick_device

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or pick_device())
    frame = pl.read_parquet(cfg.pairs)
    examples = encode_pairs(frame, cfg.max_pairs)
    if not examples:
        raise ValueError(f"no usable pairs in {cfg.pairs}")
    train_set, val_set = split_examples(examples, cfg.val_fraction, cfg.seed)
    if not train_set or not val_set:
        raise ValueError(
            f"the split left {len(train_set)} train and {len(val_set)} val pairs; "
            "check `val_fraction` and that the pairs carry more than one game"
        )
    log.info(
        "reward model: %d pairs (%d train, %d val) over %d games",
        len(examples),
        len(train_set),
        len(val_set),
        len({example.game_id for example in examples}),
    )

    model, from_scratch = build_model(cfg, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs):
        seen = 0
        running = 0.0
        for chosen, rejected, sign, _delta in batches(
            train_set, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch
        ):
            chosen, rejected, sign = chosen.to(device), rejected.to(device), sign.to(device)
            loss = -F.logsigmoid(margins(model, chosen, rejected, sign, cfg.point_of_view)).mean()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            running += float(loss.detach()) * int(chosen.shape[0])
            seen += int(chosen.shape[0])
        measured = evaluate(model, val_set, cfg.batch_size, device, cfg.point_of_view)
        log.info(
            "epoch %d/%d  train loss %.4f  val loss %.4f  val accuracy %.4f",
            epoch + 1,
            cfg.epochs,
            running / max(seen, 1),
            measured["loss"],
            measured["accuracy"],
        )

    final = evaluate(model, val_set, cfg.batch_size, device, cfg.point_of_view)
    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    directory = Path(cfg.out_dir) / f"{cfg.run_name}-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "reward.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "encoder_config": model.encoder.cfg.model_dump(mode="json"),
            "pooling": cfg.pooling,
            "config": cfg.model_dump(mode="json"),
        },
        checkpoint,
    )
    result = RewardResult(
        checkpoint=checkpoint.as_posix(),
        pairs=len(examples),
        train_pairs=len(train_set),
        val_pairs=len(val_set),
        accuracy=final["accuracy"],
        loss=final["loss"],
        pearson=_clean(final["pearson"]),
        spearman=_clean(final["spearman"]),
        pearson_without_mates=_clean(final["pearson_without_mates"]),
        spearman_without_mates=_clean(final["spearman_without_mates"]),
        epochs=cfg.epochs,
        from_scratch=from_scratch,
        point_of_view=cfg.point_of_view,
        bands=list(final["bands"]),
        accuracy_without_mates=_without_mates(final["bands"]),
    )
    (directory / "run.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _clean(value: float) -> float | None:
    """`None` instead of NaN, so the run file reads as "not measured" and not as a number."""
    return None if math.isnan(value) else float(value)


def _without_mates(bands: Sequence[RewardBand]) -> float | None:
    """Accuracy over everything but the mate band: what the model could actually have known."""
    usable = [row for row in bands if row.band != "mate"]
    pairs = sum(row.pairs for row in usable)
    if not pairs:
        return None
    return sum(row.accuracy * row.pairs for row in usable) / pairs


def load_reward(path: Path | str, map_location: str | None = None) -> RewardModel:
    """Read a reward-model checkpoint back."""
    payload = torch.load(Path(path), map_location=map_location or "cpu", weights_only=False)
    encoder = PositionEncoder(EncoderConfig.model_validate(payload["encoder_config"]))
    model = RewardModel(encoder, payload.get("pooling", "mean"))
    model.load_state_dict(payload["model"])
    return model.eval()
