"""GRPO on the decoder: a group of candidate moves, scored by a function instead of a model.

DPO needs a dataset of pairs somebody else decided. GRPO needs neither the pairs nor a reward
model: for each position the policy proposes ``group_size`` moves, a reward *function* scores
them, and each move is pushed up or down according to how it did **relative to its own group**.
The group is the baseline, which is the whole trick -- no value network, no critic, nothing to
train twice.

Four things about this file that are decisions, not details.

**One token, one exact gradient.** A completion here is a single move, so the usual
sequence-level machinery collapses: the policy gradient is ``advantage * log pi(move | prompt)``
at one position, and one forward pass gives the whole distribution. That is why ``group_size``
candidates cost one forward pass rather than ``group_size`` of them.

**The group is the baseline, so a group with no spread is worth nothing.** If all eight samples
are the same move, every advantage is zero and the step is a no-op with the engine already paid
for. Those groups are skipped and counted, because the count is a measurement: it says how much
of the budget the model's own confidence is throwing away.

**The engine is a cache, not a call.** Scoring every candidate at depth ``depth`` is the whole
cost of the run, and across steps the policy proposes the same moves in the same positions over
and over. A ``(fen, move) -> centipawns`` cache turns the second visit into a dictionary lookup;
it is keyed by the FEN so it stays correct when the same position arrives by a different route.

**KL is a term, not a clip.** The reference model is the frozen starting policy and the penalty
is the k3 estimator, which is non-negative and low-variance. Without it a verifiable reward is an
invitation to stop playing chess and start farming the reward -- and `labs/m5` has the gallery of
what that looks like when the invitation is accepted.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field

from rukh.config import BaseConfig
from rukh.tokenize.uci_vocab import UciTokenizer, elo_token
from rukh.train.rewards import RewardWeights, illegal_value, reward

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LOGGER = logging.getLogger(__name__)

__all__ = [
    "EngineCache",
    "best_available",
    "GrpoConfig",
    "GrpoResult",
    "group_advantages",
    "grpo_loss",
    "sample_group",
    "train_grpo",
]


class GrpoConfig(BaseConfig):
    """The prompts, the policy, the size of a group and the price of scoring one."""

    prompts: str = "data/pairs/dpo-prompts.parquet"
    """Only the ``prefix`` column is read. GRPO needs positions, not preferences -- which is the
    point worth noticing: the same file feeds both methods and only one of them uses the labels."""
    checkpoint: str = ""
    out_dir: str = "checkpoints"
    run_name: str = "grpo"
    group_size: int = Field(default=8, ge=2)
    """Candidates per position. The baseline is their mean, so two is the minimum that has one."""
    temperature: float = Field(default=1.0, gt=0)
    top_k: int | None = 20
    """How many candidate tokens the group is drawn from, after the restriction below."""
    restrict_to_legal: bool = True
    """Whether the group is drawn only from moves that can be played here.

    ``True`` spends the whole engine budget on moves the model could actually play, and the
    legality gate then never fires -- which is fine, because the demo masks illegal moves before
    they reach the board, so legality is a property the reward *guarantees* rather than one it
    teaches.

    ``False`` draws from the whole vocabulary, so an illegal proposal can land in the group and
    score the floor. That is the only configuration in which GRPO can teach legality at all, and
    M5 needed it: DPO doubled the illegal-proposal rate of its base (D-121), and a reward whose
    first rule is a legality gate ought to be able to pay that back."""
    beta_kl: float = Field(default=0.02, ge=0)
    """Weight of the KL term against the frozen starting policy."""
    lr: float = Field(default=1e-6, gt=0)
    positions_per_step: int = Field(default=8, ge=1)
    steps: int = Field(default=200, ge=1)
    depth: int = Field(default=8, ge=1)
    """Engine depth per candidate. Depth, not time, so a run is reproducible (D-107)."""
    block: int = Field(default=200, ge=8)
    weights: RewardWeights = RewardWeights()
    cache: str | None = "artifacts/grpo-engine-cache.json"
    """Where the ``(fen, move) -> centipawns`` cache lives between runs. ``None`` disables it."""
    val_positions: int = Field(default=200, ge=0)
    """Held-out positions the mean reward is measured on, before and after."""
    seed: int = 42
    device: str | None = None


class GrpoResult(BaseConfig):
    """What the run produced, in the shape the lesson quotes it."""

    checkpoint: str
    steps: int
    groups: int
    """Groups that contributed a gradient."""
    flat_groups: int
    """Groups where every candidate scored the same, so the baseline ate the whole signal."""
    engine_calls: int
    cache_hits: int
    reward_before: float
    reward_after: float
    """Mean reward with ``cp_best`` taken **inside the group**, which is what training optimises.

    Read it as a training diagnostic and never as a score. It is maximised by a policy that
    proposes the same move eight times out of eight: then the best candidate *is* every candidate,
    every quality term is 1.0, and the mean is as high as it goes. A model that collapses makes
    this number go up. ``reward_absolute_*`` is the one that cannot be gamed that way."""
    reward_absolute_before: float = math.nan
    reward_absolute_after: float = math.nan
    """Mean reward with ``cp_best`` taken from the **engine's own best move** in the position.

    One extra analysis per position, not per candidate, and it is the number that says whether the
    moves got better. Collapsing the policy does not move it: if the group agrees on a mediocre
    move, all eight are still mediocre against what was available."""
    flat_share_before: float = math.nan
    flat_share_after: float = math.nan
    """Share of validation groups whose candidates were all the same move. The collapse detector."""
    kl_after: float
    """Mean KL over the last tenth of the run, not the last batch.

    It was the last batch until a sweep reported 0.00048 and 0.126 for the same learning rate on
    two runs that differed in nothing that mattered: one group's KL is a sample, not a level."""
    illegal_before: float
    illegal_after: float


class EngineCache:
    """``(fen, move) -> centipawns for the mover``, kept between steps and between runs.

    Keyed by the FEN rather than by the move list, so a position reached by transposition is the
    same key. That is not a micro-optimisation: with eight candidates per position and two hundred
    steps, the engine is the entire cost of the run, and the policy keeps proposing the moves it
    already likes.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.data: dict[str, int] = {}
        self.hits = 0
        self.calls = 0
        if path is not None and path.exists():
            self.data = json.loads(path.read_text("utf-8"))
            LOGGER.info("cache: %d analisis reutilizables en %s", len(self.data), path)

    @staticmethod
    def key(fen: str, move: chess.Move) -> str:
        return f"{fen}|{move.uci()}"

    def score(
        self, engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move, depth: int
    ) -> int:
        """Centipawns for the side that plays ``move``, from the cache when it is there."""
        from rukh.data.onpolicy import score_move

        key = self.key(board.fen(), move)
        cached = self.data.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        value = score_move(engine, board, move, depth)
        self.data[key] = value
        self.calls += 1
        return value

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), "utf-8")


def sample_group(
    logits: torch.Tensor,
    legal_ids: Sequence[int] | None,
    size: int,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``size`` move tokens drawn from the policy at one position.

    Sampling is **with replacement**: the group is meant to be a sample of what the policy would
    actually do, and a policy that would play the same move eight times out of eight is telling
    the loss something true about itself. De-duplicating here would quietly turn the baseline into
    the mean of a distribution nobody is following.
    """
    scaled = logits.float() / temperature
    if legal_ids is None:
        mask = scaled
        width = scaled.numel()
    else:
        mask = torch.full_like(scaled, float("-inf"))
        index = torch.tensor(list(legal_ids), dtype=torch.long, device=scaled.device)
        mask[index] = scaled[index]
        width = index.numel()
    if top_k is not None and top_k < width:
        cut = torch.topk(mask, top_k).values[-1]
        mask = torch.where(mask < cut, torch.full_like(mask, float("-inf")), mask)
    probabilities = F.softmax(mask, dim=-1)
    return torch.multinomial(probabilities, size, replacement=True, generator=generator)


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> torch.Tensor:
    """``(r - mean) / (std + eps)``: how each candidate did against its own group.

    The normalisation by the spread is what makes a group of near-identical moves and a group with
    a blunder in it contribute comparably. When the spread is zero the advantages are zero, which
    is the honest answer -- the group said nothing -- and the caller drops the group rather than
    letting ``eps`` manufacture a gradient out of floating-point dust.
    """
    values = torch.tensor(list(rewards), dtype=torch.float)
    spread = float(values.std(unbiased=False))
    if spread < eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / (spread + eps)


def grpo_loss(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    beta_kl: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The objective and the KL, for a one-token completion.

    The KL is the k3 estimator ``exp(d) - d - 1`` with ``d = logp_ref - logp_policy``. It is
    non-negative by construction, which matters for a term that is supposed to be a *penalty*:
    the plain ``logp_ref - logp_policy`` difference is negative half the time and would pay the
    model for moving away from the reference on exactly the samples it moved away on.
    """
    chosen_policy = policy_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    chosen_reference = reference_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    objective = -(advantages * chosen_policy).mean()
    difference = chosen_reference - chosen_policy
    kl = (torch.exp(difference) - difference - 1.0).mean()
    return objective + beta_kl * kl, kl


def _prompt(tok: UciTokenizer, prefix: str, white: int, black: int, block: int) -> list[int] | None:
    """The same three header tokens every other measurement uses, then the game so far."""
    ids = [tok.bos_id, tok.vocab[elo_token(white, "w")], tok.vocab[elo_token(black, "b")]]
    for uci in prefix.split():
        token = tok.vocab.get(uci)
        if token is None:
            return None
        ids.append(int(token))
    return ids if len(ids) <= block else None


def _board_for(prefix: str) -> chess.Board | None:
    board = chess.Board()
    try:
        for uci in prefix.split():
            board.push_uci(uci)
    except ValueError:
        return None
    return None if board.is_game_over() else board


def _move_of(tok: UciTokenizer, token: int) -> chess.Move | None:
    """The move a token spells, or ``None`` when the token is not a move at all.

    The reverse table is built once per tokenizer and cached on it, because this runs eight times
    per position and rebuilding a 2 030-entry dict each time would be the second cost of the run
    after the engine.
    """
    inverse = getattr(tok, "_rukh_inverse", None)
    if inverse is None:
        inverse = {index: text for text, index in tok.vocab.items()}
        tok._rukh_inverse = inverse  # type: ignore[attr-defined]
    text = inverse.get(token)
    if text is None or len(text) < 4:
        return None
    try:
        return chess.Move.from_uci(text)
    except ValueError:
        return None


def best_available(
    cache: EngineCache, engine: chess.engine.SimpleEngine, board: chess.Board, cfg: GrpoConfig
) -> float:
    """Centipawns of the position with best play, for the side to move.

    The honest reference for "how good was this move", and the one `score_candidates` deliberately
    does **not** use while training. One analysis of the position itself, cached like the rest.
    """
    from rukh.data.scoring import MATE_SCORE

    key = f"{board.fen()}|BEST"
    cached = cache.data.get(key)
    if cached is not None:
        cache.hits += 1
        return float(cached)
    info = engine.analyse(board, chess.engine.Limit(depth=cfg.depth))
    value = int(info["score"].pov(board.turn).score(mate_score=MATE_SCORE))
    cache.data[key] = value
    cache.calls += 1
    return float(value)


def score_candidates(
    cache: EngineCache,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    moves: Sequence[chess.Move | None],
    cfg: GrpoConfig,
    best: float | None = None,
) -> list[float]:
    """The reward of each candidate, with ``cp_best`` taken over the candidates themselves.

    The best *available* move would be the honest reference, and it costs a full multi-PV search
    per position. The best move **in the group** costs nothing extra and asks the question GRPO
    actually asks: of what you proposed, which was best? A model whose whole group is bad still
    learns the ordering inside it, and the reward stays in the same closed interval either way.
    """
    legal = {move.uci(): move for move in moves if move is not None and move in board.legal_moves}
    scores = {
        uci: float(cache.score(engine, board, move, cfg.depth)) for uci, move in legal.items()
    }
    if best is None:
        best = max(scores.values()) if scores else 0.0
    floor = illegal_value(cfg.weights)
    return [
        floor
        if move is None
        else reward(
            board,
            move,
            cp_move=scores.get(move.uci()),
            cp_best=best if move.uci() in scores else None,
            weights=cfg.weights,
        ).total
        for move in moves
    ]


def train_grpo(cfg: GrpoConfig) -> GrpoResult:
    """Run GRPO and return what it measured, including the two numbers the lesson needs."""
    import copy

    import polars as pl

    from rukh import paths
    from rukh.engine import find_stockfish
    from rukh.infer.sampler import legal_token_ids
    from rukh.train.checkpoint import load_model, save_checkpoint
    from rukh.train.common import pick_device

    binary = find_stockfish()
    if binary is None:
        raise FileNotFoundError("Stockfish not found; set RUKH_STOCKFISH")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or pick_device())
    tok = UciTokenizer()

    policy, payload = load_model(paths.resolve(cfg.checkpoint), device)
    policy = policy.to(device).train()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    frame = pl.read_parquet(paths.resolve(cfg.prompts).as_posix())
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(frame.height)
    val_index = order[: cfg.val_positions]
    train_index = order[cfg.val_positions :]

    cache = EngineCache(paths.resolve(cfg.cache) if cfg.cache else None)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    def group_at(row: dict[str, object]) -> tuple[chess.Board, list[int], torch.Tensor] | None:
        """Prompt, board and a sampled group for one row, or ``None`` when the row is unusable."""
        board = _board_for(str(row["prefix"] or ""))
        ids = _prompt(
            tok, str(row["prefix"] or ""), int(row["white_elo"]), int(row["black_elo"]), cfg.block
        )
        if board is None or ids is None:
            return None
        with torch.no_grad():
            logits, _ = policy(torch.tensor([ids], dtype=torch.long, device=device))
        tokens = sample_group(
            logits[0, -1],
            legal_token_ids(board, tok) if cfg.restrict_to_legal else None,
            cfg.group_size,
            cfg.temperature,
            cfg.top_k,
            generator,
        )
        return board, ids, tokens

    groups = flat = 0
    last_kl = math.nan
    recent_kl: list[float] = []
    tail = max(cfg.steps // 10, 1)
    """How many of the final steps the reported KL averages over."""
    with chess.engine.SimpleEngine.popen_uci(str(binary)) as engine:

        def measure(indices: np.ndarray) -> tuple[float, float, float, float]:
            """Mean reward two ways, the illegal share, and how many groups collapsed.

            Two ways because they answer different questions and only one of them is a score. With
            ``cp_best`` inside the group -- what training uses -- a policy that proposes one move
            eight times scores the maximum, so the number goes up when the model collapses. With
            ``cp_best`` from the engine's own best move it cannot be gamed that way, and the gap
            between the two is exactly how much of the first is the collapse.
            """
            policy.eval()
            relative: list[float] = []
            absolute: list[float] = []
            illegal = total = flat = groups_seen = 0
            for row_index in indices:
                drawn = group_at(frame.row(int(row_index), named=True))
                if drawn is None:
                    continue
                board, _ids, tokens = drawn
                moves = [_move_of(tok, int(token)) for token in tokens]
                illegal += sum(1 for move in moves if move is None or move not in board.legal_moves)
                total += len(moves)
                groups_seen += 1
                flat += int(len({int(token) for token in tokens}) == 1)
                relative.extend(score_candidates(cache, engine, board, moves, cfg))
                absolute.extend(
                    score_candidates(
                        cache,
                        engine,
                        board,
                        moves,
                        cfg,
                        best=best_available(cache, engine, board, cfg),
                    )
                )
            policy.train()
            return (
                float(np.mean(relative)) if relative else math.nan,
                float(np.mean(absolute)) if absolute else math.nan,
                illegal / total if total else math.nan,
                flat / groups_seen if groups_seen else math.nan,
            )

        before = measure(val_index)
        LOGGER.info(
            "antes: recompensa grupo %.4f - absoluta %.4f - ilegales %.4f - planos %.3f", *before
        )

        for step in range(cfg.steps):
            picks = rng.choice(train_index, size=cfg.positions_per_step, replace=False)
            losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            used = 0
            for row_index in picks:
                drawn = group_at(frame.row(int(row_index), named=True))
                if drawn is None:
                    continue
                board, ids, tokens = drawn
                moves = [_move_of(tok, int(token)) for token in tokens]
                if all(move is None for move in moves):
                    continue
                rewards = score_candidates(cache, engine, board, moves, cfg)
                advantages = group_advantages(rewards).to(device)
                if float(advantages.abs().sum()) == 0.0:
                    flat += 1
                    continue
                prompt = torch.tensor([ids], dtype=torch.long, device=device)
                policy_logits, _ = policy(prompt)
                policy_logprobs = F.log_softmax(policy_logits[0, -1].float(), dim=-1)
                with torch.no_grad():
                    reference_logits, _ = reference(prompt)
                    reference_logprobs = F.log_softmax(reference_logits[0, -1].float(), dim=-1)
                loss, kl = grpo_loss(
                    policy_logprobs.unsqueeze(0).expand(cfg.group_size, -1),
                    reference_logprobs.unsqueeze(0).expand(cfg.group_size, -1),
                    tokens,
                    advantages,
                    cfg.beta_kl,
                )
                (loss / cfg.positions_per_step).backward()
                losses.append(float(loss.detach()))
                last_kl = float(kl.detach())
                if step >= cfg.steps - tail:
                    recent_kl.append(last_kl)
                groups += 1
                used += 1
            if used:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
            if (step + 1) % 10 == 0:
                LOGGER.info(
                    "paso %d - loss %.4f - kl %.5f - grupos %d - planos %d - cache %d/%d",
                    step + 1,
                    float(np.mean(losses)) if losses else math.nan,
                    last_kl,
                    groups,
                    flat,
                    cache.hits,
                    cache.hits + cache.calls,
                )
                cache.save()

        if recent_kl:
            last_kl = float(np.mean(recent_kl))
        after = measure(val_index)
        LOGGER.info(
            "despues: recompensa grupo %.4f - absoluta %.4f - ilegales %.4f - planos %.3f", *after
        )
    cache.save()

    out_dir = paths.resolve(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "grpo.pt"
    save_checkpoint(
        final,
        step=cfg.steps,
        model=policy,
        optimizer=None,
        cfg={**cfg.model_dump(mode="json"), "reward_before": before[0], "reward_after": after[0]},
        model_cfg=payload["model_cfg"],
        vocab_hash=payload.get("vocab_hash"),
        data_manifest_sha=payload.get("data_manifest_sha"),
    )
    return GrpoResult(
        checkpoint=str(final),
        steps=cfg.steps,
        groups=groups,
        flat_groups=flat,
        engine_calls=cache.calls,
        cache_hits=cache.hits,
        reward_before=before[0],
        reward_after=after[0],
        reward_absolute_before=before[1],
        reward_absolute_after=after[1],
        flat_share_before=before[3],
        flat_share_after=after[3],
        kl_after=last_kl,
        illegal_before=before[2],
        illegal_after=after[2],
    )
