"""Tests for rukh.train.mmm: the 80/10/10 masking, the control tokens and a toy pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rukh.models import EncoderConfig, PositionEncoder
from rukh.models.encoder import MMM_IGNORE_INDEX
from rukh.models.squares import CONTROL_IDS as SQUARE_CONTROL_IDS
from rukh.models.squares import MASK_ID as SQUARE_MASK_ID
from rukh.models.squares import SQUARE_VOCAB_SIZE, fen_to_tokens
from rukh.tokenize.pack import META_FILE, STARTS_FILE, TOKENS_FILE, PackInfo
from rukh.tokenize.uci_vocab import MASK_ID, UciTokenizer
from rukh.train import MaskingConfig, MmmConfig, apply_masking, load_encoder, train_mmm
from rukh.train.common import run_dir
from rukh.train.mmm import MOVE_CONTROL_IDS, control_ids, masked_step, masking_generator

pytestmark = pytest.mark.unit

VOCAB = 128  # above the 62 control ids, so there are real moves to hide
BLOCK = 16
FIRST_MOVE = max(MOVE_CONTROL_IDS) + 1
TOY = EncoderConfig(vocab_size=VOCAB, n_layer=2, n_head=2, d_model=32, block=BLOCK, dropout=0.0)
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def moves_batch(rows: int = 50, cols: int = 200, seed: int = 0) -> torch.Tensor:
    """``rows * cols`` move tokens, none of them a control token."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(FIRST_MOVE, 2030, (rows, cols), generator=generator)


def write_pack(directory: Path, n_games: int = 24, plies: int = 24) -> Path:
    """A toy pack shaped like ``pack.pack_month``: a cyclic, easily learnable token stream."""
    directory.mkdir(parents=True, exist_ok=True)
    tokens: list[int] = []
    starts: list[int] = []
    for game in range(n_games):
        starts.append(len(tokens))
        tokens.append(1)  # <bos>
        tokens.extend(FIRST_MOVE + (game + ply) % 5 for ply in range(plies))
        tokens.append(2)  # <eos>
    np.save(directory / TOKENS_FILE, np.asarray(tokens, dtype=np.uint16))
    np.save(directory / STARTS_FILE, np.asarray(starts, dtype=np.int64))
    info = PackInfo(
        n_games=n_games,
        n_tokens=len(tokens),
        scheme="uci",
        vocab_size=VOCAB,
        vocab_hash="toyhash",
        source="toy.parquet",
    )
    (directory / META_FILE).write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def tokens_dir(rukh_home: Path) -> Path:
    for split in ("train", "val"):
        write_pack(rukh_home / "tokens" / split)
    return rukh_home / "tokens"


def toy_config(**overrides: object) -> MmmConfig:
    cfg = MmmConfig(
        model=TOY,
        tokens_dir="tokens",
        block=BLOCK,
        batch_size=4,
        grad_accum=2,
        lr=3e-2,
        warmup=1,
        max_steps=6,
        precision="fp32",
        compile=False,
        eval_every=3,
        eval_batches=2,
        ckpt_every=3,
        log_every=1,
        run_name="toy-mmm",
        unique_run_name=False,
        masking=MaskingConfig(prob=0.4),
    )
    return cfg.model_copy(update=dict(overrides))


def metric_history(run_id: str, key: str) -> list[float]:
    import mlflow

    from rukh.tracking import tracking_uri

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri())
    return [point.value for point in client.get_metric_history(run_id, key)]


def last_run_id() -> str:
    import mlflow

    from rukh.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    runs = mlflow.search_runs(experiment_names=["rukh"], output_format="list")
    return max(runs, key=lambda r: r.info.start_time).info.run_id


def test_the_80_10_10_split_holds_over_ten_thousand_tokens() -> None:
    batch = moves_batch()
    assert batch.numel() == 10_000
    cfg = MaskingConfig()
    inputs, labels = apply_masking(batch, cfg, 2030, generator=masking_generator(7))

    selected = labels != MMM_IGNORE_INDEX
    picked = int(selected.sum())
    assert abs(picked / batch.numel() - cfg.prob) < 0.02
    masked = int((inputs == MASK_ID).sum())
    kept = int((selected & (inputs == batch)).sum())
    replaced = picked - masked - kept
    assert abs(masked / picked - cfg.mask_ratio) < 0.04
    assert abs(replaced / picked - cfg.random_ratio) < 0.04
    assert abs(kept / picked - cfg.keep_ratio) < 0.04
    # Nothing outside the selection was touched.
    assert torch.equal(inputs[~selected], batch[~selected])


def test_labels_are_minus_100_exactly_where_nothing_was_hidden() -> None:
    batch = moves_batch(rows=8, cols=32, seed=1)
    inputs, labels = apply_masking(batch, MaskingConfig(), 2030, generator=masking_generator(3))
    selected = labels != MMM_IGNORE_INDEX
    assert torch.equal(labels[selected], batch[selected])  # the label is the original token
    assert set(labels[~selected].tolist()) == {MMM_IGNORE_INDEX}
    assert inputs.shape == labels.shape == batch.shape
    # A kept 10 % looks untouched in the input but still carries a label: that is the point.
    assert bool((selected & (inputs == batch)).any())


def test_control_tokens_are_never_hidden() -> None:
    tok = UciTokenizer()
    games = [
        tok.encode_game("e2e4 c7c5 g1f3 d7d6 d2d4", 1800, 1900, "1-0"),
        tok.encode_game("d2d4 g8f6 c2c4 e7e6", 2100, 2000, "0-1"),
    ]
    width = max(len(game) for game in games)
    batch = torch.tensor([game + [0] * (width - len(game)) for game in games])
    inputs, labels = apply_masking(
        batch, MaskingConfig(prob=1.0), len(tok), generator=masking_generator(5)
    )
    control = torch.isin(batch, torch.tensor(sorted(MOVE_CONTROL_IDS)))
    assert torch.equal(inputs[control], batch[control])  # <bos>, Elo, result, <eos>, <pad>
    assert set(labels[control].tolist()) == {MMM_IGNORE_INDEX}
    assert torch.equal(labels[~control], batch[~control])  # every move is hidden at prob 1
    assert len(MOVE_CONTROL_IDS) == 62 and sorted(MOVE_CONTROL_IDS) == list(range(62))


def test_a_random_replacement_is_always_a_real_move() -> None:
    batch = moves_batch(rows=20, cols=100, seed=2)
    only_random = MaskingConfig(prob=1.0, mask_ratio=0.0, random_ratio=1.0, keep_ratio=0.0)
    inputs, _ = apply_masking(batch, only_random, 2030, generator=masking_generator(11))
    assert int(inputs.min()) >= FIRST_MOVE  # never a fake Elo, result or <mask>
    assert int(inputs.max()) < 2030
    assert not torch.equal(inputs, batch)


def test_the_masking_is_reproducible_with_a_fixed_seed() -> None:
    batch = moves_batch(rows=8, cols=64, seed=4)
    cfg = MaskingConfig()
    first = apply_masking(batch, cfg, 2030, generator=masking_generator(cfg.seed))
    same = apply_masking(batch, cfg, 2030, generator=masking_generator(cfg.seed))
    other = apply_masking(batch, cfg, 2030, generator=masking_generator(cfg.seed + 1))
    assert torch.equal(first[0], same[0]) and torch.equal(first[1], same[1])
    assert not torch.equal(first[0], other[0])
    # One generator used twice keeps moving, so a run does not hide the same moves every step.
    generator = masking_generator(cfg.seed)
    step_one = apply_masking(batch, cfg, 2030, generator=generator)
    step_two = apply_masking(batch, cfg, 2030, generator=generator)
    assert torch.equal(step_one[0], first[0])
    assert not torch.equal(step_two[0], step_one[0])


def test_the_squares_scheme_hides_squares_and_keeps_its_own_control_tokens() -> None:
    batch = torch.tensor([fen_to_tokens(START), fen_to_tokens(f"{START} 30 40")])
    inputs, labels = apply_masking(
        batch,
        MaskingConfig(prob=1.0),
        SQUARE_VOCAB_SIZE,
        scheme="squares",
        generator=masking_generator(1),
    )
    control = torch.isin(batch, torch.tensor(sorted(SQUARE_CONTROL_IDS)))
    assert control_ids("squares") == SQUARE_CONTROL_IDS
    assert bool(control[:, 0].all())  # <cls> is the only control token a FEN produces
    assert torch.equal(inputs[control], batch[control])
    assert set(labels[control].tolist()) == {MMM_IGNORE_INDEX}
    assert bool((inputs == SQUARE_MASK_ID).any())


def test_a_masking_config_must_add_up() -> None:
    with pytest.raises(ValueError, match="add up to 1"):
        MaskingConfig(mask_ratio=0.5, random_ratio=0.1, keep_ratio=0.1)
    with pytest.raises(ValueError, match="prob must be"):
        MaskingConfig(prob=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        MaskingConfig(mask_ratio=1.2, random_ratio=-0.2, keep_ratio=0.0)
    with pytest.raises(ValueError):
        MaskingConfig(probability=0.15)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="all control tokens"):
        apply_masking(torch.ones((2, 2), dtype=torch.long), MaskingConfig(), 10)


def test_masked_step_agrees_with_the_model_method() -> None:
    torch.manual_seed(0)
    model = PositionEncoder(TOY).eval()
    batch = torch.randint(FIRST_MOVE, VOCAB, (3, BLOCK))
    inputs, labels = apply_masking(batch, MaskingConfig(), VOCAB, generator=masking_generator(2))
    with torch.no_grad():
        logits, loss = masked_step(model, model, inputs, labels, model.padding_mask(inputs))
        reference = model.masked_lm(inputs, labels, model.padding_mask(inputs))
    assert torch.allclose(logits, reference[0], atol=1e-6)
    assert torch.allclose(loss, reference[1], atol=1e-6)


def test_a_short_run_lowers_the_loss_and_writes_checkpoints(
    rukh_home: Path, tokens_dir: Path
) -> None:
    final = train_mmm(toy_config(), device="cpu")
    out = rukh_home / "checkpoints" / "toy-mmm"
    assert final == out / "step-6.pt"
    assert {p.name for p in out.glob("*.pt")} >= {"step-3.pt", "step-6.pt", "best.pt"}

    model, payload = load_encoder(final)
    assert payload["step"] == 6
    assert payload["vocab_hash"] == "toyhash"
    assert payload["model_cfg"]["d_model"] == TOY.d_model
    assert payload["model_cfg"]["input"] == "moves"
    assert isinstance(model, PositionEncoder)

    run_id = last_run_id()
    losses = metric_history(run_id, "train/loss")
    assert len(losses) == 6
    assert losses[-1] < losses[0]
    for key in ("lr", "grad_norm", "tokens_per_s", "masked_tokens_per_s"):
        assert len(metric_history(run_id, key)) == 6
    assert len(metric_history(run_id, "val/loss")) == 2
    top1 = metric_history(run_id, "val/top1")
    assert len(top1) == 2 and all(0.0 <= value <= 1.0 for value in top1)


def test_a_resumed_run_continues_where_it_stopped(rukh_home: Path, tokens_dir: Path) -> None:
    first = train_mmm(toy_config(max_steps=3, ckpt_every=3), device="cpu")
    assert first.name == "step-3.pt"
    second = train_mmm(toy_config(max_steps=6, ckpt_every=3), resume=first, device="cpu")
    assert second == first.parent / "step-6.pt"
    from rukh.train import load_checkpoint

    assert load_checkpoint(second)["step"] == 6
    assert load_checkpoint(second)["run_id"] == load_checkpoint(first)["run_id"]


def test_a_mismatched_vocabulary_is_refused(rukh_home: Path, tokens_dir: Path) -> None:
    cfg = toy_config(model=TOY.model_copy(update={"vocab_size": 64}))
    with pytest.raises(ValueError, match="vocab_size"):
        train_mmm(cfg, device="cpu")
    with pytest.raises(ValueError, match="fewer than"):
        train_mmm(toy_config(batch_size=1000), device="cpu")


def test_the_run_is_named_after_the_encoder_by_default() -> None:
    assert run_dir(MmmConfig()).name.startswith("encoder-moves-")
    assert run_dir(MmmConfig(run_name="mmm", unique_run_name=False)).name == "mmm"


def test_cli_train_encoder_reads_the_shipped_config(repo_root: Path) -> None:
    from typer.testing import CliRunner

    from rukh.cli import app
    from rukh.config import load_yaml

    cfg = load_yaml(repo_root / "configs" / "train" / "encoder-mmm.yaml", MmmConfig)
    assert cfg.block == 200 and cfg.input == "moves"
    assert cfg.encoder().n_layer == 8 and cfg.encoder().d_model == 384
    assert cfg.masking.mask_ratio == 0.8
    result = CliRunner().invoke(app, ["train", "encoder", "--help"])
    assert result.exit_code == 0, result.output
    for option in ("--config", "--resume", "--max-steps"):
        assert option in result.output
    # The decoder keeps the old spelling: `rukh train --config ...`, no subcommand.
    plain = CliRunner().invoke(app, ["train", "--help"])
    assert plain.exit_code == 0 and "--preset" in plain.output
    assert CliRunner().invoke(app, ["train"]).exit_code == 2
