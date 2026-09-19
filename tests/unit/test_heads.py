"""Tests for the three heads and their staged fine-tuning: gradients, freezing and the curve."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch
from conftest import cli_options

from rukh.models import EncoderConfig, MultiHead, PositionEncoder
from rukh.models.heads import HEADS, BlunderHead, HeadWeights, ResultHead, ValueHead
from rukh.models.squares import SQUARE_TOKENS, fen_to_tokens
from rukh.train import HeadsConfig, freeze_encoder, label_curve, train_heads
from rukh.train.checkpoint import load_checkpoint, save_checkpoint
from rukh.train.heads import LabelledPositions, build_frames, set_training_mode

pytestmark = pytest.mark.unit

TOY = EncoderConfig(input="squares", n_layer=3, n_head=2, d_model=32, dropout=0.0)
# Four boards whose evaluation the encoder can actually read off the pieces: the full set, the
# same without Black's queen, without White's queen, and without either. A toy task has to be
# learnable or "the loss went down" measures nothing.
BOARDS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR",
)
BOARD_CP = (0, 300, -300, 0)
BOARD_RESULT = ("1/2-1/2", "1-0", "0-1", "1/2-1/2")
WHITE_TO_MOVE = f"{BOARDS[0]} w KQkq -"


def fen_at(ply: int) -> str:
    """The position after ``ply`` half-moves: a ply-N position has White to move on even N."""
    return f"{BOARDS[ply % 4]} {'w' if ply % 2 == 0 else 'b'} KQkq -"


def toy_encoder(seed: int = 0) -> PositionEncoder:
    torch.manual_seed(seed)
    return PositionEncoder(TOY)


def toy_batch(rows: int = 4) -> dict[str, torch.Tensor]:
    idx = torch.tensor([fen_to_tokens(WHITE_TO_MOVE) for _ in range(rows)])
    return {
        "idx": idx,
        "value": torch.linspace(-0.9, 0.9, rows),
        "blunder": torch.tensor([1.0, 0.0] * (rows // 2)),
        "blunder_mask": torch.tensor([True, False] * (rows // 2)),
        "result": torch.tensor([0, 1, 2, 0][:rows]),
    }


def eval_rows(games: int = 40, plies: tuple[int, ...] = (1, 2, 3, 4)) -> pl.DataFrame:
    """A toy ``positions-eval.parquet`` whose labels are a function of the position."""
    return pl.DataFrame(
        [
            {
                "fen": fen_at(ply),
                "game_id": game,
                "ply": ply,
                "last_move": "e2e4",
                "result": BOARD_RESULT[ply % 4],
                "phase": "middlegame",
                "cp": BOARD_CP[ply % 4],
                "mate": None,
                "best_move": "e2e4",
            }
            for game in range(games)
            for ply in plies
        ]
    )


@pytest.fixture
def labels_source(rukh_home: Path) -> str:
    path = rukh_home / "evals" / "positions-eval.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows().write_parquet(path)
    return "evals/positions-eval.parquet"


def toy_heads_config(labels_source: str, **overrides: object) -> HeadsConfig:
    cfg = HeadsConfig(
        model=TOY,
        labels={"positions_eval": labels_source, "out_dir": "labels", "val_fraction": 0.25},
        batch_size=8,
        grad_accum=1,
        lr=1e-2,
        warmup=1,
        max_steps=4,
        precision="fp32",
        compile=False,
        eval_every=2,
        eval_batches=2,
        ckpt_every=4,
        log_every=1,
        run_name="toy-heads",
        unique_run_name=False,
    )
    return cfg.model_copy(update=dict(overrides))


def test_each_head_has_the_shape_its_label_needs() -> None:
    pooled = torch.randn(5, 32)
    value = ValueHead(32)(pooled)
    assert value.shape == (5,) and bool((value.abs() < 1.0).all())
    assert BlunderHead(32)(pooled).shape == (5,)  # a logit, not a probability
    assert ResultHead(32)(pooled).shape == (5, 3)


def test_multihead_answers_the_three_questions_at_once() -> None:
    model = MultiHead(toy_encoder())
    batch = toy_batch()
    outputs = model(batch["idx"])
    assert set(outputs) == set(HEADS)
    assert outputs["value"].shape == (4,)
    assert outputs["blunder"].shape == (4,)
    assert outputs["result"].shape == (4, 3)
    assert model.pooled(batch["idx"]).shape == (4, 32)
    assert model.encoder.cfg.seq == SQUARE_TOKENS


def test_multihead_propagates_the_three_gradients() -> None:
    model = MultiHead(toy_encoder(), HeadWeights(value=1.0, blunder=1.0, result=1.0))
    batch = toy_batch()
    total, parts = model.loss(model(batch["idx"]), batch)
    assert set(parts) == set(HEADS)
    assert all(torch.isfinite(part) for part in parts.values())
    total.backward()
    for name in HEADS:
        grad = getattr(model, name).proj.weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0.0, name
    body = model.encoder.blocks[0].mlp.fc.weight.grad
    assert body is not None and float(body.abs().sum()) > 0.0  # the encoder gets them too


def test_a_head_with_weight_zero_stops_contributing() -> None:
    model = MultiHead(toy_encoder(), HeadWeights(value=1.0, blunder=0.0, result=0.0))
    batch = toy_batch()
    total, parts = model.loss(model(batch["idx"]), batch)
    assert torch.allclose(total, parts["value"], atol=1e-6)
    total.backward()
    assert float(model.blunder.proj.weight.grad.abs().sum()) == 0.0
    assert float(model.value.proj.weight.grad.abs().sum()) > 0.0


def test_a_batch_without_a_blunder_label_costs_nothing_instead_of_nan() -> None:
    model = MultiHead(toy_encoder())
    batch = toy_batch()
    batch["blunder_mask"] = torch.zeros_like(batch["blunder_mask"])
    total, parts = model.loss(model(batch["idx"]), batch)
    assert float(parts["blunder"]) == 0.0
    assert torch.isfinite(total)


@pytest.mark.parametrize("mode", ["probe", "last-n", "full"])
def test_the_mode_decides_what_is_allowed_to_move(mode: str) -> None:
    model = MultiHead(toy_encoder())
    trainable = freeze_encoder(model, mode, last_n=1)  # type: ignore[arg-type]
    total = sum(1 for _ in model.encoder.parameters())
    one_block = sum(1 for _ in model.encoder.blocks[-1].parameters())
    final_norm = sum(1 for _ in model.encoder.ln_f.parameters())
    expected = {"probe": 0, "last-n": one_block + final_norm, "full": total}[mode]
    assert trainable == expected
    assert all(param.requires_grad for param in model.head_parameters())
    if mode == "last-n":  # the last block and the final norm, nothing earlier
        assert all(not p.requires_grad for p in model.encoder.blocks[0].parameters())
        assert all(p.requires_grad for p in model.encoder.blocks[-1].parameters())
        assert all(p.requires_grad for p in model.encoder.ln_f.parameters())


def test_a_frozen_encoder_is_kept_in_eval_mode() -> None:
    model = MultiHead(toy_encoder())
    set_training_mode(model, "probe")
    assert model.training and not model.encoder.training  # no dropout on a frozen feature
    set_training_mode(model, "full")
    assert model.encoder.training
    # Under last-n the frozen prefix is frozen for dropout too: only what is learning is
    # regularised, and the early blocks give the same vector for the same position every epoch.
    set_training_mode(model, "last-n", last_n=1)
    assert model.training and not model.encoder.blocks[0].training
    assert model.encoder.blocks[-1].training and model.encoder.ln_f.training


def test_the_dataset_turns_a_row_into_69_tokens_and_three_labels(labels_source: str) -> None:
    cfg = toy_heads_config(labels_source)
    train, val = build_frames(cfg)
    dataset = LabelledPositions(train)
    item = dataset[0]
    assert item["idx"].shape == (SQUARE_TOKENS,)
    assert item["value"].dtype == torch.float32
    assert item["result"].dtype == torch.int64
    assert item["blunder_mask"].dtype == torch.bool
    assert len(dataset) == train.height and val.height
    with pytest.raises(ValueError, match="needs the games of the labelled rows"):
        LabelledPositions(train, "moves")


def test_probe_leaves_the_encoder_bit_identical(rukh_home: Path, labels_source: str) -> None:
    torch.manual_seed(0)
    pretrained = PositionEncoder(TOY)
    before = {key: value.clone() for key, value in pretrained.state_dict().items()}
    ckpt = save_checkpoint(
        rukh_home / "encoder.pt",
        step=0,
        model=pretrained,
        optimizer=None,
        cfg={},
        model_cfg=TOY.model_dump(mode="json"),
    )

    result = train_heads(toy_heads_config(labels_source, encoder_ckpt="encoder.pt"), device="cpu")
    after = load_checkpoint(Path(result.checkpoint))["model_state"]
    for key, value in before.items():
        assert torch.equal(after[f"encoder.{key}"], value), key
    fresh = MultiHead(pretrained)
    assert any(
        not torch.equal(after[f"{name}.proj.weight"], getattr(fresh, name).proj.weight)
        for name in HEADS
    )  # the heads did move
    assert result.mode == "probe" and result.train_labels > 0
    assert "val/loss" in result.metrics

    full_cfg = toy_heads_config(
        labels_source, encoder_ckpt="encoder.pt", mode="full", run_name="toy-full"
    )
    moved = train_heads(full_cfg, device="cpu")
    full = load_checkpoint(Path(moved.checkpoint))["model_state"]
    assert not torch.equal(full["encoder.blocks.0.mlp.fc.weight"], before["blocks.0.mlp.fc.weight"])
    assert ckpt.is_file()


def test_a_short_run_lowers_the_loss_and_writes_a_checkpoint(
    rukh_home: Path, labels_source: str
) -> None:
    result = train_heads(toy_heads_config(labels_source, mode="full", max_steps=8), device="cpu")
    out = rukh_home / "checkpoints" / "toy-heads"
    assert Path(result.checkpoint) == out / "step-8.pt"
    assert (out / "best.pt").is_file()
    assert result.metrics["val/result_acc"] >= 0.0
    assert 0.0 <= result.metrics["val/blunder_acc"] <= 1.0

    import mlflow

    from rukh.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    runs = mlflow.search_runs(experiment_names=["rukh"], output_format="list")
    run_id = max(runs, key=lambda r: r.info.start_time).info.run_id
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri())
    losses = [p.value for p in client.get_metric_history(run_id, "train/loss")]
    assert len(losses) == 8
    assert sum(losses[4:]) / 4 < sum(losses[:4]) / 4  # noisy by the batch, down on average
    for name in HEADS:
        assert client.get_metric_history(run_id, f"train/{name}_loss")


def test_the_label_curve_trains_one_run_per_fraction(rukh_home: Path, labels_source: str) -> None:
    cfg = toy_heads_config(labels_source, max_steps=2, ckpt_every=2, eval_every=2)
    results = label_curve(cfg, [0.5, 1.0], device="cpu")
    assert [result.fraction for result in results] == [0.5, 1.0]
    assert results[0].train_labels < results[1].train_labels
    assert results[0].val_labels == results[1].val_labels  # the same held-out games
    assert {Path(result.checkpoint).parent.name for result in results} == {
        "toy-heads-50",
        "toy-heads-100",
    }


def test_the_config_refuses_a_mismatched_encoder(rukh_home: Path, labels_source: str) -> None:
    torch.manual_seed(0)
    moves = PositionEncoder(EncoderConfig(input="moves", n_layer=2, n_head=2, d_model=32))
    save_checkpoint(
        rukh_home / "moves.pt",
        step=0,
        model=moves,
        optimizer=None,
        cfg={},
        model_cfg=moves.cfg.model_dump(mode="json"),
    )
    cfg = toy_heads_config(labels_source, encoder_ckpt="moves.pt")
    with pytest.raises(ValueError, match="was trained on the 'moves' scheme"):
        train_heads(cfg, device="cpu")
    with pytest.raises(ValueError):
        HeadsConfig(fraction=0.0)  # a run on no labels at all is a mistake, not a corner case


def test_cli_train_heads_reads_the_shipped_config(repo_root: Path) -> None:
    from typer.testing import CliRunner

    from rukh.cli import app
    from rukh.config import load_yaml

    cfg = load_yaml(repo_root / "configs" / "train" / "encoder-heads.yaml", HeadsConfig)
    assert cfg.mode == "probe" and cfg.input == "squares"
    assert cfg.curve == [0.1, 0.25, 0.5, 1.0]
    assert cfg.labels.blunder_cp == 100 and cfg.labels.value_scale == 400.0
    result = CliRunner().invoke(app, ["train", "heads", "--help"])
    assert result.exit_code == 0, result.output
    assert {"--config", "--mode", "--fraction", "--curve"} <= cli_options("train", "heads")


# --- the `moves` scheme: the line that reached the position ------------------------------------

GAME_UCI = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"
"""Eight real half-moves, so a prefix of any ply in ``eval_rows`` exists."""


def toy_games(frame: pl.DataFrame) -> pl.DataFrame:
    """The ``rukh.data.labels.game_moves`` frame for ``eval_rows``: one line per game."""
    games = sorted({int(game) for game in frame["game_id"].to_list()})
    return pl.DataFrame(
        {
            "game_id": games,
            "uci": [GAME_UCI] * len(games),
            "white_elo": [1850] * len(games),
            "black_elo": [1920] * len(games),
        }
    )


def test_a_moves_item_is_the_header_plus_the_prefix_of_the_game(labels_source: str) -> None:
    from rukh.tokenize.uci_vocab import UciTokenizer, elo_token

    cfg = toy_heads_config(labels_source, input="moves", block=200)
    train, _val = build_frames(cfg)
    dataset = LabelledPositions(train, "moves", toy_games(train), cfg.block)
    tok = UciTokenizer()
    for index in range(min(8, len(dataset))):
        ids = dataset.tokens(index)
        ply = dataset.plies[index]
        assert tok.decode(ids[:3]) == ["<bos>", elo_token(1850, "w"), elo_token(1920, "b")]
        # ``ply`` counts the move that led to this position, so it is inside the prefix.
        assert tok.decode(ids[3:]) == GAME_UCI.split()[:ply]
        assert len(ids) == 3 + ply


def test_a_prefix_longer_than_the_block_is_cropped_after_the_header(labels_source: str) -> None:
    from rukh.tokenize.uci_vocab import UciTokenizer

    cfg = toy_heads_config(labels_source, input="moves", block=5)
    train, _val = build_frames(cfg)
    dataset = LabelledPositions(train, "moves", toy_games(train), cfg.block)
    tok = UciTokenizer()
    longest = max(range(len(dataset)), key=lambda index: dataset.plies[index])
    ply = dataset.plies[longest]
    assert ply + 3 > cfg.block  # otherwise this test is not testing the crop
    ids = dataset.tokens(longest)
    assert len(ids) == cfg.block
    # The Elo conditioning survives the crop and the *oldest* moves are the ones dropped.
    assert tok.decode(ids[:3])[0] == "<bos>"
    assert tok.decode(ids[3:]) == GAME_UCI.split()[:ply][-(cfg.block - 3) :]


def test_padding_does_not_reach_the_heads(labels_source: str) -> None:
    from rukh.train.heads import collate

    cfg = toy_heads_config(labels_source, input="moves", block=200)
    train, _val = build_frames(cfg)
    dataset = LabelledPositions(train, "moves", toy_games(train), cfg.block)
    short = min(range(len(dataset)), key=lambda index: dataset.plies[index])
    long = max(range(len(dataset)), key=lambda index: dataset.plies[index])
    assert dataset.plies[short] < dataset.plies[long]

    torch.manual_seed(0)
    model = MultiHead(
        PositionEncoder(EncoderConfig(input="moves", n_layer=2, n_head=2, d_model=32, dropout=0.0))
    ).eval()
    padded = collate([dataset[short], dataset[long]])
    alone = collate([dataset[short]])
    assert bool(padded["attention_mask"][0, dataset.plies[short] + 3 :].eq(False).all())
    with torch.no_grad():
        mixed = model(padded["idx"], padded["attention_mask"])
        tight = model(alone["idx"], alone["attention_mask"])
    for head in HEADS:
        assert torch.allclose(mixed[head][:1], tight[head], atol=1e-6)


def test_each_scheme_only_loads_its_own_pretrained_encoder(
    rukh_home: Path, labels_source: str
) -> None:
    from rukh.train.heads import load_encoder_for

    torch.manual_seed(0)
    for scheme in ("moves", "squares"):
        encoder = PositionEncoder(
            EncoderConfig(input=scheme, n_layer=2, n_head=2, d_model=32, block=200)
        )
        save_checkpoint(
            rukh_home / f"{scheme}.pt",
            step=0,
            model=encoder,
            optimizer=None,
            cfg={},
            model_cfg=encoder.cfg.model_dump(mode="json"),
        )
    where = torch.device("cpu")
    for scheme, other in (("moves", "squares"), ("squares", "moves")):
        cfg = toy_heads_config(labels_source, input=scheme, encoder_ckpt=f"{scheme}.pt")
        assert load_encoder_for(cfg, where).cfg.input == scheme
        wrong = cfg.model_copy(update={"encoder_ckpt": f"{other}.pt"})
        with pytest.raises(ValueError, match=f"was trained on the {other!r} scheme"):
            load_encoder_for(wrong, where)
