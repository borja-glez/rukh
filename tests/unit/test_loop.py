"""Tests for rukh.train.loop: a toy CPU run that logs to MLflow, checkpoints and resumes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rukh.models import DecoderConfig, MoveDecoder
from rukh.tokenize.pack import META_FILE, STARTS_FILE, TOKENS_FILE, PackInfo
from rukh.train import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    param_groups,
    pick_device,
    run_dir,
    skip_batches,
    train,
)
from rukh.train.loop import forever, maybe_compile

pytestmark = pytest.mark.unit

VOCAB = 32
BLOCK = 16
TOY = DecoderConfig(vocab_size=VOCAB, n_layer=2, n_head=2, d_model=32, block=BLOCK)


def write_pack(directory: Path, n_games: int = 24, plies: int = 24) -> Path:
    """A toy pack shaped like ``pack.pack_month``: a cyclic, easily learnable token stream."""
    directory.mkdir(parents=True, exist_ok=True)
    tokens: list[int] = []
    starts: list[int] = []
    for game in range(n_games):
        starts.append(len(tokens))
        tokens.append(1)  # <bos>
        tokens.extend(8 + (game + ply) % 5 for ply in range(plies))
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


def toy_config(**overrides: object) -> TrainConfig:
    cfg = TrainConfig(
        preset="tiny",
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
        run_name="toy",
        unique_run_name=False,
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


def test_a_short_run_lowers_the_loss_and_writes_checkpoints(
    rukh_home: Path, tokens_dir: Path
) -> None:
    final = train(toy_config(), device="cpu")
    out = rukh_home / "checkpoints" / "toy"
    assert final == out / "step-6.pt"
    assert {p.name for p in out.glob("*.pt")} >= {"step-3.pt", "step-6.pt", "best.pt"}

    payload = load_checkpoint(final)
    assert payload["step"] == 6
    assert payload["vocab_hash"] == "toyhash"
    assert payload["data_manifest_sha"] is None  # no data/raw/manifest.json under RUKH_HOME
    assert payload["model_cfg"]["d_model"] == TOY.d_model

    run_id = last_run_id()
    losses = metric_history(run_id, "train/loss")
    assert len(losses) == 6
    assert losses[-1] < losses[0]
    for key in ("lr", "grad_norm", "tokens_per_s", "real_tokens_per_s"):
        assert len(metric_history(run_id, key)) == 6
    # Every position in the window is processed; only the non-pad targets are learned from.
    for window, real in zip(
        metric_history(run_id, "tokens_per_s"),
        metric_history(run_id, "real_tokens_per_s"),
        strict=True,
    ):
        assert 0.0 < real <= window
    assert len(metric_history(run_id, "val/loss")) == 2
    top1 = metric_history(run_id, "val/top1")
    assert len(top1) == 2 and all(0.0 <= value <= 1.0 for value in top1)


def test_a_run_name_is_unique_by_default(rukh_home: Path) -> None:
    """Two runs of the same config must not write the same ``step-*.pt`` series."""
    cfg = toy_config(unique_run_name=True)
    first = run_dir(cfg)
    assert first.name.startswith("toy-") and first.name != "toy"
    assert run_dir(TrainConfig(preset="tiny")).name.startswith("tiny-")
    assert run_dir(cfg.model_copy(update={"unique_run_name": False})).name == "toy"
    resumed = run_dir(cfg, resume=rukh_home / "checkpoints" / "toy" / "step-3.pt")
    assert resumed.name == "toy"  # a resumed run stays where it was


def test_skip_batches_winds_the_stream_forward(rukh_home: Path, tokens_dir: Path) -> None:
    from rukh.tokenize.loader import PackedDataset, make_loader

    def stream() -> object:
        loader = make_loader(PackedDataset(tokens_dir / "train", block=BLOCK), 4, seed=0)
        return forever(loader)

    straight = stream()
    for _ in range(5):
        next(straight)
    expected = next(straight)

    skipped = stream()
    assert skip_batches(skipped, 5) == 5
    assert torch.equal(next(skipped)[0], expected[0])
    assert skip_batches(stream(), 0) == 0


def test_resume_does_not_replay_the_windows_it_already_saw(
    rukh_home: Path, tokens_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rukh.train.loop as loop_module

    seen: list[int] = []
    real = loop_module.skip_batches
    monkeypatch.setattr(
        loop_module,
        "skip_batches",
        lambda batches, count: (seen.append(count), real(batches, count))[1],
    )
    cfg = toy_config(max_steps=3, ckpt_every=3)
    first = train(cfg, device="cpu")
    assert seen == [0]
    train(toy_config(max_steps=6, ckpt_every=3), resume=first, device="cpu")
    assert seen == [0, 3 * cfg.grad_accum]  # start_step * grad_accum batches


def test_a_resumed_run_keeps_one_mlflow_curve(rukh_home: Path, tokens_dir: Path) -> None:
    first = train(toy_config(max_steps=3, ckpt_every=3), device="cpu")
    run_id = load_checkpoint(first)["run_id"]
    assert run_id == last_run_id()

    train(toy_config(max_steps=6, ckpt_every=3), resume=first, device="cpu")
    assert load_checkpoint(first.parent / "step-6.pt")["run_id"] == run_id
    assert last_run_id() == run_id  # no second run was started
    losses = metric_history(run_id, "train/loss")
    assert len(losses) == 6  # steps 1-3 from the first half, 4-6 from the resumed one


def test_resume_continues_from_the_saved_step(rukh_home: Path, tokens_dir: Path) -> None:
    first = train(toy_config(max_steps=3, ckpt_every=3), device="cpu")
    assert first.name == "step-3.pt"
    second = train(toy_config(max_steps=6, ckpt_every=3), resume=first, device="cpu")
    assert second == first.parent / "step-6.pt"  # the resumed run stays in the same folder
    assert load_checkpoint(second)["step"] == 6

    restored = MoveDecoder(TOY)
    restored.load_state_dict(load_checkpoint(first)["model_state"])
    idx = torch.randint(1, VOCAB, (2, BLOCK))
    with torch.no_grad():
        before = restored.eval()(idx)[0]
    moved = MoveDecoder(TOY)
    moved.load_state_dict(load_checkpoint(second)["model_state"])
    with torch.no_grad():
        assert not torch.allclose(before, moved.eval()(idx)[0], atol=1e-5)


def test_a_mismatched_vocabulary_is_refused(rukh_home: Path, tokens_dir: Path) -> None:
    cfg = toy_config(model=TOY.model_copy(update={"vocab_size": 64}))
    with pytest.raises(ValueError, match="vocab_size"):
        train(cfg, device="cpu")


def test_too_few_windows_for_one_batch_is_refused(rukh_home: Path, tokens_dir: Path) -> None:
    with pytest.raises(ValueError, match="fewer than"):
        train(toy_config(batch_size=1000), device="cpu")


def test_weight_decay_only_applies_to_matrices() -> None:
    groups = param_groups(MoveDecoder(TOY), 0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in groups[0]["params"])
    assert all(p.dim() < 2 for p in groups[1]["params"])
    counted = sum(p.numel() for group in groups for p in group["params"])
    assert counted == MoveDecoder(TOY).num_params(non_embedding=False)


def test_evaluate_reports_a_loss_and_an_accuracy(rukh_home: Path, tokens_dir: Path) -> None:
    from rukh.tokenize.loader import PackedDataset, make_loader

    loader = make_loader(
        PackedDataset(tokens_dir / "val", block=BLOCK), 4, shuffle=False, drop_last=False
    )
    torch.manual_seed(0)
    loss, top1 = evaluate(MoveDecoder(TOY).eval(), loader, batches=2, device=torch.device("cpu"))
    assert loss > 0.0
    assert 0.0 <= top1 <= 1.0


def test_forever_repeats_the_loader(rukh_home: Path, tokens_dir: Path) -> None:
    from rukh.tokenize.loader import PackedDataset, make_loader

    loader = make_loader(PackedDataset(tokens_dir / "train", block=BLOCK), 8, seed=0)
    batches = [x for x, _ in zip(forever(loader), range(len(loader) + 2), strict=False)]
    assert len(batches) == len(loader) + 2


def test_a_failing_compile_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MoveDecoder(TOY)
    assert maybe_compile(model, enabled=False) is model

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no compiler on this machine")

    monkeypatch.setattr(torch, "compile", boom)
    assert maybe_compile(model, enabled=True) is model


def test_a_compile_that_only_fails_on_the_first_forward_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MoveDecoder(TOY)

    def lazy(_: object) -> object:
        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Compiler: cl is not found.")

        return explode

    monkeypatch.setattr(torch, "compile", lazy)
    sample = torch.ones((2, BLOCK), dtype=torch.long)
    assert maybe_compile(model, enabled=True, sample=sample) is model
    assert all(p.grad is None for p in model.parameters())


def test_device_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUKH_DEVICE", "cpu")
    assert pick_device() == "cpu"
    monkeypatch.delenv("RUKH_DEVICE")
    assert pick_device() in ("cpu", "cuda")


def test_the_decoder_config_follows_the_preset_and_block() -> None:
    cfg = TrainConfig(preset="tiny", block=64)
    assert cfg.decoder().n_layer == 6 and cfg.decoder().block == 64
    assert TrainConfig(model=TOY, block=BLOCK).decoder().d_model == 32


def test_cli_train_reads_the_shipped_configs(repo_root: Path) -> None:
    from typer.testing import CliRunner

    from rukh.cli import app
    from rukh.config import load_yaml

    for name in ("tiny", "small"):
        cfg = load_yaml(repo_root / "configs" / "train" / f"{name}.yaml", TrainConfig)
        assert cfg.preset == name and cfg.block == 200
    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    for option in ("--config", "--preset", "--resume", "--max-steps"):
        assert option in result.output


def test_toy_pack_matches_the_packer_layout(tmp_path: Path) -> None:
    directory = write_pack(tmp_path / "pack", n_games=3, plies=4)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    assert meta["n_games"] == 3
    assert np.load(directory / TOKENS_FILE).dtype == np.uint16
    assert np.load(directory / STARTS_FILE).tolist() == [0, 6, 12]
