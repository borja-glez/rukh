"""Tests for rukh.eval.nightly: the catalogue drives the run, and the run says what it did."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from rukh.cli import app
from rukh.eval import nightly
from rukh.hub import catalogue
from rukh.models import DecoderConfig, MoveDecoder
from rukh.train import load_model, save_checkpoint

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=64, n_layer=1, n_head=2, d_model=16, block=32)


def test_every_published_model_the_table_cites_is_in_the_plan() -> None:
    """The stages of `docs/benchmarks.md` that a reader can pull are all measured by nightly."""
    stages = {a.stage for a in nightly.plan()}
    assert stages >= {
        "tiny-greedy",
        "small-v3-greedy",
        "medium-v4-greedy",
        "medium-elo",
        "medium-masters",
        "lora-e4",
        "lora-d4",
        "qwen3-pgn-qlora",
        "medium-v4-dpo-onpolicy-greedy",
        "medium-v4-grpo-greedy",
        "encoder-v4",
    }
    # Nothing without a way to measure it: datasets, the reward model, the pretraining encoder.
    assert all(a.measure != "none" for a in nightly.plan())
    assert {a.name for a in catalogue() if a.measure == "none"} >= {"rm", "encoder-mmm-v4"}
    assert [a.name for a in nightly.plan({"tiny", "encoder-v4"})] == ["tiny", "encoder-v4"]
    assert [a.stage for a in nightly.plan({"medium-elo"})] == ["medium-elo"]


def _toy(path: Path, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    return save_checkpoint(
        path, step=1, model=MoveDecoder(TOY), optimizer=None, cfg={}, model_cfg=TOY.model_dump()
    )


def test_a_dry_run_lists_the_plan_and_writes_nothing(rukh_home: Path) -> None:
    report = nightly.run_nightly(dry_run=True, only={"tiny", "encoder-v4"})
    assert [r.status for r in report.records] == ["planned", "planned"]
    assert not (rukh_home / "artifacts" / "eval" / "nightly.json").exists()
    assert nightly.render_plan(report).count("planned") == 2


def test_the_run_measures_in_catalogue_order_and_records_what_it_did(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured: list[tuple[str, str]] = []

    def fake_decoder(checkpoint, config, stage, use_cache, device):  # noqa: ANN001
        measured.append((stage, Path(checkpoint).name))
        return None

    def fake_encoder(checkpoint, config, stage, use_cache, device):  # noqa: ANN001
        measured.append((stage, Path(checkpoint).name))
        return None

    monkeypatch.setattr(nightly, "_run_decoder", fake_decoder)
    monkeypatch.setattr(nightly, "_run_encoder", fake_encoder)
    monkeypatch.setattr(
        nightly, "_run_qwen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no gpu"))
    )
    # `tiny` under its stamped run folder: the stable name resolves to it.
    _toy(rukh_home / "checkpoints" / "tiny-20260919-060101" / "best.pt")
    _toy(rukh_home / "checkpoints" / "encoder-heads-v4" / "step-4000.pt")
    (rukh_home / "checkpoints" / "qwen3-pgn-qlora").mkdir(parents=True)
    results = rukh_home / "artifacts" / "web" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"rows": [{"stage": "kept", "params": 1}]}), encoding="utf-8")
    benchmarks = rukh_home / "docs" / "benchmarks.md"
    benchmarks.parent.mkdir(parents=True)
    benchmarks.write_text("# B\n\ntexto\n\n## Resultados\n\nold\n", encoding="utf-8")

    report = nightly.run_nightly(
        only={"tiny", "encoder-v4", "qwen3-pgn-qlora", "medium-v4"},
        pull_missing=False,
        results=results,
        benchmarks=benchmarks,
    )
    assert measured == [("tiny-greedy", "best.pt"), ("encoder-v4", "step-4000.pt")]
    by_name = {r.name: r for r in report.records}
    assert by_name["tiny"].status == "measured" and by_name["tiny"].model_sha
    assert by_name["tiny"].checkpoint.endswith("tiny-20260919-060101/best.pt")
    assert by_name["qwen3-pgn-qlora"].status == "failed"
    assert "no gpu" in (by_name["qwen3-pgn-qlora"].detail or "")
    # A model that is not on disk and may not be pulled fails on its own, not the run.
    assert by_name["medium-v4"].status == "failed"
    assert "rukh pull medium-v4" in (by_name["medium-v4"].detail or "")
    # The rows nightly did not measure are untouched; the benchmarks doc was rewritten.
    assert json.loads(results.read_text(encoding="utf-8"))["rows"] == [
        {"stage": "kept", "params": 1}
    ]
    assert "texto" in benchmarks.read_text(encoding="utf-8")
    assert "old" not in benchmarks.read_text(encoding="utf-8")
    written = json.loads((rukh_home / "artifacts" / "eval" / "nightly.json").read_text("utf-8"))
    assert [r["name"] for r in written["records"]] == [
        "tiny",
        "medium-v4",
        "encoder-v4",
        "qwen3-pgn-qlora",
    ]


def test_an_adapter_is_merged_onto_its_base_before_it_is_measured(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rukh.models.lora import LoraConfig, apply_lora, save_adapter

    base = _toy(rukh_home / "checkpoints" / "medium-v4" / "best.pt")
    torch.manual_seed(1)
    model = MoveDecoder(TOY)
    model.load_state_dict(load_model(base)[0].state_dict())
    apply_lora(model, LoraConfig(r=2, alpha=4, targets=["q", "v"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".b." in name:
                parameter.add_(0.5)
    adapter_dir = rukh_home / "checkpoints" / "lora-e4"
    adapter_dir.mkdir(parents=True)
    save_adapter(
        model, adapter_dir / "adapter.safetensors", LoraConfig(r=2, alpha=4, targets=["q", "v"])
    )

    seen: dict[str, Path] = {}
    monkeypatch.setattr(
        nightly, "_run_decoder", lambda ckpt, *a, **k: seen.__setitem__("ckpt", Path(ckpt))
    )
    report = nightly.run_nightly(only={"lora-e4"}, pull_missing=False)
    assert report.records[0].status == "measured"
    merged, payload = load_model(seen["ckpt"])
    assert payload["cfg"]["adapter"].endswith("lora-e4")
    plain, _ = load_model(base)
    # The merged weights differ from the base exactly where the adapter pushed.
    assert not torch.equal(merged.blocks[0].attn.qkv.weight, plain.blocks[0].attn.qkv.weight)
    assert not any(".lora" in k or ".a." in k for k in merged.state_dict())


def test_the_cli_dry_run_prints_the_plan(rukh_home: Path) -> None:
    result = CliRunner().invoke(app, ["eval", "nightly", "--dry-run", "--only", "tiny,rm"])
    assert result.exit_code == 0, result.output
    assert "tiny-greedy" in result.output and "planned" in result.output
    assert "rm" not in result.output.replace("tiny-greedy", "")


def test_an_adapter_base_is_looked_up_and_not_hard_coded() -> None:
    """Merging onto the wrong base does not fail, it just measures another model."""
    from rukh.eval.nightly import _artefact_at

    assert _artefact_at("checkpoints/medium-v4/best.pt").name == "medium-v4"
    assert _artefact_at("checkpoints/small-v3/best.pt").name == "small"
    with pytest.raises(FileNotFoundError):
        _artefact_at("checkpoints/nothing-builds-this/best.pt")
