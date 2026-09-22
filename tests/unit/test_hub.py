"""Tests for rukh.hub: the way back from the Hub to the paths the configs read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from rukh.cli import app
from rukh.hub import DATASET_MODULES, MODELS, catalogue, for_module, lookup, pull
from rukh.models import DecoderConfig, EncoderConfig, MoveDecoder, PositionEncoder
from rukh.train import load_any, load_checkpoint

pytestmark = pytest.mark.unit

TOY = DecoderConfig(vocab_size=64, n_layer=1, n_head=2, d_model=16, block=32)
TOY_ENCODER = EncoderConfig(
    input="squares", vocab_size=64, square_vocab=47, n_layer=1, n_head=2, d_model=16, block=69
)


def _published(folder: Path, config: dict[str, object], state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in state.items()}, str(folder / "model.safetensors"))


def _fake_hub(monkeypatch: pytest.MonkeyPatch, store: Path) -> list[tuple[str, str]]:
    """``hf_hub_download`` and ``snapshot_download`` served from ``store/<repo id>``."""
    calls: list[tuple[str, str]] = []

    def download(repo_id: str, filename: str, **_: object) -> str:
        calls.append((repo_id, filename))
        path = store / repo_id / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return str(path)

    def snapshot(
        repo_id: str, local_dir: str, allow_patterns: list[str] | None = None, **_: object
    ):
        calls.append((repo_id, "snapshot"))
        for path in (store / repo_id).rglob("*"):
            if path.is_file():
                target = Path(local_dir) / path.relative_to(store / repo_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        return local_dir

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download, raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot, raising=False)
    return calls


def test_every_module_after_m0_has_a_starting_kit() -> None:
    for module in ("m1", "m2", "m3", "m4", "m5", "m6"):
        assert for_module(module), module
    # M6 measures everything the course published: every model is part of its kit.
    assert {a.name for a in for_module("m6")} >= {
        a.name for a in catalogue() if a.kind != "dataset"
    }
    assert {a.name for a in for_module("m5")} >= {"medium-v4", "rukh-pairs-dpo"}
    assert {a.name for a in for_module("m3")} >= {"encoder-mmm-v4", "rukh-positions-eval"}


def test_the_catalogue_covers_every_published_dataset_once() -> None:
    names = [a.name for a in catalogue()]
    assert len(names) == len(set(names))
    assert set(DATASET_MODULES) == {a.name for a in catalogue() if a.kind == "dataset"}
    assert lookup("chorcat/rukh-medium").name == "medium-v4"
    assert lookup("rukh-rm").target == "checkpoints/rm/reward.pt"
    with pytest.raises(KeyError, match="nothing to pull called 'nope'"):
        lookup("nope")


def test_no_entry_lists_a_module_twice() -> None:
    """`for_module` uses `in`, so a repeat is invisible until someone counts modules per model."""
    for artefact in catalogue():
        assert len(artefact.modules) == len(set(artefact.modules)), artefact.name
    for name, modules in DATASET_MODULES.items():
        assert len(modules) == len(set(modules)), name


def test_model_targets_are_run_names_the_configs_use() -> None:
    """A pulled model must sit exactly where the shipped configs look for it."""
    from rukh.paths import package_root

    configs = (path for path in (package_root() / "configs").rglob("*.yaml") if path.name[0] != ".")
    referenced = set()
    for config in configs:
        for line in config.read_text(encoding="utf-8").splitlines():
            if "checkpoints/" in line and not line.strip().startswith("#"):
                referenced.add(line.split("checkpoints/", 1)[1].split()[0].strip("'\""))
    targets = {a.target.removeprefix("checkpoints/") for a in MODELS}
    # Runs the reader always trains themselves, because they take minutes and the lesson is in
    # running them: the 15 M masked-move encoder of M3 and the off-policy DPO arm of M5.
    local_only = {"encoder-mmm/best.pt", "medium-v4-dpo-offpolicy/dpo.pt"}
    # Every other checkpoint a shipped config starts from is something `rukh pull` can provide.
    missing = referenced - targets - local_only
    assert not missing, missing


def test_a_pulled_decoder_is_a_checkpoint_every_loader_reads(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    model = MoveDecoder(TOY)
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    config = {"stage": "medium-v4-greedy", "step": 48000, "vocab_hash": "v", **TOY.model_dump()}
    store = tmp_path / "hub"
    _published(store / "chorcat" / "rukh-medium", config, state)
    calls = _fake_hub(monkeypatch, store)

    path = pull(lookup("medium-v4"))
    assert path == rukh_home / "checkpoints" / "medium-v4" / "best.pt"
    payload = load_checkpoint(path)
    assert payload["step"] == 48000 and payload["vocab_hash"] == "v"
    assert payload["cfg"]["pulled_from"] == "chorcat/rukh-medium"
    loaded, _payload, kind = load_any(path)
    assert kind == "decoder"
    assert torch.equal(loaded.tokens.weight, model.tokens.weight)
    # A second pull leaves the file alone: it may be the reader's own run by now.
    (path).write_bytes(b"mine")
    assert pull(lookup("medium-v4")) == path and path.read_bytes() == b"mine"
    assert calls.count(("chorcat/rukh-medium", "config.json")) == 1
    pull(lookup("medium-v4"), force=True)
    assert path.read_bytes() != b"mine"


def test_a_pulled_encoder_keeps_its_pooling_for_the_heads(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    encoder = PositionEncoder(TOY_ENCODER)
    state = {k: v for k, v in encoder.state_dict().items() if k != "mlm_head.weight"}
    config = {
        "stage": "encoder-mmm-v4",
        "step": 16000,
        "pooling": "mean",
        **TOY_ENCODER.model_dump(),
    }
    store = tmp_path / "hub"
    _published(store / "chorcat" / "rukh-encoder-mmm", config, state)
    _fake_hub(monkeypatch, store)

    path = pull(lookup("encoder-mmm-v4"))
    payload = load_checkpoint(path)
    assert payload["model_cfg"]["input"] == "squares"
    assert payload["cfg"]["pooling"] == "mean"
    _model, _payload, kind = load_any(path)
    assert kind == "encoder"


def test_a_pulled_reward_model_has_the_layout_its_trainer_wrote(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = {"encoder.tokens.weight": torch.zeros(2, 2), "head.weight": torch.ones(1, 2)}
    config = {
        "model_type": "rukh-reward",
        "pooling": "mean",
        "encoder": TOY_ENCODER.model_dump(),
    }
    store = tmp_path / "hub"
    _published(store / "chorcat" / "rukh-rm", config, state)
    _fake_hub(monkeypatch, store)

    path = pull(lookup("rm"))
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"model", "encoder_config", "pooling", "config"}
    assert payload["encoder_config"]["input"] == "squares"
    assert torch.equal(payload["model"]["head.weight"], torch.ones(1, 2))


def test_adapters_and_datasets_land_in_their_folders(
    rukh_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "hub"
    adapter = store / "chorcat" / "rukh-lora-e4"
    adapter.mkdir(parents=True)
    (adapter / "adapter.safetensors").write_bytes(b"A")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    pairs = store / "chorcat" / "rukh-pairs-dpo"
    pairs.mkdir(parents=True)
    (pairs / "pairs.parquet").write_bytes(b"P")
    (pairs / "dpo-prompts.parquet").write_bytes(b"Q")
    (pairs / "manifest.json").write_text("{}", encoding="utf-8")
    _fake_hub(monkeypatch, store)

    assert pull(lookup("lora-e4")) == rukh_home / "checkpoints" / "lora-e4"
    assert (rukh_home / "checkpoints" / "lora-e4" / "adapter.safetensors").read_bytes() == b"A"
    assert pull(lookup("rukh-pairs-dpo")) == rukh_home / "data" / "pairs"
    assert (rukh_home / "data" / "pairs" / "dpo-prompts.parquet").read_bytes() == b"Q"


def test_the_cli_lists_and_refuses_unknown_names(rukh_home: Path) -> None:
    listed = CliRunner().invoke(app, ["pull", "--list"])
    assert listed.exit_code == 0
    assert "medium-v4" in listed.output and "rukh-pairs-dpo" in listed.output
    unknown = CliRunner().invoke(app, ["pull", "nothing-like-this"])
    assert unknown.exit_code == 2
    assert "nothing to pull called" in unknown.output
    empty = CliRunner().invoke(app, ["pull"])
    assert empty.exit_code == 2
    no_module = CliRunner().invoke(app, ["pull", "--module", "m9"])
    assert no_module.exit_code == 2
