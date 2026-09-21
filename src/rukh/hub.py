"""``rukh pull``: the published artefacts of a module, written where the configs expect them.

Every model of the course is on the Hub in the Hub's own layout -- ``config.json`` plus
``model.safetensors``, or a ``peft`` folder -- and every dataset under its own name. The
configs and the lessons, on the other hand, read files at fixed local paths:
``checkpoints/medium-v4/best.pt``, ``data/pairs/dpo-prompts.parquet``. This module maps one
onto the other, so a reader who skipped a module, or a machine that never trained anything, can
start the next module from exactly what was published, with one command:

    uv run rukh pull --module m4        # everything M4 starts from
    uv run rukh pull medium-v4 rukh-pairs-dpo

The local path of a model is its **run name** (``checkpoints/<run>/<file>``), the same spelling
``rukh.train.checkpoint.resolve_run`` accepts for a run the reader trained themselves. A pulled
model is therefore indistinguishable, to every config, from one trained locally.

Nothing here decides what is worth publishing; ``rukh publish`` does that. This is the way back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rukh import paths
from rukh.data.publish import DATASETS, DatasetSpec

log = logging.getLogger(__name__)

Kind = Literal["decoder", "encoder", "reward", "adapter", "peft", "dataset"]
OWNER = "chorcat"


@dataclass(frozen=True)
class Artefact:
    """One thing ``rukh pull`` knows how to bring back, and where it lands."""

    name: str
    """What the reader types: the run or stage name of a model, the Hub name of a dataset."""
    repo: str
    kind: Kind
    target: str
    """Path under the repository root: the checkpoint file, the adapter folder or the data dir."""
    modules: tuple[str, ...]
    """The modules that start from this artefact when the reader skipped what produced it."""
    note: str
    stage: str | None = None
    """Its row in the single results table, when it has one; ``rukh eval nightly`` measures it
    under that name so the row it rewrites is the row the cards and the course already cite."""
    measure: Literal["decoder", "encoder", "qwen", "none"] = "none"
    """How ``nightly`` measures it: the decoder suite (adapters are merged onto their base
    first), the encoder suite, the general-model suite, or not at all (datasets, the reward
    model, the pretraining encoder)."""
    base: str | None = None
    """For an adapter: the checkpoint it is merged onto before it is measured."""

    @property
    def repo_id(self) -> str:
        return self.repo if "/" in self.repo else f"{OWNER}/{self.repo}"


MODELS: tuple[Artefact, ...] = (
    Artefact(
        "tiny",
        "rukh-tiny",
        "decoder",
        "checkpoints/tiny/best.pt",
        ("m2",),
        "the 5 M decoder M2 iterates on (`configs/train/tiny.yaml`)",
        stage="tiny-greedy",
        measure="decoder",
    ),
    Artefact(
        "small",
        "rukh-small",
        "decoder",
        "checkpoints/small/best.pt",
        ("m2",),
        "the 39 M course decoder, `small-v3` on the corpus with 24 months of Elite games",
        stage="small-v3-greedy",
        measure="decoder",
    ),
    Artefact(
        "medium-v4",
        "rukh-medium",
        "decoder",
        "checkpoints/medium-v4/best.pt",
        ("m4", "m5"),
        "the 115 M decoder every M4 fine-tune and every M5 alignment run starts from",
        stage="medium-v4-greedy",
        measure="decoder",
    ),
    Artefact(
        "encoder-mmm-v4",
        "rukh-encoder-mmm",
        "encoder",
        "checkpoints/encoder-mmm-v4/best.pt",
        ("m3",),
        "the masked-move pretraining the published encoder's heads were fine-tuned on",
    ),
    Artefact(
        "encoder-v4",
        "rukh-encoder",
        "encoder",
        "checkpoints/encoder-v4/best.pt",
        ("m3",),
        "the encoder with its three heads, as published and served by the demo",
        stage="encoder-v4",
        measure="encoder",
    ),
    Artefact(
        "medium-elo",
        "rukh-medium-elo",
        "decoder",
        "checkpoints/medium-elo/step-3800.pt",
        ("m4",),
        "`medium-v4` fine-tuned on the Elo-balanced corpus (the last step, not `best.pt`)",
        stage="medium-elo",
        measure="decoder",
    ),
    Artefact(
        "medium-masters",
        "rukh-medium-masters",
        "decoder",
        "checkpoints/medium-masters/step-3800.pt",
        ("m4",),
        "`medium-v4` fine-tuned on Elite games (the last step, not `best.pt`)",
        stage="medium-masters",
        measure="decoder",
    ),
    Artefact(
        "lora-e4",
        "rukh-lora-e4",
        "adapter",
        "checkpoints/lora-e4",
        ("m4",),
        "the 1.6 MB style adapter that opens 1. e4",
        stage="lora-e4",
        measure="decoder",
        base="checkpoints/medium-v4/best.pt",
    ),
    Artefact(
        "lora-d4",
        "rukh-lora-d4",
        "adapter",
        "checkpoints/lora-d4",
        ("m4",),
        "the 1.6 MB style adapter that opens 1. d4",
        stage="lora-d4",
        measure="decoder",
        base="checkpoints/medium-v4/best.pt",
    ),
    Artefact(
        "qwen3-pgn-qlora",
        "rukh-qwen3-pgn-qlora",
        "peft",
        "checkpoints/qwen3-pgn-qlora",
        ("m4",),
        "the `peft` adapter of Qwen3-0.6B over PGN text, as `trl` wrote it",
        stage="qwen3-pgn-qlora",
        measure="qwen",
    ),
    Artefact(
        "rm",
        "rukh-rm",
        "reward",
        "checkpoints/rm/reward.pt",
        ("m5",),
        "the Bradley-Terry reward model over 69 square tokens",
    ),
    Artefact(
        "medium-v4-dpo-onpolicy",
        "rukh-medium-dpo",
        "decoder",
        "checkpoints/medium-v4-dpo-onpolicy/dpo.pt",
        ("m5",),
        "`medium-v4` after DPO on its own pairs, the aligned decoder the demo serves",
        stage="medium-v4-dpo-onpolicy-greedy",
        measure="decoder",
    ),
    Artefact(
        "medium-v4-grpo",
        "rukh-medium-grpo",
        "decoder",
        "checkpoints/medium-v4-grpo/grpo.pt",
        ("m5",),
        "`medium-v4` after GRPO against the verifiable reward",
        stage="medium-v4-grpo-greedy",
        measure="decoder",
    ),
)

DATASET_MODULES: dict[str, tuple[str, ...]] = {
    "rukh-games-1800": ("m1", "m2", "m3", "m4", "m5"),
    "rukh-tokenizer": ("m2", "m3", "m4", "m5"),
    "rukh-games-elite": ("m2", "m4"),
    "rukh-positions-eval": ("m2", "m3", "m5"),
    "rukh-puzzles-split": ("m2", "m3", "m4", "m5"),
    "rukh-elo-bins": (),
    "rukh-pairs-dpo": ("m5",),
    "rukh-pairs-onpolicy": ("m5",),
}
"""Which modules start from each dataset. ``rukh-elo-bins`` is M1's 100-Elo sample of the
1800+ corpus and nothing downstream reads it: M4 builds its own, from the club games it fetches.
"""


def _dataset_artefact(spec: DatasetSpec) -> Artefact:
    return Artefact(
        spec.name,
        spec.name,
        "dataset",
        spec.local_dir,
        DATASET_MODULES.get(spec.name, ()),
        spec.description.split(". ")[0].rstrip("."),
    )


def catalogue() -> list[Artefact]:
    """Every artefact ``rukh pull`` knows, models first and then the datasets."""
    return [*MODELS, *(_dataset_artefact(spec) for spec in DATASETS.values())]


def lookup(name: str) -> Artefact:
    """The artefact called ``name``: a run name, a dataset name or a Hub id."""
    for artefact in catalogue():
        if name in (artefact.name, artefact.repo, artefact.repo_id):
            return artefact
    known = ", ".join(a.name for a in catalogue())
    raise KeyError(f"nothing to pull called {name!r}; known: {known}")


def for_module(module: str) -> list[Artefact]:
    """What a reader who skipped everything before ``module`` needs on disk to start it."""
    return [artefact for artefact in catalogue() if module in artefact.modules]


def _download(repo_id: str, filename: str) -> Path:
    import huggingface_hub

    return Path(huggingface_hub.hf_hub_download(repo_id, filename))


def _load_state(repo_id: str) -> dict[str, Any]:
    """The weights of a Hub model, whichever of the two names they were published under."""
    import torch

    for name in ("model.safetensors", "pytorch_model.bin"):
        try:
            weights = _download(repo_id, name)
        except Exception as exc:  # noqa: BLE001 - a missing file only means "try the next one"
            log.debug("%s has no %s (%s)", repo_id, name, exc)
            continue
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file

            return dict(load_file(str(weights)))
        return dict(torch.load(weights, map_location="cpu", weights_only=True))
    raise FileNotFoundError(f"{repo_id} holds neither model.safetensors nor pytorch_model.bin")


def _model_payload(artefact: Artefact, config: dict[str, Any], state: dict[str, Any]) -> dict:
    """Fold a Hub layout back into the checkpoint payload every loader of this project reads.

    ``rukh publish`` writes the model's shape and the run's provenance into ``config.json``;
    this puts them back under the keys ``save_checkpoint`` uses. The heads of an encoder need
    the pooling they were trained with, which the card's config carries and ``build_heads``
    reads from the run config.
    """
    from rukh.models import DecoderConfig, EncoderConfig

    if artefact.kind == "reward":
        return {
            "model": state,
            "encoder_config": config["encoder"],
            "pooling": config.get("pooling"),
            "config": {"pulled_from": artefact.repo_id},
        }
    shape = EncoderConfig if artefact.kind == "encoder" else DecoderConfig
    model_cfg = {key: value for key, value in config.items() if key in shape.model_fields}
    run_cfg: dict[str, Any] = {"pulled_from": artefact.repo_id}
    if artefact.kind == "encoder" and config.get("pooling"):
        run_cfg["pooling"] = config["pooling"]
    return {
        "step": int(config.get("step") or 0),
        "model_state": state,
        "opt_state": None,
        "cfg": run_cfg,
        "model_cfg": model_cfg,
        "vocab_hash": config.get("vocab_hash"),
        "data_manifest_sha": config.get("data_manifest_sha"),
        "git_sha": config.get("git_sha"),
        "best_val": None,
    }


def pull(artefact: Artefact, force: bool = False) -> Path:
    """Bring ``artefact`` to its local path and return that path.

    An artefact already on disk is left alone unless ``force``: a reader's own training run,
    sitting at the same stable path, must never be replaced silently by the published one.
    """
    import torch

    target = paths.resolve(artefact.target)
    if target.exists() and not force:
        log.info("%s is already there; --force replaces it", target)
        return target
    if artefact.kind == "dataset":
        import huggingface_hub

        spec = DATASETS[artefact.name]
        target.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            artefact.repo_id,
            repo_type=spec.repo_type,
            local_dir=str(target),
            allow_patterns=[*spec.patterns, "manifest.json"],
        )
        return target
    if artefact.kind == "peft":
        import huggingface_hub

        target.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(artefact.repo_id, local_dir=str(target))
        return target
    if artefact.kind == "adapter":
        target.mkdir(parents=True, exist_ok=True)
        for name in ("adapter.safetensors", "adapter_config.json"):
            source = _download(artefact.repo_id, name)
            (target / name).write_bytes(source.read_bytes())
        return target
    config = json.loads(_download(artefact.repo_id, "config.json").read_text(encoding="utf-8"))
    state = _load_state(artefact.repo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(_model_payload(artefact, config, state), tmp)
    tmp.replace(target)
    return target
