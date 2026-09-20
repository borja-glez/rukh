"""QLoRA of a general language model on the same games, so the comparison is fair.

M4 owes the reader a number for a question the whole project raises by existing: why build a
115 M decoder for chess when a general model five times its size can be fine-tuned in an
afternoon? The only answer worth printing is a measurement, and a measurement is only worth
printing when the two sides differ in one thing. So:

- the **games** are the same, drawn from the same month and held out against the same validation
  split (``rukh.data.pgn_text``);
- the **metrics** are the same, because the fine-tuned Qwen is plugged into the harness of P2 as
  another ``MoveSource`` (``rukh.eval.qwen_source``) and plays the same Stockfish ladder;
- the **representation** is what differs, and that is the point. The decoder reads one token per
  legal move; Qwen reads ``1. e4 e5 2. Nf3`` cut into whatever pieces its BPE makes of it, and
  nothing in it forbids writing a move that does not exist.

This file is the training half, and it is deliberately somebody else's code: ``transformers`` for
the model, ``peft`` for the adapter, ``trl`` for the loop. The project has its own LoRA and its own
loop and does not need either here -- what it needs is the experience of the ecosystem path,
which is what a reader will actually reach for, and a run that is comparable to the published
QLoRA recipes rather than to ours.

Four-bit is attempted and not required. The memory it saves is the reason QLoRA exists, and at
0.6 B parameters on a 32 GB card there is nothing to save: the honest thing is to try it, report
whether it worked, and fall back to bf16 rather than pretend a technique was demonstrated when
the dependency would not build.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from rukh.config import BaseConfig
from rukh.paths import resolve

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

__all__ = ["QwenConfig", "QwenReport", "four_bit_config", "train_qwen"]


class QwenConfig(BaseConfig):
    """One QLoRA run of a general model over PGN text."""

    model: str = "Qwen/Qwen3-0.6B"
    data_dir: str = "data/pgn-text"
    out_dir: str = "checkpoints"
    run_name: str = "qwen3-pgn-qlora"
    max_length: int = Field(default=512, ge=64)
    """Tokens per sample. A 120-ply game in SAN with the rating header is 350-450 of Qwen's
    tokens, so 512 fits a whole game and truncates the longest few rather than half of them."""
    batch_size: int = Field(default=8, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    lr: float = Field(default=2e-4, gt=0)
    max_steps: int = Field(default=1500, ge=1)
    warmup: int = Field(default=100, ge=0)
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    four_bit: bool = True
    """Try to load the base model in 4 bits with ``bitsandbytes``; fall back to bf16 if not."""
    n_train: int | None = None
    """Samples to read from ``train.jsonl``; ``None`` reads them all."""
    seed: int = 42


class QwenReport(BaseConfig):
    """What the run did, including the parts that did not go as configured."""

    model: str
    adapter_dir: str
    four_bit: bool
    """Whether 4-bit quantisation was actually used, not whether it was asked for."""
    fallback_reason: str | None = None
    trainable_params: int
    total_params: int
    train_samples: int
    steps: int
    final_loss: float | None = None
    weights_memory_mb: float | None = None
    """What the loaded model occupies before a single step. This is the number QLoRA exists for,
    and the only one that separates the two precisions: the peak during training is dominated by
    activations and optimizer state, which 4-bit does not touch."""
    peak_memory_mb: float | None = None

    @property
    def trainable_share(self) -> float:
        return self.trainable_params / max(self.total_params, 1)


def four_bit_config() -> tuple[Any | None, str | None]:
    """``(BitsAndBytesConfig, None)`` when 4-bit is available, ``(None, reason)`` when it is not.

    NF4 with double quantisation and a bf16 compute dtype: the recipe of the QLoRA paper, so the
    run is comparable with the published ones rather than with a variant of our own.
    """
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        return None, f"bitsandbytes is not installed ({exc})"
    return (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        None,
    )


def _dataset(cfg: QwenConfig, split: str) -> Any:
    from datasets import load_dataset

    path = resolve(cfg.data_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `rukh data pgn-text` first")
    data = load_dataset("json", data_files={split: path.as_posix()})[split]
    if split == "train" and cfg.n_train is not None:
        data = data.select(range(min(cfg.n_train, len(data))))
    return data


def _count_params(model: Any) -> tuple[int, int]:
    """``(trainable, total)``, counting a 4-bit weight as the parameters it stands for.

    ``bitsandbytes`` stores two 4-bit values per byte, so ``numel()`` on a quantised weight
    returns half the parameter count and the "0.1 % trainable" line would come out twice as
    flattering as the truth.
    """
    trainable = total = 0
    for param in model.parameters():
        count = param.numel()
        if param.__class__.__name__ == "Params4bit":
            count *= 2
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def train_qwen(cfg: QwenConfig, device: str | None = None) -> QwenReport:
    """Fine-tune ``cfg.model`` on PGN text with LoRA, in 4 bits when that is possible."""
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    where = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantization, reason = (None, "not requested")
    if cfg.four_bit:
        quantization, reason = four_bit_config()
    if quantization is None and cfg.four_bit:
        log.warning("4-bit unavailable, training in bf16 instead: %s", reason)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": where} if quantization is not None else None,
    )
    if quantization is not None:
        model = prepare_model_for_kbit_training(model)
    elif where == "cuda":
        model = model.to(where)
    model.config.use_cache = False

    weights_memory = torch.cuda.memory_allocated() / 1e6 if where == "cuda" else None
    adapted = get_peft_model(
        model,
        PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.targets),
        ),
    )
    trainable, total = _count_params(adapted)
    log.info(
        "%s: %s trainable of %s (%.3f %%), %s",
        cfg.model,
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(total, 1),
        "4-bit" if quantization is not None else "bf16",
    )

    out_dir = resolve(cfg.out_dir) / cfg.run_name
    args = SFTConfig(
        output_dir=out_dir.as_posix(),
        dataset_text_field="text",
        max_length=cfg.max_length,
        packing=False,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="no",
        bf16=True,
        seed=cfg.seed,
        report_to=["mlflow"],
        run_name=cfg.run_name,
    )
    trainer = SFTTrainer(
        model=adapted,
        args=args,
        train_dataset=_dataset(cfg, "train"),
        processing_class=tokenizer,
    )
    if where == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e6 if where == "cuda" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    adapted.save_pretrained(out_dir.as_posix())
    tokenizer.save_pretrained(out_dir.as_posix())
    report = QwenReport(
        model=cfg.model,
        adapter_dir=out_dir.as_posix(),
        four_bit=quantization is not None,
        fallback_reason=None if quantization is not None else reason,
        trainable_params=trainable,
        total_params=total,
        train_samples=len(trainer.train_dataset),
        steps=cfg.max_steps,
        final_loss=float(result.training_loss) if result.training_loss is not None else None,
        weights_memory_mb=weights_memory,
        peak_memory_mb=peak,
    )
    (out_dir / "run.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def load_qwen_adapter(adapter_dir: Path | str, device: str | None = None) -> tuple[Any, Any]:
    """``(model, tokenizer)`` of a finished run, merged and in eval mode.

    Merged on purpose: the evaluation plays hundreds of games move by move, and a merged model
    runs the base matrices instead of the base plus two extra matmuls per adapted projection.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = Path(adapter_dir)
    config: Mapping[str, Any] = json.loads(
        (directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    where = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(
        str(config["base_model_name_or_path"]), dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, directory.as_posix())
    model = model.merge_and_unload().to(where).eval()
    tokenizer = AutoTokenizer.from_pretrained(directory.as_posix())
    return model, tokenizer
