"""Export: ONNX for the browser, fp16 and int8 quantization and PyTorch parity.

``export_all`` is the whole of ``rukh export``: write the fp32 graph, derive the fp16 and int8
files the two browser backends need, and check every file it produced against PyTorch. It
exports two kinds of model - the decoder's next move and the encoder's ``value`` and ``blunder``
heads - through the same path, with one wrapper and one parity function each.

The parity is not only printed: it is written to ``parity.json`` next to the ONNX files, so that
whoever publishes the model can quote the measurement instead of repeating the claim. The file
travels with the ONNX files to the Hub, because a number nobody can check is not evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rukh import __version__
from rukh.data.labels import LabelsConfig
from rukh.export.adapter import (
    A_INPUT,
    B_INPUT,
    adaptable_inputs,
    export_adaptable_onnx,
    zero_inputs,
)
from rukh.export.embed import EmbedResult, embed_positions, fen4
from rukh.export.onnx import (
    BLUNDER_OUTPUT,
    DEFAULT_OPSET,
    ENCODER_OUTPUTS,
    INPUT_NAME,
    MODEL_NAME,
    OUTPUT_NAME,
    VALUE_OUTPUT,
    EncoderHeads,
    ExportResult,
    LastStepLogits,
    clear_value_info,
    export_encoder_onnx,
    export_onnx,
    read_metadata,
    sequence_lengths,
    set_metadata,
    target_path,
    verify_dynamic_seq,
    write_metadata,
)
from rukh.export.parity import (
    DEFAULT_GAMES,
    EncoderParityResult,
    ParityResult,
    encoder_parity,
    encoder_parity_positions,
    label_positions,
    parity,
    parity_positions,
    random_prefixes,
    validation_prefixes,
)
from rukh.export.quantize import (
    FP16_NAME,
    INT8_NAME,
    QUANTIZED_OPS,
    QuantizeResult,
    quantize_int8,
    to_fp16,
)
from rukh.models.lora import (
    ADAPTER_FILE,
    AdapterLayout,
    LoraConfig,
    adapter_layout,
    apply_lora,
)

KINDS = ("decoder", "encoder")
"""What ``rukh export --kind`` accepts."""

PARITY_NAME = "parity.json"
"""Where ``export_all`` records the parity it measured, next to the files it measured."""

PARITY_VERSION = 1
"""Schema of ``parity.json``; a reader that does not know this number should not trust the rest."""

PARITY_MEASURES = {
    "decoder": "the argmax move",
    "encoder": "the blunder decision at p >= 0.5",
}
"""What ``agreement`` counts for each kind of model, spelled out for whoever reads the file."""


class ExportBundle(BaseModel):
    """Everything one ``rukh export`` produced."""

    model_config = ConfigDict(extra="forbid")

    onnx: ExportResult
    fp16: QuantizeResult | None = None
    int8: QuantizeResult | None = None
    parity: dict[str, ParityResult] = {}
    """Parity per file kind: ``fp32``, ``fp16``, ``int8``."""
    adapter_inputs: bool = False
    """Whether the graph takes its LoRA factors as inputs instead of baking them in."""
    adapter_parity: dict[str, ParityResult] = {}
    """The same parity measured again with a real adapter fed into both sides."""
    adapter_name: str | None = None
    """Which adapter that second measurement used, by the folder it was read from."""
    heads_parity: dict[str, EncoderParityResult] = {}
    """The encoder's parity per file kind: the blunder decision and the drift of ``value``."""
    parity_source: str | None = None
    """``validation`` (the positions of the design spec 02 §6), ``random-walk``, or, for the
    encoder, ``validation-labels``."""
    parity_warning: str | None = None
    parity_path: str | None = None
    """``parity.json``, written next to the ONNX files when parity was measured."""


def parity_payload(bundle: ExportBundle, files: dict[str, str]) -> dict[str, Any]:
    """The parity of one export as ``parity.json`` records it.

    One entry per precision, each naming the file it was measured on, so that a card can quote
    the number for the very file it is telling the reader to download. The population is part of
    the record: an agreement measured on random legal walks is not the same claim as one measured
    on validation positions, and the difference is exactly what ``parity_source`` carries.
    """
    kind = bundle.onnx.kind
    measured: dict[str, Any] = dict(bundle.heads_parity or bundle.parity)
    precisions: dict[str, dict[str, float | str]] = {}
    for name, result in measured.items():
        entry: dict[str, float | str] = {
            "file": Path(files[name]).name,
            "agreement": result.agreement,
        }
        if isinstance(result, EncoderParityResult):
            entry["max_abs_value_delta"] = result.max_abs_value_delta
            entry["max_abs_blunder_delta"] = result.max_abs_blunder_delta
        else:
            entry["max_abs_logit_delta"] = result.max_abs_logit_delta
        precisions[name] = entry
    first = next(iter(measured.values()))
    payload: dict[str, Any] = {
        "version": PARITY_VERSION,
        "kind": kind,
        "measures": PARITY_MEASURES[kind],
        "positions": first.positions,
        "source": bundle.parity_source,
        "exporter": bundle.onnx.exporter,
        "rukh_version": __version__,
        "warning": bundle.parity_warning,
        "precisions": precisions,
    }
    if bundle.adapter_inputs:
        # A graph with the factors as inputs has two claims to make, and they are different
        # claims: fed zeros it has to be the plain model, and fed a real adapter it has to be
        # the adapted model. The first is what `precisions` above measured.
        payload["adapter_inputs"] = True
        if bundle.adapter_parity:
            payload["with_adapter"] = {
                "adapter": bundle.adapter_name,
                "precisions": {
                    name: {
                        "file": Path(files[name]).name,
                        "agreement": result.agreement,
                        "max_abs_logit_delta": result.max_abs_logit_delta,
                    }
                    for name, result in bundle.adapter_parity.items()
                },
            }
    return payload


def write_parity(bundle: ExportBundle, files: dict[str, str], out: Path) -> str | None:
    """Write ``parity.json`` beside the ONNX files; ``None`` when nothing was measured.

    Nothing measured means nothing written: an absent file says "not checked", which is the one
    thing a card must never confuse with "checked and perfect".
    """
    if not (bundle.parity or bundle.heads_parity):
        return None
    path = target_path(out).parent / PARITY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(parity_payload(bundle, files), indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return path.as_posix()


def _quantize(bundle: ExportBundle, fp16: bool, int8: bool) -> dict[str, str]:
    """Derive the browser's files from the fp32 graph and return every file to check."""
    if fp16:
        bundle.fp16 = to_fp16(Path(bundle.onnx.path))
    if int8:
        bundle.int8 = quantize_int8(Path(bundle.onnx.path))
    # The demo loads the fp16 or the int8 file, not the fp32 one, so the context length has to
    # travel with them too; neither converter promises to keep the metadata of its input.
    for quantized in (bundle.fp16, bundle.int8):
        if quantized is not None and bundle.onnx.metadata:
            set_metadata(Path(quantized.path), bundle.onnx.metadata)
    checks = {"fp32": bundle.onnx.path}
    if bundle.fp16 is not None:
        checks["fp16"] = bundle.fp16.path
    if bundle.int8 is not None:
        checks["int8"] = bundle.int8.path
    return checks


def export_all(
    ckpt: Path,
    out: Path,
    kind: str = "decoder",
    opset: int = DEFAULT_OPSET,
    seq_len: int = 200,
    fp16: bool = False,
    int8: bool = False,
    check_parity: bool = False,
    positions: int = 1_000,
    seed: int = 0,
    games: Path | None = None,
    labels: LabelsConfig | None = None,
    split: str = "val",
    adapter_inputs: bool = False,
    adapter: Path | None = None,
    lora: LoraConfig | None = None,
) -> ExportBundle:
    """Export, quantize and check one checkpoint in a single pass.

    For the decoder, parity is measured on validation positions (``games``, or the validation
    month of P1 when it is on disk) and falls back to random legal walks with a warning when
    there are none. For the encoder it is measured on the held-out labelled positions of
    ``rukh.data.labels`` and is skipped, with a warning, when those have not been built: a
    random board is not a position the demo will ever be asked to evaluate.

    ``adapter_inputs`` writes the graph that takes its LoRA factors as two extra inputs
    (``rukh.export.adapter``) instead of a graph with fixed weights. Its parity is then read
    twice: once with an adapter of zeros, which has to reproduce the checkpoint exactly, and, if
    ``adapter`` names a trained one, once with that adapter fed to both sides.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    if kind == "encoder":
        return _export_encoder(ckpt, out, opset, fp16, int8, check_parity, positions, labels, split)
    from rukh.tokenize.uci_vocab import UciTokenizer
    from rukh.train import load_model

    model, _payload = load_model(Path(ckpt))
    block = model.cfg.block
    if adapter_inputs:
        # `export_adaptable_onnx` swaps the adapted matrices for holders, so the model it is
        # given stops being a plain decoder; everything downstream reloads from the checkpoint.
        lora = lora or LoraConfig()
        bundle = ExportBundle(
            onnx=export_adaptable_onnx(model, out, lora, opset=opset, seq_len=seq_len),
            adapter_inputs=True,
        )
        base_feed = zero_inputs(_layout_of(Path(ckpt), lora))
    else:
        bundle = ExportBundle(onnx=export_onnx(model, out, opset=opset, seq_len=seq_len))
        base_feed = None
    checks = _quantize(bundle, fp16, int8)
    if check_parity:
        prefixes, source, warning = parity_positions(
            UciTokenizer(), n=positions, block=block, seed=seed, games=games
        )
        bundle.parity_source = source
        bundle.parity_warning = warning
        # The reference for the zero-adapter parity is the plain checkpoint: with no correction
        # the adaptable file has to be the model it was exported from, and nothing else.
        plain = load_model(Path(ckpt))[0] if adapter_inputs else model
        bundle.parity = {
            name: parity(plain, Path(path), prefixes, n=positions, extra=base_feed)
            for name, path in checks.items()
        }
        if adapter_inputs and adapter is not None:
            bundle.adapter_name = Path(adapter).name
            bundle.adapter_parity = _adapter_parity(ckpt, adapter, checks, prefixes, positions)
        bundle.parity_path = write_parity(bundle, checks, out)
    return bundle


def adapter_file(adapter: Path | str) -> Path:
    """``adapter.safetensors`` inside a training run, or the file itself when given directly.

    ``rukh export --adapter`` takes the *folder* a run wrote, because that is what a reader has in
    front of them, while ``load_adapter`` takes the weights file and finds the config next to it.
    Handing it the folder makes it look for ``adapter_config.json`` one level too high and fail on
    a path nobody wrote, so the translation happens once and here.
    """
    path = Path(adapter)
    return path / ADAPTER_FILE if path.is_dir() else path


def _layout_of(ckpt: Path, lora: LoraConfig) -> AdapterLayout:
    """The shape of the adapters a checkpoint's adaptable graph accepts."""
    from rukh.train import load_model

    model, _payload = load_model(Path(ckpt))
    apply_lora(model, lora)
    return adapter_layout(model)


def _adapter_parity(
    ckpt: Path,
    adapter: Path,
    checks: dict[str, str],
    prefixes: list[list[int]],
    positions: int,
) -> dict[str, ParityResult]:
    """Parity again, with a real adapter on both sides: the claim the swapping rests on."""
    from rukh.models.lora import load_adapter
    from rukh.train import load_model

    adapted, _payload = load_model(Path(ckpt))
    load_adapter(adapted, adapter_file(adapter))
    feed = adaptable_inputs(adapted)
    return {
        name: parity(adapted, Path(path), prefixes, n=positions, extra=feed)
        for name, path in checks.items()
    }


def _export_encoder(
    ckpt: Path,
    out: Path,
    opset: int,
    fp16: bool,
    int8: bool,
    check_parity: bool,
    positions: int,
    labels: LabelsConfig | None,
    split: str,
) -> ExportBundle:
    """``export_all(kind="encoder")``: the two heads, the same files, its own parity."""
    from rukh.train import load_heads

    model, _payload = load_heads(Path(ckpt))
    bundle = ExportBundle(onnx=export_encoder_onnx(model, out, opset=opset))
    checks = _quantize(bundle, fp16, int8)
    if check_parity:
        items, source, warning = encoder_parity_positions(model, labels, n=positions, split=split)
        bundle.parity_source = source
        bundle.parity_warning = warning
        if items:
            bundle.heads_parity = {
                name: encoder_parity(model, Path(path), items, n=positions)
                for name, path in checks.items()
            }
            bundle.parity_path = write_parity(bundle, checks, out)
    return bundle


__all__ = [
    "BLUNDER_OUTPUT",
    "DEFAULT_GAMES",
    "DEFAULT_OPSET",
    "ENCODER_OUTPUTS",
    "FP16_NAME",
    "INPUT_NAME",
    "INT8_NAME",
    "KINDS",
    "MODEL_NAME",
    "OUTPUT_NAME",
    "PARITY_MEASURES",
    "PARITY_NAME",
    "PARITY_VERSION",
    "QUANTIZED_OPS",
    "A_INPUT",
    "B_INPUT",
    "VALUE_OUTPUT",
    "EmbedResult",
    "EncoderHeads",
    "EncoderParityResult",
    "ExportBundle",
    "ExportResult",
    "LastStepLogits",
    "ParityResult",
    "QuantizeResult",
    "clear_value_info",
    "embed_positions",
    "encoder_parity",
    "encoder_parity_positions",
    "adaptable_inputs",
    "adapter_file",
    "export_adaptable_onnx",
    "export_all",
    "export_encoder_onnx",
    "export_onnx",
    "fen4",
    "label_positions",
    "parity",
    "parity_payload",
    "parity_positions",
    "quantize_int8",
    "random_prefixes",
    "read_metadata",
    "sequence_lengths",
    "set_metadata",
    "target_path",
    "to_fp16",
    "validation_prefixes",
    "verify_dynamic_seq",
    "write_metadata",
    "write_parity",
    "zero_inputs",
]
