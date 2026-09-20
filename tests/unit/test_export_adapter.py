"""The adaptable graph: zeros are the base model, and the factors really are inputs."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from rukh.export import export_onnx
from rukh.export.adapter import (
    A_INPUT,
    B_INPUT,
    WEB_ADAPTER_META,
    adaptable_inputs,
    browser_lora,
    build_adaptable,
    export_adaptable_onnx,
    restore_lora_modules,
    write_web_adapter,
    zero_inputs,
)
from rukh.export.onnx import INPUT_NAME, read_metadata
from rukh.models import DecoderConfig, MoveDecoder
from rukh.models.lora import LoraConfig, adapter_layout, apply_lora, lora_modules

ort = pytest.importorskip("onnxruntime")

TOY = DecoderConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, block=16, dropout=0.0)
LORA = LoraConfig(r=4, alpha=8, targets=("q", "v"))
SEQ = 8  # the toy block is 16; the exporter refuses an example longer than the model


def _toy(seed: int = 0) -> MoveDecoder:
    torch.manual_seed(seed)
    return MoveDecoder(TOY).eval()


def _trained_adapter(model: MoveDecoder) -> None:
    """Give every ``B`` something other than zero, which is what training would do."""
    torch.manual_seed(7)
    with torch.no_grad():
        for _, adapter in lora_modules(model):
            for b in adapter.b:
                b.normal_(std=0.05)


def _session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_the_graph_declares_the_two_factors_as_inputs(tmp_path):
    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=SEQ)
    session = _session(result.path)
    assert [i.name for i in session.get_inputs()] == [INPUT_NAME, A_INPUT, B_INPUT]
    metadata = read_metadata(result.path)
    assert metadata["rukh_adapter_inputs"] == f"{A_INPUT},{B_INPUT}"
    assert metadata["rukh_adapter_shape_a"] == "2,2,4,32"
    assert metadata["rukh_adapter_shape_b"] == "2,2,32,4"
    assert metadata["rukh_adapter_module"] == "attn.qkv"


def test_zeros_are_the_base_model(tmp_path):
    """The claim that lets the adaptable file replace the ordinary one instead of joining it."""
    plain = export_onnx(_toy(), tmp_path / "plain" / "model.onnx", seq_len=SEQ)
    adaptable = export_adaptable_onnx(_toy(), tmp_path / "lora" / "model.onnx", LORA, seq_len=SEQ)
    idx = torch.randint(1, 64, (1, 9)).numpy()
    layout = adapter_layout(_with_lora(_toy()))

    base = _session(plain.path).run(None, {INPUT_NAME: idx})[0]
    adapted = _session(adaptable.path).run(None, {INPUT_NAME: idx, **zero_inputs(layout)})[0]
    assert np.abs(base - adapted).max() < 1e-5


def test_the_adapter_changes_the_logits_without_touching_the_file(tmp_path):
    model = _with_lora(_toy())
    _trained_adapter(model)
    feed = adaptable_inputs(model)
    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=SEQ)
    layout = adapter_layout(_with_lora(_toy()))
    idx = torch.randint(1, 64, (1, 9)).numpy()

    session = _session(result.path)
    without = session.run(None, {INPUT_NAME: idx, **zero_inputs(layout)})[0]
    with_adapter = session.run(None, {INPUT_NAME: idx, **feed})[0]
    assert np.abs(with_adapter - without).max() > 1e-3


def test_the_graph_agrees_with_pytorch_on_the_same_adapter(tmp_path):
    """Parity: the file and ``apply_lora`` have to compute the same next move."""
    model = _with_lora(_toy())
    _trained_adapter(model)
    idx = torch.randint(1, 64, (1, 9))
    with torch.no_grad():
        expected = model(idx)[0][:, -1, :].numpy()

    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=SEQ)
    feed = adaptable_inputs(model)
    got = _session(result.path).run(None, {INPUT_NAME: idx.numpy(), **feed})[0]
    assert np.abs(got - expected).max() < 1e-4


def test_two_adapters_give_two_different_answers_from_one_file(tmp_path):
    """Swapping is the whole feature: one download, as many styles as there are files."""
    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=SEQ)
    session = _session(result.path)
    idx = torch.randint(1, 64, (1, 9)).numpy()
    answers = []
    for seed in (1, 2):
        model = _with_lora(_toy())
        torch.manual_seed(seed)
        with torch.no_grad():
            for _, adapter in lora_modules(model):
                for b in adapter.b:
                    b.normal_(std=0.1)
        answers.append(session.run(None, {INPUT_NAME: idx, **adaptable_inputs(model)})[0])
    assert np.abs(answers[0] - answers[1]).max() > 1e-3


def test_the_sequence_axis_survives_the_extra_inputs(tmp_path):
    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=8)
    assert result.dynamic_seq is True
    assert result.dynamic_seq_verified is True


def test_an_adapter_that_cannot_be_stacked_says_so():
    """The MLP's ``fc`` is four times as wide, so the factors stop being one tensor."""
    model = _toy()
    apply_lora(model, LoraConfig(r=4, alpha=8, targets=("q", "mlp_in")))
    with pytest.raises(ValueError, match="more than one matrix per block"):
        adapter_layout(model)


def test_the_holders_can_be_turned_back_into_ordinary_adapters():
    model = _toy()
    build_adaptable(model, LORA)
    assert not list(lora_modules(model))
    assert restore_lora_modules(model, LORA) == TOY.n_layer
    assert len(list(lora_modules(model))) == TOY.n_layer


def _with_lora(model: MoveDecoder) -> MoveDecoder:
    apply_lora(model, LORA)
    return model


def test_the_browser_file_feeds_the_unit_scale_graph_and_matches_pytorch(tmp_path):
    """The whole browser path end to end: write the flat file, read it back, run the graph.

    The scale lives in the file and not in the graph (``browser_lora``), so this is also the
    check that the convention holds: a ``B`` written scaled, fed to a graph that multiplies by
    one, has to reproduce the model that applies ``alpha / r`` itself.
    """
    import json

    model = _with_lora(_toy())
    _trained_adapter(model)
    idx = torch.randint(1, 64, (1, 9))
    with torch.no_grad():
        expected = model(idx)[0][:, -1, :].numpy()

    binary = write_web_adapter(model, tmp_path / "web" / "adapter.bin", base_repo="chorcat/toy")
    meta = json.loads((tmp_path / "web" / pathlib.Path(WEB_ADAPTER_META).name).read_text("utf-8"))
    assert meta["shape_a"] == [2, 2, 4, 32]
    assert meta["scale_applied"] == LORA.scale

    flat = np.frombuffer(binary.read_bytes(), dtype="<f4")
    size_a = int(np.prod(meta["shape_a"]))
    feed = {
        A_INPUT: flat[:size_a].reshape(meta["shape_a"]).copy(),
        B_INPUT: flat[size_a:].reshape(meta["shape_b"]).copy(),
    }

    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", browser_lora(LORA), seq_len=SEQ)
    got = _session(result.path).run(None, {INPUT_NAME: idx.numpy(), **feed})[0]
    assert np.abs(got - expected).max() < 1e-4


def test_the_web_file_is_the_same_from_the_model_and_from_the_saved_adapter(tmp_path):
    """Publishing must not need 440 MB of base weights on disk to write 1.6 MB of floats."""
    from rukh.export.adapter import write_web_adapter_from_file
    from rukh.models.lora import save_adapter

    model = _with_lora(_toy())
    _trained_adapter(model)
    save_adapter(model, tmp_path / "adapter.safetensors", LORA)

    from_model = write_web_adapter(model, tmp_path / "a" / "adapter.bin", "chorcat/toy")
    from_file = write_web_adapter_from_file(
        tmp_path / "adapter.safetensors", tmp_path / "b" / "adapter.bin", "chorcat/toy"
    )
    assert from_model.read_bytes() == from_file.read_bytes()


def test_the_quantized_files_still_take_the_adapter(tmp_path):
    """int8 and fp16 are what the browser actually loads, so the swap has to survive both.

    Dynamic quantization rewrites the `MatMul`s whose other side is an initialiser, and the
    adapter's two are not: both of their inputs come from outside the graph. Nothing guarantees
    the converters leave them alone, so it is checked rather than assumed.
    """
    from rukh.export.quantize import quantize_int8, to_fp16

    model = _with_lora(_toy())
    _trained_adapter(model)
    feed = adaptable_inputs(model)
    result = export_adaptable_onnx(_toy(), tmp_path / "model.onnx", LORA, seq_len=SEQ)
    layout = adapter_layout(_with_lora(_toy()))
    idx = torch.randint(1, 64, (1, 9)).numpy()

    for quantized in (to_fp16(pathlib.Path(result.path)), quantize_int8(pathlib.Path(result.path))):
        session = _session(quantized.path)
        assert [i.name for i in session.get_inputs()] == [INPUT_NAME, A_INPUT, B_INPUT]
        without = session.run(None, {INPUT_NAME: idx, **zero_inputs(layout)})[0]
        with_adapter = session.run(None, {INPUT_NAME: idx, **feed})[0]
        assert np.abs(with_adapter - without).max() > 1e-3, quantized.kind


def test_the_adapter_parity_takes_the_folder_a_run_wrote(tmp_path):
    """`--adapter` names a training run's folder; `load_adapter` wants the weights file.

    The translation used to be missing, so the export looked for `adapter_config.json` one level
    above the run and died on a path nobody had written -- after fifteen minutes of exporting.
    """
    from rukh.export import adapter_file
    from rukh.models.lora import ADAPTER_FILE, save_adapter

    model = _with_lora(_toy())
    run = tmp_path / "lora-e4-20260920-160133"
    save_adapter(model, run / ADAPTER_FILE, LORA)

    assert adapter_file(run) == run / ADAPTER_FILE
    assert adapter_file(run / ADAPTER_FILE) == run / ADAPTER_FILE
    # And the file it points at is the one `load_adapter` can actually read.
    plain = _toy()
    from rukh.models.lora import load_adapter

    assert load_adapter(plain, adapter_file(run)).r == LORA.r
