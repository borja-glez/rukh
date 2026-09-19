"""Half precision and dynamic int8 for the two browser backends.

WebGPU runs the fp16 graph (``small`` is about 80 MB that way); the WASM fallback runs the int8
one (about 40 MB), which is the difference between a demo that loads on a phone and one that
does not. Quantization is restricted to ``MatMul`` and ``Gemm``: those are the weights that make
up almost the whole file, and quantizing the rest costs accuracy for nothing.

``onnxconverter_common`` does the fp16 conversion properly (it keeps the graph's inputs and
outputs in float32 and leaves the ops that overflow in fp32). When it is not installed there is
a smaller fallback here that casts the float initializers with ``onnx.numpy_helper`` and puts a
``Cast`` back to float32 in front of the output; ``to_fp16`` reports which of the two ran.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

FP16_NAME = "model-fp16.onnx"
INT8_NAME = "model-int8.onnx"
QUANTIZED_OPS = ["MatMul", "Gemm"]


class QuantizeResult(BaseModel):
    """One converted file."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["fp16", "int8"]
    method: str
    bytes: int
    source_bytes: int

    @property
    def ratio(self) -> float:
        return self.bytes / self.source_bytes if self.source_bytes else 0.0


def _sibling(path: Path, name: str, out: Path | None) -> Path:
    if out is None:
        return Path(path).with_name(name)
    out = Path(out)
    return out if out.suffix == ".onnx" else out / name


def to_fp16(path: Path, out: Path | None = None) -> QuantizeResult:
    """Convert an fp32 ONNX model to fp16, in place of ``model.onnx``'s sibling by default."""
    import onnx

    source = Path(path)
    target = _sibling(source, FP16_NAME, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    try:
        from onnxconverter_common import float16

        converted = float16.convert_float_to_float16(model, keep_io_types=True)
        method = "onnxconverter_common.float16"
    except ImportError:
        converted = _cast_initializers(model)
        method = "onnx.numpy_helper fallback"
        log.warning("onnxconverter_common is not installed; used the initializer-casting fallback")
    repaired = _align_cast_outputs(converted)
    if repaired:
        log.info(
            "retyped %d Cast node(s) the fp16 conversion left disagreeing with the graph", repaired
        )
    onnx.save(converted, str(target))
    return QuantizeResult(
        path=target.as_posix(),
        kind="fp16",
        method=method,
        bytes=target.stat().st_size,
        source_bytes=source.stat().st_size,
    )


def _align_cast_outputs(model: object) -> int:
    """Make every ``Cast`` say the type its own output is declared to be; count the repairs.

    ``convert_float_to_float16`` retypes the values of the graph but does not touch the ``to``
    attribute of a ``Cast`` node, so a cast that used to produce float32 keeps saying so while
    its declared output is now float16. onnxruntime refuses to load that file — the symptom is a
    ``Type Error: Type (tensor(float16)) of output arg ... does not match expected type
    (tensor(float))`` — and the encoder hits it because pooling casts its padding mask to the
    hidden dtype. Only intermediate values are considered: the graph's own inputs and outputs
    keep the types ``keep_io_types`` promised the caller.
    """
    from onnx import TensorProto

    graph = model.graph  # type: ignore[attr-defined]
    declared = {value.name: value.type.tensor_type.elem_type for value in graph.value_info}
    repaired = 0
    for node in graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        wanted = declared.get(node.output[0])
        if wanted not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i != wanted:
                attribute.i = int(wanted)
                repaired += 1
    return repaired


def _cast_initializers(model: object) -> object:
    """Fallback fp16 conversion: cast every float initializer and cast the outputs back.

    Only the weights change type. Every float-typed value in the graph is retyped to fp16 and a
    ``Cast`` node is appended so the model still hands the caller float32 logits, which keeps the
    demo's input and output contract identical to the fp32 file.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph  # type: ignore[attr-defined]
    for initializer in graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(initializer).astype(np.float16)
        initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
    for value in list(graph.value_info) + list(graph.input):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT:
            value.type.tensor_type.elem_type = TensorProto.FLOAT16
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensor = attribute.t
                if tensor.data_type == TensorProto.FLOAT:
                    array = numpy_helper.to_array(tensor).astype(np.float16)
                    tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
    for output in graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        inner = f"{output.name}_fp16"
        for node in graph.node:
            node.output[:] = [inner if name == output.name else name for name in node.output]
        graph.node.append(
            helper.make_node("Cast", [inner], [output.name], to=int(TensorProto.FLOAT))
        )
    return model


def quantize_int8(path: Path, out: Path | None = None) -> QuantizeResult:
    """Dynamically quantize the MatMul and Gemm weights of an ONNX model to int8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(path)
    target = _sibling(source, INT8_NAME, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=QUANTIZED_OPS,
    )
    return QuantizeResult(
        path=target.as_posix(),
        kind="int8",
        method="onnxruntime.quantization.quantize_dynamic",
        bytes=target.stat().st_size,
        source_bytes=source.stat().st_size,
    )
