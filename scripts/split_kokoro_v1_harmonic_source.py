#!/usr/bin/env python3
"""Split Kokoro v1's small harmonic source from its QNN-heavy vocoder.

The HTP v75 compiler used by this project miscompiles the voiced-gate branch
inside ``m_source``.  This tool preserves the original math byte-for-byte by
running that small branch in the CPU session and exposing its terminal tensor
as one additional input to the QNN suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


SOURCE_OUTPUT = "/decoder/decoder/generator/Transpose_1_output_0"
QNN_SOURCE_INPUT = "kokoro_harmonic_source"


def value_info(model: onnx.ModelProto, name: str, replacement: str) -> onnx.ValueInfoProto:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == name:
            copied = onnx.ValueInfoProto()
            copied.CopyFrom(value)
            copied.name = replacement
            dims = copied.type.tensor_type.shape.dim
            if dims and all(dim.HasField("dim_value") for dim in dims):
                return copied
            shape_model = onnx.ModelProto()
            shape_model.CopyFrom(model)
            del shape_model.graph.output[:]
            shape_model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
            session = ort.InferenceSession(
                shape_model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            metadata = session.get_outputs()[0]
            if metadata.type != "tensor(float)" or not all(
                isinstance(dimension, int) for dimension in metadata.shape
            ):
                raise RuntimeError(f"Cut tensor is not static FP32: {metadata.type} {metadata.shape}")
            return helper.make_tensor_value_info(replacement, TensorProto.FLOAT, metadata.shape)
    raise RuntimeError(f"No type/shape metadata for {name!r}")


def prune_to_outputs(model: onnx.ModelProto) -> None:
    graph = model.graph
    required = {value.name for value in graph.output}
    retained: list[onnx.NodeProto] = []
    for node in reversed(graph.node):
        if any(output in required for output in node.output):
            retained.append(node)
            required.update(name for name in node.input if name)
    retained.reverse()
    del graph.node[:]
    graph.node.extend(retained)

    inputs = [value for value in graph.input if value.name in required]
    del graph.input[:]
    graph.input.extend(inputs)
    initializers = [tensor for tensor in graph.initializer if tensor.name in required]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    sparse = [tensor for tensor in graph.sparse_initializer if tensor.values.name in required]
    del graph.sparse_initializer[:]
    graph.sparse_initializer.extend(sparse)
    retained_names = required | {name for node in retained for name in (*node.input, *node.output)}
    infos = [value for value in graph.value_info if value.name in retained_names]
    del graph.value_info[:]
    graph.value_info.extend(infos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cpu-source-output", required=True, type=Path)
    parser.add_argument("--qnn-suffix-output", required=True, type=Path)
    parser.add_argument("--cut-output", default=SOURCE_OUTPUT)
    parser.add_argument("--qnn-input-name", default=QNN_SOURCE_INPUT)
    args = parser.parse_args()
    for output in (args.cpu_source_output, args.qnn_suffix_output):
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")

    original = onnx.load(args.source)
    source_input_info = value_info(original, args.cut_output, args.qnn_input_name)

    cpu_source = onnx.ModelProto()
    cpu_source.CopyFrom(original)
    del cpu_source.graph.output[:]
    terminal = onnx.ValueInfoProto()
    terminal.CopyFrom(source_input_info)
    terminal.name = args.cut_output
    cpu_source.graph.output.append(terminal)
    prune_to_outputs(cpu_source)
    cpu_source.ir_version = min(cpu_source.ir_version, 10)
    onnx.checker.check_model(cpu_source)

    qnn_suffix = onnx.ModelProto()
    qnn_suffix.CopyFrom(original)
    replacements = 0
    for node in qnn_suffix.graph.node:
        for index, name in enumerate(node.input):
            if name == args.cut_output:
                node.input[index] = args.qnn_input_name
                replacements += 1
    if replacements < 1:
        raise RuntimeError(f"Cut tensor has no consumers: {args.cut_output}")
    qnn_suffix.graph.input.append(source_input_info)
    prune_to_outputs(qnn_suffix)
    qnn_suffix.ir_version = min(qnn_suffix.ir_version, 10)
    onnx.checker.check_model(qnn_suffix)

    args.cpu_source_output.parent.mkdir(parents=True, exist_ok=True)
    args.qnn_suffix_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(cpu_source, args.cpu_source_output)
    onnx.save(qnn_suffix, args.qnn_suffix_output)
    print(
        f"wrote CPU source {args.cpu_source_output} "
        f"nodes={len(cpu_source.graph.node)} inputs={[v.name for v in cpu_source.graph.input]} "
        f"output={args.cut_output}",
        flush=True,
    )
    print(
        f"wrote QNN suffix {args.qnn_suffix_output} "
        f"nodes={len(qnn_suffix.graph.node)} inputs={[v.name for v in qnn_suffix.graph.input]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
