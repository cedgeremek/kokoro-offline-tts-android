#!/usr/bin/env python3
"""Create an exact-length static Kokoro v1 acoustic graph for QNN diagnosis.

Unlike the production bucket graphs this graph performs no padding or temporal
mask transformation.  It is useful only when the test utterance's measured
duration exactly equals the selected bucket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


INPUTS = {
    "/decoder/Slice_output_0": lambda bucket: [1, 128],
    "/decoder/decoder/Unsqueeze_output_0": lambda bucket: [1, 1, bucket],
    "/decoder/decoder/decode.3/Div_4_output_0": lambda bucket: [1, 512, bucket],
}
OUTPUT_SOURCE = "/decoder/decoder/generator/conv_post/Conv_output_0"


def set_shape(value, dimensions: list[int]) -> None:
    value.type.tensor_type.elem_type = TensorProto.FLOAT
    del value.type.tensor_type.shape.dim[:]
    for size in dimensions:
        value.type.tensor_type.shape.dim.add().dim_value = size


def find_value(graph, name: str):
    for values in (graph.input, graph.output, graph.value_info):
        for value in values:
            if value.name == name:
                return value
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", required=True, type=int)
    args = parser.parse_args()
    if args.bucket < 16:
        parser.error("--bucket must be at least 16 frames")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    model = onnx.load(args.source)
    graph = model.graph
    for name, shape in INPUTS.items():
        set_shape(find_value(graph, name), shape(args.bucket))
    set_shape(find_value(graph, OUTPUT_SOURCE), [1, 22, 60 * args.bucket + 1])

    model = shape_inference.infer_shapes(model)
    graph = model.graph
    for name, shape in INPUTS.items():
        set_shape(find_value(graph, name), shape(args.bucket))

    required = {OUTPUT_SOURCE}
    retained = []
    for node in reversed(graph.node):
        if any(output in required for output in node.output):
            retained.append(node)
            required.update(value for value in node.input if value)
    retained.reverse()
    del graph.node[:]
    graph.node.extend(retained)

    del graph.input[:]
    for name, shape in INPUTS.items():
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape(args.bucket)))
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("acoustic", TensorProto.FLOAT, [1, 22, 60 * args.bucket + 1])
    )
    for node in graph.node:
        for index, output in enumerate(node.output):
            if output == OUTPUT_SOURCE:
                node.output[index] = "acoustic"

    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output} bucket={args.bucket} nodes={len(graph.node)}", flush=True)


if __name__ == "__main__":
    main()
