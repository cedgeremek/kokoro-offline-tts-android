#!/usr/bin/env python3
"""Expose a static Kokoro v1 internal tensor for physical HTP bisection."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--node-index", required=True, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    model = onnx.load(args.source)
    graph = model.graph
    if not 0 <= args.node_index < len(graph.node):
        parser.error(f"--node-index must be within 0..{len(graph.node) - 1}")
    selected = graph.node[args.node_index]
    if len(selected.output) != 1:
        parser.error("selected probe node must have exactly one output")
    target = selected.output[0]

    # Ask ORT for the concrete output shape before pruning. The input graph is
    # already fixed to B64, so the result must contain integers only.
    shape_model = onnx.ModelProto()
    shape_model.CopyFrom(model)
    del shape_model.graph.output[:]
    shape_model.graph.output.append(helper.make_tensor_value_info(target, TensorProto.FLOAT, None))
    session = ort.InferenceSession(
        shape_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    metadata = session.get_outputs()[0]
    if metadata.type != "tensor(float)" or not all(isinstance(value, int) for value in metadata.shape):
        raise RuntimeError(f"probe output is not static FP32: {metadata.type} {metadata.shape}")
    output_shape = list(metadata.shape)

    required = {target}
    retained = []
    for node in reversed(graph.node):
        if any(name in required for name in node.output):
            retained.append(node)
            required.update(name for name in node.input if name)
    retained.reverse()
    del graph.node[:]
    graph.node.extend(retained)

    original_inputs = list(graph.input)
    del graph.input[:]
    graph.input.extend(value for value in original_inputs if value.name in required)

    # The target is terminal after backward pruning, so renaming it cannot
    # affect another node and gives every diagnostic context one stable name.
    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == target:
                node.output[index] = "probe"
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("probe", TensorProto.FLOAT, output_shape))
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"wrote {args.output} index={args.node_index} op={selected.op_type} "
        f"name={selected.name} nodes={len(graph.node)} inputs={[v.name for v in graph.input]} "
        f"shape={output_shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
