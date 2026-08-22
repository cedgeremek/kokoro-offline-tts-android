#!/usr/bin/env python3
"""Build a tiny static graph with Kokoro's acoustic QNN I/O contract.

Every input contributes one finite scalar multiplied by zero, while the output
is a known constant acoustic tensor. The physical-device result therefore
separates EPContext/FastRPC tensor transport from Kokoro graph arithmetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", required=True, type=int)
    args = parser.parse_args()
    if args.bucket < 1:
        parser.error("--bucket must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    bucket = args.bucket
    shapes = {
        "/decoder/Slice_output_0": [1, 128],
        "/decoder/decoder/Unsqueeze_output_0": [1, 1, bucket],
        "/decoder/decoder/decode.3/Div_4_output_0": [1, 512, bucket],
        "valid_mask_10": [1, 1, bucket * 10],
        "valid_mask_60": [1, 1, bucket * 60 + 1],
    }
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes.items()]
    nodes = []
    scalars = []
    zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero")
    index = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="first_index")
    known = np.linspace(-0.25, 0.25, 22 * (bucket * 60 + 1), dtype=np.float32).reshape(
        1, 22, bucket * 60 + 1
    )
    initializers = [zero, index, numpy_helper.from_array(known, name="known_acoustic")]
    for number, name in enumerate(shapes):
        flat = f"flat_{number}"
        first = f"first_{number}"
        nodes.append(helper.make_node("Flatten", [name], [flat], name=f"flatten_{number}", axis=1))
        nodes.append(helper.make_node("Gather", [flat, "first_index"], [first], name=f"gather_{number}", axis=1))
        scalars.append(first)
    total = scalars[0]
    for number, scalar in enumerate(scalars[1:], start=1):
        added = f"sum_{number}"
        nodes.append(helper.make_node("Add", [total, scalar], [added], name=f"add_{number}"))
        total = added
    nodes.append(helper.make_node("Mul", [total, "zero"], ["input_zero"], name="make_zero"))
    nodes.append(helper.make_node("Add", ["known_acoustic", "input_zero"], ["acoustic"], name="known_output"))
    output = helper.make_tensor_value_info("acoustic", TensorProto.FLOAT, [1, 22, bucket * 60 + 1])
    graph = helper.make_graph(nodes, "kokoro_qnn_io_probe", inputs, [output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output} bucket={bucket} nodes={len(nodes)}")


if __name__ == "__main__":
    main()
