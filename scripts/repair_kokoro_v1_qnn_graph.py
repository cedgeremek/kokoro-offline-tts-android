#!/usr/bin/env python3
"""Apply numerically stable QNN workarounds to a fixed Kokoro v1 graph.

QAIRT 2.48.40 on HTP v75 miscompiles the harmonic-source voiced gate when a
``[1, samples, 9]`` tensor is multiplied by a broadcast ``[1, samples, 1]``
boolean-to-float mask.  Both inputs are correct on device, but the broadcast
Mul produces all zeros.  Repeating the mask explicitly makes the Mul shapes
identical without changing CPU results.

QAIRT 2.48.40 also miscompiles ``Pow(x, 2)`` in the Snake activations on HTP
v75: a sine input bounded to [-1, 1] has produced a squared output as large as
4 on the physical SM8650 gate.  Replacing that special case with ``Mul(x, x)``
is algebraically equivalent and avoids the faulty power kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


GATE_MUL_NAME = "/decoder/decoder/generator/m_source/l_sin_gen/Mul_6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gate-strategy",
        choices=("none", "concat-mul", "where"),
        default="none",
        help="Optional voiced-gate implementation (default: none).",
    )
    parser.add_argument(
        "--square-pow-to-mul",
        action="store_true",
        help="Replace every Pow whose constant exponent is exactly 2 with Mul(x, x).",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    model = onnx.load(args.source)
    graph = model.graph
    transformations: list[str] = []

    if args.gate_strategy != "none":
        matches = [index for index, node in enumerate(graph.node) if node.name == GATE_MUL_NAME]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {GATE_MUL_NAME!r}, found {len(matches)}")

        index = matches[0]
        original = graph.node[index]
        if original.op_type != "Mul" or len(original.input) != 2 or len(original.output) != 1:
            raise RuntimeError(f"Unexpected voiced-gate node contract: {original}")

        harmonic, gate = original.input
        producers = {output: node for node in graph.node for output in node.output}
        cast = producers.get(gate)
        if cast is None or cast.op_type != "Cast" or len(cast.input) != 1:
            raise RuntimeError("Voiced gate is not produced by the expected bool-to-float Cast")
        condition = cast.input[0]
        expanded_gate = f"{gate}.qnn_explicit_9"
        expanded_condition = f"{condition}.qnn_explicit_9"

        # Concat repeats the scalar harmonic gate along its final channel axis.
        # Unlike Expand/Tile, this leaves no broadcasted mask for QNN to infer.
        concat = helper.make_node(
            "Concat",
            inputs=([condition] * 9 if args.gate_strategy == "where" else [gate] * 9),
            outputs=[expanded_condition if args.gate_strategy == "where" else expanded_gate],
            name=f"{GATE_MUL_NAME}/QnnExplicitGateConcat",
            axis=2,
        )
        if args.gate_strategy == "where":
            shape = None
            for value in graph.value_info:
                if value.name == harmonic:
                    dims = value.type.tensor_type.shape.dim
                    if dims and all(dim.HasField("dim_value") for dim in dims):
                        shape = [dim.dim_value for dim in dims]
                    break
            if shape is None:
                raise RuntimeError(f"No static inferred shape for harmonic tensor {harmonic!r}")
            zeros = f"{original.output[0]}.qnn_zeros"
            graph.initializer.append(
                numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name=zeros)
            )
            replacement = helper.make_node(
                "Where",
                inputs=[expanded_condition, harmonic, zeros],
                outputs=list(original.output),
                name=f"{GATE_MUL_NAME}/QnnSelect",
            )
        else:
            replacement = helper.make_node(
                "Mul",
                inputs=[harmonic, expanded_gate],
                outputs=list(original.output),
                name=f"{GATE_MUL_NAME}/QnnExactShape",
            )
        nodes = list(graph.node)
        nodes[index : index + 1] = [concat, replacement]
        del graph.node[:]
        graph.node.extend(nodes)
        if args.gate_strategy == "where":
            graph.value_info.append(
                helper.make_tensor_value_info(expanded_condition, TensorProto.BOOL, shape)
            )
        else:
            graph.value_info.append(
                helper.make_tensor_value_info(expanded_gate, TensorProto.FLOAT, [1, None, 9])
            )
        transformations.append(f"{args.gate_strategy} voiced gate")

    if args.square_pow_to_mul:
        initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
        replaced = 0
        for node in graph.node:
            if node.op_type != "Pow" or len(node.input) != 2:
                continue
            exponent = initializers.get(node.input[1])
            if exponent is None or exponent.size != 1 or float(exponent.reshape(-1)[0]) != 2.0:
                continue
            base = node.input[0]
            node.op_type = "Mul"
            del node.input[:]
            node.input.extend([base, base])
            node.name = f"{node.name}/QnnSquareMul"
            replaced += 1
        if replaced == 0:
            raise RuntimeError("--square-pow-to-mul found no constant exponent-2 Pow nodes")
        transformations.append(f"{replaced} Pow(x,2)->Mul(x,x)")

    if not transformations:
        parser.error("No repair selected")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"wrote {args.output} with {', '.join(transformations)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
