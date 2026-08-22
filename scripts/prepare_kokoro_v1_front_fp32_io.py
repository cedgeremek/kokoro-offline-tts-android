#!/usr/bin/env python3
"""Expose the v1 FP16 front graph through the existing FP32 ORT contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build(source: Path, output: Path) -> None:
    model = onnx.load(str(source))
    g = model.graph
    # The v1 front exporter already has the correct input_ids/style/speed
    # contract.  Only cast its three FP16 cut outputs for the Java runtime.
    output_names = [x.name for x in g.output]
    cast_nodes = []
    new_outputs = []
    for old in list(g.output):
        if old.type.tensor_type.elem_type != TensorProto.FLOAT16:
            raise ValueError(f"expected FP16 front output: {old.name}")
        internal = old.name + "__fp16"
        for node in g.node:
            for i, name in enumerate(node.output):
                if name == old.name:
                    node.output[i] = internal
            for i, name in enumerate(node.input):
                if name == old.name:
                    node.input[i] = internal
        cast_nodes.append(helper.make_node("Cast", [internal], [old.name], name=old.name + "__to_fp32", to=TensorProto.FLOAT))
        shape = [d.dim_value or d.dim_param for d in old.type.tensor_type.shape.dim]
        new_outputs.append(helper.make_tensor_value_info(old.name, TensorProto.FLOAT, shape))
    g.node.extend(cast_nodes)
    del g.output[:]
    g.output.extend(new_outputs)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    build(a.source, a.output)
    print(f"wrote {a.output}")
