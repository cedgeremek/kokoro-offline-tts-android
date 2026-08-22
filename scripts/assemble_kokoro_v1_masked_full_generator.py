#!/usr/bin/env python3
"""Join a fixed masked v1 acoustic graph with its iSTFT suffix.

This restores the v0.19-style *full waveform* QNN route for Kokoro v1: the
HTP graph receives the three front-end tensors and emits waveform directly.
The acoustic source must already be fixed-shape and mask-correct; this tool
does not alter its numerical operations or introduce padding semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


ISTFT_INPUT = "/decoder/decoder/generator/conv_post/Conv_output_0"


def _tensor_bytes(tensor: onnx.TensorProto) -> bytes:
    copy = onnx.TensorProto()
    copy.CopyFrom(tensor)
    return copy.SerializeToString(deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, required=True)
    parser.add_argument("--istft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    acoustic = onnx.load(args.acoustic)
    suffix = onnx.load(args.istft)
    graph = acoustic.graph
    if len(graph.output) != 1 or graph.output[0].name != "acoustic":
        raise RuntimeError("Acoustic graph must expose exactly the `acoustic` output")
    if len(suffix.graph.input) != 1 or suffix.graph.input[0].name != ISTFT_INPUT:
        raise RuntimeError("Unexpected iSTFT input contract")
    if len(suffix.graph.output) != 1 or suffix.graph.output[0].name != "waveform":
        raise RuntimeError("Unexpected iSTFT output contract")

    # Link the producer directly to the original iSTFT input name. The suffix
    # had that value as a graph input, so no bridge node or copy is introduced.
    for node in graph.node:
        for index, output in enumerate(node.output):
            if output == "acoustic":
                node.output[index] = ISTFT_INPUT

    existing = {tensor.name: tensor for tensor in graph.initializer}
    for tensor in suffix.graph.initializer:
        prior = existing.get(tensor.name)
        if prior is None:
            copied = graph.initializer.add()
            copied.CopyFrom(tensor)
            existing[tensor.name] = copied
        elif _tensor_bytes(prior) != _tensor_bytes(tensor):
            raise RuntimeError(f"Conflicting initializer named {tensor.name}")
    graph.node.extend(suffix.graph.node)
    del graph.output[:]
    graph.output.add().CopyFrom(suffix.graph.output[0])
    acoustic.ir_version = min(acoustic.ir_version, suffix.ir_version, 10)
    onnx.checker.check_model(acoustic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(acoustic, args.output)
    print(f"wrote {args.output}: nodes={len(graph.node)} initializers={len(graph.initializer)}", flush=True)


if __name__ == "__main__":
    main()
