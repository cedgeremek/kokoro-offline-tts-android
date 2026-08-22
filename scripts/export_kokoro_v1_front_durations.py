"""Expose Kokoro v1's exact token-duration alignment from the CPU front graph.

The shipped front already computes this INT64 tensor and consumes it to build
the duration-expanded frame path.  This script adds that existing value as a
fourth graph output; it does not add a model input or invent a `weights`
contract.  The three established floating-point outputs stay unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx


ESTABLISHED_OUTPUTS = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)
DURATION_OUTPUT = "/encoder/Cast_output_0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.source)
    graph = model.graph
    actual = tuple(value.name for value in graph.output[:3])
    if actual != ESTABLISHED_OUTPUTS:
        raise ValueError(f"Unexpected Kokoro front outputs: {actual}")

    if not any(value.name == DURATION_OUTPUT for value in graph.output):
        duration_info = next(
            (value for value in graph.value_info if value.name == DURATION_OUTPUT),
            None,
        )
        if duration_info is None:
            raise ValueError(f"Missing internal duration tensor {DURATION_OUTPUT}")
        graph.output.add().CopyFrom(duration_info)

    output_names = tuple(value.name for value in graph.output)
    if output_names != (*ESTABLISHED_OUTPUTS, DURATION_OUTPUT):
        raise ValueError(f"Unexpected exported output contract: {output_names}")
    duration_type = graph.output[-1].type.tensor_type
    if duration_type.elem_type != onnx.TensorProto.INT64:
        raise ValueError("Kokoro token durations are no longer INT64")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.part")
    try:
        onnx.save(model, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
