"""Create a host-runnable FP16 sensitivity model with FLOAT32 external I/O.

The shipping QNN route uses ``enable_htp_fp16_precision=1`` while preserving
the five-input FLOAT contract.  This conversion is intentionally only a host
quality probe: it rounds eligible graph math and weights to IEEE FP16 so gross
numeric sensitivity is caught before QAIRT compilation.  It is not claimed to
reproduce Qualcomm HTP kernels bit-for-bit and must never be packaged as the
QNN context source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topologically_sort(model: onnx.ModelProto) -> None:
    """Repair converter-appended Cast placement without changing graph math."""
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)
    available.update(value.values.name for value in graph.sparse_initializer)
    pending = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        next_pending: list[onnx.NodeProto] = []
        progressed = False
        for node in pending:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(name for name in node.output if name)
                progressed = True
            else:
                next_pending.append(node)
        if not progressed:
            missing = sorted(
                {
                    name
                    for node in next_pending
                    for name in node.input
                    if name and name not in available
                }
            )
            raise RuntimeError(f"Unable to topologically sort converted graph; missing={missing[:10]}")
        pending = next_pending
    del graph.node[:]
    graph.node.extend(ordered)


def deduplicate_identical_producers(model: onnx.ModelProto) -> int:
    """Collapse an ORT float16-converter bug that repeats identical Casts."""
    kept: list[onnx.NodeProto] = []
    producer: dict[str, onnx.NodeProto] = {}
    removed = 0
    for node in model.graph.node:
        duplicate = None
        for output in node.output:
            if output and output in producer:
                duplicate = producer[output]
                break
        if duplicate is not None:
            equivalent = (
                node.op_type == duplicate.op_type
                and node.domain == duplicate.domain
                and list(node.input) == list(duplicate.input)
                and list(node.output) == list(duplicate.output)
                and [item.SerializeToString() for item in node.attribute]
                == [item.SerializeToString() for item in duplicate.attribute]
            )
            if not equivalent:
                raise RuntimeError(
                    f"Converted graph has conflicting producers for {list(node.output)}"
                )
            removed += 1
            continue
        kept.append(node)
        for output in node.output:
            if output:
                producer[output] = node
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--block-op",
        action="append",
        default=[],
        help="Additional operator type to retain in FP32 (repeatable).",
    )
    parser.add_argument(
        "--only-fp16-op",
        action="append",
        default=[],
        help="If present, retain every other operator type in FP32 (repeatable).",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing sensitivity model: {args.output}")

    # Passing a path makes ORT's helper reopen a still-locked NamedTemporaryFile
    # on Windows.  Loading the ModelProto first uses the equivalent in-memory
    # shape-inference route and is deterministic for this sub-100 MB graph.
    source_model = onnx.load(str(args.model.resolve()))
    selected_block_ops = list(args.block_op)
    if args.only_fp16_op:
        allowed = set(args.only_fp16_op)
        selected_block_ops.extend(
            sorted({node.op_type for node in source_model.graph.node} - allowed)
        )
    selected_block_ops = sorted(set(selected_block_ops))
    converted = convert_float_to_float16(
        source_model,
        min_positive_val=5.96e-8,
        max_finite_val=65504.0,
        keep_io_types=True,
        op_block_list=sorted(set(DEFAULT_OP_BLOCK_LIST) | set(selected_block_ops)),
    )
    converted.metadata_props.add(
        key="kokoro_host_probe",
        value="FP16 sensitivity only; production uses QNN native HTP FP16",
    )
    duplicate_casts_removed = deduplicate_identical_producers(converted)
    topologically_sort(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(converted, str(args.output))
    onnx.checker.check_model(str(args.output), full_check=True)
    receipt = {
        "purpose": "host-only FP16 numeric sensitivity probe",
        "source": str(args.model.resolve()),
        "source_sha256": sha256(args.model),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "keep_io_types": True,
        "min_positive_val": 5.96e-8,
        "max_finite_val": 65504.0,
        "additional_fp32_op_types": selected_block_ops,
        "only_fp16_op_types": args.only_fp16_op,
        "duplicate_converter_nodes_removed": duplicate_casts_removed,
        "production_asset": False,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
