"""Compile a fixed-shape QNN HTP context from the masked Kokoro generator.

This uses Qualcomm's official ONNX Runtime QNN plugin wheel as a cross-platform
context generator.  The generated EPContext model is targeted explicitly at
Snapdragon 8 Gen 3 (SM8650, HTP v75) and has CPU fallback disabled so a receipt
cannot be produced from a partially delegated graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import onnxruntime as ort
import onnxruntime_qnn as ort_qnn
from onnx import TensorProto


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_contract(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return {
        "name": value.name,
        "element_type": TensorProto.DataType.Name(tensor.elem_type),
        "shape": dimensions,
    }


def attribute_value(node: onnx.NodeProto, name: str) -> object | None:
    attribute = next((item for item in node.attribute if item.name == name), None)
    return None if attribute is None else onnx.helper.get_attribute_value(attribute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", required=True, type=int)
    parser.add_argument("--soc-model", default="57")
    parser.add_argument("--htp-arch", default="75")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--v1-full-waveform",
        action="store_true",
        help="Validate Kokoro v1's seven-input masked full-waveform contract.",
    )
    parser.add_argument(
        "--v1-acoustic",
        action="store_true",
        help="Validate Kokoro v1's masked acoustic-output contract.",
    )
    parser.add_argument(
        "--v1-acoustic-unmasked",
        action="store_true",
        help="Diagnostic only: validate Kokoro v1's three-input exact-length acoustic contract.",
    )
    parser.add_argument(
        "--vtcm-mb",
        type=int,
        default=0,
        help="Explicit HTP VTCM reservation in MiB (0 uses QAIRT default).",
    )
    parser.add_argument(
        "--finalizer-mode",
        choices=("0", "1", "2", "3"),
        default="3",
        help="QNN HTP graph finalization optimization mode (default: 3).",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Diagnostic only: permit unsupported nodes to remain on CPU.",
    )
    parser.add_argument(
        "--minimum-context-bytes",
        type=int,
        default=1_000_000,
        help=(
            "Minimum embedded EPContext payload size. The production default "
            "rejects suspiciously small contexts; deterministic diagnostic "
            "probes may explicitly lower it."
        ),
    )
    parser.add_argument(
        "--diagnostic-probe",
        action="store_true",
        help="Diagnostic only: validate the source model's exact input/output contract.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if sum((args.v1_full_waveform, args.v1_acoustic, args.v1_acoustic_unmasked)) > 1:
        parser.error("v1 contract modes are mutually exclusive")
    if args.bucket < 16:
        parser.error("--bucket must be at least 16 frames")
    if args.minimum_context_bytes < 1:
        parser.error("--minimum-context-bytes must be positive")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing context: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ort_qnn.setup_library_path()
    ort.register_execution_provider_library(
        ort_qnn.get_ep_name(), ort_qnn.get_library_path()
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.verbose:
        options.log_severity_level = 0
        options.log_verbosity_level = 1
    dimensions = {
        "unk__443": 1,
        "unk__445": args.bucket,
        "unk__648": args.bucket,
        "kokoro_mask_10_length": args.bucket * 10,
        "kokoro_mask_60_length": args.bucket * 60 + 1,
    }
    if not (args.v1_acoustic or args.v1_acoustic_unmasked):
        dimensions["audio_length"] = args.bucket * 300
    for name, value in dimensions.items():
        options.add_free_dimension_override_by_name(name, value)
    if not args.allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("ep.context_embed_mode", "1")
    options.add_session_config_entry("ep.context_file_path", str(args.output.resolve()))

    provider_options = {
        "backend_path": ort_qnn.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": args.finalizer_mode,
        "soc_model": args.soc_model,
        "htp_arch": args.htp_arch,
        "enable_htp_fp16_precision": "1" if args.fp16 else "0",
    }
    if args.vtcm_mb > 0:
        provider_options["vtcm_mb"] = str(args.vtcm_mb)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        provider_options.update(
            {
                "enable_framework_op_trace": "1",
                "framework_op_trace_dir": str(args.trace_dir.resolve()),
                "dump_json_qnn_graph": "1",
                "json_qnn_graph_dir": str(args.trace_dir.resolve()),
            }
        )
    qnn_devices = [
        device for device in ort.get_ep_devices() if device.ep_name == ort_qnn.get_ep_name()
    ]
    if not qnn_devices:
        raise RuntimeError("Registered QNN plugin exposed no OrtEpDevice")
    # Plugin EPs must be selected through the device API. Passing the legacy
    # providers=[...] list can silently leave every node on the built-in CPU EP.
    options.add_provider_for_devices(qnn_devices, provider_options)
    session = ort.InferenceSession(str(args.model.resolve()), sess_options=options)
    del session
    if not args.output.is_file():
        raise RuntimeError("QNN session initialized but did not emit an EPContext model")

    context = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(context)
    context_nodes = [node for node in context.graph.node if node.op_type == "EPContext"]
    if not context_nodes:
        raise RuntimeError("Generated model contains no QNN EPContext node")
    if not args.allow_cpu_fallback and len(context.graph.node) != 1:
        raise RuntimeError("Generated model is not a single fully delegated EPContext node")
    if context.graph.initializer or context.graph.sparse_initializer or context.functions:
        raise RuntimeError("Generated EPContext unexpectedly retains graph weights or functions")
    if any(tensor.data_location == TensorProto.EXTERNAL for tensor in context.graph.initializer):
        raise RuntimeError("Generated EPContext unexpectedly references external tensor data")

    source_model = onnx.load(str(args.model), load_external_data=False)
    if args.diagnostic_probe:
        expected_input_contracts = [value_contract(value) for value in source_model.graph.input]
        expected_inputs = {
            item["name"]: item["shape"] for item in expected_input_contracts
        }
    else:
        expected_inputs = {
            "/decoder/Slice_output_0": [1, 128],
            "/decoder/decoder/Unsqueeze_output_0": [1, 1, args.bucket],
            "/decoder/decoder/decode.3/Div_4_output_0": [1, 512, args.bucket],
        }
    if not args.diagnostic_probe and not args.v1_acoustic_unmasked:
        expected_inputs.update(
            {
                "valid_mask_10": [1, 1, args.bucket * 10],
                "valid_mask_60": [1, 1, args.bucket * 60 + 1],
            }
        )
    model_input_names = {value.name for value in source_model.graph.input}
    if "valid_length_10" in model_input_names or "valid_length_60" in model_input_names:
        if not {"valid_length_10", "valid_length_60"}.issubset(model_input_names):
            raise RuntimeError("Masked source exposes only one valid-length input")
        expected_inputs.update(
            {
                "valid_length_10": [1],
                "valid_length_60": [1],
            }
        )
    input_contracts = [value_contract(value) for value in context.graph.input]
    output_contracts = [value_contract(value) for value in context.graph.output]
    actual_inputs = {item["name"]: item for item in input_contracts}
    if set(actual_inputs) != set(expected_inputs):
        raise RuntimeError(f"Generated EPContext inputs changed: {sorted(actual_inputs)}")
    for name, shape in expected_inputs.items():
        contract = actual_inputs[name]
        if contract["element_type"] != "FLOAT" or contract["shape"] != shape:
            raise RuntimeError(f"Generated EPContext input contract changed: {contract}")
    if args.diagnostic_probe:
        expected_outputs = [value_contract(value) for value in source_model.graph.output]
        if output_contracts != expected_outputs:
            raise RuntimeError(
                f"Generated diagnostic output contract changed: {output_contracts} != {expected_outputs}"
            )
        expected_output = None
    elif args.v1_full_waveform:
        expected_output = {
            "name": "waveform",
            "element_type": "FLOAT",
            "shape": [1, args.bucket * 300],
        }
    elif args.v1_acoustic or args.v1_acoustic_unmasked:
        expected_output = {
            "name": "acoustic",
            "element_type": "FLOAT",
            "shape": [1, 22, args.bucket * 60 + 1],
        }
    else:
        expected_output = {
            "name": "audio",
            "element_type": "FLOAT",
            "shape": [args.bucket * 300],
        }
    if expected_output is not None and output_contracts != [expected_output]:
        raise RuntimeError(f"Generated EPContext output contract changed: {output_contracts}")

    context_node = context_nodes[0]
    embedded = attribute_value(context_node, "ep_cache_context")
    if not isinstance(embedded, bytes) or len(embedded) < args.minimum_context_bytes:
        raise RuntimeError("Generated EPContext does not contain a substantial embedded context")
    if attribute_value(context_node, "main_context") != 1:
        raise RuntimeError("Generated EPContext is not marked as the main context")
    if attribute_value(context_node, "embed_mode") != 1:
        raise RuntimeError("Generated EPContext is not embedded")
    if attribute_value(context_node, "source") != b"QNNExecutionProvider":
        raise RuntimeError("Generated EPContext source is not QNNExecutionProvider")
    receipt = {
        "source": str(args.model.resolve()),
        "source_sha256": sha256(args.model),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "bucket_frames": args.bucket,
        "dimensions": dimensions,
        "soc_model": args.soc_model,
        "htp_arch": args.htp_arch,
        "fp16": args.fp16,
        "vtcm_mb": args.vtcm_mb,
        "finalizer_mode": args.finalizer_mode,
        "provider_options": provider_options,
        "onnxruntime": ort.__version__,
        "onnxruntime_qnn": ort_qnn.__version__,
        "context_metadata": {item.key: item.value for item in context.metadata_props},
        "context_nodes": len(context_nodes),
        "model_nodes": len(context.graph.node),
        "cpu_fallback_disabled": not args.allow_cpu_fallback,
        "inputs": input_contracts,
        "outputs": output_contracts,
        "ep_sdk_version": attribute_value(context_node, "ep_sdk_version").decode("utf-8"),
        "embedded_context_bytes": len(embedded),
        "embedded_context_sha256": hashlib.sha256(embedded).hexdigest(),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
