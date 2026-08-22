"""Compile multiple fixed Kokoro graphs into one weight-shared QNN context.

The source models are compiled sequentially into a single external QNN binary.
Each emitted ONNX wrapper selects one graph from that binary.  This deliberately
uses non-embedded EPContext models: ONNX Runtime's QNN provider documents that
embedded contexts and shared EP contexts are incompatible.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import onnx
import onnxruntime as ort
import onnxruntime_qnn as ort_qnn
from onnx import TensorProto


@dataclass(frozen=True)
class GraphSpec:
    bucket: int
    model: Path
    waveform: bool
    vtcm_mb: int
    finalizer_mode: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attribute_value(node: onnx.NodeProto, name: str) -> object | None:
    attribute = next((item for item in node.attribute if item.name == name), None)
    return None if attribute is None else onnx.helper.get_attribute_value(attribute)


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


def parse_graph(raw: str) -> GraphSpec:
    # bucket|model|kind|vtcm_mb|finalizer_mode (a pipe avoids Windows drive-colon ambiguity)
    parts = raw.split("|", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--graph must be BUCKET|MODEL|acoustic-or-waveform|VTCM_MB|FINALIZER_MODE"
        )
    bucket_text, model_text, kind, vtcm_text, finalizer_mode = parts
    try:
        bucket = int(bucket_text)
        vtcm_mb = int(vtcm_text)
    except ValueError as failure:
        raise argparse.ArgumentTypeError(str(failure)) from failure
    if bucket < 16:
        raise argparse.ArgumentTypeError("bucket must be at least 16")
    if kind not in {"acoustic", "waveform"}:
        raise argparse.ArgumentTypeError("kind must be acoustic or waveform")
    if vtcm_mb < 0:
        raise argparse.ArgumentTypeError("VTCM_MB cannot be negative")
    if finalizer_mode not in {"0", "1", "2", "3"}:
        raise argparse.ArgumentTypeError("FINALIZER_MODE must be 0, 1, 2, or 3")
    return GraphSpec(bucket, Path(model_text), kind == "waveform", vtcm_mb, finalizer_mode)


def dimensions(spec: GraphSpec) -> dict[str, int]:
    values = {
        "unk__443": 1,
        "unk__445": spec.bucket,
        "unk__648": spec.bucket,
        "kokoro_mask_10_length": spec.bucket * 10,
        "kokoro_mask_60_length": spec.bucket * 60 + 1,
    }
    if spec.waveform:
        values["audio_length"] = spec.bucket * 300
    return values


def expected_inputs(spec: GraphSpec) -> dict[str, list[int]]:
    result = {
        "/decoder/Slice_output_0": [1, 128],
        "/decoder/decoder/Unsqueeze_output_0": [1, 1, spec.bucket],
        "/decoder/decoder/decode.3/Div_4_output_0": [1, 512, spec.bucket],
        "valid_mask_10": [1, 1, spec.bucket * 10],
        "valid_mask_60": [1, 1, spec.bucket * 60 + 1],
        "valid_length_10": [1],
        "valid_length_60": [1],
    }
    return result


def validate_wrapper(path: Path, spec: GraphSpec) -> tuple[str, dict[str, object]]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    nodes = [node for node in model.graph.node if node.op_type == "EPContext"]
    if len(nodes) != 1 or len(model.graph.node) != 1:
        raise RuntimeError(f"{path.name} is not one fully delegated EPContext node")
    if model.graph.initializer or model.graph.sparse_initializer or model.functions:
        raise RuntimeError(f"{path.name} unexpectedly retains graph weights/functions")
    node = nodes[0]
    if attribute_value(node, "embed_mode") != 0:
        raise RuntimeError(f"{path.name} is not an external EPContext wrapper")
    if attribute_value(node, "source") != b"QNNExecutionProvider":
        raise RuntimeError(f"{path.name} has the wrong execution provider")
    external = attribute_value(node, "ep_cache_context")
    if not isinstance(external, bytes) or not external:
        raise RuntimeError(f"{path.name} does not reference an external QNN binary")
    context_name = external.decode("utf-8")

    contracts = [value_contract(value) for value in model.graph.input]
    actual = {item["name"]: item for item in contracts}
    expected = expected_inputs(spec)
    if set(actual) != set(expected):
        raise RuntimeError(f"{path.name} inputs changed: {sorted(actual)}")
    for name, shape in expected.items():
        if actual[name]["element_type"] != "FLOAT" or actual[name]["shape"] != shape:
            raise RuntimeError(f"{path.name} input changed: {actual[name]}")
    outputs = [value_contract(value) for value in model.graph.output]
    expected_output = {
        "name": "waveform" if spec.waveform else "acoustic",
        "element_type": "FLOAT",
        "shape": [1, spec.bucket * 300] if spec.waveform else [1, 22, spec.bucket * 60 + 1],
    }
    if outputs != [expected_output]:
        raise RuntimeError(f"{path.name} output changed: {outputs}")
    return context_name, {
        "wrapper": str(path.resolve()),
        "wrapper_bytes": path.stat().st_size,
        "wrapper_sha256": sha256(path),
        "source": str(spec.model.resolve()),
        "source_bytes": spec.model.stat().st_size,
        "source_sha256": sha256(spec.model),
        "bucket": spec.bucket,
        "waveform": spec.waveform,
        "vtcm_mb": spec.vtcm_mb,
        "finalizer_mode": spec.finalizer_mode,
        "inputs": contracts,
        "outputs": outputs,
        "main_context": attribute_value(node, "main_context"),
        "ep_sdk_version": attribute_value(node, "ep_sdk_version").decode("utf-8"),
        "external_context": context_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graph", action="append", required=True, type=parse_graph)
    parser.add_argument("--soc-model", default="57")
    parser.add_argument("--htp-arch", default="75")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    specs: list[GraphSpec] = args.graph
    if len({spec.bucket for spec in specs}) != len(specs):
        parser.error("Every shared graph must have a unique bucket")
    missing = [str(spec.model) for spec in specs if not spec.model.is_file()]
    if missing:
        parser.error(f"Missing source models: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Refusing to use non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ort_qnn.setup_library_path()
    ort.register_execution_provider_library(ort_qnn.get_ep_name(), ort_qnn.get_library_path())
    qnn_devices = [
        device for device in ort.get_ep_devices() if device.ep_name == ort_qnn.get_ep_name()
    ]
    if not qnn_devices:
        raise RuntimeError("Registered QNN plugin exposed no OrtEpDevice")

    wrappers: list[Path] = []
    for index, spec in enumerate(specs):
        wrapper = args.output_dir / f"kokoro-v1-b{spec.bucket}.shared.ctx.onnx"
        wrappers.append(wrapper)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if args.verbose:
            options.log_severity_level = 0
            options.log_verbosity_level = 1
        for name, value in dimensions(spec).items():
            options.add_free_dimension_override_by_name(name, value)
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_embed_mode", "0")
        options.add_session_config_entry("ep.context_file_path", str(wrapper.resolve()))
        options.add_session_config_entry("ep.context_node_name_prefix", f"kokoro_b{spec.bucket}")
        options.add_session_config_entry("ep.share_ep_contexts", "1")
        if index == len(specs) - 1:
            options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
        provider_options = {
            "backend_path": ort_qnn.get_qnn_htp_path(),
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": spec.finalizer_mode,
            "soc_model": args.soc_model,
            "htp_arch": args.htp_arch,
            "enable_htp_fp16_precision": "1",
        }
        if spec.vtcm_mb:
            provider_options["vtcm_mb"] = str(spec.vtcm_mb)
        options.add_provider_for_devices(qnn_devices, provider_options)
        print(
            f"[{index + 1}/{len(specs)}] compiling B{spec.bucket} "
            f"kind={'waveform' if spec.waveform else 'acoustic'} "
            f"vtcm={spec.vtcm_mb or 'default'} mode={spec.finalizer_mode}",
            flush=True,
        )
        # The QNN backend manager is retained by the provider's shared
        # workspace until the final graph sets ep.stop_share_ep_contexts.
        # Releasing each ORT session here drops the parsed ONNX graph and
        # its initializer copies instead of accumulating all source models
        # in host RAM during an eleven-graph compile.
        session = ort.InferenceSession(str(spec.model.resolve()), sess_options=options)
        del session
        del options
        gc.collect()

    wrapper_receipts: list[dict[str, object]] = []
    context_names: set[str] = set()
    for path, spec in zip(wrappers, specs, strict=True):
        if not path.is_file():
            raise RuntimeError(f"Shared compilation did not emit {path}")
        context_name, receipt = validate_wrapper(path, spec)
        context_names.add(context_name)
        wrapper_receipts.append(receipt)
    if len(context_names) != 1:
        raise RuntimeError(f"Wrappers do not share one context binary: {sorted(context_names)}")
    context_path = args.output_dir / next(iter(context_names))
    if not context_path.is_file() or context_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Shared QNN binary is missing or too small: {context_path}")
    receipt = {
        "soc_model": args.soc_model,
        "htp_arch": args.htp_arch,
        "onnxruntime": ort.__version__,
        "onnxruntime_qnn": ort_qnn.__version__,
        "cpu_fallback_disabled": True,
        "weight_sharing": True,
        "embedded": False,
        "graph_count": len(specs),
        "context": str(context_path.resolve()),
        "context_bytes": context_path.stat().st_size,
        "context_sha256": sha256(context_path),
        "wrappers": wrapper_receipts,
    }
    receipt_path = args.output_dir / "shared-context-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
