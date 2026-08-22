"""Create the QDQ candidate required by ONNX Runtime's QNN HTP provider.

The output is experimental until both the host ``validate_audio_quality.py
--full`` gate and the physical Snapdragon gate pass. It is excluded from normal
CPU APKs and never replaces the clear FP32 CPU model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config,
    qnn_preprocess_model,
)

from optimize_kokoro_mobile import CALIBRATION_PROFILES, KokoroCalibrationReader


DIAGNOSTIC_SCOPES = ("all", "decoder", "decoder-no-generator")


def _layernorm_per_tensor_overrides(model: onnx.ModelProto) -> dict[str, list[dict[str, object]]]:
    """Keep LayerNorm weights/biases per-tensor while other weights use per-channel QDQ."""
    overrides: dict[str, list[dict[str, object]]] = {}
    for node in model.graph.node:
        if node.op_type != "LayerNormalization":
            continue
        for input_name in node.input[1:3]:
            if input_name:
                overrides[input_name] = [{"quant_type": QuantType.QUInt8, "symmetric": False}]
    return overrides


def _nodes_for_scope(model: onnx.ModelProto, scope: str) -> list[str] | None:
    if scope == "all":
        return None
    if scope == "decoder":
        return [node.name for node in model.graph.node if node.name.startswith("/decoder/")]
    if scope == "decoder-no-generator":
        return [
            node.name
            for node in model.graph.node
            if node.name.startswith("/decoder/") and "/decoder/decoder/generator/" not in node.name
        ]
    raise ValueError(f"Unknown diagnostic scope {scope!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Fidelity-qualified mobile FP32 graph")
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibration-profile",
        choices=CALIBRATION_PROFILES,
        default="all-voices",
        help="Use all bundled voices for candidate work; smoke is only for bounded diagnostics",
    )
    parser.add_argument(
        "--diagnostic-scope",
        choices=DIAGNOSTIC_SCOPES,
        default="all",
        help="Quantize a named graph region to localize fidelity loss; partial scopes are never HTP candidates",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    assets = args.assets.resolve()
    if output.is_relative_to(assets):
        parser.error("Refusing to write an ungated QDQ model into app assets; generate under build/ and run --full first")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = args.output.with_suffix(".preprocessed.onnx")
    print("QNN QDQ: preprocessing the clear FP32 graph", flush=True)
    changed = qnn_preprocess_model(args.input, preprocessed)
    source = preprocessed if changed else args.input

    model = onnx.load(str(source), load_external_data=False)
    calibration_reader = KokoroCalibrationReader(
        args.assets,
        profile=args.calibration_profile,
        report_progress=True,
    )
    print(
        "QNN QDQ: calibrating uint16 activations and per-channel uint8 weights "
        f"with {calibration_reader.record_count} records",
        flush=True,
    )
    config = get_qnn_qdq_config(
        source,
        calibration_reader,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        # ORT's QNN helper requires explicit per-tensor LayerNorm parameters
        # before global per-channel quantization can be enabled.
        init_overrides=_layernorm_per_tensor_overrides(model),
    )
    config.nodes_to_quantize = _nodes_for_scope(model, args.diagnostic_scope)
    if args.diagnostic_scope != "all":
        print(
            f"DIAGNOSTIC ONLY: scope={args.diagnostic_scope} leaves floating-point compute and is not an HTP candidate.",
            flush=True,
        )
    quantize(source, args.output, config)
    if changed:
        preprocessed.unlink(missing_ok=True)

    quantized = onnx.load(str(args.output), load_external_data=False)
    q_nodes = sum(node.op_type == "QuantizeLinear" for node in quantized.graph.node)
    dq_nodes = sum(node.op_type == "DequantizeLinear" for node in quantized.graph.node)
    receipt = {
        "activation_type": "QUInt16",
        "calibration_profile": args.calibration_profile,
        "calibration_records": calibration_reader.record_count,
        "dequantize_linear_nodes": dq_nodes,
        "diagnostic_scope": args.diagnostic_scope,
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "per_channel_weights": True,
        "quantize_linear_nodes": q_nodes,
        "weight_type": "QUInt8",
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB)", flush=True)
    print(f"Wrote generation receipt {receipt_path} (Q={q_nodes}, DQ={dq_nodes})", flush=True)
    print("This candidate is not releasable until both mandatory audio gates pass.", flush=True)


if __name__ == "__main__":
    main()
