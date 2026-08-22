"""Bounded, no-fallback QNN HTP context probe for the current Kokoro v1 ONNX.

This is diagnostic only.  It writes nothing into app assets and never changes
the installed APK.  A successful result still needs waveform and S24 gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort
import onnxruntime_qnn as ort_qnn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=48_000)
    parser.add_argument("--length-symbol", action="append", default=["sequence_length"])
    parser.add_argument("--sample-symbol", default="num_samples")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--finalization-mode", default="3")
    parser.add_argument("--htp-fp16", action="store_true")
    parser.add_argument(
        "--vtcm-mb",
        type=int,
        default=0,
        help="Explicit HTP VTCM reservation in MiB (0 leaves QAIRT's default policy intact)",
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    ort_qnn.setup_library_path()
    ort.register_execution_provider_library(ort_qnn.get_ep_name(), ort_qnn.get_library_path())

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 0
    options.log_verbosity_level = 1
    for symbol in args.length_symbol:
        options.add_free_dimension_override_by_name(symbol, args.tokens)
    options.add_free_dimension_override_by_name(args.sample_symbol, args.samples)
    if not args.allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if not args.no_context:
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_embed_mode", "1")
        options.add_session_config_entry("ep.context_file_path", str(args.output.resolve()))
    devices = [item for item in ort.get_ep_devices() if item.ep_name == ort_qnn.get_ep_name()]
    if not devices:
        raise RuntimeError("No QNN EP device was registered")
    provider_options = {
        "backend_path": ort_qnn.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": args.finalization_mode,
        "soc_model": "57",
        "htp_arch": "75",
        "enable_htp_fp16_precision": "1" if args.htp_fp16 else "0",
        "enable_framework_op_trace": "1",
        "framework_op_trace_dir": str(args.trace_dir.resolve()),
    }
    if args.vtcm_mb > 0:
        provider_options["vtcm_mb"] = str(args.vtcm_mb)
    options.add_provider_for_devices(
        devices,
        provider_options,
    )
    try:
        session = ort.InferenceSession(str(args.model.resolve()), sess_options=options)
        del session
    finally:
        # OrtSession.SessionOptions is owned by the created session in this
        # API; ORT 1.26 does not expose a close() method on the options object.
        pass

    result = {
        "context_exists": args.output.is_file(),
        "context_bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "trace_files": [item.name for item in args.trace_dir.glob("*")],
    }
    if args.output.is_file():
        context = onnx.load(args.output, load_external_data=False)
        result["nodes"] = [(node.domain, node.op_type) for node in context.graph.node]
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
