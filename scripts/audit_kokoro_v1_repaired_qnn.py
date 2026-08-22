#!/usr/bin/env python3
"""Audit the repaired Kokoro v1 QNN artifact family before APK packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


BUCKETS = (64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def audit_context(path: Path, bucket: int) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert (node.domain, node.op_type) == ("com.microsoft", "EPContext")
    assert not model.graph.initializer and not model.graph.sparse_initializer
    expected_inputs = {
        "/decoder/Slice_output_0": [1, 128],
        "/decoder/decoder/decode.3/Div_4_output_0": [1, 512, bucket],
        "valid_mask_10": [1, 1, 10 * bucket],
        "valid_mask_60": [1, 1, 60 * bucket + 1],
        "valid_length_10": [1],
        "valid_length_60": [1],
        "kokoro_source_spectrum": [1, 22, 60 * bucket + 1],
    }
    actual_inputs = {value.name: shape(value) for value in model.graph.input}
    assert actual_inputs == expected_inputs, (bucket, actual_inputs)
    assert len(model.graph.output) == 1
    output = model.graph.output[0]
    assert output.name == "acoustic"
    assert output.type.tensor_type.elem_type == TensorProto.FLOAT
    assert shape(output) == [1, 22, 60 * bucket + 1]
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
    assert attributes["embed_mode"] == 1 and attributes["main_context"] == 1
    assert attributes["source"] == b"QNNExecutionProvider"
    return {"bytes": path.stat().st_size, "sha256": sha256(path), "inputs": actual_inputs}


def audit_source(path: Path, bucket: int) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    assert {value.name: shape(value) for value in model.graph.input} == {
        "/decoder/decoder/Unsqueeze_output_0": [1, 1, bucket]
    }
    assert len(model.graph.output) == 1
    assert shape(model.graph.output[0]) == [1, 22, 60 * bucket + 1]
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def audit_device_outputs(directory: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for path in sorted(directory.glob("*-f32le.raw")):
        values = np.fromfile(path, dtype="<f4")
        assert values.size and np.isfinite(values).all(), path
        peak = float(np.max(np.abs(values)))
        rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        assert peak > 1e-4 and rms > 1e-5, path
        spectrum = np.abs(np.fft.rfft(values.astype(np.float64))) ** 2
        frequencies = np.fft.rfftfreq(values.size, 1 / 24_000)
        high_fraction = float(spectrum[frequencies >= 8_000].sum() / max(spectrum.sum(), 1e-30))
        assert high_fraction < 0.05, (path, high_fraction)
        outputs[path.name] = {
            "samples": int(values.size),
            "peak": peak,
            "rms": rms,
            "high_frequency_fraction_8khz": high_fraction,
        }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--device-outputs",
        type=Path,
        help="Directory containing silent physical-device float captures (defaults to ARTIFACTS/device_gate)",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    contexts: dict[str, object] = {}
    sources: dict[str, object] = {}
    for bucket in BUCKETS:
        contexts[str(bucket)] = audit_context(
            args.artifacts / f"kokoro-v1-neural-vocoder-b{bucket}.qnn248.powmul.ctx.onnx", bucket
        )
        sources[str(bucket)] = audit_source(
            args.artifacts / f"kokoro-v1-source-spectrum-b{bucket}.fp32.onnx", bucket
        )
    device = audit_device_outputs(args.device_outputs or args.artifacts / "device_gate")
    assert device, "No physical-device audio captures were supplied"
    receipt = {"result": "PASS", "buckets": list(BUCKETS), "contexts": contexts, "sources": sources,
               "device_outputs": device}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(contexts)} contexts, {len(sources)} CPU prefixes, {len(device)} device outputs")


if __name__ == "__main__":
    main()
