"""Split Kokoro into an exact CPU front-end and a static-bucket QNN generator.

The three cut tensors are natural inputs to Kokoro's waveform generator.  The
two emitted models contain disjoint initializer sets and together reproduce the
source model bit-for-bit on ONNX Runtime CPU.  At runtime the generator's frame
dimensions can be fixed with ORT free-dimension overrides for QNN HTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx.utils import extract_model


CUT_TENSORS = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)
SOURCE_INPUTS = ("tokens", "style", "speed")
SOURCE_OUTPUT = "audio"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dimensions(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        result.append(dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Fidelity-qualified mobile FP32 model")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    front = output_dir / "kokoro-front.fp32.onnx"
    generator = output_dir / "kokoro-generator.fp32.onnx"

    print("Extracting exact CPU front-end", flush=True)
    extract_model(
        str(args.input.resolve()),
        str(front),
        list(SOURCE_INPUTS),
        list(CUT_TENSORS),
        check_model=True,
    )
    print("Extracting waveform generator", flush=True)
    extract_model(
        str(args.input.resolve()),
        str(generator),
        list(CUT_TENSORS),
        [SOURCE_OUTPUT],
        check_model=True,
    )

    generator_model = onnx.load(str(generator), load_external_data=False)
    generator_inputs = {item.name: dimensions(item) for item in generator_model.graph.input}
    expected = {
        CUT_TENSORS[0]: [1, 128],
        CUT_TENSORS[1]: ["unk__443", 1, "unk__445"],
        CUT_TENSORS[2]: [1, 512, "unk__648"],
    }
    if generator_inputs != expected:
        raise RuntimeError(f"Unexpected generator input contract: {generator_inputs!r}")

    receipt = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "cut_tensors": list(CUT_TENSORS),
        "front": {"path": str(front), "bytes": front.stat().st_size, "sha256": sha256(front)},
        "generator": {
            "path": str(generator),
            "bytes": generator.stat().st_size,
            "sha256": sha256(generator),
            "inputs": generator_inputs,
            "frame_samples": 300,
        },
    }
    receipt_path = output_dir / "kokoro-split-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {front} ({front.stat().st_size / 1_000_000:.1f} MB)", flush=True)
    print(f"Wrote {generator} ({generator.stat().st_size / 1_000_000:.1f} MB)", flush=True)
    print(f"Wrote {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
