"""Build a mobile accelerator graph for Kokoro v0.19.

Kokoro's decoder is dominated by 1-D convolutions. Android NNAPI only accepts
2-D QLinearConv, so the normal dynamic-int8 export cannot use the Snapdragon
accelerator. This tool performs a semantics-preserving 1-D -> 2-D rewrite,
calibrates convolution activations into an unsigned static-int8 QLinearConv
graph, and can optionally rewrite the tiny-frame STFT as a convolution. The
original dynamic-int8 graph remains useful as a portable comparison baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


CALIBRATION_SENTENCES = (
    "Ready when you are.",
    "This is an example of speech synthesis in English.",
    "A fast offline voice should begin speaking without an awkward pause.",
    "Numbers like 24 and punctuation test timing, emphasis, and cadence!",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Battery efficient neural speech is useful during a long afternoon of reading.",
)

# The broad profile intentionally overlaps the audio-quality gate's harder text
# classes while keeping calibration bounded. Every bundled voice is exercised
# with both records by ``KokoroCalibrationReader(profile="all-voices")``.
BROAD_CALIBRATION_SENTENCES = (
    "This is an example of speech synthesis in English.",
    "Numbers like 24, 3.14159, and 2026 should sound natural, not noisy.",
)
CALIBRATION_PROFILES = ("smoke", "all-voices")


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def replace_stft_with_convolution(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Kokoro's 20-point STFT with an equivalent strided DFT Conv."""
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in rewritten.graph.initializer}
    nodes: list[onnx.NodeProto] = []
    replacements = 0
    for index, node in enumerate(rewritten.graph.node):
        if node.op_type != "STFT" or len(node.input) < 4:
            nodes.append(node)
            continue
        frame_step = int(np.asarray(initializers[node.input[1]]).reshape(-1)[0])
        window = np.asarray(initializers[node.input[2]], dtype=np.float32).reshape(-1)
        frame_length = int(np.asarray(initializers[node.input[3]]).reshape(-1)[0])
        if frame_length != window.size:
            raise ValueError("STFT frame length and window length differ")

        bins = frame_length // 2 + 1
        sample = np.arange(frame_length, dtype=np.float32)
        kernels = []
        for frequency in range(bins):
            angle = 2.0 * np.pi * frequency * sample / frame_length
            kernels.append(window * np.cos(angle))
            kernels.append(window * -np.sin(angle))
        weights = np.asarray(kernels, dtype=np.float32).reshape(bins * 2, 1, frame_length)

        stem = f"{node.name or 'STFT'}.mobile_dft.{index}"
        axes_name = f"{stem}.axes"
        weights_name = f"{stem}.weights"
        shape_name = f"{stem}.shape"
        rewritten.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray([1], dtype=np.int64), axes_name),
                numpy_helper.from_array(weights, weights_name),
                numpy_helper.from_array(np.asarray([0, 0, bins, 2], dtype=np.int64), shape_name),
            ]
        )
        expanded = f"{stem}.expanded"
        convolved = f"{stem}.convolved"
        transposed = f"{stem}.transposed"
        nodes.extend(
            [
                helper.make_node("Unsqueeze", [node.input[0], axes_name], [expanded], name=f"{stem}.unsqueeze"),
                helper.make_node(
                    "Conv",
                    [expanded, weights_name],
                    [convolved],
                    name=f"{stem}.conv",
                    kernel_shape=[frame_length],
                    strides=[frame_step],
                ),
                helper.make_node("Transpose", [convolved], [transposed], name=f"{stem}.transpose", perm=[0, 2, 1]),
                helper.make_node("Reshape", [transposed, shape_name], list(node.output), name=f"{stem}.reshape"),
            ]
        )
        replacements += 1

    if replacements != 1:
        raise ValueError(f"Expected one Kokoro STFT node, found {replacements}")
    del rewritten.graph.node[:]
    rewritten.graph.node.extend(nodes)
    onnx.checker.check_model(rewritten)
    return rewritten


def promote_convolutions_to_2d(model: onnx.ModelProto) -> onnx.ModelProto:
    """Represent Conv1D/ConvTranspose1D as equivalent height-one 2-D ops."""
    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    nodes: list[onnx.NodeProto] = []
    initializers = {initializer.name: initializer for initializer in rewritten.graph.initializer}
    promoted_weights: set[str] = set()
    axes_name = "kokoro.mobile.axes_2"
    if not any(initializer.name == axes_name for initializer in rewritten.graph.initializer):
        rewritten.graph.initializer.append(
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), axes_name)
        )

    for index, node in enumerate(rewritten.graph.node):
        attrs = _attributes(node)
        kernel = attrs.get("kernel_shape")
        if node.op_type not in {"Conv", "ConvTranspose"} or not kernel or len(kernel) != 1:
            nodes.append(node)
            continue

        stem = f"{node.name or node.op_type}.mobile2d.{index}"
        input_2d = f"{stem}.input"
        weight_2d = f"{stem}.weight"
        output_2d = f"{stem}.output"
        nodes.append(helper.make_node("Unsqueeze", [node.input[0], axes_name], [input_2d], name=f"{stem}.unsqueeze_input"))
        weight_name = node.input[1]
        weight = initializers.get(weight_name)
        if weight is not None:
            if weight_name not in promoted_weights:
                weight.dims.insert(2, 1)
                promoted_weights.add(weight_name)
            weight_2d = weight_name
        else:
            nodes.append(helper.make_node("Unsqueeze", [weight_name, axes_name], [weight_2d], name=f"{stem}.unsqueeze_weight"))

        converted = dict(attrs)
        converted["kernel_shape"] = [1, int(kernel[0])]
        if "dilations" in converted:
            converted["dilations"] = [1, int(converted["dilations"][0])]
        if "strides" in converted:
            converted["strides"] = [1, int(converted["strides"][0])]
        if "pads" in converted:
            begin, end = converted["pads"]
            converted["pads"] = [0, int(begin), 0, int(end)]
        if "output_padding" in converted:
            converted["output_padding"] = [0, int(converted["output_padding"][0])]
        if "output_shape" in converted:
            converted["output_shape"] = [1, int(converted["output_shape"][0])]

        inputs = [input_2d, weight_2d, *node.input[2:]]
        nodes.append(helper.make_node(node.op_type, inputs, [output_2d], name=f"{stem}.{node.op_type.lower()}", **converted))
        nodes.append(helper.make_node("Squeeze", [output_2d, axes_name], list(node.output), name=f"{stem}.squeeze_output"))

    del rewritten.graph.node[:]
    rewritten.graph.node.extend(nodes)
    onnx.checker.check_model(rewritten)
    return rewritten


def _load_dictionary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line or line.startswith(";;;"):
            continue
        word, phonemes = line.split("\t", 1)
        result[word.split("(", 1)[0]] = phonemes.split(",", 1)[0].strip()
    return result


def _phonemize(text: str, dictionary: dict[str, str]) -> str:
    pieces = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    return "".join(
        dictionary.get(piece.upper(), piece.lower()) if any(char.isalpha() for char in piece) else piece
        for piece in pieces
    )


class KokoroCalibrationReader(CalibrationDataReader):
    """Deterministic Kokoro calibration records shared by mobile exporters.

    ``smoke`` retains the original six-record development pass. ``all-voices``
    covers every bundled style with two difficult utterances and three speeds;
    it is the minimum profile appropriate for a release-candidate experiment.
    """

    def __init__(self, assets: Path, profile: str = "smoke", report_progress: bool = False):
        if profile not in CALIBRATION_PROFILES:
            raise ValueError(f"Unknown calibration profile {profile!r}; expected one of {CALIBRATION_PROFILES}")
        vocabulary = json.loads((assets / "vocab.json").read_text(encoding="utf-8"))["vocab"]
        dictionary = _load_dictionary(assets / "cmudict_ipa.dict")
        voices = sorted((assets / "voices").glob("*.npy"))
        if not voices:
            raise ValueError(f"No voice .npy files found under {assets / 'voices'}")

        if profile == "smoke":
            cases = [
                (voices[(index * 5) % len(voices)], sentence, 0.9 + (index % 3) * 0.1)
                for index, sentence in enumerate(CALIBRATION_SENTENCES)
            ]
        else:
            cases = [
                (voice_path, sentence, 0.9 + ((voice_index + sentence_index) % 3) * 0.1)
                for voice_index, voice_path in enumerate(voices)
                for sentence_index, sentence in enumerate(BROAD_CALIBRATION_SENTENCES)
            ]

        records: list[dict[str, np.ndarray]] = []
        loaded_voices: dict[Path, np.ndarray] = {}
        for voice_path, sentence, speed in cases:
            phonemes = _phonemize(sentence, dictionary)
            token_ids = [vocabulary[char] for char in phonemes if char in vocabulary]
            tokens = np.asarray([[0, *token_ids, 0]], dtype=np.int64)
            if voice_path not in loaded_voices:
                loaded_voices[voice_path] = np.load(voice_path, mmap_mode="r")
            voice = loaded_voices[voice_path]
            # Voice rows use the zero-based phoneme count. This must match the
            # Android VoiceStyleStore and Kokoro's official export helper.
            style = np.asarray(
                voice[min(509, max(0, len(token_ids) - 1))],
                dtype=np.float32,
            ).reshape(1, 256)
            records.append(
                {
                    "tokens": tokens,
                    "style": style,
                    "speed": np.asarray([speed], dtype=np.float32),
                }
            )
        self._records = records
        self.profile = profile
        self.record_count = len(records)
        self._report_progress = report_progress
        self.rewind()

    def get_next(self):
        record = next(self._iterator, None)
        if record is not None:
            self._delivered += 1
            if self._report_progress and (
                self._delivered == 1
                or self._delivered % 8 == 0
                or self._delivered == self.record_count
            ):
                print(f"Calibration {self._delivered}/{self.record_count} ({self.profile})", flush=True)
        return record

    def rewind(self):
        self._iterator = iter(self._records)
        self._delivered = 0


def optimize_for_mobile(
    source: Path,
    assets: Path,
    output: Path,
    transformed: Path,
    rewrite_stft: bool = False,
    quantize: bool = True,
) -> None:
    basic = transformed.with_suffix(".basic.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(basic)
    ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])

    model = onnx.load(basic)
    if rewrite_stft:
        model = replace_stft_with_convolution(model)
    model_2d = promote_convolutions_to_2d(model)
    onnx.save(model_2d, transformed)
    if not quantize:
        if output.resolve() != transformed.resolve():
            shutil.copyfile(transformed, output)
        return
    # ORT 1.20's range-adjustment pass expects Softmax ranges even when only
    # Conv is selected. Calibrate those tensors but explicitly leave the
    # Softmax nodes in float so the acoustic model is not needlessly altered.
    float_only_nodes = [
        node.name
        for node in model_2d.graph.node
        if node.op_type == "Softmax" or (node.op_type == "Conv" and "mobile_dft" in node.name)
    ]
    quantize_static(
        model_input=transformed,
        model_output=output,
        calibration_data_reader=KokoroCalibrationReader(assets),
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=["Conv", "Softmax"],
        nodes_to_exclude=float_only_nodes,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Kokoro v0.19 fp32 ONNX")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transformed", type=Path, required=True)
    parser.add_argument(
        "--rewrite-stft",
        action="store_true",
        help="Experimental exact DFT-convolution rewrite; faster but more numerically sensitive",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Write the fidelity-qualified 2-D FP32 graph instead of a lossy static-INT8 export",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimize_for_mobile(
        args.input,
        args.assets,
        args.output,
        args.transformed,
        args.rewrite_stft,
        quantize=not args.fp32,
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
