"""Qualify a smallest-fitting fixed masked-generator bucket policy on the host.

The Qualcomm EPContext itself can only be executed on the target Android HTP.
This gate therefore runs the exact fixed source graphs (or an explicitly supplied
FP16 simulation graph) on CPU, applies the same padding masks/crop/headroom as the
app, and compares them with the unpadded FP32 generator.  It proves bucket and
numeric fidelity; it does not replace the physical-device QNN gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

from validate_audio_quality import (
    DEFAULT_SENTENCES,
    DEFAULT_VOICES,
    load_dictionary,
    phonemize,
    spectral_metrics,
    validate_case,
)


CUT_TENSORS = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)
FRAME_SAMPLES = 300
HEADROOM = 0.95
READY_SENTENCE = "Ready when you are."


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def parse_bucket(value: str) -> tuple[int, Path]:
    try:
        bucket_text, path_text = value.split("=", 1)
        bucket = int(bucket_text)
    except ValueError as problem:
        raise argparse.ArgumentTypeError("bucket must be FRAMES=MODEL.onnx") from problem
    path = Path(path_text)
    if bucket < 16:
        raise argparse.ArgumentTypeError("bucket must be at least 16 frames")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"bucket model does not exist: {path}")
    return bucket, path


def parse_integer_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split("=", 1)
        return int(left), int(right)
    except ValueError as problem:
        raise argparse.ArgumentTypeError("value must be BUCKET=MINIMUM_FRAMES") from problem


def headroom(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return audio
    gain = min(1.0, HEADROOM / peak)
    return np.asarray(audio * gain, dtype=np.float32)


def fixed_feeds(middle: list[np.ndarray], frames: int, bucket: int) -> dict[str, np.ndarray]:
    pad = bucket - frames
    if pad < 0:
        raise ValueError(f"T={frames} exceeds B={bucket}")
    return {
        CUT_TENSORS[0]: np.asarray(middle[0], dtype=np.float32),
        CUT_TENSORS[1]: np.pad(
            np.asarray(middle[1], dtype=np.float32), ((0, 0), (0, 0), (0, pad))
        ),
        CUT_TENSORS[2]: np.pad(
            np.asarray(middle[2], dtype=np.float32), ((0, 0), (0, 0), (0, pad))
        ),
        "valid_mask_10": np.concatenate(
            [
                np.ones(frames * 10, dtype=np.float32),
                np.zeros(pad * 10, dtype=np.float32),
            ]
        ).reshape(1, 1, bucket * 10),
        "valid_mask_60": np.concatenate(
            [
                np.ones(frames * 60 + 1, dtype=np.float32),
                np.zeros(pad * 60, dtype=np.float32),
            ]
        ).reshape(1, 1, bucket * 60 + 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--reference-generator", required=True, type=Path)
    parser.add_argument(
        "--bucket",
        required=True,
        action="append",
        type=parse_bucket,
        metavar="FRAMES=MODEL.onnx",
    )
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--all-voices", action="store_true")
    parser.add_argument("--speeds", nargs="+", type=float, default=[1.0])
    parser.add_argument("--text", action="append")
    parser.add_argument(
        "--minimum-use",
        action="append",
        type=parse_integer_pair,
        default=[],
        metavar="BUCKET=MINIMUM_FRAMES",
        help="Require this many real frames before selecting a larger padded bucket.",
    )
    parser.add_argument(
        "--maximum-use",
        action="append",
        type=parse_integer_pair,
        default=[],
        metavar="BUCKET=MAXIMUM_FRAMES",
        help="Stop selecting a bucket above a qualified real-frame limit.",
    )
    parser.add_argument(
        "--allow-policy-split",
        action="store_true",
        help="Report but do not fail cases in a deliberate gap between bucket ranges.",
    )
    parser.add_argument(
        "--allow-oversize",
        action="store_true",
        help="Report but do not fail cases whose front output exceeds the largest bucket.",
    )
    args = parser.parse_args()

    bucket_paths = dict(args.bucket)
    if len(bucket_paths) != len(args.bucket):
        parser.error("duplicate bucket size")
    buckets = sorted(bucket_paths)
    minimum_use = {bucket: 1 for bucket in buckets}
    maximum_use = {bucket: bucket for bucket in buckets}
    for bucket, minimum in args.minimum_use:
        if bucket not in bucket_paths:
            parser.error(f"--minimum-use refers to undeclared B{bucket}")
        if minimum < 1 or minimum > bucket:
            parser.error(f"invalid minimum-use B{bucket}={minimum}")
        minimum_use[bucket] = minimum
    for bucket, maximum in args.maximum_use:
        if bucket not in bucket_paths:
            parser.error(f"--maximum-use refers to undeclared B{bucket}")
        if maximum < 1 or maximum > bucket:
            parser.error(f"invalid maximum-use B{bucket}={maximum}")
        maximum_use[bucket] = maximum
    for bucket in buckets:
        if minimum_use[bucket] > maximum_use[bucket]:
            parser.error(
                f"empty policy range for B{bucket}: "
                f"{minimum_use[bucket]}-{maximum_use[bucket]}"
            )
    sessions = {bucket: make_session(bucket_paths[bucket]) for bucket in buckets}
    reference_generator = make_session(args.reference_generator)
    front = make_session(args.front)

    for bucket, session in sessions.items():
        inputs = {item.name: list(item.shape) for item in session.get_inputs()}
        expected = {
            CUT_TENSORS[0]: [1, 128],
            CUT_TENSORS[1]: [1, 1, bucket],
            CUT_TENSORS[2]: [1, 512, bucket],
            "valid_mask_10": [1, 1, bucket * 10],
            "valid_mask_60": [1, 1, bucket * 60 + 1],
        }
        if inputs != expected:
            raise RuntimeError(f"B{bucket} input contract changed: {inputs}")
        outputs = session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "audio" or list(outputs[0].shape) != [bucket * 300]:
            raise RuntimeError(f"B{bucket} output contract changed")

    vocabulary = json.loads((args.assets / "vocab.json").read_text(encoding="utf-8"))["vocab"]
    dictionary = load_dictionary(args.assets / "cmudict_ipa.dict")
    voices = (
        sorted(path.stem for path in (args.assets / "voices").glob("*.npy"))
        if args.all_voices
        else list(DEFAULT_VOICES)
    )
    texts = list(args.text or (READY_SENTENCE, *DEFAULT_SENTENCES))
    failures: list[str] = []
    skipped: list[str] = []
    policy_splits: list[str] = []
    bucket_counts: Counter[int] = Counter()
    worst_log = (-1.0, "")
    worst_corr = (2.0, "")
    worst_rms = (-1.0, "")
    worst_high_margin = (-1.0, "")

    for voice_name in voices:
        voice = np.load(args.assets / "voices" / f"{voice_name}.npy", mmap_mode="r")
        for text_index, text in enumerate(texts, start=1):
            phonemes = phonemize(text, dictionary)
            token_ids = [vocabulary[char] for char in phonemes if char in vocabulary]
            tokens = np.asarray([[0, *token_ids, 0]], dtype=np.int64)
            style = np.asarray(
                voice[min(509, max(0, len(token_ids) - 1))], dtype=np.float32
            ).reshape(1, 256)
            for speed in args.speeds:
                middle = front.run(
                    list(CUT_TENSORS),
                    {
                        "tokens": tokens,
                        "style": style,
                        "speed": np.asarray([speed], dtype=np.float32),
                    },
                )
                frames = int(middle[1].shape[-1])
                label = f"{voice_name}/text{text_index}/speed{speed:g}/T{frames}"
                bucket = next(
                    (
                        candidate
                        for candidate in buckets
                        if minimum_use[candidate] <= frames <= maximum_use[candidate]
                    ),
                    None,
                )
                if bucket is None:
                    if frames <= buckets[-1]:
                        policy_splits.append(label)
                        print(
                            f"POLICY_SPLIT {label} ranges="
                            + ",".join(
                                f"{minimum_use[candidate]}-{maximum_use[candidate]}"
                                for candidate in buckets
                            ),
                            flush=True,
                        )
                    else:
                        skipped.append(label)
                        print(f"OVERSIZE {label} maxB={buckets[-1]}", flush=True)
                    continue

                unpadded = dict(zip(CUT_TENSORS, middle))
                reference = np.asarray(
                    reference_generator.run(None, unpadded)[0], dtype=np.float32
                ).reshape(-1)
                candidate = np.asarray(
                    sessions[bucket].run(None, fixed_feeds(middle, frames, bucket))[0],
                    dtype=np.float32,
                ).reshape(-1)[: frames * FRAME_SAMPLES]
                reference = headroom(reference)
                candidate = headroom(candidate)
                case_failures = validate_case(reference, candidate)
                full_label = f"{label}/B{bucket}/pad{bucket - frames}"
                if case_failures:
                    failures.append(f"{full_label}: {', '.join(case_failures)}")
                    print(f"FAIL {failures[-1]}", flush=True)
                    continue

                log_mae, correlation, reference_high, candidate_high = spectral_metrics(
                    reference, candidate
                )
                delta = candidate - reference[: candidate.size]
                rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
                high_margin = candidate_high - (reference_high * 1.20 + 0.003)
                if log_mae > worst_log[0]:
                    worst_log = (log_mae, full_label)
                if correlation < worst_corr[0]:
                    worst_corr = (correlation, full_label)
                if rms > worst_rms[0]:
                    worst_rms = (rms, full_label)
                if high_margin > worst_high_margin[0]:
                    worst_high_margin = (high_margin, full_label)
                bucket_counts[bucket] += 1
                print(
                    f"PASS {full_label} logMAE={log_mae:.6f} corr={correlation:.6f} "
                    f"rms={rms:.7f}",
                    flush=True,
                )

    if len(bucket_counts) != len(buckets):
        failures.append(
            f"policy did not exercise every bucket: observed={dict(bucket_counts)}, expected={buckets}"
        )
    if skipped and not args.allow_oversize:
        failures.append(f"{len(skipped)} cases exceeded B{buckets[-1]}")
    if policy_splits and not args.allow_policy_split:
        failures.append(f"{len(policy_splits)} cases require recursive policy splitting")

    summary = {
        "voices": len(voices),
        "texts": len(texts),
        "speeds": args.speeds,
        "passed": sum(bucket_counts.values()),
        "failed": len(failures),
        "oversize": len(skipped),
        "policy_splits": len(policy_splits),
        "bucket_ranges": {
            str(bucket): [minimum_use[bucket], maximum_use[bucket]] for bucket in buckets
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "worst_log_mae": worst_log,
        "minimum_correlation": worst_corr,
        "worst_rms": worst_rms,
        "worst_high_band_margin": worst_high_margin,
    }
    print("BUCKET POLICY SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if failures:
        print("BUCKET POLICY AUDIO GATE FAILED", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("BUCKET POLICY AUDIO GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
