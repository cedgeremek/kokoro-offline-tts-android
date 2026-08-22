"""Validate lossless Kokoro splitting and fixed-bucket generator padding.

The optional masked generator path validates a single fixed QNN bucket whose
normalization receives the real frame count through ``valid_frames``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from validate_audio_quality import (
    DEFAULT_SENTENCES,
    DEFAULT_VOICES,
    FULL_SENTENCES,
    load_dictionary,
    phonemize,
    validate_case,
)


CUT_TENSORS = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)
FRAME_SAMPLES = 300


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--fixed-generator", type=Path)
    parser.add_argument("--fixed-bucket", type=int)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--bucket-step", type=int, default=16)
    parser.add_argument("--speeds", nargs="+", type=float, default=[1.0])
    parser.add_argument("--text", action="append", help="Override the built-in gate sentence(s)")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--all-voices", action="store_true")
    parser.add_argument("--sentence-limit", type=int)
    args = parser.parse_args()
    if args.bucket_step < 1:
        parser.error("--bucket-step must be positive")
    if (args.fixed_generator is None) != (args.fixed_bucket is None):
        parser.error("--fixed-generator and --fixed-bucket must be supplied together")
    if args.fixed_bucket is not None and args.fixed_bucket < 1:
        parser.error("--fixed-bucket must be positive")

    vocabulary = json.loads((args.assets / "vocab.json").read_text(encoding="utf-8"))["vocab"]
    dictionary = load_dictionary(args.assets / "cmudict_ipa.dict")
    voices = (
        sorted(path.stem for path in (args.assets / "voices").glob("*.npy"))
        if args.full or args.all_voices
        else list(DEFAULT_VOICES)
    )
    sentences = list(args.text or (FULL_SENTENCES if args.full else DEFAULT_SENTENCES))
    if args.sentence_limit is not None:
        if args.sentence_limit < 1:
            parser.error("--sentence-limit must be positive")
        sentences = sentences[: args.sentence_limit]

    front = session(args.front)
    generator = session(args.generator)
    fixed_generator = session(args.fixed_generator) if args.fixed_generator else None
    fixed_input_names = (
        {item.name for item in fixed_generator.get_inputs()} if fixed_generator else set()
    )
    failures: list[str] = []
    observed_deltas: set[int] = set()
    for voice_name in voices:
        voice = np.load(args.assets / "voices" / f"{voice_name}.npy", mmap_mode="r")
        for sentence_index, text in enumerate(sentences, start=1):
            phonemes = phonemize(text, dictionary)
            token_ids = [vocabulary[char] for char in phonemes if char in vocabulary]
            tokens = np.asarray([[0, *token_ids, 0]], dtype=np.int64)
            style = np.asarray(
                voice[min(509, max(0, len(token_ids) - 1))],
                dtype=np.float32,
            ).reshape(1, 256)
            for speed in args.speeds:
                middle = front.run(
                    list(CUT_TENSORS),
                    {"tokens": tokens, "style": style, "speed": np.asarray([speed], dtype=np.float32)},
                )
                unpadded = dict(zip(CUT_TENSORS, middle))
                reference = np.asarray(generator.run(None, unpadded)[0], dtype=np.float32).reshape(-1)
                frames = int(middle[1].shape[-1])
                bucket = (
                    args.fixed_bucket
                    if args.fixed_bucket is not None
                    else ((frames + args.bucket_step - 1) // args.bucket_step) * args.bucket_step
                )
                if frames > bucket:
                    failures.append(
                        f"{voice_name} sentence-{sentence_index} speed={speed} "
                        f"T={frames}: exceeds fixed bucket B={bucket}"
                    )
                    continue
                observed_deltas.add(bucket - frames)
                padded = {
                    CUT_TENSORS[0]: middle[0],
                    CUT_TENSORS[1]: np.pad(middle[1], ((0, 0), (0, 0), (0, bucket - frames))),
                    CUT_TENSORS[2]: np.pad(middle[2], ((0, 0), (0, 0), (0, bucket - frames))),
                }
                candidate_session = fixed_generator or generator
                if fixed_generator is not None:
                    if "valid_frames" in fixed_input_names:
                        padded["valid_frames"] = np.asarray([frames], dtype=np.int64)
                    elif "valid_mask_10" in fixed_input_names:
                        padded["valid_mask_10"] = np.concatenate(
                            [
                                np.ones(frames * 10, dtype=np.float32),
                                np.zeros((bucket - frames) * 10, dtype=np.float32),
                            ]
                        ).reshape(1, 1, bucket * 10)
                        if "valid_length_10" in fixed_input_names:
                            padded["valid_length_10"] = np.asarray(
                                [frames * 10], dtype=np.float32
                            ).reshape(1, 1, 1)
                        padded["valid_mask_60"] = np.concatenate(
                            [
                                np.ones(frames * 60 + 1, dtype=np.float32),
                                np.zeros((bucket - frames) * 60, dtype=np.float32),
                            ]
                        ).reshape(1, 1, bucket * 60 + 1)
                        if "valid_length_60" in fixed_input_names:
                            padded["valid_length_60"] = np.asarray(
                                [frames * 60 + 1], dtype=np.float32
                            ).reshape(1, 1, 1)
                    else:
                        raise RuntimeError(
                            "Fixed generator lacks valid_frames or external mask inputs"
                        )
                candidate = np.asarray(candidate_session.run(None, padded)[0], dtype=np.float32).reshape(-1)
                candidate = candidate[: frames * FRAME_SAMPLES]
                case_failures = validate_case(reference, candidate)
                label = (
                    f"{voice_name} sentence-{sentence_index} speed={speed} "
                    f"T={frames} B={bucket}"
                )
                if case_failures:
                    failures.append(f"{label}: {', '.join(case_failures)}")
                else:
                    print(f"PASS {label}", flush=True)

    if failures:
        print("SPLIT GENERATOR AUDIO GATE FAILED", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"SPLIT GENERATOR AUDIO GATE PASSED ({len(voices)} voices x {len(sentences)} "
        f"utterances x {len(args.speeds)} speeds; "
        f"pad deltas={sorted(observed_deltas)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
