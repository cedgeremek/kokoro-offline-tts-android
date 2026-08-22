#!/usr/bin/env python3
"""Host audio gate for a static masked Kokoro v1 acoustic bucket.

The Android HTP EPContext cannot run on this x64 host.  This validator instead
executes the exact FP32 masked source graph with the same left-padding, valid
masks, crop and CPU iSTFT used by the Android runtime.  It rejects a bucket
unless static synthesis remains close to the dynamic FP32 acoustic reference.
It is a source-fidelity gate, not a substitute for the physical S24 gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


CUT = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)
SAMPLE_RATE = 24_000
FRAME_SAMPLES = 300
DEFAULT_TEXTS = (
    "Ready when you are.",
    "This is an example of speech synthesis in English.",
    "The morning train moved slowly through the valley as sunlight reached the hills.",
)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def load_g2p(root: Path, british: bool):
    # Reuse the same lightweight official-Misaki loader used by the v1 host
    # render receipt, without importing its audio-writing gate.
    sys.path.insert(0, str(root / "scripts"))
    from validate_kokoro_v1_audio import load_official_misaki  # pylint: disable=import-outside-toplevel

    en, espeak = load_official_misaki(root / ".build-temp" / "misaki")
    return en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british))


def style_for(path: Path, token_count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != 510 * 256:
        raise RuntimeError(f"Unexpected v1 style shape in {path}")
    return values.reshape(510, 1, 256)[min(509, max(0, token_count))]


def static_inputs(
    middle: list[np.ndarray],
    frames: int,
    bucket: int,
    padding: str = "left",
) -> dict[str, np.ndarray]:
    pad = bucket - frames
    if pad < 0:
        raise ValueError(f"T={frames} exceeds B={bucket}")
    if padding not in {"left", "right"}:
        raise ValueError(f"Unsupported padding direction: {padding}")
    before, after = (pad, 0) if padding == "left" else (0, pad)
    return {
        CUT[0]: np.asarray(middle[0], dtype=np.float32),
        CUT[1]: np.pad(np.asarray(middle[1], dtype=np.float32), ((0, 0), (0, 0), (before, after))),
        CUT[2]: np.pad(np.asarray(middle[2], dtype=np.float32), ((0, 0), (0, 0), (before, after))),
        "valid_mask_10": np.concatenate((
            np.zeros(before * 10, np.float32),
            np.ones(frames * 10, np.float32),
            np.zeros(after * 10, np.float32),
        )).reshape(1, 1, -1),
        "valid_mask_60": np.concatenate((
            np.zeros(before * 60, np.float32),
            np.ones(frames * 60 + 1, np.float32),
            np.zeros(after * 60, np.float32),
        )).reshape(1, 1, -1),
        "valid_length_10": np.asarray((frames * 10,), dtype=np.float32),
        "valid_length_60": np.asarray((frames * 60 + 1,), dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bucket", required=True, type=int)
    parser.add_argument("--minimum-frames", required=True, type=int)
    parser.add_argument("--masked", required=True, type=Path)
    parser.add_argument(
        "--full-waveform",
        action="store_true",
        help="Validate a joined masked acoustic+iSTFT source graph instead of the acoustic cut",
    )
    parser.add_argument("--padding", choices=("left", "right"), default="left")
    parser.add_argument("--speeds", nargs="+", type=float, default=(0.8, 1.0, 1.2))
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help="Restrict the gate to one or more voice IDs; default is every packaged voice.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.minimum_frames <= args.bucket:
        parser.error("minimum frame count must be inside the bucket")

    root = args.root.resolve()
    assets = root / "app" / "src" / "main" / "assets"
    vocabulary = json.loads((assets / "kokoro-v1.0-tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    front = session(root / "build" / "qnn" / "v1_fp32" / "front-fp32.onnx")
    reference = session(root / "build" / "qnn" / "v1_fp32" / "generator-fp32.onnx")
    masked = session(args.masked.resolve())
    istft = None if args.full_waveform else session(root / "build" / "qnn" / "v1_fp32" / "istft-fp32.onnx")
    expected_inputs = {name: list(shape) for name, shape in (
        (CUT[0], [1, 128]), (CUT[1], [1, 1, args.bucket]), (CUT[2], [1, 512, args.bucket]),
        ("valid_mask_10", [1, 1, args.bucket * 10]),
        ("valid_mask_60", [1, 1, args.bucket * 60 + 1]),
        ("valid_length_10", [1]), ("valid_length_60", [1]),
    )}
    actual_inputs = {item.name: list(item.shape) for item in masked.get_inputs()}
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"Masked model input contract differs: {actual_inputs}")
    output = masked.get_outputs()
    expected_output = [1, args.bucket * FRAME_SAMPLES] if args.full_waveform else [1, 22, args.bucket * 60 + 1]
    if len(output) != 1 or list(output[0].shape) != expected_output:
        raise RuntimeError(f"Masked model output contract differs: {[(item.name, item.shape) for item in output]}")

    texts = tuple(args.text) or DEFAULT_TEXTS
    voices = sorted((assets / "voices_v1").glob("*.bin"))
    if args.voice:
        requested_voices = set(args.voice)
        voices = [path for path in voices if path.stem in requested_voices]
        missing_voices = sorted(requested_voices - {path.stem for path in voices})
        if missing_voices:
            raise RuntimeError(f"Unknown voice IDs: {missing_voices}")
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    skipped = 0
    for voice_path in voices:
        british = voice_path.stem.startswith(("bf_", "bm_"))
        g2p = load_g2p(root, british)
        for text in texts:
            phonemes, _ = g2p(text)
            ids = [vocabulary[ch] for ch in phonemes if ch in vocabulary]
            if not ids:
                raise RuntimeError(f"No model tokens for {text!r}")
            for speed in args.speeds:
                model_inputs = {
                    "input_ids": np.asarray([[0, *ids, 0]], dtype=np.int64),
                    "style": style_for(voice_path, len(ids)),
                    "speed": np.asarray((speed,), dtype=np.float32),
                }
                middle = front.run(None, model_inputs)
                frames = int(middle[1].shape[-1])
                label = f"{voice_path.stem}|{speed:g}|T{frames}|{text}"
                if not args.minimum_frames <= frames <= args.bucket:
                    skipped += 1
                    continue
                expected = reference.run(None, dict(zip(CUT, middle)))[0].reshape(-1)
                pad = args.bucket - frames
                generated = masked.run(None, static_inputs(middle, frames, args.bucket, args.padding))[0]
                if args.full_waveform:
                    start = pad * FRAME_SAMPLES if args.padding == "left" else 0
                    candidate = generated.reshape(-1)[start : start + frames * FRAME_SAMPLES]
                else:
                    start = pad * 60 if args.padding == "left" else 0
                    acoustic = generated[:, :, start : start + frames * 60 + 1]
                    candidate = istft.run(None, {"/decoder/decoder/generator/conv_post/Conv_output_0": acoustic})[0].reshape(-1)[: frames * FRAME_SAMPLES]
                expected = expected[: candidate.size]
                finite = bool(np.isfinite(candidate).all())
                corr = float(np.corrcoef(expected, candidate)[0, 1])
                mae = float(np.mean(np.abs(expected - candidate)))
                dynamic = float(np.max(candidate) - np.min(candidate))
                passed = finite and candidate.size == frames * FRAME_SAMPLES and dynamic > 1e-4 and corr >= 0.985 and mae <= 0.01
                item = {"label": label, "frames": frames, "pad": pad, "correlation": corr, "mae": mae, "finite": finite, "dynamic_range": dynamic, "passed": passed}
                cases.append(item)
                print(json.dumps(item), flush=True)
                if not passed:
                    failures.append(label)
    report = {"bucket": args.bucket, "minimum_frames": args.minimum_frames, "full_waveform": args.full_waveform, "padding": args.padding, "cases": cases, "skipped": skipped, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"tested={len(cases)} skipped={skipped} failures={len(failures)} report={args.output}", flush=True)
    return 1 if failures or not cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
