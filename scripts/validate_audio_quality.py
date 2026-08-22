"""Reject mobile Kokoro exports that add audible quantization/noise artifacts.

This is deliberately a fidelity gate, not a perceptual-quality score. It compares a
candidate graph with the official FP32 graph using identical phonemes and styles.
Every candidate must be finite, unclipped, duration-stable, spectrally similar, and
must not add an abnormal amount of high-frequency energy (a useful hiss/static
signal). Run with ``--full`` before shipping a new graph optimization.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 24_000
DEFAULT_VOICES = ("af_heart", "af_bella", "am_adam", "bm_george")
DEFAULT_SENTENCES = (
    "This is an example of speech synthesis in English.",
    "Numbers like 24, 3.14159, and 2026 should sound natural, not noisy.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "A clean offline voice should handle punctuation, pauses, and emphasis!",
)
FULL_SENTENCES = DEFAULT_SENTENCES + (
    "She sells seashells by the seashore; six sleek swans swam swiftly south.",
    "Please read this longer paragraph with a calm, continuous cadence so the waveform has time to reveal any unwanted high-frequency texture or clicking between syllables.",
    "Ready when you are. The battery-friendly reader starts speaking promptly.",
    "Quotes, parentheses (like these), abbreviations such as Dr., and contractions shouldn't create artifacts.",
)


def load_dictionary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line or line.startswith(";;;"):
            continue
        word, phonemes = line.split("\t", 1)
        result.setdefault(word.split("(", 1)[0], phonemes.split(",", 1)[0].strip())
    return result


def phonemize(text: str, dictionary: dict[str, str]) -> str:
    pieces = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    return "".join(
        dictionary.get(piece.upper(), piece.lower()) if any(char.isalpha() for char in piece) else piece
        for piece in pieces
    )


def make_session(model: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def synthesize(
    session: ort.InferenceSession,
    vocabulary: dict[str, int],
    dictionary: dict[str, str],
    voice: np.ndarray,
    text: str,
) -> np.ndarray:
    phonemes = phonemize(text, dictionary)
    token_ids = [vocabulary[char] for char in phonemes if char in vocabulary]
    tokens = np.asarray([[0, *token_ids, 0]], dtype=np.int64)
    if tokens.shape[1] > 512:
        raise ValueError(f"Golden sentence tokenized to {tokens.shape[1]} tokens; Kokoro supports at most 512")
    # Kokoro voice packs are indexed by the zero-based phoneme count. Keep
    # this identical to VoiceStyleStore on Android and Kokoro's export helper.
    style = np.asarray(voice[min(509, max(0, len(token_ids) - 1))], dtype=np.float32)
    audio = session.run(None, {"tokens": tokens, "style": style, "speed": np.asarray([1.0], dtype=np.float32)})[0]
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def spectral_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    length = min(reference.size, candidate.size)
    if length < 1024:
        raise ValueError(f"Only {length} samples; a golden utterance unexpectedly produced too little audio")
    reference = reference[:length]
    candidate = candidate[:length]
    window = np.hanning(1024).astype(np.float32)
    ref_frames = np.lib.stride_tricks.sliding_window_view(reference, 1024)[::256] * window
    candidate_frames = np.lib.stride_tricks.sliding_window_view(candidate, 1024)[::256] * window
    ref_power = np.abs(np.fft.rfft(ref_frames, axis=1)) ** 2
    candidate_power = np.abs(np.fft.rfft(candidate_frames, axis=1)) ** 2
    ref_log = np.log(ref_power + 1e-10)
    candidate_log = np.log(candidate_power + 1e-10)
    log_mae = float(np.mean(np.abs(candidate_log - ref_log)))
    spectral_correlation = float(np.corrcoef(ref_log.ravel(), candidate_log.ravel())[0, 1])
    # 8 kHz begins at bin 342 for a 1024-point FFT at 24 kHz.
    reference_high_band = float(ref_power[:, 342:].sum() / max(ref_power.sum(), 1e-20))
    candidate_high_band = float(candidate_power[:, 342:].sum() / max(candidate_power.sum(), 1e-20))
    return log_mae, spectral_correlation, reference_high_band, candidate_high_band


def validate_case(reference: np.ndarray, candidate: np.ndarray) -> list[str]:
    failures: list[str] = []
    if not np.isfinite(candidate).all():
        failures.append("non-finite samples")
        return failures
    if candidate.size == 0:
        failures.append("empty audio")
        return failures
    if float(np.max(np.abs(candidate))) > 0.98:
        failures.append(f"PCM-clipping risk (peak={float(np.max(np.abs(candidate))):.4f})")
    duration_delta = abs(candidate.size - reference.size) / max(reference.size, 1)
    if duration_delta > 0.03:
        failures.append(f"duration drift={duration_delta:.2%}")
    log_mae, spectral_correlation, reference_high, candidate_high = spectral_metrics(reference, candidate)
    # The 1-D -> 2-D FP32 rewrite has tiny operator-order differences across
    # voices, so allow its measured envelope while still rejecting the previous
    # all-convolution QOperator export (~0.9 on this metric).
    if log_mae > 0.30:
        failures.append(f"spectral log-MAE={log_mae:.3f}")
    if spectral_correlation < 0.985:
        failures.append(f"spectral correlation={spectral_correlation:.4f}")
    allowed_high = reference_high * 1.20 + 0.003
    if candidate_high > allowed_high:
        failures.append(
            f"excess high-band energy={candidate_high:.3%} (reference={reference_high:.3%}, limit={allowed_high:.3%})"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Official unquantized Kokoro FP32 ONNX model")
    parser.add_argument("--candidate", required=True, type=Path, help="Graph proposed for the Android APK")
    parser.add_argument("--assets", required=True, type=Path, help="Directory containing vocab.json, dictionary, and voices")
    parser.add_argument("--full", action="store_true", help="Exercise every included voice and the extended sentence corpus")
    parser.add_argument("--all-voices", action="store_true", help="Exercise every included voice with the selected sentence corpus")
    parser.add_argument("--sentence-limit", type=int, help="Limit the selected corpus for a bounded broad voice sweep")
    args = parser.parse_args()

    vocabulary = json.loads((args.assets / "vocab.json").read_text(encoding="utf-8"))["vocab"]
    dictionary = load_dictionary(args.assets / "cmudict_ipa.dict")
    voice_names = sorted(path.stem for path in (args.assets / "voices").glob("*.npy")) if (args.full or args.all_voices) else DEFAULT_VOICES
    sentences = FULL_SENTENCES if args.full else DEFAULT_SENTENCES
    if args.sentence_limit is not None:
        if args.sentence_limit < 1:
            parser.error("--sentence-limit must be positive")
        sentences = sentences[:args.sentence_limit]
    reference_session = make_session(args.reference)
    candidate_session = make_session(args.candidate)
    failures: list[str] = []
    for voice_name in voice_names:
        voice_path = args.assets / "voices" / f"{voice_name}.npy"
        if not voice_path.exists():
            failures.append(f"missing voice asset {voice_name}")
            continue
        voice = np.load(voice_path, mmap_mode="r")
        for sentence_index, sentence in enumerate(sentences, start=1):
            reference = synthesize(reference_session, vocabulary, dictionary, voice, sentence)
            candidate = synthesize(candidate_session, vocabulary, dictionary, voice, sentence)
            case_failures = validate_case(reference, candidate)
            label = f"{voice_name} sentence-{sentence_index}"
            if case_failures:
                failures.append(f"{label}: {', '.join(case_failures)}")
            else:
                print(f"PASS {label}: {candidate.size / SAMPLE_RATE:.2f}s", flush=True)
    if failures:
        print("AUDIO QUALITY GATE FAILED", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"AUDIO QUALITY GATE PASSED ({len(voice_names)} voices x {len(sentences)} utterances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
