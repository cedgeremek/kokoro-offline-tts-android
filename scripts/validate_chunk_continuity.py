"""Host-side continuity gate for Kokoro's sentence-aware Android chunking.

This exercises the exact CPU front/generator assets shipped beside the QNN
contexts.  It is intentionally independent from Android playback: the gate
checks that the Settings sentence stays whole, then measures model-generated
edge silence after the same sustained-RMS trimming policy used by the app.

The default sweep covers all packaged voices for the Settings sentence and a
long input with artificial continuation seams.  A smaller accent/gender set is
also compared against unsplit punctuation references.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnxruntime as ort

from validate_audio_quality import load_dictionary, phonemize


SAMPLE_RATE = 24_000
FRAME_SAMPLES = 300
MAX_TOKENS = 510
FIRST_TARGET = 30
FOLLOWING_TARGET = 72
MIN_TAIL_TOKENS = 10

ACTIVE_RELATIVE_RMS = 10.0 ** (-45.0 / 20.0)
ACTIVE_WINDOW = SAMPLE_RATE // 100  # 10 ms
ACTIVE_HOP = SAMPLE_RATE // 200  # 5 ms
ACTIVE_SUSTAINED_WINDOWS = 3
MIN_TRIM = SAMPLE_RATE // 200  # 5 ms
MAX_TRIM = SAMPLE_RATE * 4 // 5  # 800 ms

LEADING_KEEP = SAMPLE_RATE * 30 // 1_000
TRAILING_KEEP = {
    "continuation": SAMPLE_RATE * 40 // 1_000,
    "comma": SAMPLE_RATE * 400 // 1_000,
    "semicolon": SAMPLE_RATE * 420 // 1_000,
    "colon": SAMPLE_RATE * 565 // 1_000,
    "period": SAMPLE_RATE * 530 // 1_000,
    "question": SAMPLE_RATE * 545 // 1_000,
}

SETTINGS_TEXT = "This is an example of speech synthesis in English."
LONG_CONTINUATION_TEXT = (
    "The patient reader follows the winding road through a quiet valley while the morning "
    "light crosses every window and the distant river carries a steady rhythm beneath the "
    "trees where travelers continue toward the old stone bridge without stopping because "
    "each careful step brings the village closer"
)
PUNCTUATION_LEFT = "The careful reader follows the road"
PUNCTUATION_RIGHT = "the river keeps a steady rhythm beneath the trees"
PUNCTUATION_MARKS = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "period": ".",
    "question": "?",
}
REPRESENTATIVE_VOICES = (
    "af_heart",
    "af_bella",
    "am_adam",
    "bf_lily",
    "bm_george",
)

CUT_TENSORS = (
    "/decoder/Slice_output_0",
    "/decoder/decoder/Unsqueeze_output_0",
    "/decoder/decoder/decode.3/Div_4_output_0",
)


@dataclass(frozen=True)
class ActiveRange:
    first: int
    last: int  # exclusive


@dataclass(frozen=True)
class TrimmedAudio:
    values: np.ndarray
    start: int
    end: int
    active: ActiveRange | None


@dataclass(frozen=True)
class BoundaryCandidate:
    end: int
    tokens: int
    tier: int


# Keep this grammar synchronized with KokoroSynthesizer.textBoundaryPattern.
TEXT_BOUNDARY_PATTERN = re.compile(
    r"[.!?\u2026]+[\"')}\]\u2019\u201d\u00bb]*(?:\s+|$)|"
    r"[,;:]+[\"')}\]\u2019\u201d\u00bb]*(?:\s+|$)|"
    r"[\u2013\u2014]+\s*|\s+"
)


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def token_ids(phonemes: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary[char] for char in phonemes if char in vocabulary]


def preferred_boundary(
    value: str,
    target: int,
    hard_limit: int,
    minimum_tail: int,
    count_tokens: Callable[[str], int],
) -> int | None:
    candidates: list[BoundaryCandidate] = []
    for match in TEXT_BOUNDARY_PATTERN.finditer(value):
        end = match.end()
        if end >= len(value):
            continue
        head_tokens = count_tokens(value[:end])
        tail_tokens = count_tokens(value[end:])
        if not 1 <= head_tokens <= hard_limit or tail_tokens < minimum_tail:
            continue
        matched = match.group(0)
        if any(char in ".!?\u2026" for char in matched):
            tier = 2
        elif any(char in ",;:\u2013\u2014" for char in matched):
            tier = 1
        else:
            tier = 0
        candidates.append(BoundaryCandidate(end, head_tokens, tier))
    if not candidates:
        return None
    punctuation_floor = max(1, target * 2 // 3)
    for tier in (2, 1):
        tier_candidates = [
            item for item in candidates if item.tier == tier and item.tokens >= punctuation_floor
        ]
        if tier_candidates:
            return min(tier_candidates, key=lambda item: (abs(item.tokens - target), -item.tokens)).end
    for tier in (2, 1):
        tier_candidates = [item for item in candidates if item.tier == tier]
        if tier_candidates:
            return min(tier_candidates, key=lambda item: (abs(item.tokens - target), -item.tokens)).end
    whitespace = [item for item in candidates if item.tier == 0]
    pool = whitespace or candidates
    return min(pool, key=lambda item: (abs(item.tokens - target), -item.tokens)).end


def fallback_boundary(
    value: str,
    target: int,
    minimum_tail: int,
    count_tokens: Callable[[str], int],
) -> int | None:
    low, high, best = 1, len(value) - 1, -1
    while low <= high:
        middle = (low + high) // 2
        count = count_tokens(value[:middle])
        if count <= target:
            if count > 0:
                best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best <= 0:
        return None
    if count_tokens(value[:best]) <= 0 or count_tokens(value[best:]) < minimum_tail:
        return None
    return best


def plan_chunks(
    value: str,
    count_tokens: Callable[[str], int],
    first_target: int = FIRST_TARGET,
    following_target: int = FOLLOWING_TARGET,
) -> list[str]:
    remaining = value.strip()
    if not remaining:
        return []
    semantic_whole_limit = min(
        MAX_TOKENS - 2,
        following_target + max(12, following_target // 2),
    )
    if count_tokens(remaining) <= semantic_whole_limit:
        return [remaining]
    chunks: list[str] = []
    first = True
    while remaining:
        target = first_target if first else following_target
        slack = max(MIN_TAIL_TOKENS, target // 3) if first else max(12, target // 2)
        hard_limit = min(MAX_TOKENS - 2, target + slack)
        if count_tokens(remaining) <= hard_limit:
            chunks.append(remaining)
            break
        semantic_lookahead = semantic_whole_limit if first else hard_limit
        boundary = preferred_boundary(
            remaining,
            target,
            max(hard_limit, semantic_lookahead),
            MIN_TAIL_TOKENS,
            count_tokens,
        )
        if boundary is None:
            boundary = fallback_boundary(remaining, target, MIN_TAIL_TOKENS, count_tokens)
        if boundary is None:
            chunks.append(remaining)
            break
        head, tail = remaining[:boundary].strip(), remaining[boundary:].strip()
        if not head or not tail:
            chunks.append(remaining)
            break
        chunks.append(head)
        remaining = tail
        first = False
    return chunks


def seam_after(value: str) -> str:
    boundary = value.rstrip().rstrip("\"')]}\u2019\u201d\u00bb")[-1:] or ""
    if boundary == ",":
        return "comma"
    if boundary in ";\u2013\u2014":
        return "semicolon"
    if boundary == ":":
        return "colon"
    if boundary in ".!\u2026":
        return "period"
    if boundary == "?":
        return "question"
    return "continuation"


def activity_windows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < ACTIVE_WINDOW:
        return np.empty(0, dtype=np.bool_)
    peak = float(np.max(np.abs(values)))
    if not math.isfinite(peak) or peak <= 0.0:
        return np.empty(0, dtype=np.bool_)
    squared = np.square(values.astype(np.float64))
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(squared)))
    starts = np.arange(0, values.size - ACTIVE_WINDOW + 1, ACTIVE_HOP)
    energies = prefix[starts + ACTIVE_WINDOW] - prefix[starts]
    threshold_squared = (peak * ACTIVE_RELATIVE_RMS) ** 2
    return energies / ACTIVE_WINDOW >= threshold_squared


def sustained_activity(values: np.ndarray) -> ActiveRange | None:
    windows = activity_windows(values)
    if windows.size == 0:
        return None
    first_sustained = -1
    last_sustained = -1
    run_start = 0
    run_length = 0
    for index, active in enumerate(windows):
        if active:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= ACTIVE_SUSTAINED_WINDOWS:
                if first_sustained < 0:
                    first_sustained = run_start
                last_sustained = index
        else:
            run_length = 0
    if first_sustained < 0:
        return None
    return ActiveRange(
        first_sustained * ACTIVE_HOP,
        min(values.size, last_sustained * ACTIVE_HOP + ACTIVE_WINDOW),
    )


def trim_edges(
    values: np.ndarray,
    keep_leading: int | None,
    keep_trailing: int | None,
) -> TrimmedAudio:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    active = sustained_activity(values)
    if active is None:
        return TrimmedAudio(values, 0, values.size, None)

    def removable(quiet: int, keep: int | None) -> int:
        if keep is None:
            return 0
        excess = quiet - keep
        return min(excess, MAX_TRIM) if excess >= MIN_TRIM else 0

    remove_leading = removable(active.first, keep_leading)
    remove_trailing = removable(values.size - active.last, keep_trailing)
    remove_trailing = min(remove_trailing, max(0, values.size - remove_leading - 1))
    end = values.size - remove_trailing
    return TrimmedAudio(values[remove_leading:end], remove_leading, end, active)


def synthesize(
    front: ort.InferenceSession,
    generator: ort.InferenceSession,
    vocabulary: dict[str, int],
    voice: np.ndarray,
    phoneme_text: str,
) -> np.ndarray:
    ids = token_ids(phoneme_text, vocabulary)
    if not ids or len(ids) > MAX_TOKENS - 2:
        raise ValueError(f"Invalid model input token count: {len(ids)}")
    tokens = np.asarray([[0, *ids, 0]], dtype=np.int64)
    style = np.asarray(voice[min(509, max(0, len(ids) - 1))], dtype=np.float32).reshape(1, 256)
    middle = front.run(
        list(CUT_TENSORS),
        {"tokens": tokens, "style": style, "speed": np.asarray([1.0], dtype=np.float32)},
    )
    frames = int(middle[1].shape[-1])
    feeds = dict(zip(CUT_TENSORS, middle))
    feeds["valid_mask_10"] = np.ones((1, 1, frames * 10), dtype=np.float32)
    feeds["valid_mask_60"] = np.ones((1, 1, frames * 60 + 1), dtype=np.float32)
    audio = np.asarray(generator.run(None, feeds)[0], dtype=np.float32).reshape(-1)
    expected = frames * FRAME_SAMPLES
    if audio.size != expected:
        raise ValueError(f"Generator returned {audio.size} samples for T={frames}; expected {expected}")
    return audio


def final_pcm_float(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if not math.isfinite(peak) or peak <= 0.0:
        return values
    gain = min(1.0, 0.95 / peak)
    return values * gain


def signal_failures(values: np.ndarray, label: str) -> list[str]:
    failures: list[str] = []
    if values.size == 0:
        return [f"{label}: empty audio"]
    if not np.isfinite(values).all():
        return [f"{label}: non-finite audio"]
    dynamic_range = float(np.max(values) - np.min(values))
    if dynamic_range < 1e-5:
        failures.append(f"{label}: static audio (range={dynamic_range:.3g})")
    final = final_pcm_float(values)
    peak = float(np.max(np.abs(final)))
    if peak > 0.950001:
        failures.append(f"{label}: post-headroom clipping (peak={peak:.6f})")
    return failures


def removed_energy_fraction(original: np.ndarray, trimmed: TrimmedAudio) -> float:
    total = float(np.sum(np.square(original.astype(np.float64))))
    if total <= 0.0:
        return 0.0
    removed = float(np.sum(np.square(original[: trimmed.start].astype(np.float64))))
    removed += float(np.sum(np.square(original[trimmed.end :].astype(np.float64))))
    return removed / total


def seam_gap_ms(left: np.ndarray, right: np.ndarray) -> float:
    left_active = sustained_activity(left)
    right_active = sustained_activity(right)
    if left_active is None or right_active is None:
        return math.inf
    quiet_samples = left.size - left_active.last + right_active.first
    return quiet_samples * 1_000.0 / SAMPLE_RATE


def longest_interior_quiet_ms(values: np.ndarray) -> float:
    windows = activity_windows(values)
    if windows.size == 0:
        return math.inf
    sustained = np.convolve(
        windows.astype(np.int8),
        np.ones(ACTIVE_SUSTAINED_WINDOWS, dtype=np.int8),
        mode="same",
    ) > 0
    active_indices = np.flatnonzero(sustained)
    if active_indices.size < 2:
        return math.inf
    start, end = int(active_indices[0]), int(active_indices[-1])
    longest = 0
    current = 0
    for active in sustained[start : end + 1]:
        if active:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    longest = max(longest, current)
    return longest * ACTIVE_HOP * 1_000.0 / SAMPLE_RATE


def percentile(values: Iterable[float], quantile: float) -> float:
    materialized = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(materialized, quantile))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--front",
        type=Path,
        default=Path("app/src/main/assets/kokoro-front.fp32.onnx"),
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("app/src/main/assets/kokoro-generator.masked-dynamic.fp32.onnx"),
    )
    parser.add_argument("--assets", type=Path, default=Path("app/src/main/assets"))
    parser.add_argument(
        "--punctuation-voices",
        nargs="+",
        default=list(REPRESENTATIVE_VOICES),
    )
    args = parser.parse_args()

    vocabulary = json.loads((args.assets / "vocab.json").read_text(encoding="utf-8"))["vocab"]
    dictionary = load_dictionary(args.assets / "cmudict_ipa.dict")
    voice_names = sorted(path.stem for path in (args.assets / "voices").glob("*.npy"))
    punctuation_voices = (
        voice_names if args.punctuation_voices == ["all"] else args.punctuation_voices
    )
    failures: list[str] = []

    def to_phonemes(text: str) -> str:
        return phonemize(text, dictionary)

    def count(value: str) -> int:
        return len(token_ids(value, vocabulary))

    settings_phonemes = to_phonemes(SETTINGS_TEXT)
    settings_chunks = plan_chunks(settings_phonemes, count)
    print(
        f"PLAN settings tokens={count(settings_phonemes)} chunks={len(settings_chunks)} "
        f"sizes={[count(item) for item in settings_chunks]}",
        flush=True,
    )
    if len(settings_chunks) != 1:
        failures.append(f"Settings sentence planned as {len(settings_chunks)} chunks, expected 1")

    continuation_phonemes = to_phonemes(LONG_CONTINUATION_TEXT)
    continuation_chunks = plan_chunks(continuation_phonemes, count)
    print(
        f"PLAN continuation tokens={count(continuation_phonemes)} chunks={len(continuation_chunks)} "
        f"sizes={[count(item) for item in continuation_chunks]}",
        flush=True,
    )
    if len(continuation_chunks) < 2:
        failures.append("Long continuation case did not produce an artificial seam")
    if any(count(item) > MAX_TOKENS - 2 for item in continuation_chunks):
        failures.append("Long continuation plan exceeds the model token limit")

    front = make_session(args.front)
    generator = make_session(args.generator)
    continuation_gaps: list[float] = []
    trim_energy_losses: list[float] = []

    for voice_name in voice_names:
        voice = np.load(args.assets / "voices" / f"{voice_name}.npy", mmap_mode="r")
        settings_audio = synthesize(
            front, generator, vocabulary, voice, settings_chunks[0]
        )
        failures.extend(signal_failures(settings_audio, f"{voice_name} settings"))

        generated = [
            synthesize(front, generator, vocabulary, voice, chunk)
            for chunk in continuation_chunks
        ]
        adjusted: list[np.ndarray] = []
        for index, audio in enumerate(generated):
            leading = LEADING_KEEP if index > 0 else None
            trailing = TRAILING_KEEP["continuation"] if index < len(generated) - 1 else None
            trimmed = trim_edges(audio, leading, trailing)
            trim_energy_losses.append(removed_energy_fraction(audio, trimmed))
            if trimmed.active is not None and not (
                trimmed.start <= trimmed.active.first and trimmed.end >= trimmed.active.last
            ):
                failures.append(f"{voice_name} continuation chunk-{index + 1}: activity clipped")
            failures.extend(
                signal_failures(trimmed.values, f"{voice_name} continuation chunk-{index + 1}")
            )
            adjusted.append(trimmed.values)
        for index in range(len(adjusted) - 1):
            gap = seam_gap_ms(adjusted[index], adjusted[index + 1])
            continuation_gaps.append(gap)
            print(
                f"PASS {voice_name} continuation-seam-{index + 1}: gap={gap:.1f}ms",
                flush=True,
            )

    if continuation_gaps:
        gap_p90 = percentile(continuation_gaps, 90)
        gap_max = max(continuation_gaps)
        print(
            "CONTINUATION SUMMARY "
            f"n={len(continuation_gaps)} min={min(continuation_gaps):.1f}ms "
            f"p50={percentile(continuation_gaps, 50):.1f}ms "
            f"p90={gap_p90:.1f}ms max={gap_max:.1f}ms",
            flush=True,
        )
        if gap_p90 > 150.0:
            failures.append(f"Artificial seam p90 {gap_p90:.1f}ms exceeds 150ms")
        if gap_max > 200.0:
            failures.append(f"Artificial seam max {gap_max:.1f}ms exceeds 200ms")
    if trim_energy_losses:
        worst_loss = max(trim_energy_losses)
        print(
            f"TRIM ENERGY max_removed={worst_loss:.6%} "
            f"p90={percentile(trim_energy_losses, 90):.6%}",
            flush=True,
        )
        if worst_loss > 0.001:
            failures.append(f"Edge trim removed {worst_loss:.4%} of waveform energy (limit 0.1%)")

    punctuation_deltas: list[float] = []
    for voice_name in punctuation_voices:
        if voice_name not in voice_names:
            failures.append(f"Unknown punctuation voice {voice_name}")
            continue
        voice = np.load(args.assets / "voices" / f"{voice_name}.npy", mmap_mode="r")
        for seam, mark in PUNCTUATION_MARKS.items():
            text = f"{PUNCTUATION_LEFT}{mark} {PUNCTUATION_RIGHT}"
            phonemes = to_phonemes(text)
            chunks = plan_chunks(phonemes, count)
            if len(chunks) != 1:
                failures.append(
                    f"{voice_name} {seam}: moderate punctuation case planned as "
                    f"{len(chunks)} chunks, expected native whole synthesis"
                )
                continue
            reference = synthesize(front, generator, vocabulary, voice, phonemes)
            reference_pause = longest_interior_quiet_ms(reference)
            native_pause = longest_interior_quiet_ms(reference)
            delta = abs(native_pause - reference_pause)
            punctuation_deltas.append(delta)
            failures.extend(signal_failures(reference, f"{voice_name} {seam} native-whole"))
            print(
                f"PASS {voice_name} {seam}: native-whole={native_pause:.1f}ms "
                f"reference={reference_pause:.1f}ms delta={delta:.1f}ms",
                flush=True,
            )
            if delta > 175.0:
                failures.append(
                    f"{voice_name} {seam}: pause delta {delta:.1f}ms exceeds 175ms"
                )

    if punctuation_deltas:
        print(
            "PUNCTUATION SUMMARY "
            f"n={len(punctuation_deltas)} p50={percentile(punctuation_deltas, 50):.1f}ms "
            f"p90={percentile(punctuation_deltas, 90):.1f}ms "
            f"max={max(punctuation_deltas):.1f}ms",
            flush=True,
        )

    if failures:
        print("CHUNK CONTINUITY GATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print(
        f"CHUNK CONTINUITY GATE PASSED ({len(voice_names)} Settings voices, "
        f"{len(continuation_gaps)} artificial seams, "
        f"{len(punctuation_deltas)} punctuation comparisons)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
