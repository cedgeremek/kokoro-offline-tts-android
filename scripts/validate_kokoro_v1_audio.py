#!/usr/bin/env python3
"""Generate host-side Kokoro v1.0 FP32/q8 audio and numerical checks.

The phonemes come from the pinned official Misaki checkout with its eSpeak
fallback.  This script is a host validation tool only; it does not stand in
for installation, JNI, audio, or performance validation on the target phone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


SAMPLE_RATE = 24_000
CASES = (
    {
        "id": "us_numbers_oov",
        "accent": "us",
        "voice": "af_heart",
        "text": "Dr. Smith will record a $12.50 API demo for Quorvax on August 21st.",
    },
    {
        "id": "gb_context_oov",
        "accent": "gb",
        "voice": "bf_emma",
        "text": "The project lead read the latest data near the Quorvax theatre.",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_misaki(repo: Path):
    """Import Misaki without installing unused transformer/torch dependencies."""
    transformers = types.ModuleType("transformers")
    transformers.BartForConditionalGeneration = object
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.tensor = np.asarray
    torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: self,
        __exit__=lambda self, *args: None,
    )
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("torch", torch)
    sys.path.insert(0, str(repo))
    from misaki import en, espeak  # pylint: disable=import-outside-toplevel

    return en, espeak


def metrics(audio: np.ndarray) -> dict[str, float | int | bool]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    dynamic_range = float(np.max(audio) - np.min(audio)) if audio.size else 0.0
    return {
        "samples": int(audio.size),
        "seconds": float(audio.size / SAMPLE_RATE),
        "finite": finite,
        "peak": peak,
        "rms": rms,
        "dynamic_range": dynamic_range,
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    sample_count_delta = int(candidate.size - reference.size)
    common = min(reference.size, candidate.size)
    if common == 0:
        raise RuntimeError("cannot compare empty audio")
    reference = reference[:common]
    candidate = candidate[:common]
    error = candidate - reference
    signal_power = float(np.mean(reference * reference))
    error_power = float(np.mean(error * error))
    correlation = float(np.corrcoef(reference, candidate)[0, 1])
    snr_db = float(10.0 * math.log10(signal_power / max(error_power, 1e-30)))
    return {
        "common_samples": int(common),
        "sample_count_delta": sample_count_delta,
        "correlation": correlation,
        "snr_db": snr_db,
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "build" / "host-validation-v1").resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "fp32": root / ".build-temp" / "kokoro-v1" / "model.onnx",
        "q8": root / "app" / "src" / "main" / "assets" / "kokoro-v1.0-q8.onnx",
    }
    tokenizer_path = root / "app" / "src" / "main" / "assets" / "kokoro-v1.0-tokenizer.json"
    misaki_repo = root / ".build-temp" / "misaki"
    for path in (*model_paths.values(), tokenizer_path, misaki_repo):
        if not path.exists():
            raise FileNotFoundError(path)

    en, espeak = load_official_misaki(misaki_repo)
    vocab = json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    g2p = {
        accent: en.G2P(
            trf=False,
            british=accent == "gb",
            fallback=espeak.EspeakFallback(accent == "gb"),
        )
        for accent in ("us", "gb")
    }

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {}
    for name, path in model_paths.items():
        print(f"loading {name}: {path}", flush=True)
        sessions[name] = ort.InferenceSession(
            str(path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )

    report: dict[str, object] = {
        "schema": "kokoro-v1.0-host-audio-v1",
        "sample_rate": SAMPLE_RATE,
        "onnxruntime_version": ort.__version__,
        "misaki_version": getattr(sys.modules.get("misaki"), "__version__", "0.9.4"),
        "models": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in model_paths.items()
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
            "sha256": sha256(tokenizer_path),
        },
        "cases": [],
    }

    for case in CASES:
        print(f"phonemizing {case['id']} ({case['accent']})", flush=True)
        phonemes, _tokens = g2p[case["accent"]](case["text"])
        token_ids = [vocab[ch] for ch in phonemes if ch in vocab]
        if not token_ids or len(token_ids) > 510:
            raise RuntimeError(f"invalid token count for {case['id']}: {len(token_ids)}")
        voice_path = (
            root / "app" / "src" / "main" / "assets" / "voices_v1" / f"{case['voice']}.bin"
        )
        voice = np.fromfile(voice_path, dtype="<f4")
        if voice.size != 510 * 256:
            raise RuntimeError(f"unexpected voice shape: {voice_path}: {voice.size}")
        style = voice.reshape(510, 1, 256)[len(token_ids)]
        inputs = {
            "input_ids": np.asarray([[0, *token_ids, 0]], dtype=np.int64),
            "style": style,
            "speed": np.ones(1, dtype=np.float32),
        }
        case_report: dict[str, object] = {
            **case,
            "phonemes": phonemes,
            "phoneme_codepoints": len(phonemes),
            "model_tokens": len(token_ids),
            "voice_sha256": sha256(voice_path),
            "outputs": {},
        }
        generated = {}
        for model_name, session in sessions.items():
            print(f"running {case['id']} with {model_name}", flush=True)
            started = time.perf_counter()
            audio = np.asarray(session.run(None, inputs)[0], dtype=np.float32).reshape(-1)
            elapsed = time.perf_counter() - started
            audio_metrics = metrics(audio)
            if (
                not audio_metrics["finite"]
                or audio_metrics["samples"] <= 0
                or audio_metrics["rms"] <= 1e-5
                or audio_metrics["dynamic_range"] <= 1e-4
            ):
                raise RuntimeError(f"invalid {model_name} output for {case['id']}: {audio_metrics}")
            wav_path = output / f"{case['id']}-{case['voice']}-{model_name}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
            audio_metrics.update(
                {
                    "inference_seconds": elapsed,
                    "host_rtf": elapsed / float(audio_metrics["seconds"]),
                    "wav": str(wav_path),
                    "wav_bytes": wav_path.stat().st_size,
                    "wav_sha256": sha256(wav_path),
                }
            )
            case_report["outputs"][model_name] = audio_metrics
            generated[model_name] = audio
        case_report["q8_vs_fp32"] = compare(generated["fp32"], generated["q8"])
        report["cases"].append(case_report)

    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
