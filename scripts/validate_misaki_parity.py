#!/usr/bin/env python3
"""Compare the native Android G2P source against pinned official Misaki."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import espeakng_loader
import numpy as np


CORPUS = (
    "A cat and a dog waited by the gate.",
    "I read the book yesterday.",
    "I will read the book tomorrow.",
    "The wind will wind around the old tower.",
    "They record a new record every week.",
    "The project lead will lead the review.",
    "Please close the close-fitting case.",
    "The dove dove quietly into the lake.",
    "I object to the object on the table.",
    "By and by, the children walked by the river.",
    "The quick brown fox jumps over the lazy dog.",
    "Dr. Smith met Mr. Jones on St. James Street.",
    "Prof. Adams arrived at 3:45 p.m.",
    "The U.S. team won the match.",
    "G2P, TTS, API, and ONNX are technical acronyms.",
    "Kokoro-82M runs an offline demo.",
    "I can't believe she'd already left.",
    "We'll finish when they're ready, won't we?",
    "It's John's book; I'd return it.",
    "You shouldn't've done that.",
    "$1 costs less than $12.50.",
    "The total was £3.25 plus €4.10.",
    "There were 1,234 entries and 56 winners.",
    "She finished 1st, he finished 2nd, and I finished 21st.",
    "The date is August 21st, 2026.",
    "Temperatures fell from 12.5 to -3 degrees.",
    "Call 555-0123 after 9:30.",
    "One half is 0.5, and one quarter is 0.25.",
    "He bought 2 apples, 3 pears, and 10 plums.",
    "Room 101 is on the 4th floor.",
    "Wait... did she say \"hello\"?",
    "Yes—absolutely! No: not yet.",
    "(This parenthetical phrase) should remain audible.",
    "First, pause; second, continue: finally, stop.",
    "‘Single quotes’ and “curly quotes” are punctuation.",
    "well-known state-of-the-art methods",
    "camelCase and snake_case identifiers",
    "PDFReader calls textToSpeechService.",
    "preprocessing, recorded, and happiness",
    "cats, boxes, wishes, and churches",
    "running, tried, studies, and faster",
    "Café patrons discussed naïve résumés.",
    "The coöperative façade was restored.",
    "Quorvax calibrates the blenforth engine.",
    "Zyphrax and mernovian are invented words.",
    "quorvaxium blenforthic zyzzyphonic",
    "She presents the present to her friend.",
    "Does he use the old permit to permit entry?",
    "The bass swam beneath the bass clef poster.",
    "After reading, the reader records the results.",
    "useEffect",
    "getInputValue",
    "HTMLIFrameElement",
    "HTMLElement reads URLSearchParams.",
    "cat-dog and cat_dog",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_misaki(repo: Path):
    transformers = types.ModuleType("transformers")
    transformers.BartForConditionalGeneration = object
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.tensor = np.asarray
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("torch", torch)
    sys.path.insert(0, str(repo))
    from misaki import en, espeak  # pylint: disable=import-outside-toplevel

    return en, espeak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "build" / "misaki-parity-v1" / "report.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    misaki_repo = root / ".build-temp" / "misaki"
    native_exe = root / "native" / "misaki-android" / "target" / "release" / "misaki-parity.exe"
    if not misaki_repo.is_dir():
        raise FileNotFoundError(misaki_repo)
    if not native_exe.is_file():
        raise FileNotFoundError(native_exe)

    en, espeak = load_official_misaki(misaki_repo)
    library = Path(espeakng_loader.get_library_path()).resolve()
    data = Path(espeakng_loader.get_data_path()).resolve()
    report: dict[str, object] = {
        "schema": "misaki-native-parity-v1",
        "misaki_version": "0.9.4",
        "misaki_revision": "fba1236595f2d2bf21d414ba6e57d25256afada3",
        "native_executable": {
            "path": str(native_exe),
            "bytes": native_exe.stat().st_size,
            "sha256": sha256(native_exe),
        },
        "espeak_library": str(library),
        "espeak_data": str(data),
        "corpus_size_per_accent": len(CORPUS),
        "accents": {},
    }
    total_matches = 0
    for accent in ("us", "gb"):
        print(f"official Misaki {accent}: {len(CORPUS)} cases", flush=True)
        official_g2p = en.G2P(
            trf=False,
            british=accent == "gb",
            fallback=espeak.EspeakFallback(accent == "gb"),
        )
        official = [official_g2p(text)[0] for text in CORPUS]
        print(f"native frontend {accent}: {len(CORPUS)} cases", flush=True)
        completed = subprocess.run(
            [str(native_exe), str(library), str(data), accent],
            input="\n".join(CORPUS) + "\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"native parity executable failed ({completed.returncode}): {completed.stderr}"
            )
        native = completed.stdout.splitlines()
        if len(native) != len(CORPUS):
            raise RuntimeError(f"native {accent} line count: {len(native)} != {len(CORPUS)}")
        cases = []
        matches = 0
        for text, expected, actual in zip(CORPUS, official, native, strict=True):
            match = expected == actual
            matches += int(match)
            if not match:
                cases.append({"text": text, "official": expected, "native": actual})
        total_matches += matches
        report["accents"][accent] = {
            "matches": matches,
            "total": len(CORPUS),
            "mismatches": cases,
        }

    report["matches"] = total_matches
    report["total"] = len(CORPUS) * 2
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"parity: {total_matches}/{len(CORPUS) * 2}; wrote {output}", flush=True)
    return 0 if total_matches == len(CORPUS) * 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
