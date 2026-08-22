#!/usr/bin/env python3
"""Build resumable, HTP-safe Kokoro v1 vocoder contexts for every bucket.

The stock v1 graph is not safe on QAIRT 2.48.40/HTP v75: its voiced source
gate and exponent-2 Pow kernels have both produced invalid tensors on a real
SM8650.  Production therefore computes the small source-spectrum prefix on
CPU, rewrites every exact square as Mul(x, x), and delegates the masked neural
vocoder suffix completely to QNN.  The existing CPU iSTFT suffix is retained.

Every intermediate is written below build/qnn and every completed context has
the compiler's JSON receipt.  Re-running this script resumes at the first
missing artifact; it never overwrites an existing file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BUCKETS = (64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640)


def run(command: list[str], root: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def complete_context(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1_000_000 and path.with_suffix(
        path.suffix + ".json"
    ).is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "build" / "qnn" / "v1_repaired_powmul").resolve()
    output.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.buckets) - set(BUCKETS))
    if unknown:
        parser.error(f"unsupported buckets: {unknown}")

    host_python = Path(sys.executable).resolve()
    qnn_python = root / "build" / "qnn" / "qnn-tooling" / ".venv" / "Scripts" / "python.exe"
    source = root / "build" / "qnn" / "v1_fp32" / "acoustic-fp32.onnx"
    for required in (qnn_python, source):
        if not required.is_file():
            raise FileNotFoundError(required)

    for ordinal, bucket in enumerate(args.buckets, 1):
        print(f"[{ordinal}/{len(args.buckets)}] B{bucket}", flush=True)
        masked = output / f"kokoro-v1-acoustic-b{bucket}.masked.fp32.onnx"
        repaired = output / f"kokoro-v1-acoustic-b{bucket}.masked.powmul.fp32.onnx"
        cpu_source = output / f"kokoro-v1-source-spectrum-b{bucket}.fp32.onnx"
        qnn_source = output / f"kokoro-v1-neural-vocoder-b{bucket}.masked.powmul.fp32.onnx"
        context = output / f"kokoro-v1-neural-vocoder-b{bucket}.qnn248.powmul.ctx.onnx"

        if not masked.exists():
            run(
                [
                    str(host_python),
                    "scripts/prepare_kokoro_v1_masked_acoustic.py",
                    "--source", str(source),
                    "--output", str(masked),
                    "--bucket", str(bucket),
                    "--fp32-internal",
                ],
                root,
            )
        if not repaired.exists():
            run(
                [
                    str(host_python),
                    "scripts/repair_kokoro_v1_qnn_graph.py",
                    "--source", str(masked),
                    "--output", str(repaired),
                    "--square-pow-to-mul",
                ],
                root,
            )
        if not cpu_source.exists() or not qnn_source.exists():
            if cpu_source.exists() or qnn_source.exists():
                raise RuntimeError(f"partial B{bucket} split exists; refusing to overwrite it")
            run(
                [
                    str(host_python),
                    "scripts/split_kokoro_v1_harmonic_source.py",
                    "--source", str(repaired),
                    "--cpu-source-output", str(cpu_source),
                    "--qnn-suffix-output", str(qnn_source),
                    "--cut-output", "/decoder/decoder/generator/Concat_3_output_0",
                    "--qnn-input-name", "kokoro_source_spectrum",
                ],
                root,
            )
        if not complete_context(context):
            if context.exists() or context.with_suffix(context.suffix + ".json").exists():
                raise RuntimeError(f"incomplete B{bucket} context exists; inspect before retry: {context}")
            command = [
                str(qnn_python),
                "scripts/compile_qnn_context.py",
                "--model", str(qnn_source),
                "--output", str(context),
                "--bucket", str(bucket),
                "--diagnostic-probe",
                "--minimum-context-bytes", "1",
            ]
            if bucket in (192, 512, 640):
                command.extend(("--vtcm-mb", "8"))
            if bucket in (512, 640):
                command.extend(("--finalizer-mode", "0"))
            run(command, root)
        print(
            f"B{bucket} ready: context={context.stat().st_size} bytes "
            f"cpu_source={cpu_source.stat().st_size} bytes",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
