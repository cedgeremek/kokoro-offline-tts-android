# Building

## Requirements

- Windows or Linux with JDK 17 and Android SDK/NDK.
- Android compile SDK 36 and build tools 35 or newer.
- Python environment for ONNX graph preparation/validation.
- Rust 1.85+ for rebuilding the native Misaki frontend.
- Qualcomm QAIRT/QNN 2.48 tooling for SM8650 AOT contexts.

## Assets

Large generated/downloaded assets are excluded from Git. The Gradle build
expects these under `app/src/main/assets`:

- `kokoro-v1-front.fp16io.onnx`
- `kokoro-generator.masked-dynamic.fp32.onnx`
- `kokoro-v1-istft.fp32.onnx`
- `kokoro-v1.0-q8.onnx`
- `kokoro-v1.0-tokenizer.json`
- B64-B640 `kokoro-v1-neural-vocoder-*.qnn248.powmul.ctx.onnx`
- matching B64-B640 `kokoro-v1-source-spectrum-*.fp32.onnx`
- `voices_v1/` and `espeak-ng-data/`

Gradle pins byte counts and SHA-256 values. A partial or altered context set is
rejected instead of silently producing a mixed build.

The graph scripts in `scripts/` document the transformation chain. The main
entry point for the production contexts is:

```powershell
py scripts/build_kokoro_v1_repaired_qnn.py
```

This expects QAIRT tooling and prepared intermediate ONNX graphs under
`build/qnn`. Context binaries are SoC-specific; use the exact target described
in [COMPATIBILITY.md](COMPATIBILITY.md).

The Samsung QNN provider compatibility AAR is recreated with:

```powershell
./scripts/build_samsung_qnn_2_4_compat.ps1
```

## Native frontend

`native/misaki-android` builds a `cdylib` exporting JNI symbols for
`com.local.kokorotts.NativeMisaki`. Build for `aarch64-linux-android`, remap
local source paths in Rust debug metadata, and strip nonessential symbols before
copying the result to `app/src/main/jniLibs/arm64-v8a/libmisaki_android.so`.
Update the pinned SHA-256 only after parity tests pass.

eSpeak NG's exact source, configuration, and build command are in
`third_party/espeak-ng-1.52.0/PROVENANCE.md`.

## Build and test

```powershell
./gradlew test
./gradlew lint
./gradlew assembleRelease
```

Release builds are unsigned. Sign with a private release key that is never
committed, then run `zipalign -c -P 16 -v 4` and `apksigner verify --verbose`.
