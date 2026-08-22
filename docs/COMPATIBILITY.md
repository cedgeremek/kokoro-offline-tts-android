# Device compatibility and porting

## Current matrix

| Device class | Expected behavior | Work needed |
|---|---|---|
| Galaxy S24 Ultra SM-S928U1, Snapdragon SM8650 | Full QNN HTP v75 acceleration; physically qualified | None |
| Other Galaxy S24 Ultra / SM8650 variants | Likely close, but not release-qualified; OEM/FastRPC differences can matter | Test device guard, FastRPC node, audio, callbacks, and thermal behavior |
| Other Snapdragon 8 Gen 3 / SM8650 phones | AOT contexts target the right HTP generation, but vendor DSP access may differ | Adapt the FastRPC probe and allowlist only after on-device qualification |
| Snapdragon SM8750 devices | q8 CPU fallback with the current APK | Recompile all buckets for HTP v79, ship matching runtime/stub/skel, update hashes and guard |
| Snapdragon SM8850 devices | q8 CPU fallback with the current APK | Recompile for HTP v81 and repeat the same qualification |
| Older/different Snapdragon SoCs | q8 CPU fallback | Build contexts for the exact SoC/Hexagon architecture; do not reuse v75 binaries blindly |
| Exynos, Tensor, MediaTek | arm64 q8 CPU fallback may function but will be much slower | Add and validate another accelerator backend such as a suitable LiteRT/NNAPI/GPU path |

Google's Qualcomm LiteRT documentation maps SM8650 to HTP v75, SM8750 to v79,
and SM8850 to v81, and explains why AOT artifacts are target-specific:
<https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/qualcomm/doc/HTP_INSTRUCTIONS.md>.

The Android project declares arm64-v8a only, minSdk 27 (Android 8.1), targetSdk
35, and compileSdk 36.

## Porting recipe for another Snapdragon phone

1. Identify the exact SoC model and Hexagon/HTP architecture.
2. Compile every B64-B640 repaired vocoder suffix with the matching QAIRT target.
3. Package the matching QNN runtime, HTP stub, and skel libraries.
4. Inspect the OEM FastRPC device node and linker namespace. Do not assume the
   Samsung `adsprpc-smd` probe exists elsewhere.
5. Update context byte counts/SHA-256 values, HTP architecture, and the runtime
   device allowlist.
6. Run CPU-reference numerical gates, direct PCM and Android callback parity,
   long-reader starvation tests, seam/click analysis, and a sustained thermal
   test on the target phone.

If an OEM denies normal-app FastRPC/unsigned-PD access, this is not something a
Kotlin flag can bypass safely. Keep the CPU fallback or choose a platform API
the OEM exposes.

## Is it the fastest phone Kokoro implementation?

It is a credible high-performance contender on the tested S24 Ultra, and the
published measurements are stronger evidence than an unqualified claim. A
search of public Android Kokoro engines found broad CPU/sherpa-onnx solutions
and projects that explicitly report QNN as not yet viable, but no same-phone,
same-model, same-text comparison was available. Therefore the project claims
**physically verified QNN HTP execution and near-real-time continuous use**, not
an unverifiable universal record.
