# Kokoro Offline TTS for Android

Near-real-time, fully local Kokoro-82M speech as a system-wide Android TTS
engine. This build is aggressively specialized for the Snapdragon Galaxy S24
Ultra: the CPU handles text, duration, conditioning, exact harmonic-source
math, and iSTFT while Qualcomm HTP executes the 1,153-node neural-vocoder
suffix. Android can play one window while the next is being produced.

The release APK needs no account, model download, cloud service, or network
permission. It exposes 28 English Kokoro v1 voices to reader apps, navigation,
accessibility tools, and anything else using Android's `TextToSpeech` API.

> **Hardware status:** physically qualified on Samsung Galaxy S24 Ultra
> SM-S928U1 / Snapdragon 8 Gen 3 (SM8650, HTP v75). Other arm64 phones can use
> the q8 CPU fallback; extending hardware acceleration requires target-specific
> QNN contexts. See [compatibility](docs/COMPATIBILITY.md).

## What makes this build unusual

| Enhancement | What it does | Why it matters |
|---|---|---|
| CPU + NPU pipeline overlap | While the opening HTP job runs, a separate CPU worker prepares the next exact source spectrum; playback overlaps later generation | Hides serial work without running two HTP jobs against each other |
| 11 AOT HTP buckets | B64, 96, 128, 192, 208, 224, 256, 320, 384, 512, and 640 contexts, compiled for SM8650/HTP v75 | Covers short through long windows with no phone-side JIT |
| Graph surgery for real silicon | Replaced 50 exact `Pow(x, 2)` operations with `Mul(x, x)` and kept the small source-spectrum prefix on CPU | Avoids an observed HTP numerical failure while leaving the 1,153-node neural-vocoder suffix accelerated |
| Proved no silent CPU fallback | Every HTP session sets `session.disable_cpu_ep_fallback=1` | A logged `QNN_HTP` window really used QNN rather than quietly dropping nodes onto CPU |
| Duration-aligned global planning | Runs the full-sentence front once, maps word-safe cuts onto the actual duration-expanded frame path, and refines with bounded depth | Preserves full-sentence prosody without sending giant cold windows to the accelerator |
| Three-session LRU | Keeps the common B128/B192/B208 reading contexts warm and bounds live accelerator sessions | Cuts repeat context startup while containing memory/resource use |
| Scoped Qualcomm power modes | Provider is `balanced` while resident; `qnn.perf_mode=burst` exists only for a live inference call | Keeps high-performance mode out of idle time |
| Window-specific compiler policy | 8 MiB VTCM is used on selected large buckets; finalizer mode was tuned by physical-device A/B testing | Avoids assuming one compiler recipe is optimal for every shape |
| Seam-safe streaming | 50 ms raised-cosine shared-timeline joins, no duplicated samples, plus a 120 ms leading / 30 ms trailing request-edge budget | Smooth continuous reading and protects first-word onset |
| Robust PCM conditioning | p99.5 encoder, continuous soft limiter, speech-band median loudness, 1 dB corridor, 6 dB cap, carried/ramped gain, 0.95 headroom | Reduces clicks, pumping, clipping, and loudness jumps |
| Retry-aware PCM cache | Cache identity includes backend, context, and retry generation; fallback is lazy and cancellation-safe | Fast repeats without serving stale backend output |
| Exact native English frontend | Misaki/eSpeak 1.52 frontend with persisted US/GB parity checks | More faithful names, numbers, punctuation, and British/American pronunciation than a toy tokenizer |

The accelerator here is the Qualcomm Hexagon **HTP/NPU**, not the Adreno GPU.
The clever part is the CPU/HTP scheduling: CPU-only preparation overlaps the
serialized HTP critical path, and the Android audio consumer overlaps both.

## Measured on a real S24 Ultra

All figures below are from physical-device runs, not emulator estimates.
`RTF` is compute seconds per generated audio second; lower is better.

| Measurement | Before | After | Improvement |
|---|---:|---:|---:|
| First PCM, q8 CPU vs progressive QNN | 1,252 ms | 779 ms | **37.8% lower latency / 1.61x faster** |
| Duplicate QNN-context repair, first PCM | 2,333 ms | 1,130 ms | **51.6% lower / 2.06x faster** |
| Duplicate QNN-context repair, total request | 4,662 ms | 3,488 ms | **25.2% lower / 1.34x faster** |
| Generator RTF after context-cache repair | 0.7379 | 0.3104 | **57.9% lower / 2.38x throughput** |
| Whole-window to progressive first PCM | 1,257 ms | 779 ms | **38.0% lower / 1.61x faster** |
| Default model delivery speed | 1.0x | 1.3x | **23.1% shorter playback time** (mathematical, not compute acceleration) |

### Whole-article usage run

The v1.30 acceptance run read a Bloomberg article through the public Android
callback path using `am_puck`, Android rate 1.0, and model speed 1.3:

| Actual reading-session statistic | Result |
|---|---:|
| Delivered speech | 460.470 s / **7 min 40.47 s** |
| Wall-clock session | 497.775 s / **8 min 17.78 s** |
| Delivered audio / wall time | **92.51%** |
| Requests planned / completed | **76 / 76** |
| Generator windows | **379 / 379 on QNN HTP** |
| Generator RTF | mean **0.2911**, median **0.29**, p95 **0.32**, max **0.39** |
| Equivalent mean generator throughput | **3.44x real time** |
| First PCM | median **949.8 ms**, p95 **1,372.8 ms**, max **1,642.2 ms** |
| QNN failures / rejected callback chunks / CPU generator windows | **0 / 0 / 0** |

The session wall time includes prewarm, Android callback handoff, request gaps,
and app scheduling, so it is deliberately not presented as generator RTF.
Reproducible summary data and calculation notes are in
[performance.md](docs/PERFORMANCE.md).

## Power-minded design

No calibrated wattmeter or battery-drain experiment was captured for v1.30, so
this project does not invent a watts or battery-per-hour claim. It does make
power-conscious architectural choices:

- The dominant vocoder runs on HTP instead of sustaining the general-purpose CPU.
- HTP sessions stay `balanced` at rest and switch to `burst` only around inference.
- The q8 CPU fallback is lazy; it does not occupy an idle session on the normal path.
- HTP work is serialized, avoiding two accelerator jobs fighting for bandwidth and power.
- The three-context LRU bounds resident sessions while preserving the common reading buckets.
- AOT contexts remove repeated on-device compilation; session reuse removes duplicate context creation.
- The CPU prefix is overlapped with work that must happen anyway; it is not duplicate speculation.
- There is no network permission, so synthesis never wakes Wi-Fi/cellular for model or telemetry traffic.

These choices should improve energy per spoken minute and race-to-idle behavior,
but that conclusion remains architectural until a controlled battery/thermal
benchmark is published.

## Fresh Qualcomm stack

This release moved quickly onto the then-newly published
[`com.qualcomm.qti:qnn-runtime:2.48.0`](https://central.sonatype.com/artifact/com.qualcomm.qti/qnn-runtime/2.48.0),
with contexts produced by QAIRT 2.48.40. [Qualcomm AI Hub added QAIRT
2.48.0](https://dev.aihub.qualcomm.com/docs/hub/release_notes.html) on August 3,
2026. The useful part was not the version number alone: the app ships
SM8650/HTP-v75 AOT contexts, registers QNN directly through ONNX Runtime, uses
8 MiB VTCM where it helped, and never JIT-compiles a speech graph on the phone.

Samsung exposes the compute DSP through `adsprpc-smd`; Qualcomm's QNN 2.4
provider originally probed `fastrpc-cdsp`. The included reproducible
compatibility build changes exactly that 12-byte NUL-padded device probe and
nothing in model execution. See
[`third_party/qualcomm-qnn-samsung-compat/PROVENANCE.md`](third_party/qualcomm-qnn-samsung-compat/PROVENANCE.md).

## Install

1. Download the APK from the latest GitHub Release.
2. Allow installation from the browser/file manager and install it.
3. Open **Kokoro Offline TTS** once and wait for the readiness check.
4. In Android Settings, choose Kokoro Offline TTS under **Text-to-speech**.
5. Select a voice and use it from a reader such as @Voice Aloud Reader.

The APK is large (about 1.40 GiB / 1.50 GB) because all model weights, 28 voices, native
libraries, and 11 no-JIT HTP contexts are bundled for completely offline use.

## Does something similar already exist?

Yes—this is not the first Android Kokoro project. [HayaiTTS](https://github.com/HayaiApp/HayaiTTS)
offers a broad multi-engine catalog, [NekoSpeak](https://github.com/siva-sub/NekoSpeak)
is a polished private/offline system engine, and [JokobeeTTS](https://github.com/Jokobee/JokobeeTTS)
packages a developer-facing Android Kokoro library. They are useful projects
with different goals.

The non-duplicate contribution here is the narrow, deeply optimized Snapdragon
path: physically verified Kokoro-v1 generation on Qualcomm HTP, eleven static
AOT buckets, explicit no-CPU-fallback proof, real CPU/NPU pipeline overlap, and
continuous-reader callback measurements. The comparison is architectural, not
a claim of a universal speed record; no fair same-phone, same-text benchmark
against every other app has been run.

## Source and building

The repository contains the Android engine, native Misaki frontend, graph
preparation/repair scripts, validators, and provenance. Giant generated ONNX
models and QNN contexts are intentionally excluded from ordinary Git history;
GitHub Releases carry the installable APK. Rebuilding the exact accelerated
variant requires the Kokoro v1 assets plus Qualcomm QAIRT 2.48 tooling.

Start with [BUILDING.md](docs/BUILDING.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md),
and [COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Privacy and licenses

The manifest requests no Internet permission, `allowBackup` is false, and there
is no account, analytics SDK, telemetry, ad ID, or model downloader. See
[PRIVACY.md](docs/PRIVACY.md).

Project-authored source is Apache-2.0. Bundled components retain their own
licenses: Kokoro/Misaki (Apache-2.0), voice-g2p and ONNX Runtime/QNN EP (MIT),
eSpeak NG (GPL-3.0-or-later, with complete corresponding source), and Qualcomm
QNN runtime under Qualcomm's distribution terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[DISTRIBUTION_REVIEW.md](docs/DISTRIBUTION_REVIEW.md).
