# Kokoro Offline TTS v1.30.1

This patch rebuilds the native Misaki frontend for the public
`com.local.kokorotts` application identity. Version 1.30.0's APK retained JNI
symbols from the pre-sanitization development package and could not initialize
the frontend after installation. The source-level namespace was already
correct; v1.30.1 replaces the stale native artifact and adds a build-time JNI
symbol guard so that mismatch cannot silently recur.

The speech model, 28 voices, Qualcomm QNN 2.48 contexts, CPU/HTP pipeline,
streaming policy, and measured performance architecture are unchanged from
v1.30.0.

## Validation

- Native Misaki host tests: 156/156 passed.
- Exported JNI symbols verified for `com.local.kokorotts`.
- Installed v1.30.1 identity and behavior verified on SM-S928U1.
- Physical-device synthesis, QNN/HTP execution, and playback callbacks passed.
- After the privacy-only Rust path-remapping rebuild, the final APK passed unit
  tests, lint, JNI-symbol checks, signing verification, and 16 KiB alignment.

## Real-device proof

[Watch the 16.3-second native self-test](https://github.com/cedgeremek/kokoro-offline-tts-android/releases/download/v1.30.1-s24-qnn/Kokoro-v1.30.1-S24-Ultra-QNN-native-self-test.mp4),
recorded from v1.30.1 on an SM-S928U1 in airplane mode.
The warm run returned first audio in **306 ms**, completed its self-test in
2,731 ms, and reported `QNN_HTP` on the T=192 AOT context at RTF 0.2644
(approximately **3.78x real-time generator throughput**). The clip contains
only the app's own interface, bundled `af_alloy` voice, and device-playback
audio.

## Installation

Install this APK, open **Kokoro Offline TTS** once, and select it in Android's
Text-to-speech settings or directly in a compatible reader. Version 1.30.1 is
the first working release under the privacy-sanitized public package identity.
