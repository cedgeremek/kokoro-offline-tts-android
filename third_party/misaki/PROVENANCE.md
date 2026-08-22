# Current Misaki English frontend provenance

The active Android frontend targets `misaki` 0.9.4 at commit
`fba1236595f2d2bf21d414ba6e57d25256afada3` (Apache-2.0).

`native/misaki-android` is derived from `rgbkrk/voice`'s `voice-g2p` crate at
commit `0e911aaff676ad4c5e08c395df790041418f87b9` (MIT). Product-specific
pronunciation overrides, the generated bronze dictionary, and its rough OOV
fallback are compile-disabled. The Android crate instead embeds the exact
pinned Misaki US/GB gold and silver JSON, implements the current normalization,
context, number, inflection, identifier, and accent branches, and sends genuine
OOV text through eSpeak NG with Misaki's conversion tables.

Python Misaki uses spaCy. Android uses the self-contained averaged-perceptron POS
model supplied by `voice-g2p`, so this is described as a behavioral port rather
than a byte-for-byte reimplementation of the Python dependency stack. The
persisted oracle comparison at `build/misaki-parity-v1/report.json` covers 55
representative inputs in each accent and currently matches 110/110 exactly.

The older `misaki_en_us.mlex` and `misaki_en_gb.mlex` assets and the Kotlin
`OfflineEnglishG2p` implementation remain only as historical source/test material.
Gradle excludes all `.mlex` files, and the active `EnglishPhonemizer` exclusively
calls the native frontend.

The Apache-2.0 text is retained in `LICENSE` and bundled in the APK as
`assets/licenses/Apache-2.0.txt`. The upstream MIT text is retained in
`native/misaki-android/LICENSE-UPSTREAM` and bundled as
`assets/licenses/voice-g2p-MIT.txt`.
