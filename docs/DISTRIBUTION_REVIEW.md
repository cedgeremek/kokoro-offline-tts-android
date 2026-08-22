# Distribution review

Review result for v1.30.0: suitable for public GitHub source and APK distribution
with the component notices and source offer included here.

| Component | Distribution basis | Included compliance material |
|---|---|---|
| Project-authored Kotlin, Rust, and scripts | Apache-2.0 | Top-level `LICENSE` |
| Kokoro-82M model/tokenizer/voices | Apache-2.0 | Apache license and source/revision notice |
| Misaki data/behavior | Apache-2.0 | Apache license and pinned provenance |
| voice-g2p base/POS model | MIT | MIT license and pinned provenance |
| ONNX Runtime and QNN EP source | MIT | Upstream links and compatibility-patch provenance |
| eSpeak NG 1.52 | GPL-3.0-or-later | License, exact source revision, build recipe, and complete corresponding source archive |
| Qualcomm QNN runtime 2.48 | Qualcomm AI Hub Model License | Exact runtime `LICENSE.pdf` bundled in the APK and repository |

The Qualcomm runtime license grants distribution of object code when
incorporated into an application; the runtime is not offered here as a
standalone product. Required copyright/license notices are retained. The local
Samsung QNN provider patch is derived from MIT-licensed provider source and is
fully described in its provenance file.

The release repository deliberately excludes signing keys, `local.properties`,
Gradle/build caches, device logs, local paths, and giant generated model/context
blobs from ordinary Git history. The combined installable APK is published as a
GitHub Release asset.

This is a practical engineering/licensing review, not legal advice. No trademark
affiliation with Kokoro, Qualcomm, Samsung, Microsoft, or the related projects
is implied.
