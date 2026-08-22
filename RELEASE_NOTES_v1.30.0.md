# v1.30.0 — duration-refined global streaming for S24 Ultra

This is the first public, privacy-sanitized release of the physically qualified
S24 Ultra build. It installs under the neutral package `com.local.kokorotts`
with a separate release certificate, so it can coexist with private development
builds and will not update them in place.

Highlights:

- 379/379 article-run generator windows executed on Qualcomm QNN HTP.
- 7 min 40.47 s of delivered speech in an 8 min 17.78 s wall-clock session.
- Mean generator RTF 0.2911 (3.44x audio-time throughput); p95 0.32.
- Median first PCM 949.8 ms across 76 completed requests.
- Full-sentence CPU front with exact duration-aligned, word-safe HTP windows.
- CPU/HTP pipeline overlap plus concurrent Android playback.
- 11 no-JIT SM8650/HTP-v75 AOT buckets generated with QAIRT 2.48.40.
- Exact-square graph repair and CPU source-spectrum cut for HTP numerical safety.
- Scoped balanced/burst power policy, three-session LRU, and lazy CPU fallback.
- 50 ms shared-timeline raised-cosine joins and onset-safe 120/30 ms edge budget.
- 28 English voices, fully local operation, and no Internet permission.

Qualified device: Samsung Galaxy S24 Ultra SM-S928U1, Snapdragon 8 Gen 3
(SM8650). Other arm64 devices can use the slower q8 CPU route; QNN acceleration
must be rebuilt and qualified for their SoC/HTP generation.

Public-build verification:

- Debug unit tests: 46/46.
- Release unit tests: 46/46.
- Android lint: zero errors.
- APK Signature Schemes v2 and v3 verified with a neutral 4096-bit RSA release certificate.
- 16 KiB ZIP/page alignment verified.
- Manifest application ID: `com.local.kokorotts`; no Internet permission.
- APK privacy scan: no developer username, home-directory path, original private
  package ID, email address, credential, token, or private signing material.
- APK size: 1,499,532,561 bytes (about 1.40 GiB).
- SHA-256: `cadd795947106d9f0652dc858d1dd51863c1c66155997f661c12a212203e50fa`.
