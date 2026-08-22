# Native Misaki-compatible English frontend

This crate is derived from `rgbkrk/voice`'s `voice-g2p` crate at commit
`0e911aaff676ad4c5e08c395df790041418f87b9` (MIT). The Android integration
removes that fork's product-specific pronunciation overrides and generated
"bronze" fallback dictionary. It embeds the unmodified English gold/silver
lexicons from `hexgrad/misaki` commit
`fba1236595f2d2bf21d414ba6e57d25256afada3` (Apache-2.0), adds the matching
GB branch, and sends OOV words through eSpeak NG with Misaki's conversion
table.

The perceptual tagger is the self-contained averaged-perceptron model shipped
by `voice-g2p`; it replaces spaCy on Android. Behavioral parity is checked
against the pinned official Python Misaki oracle. Any known mismatch must be
reported rather than describing this implementation as byte-identical Python
Misaki.
