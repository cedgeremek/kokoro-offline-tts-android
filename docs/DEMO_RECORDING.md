# Reproducible demo recording

The public demos use only project-authored text from [`demo/`](demo/). They do
not reproduce the Bloomberg article used during private long-form acceptance
testing.

## Planned clips

1. **Native self-test:** open Kokoro Offline TTS, run **Test selected voice**,
   and leave the resulting first-audio latency, QNN/HTP bucket, and generator
   RTF visible on screen.
2. **Continuous reader:** open [`LONGFORM_DEMO.txt`](demo/LONGFORM_DEMO.txt) in
   @Voice Aloud Reader, enable airplane mode, and record uninterrupted playback
   with sentence highlighting visible.

@Voice Aloud Reader is a third-party application used only to demonstrate
compatibility with Android's standard `TextToSpeech` API. This project is not
affiliated with or endorsed by Hyperionics Technology.

## Privacy checklist

- Enable Do Not Disturb and clear notifications before recording.
- Use airplane mode for the synthesis portion.
- Keep account names, recent-file lists, browser tabs, and unrelated apps out
  of frame.
- Use the ad-free @Voice view or crop the recording so no personalized ad is
  included.
- Review every final frame and the audio track before publishing.
