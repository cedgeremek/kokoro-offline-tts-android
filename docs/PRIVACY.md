# Privacy

Kokoro Offline TTS performs synthesis on the device.

- No `INTERNET` permission.
- No account or sign-in.
- No analytics, telemetry, advertising SDK, crash uploader, or ad identifier.
- No cloud TTS call and no model downloader.
- Android backup is disabled (`android:allowBackup="false"`).
- The public application ID is the neutral `com.local.kokorotts`.
- The release APK and tracked repository files are scanned for local usernames,
  home-directory paths, email addresses, credentials, signing keys, and tokens.

Text supplied by another app enters the Android TTS service and is converted to
PCM locally. Ordinary Android/system logging may contain timing and technical
diagnostics; the app does not transmit those logs.
