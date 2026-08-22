# eSpeak NG 1.52.0 Android provenance

The packaged arm64 library is built from official `espeak-ng/espeak-ng` tag
`1.52.0`, commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`, retrieved
2026-08-18. It is GPL-3.0-or-later; the verbatim license is retained as `COPYING`
and bundled at `assets/licenses/espeak-ng-GPL-3.0-or-later.txt`.

Build inputs were Android NDK 28.2.13676358, CMake 3.31.6, Ninja, ABI
`arm64-v8a`, API 27, and Release shared-library mode. Klatt was enabled;
asynchronous playback, speech-player, PcAudio, Sonic, and MBROLA were disabled
because the app uses only the synchronous phoneme API.

Reproduction command from the repository root:

```powershell
git clone --depth 1 --branch 1.52.0 https://github.com/espeak-ng/espeak-ng.git .build-temp\espeak-ng-1.52.0
$cmake = "$env:LOCALAPPDATA\Android\Sdk\cmake\3.31.6\bin\cmake.exe"
$ninja = "$env:LOCALAPPDATA\Android\Sdk\cmake\3.31.6\bin\ninja.exe"
$toolchain = "$env:LOCALAPPDATA\Android\Sdk\ndk\28.2.13676358\build\cmake\android.toolchain.cmake"
& $cmake -S .build-temp\espeak-ng-1.52.0 -B .build-temp\espeak-ng-android-arm64 -G Ninja `
  "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-27 -DANDROID_STL=c++_static `
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DUSE_MBROLA=OFF `
  -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=ON `
  -DUSE_SPEECHPLAYER=OFF -DUSE_ASYNC=OFF -DCMAKE_BUILD_TYPE=Release
& $cmake --build .build-temp\espeak-ng-android-arm64 --target libespeak-ng.so --parallel 8
```

The output was renamed to `libttsespeak.so`, stripped with the matching NDK
`llvm-strip --strip-unneeded`, and pinned at 452,608 bytes with SHA-256
`1a887ac7fad29783369630020708aeeeecfbc2036b8ac7e27898d11bdaba673d`.
It exports `espeak_Info`, `espeak_Initialize`, `espeak_SetVoiceByName`, and
`espeak_TextToPhonemes`; its only `DT_NEEDED` entries are Android `libm`,
`libdl`, and `libc`.

The complete pinned corresponding source is staged beside this file as
`espeak-ng-1.52.0-source.tar.gz` (17,823,678 bytes, SHA-256
`8ccd21f4b4ba500e89a1e132b1f14e165e20f3056145cb0b7b86de807ee388e4`) so a
transfer bundle does not depend on a future network checkout.

The 364 compiled data files (18,373,365 bytes) are from the current
`espeakng-loader` 0.2.4 wheel used by pinned Misaki validation. That library and
the official source build both report eSpeak 1.52.0. The principal data hashes
are: `en_dict` `1053f74a...`, `phondata` `a0b643b1...`, `phonindex`
`384e5fa6...`, and `phontab` `1b406906...`.
