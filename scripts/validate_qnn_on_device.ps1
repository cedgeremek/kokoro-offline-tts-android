[CmdletBinding()]
param(
    [string]$DeviceSerial,
    [string]$HostAudioGateReceipt,
    [string]$AndroidSdk = (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
    [string]$JavaHome = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($HostAudioGateReceipt)) {
    $HostAudioGateReceipt = Join-Path $projectRoot `
        'build\qnn\split_experiment\aot-masked-b256-b384-fp32\HOST_B256_B384_GUARDED_28_VOICE_GATE.txt'
}
$frontAssetName = 'kokoro-front.fp32.onnx'
$generatorAssetName = 'kokoro-generator.masked-dynamic.fp32.onnx'
$qnnB256ContextAssetName = 'kokoro-generator.masked-b256.qnn248.fp32.ctx.onnx'
$qnnB384ContextAssetName = 'kokoro-generator.masked-b384.qnn248.fp32.ctx.onnx'
$frontAsset = Join-Path $projectRoot "app\src\main\assets\$frontAssetName"
$generatorAsset = Join-Path $projectRoot "app\src\main\assets\$generatorAssetName"
$qnnB256ContextAsset = Join-Path $projectRoot "app\src\main\assets\$qnnB256ContextAssetName"
$qnnB384ContextAsset = Join-Path $projectRoot "app\src\main\assets\$qnnB384ContextAssetName"
$expectedHostGateSha256 = 'afe043efbf1a2fd01d3e9a32d0a533f2004f07bc9eb5f63f899d332cc4c0dfd7'
$expectedFrontSha256 = '9e9cb970bf8a004ebdd6b27b91ee935fc400aea9e905037c657185cf0caeb0c1'
$expectedGeneratorSha256 = 'e272c38842e7b913d81d79acdd4a86055891c48ae699b6857468812236894cf0'
$expectedB256ContextSha256 = 'e59874d7f11675151920d45ca75da7516e9ba952379bd5ae4aa719f7ac8a31db'
$expectedB384ContextSha256 = 'e48d42eab91be0ef7cd9d0f7bf68fbc949ef0d652f725222933f8cf7fd6473ae'
$expectedB256ContextBytes = 132882915L
$expectedB384ContextBytes = 168993252L
$ortCoordinate = 'com.microsoft.onnxruntime:onnxruntime-android:1.24.3'
$qnnProviderCoordinate = 'com.qualcomm.qti:onnxruntime-android-qnn:2.4.0'
$qnnRuntimeCoordinate = 'com.qualcomm.qti:qnn-runtime:2.48.0'
$qnnContextProducer = 'QAIRT_2.48.40_FP32_B256_B384'
$qnnSessionSource = 'PACKAGED_AOT_QNN248_FP32'
$qnnPerformancePolicy = 'provider=balanced;run.qnn.perf_mode=burst'
$qnnPrecisionPolicy = 'graph_io=FP32;htp_math=FP32'
$qnnAssignmentPolicy = 'session.disable_cpu_ep_fallback=1'
$receiptPath = Join-Path $projectRoot 'qnn-validation.properties'
$receiptPartPath = "$receiptPath.$PID.part"
$adb = Join-Path $AndroidSdk 'platform-tools\adb.exe'

# A prior receipt must never survive a failed, skipped, or incomplete attempt.
Remove-Item -LiteralPath $receiptPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $receiptPartPath -Force -ErrorAction SilentlyContinue

foreach ($required in @(
    $HostAudioGateReceipt,
    $frontAsset,
    $generatorAsset,
    $qnnB256ContextAsset,
    $qnnB384ContextAsset,
    (Join-Path $JavaHome 'bin\java.exe'),
    $adb
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing prerequisite: $required"
    }
}

function Get-LowerSha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-TextSha256([string]$Text) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
}

function Get-ZipEntrySha256([object]$Archive, [string]$EntryName) {
    $entry = $Archive.GetEntry($EntryName)
    if ($null -eq $entry) { throw "APK is missing $EntryName" }
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = $entry.Open()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

$hostGateSha256 = Get-LowerSha256 $HostAudioGateReceipt
if ($hostGateSha256 -cne $expectedHostGateSha256) {
    throw "Unexpected 28-voice B256/B384 host-gate receipt SHA-256: $hostGateSha256"
}
$generatorSha256 = Get-LowerSha256 $generatorAsset
if ($generatorSha256 -cne $expectedGeneratorSha256) {
    throw "Unexpected masked generator SHA-256: $generatorSha256"
}
$frontSha256 = Get-LowerSha256 $frontAsset
if ($frontSha256 -cne $expectedFrontSha256) {
    throw "Unexpected split CPU front-end SHA-256: $frontSha256"
}
$qnnB256ContextSha256 = Get-LowerSha256 $qnnB256ContextAsset
$qnnB384ContextSha256 = Get-LowerSha256 $qnnB384ContextAsset
if ($qnnB256ContextSha256 -cne $expectedB256ContextSha256) {
    throw "Unexpected audited B256 FP32 QNN context SHA-256: $qnnB256ContextSha256"
}
if ($qnnB384ContextSha256 -cne $expectedB384ContextSha256) {
    throw "Unexpected audited B384 FP32 QNN context SHA-256: $qnnB384ContextSha256"
}
if ((Get-Item -LiteralPath $qnnB256ContextAsset).Length -ne $expectedB256ContextBytes -or
    (Get-Item -LiteralPath $qnnB384ContextAsset).Length -ne $expectedB384ContextBytes) {
    throw 'Packaged QNN context byte counts do not match the audited artifacts.'
}

$adbArguments = @()
if (-not [string]::IsNullOrWhiteSpace($DeviceSerial)) {
    $adbArguments = @('-s', $DeviceSerial)
}

function Get-DeviceProperty([string]$Name) {
    $value = (& $adb @adbArguments shell getprop $Name) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "adb getprop failed for $Name" }
    return $value.Trim()
}

if ((Get-DeviceProperty 'ro.kernel.qemu') -eq '1') {
    throw 'QNN validation is not supported on an emulator; connect the physical Snapdragon phone.'
}
if ((Get-DeviceProperty 'ro.product.cpu.abi') -cne 'arm64-v8a') {
    throw 'The connected target is not arm64-v8a.'
}
if ((Get-DeviceProperty 'ro.soc.model') -cne 'SM8650') {
    throw 'The connected target is not the Snapdragon 8 Gen 3 SM8650 device required by this candidate.'
}
if ((Get-DeviceProperty 'ro.product.model') -notmatch '^SM-S928') {
    throw 'The connected target is not a Galaxy S24 Ultra (SM-S928 family).'
}

$env:JAVA_HOME = (Resolve-Path -LiteralPath $JavaHome).Path
Push-Location $projectRoot
try {
    & .\gradlew.bat testDebugUnitTest assembleDebug assembleDebugAndroidTest lintDebug --console=plain
    if ($LASTEXITCODE -ne 0) { throw 'Focused Android build/tests/lint failed.' }
} finally {
    Pop-Location
}

$apk = Join-Path $projectRoot 'app\build\outputs\apk\debug\app-debug.apk'
$testApk = Join-Path $projectRoot 'app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk'
$metadataPath = Join-Path $projectRoot 'app\build\outputs\apk\debug\output-metadata.json'
foreach ($builtArtifact in @($apk, $testApk, $metadataPath)) {
    if (-not (Test-Path -LiteralPath $builtArtifact -PathType Leaf)) {
        throw "Android build did not produce $builtArtifact"
    }
}
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
if ($metadata.applicationId -cne 'com.local.kokorotts' -or $metadata.elements.Count -ne 1) {
    throw 'Unexpected APK output metadata.'
}
$apkElement = $metadata.elements[0]
$apkSha256 = Get-LowerSha256 $apk
$testApkSha256 = Get-LowerSha256 $testApk
$apkBytes = (Get-Item -LiteralPath $apk).Length

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($apk)
try {
    $entries = @($archive.Entries.FullName)
    $expectedOnnxEntries = @(
        "assets/$frontAssetName",
        "assets/$generatorAssetName",
        "assets/$qnnB256ContextAssetName",
        "assets/$qnnB384ContextAssetName"
    )
    $actualOnnxEntries = @($entries | Where-Object { $_ -like 'assets/*.onnx' } | Sort-Object)
    if (($actualOnnxEntries -join '|') -cne (($expectedOnnxEntries | Sort-Object) -join '|')) {
        throw "APK ONNX asset set is not exact: $($actualOnnxEntries -join ', ')"
    }
    foreach ($onnxEntryName in $expectedOnnxEntries) {
        $onnxEntry = $archive.GetEntry($onnxEntryName)
        if ($null -eq $onnxEntry -or $onnxEntry.CompressedLength -ne $onnxEntry.Length) {
            throw "APK must store $onnxEntryName uncompressed for direct mmap"
        }
    }
    $abiDirectories = @(
        $entries |
            Where-Object { $_ -like 'lib/*/*.so' } |
            ForEach-Object { ($_ -split '/')[1] } |
            Sort-Object -Unique
    )
    if (($abiDirectories -join ',') -cne 'arm64-v8a') {
        throw "APK contains unexpected native ABIs: $($abiDirectories -join ', ')"
    }
    foreach ($library in @(
        'lib/arm64-v8a/libonnxruntime.so',
        'lib/arm64-v8a/libonnxruntime_providers_qnn.so',
        'lib/arm64-v8a/libQnnHtp.so',
        'lib/arm64-v8a/libQnnHtpV75Stub.so',
        'lib/arm64-v8a/libQnnHtpV75Skel.so',
        'lib/arm64-v8a/libQnnSystem.so'
    )) {
        if (-not ($entries -ccontains $library)) {
            throw "The required ORT/QNN runtime library is absent from the APK: $library"
        }
    }
    $packagedGeneratorSha256 = Get-ZipEntrySha256 $archive "assets/$generatorAssetName"
    if ($packagedGeneratorSha256 -cne $generatorSha256) {
        throw "Packaged generator hash mismatch: $packagedGeneratorSha256"
    }
    $packagedFrontSha256 = Get-ZipEntrySha256 $archive "assets/$frontAssetName"
    if ($packagedFrontSha256 -cne $frontSha256) {
        throw "Packaged front-end hash mismatch: $packagedFrontSha256"
    }
    $packagedB256ContextSha256 = Get-ZipEntrySha256 $archive "assets/$qnnB256ContextAssetName"
    if ($packagedB256ContextSha256 -cne $qnnB256ContextSha256) {
        throw "Packaged B256 FP32 QNN context hash mismatch: $packagedB256ContextSha256"
    }
    $packagedB384ContextSha256 = Get-ZipEntrySha256 $archive "assets/$qnnB384ContextAssetName"
    if ($packagedB384ContextSha256 -cne $qnnB384ContextSha256) {
        throw "Packaged B384 FP32 QNN context hash mismatch: $packagedB384ContextSha256"
    }
} finally {
    $archive.Dispose()
}

& $adb @adbArguments install -r $apk | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Target APK installation failed.' }
& $adb @adbArguments install -r $testApk | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Instrumentation APK installation failed.' }

# Remove only this version's possible compressed-asset fallback copies. Older
# app data is preserved and compared as a baseline instead of being blamed on
# this packaged-AOT validation run.
$currentFallbackFiles = @(
    "./no_backup/v$($apkElement.versionCode)-$qnnB256ContextAssetName",
    "./no_backup/v$($apkElement.versionCode)-$qnnB256ContextAssetName.part",
    "./no_backup/v$($apkElement.versionCode)-$qnnB384ContextAssetName",
    "./no_backup/v$($apkElement.versionCode)-$qnnB384ContextAssetName.part"
)
& $adb @adbArguments shell run-as com.local.kokorotts rm -f @($currentFallbackFiles | ForEach-Object {
    $_.Substring(2)
}) | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Unable to clear current-version AOT fallback-copy targets.' }

function Get-AppPrivateFiles {
    $files = @(& $adb @adbArguments shell run-as com.local.kokorotts find . -type f)
    if ($LASTEXITCODE -ne 0) { throw 'Unable to audit app-private files.' }
    return @($files | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' } | Sort-Object -Unique)
}

$privateFilesBefore = @(Get-AppPrivateFiles)

# Delete only this test's marker. The test repeats this before every assumption.
& $adb @adbArguments shell run-as com.local.kokorotts rm -f `
    files/qnn-audio-gate.txt files/qnn-audio-gate.txt.part | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Unable to delete the stale device validation marker.' }

$testName = 'com.local.kokorotts.KokoroRuntimeInstrumentedTest#qnnHtpMatchesClearCpuReference'
$testStartedUtc = [DateTimeOffset]::UtcNow
$instrumentation = @(
    & $adb @adbArguments shell am instrument -w -r -e class $testName `
        com.local.kokorotts.test/androidx.test.runner.AndroidJUnitRunner
)
$instrumentationExitCode = $LASTEXITCODE
$testEndedUtc = [DateTimeOffset]::UtcNow
$instrumentation | Out-Host
if ($instrumentationExitCode -ne 0 -or ($instrumentation -join "`n") -notmatch 'OK \(1 test\)') {
    throw 'Physical QNN instrumentation failed; no validation receipt was written.'
}

$markerOutput = @(
    & $adb @adbArguments shell run-as com.local.kokorotts cat files/qnn-audio-gate.txt
)
if ($LASTEXITCODE -ne 0) { throw 'The passing test did not publish its QNN audio-gate marker.' }
$markerText = ($markerOutput -join "`n").Trim()
$markerFields = @{}
foreach ($line in ($markerText -split "`r?`n")) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $separator = $line.IndexOf('=')
    if ($separator -le 0) { throw "Malformed device marker line: $line" }
    $key = $line.Substring(0, $separator)
    $value = $line.Substring($separator + 1)
    if ($markerFields.ContainsKey($key)) { throw "Duplicate device marker field: $key" }
    $markerFields[$key] = $value
}

function Assert-MarkerField([string]$Name, [string]$Expected) {
    if (-not $markerFields.ContainsKey($Name)) { throw "Device marker is missing $Name" }
    if ($markerFields[$Name] -cne $Expected) {
        throw "Device marker $Name mismatch: '$($markerFields[$Name])' != '$Expected'"
    }
}

$expectedCases = @(
    [pscustomobject]@{ Voice = 'af_heart'; Speed = '1.0'; Text = 'Ready when you are.' },
    [pscustomobject]@{ Voice = 'af_heart'; Speed = '1.0'; Text = 'This is an example of speech synthesis in English.' },
    [pscustomobject]@{ Voice = 'af_bella'; Speed = '1.0'; Text = 'Numbers like 24, 3.14159, and 2026 should sound natural, not noisy.' },
    [pscustomobject]@{ Voice = 'am_adam'; Speed = '1.0'; Text = 'The quick brown fox jumps over the lazy dog near the river bank.' },
    [pscustomobject]@{ Voice = 'bm_george'; Speed = '1.0'; Text = 'Punctuation, timing, voices.' }
)
$expectedBaseFields = @(
    'receiptVersion', 'result', 'testedAtUtc', 'applicationId', 'versionCode', 'versionName',
    'apkBytes', 'apkSha256', 'frontModelAsset', 'frontModelSha256', 'generatorModelAsset',
    'generatorModelSha256', 'ortCoordinate',
    'qnnProviderCoordinate', 'qnnRuntimeCoordinate', 'qnnContextCacheId', 'qnnContextProducer',
    'qnnSessionSource', 'qnnPerformancePolicy', 'qnnPrecisionPolicy', 'qnnAssignmentPolicy', 'qnnB256ContextAsset',
    'qnnB256ContextSha256',
    'qnnB384ContextAsset', 'qnnB384ContextSha256', 'qnnObservedContextSources',
    'qnnObservedContextHashes', 'deviceFingerprint',
    'deviceManufacturer', 'deviceModel', 'deviceName', 'deviceSoc', 'deviceSdk', 'deviceAbis',
    'backend', 'cases', 'buckets', 'maxBucketSessions'
)
$expectedCaseSuffixes = @(
    'voice', 'textSha256', 'speed', 'referenceSamples', 'candidateSamples', 'cpuElapsedMs',
    'qnnElapsedMs', 'qnnMaxGeneratorRtf', 'qnnBackends', 'qnnContextSources',
    'qnnContextHashes', 'durationDelta', 'peak', 'nrmse', 'correlation', 'roughnessRatio'
)
$expectedMarkerFields = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($field in $expectedBaseFields) { [void]$expectedMarkerFields.Add($field) }
for ($index = 0; $index -lt $expectedCases.Count; $index++) {
    foreach ($suffix in $expectedCaseSuffixes) { [void]$expectedMarkerFields.Add("case.$index.$suffix") }
}
foreach ($field in $markerFields.Keys) {
    if (-not $expectedMarkerFields.Contains($field)) { throw "Unexpected device marker field: $field" }
}
foreach ($field in $expectedMarkerFields) {
    if (-not $markerFields.ContainsKey($field)) { throw "Device marker is missing $field" }
}

Assert-MarkerField 'receiptVersion' '5'
Assert-MarkerField 'result' 'PASSED'
Assert-MarkerField 'applicationId' $metadata.applicationId
Assert-MarkerField 'versionCode' ([string]$apkElement.versionCode)
Assert-MarkerField 'versionName' ([string]$apkElement.versionName)
Assert-MarkerField 'apkBytes' ([string]$apkBytes)
Assert-MarkerField 'apkSha256' $apkSha256
Assert-MarkerField 'frontModelAsset' $frontAssetName
Assert-MarkerField 'frontModelSha256' $frontSha256
Assert-MarkerField 'generatorModelAsset' $generatorAssetName
Assert-MarkerField 'generatorModelSha256' $generatorSha256
Assert-MarkerField 'ortCoordinate' $ortCoordinate
Assert-MarkerField 'qnnProviderCoordinate' $qnnProviderCoordinate
Assert-MarkerField 'qnnRuntimeCoordinate' $qnnRuntimeCoordinate
Assert-MarkerField 'qnnContextProducer' $qnnContextProducer
Assert-MarkerField 'qnnSessionSource' $qnnSessionSource
Assert-MarkerField 'qnnPerformancePolicy' $qnnPerformancePolicy
Assert-MarkerField 'qnnPrecisionPolicy' $qnnPrecisionPolicy
Assert-MarkerField 'qnnAssignmentPolicy' $qnnAssignmentPolicy
Assert-MarkerField 'qnnB256ContextAsset' $qnnB256ContextAssetName
Assert-MarkerField 'qnnB256ContextSha256' $qnnB256ContextSha256
Assert-MarkerField 'qnnB384ContextAsset' $qnnB384ContextAssetName
Assert-MarkerField 'qnnB384ContextSha256' $qnnB384ContextSha256
Assert-MarkerField 'qnnObservedContextSources' "256:$qnnSessionSource,384:$qnnSessionSource"
Assert-MarkerField 'qnnObservedContextHashes' "256:$qnnB256ContextSha256,384:$qnnB384ContextSha256"
Assert-MarkerField 'deviceFingerprint' (Get-DeviceProperty 'ro.build.fingerprint')
Assert-MarkerField 'deviceManufacturer' (Get-DeviceProperty 'ro.product.manufacturer')
Assert-MarkerField 'deviceModel' (Get-DeviceProperty 'ro.product.model')
Assert-MarkerField 'deviceName' (Get-DeviceProperty 'ro.product.device')
Assert-MarkerField 'deviceSoc' (Get-DeviceProperty 'ro.soc.model')
Assert-MarkerField 'deviceSdk' (Get-DeviceProperty 'ro.build.version.sdk')
Assert-MarkerField 'deviceAbis' (Get-DeviceProperty 'ro.product.cpu.abilist')
Assert-MarkerField 'backend' 'QNN_HTP'
Assert-MarkerField 'cases' ([string]$expectedCases.Count)
if ($markerFields.qnnContextCacheId -cnotmatch '^[0-9a-f]{24}$') {
    throw "Invalid QNN context-cache fingerprint: $($markerFields.qnnContextCacheId)"
}

$markerTimestamp = [DateTimeOffset]::Parse(
    $markerFields.testedAtUtc,
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::RoundtripKind
)
if ($markerTimestamp -lt $testStartedUtc -or $markerTimestamp -gt $testEndedUtc.AddMinutes(1)) {
    throw "Device marker timestamp is outside this instrumentation run: $markerTimestamp"
}

$bucketValues = @($markerFields.buckets -split ',' | Where-Object { $_ -ne '' } | ForEach-Object {
    [int]::Parse($_, [Globalization.CultureInfo]::InvariantCulture)
})
if (($bucketValues -join ',') -cne '256,384') {
    throw "The device gate did not exercise exactly packaged B256 and B384: $($markerFields.buckets)"
}
$maxBucketSessions = [int]::Parse($markerFields.maxBucketSessions, [Globalization.CultureInfo]::InvariantCulture)
if ($maxBucketSessions -ne 2) {
    throw "Invalid QNN LRU high-water mark: $maxBucketSessions"
}

for ($index = 0; $index -lt $expectedCases.Count; $index++) {
    $prefix = "case.$index"
    Assert-MarkerField "$prefix.voice" $expectedCases[$index].Voice
    Assert-MarkerField "$prefix.textSha256" (Get-TextSha256 $expectedCases[$index].Text)
    Assert-MarkerField "$prefix.speed" $expectedCases[$index].Speed
    Assert-MarkerField "$prefix.qnnBackends" 'QNN_HTP'
    $caseSources = $markerFields["$prefix.qnnContextSources"]
    $caseHashes = $markerFields["$prefix.qnnContextHashes"]
    if ([string]::IsNullOrWhiteSpace($caseSources) -or [string]::IsNullOrWhiteSpace($caseHashes)) {
        throw "$prefix did not bind its packaged QNN contexts"
    }
    $caseSourceParts = @($caseSources -split ',')
    $caseHashParts = @($caseHashes -split ',')
    if ($caseSourceParts.Count -ne $caseHashParts.Count) {
        throw "$prefix QNN source/hash bucket counts differ"
    }
    foreach ($part in $caseSourceParts) {
        if ($part -notmatch '^(256|384):PACKAGED_AOT_QNN248_FP32$') {
            throw "$prefix contains invalid QNN context source evidence: $part"
        }
        $bucket = $Matches[1]
        $expectedHash = if ($bucket -ceq '256') { $qnnB256ContextSha256 } else { $qnnB384ContextSha256 }
        if ($caseHashParts -cnotcontains "${bucket}:$expectedHash") {
            throw "$prefix is missing the exact packaged T=$bucket context hash"
        }
    }
    if ($index -eq 1 -and $caseSourceParts -cnotcontains "384:$qnnSessionSource") {
        throw "$prefix known Settings sample did not execute packaged B384"
    }
    if ($index -eq 0 -and $caseSourceParts -cnotcontains "256:$qnnSessionSource") {
        throw "$prefix known short sample did not execute packaged B256"
    }
    $referenceSamples = [long]::Parse($markerFields["$prefix.referenceSamples"], [Globalization.CultureInfo]::InvariantCulture)
    $candidateSamples = [long]::Parse($markerFields["$prefix.candidateSamples"], [Globalization.CultureInfo]::InvariantCulture)
    $cpuElapsedMs = [long]::Parse($markerFields["$prefix.cpuElapsedMs"], [Globalization.CultureInfo]::InvariantCulture)
    $qnnElapsedMs = [long]::Parse($markerFields["$prefix.qnnElapsedMs"], [Globalization.CultureInfo]::InvariantCulture)
    $qnnMaxGeneratorRtf = [double]::Parse($markerFields["$prefix.qnnMaxGeneratorRtf"], [Globalization.CultureInfo]::InvariantCulture)
    $durationDelta = [double]::Parse($markerFields["$prefix.durationDelta"], [Globalization.CultureInfo]::InvariantCulture)
    $peak = [int]::Parse($markerFields["$prefix.peak"], [Globalization.CultureInfo]::InvariantCulture)
    $nrmse = [double]::Parse($markerFields["$prefix.nrmse"], [Globalization.CultureInfo]::InvariantCulture)
    $correlation = [double]::Parse($markerFields["$prefix.correlation"], [Globalization.CultureInfo]::InvariantCulture)
    $roughnessRatio = [double]::Parse($markerFields["$prefix.roughnessRatio"], [Globalization.CultureInfo]::InvariantCulture)
    $computedDurationDelta = [Math]::Abs($candidateSamples - $referenceSamples) / [Math]::Max($referenceSamples, [long]1)
    if ($referenceSamples -le 10000 -or $candidateSamples -le 10000) { throw "$prefix produced too little audio" }
    if ($cpuElapsedMs -lt 0 -or $qnnElapsedMs -lt 0) { throw "$prefix contains an invalid elapsed time" }
    if ([double]::IsNaN($qnnMaxGeneratorRtf) -or [double]::IsInfinity($qnnMaxGeneratorRtf) -or $qnnMaxGeneratorRtf -lt 0) {
        throw "$prefix contains an invalid QNN generator RTF: $qnnMaxGeneratorRtf"
    }
    if ([Math]::Abs($computedDurationDelta - $durationDelta) -gt 0.000000002 -or $durationDelta -gt 0.03) {
        throw "$prefix duration metric is invalid: $durationDelta"
    }
    if ($peak -ge [int](32767 * 0.98)) { throw "$prefix clipping gate failed: peak=$peak" }
    if ([double]::IsNaN($nrmse) -or [double]::IsInfinity($nrmse) -or $nrmse -gt 0.12) {
        throw "$prefix NRMSE gate failed: $nrmse"
    }
    if ([double]::IsNaN($correlation) -or [double]::IsInfinity($correlation) -or $correlation -lt 0.985) {
        throw "$prefix correlation gate failed: $correlation"
    }
    if ([double]::IsNaN($roughnessRatio) -or [double]::IsInfinity($roughnessRatio) -or $roughnessRatio -gt 1.25) {
        throw "$prefix roughness gate failed: $roughnessRatio"
    }
}

# A packaged-context run must not generate/copy a source-JIT cache or leave a
# partial file. Preserve unrelated legacy files and reject only new paths.
$privateFilesAfter = @(Get-AppPrivateFiles)
$newPrivateFiles = @($privateFilesAfter | Where-Object { $privateFilesBefore -cnotcontains $_ })
$unexpectedContextFiles = @($newPrivateFiles | Where-Object {
    $_ -match '(?i)(\.part$|[._]ctx\.onnx$|context[^/\\]*\.onnx$|qnn[^/\\]*(?:context|\.bin$))'
})
if ($unexpectedContextFiles.Count -ne 0) {
    throw "Packaged AOT validation generated a context/partial file: $($unexpectedContextFiles -join ', ')"
}
$currentFallbackFilesAfter = @($privateFilesAfter | Where-Object { $currentFallbackFiles -ccontains $_ })
if ($currentFallbackFilesAfter.Count -ne 0) {
    throw "A packaged context was copied instead of directly mapped: $($currentFallbackFilesAfter -join ', ')"
}

$resolvedSerial = ((& $adb @adbArguments get-serialno) -join "`n").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolvedSerial)) {
    throw 'Unable to resolve the validated device serial.'
}
$receiptLines = [Collections.Generic.List[string]]::new()
[void]$receiptLines.Add('receiptVersion=5')
[void]$receiptLines.Add('result=PASSED')
[void]$receiptLines.Add("hostAudioGateSha256=$hostGateSha256")
[void]$receiptLines.Add("apkSha256=$apkSha256")
[void]$receiptLines.Add("instrumentationApkSha256=$testApkSha256")
[void]$receiptLines.Add("deviceSerial=$resolvedSerial")
[void]$receiptLines.Add("validationStartedUtc=$($testStartedUtc.ToString('o'))")
[void]$receiptLines.Add("validationEndedUtc=$($testEndedUtc.ToString('o'))")
foreach ($key in @($markerFields.Keys | Sort-Object)) {
    [void]$receiptLines.Add("device.$key=$($markerFields[$key])")
}
[IO.File]::WriteAllLines($receiptPartPath, $receiptLines, [Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $receiptPartPath -Destination $receiptPath -Force
Write-Output "QNN host and physical-device gates passed; strict receipt: $receiptPath"
