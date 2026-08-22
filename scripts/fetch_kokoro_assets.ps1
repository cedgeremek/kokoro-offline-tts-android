# Historical v0.19 asset/bootstrap helper retained for provenance and graph
# experiments. It does not fetch the v1.30 AOT release asset set described in
# docs/BUILDING.md.
param([switch]$Force)
$ErrorActionPreference = 'Stop'
$assets = Join-Path $PSScriptRoot '..\app\src\main\assets'
$voices = Join-Path $assets 'voices'
New-Item -ItemType Directory -Force -Path $voices | Out-Null
function Get-Asset([string]$url, [string]$path) { if ($Force -or -not (Test-Path $path)) { Write-Host "Downloading $(Split-Path $path -Leaf)"; curl.exe -fL --retry 3 -o $path $url; if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" } } }
Get-Asset 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.int8.onnx' (Join-Path $assets 'kokoro-v0_19.int8.onnx')
$base = 'https://raw.githubusercontent.com/puff-dayo/Kokoro-82M-Android/latest/app/src/main/res/raw/'
Get-Asset ($base + 'cmudict_ipa.dict') (Join-Path $assets 'cmudict_ipa.dict')
Get-Asset 'https://raw.githubusercontent.com/thewh1teagle/kokoro-onnx/main/src/kokoro_onnx/config.json' (Join-Path $assets 'vocab.json')
$voiceBundle = Join-Path $env:TEMP 'kokoro-voices-v1.0.bin'
Get-Asset 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' $voiceBundle
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($voiceBundle)
try {
    foreach ($entry in $archive.Entries) {
        if ($entry.Name -match '^(af|am|bf|bm)_.+\.npy$') {
            $target = Join-Path $voices $entry.Name
            if ($Force -or -not (Test-Path $target)) { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true) }
        }
    }
} finally { $archive.Dispose() }
$mobileModel = Join-Path $assets 'kokoro-v0_19.mobile2d.fp32.onnx'
if ($Force -or -not (Test-Path $mobileModel)) {
    $fp32Model = Join-Path $env:TEMP 'kokoro-v0_19.onnx'
    $mobile2d = Join-Path $env:TEMP 'kokoro-v0_19.mobile2d.onnx'
    Get-Asset 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx' $fp32Model
    & py -3.10 (Join-Path $PSScriptRoot 'optimize_kokoro_mobile.py') --input $fp32Model --assets $assets --transformed $mobile2d --output $mobileModel --fp32
    if ($LASTEXITCODE -ne 0) { throw 'Mobile Kokoro graph preparation failed (Python onnx and onnxruntime 1.20+ are required)' }
    & py -3.10 (Join-Path $PSScriptRoot 'validate_audio_quality.py') --reference $fp32Model --candidate $mobileModel --assets $assets
    if ($LASTEXITCODE -ne 0) { throw 'The regenerated mobile model failed the audio-fidelity gate' }
}
Write-Host "Assets ready in $assets"
