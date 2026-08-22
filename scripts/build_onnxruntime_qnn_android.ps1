[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OrtSource,
    [Parameter(Mandatory = $true)][string]$QnnHome,
    [Parameter(Mandatory = $true)][string]$AndroidNdk,
    [Parameter(Mandatory = $true)][string]$QualcommRuntimeAar,
    [string]$AndroidSdk = (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
    [string]$JavaHome = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot',
    [string]$PythonExecutable = '',
    [switch]$ConfirmQualcommRuntimeRedistribution
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $ConfirmQualcommRuntimeRedistribution) {
    throw 'Pass -ConfirmQualcommRuntimeRedistribution only after confirming your Qualcomm license permits packaging the supplied runtime AAR.'
}
$OrtSource = (Resolve-Path -LiteralPath $OrtSource).Path
$QnnHome = (Resolve-Path -LiteralPath $QnnHome).Path
$AndroidSdk = (Resolve-Path -LiteralPath $AndroidSdk).Path
$AndroidNdk = (Resolve-Path -LiteralPath $AndroidNdk).Path
$JavaHome = (Resolve-Path -LiteralPath $JavaHome).Path
$QualcommRuntimeAar = (Resolve-Path -LiteralPath $QualcommRuntimeAar).Path

# Prefer the pinned virtual environment created beside the checked-out ORT source.
# build.bat invokes `python` by name, so make this interpreter first on PATH.
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $pinnedPython = Join-Path $OrtSource '.venv-build\Scripts\python.exe'
    if (Test-Path -LiteralPath $pinnedPython -PathType Leaf) { $PythonExecutable = $pinnedPython }
}
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $PythonExecutable = $pythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($PythonExecutable) -or -not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python was not found. Create OrtSource\.venv-build with the pinned ORT Python dependencies or pass -PythonExecutable.'
}
$PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
$env:PATH = "$(Split-Path -Parent $PythonExecutable);$env:PATH"
& $PythonExecutable -c 'import onnx; print(onnx.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'The selected Python does not have the required ONNX package.' }

# Android Studio's SDK-managed CMake bundle includes Ninja. Prefer it so the
# build does not depend on machine-wide PATH changes.
$sdkCmakeBins = @(
    Get-ChildItem -LiteralPath (Join-Path $AndroidSdk 'cmake') -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName 'bin' } |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_ 'cmake.exe') -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $_ 'ninja.exe') -PathType Leaf)
        }
)
if ($sdkCmakeBins.Count -gt 0) { $env:PATH = "$($sdkCmakeBins[0]);$env:PATH" }

if ((Get-Content -Raw (Join-Path $OrtSource 'VERSION_NUMBER')).Trim() -ne '1.20.0') {
    throw 'OrtSource must be the official ONNX Runtime v1.20.0 source tree.'
}
foreach ($required in @(
    (Join-Path $OrtSource 'build.bat'),
    (Join-Path $QnnHome 'include\QNN\QnnInterface.h'),
    (Join-Path $AndroidNdk 'source.properties'),
    (Join-Path $JavaHome 'bin\java.exe')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing prerequisite: $required" }
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw 'CMake 3.28+ must be on PATH.' }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'Ninja must be on PATH.' }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$runtimeZip = [IO.Compression.ZipFile]::OpenRead($QualcommRuntimeAar)
try {
    $entries = $runtimeZip.Entries.FullName
    foreach ($library in @('libQnnHtp.so', 'libQnnSystem.so')) {
        if (-not ($entries -match "(^|/)$([regex]::Escape($library))$")) {
            throw "The supplied Qualcomm runtime AAR does not contain $library."
        }
    }
} finally {
    $runtimeZip.Dispose()
}

$env:JAVA_HOME = $JavaHome
$buildDirectory = Join-Path $OrtSource 'build\Android-QNN-1.20.0'
& (Join-Path $OrtSource 'build.bat') `
    --build_java --build_shared_lib --android --config Release --parallel `
    --use_qnn static_lib --qnn_home $QnnHome `
    --android_sdk_path $AndroidSdk --android_ndk_path $AndroidNdk `
    --android_abi arm64-v8a --android_api 26 --cmake_generator Ninja `
    --build_dir $buildDirectory
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime QNN build failed with exit code $LASTEXITCODE" }

$builtAar = Get-ChildItem -LiteralPath $buildDirectory -Recurse -Filter 'onnxruntime-release.aar' -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $builtAar) { throw "The build succeeded but no onnxruntime-release.aar was found under $buildDirectory" }

$projectRoot = Split-Path -Parent $PSScriptRoot
$libs = Join-Path $projectRoot 'app\libs'
New-Item -ItemType Directory -Path $libs -Force | Out-Null
Copy-Item -LiteralPath $builtAar.FullName -Destination (Join-Path $libs 'onnxruntime-android-qnn-1.20.0.aar') -Force
Copy-Item -LiteralPath $QualcommRuntimeAar -Destination (Join-Path $libs 'qnn-runtime.aar') -Force
Write-Output 'Staged the custom ORT QNN AAR and the caller-supplied licensed Qualcomm runtime AAR.'
Write-Output 'QNN remains off by default until scripts\validate_qnn_on_device.ps1 creates a matching receipt.'
