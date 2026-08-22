param(
    [Parameter(Mandatory = $true)]
    [string] $SourceAar
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = (Resolve-Path -LiteralPath $SourceAar).Path
$expectedAarSha256 = "48F0AD8ACD0864D4DBA66DB283468C1D082E9D1B91B33D92F3CE40562E0C533D"
$actualAarSha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
if ($actualAarSha256 -ne $expectedAarSha256) {
    throw "Unexpected upstream QNN 2.4.0 AAR SHA-256: $actualAarSha256"
}

$jar = Join-Path $env:JAVA_HOME "bin\jar.exe"
if (-not (Test-Path -LiteralPath $jar -PathType Leaf)) {
    $jar = (Get-Command jar.exe -ErrorAction Stop).Source
}
$work = Join-Path $repoRoot "build\device-debug\qnn-2.4-samsung-compat-$([Guid]::NewGuid().ToString('N'))"
$outputDirectory = Join-Path $repoRoot "app\libs"
$output = Join-Path $outputDirectory "onnxruntime-android-qnn-2.4.0-samsung-sm8650.aar"
New-Item -ItemType Directory -Path $work -Force | Out-Null
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $output -Force

Push-Location $work
try {
    & $jar xf $source "jni/arm64-v8a/libonnxruntime_providers_qnn.so"
    if ($LASTEXITCODE -ne 0) { throw "Unable to extract the upstream QNN provider" }
    $provider = Join-Path $work "jni\arm64-v8a\libonnxruntime_providers_qnn.so"
    $bytes = [IO.File]::ReadAllBytes($provider)
    $original = [Text.Encoding]::ASCII.GetBytes("fastrpc-cdsp")
    $replacement = [Text.Encoding]::ASCII.GetBytes("adsprpc-smd`0")
    if ($original.Length -ne $replacement.Length) { throw "Compatibility probe byte lengths differ" }
    $matches = @()
    for ($offset = 0; $offset -le $bytes.Length - $original.Length; $offset++) {
        $matchesHere = $true
        for ($index = 0; $index -lt $original.Length; $index++) {
            if ($bytes[$offset + $index] -ne $original[$index]) { $matchesHere = $false; break }
        }
        if ($matchesHere) { $matches += $offset }
    }
    if ($matches.Count -ne 1) { throw "Expected one FastRPC device probe, found $($matches.Count)" }
    [Array]::Copy($replacement, 0, $bytes, $matches[0], $replacement.Length)
    [IO.File]::WriteAllBytes($provider, $bytes)
    & $jar uf $output "jni/arm64-v8a/libonnxruntime_providers_qnn.so"
    if ($LASTEXITCODE -ne 0) { throw "Unable to update the compatibility AAR" }
} finally {
    Pop-Location
}

$patchedAarSha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
Write-Output "Created $output"
Write-Output "upstreamAarSha256=$actualAarSha256"
Write-Output "patchedAarSha256=$patchedAarSha256"
Write-Output "probe=fastrpc-cdsp -> adsprpc-smd"
