<#
Builds the compact, indexed Misaki English lexicons consumed by Android.

This is a build-host-only tool.  It does not ship Python, spaCy, torch, or a
network client in the application.  The input files are the four JSON lexicons
from the pinned Misaki revision documented in third_party/misaki/PROVENANCE.md.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MisakiRoot,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\app\src\main\assets')
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$validPhoneString = (@(65, 73, 79, 81, 87, 89) + (97..122) + @(593, 592, 594, 230, 596, 240, 331, 601, 603, 604, 609, 618, 633, 638, 643, 650, 652, 658, 660, 676, 679, 712, 716, 720, 952, 7498, 7547) | ForEach-Object { [char]$_ }) -join ''
Add-Type -AssemblyName System.Web.Extensions
$json = [System.Web.Script.Serialization.JavaScriptSerializer]::new()
$json.MaxJsonLength = [int]::MaxValue

function Resolve-Pronunciation([object]$value) {
    if ($value -is [string]) { return $value }
    if ($null -eq $value) { return $null }
    if ($value -is [System.Collections.IDictionary]) {
        foreach ($key in @('DEFAULT', 'None')) {
            if ($value.ContainsKey($key) -and $value[$key] -is [string]) { return $value[$key] }
        }
        foreach ($entry in $value.GetEnumerator()) {
            if ($entry.Value -is [string]) { return $entry.Value }
        }
    }
    return $null
}

function Add-Lexicon([hashtable]$destination, [string]$path) {
    # Windows PowerShell defaults UTF-8 files without a BOM to the ANSI code
    # page. Misaki's JSON contains IPA, so the source encoding must be explicit.
    $parsed = $json.DeserializeObject([System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8))
    foreach ($entry in $parsed.GetEnumerator()) {
        $pronunciation = Resolve-Pronunciation $entry.Value
        if ($null -ne $pronunciation -and $pronunciation.Length -gt 0) {
            foreach ($phone in [char[]]$pronunciation) {
                $codePoint = [int][char]$phone
                if ($validPhoneString.IndexOf([char]$codePoint) -lt 0) {
                    throw "Invalid or mojibake Misaki phone U+$($codePoint.ToString('X4')) for $($entry.Key) in $path"
                }
            }
            $destination[$entry.Key] = $pronunciation
        }
    }
}

function Add-MisakiCaseVariants([hashtable]$destination) {
    # Matches Misaki Lexicon.grow_dictionary: preserve intentional all-caps
    # entries, while making ordinary lower/title case spellings interchangeable.
    foreach ($entry in @($destination.GetEnumerator())) {
        $word = [string]$entry.Key
        if ($word -ceq $word.ToLowerInvariant()) {
            $title = [cultureinfo]::InvariantCulture.TextInfo.ToTitleCase($word)
            if ($word -cne $title -and -not $destination.ContainsKey($title)) { $destination[$title] = $entry.Value }
        } elseif ($word -ceq [cultureinfo]::InvariantCulture.TextInfo.ToTitleCase($word)) {
            $lower = $word.ToLowerInvariant()
            if (-not $destination.ContainsKey($lower)) { $destination[$lower] = $entry.Value }
        }
    }
}

function Write-MisakiLexicon([string]$Dialect) {
    $lexicon = @{}
    # Silver is deliberately loaded first: gold takes the same precedence as
    # Misaki's runtime lookup.
    Add-Lexicon $lexicon (Join-Path $MisakiRoot "misaki\data\${Dialect}_silver.json")
    Add-Lexicon $lexicon (Join-Path $MisakiRoot "misaki\data\${Dialect}_gold.json")
    Add-MisakiCaseVariants $lexicon
    # Do not use $lexicon.Keys: a real lexicon entry named "keys" shadows the
    # Hashtable property in Windows PowerShell's member-access adapter.
    $keys = [string[]]@($lexicon.GetEnumerator() | ForEach-Object { [string]$_.Key })
    [Array]::Sort($keys, [StringComparer]::Ordinal)

    $payload = [System.IO.MemoryStream]::new()
    $offsets = [System.Collections.Generic.List[int]]::new($keys.Count)
    foreach ($key in $keys) {
        $offsets.Add([int]$payload.Position)
        $record = "$key`t$($lexicon[$key])`n"
        $bytes = $utf8.GetBytes($record)
        $payload.Write($bytes, 0, $bytes.Length)
    }

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $output = Join-Path $OutputDirectory "misaki_en_${Dialect}.mlex"
    $stream = [System.IO.File]::Open($output, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        $writer = [System.IO.BinaryWriter]::new($stream, $utf8, $true)
        $writer.Write([byte[]][char[]]'MLEX')
        $writer.Write([int]$keys.Count)
        foreach ($offset in $offsets) { $writer.Write($offset) }
        $payload.Position = 0
        $payload.CopyTo($stream)
        $writer.Flush()
    } finally {
        $stream.Dispose()
        $payload.Dispose()
    }
    Write-Host "Built $output entries=$($keys.Count) bytes=$((Get-Item $output).Length)"
}

foreach ($dialect in @('us', 'gb')) { Write-MisakiLexicon $dialect }
