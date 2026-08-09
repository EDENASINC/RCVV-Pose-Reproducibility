param(
    [Parameter(Mandatory=$true)]
    [string]$MapCsv
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Rows = Import-Csv $MapCsv

if ($Rows.Count -ne 72) {
    throw "Expected 72 checkpoint rows, found $($Rows.Count)."
}

foreach ($Row in $Rows) {
    if ([string]::IsNullOrWhiteSpace($Row.source_path)) {
        throw "Missing source_path for $($Row.logical_id). Fill the CSV before importing."
    }
    $Source = [System.IO.Path]::GetFullPath($Row.source_path)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Source checkpoint not found: $Source"
    }
    $Destination = Join-Path $RepoRoot ($Row.destination_path -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
    Write-Host "$($Row.logical_id)  $Hash"
}

Write-Host "Imported 72 checkpoints. Run: python tools/release_preflight.py --mode release"
