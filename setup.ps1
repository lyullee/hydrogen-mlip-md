param(
    [ValidateSet('chgnet', 'mace', 'none')]
    [string]$Backend = 'chgnet',
    [switch]$DownloadCheckpoint
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    $BundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $BundledPython) {
        & $BundledPython -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required.' }
}
& '.venv/Scripts/python.exe' -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
$PackageSpec = if ($Backend -eq 'none') { '.' } else { ".[${Backend}]" }
& '.venv/Scripts/python.exe' -m pip install -e $PackageSpec
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
if ($DownloadCheckpoint -and $Backend -ne 'none') {
    & '.venv/Scripts/python.exe' -m hydrogen_lab.cli fetch-model --model $Backend --out models
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed. See upstream URL and hash in docs/METHODS.md.' }
}
Write-Host 'Ready. Run .\launch.ps1 to open Hydrogen Lab.'
