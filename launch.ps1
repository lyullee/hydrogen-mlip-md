param([int]$Port = 8511, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Runtime = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$AppPath = Join-Path $PSScriptRoot 'hydrogen_lab/app.py'
$Url = "http://127.0.0.1:$Port"
$Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Listener) {
    $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($Listener.OwningProcess)"
    if ($ExistingProcess.CommandLine -notmatch [regex]::Escape($AppPath)) {
        throw "Port $Port is used by another application. Choose a different -Port."
    }
    Write-Host "Hydrogen Lab is already running: $Url"
    if (-not $NoBrowser) { Start-Process $Url }
    return
}
$NeedsSetup = -not (Test-Path -LiteralPath $Runtime)
if (-not $NeedsSetup) {
    & $Runtime -c "import streamlit, ase, torch, chgnet" 2>$null
    $NeedsSetup = $LASTEXITCODE -ne 0
}
if ($NeedsSetup -or -not (Test-Path -LiteralPath 'models/chgnet_0.3.0.pth.tar')) {
    & (Join-Path $PSScriptRoot 'setup.ps1') -Backend chgnet -DownloadCheckpoint
}
$LogDirectory = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$OutputLog = Join-Path $LogDirectory 'server-out.log'
$ErrorLog = Join-Path $LogDirectory 'server-error.log'
$Arguments = @('-m', 'streamlit', 'run', ('"{0}"' -f $AppPath),
    '--global.developmentMode', 'false', '--server.address', '127.0.0.1',
    '--server.port', [string]$Port, '--server.headless', 'true',
    '--server.showEmailPrompt', 'false', '--browser.gatherUsageStats', 'false')
$ServerProcess = Start-Process -FilePath $Runtime -ArgumentList $Arguments -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $OutputLog -RedirectStandardError $ErrorLog
for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    if ($ServerProcess.HasExited) {
        $Details = Get-Content -LiteralPath $ErrorLog -Tail 20 -ErrorAction SilentlyContinue
        throw "Server exited. $Details"
    }
    try {
        $Health = Invoke-WebRequest "$Url/_stcore/health" -UseBasicParsing -TimeoutSec 2
        if ($Health.StatusCode -eq 200) {
            Write-Host "Hydrogen Lab is ready: $Url"
            if (-not $NoBrowser) { Start-Process $Url }
            return
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw "Startup timed out. Check $ErrorLog"
