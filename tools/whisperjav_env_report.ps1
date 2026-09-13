# whisperjav_env_report.ps1 — runs whisperjav_env_report.py with the Python that runs WhisperJAV.
#
# Usage (PowerShell, from the folder holding both files):
#     .\whisperjav_env_report.ps1            # report only
#     .\whisperjav_env_report.ps1 -Probe     # also the 30-second GPU probe
#
# If PowerShell refuses to run scripts, start it with:
#     powershell -ExecutionPolicy Bypass -File .\whisperjav_env_report.ps1
#
# If Windows blocks a downloaded script, run once:  Unblock-File .\whisperjav_env_report.ps1
#
# It looks for python.exe in this order: a WHISPERJAV_PYTHON environment variable (explicit
# override always wins), the folder this script is in, the WhisperJAV installer location
# (%LOCALAPPDATA%\WhisperJAV), then whatever "python" is on PATH.

param(
    [switch]$Probe
)

$here = $PSScriptRoot
$script = Join-Path $here "whisperjav_env_report.py"
if (-not (Test-Path $script)) {
    Write-Host "whisperjav_env_report.py must be in the same folder as this script." -ForegroundColor Red
    exit 1
}

$candidates = @(
    $env:WHISPERJAV_PYTHON,
    (Join-Path $here "python.exe"),
    (Join-Path $env:LOCALAPPDATA "WhisperJAV\python.exe")
)
$python = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) { $python = $c; break }
}
if (-not $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}
if (-not $python) {
    Write-Host "No python.exe found. Set WHISPERJAV_PYTHON to the python.exe inside your WhisperJAV installation." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $python"
$scriptArgs = @($script)
if ($Probe) { $scriptArgs += "--probe" }
& $python @scriptArgs
exit $LASTEXITCODE
