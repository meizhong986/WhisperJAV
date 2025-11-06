<#
Build WhisperJAV GUI executable with PyInstaller

Requirements:
- Run from PowerShell (pwsh) with the 'WJ' conda env activated
- PyInstaller installed in the env

Usage:
  conda activate WJ
  pwsh -File installer/build_exe_gui.ps1
#>

param(
  [switch] $Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host '========================================'
Write-Host 'WhisperJAV GUI Builder'
Write-Host '========================================'
Write-Host ''

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -ne 'WJ') {
  Write-Error "This script must be run inside the 'WJ' conda environment. Current: '$($env:CONDA_DEFAULT_ENV)'.`nActivate with: conda activate WJ"
}

try {
  pyinstaller --version | Out-Null
} catch {
  Write-Error 'PyInstaller not found. Install with: pip install pyinstaller'
}

if ($Clean) {
  Write-Host '[1/4] Cleaning previous builds...'
  if (Test-Path "$RepoRoot\\build") { Remove-Item "$RepoRoot\\build" -Recurse -Force }
  if (Test-Path "$RepoRoot\\dist\\whisperjav-gui") { Remove-Item "$RepoRoot\\dist\\whisperjav-gui" -Recurse -Force }
}

Write-Host '[2/4] Building executable (GUI)...'
pyinstaller "$ScriptDir/whisperjav-gui.spec" --clean --noconfirm

if (-not (Test-Path "$RepoRoot\\dist\\whisperjav-gui\\whisperjav-gui.exe")) {
  Write-Error 'Build failed: dist/whisperjav-gui/whisperjav-gui.exe was not produced.'
}

# Optional: copy ffmpeg.exe from conda env if present
Write-Host '[3/4] Checking for ffmpeg.exe in environment...'
$CandidatePaths = @(
  Join-Path $env:CONDA_PREFIX 'Library/bin/ffmpeg.exe'),
  (Join-Path $env:CONDA_PREFIX 'Scripts/ffmpeg.exe'),
  (Join-Path $env:CONDA_PREFIX 'ffmpeg.exe')
foreach ($p in $CandidatePaths) {
  if (Test-Path $p) {
    Copy-Item $p "$RepoRoot/dist/whisperjav-gui/" -Force
    Write-Host "Copied ffmpeg.exe from: $p"
    break
  }
}

Write-Host '[4/4] Done.'
Write-Host ''
Write-Host 'Executable location:'
Write-Host "  $RepoRoot/dist/whisperjav-gui/whisperjav-gui.exe"
Write-Host ''
Write-Host 'Note: WebView2 runtime must be installed on Windows for the GUI.'
