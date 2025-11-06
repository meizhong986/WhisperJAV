<#
Build WhisperJAV CLI executable with PyInstaller

Requirements:
- Run from PowerShell (pwsh) with the 'WJ' conda env activated
- PyInstaller installed in the env

Usage:
  conda activate WJ
  pwsh -File installer/build_exe_cli.ps1
#>

param(
  [switch] $Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host '========================================'
Write-Host 'WhisperJAV CLI Builder'
Write-Host '========================================'
Write-Host ''

# Ensure we are in the repo root (script lives in installer/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# Verify conda environment
if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -ne 'WJ') {
  Write-Error "This script must be run inside the 'WJ' conda environment. Current: '$($env:CONDA_DEFAULT_ENV)'.`nActivate with: conda activate WJ"
}

# Verify PyInstaller
try {
  pyinstaller --version | Out-Null
} catch {
  Write-Error 'PyInstaller not found. Install with: pip install pyinstaller'
}

if ($Clean) {
  Write-Host '[1/4] Cleaning previous builds...'
  if (Test-Path "$RepoRoot\\build") { Remove-Item "$RepoRoot\\build" -Recurse -Force }
  if (Test-Path "$RepoRoot\\dist\\whisperjav") { Remove-Item "$RepoRoot\\dist\\whisperjav" -Recurse -Force }
}

Write-Host '[2/4] Building executable (CLI)...'
pyinstaller "$ScriptDir/whisperjav-cli.spec" --clean --noconfirm

Write-Host ''
if (-not (Test-Path "$RepoRoot\\dist\\whisperjav\\whisperjav.exe")) {
  Write-Error 'Build failed: dist/whisperjav/whisperjav.exe was not produced.'
}

# Optional: copy ffmpeg.exe from conda env if present
Write-Host '[3/4] Checking for ffmpeg.exe in environment...'
$CandidatePaths = @(
  Join-Path $env:CONDA_PREFIX 'Library/bin/ffmpeg.exe'),
  (Join-Path $env:CONDA_PREFIX 'Scripts/ffmpeg.exe'),
  (Join-Path $env:CONDA_PREFIX 'ffmpeg.exe')
foreach ($p in $CandidatePaths) {
  if (Test-Path $p) {
    Copy-Item $p "$RepoRoot/dist/whisperjav/" -Force
    Write-Host "Copied ffmpeg.exe from: $p"
    break
  }
}

Write-Host '[4/4] Done.'
Write-Host ''
Write-Host 'Executable location:'
Write-Host "  $RepoRoot/dist/whisperjav/whisperjav.exe"
Write-Host ''
Write-Host 'Tip: Ensure NVIDIA drivers and CUDA-enabled PyTorch are installed.'
