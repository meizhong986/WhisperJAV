<#
Build both CLI and GUI executables for WhisperJAV

Usage:
  conda activate WJ
  pwsh -File installer/build_all.ps1 -Clean
#>

param(
  [switch] $Clean
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $ScriptDir)

if ($Clean) { $cleanArg = '-Clean' } else { $cleanArg = $null }

& pwsh -NoProfile -ExecutionPolicy Bypass -File "$ScriptDir/build_exe_cli.ps1" $cleanArg
& pwsh -NoProfile -ExecutionPolicy Bypass -File "$ScriptDir/build_exe_gui.ps1" $cleanArg

Write-Host ''
Write-Host 'Both builds finished. Outputs:'
Write-Host "  $(Resolve-Path ./dist/whisperjav/whisperjav.exe)"
Write-Host "  $(Resolve-Path ./dist/whisperjav-gui/whisperjav-gui.exe)"
