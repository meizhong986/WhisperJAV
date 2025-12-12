@echo off
REM ===============================================================================
REM WhisperJAV Upgrade Script
REM ===============================================================================
REM
REM This script upgrades your WhisperJAV installation to the latest version
REM without re-downloading large packages like PyTorch (~2.5GB).
REM
REM Usage: Just double-click this file!
REM
REM What it does:
REM   - Updates WhisperJAV to the latest version from GitHub
REM   - Installs any new dependencies
REM   - Updates your desktop shortcut with the new version
REM   - Preserves your AI models and settings
REM
REM ===============================================================================

echo.
echo ===============================================================================
echo                     WhisperJAV Upgrade Launcher
echo ===============================================================================
echo.

REM Detect installation directory
REM Priority 1: Script's own location (if packaged with installer)
REM Priority 2: Default location as fallback

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Check if python.exe exists in script's directory (we're in the install folder)
if exist "%SCRIPT_DIR%\python.exe" (
    set "INSTALL_DIR=%SCRIPT_DIR%"
    echo Found WhisperJAV at script location: %INSTALL_DIR%
    goto :found_install
)

REM Fallback: Check default installation location
set "INSTALL_DIR=%LOCALAPPDATA%\WhisperJAV"
if exist "%INSTALL_DIR%\python.exe" (
    echo Found WhisperJAV at default location: %INSTALL_DIR%
    goto :found_install
)

REM Not found in either location
echo ERROR: WhisperJAV installation not found.
echo.
echo Checked locations:
echo   1. Script location: %SCRIPT_DIR%
echo   2. Default location: %LOCALAPPDATA%\WhisperJAV
echo.
echo Please run this script from your WhisperJAV installation folder,
echo or reinstall WhisperJAV using the Windows installer.
echo.
pause
exit /b 1

:found_install
echo.

REM Check if upgrade script exists in same directory as this bat file
set "SCRIPT_DIR=%~dp0"
set "UPGRADE_SCRIPT=%SCRIPT_DIR%upgrade_whisperjav.py"

if exist "%UPGRADE_SCRIPT%" (
    echo Using local upgrade script: %UPGRADE_SCRIPT%
    echo.
    "%INSTALL_DIR%\python.exe" "%UPGRADE_SCRIPT%"
) else (
    echo Downloading upgrade script from GitHub...
    echo.

    REM Use PowerShell to download the script
    powershell -Command "& {Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/meizhong986/whisperjav/main/installer/upgrade_whisperjav.py' -OutFile '%TEMP%\upgrade_whisperjav.py'}"

    if not exist "%TEMP%\upgrade_whisperjav.py" (
        echo ERROR: Failed to download upgrade script.
        echo Please check your internet connection.
        echo.
        pause
        exit /b 1
    )

    "%INSTALL_DIR%\python.exe" "%TEMP%\upgrade_whisperjav.py"

    REM Clean up
    del "%TEMP%\upgrade_whisperjav.py" 2>nul
)

echo.
pause
