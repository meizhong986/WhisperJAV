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

REM Find WhisperJAV installation
set "INSTALL_DIR=%LOCALAPPDATA%\WhisperJAV"

if not exist "%INSTALL_DIR%\python.exe" (
    echo ERROR: WhisperJAV installation not found at:
    echo        %INSTALL_DIR%
    echo.
    echo Please make sure WhisperJAV is installed via the Windows installer.
    echo.
    pause
    exit /b 1
)

echo Found WhisperJAV at: %INSTALL_DIR%
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
