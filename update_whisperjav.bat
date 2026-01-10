@echo off
REM ==============================================================================
REM WhisperJAV Update Script
REM ==============================================================================
REM
REM This script updates WhisperJAV to the latest version from GitHub.
REM It automatically detects your installation folder using:
REM   1. Windows Registry (from installer)
REM   2. Script's own directory (if placed in install folder)
REM   3. Default location (%LOCALAPPDATA%\WhisperJAV)
REM
REM Usage: Simply double-click this script or run from Command Prompt.
REM
REM ==============================================================================

setlocal EnableDelayedExpansion

echo.
echo ==============================================================
echo   WhisperJAV Update Script
echo ==============================================================
echo.

REM Initialize variables
set "WJPATH="
set "DISCOVERY_METHOD="

REM ==============================================================
REM Step 1: Try to find installation path from Windows Registry
REM ==============================================================
echo [Step 1/4] Searching for WhisperJAV installation...
echo.

REM Try HKCU first (user installation)
for /f "tokens=2*" %%a in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperJAV" /v "InstallLocation" 2^>nul') do (
    set "WJPATH=%%b"
    set "DISCOVERY_METHOD=Windows Registry (User)"
)

REM Try HKLM if not found (all users installation)
if not defined WJPATH (
    for /f "tokens=2*" %%a in ('reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperJAV" /v "InstallLocation" 2^>nul') do (
        set "WJPATH=%%b"
        set "DISCOVERY_METHOD=Windows Registry (All Users)"
    )
)

REM Try UninstallString as fallback (extract directory from uninstaller path)
if not defined WJPATH (
    for /f "tokens=2*" %%a in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\WhisperJAV" /v "UninstallString" 2^>nul') do (
        set "UNINST_PATH=%%b"
        REM Remove quotes and get directory
        set "UNINST_PATH=!UNINST_PATH:"=!"
        for %%i in ("!UNINST_PATH!") do set "WJPATH=%%~dpi"
        REM Remove trailing backslash
        if "!WJPATH:~-1!"=="\" set "WJPATH=!WJPATH:~0,-1!"
        set "DISCOVERY_METHOD=Windows Registry (Uninstaller Path)"
    )
)

REM ==============================================================
REM Step 2: Fallback to script's own directory
REM ==============================================================
if not defined WJPATH (
    REM Check if this script is in a WhisperJAV installation directory
    set "SCRIPT_DIR=%~dp0"
    REM Remove trailing backslash
    if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

    if exist "!SCRIPT_DIR!\python.exe" (
        set "WJPATH=!SCRIPT_DIR!"
        set "DISCOVERY_METHOD=Script Location"
    ) else if exist "!SCRIPT_DIR!\Scripts\pip.exe" (
        set "WJPATH=!SCRIPT_DIR!"
        set "DISCOVERY_METHOD=Script Location"
    )
)

REM ==============================================================
REM Step 3: Fallback to default location
REM ==============================================================
if not defined WJPATH (
    set "WJPATH=%LOCALAPPDATA%\WhisperJAV"
    set "DISCOVERY_METHOD=Default Location"
)

REM ==============================================================
REM Display discovered path prominently
REM ==============================================================
echo   Discovery method: !DISCOVERY_METHOD!
echo.
echo   ============================================================
echo   INSTALLATION FOLDER DETECTED:
echo.
echo      !WJPATH!
echo.
echo   ============================================================
echo.

REM ==============================================================
REM Validate the installation path
REM ==============================================================
echo [Step 2/4] Validating installation...
echo.

set "VALIDATION_ERRORS=0"

REM Check python.exe
if exist "!WJPATH!\python.exe" (
    echo   [OK] python.exe found
) else (
    echo   [!!] python.exe NOT FOUND
    set /a VALIDATION_ERRORS+=1
)

REM Check pip
if exist "!WJPATH!\Scripts\pip.exe" (
    echo   [OK] Scripts\pip.exe found
) else (
    echo   [!!] Scripts\pip.exe NOT FOUND
    set /a VALIDATION_ERRORS+=1
)

REM Check git
if exist "!WJPATH!\Library\bin\git.exe" (
    echo   [OK] Library\bin\git.exe found
) else (
    echo   [!!] Library\bin\git.exe NOT FOUND
    set /a VALIDATION_ERRORS+=1
)

echo.

if !VALIDATION_ERRORS! GTR 0 (
    echo   ============================================================
    echo   ERROR: This does not appear to be a valid WhisperJAV
    echo          installation folder. !VALIDATION_ERRORS! component(s) missing.
    echo.
    echo   Please check:
    echo   - Did you install WhisperJAV using the official installer?
    echo   - Is the installation folder correct?
    echo.
    echo   You can also manually specify the path by editing this script
    echo   or by running pip directly from your installation folder.
    echo   ============================================================
    echo.
    pause
    exit /b 1
)

echo   All components validated successfully!
echo.

REM ==============================================================
REM Confirm with user before proceeding
REM ==============================================================
echo [Step 3/4] Ready to update...
echo.
echo   The update will:
echo   - Download the latest WhisperJAV from GitHub
echo   - Install it to the folder shown above
echo   - This may take 1-2 minutes depending on your connection
echo.
echo   ============================================================
echo   Please verify the installation folder above is correct.
echo   ============================================================
echo.
echo   Press any key to start the update, or Ctrl+C to cancel...
pause >nul
echo.

REM ==============================================================
REM Set PATH and run update
REM ==============================================================
echo [Step 4/4] Updating WhisperJAV...
echo.

REM Temporarily add required directories to PATH
set "PATH=!WJPATH!\Library\bin;!WJPATH!\Scripts;!WJPATH!;%PATH%"

echo   Running: pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git
echo.
echo   ------------------------------------------------------------

"!WJPATH!\Scripts\pip.exe" install -U --no-deps git+https://github.com/meizhong986/whisperjav.git

if !ERRORLEVEL! EQU 0 (
    echo   ------------------------------------------------------------
    echo.
    echo   ============================================================
    echo   SUCCESS! WhisperJAV has been updated to the latest version.
    echo   ============================================================
    echo.
    echo   Please restart WhisperJAV to use the updated version.
    echo.
) else (
    echo   ------------------------------------------------------------
    echo.
    echo   ============================================================
    echo   ERROR: Update failed. Please check the error messages above.
    echo   ============================================================
    echo.
    echo   Common issues:
    echo   - Network connection problems
    echo   - GitHub rate limiting (try again in a few minutes)
    echo   - Firewall blocking git connections
    echo.
)

pause
