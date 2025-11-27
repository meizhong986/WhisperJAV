@echo off
REM ========================================
REM Build WhisperJAV wheel for installer
REM ========================================
REM
REM This script builds a wheel from the local source code
REM to be bundled with the installer
REM
REM Can be called from any directory - uses %~dp0 for navigation
REM

SETLOCAL EnableDelayedExpansion

echo Building WhisperJAV wheel from local source...
echo.

REM Get the directory where this script lives (installer/)
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Calculate repo root (parent of installer/)
for %%I in ("%SCRIPT_DIR%\..") do set REPO_ROOT=%%~fI

echo   Script directory: %SCRIPT_DIR%
echo   Repository root: %REPO_ROOT%
echo.

REM Save current directory
set ORIGINAL_DIR=%CD%

REM Go to repository root
cd /d "%REPO_ROOT%"

REM Verify we're in the right place
if not exist "setup.py" (
    echo ERROR: setup.py not found in %REPO_ROOT%
    echo Cannot build wheel - not in repository root
    cd /d "%ORIGINAL_DIR%"
    exit /b 1
)

REM Clean old builds
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build

REM Build wheel
python -m pip install --upgrade build >nul 2>nul
python -m build --wheel

if errorlevel 1 (
    echo ERROR: Wheel build failed!
    cd /d "%ORIGINAL_DIR%"
    exit /b 1
)

REM Verify wheel was created
set WHEEL_CREATED=0
for %%f in (dist\whisperjav-*.whl) do (
    set WHEEL_CREATED=1
    set WHEEL_PATH=%%f
    set WHEEL_NAME=%%~nxf
)

if !WHEEL_CREATED!==0 (
    echo ERROR: Wheel build failed - no wheel file created!
    cd /d "%ORIGINAL_DIR%"
    exit /b 1
)

echo.
echo   Built: !WHEEL_NAME!

REM Copy wheel to installer directory
copy "!WHEEL_PATH!" "%SCRIPT_DIR%\" >nul
echo   Copied to: %SCRIPT_DIR%\!WHEEL_NAME!

REM Also copy to generated/ if it exists
if exist "%SCRIPT_DIR%\generated" (
    copy "!WHEEL_PATH!" "%SCRIPT_DIR%\generated\" >nul
    echo   Copied to: %SCRIPT_DIR%\generated\!WHEEL_NAME!
)

REM Return to original directory
cd /d "%ORIGINAL_DIR%"

echo.
echo Wheel build complete!
exit /b 0
