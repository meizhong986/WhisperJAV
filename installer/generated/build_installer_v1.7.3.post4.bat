@echo off
REM ===============================================================================
REM WhisperJAV v1.7.3.post4 Installer Build Script
REM ===============================================================================
REM
REM This script builds the conda-constructor installer for WhisperJAV v1.7.3.post4
REM
REM Prerequisites:
REM - Conda with constructor package installed:
REM   conda install constructor -c conda-forge
REM - All v1.7.3.post4 installer files must be present in installer/ directory
REM
REM Build Process:
REM 1. Check prerequisites (constructor, Python)
REM 2. Clean previous builds
REM 3. Run constructor to build installer
REM 4. Verify output and generate build report
REM
REM Output: WhisperJAV-1.7.3.post4-Windows-x86_64.exe
REM Logs: build_log_v1.7.3.post4.txt
REM

SETLOCAL EnableDelayedExpansion

REM ===== Configuration =====
set CONFIG_FILE=construct_v1.7.3.post4.yaml
set BUILD_LOG=build_log_v1.7.3.post4.txt
set VERSION=1.7.3.post4

echo ===============================================================================
echo                    WhisperJAV v1.7.3.post4 Installer Build
echo ===============================================================================
echo.

REM Initialize log file
echo WhisperJAV v1.7.3.post4 Installer Build Log > "%BUILD_LOG%"
echo Build started: %DATE% %TIME% >> "%BUILD_LOG%"
echo. >> "%BUILD_LOG%"

REM ===== Phase 1: Check Prerequisites =====
echo [Phase 1/4] Checking prerequisites...
echo [Phase 1/4] Checking prerequisites... >> "%BUILD_LOG%"

REM Check if constructor is installed
constructor --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Constructor not found or not working!
    echo ERROR: Constructor not found >> "%BUILD_LOG%"
    echo.
    echo Please install constructor:
    echo   conda install constructor -c conda-forge
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('constructor --version 2^>^&1') do set CONS_VER=%%a
echo   - Constructor: %CONS_VER%
echo   - Constructor: %CONS_VER% >> "%BUILD_LOG%"

REM Check if Python is available
python --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo ERROR: Python not found >> "%BUILD_LOG%"
    echo Please ensure Python is in your PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('python --version 2^>^&1') do set PY_VER=%%a
echo   - Python: %PY_VER%
echo   - Python: %PY_VER% >> "%BUILD_LOG%"

REM Check if config file exists
if not exist "%CONFIG_FILE%" (
    echo.
    echo ERROR: Configuration file not found: %CONFIG_FILE%
    echo ERROR: Missing %CONFIG_FILE% >> "%BUILD_LOG%"
    echo.
    echo Make sure you're running this script from the installer/generated/ directory.
    pause
    exit /b 1
)

echo   - Config file: %CONFIG_FILE% (found)
echo   - Config file: %CONFIG_FILE% >> "%BUILD_LOG%"

REM Check if wheel exists
if not exist "whisperjav-%VERSION%-py3-none-any.whl" (
    echo.
    echo ERROR: Wheel file not found: whisperjav-%VERSION%-py3-none-any.whl
    echo ERROR: Missing wheel >> "%BUILD_LOG%"
    echo.
    echo Build the wheel first:
    echo   python -m pip wheel . --no-deps -w installer/generated/
    pause
    exit /b 1
)
echo   - Wheel: whisperjav-%VERSION%-py3-none-any.whl (found)
echo.

REM ===== Phase 2: Clean Previous Builds =====
echo [Phase 2/4] Cleaning previous builds...
echo [Phase 2/4] Cleaning previous builds... >> "%BUILD_LOG%"

if exist build (
    echo   - Removing build/ directory...
    rmdir /s /q build
    echo   - Cleaned: build/ >> "%BUILD_LOG%"
)

if exist _build (
    echo   - Removing _build/ directory...
    rmdir /s /q _build
    echo   - Cleaned: _build/ >> "%BUILD_LOG%"
)

if exist WhisperJAV-%VERSION%-Windows-x86_64.exe (
    echo   - Removing old installer: WhisperJAV-%VERSION%-Windows-x86_64.exe
    del WhisperJAV-%VERSION%-Windows-x86_64.exe
    echo   - Cleaned: WhisperJAV-%VERSION%-Windows-x86_64.exe >> "%BUILD_LOG%"
)

echo   - Cleanup complete
echo.

REM ===== Phase 3: Build Installer =====
echo [Phase 3/4] Building installer with constructor...
echo [Phase 3/4] Building installer... >> "%BUILD_LOG%"
echo.
echo This will take 2-5 minutes. Please wait...
echo.

REM Build with verbose output
constructor . --config "%CONFIG_FILE%" -v

if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo                          BUILD FAILED!
    echo ===============================================================================
    echo.
    echo ERROR: Constructor build failed! >> "%BUILD_LOG%"
    echo Check the error messages above for details.
    echo.
    pause
    exit /b 1
)

echo.

REM ===== Phase 4: Verify Output and Generate Report =====
echo [Phase 4/4] Verifying build output...
echo [Phase 4/4] Verifying output... >> "%BUILD_LOG%"

set INSTALLER_NAME=WhisperJAV-%VERSION%-Windows-x86_64.exe

if not exist "%INSTALLER_NAME%" (
    echo.
    echo ERROR: Installer file was not created!
    echo ERROR: Output file missing: %INSTALLER_NAME% >> "%BUILD_LOG%"
    echo.
    pause
    exit /b 1
)

REM Get file size
for %%A in ("%INSTALLER_NAME%") do set FILESIZE=%%~zA
set /a FILESIZE_MB=!FILESIZE! / 1048576

echo   - Installer created: %INSTALLER_NAME%
echo   - File size: !FILESIZE_MB! MB
echo.

REM Log success
echo. >> "%BUILD_LOG%"
echo =============================================================================== >> "%BUILD_LOG%"
echo BUILD SUCCESSFUL >> "%BUILD_LOG%"
echo =============================================================================== >> "%BUILD_LOG%"
echo. >> "%BUILD_LOG%"
echo Output file: %INSTALLER_NAME% >> "%BUILD_LOG%"
echo File size: !FILESIZE_MB! MB >> "%BUILD_LOG%"
echo Build completed: %DATE% %TIME% >> "%BUILD_LOG%"
echo. >> "%BUILD_LOG%"

REM ===== Build Summary =====
echo ===============================================================================
echo                      BUILD COMPLETED SUCCESSFULLY!
echo ===============================================================================
echo.
echo Installer Details:
echo   - File: %INSTALLER_NAME%
echo   - Size: !FILESIZE_MB! MB
echo   - Version: %VERSION%
echo   - Build log: %BUILD_LOG%
echo.
echo Next Steps:
echo   1. Test the installer on a clean Windows VM
echo   2. Verify all features work correctly
echo   3. Distribute to users
echo.
echo ===============================================================================

pause
exit /b 0
