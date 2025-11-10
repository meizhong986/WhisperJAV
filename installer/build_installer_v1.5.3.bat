@echo off
REM ===============================================================================
REM WhisperJAV v1.5.3 Installer Build Script
REM ===============================================================================
REM
REM This script builds the conda-constructor installer for WhisperJAV v1.5.3
REM
REM Prerequisites:
REM - Conda with constructor package installed:
REM   conda install constructor -c conda-forge
REM - All v1.5.3 installer files must be present in installer/ directory
REM
REM Build Process:
REM 1. Check prerequisites (constructor, Python)
REM 2. Validate installer configuration (via validate_installer_v1.5.3.py)
REM 3. Clean previous v1.5.3 builds
REM 4. Run constructor to build installer
REM 5. Verify output and generate build report
REM
REM Output: WhisperJAV-1.5.3-Windows-x86_64.exe
REM Logs: build_log_v1.5.3.txt
REM

SETLOCAL EnableDelayedExpansion

REM ===== Configuration =====
set CONFIG_FILE=construct_v1.5.3.yaml
set VALIDATOR=validate_installer_v1.5.3.py
set BUILD_LOG=build_log_v1.5.3.txt
set VERSION=1.5.3

echo ===============================================================================
echo                    WhisperJAV v1.5.3 Installer Build
echo ===============================================================================
echo.

REM Initialize log file
echo WhisperJAV v1.5.3 Installer Build Log > "%BUILD_LOG%"
echo Build started: %DATE% %TIME% >> "%BUILD_LOG%"
echo. >> "%BUILD_LOG%"

REM ===== Phase 1: Check Prerequisites =====
echo [Phase 1/5] Checking prerequisites...
echo [Phase 1/5] Checking prerequisites... >> "%BUILD_LOG%"

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
    echo If constructor is installed:
    echo   1. Activate your conda environment: conda activate base
    echo   2. Ensure conda is in your PATH
    echo   3. Try running: constructor --version
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
    echo Make sure you're running this script from the installer/ directory.
    pause
    exit /b 1
)

echo   - Config file: %CONFIG_FILE% (found)
echo   - Config file: %CONFIG_FILE% >> "%BUILD_LOG%"
echo.

REM ===== Phase 2: Validate Configuration =====
echo [Phase 2/5] Validating installer configuration...
echo [Phase 2/5] Validating configuration... >> "%BUILD_LOG%"

if exist "%VALIDATOR%" (
    python "%VALIDATOR%"
    if errorlevel 1 (
        echo.
        echo ERROR: Configuration validation failed!
        echo ERROR: Validation failed >> "%BUILD_LOG%"
        echo Check the error messages above and fix any issues.
        pause
        exit /b 1
    )
    echo   - All validation checks passed
    echo   - Validation: PASSED >> "%BUILD_LOG%"
) else (
    echo   - Validation script not found, skipping validation
    echo   - Validation: SKIPPED (validator not found) >> "%BUILD_LOG%"
)
echo.

REM ===== Phase 3: Clean Previous Builds =====
echo [Phase 3/5] Cleaning previous v1.5.3 builds...
echo [Phase 3/5] Cleaning previous builds... >> "%BUILD_LOG%"

REM Only clean v1.5.3 builds, preserve other versions
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

REM ===== Phase 4: Build Installer =====
echo [Phase 4/5] Building installer with constructor...
echo [Phase 4/5] Building installer... >> "%BUILD_LOG%"
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
    echo Common issues:
    echo   - Missing dependencies in construct_v1.5.3.yaml
    echo   - Network issues downloading packages
    echo   - Insufficient disk space
    echo   - Corrupted conda package cache
    echo.
    echo To troubleshoot:
    echo   1. Check %BUILD_LOG% for details
    echo   2. Try: conda clean --all
    echo   3. Verify all files in construct_v1.5.3.yaml extra_files exist
    echo.
    pause
    exit /b 1
)

echo.

REM ===== Phase 5: Verify Output and Generate Report =====
echo [Phase 5/5] Verifying build output...
echo [Phase 5/5] Verifying output... >> "%BUILD_LOG%"

set INSTALLER_NAME=WhisperJAV-%VERSION%-Windows-x86_64.exe

if not exist "%INSTALLER_NAME%" (
    echo.
    echo ERROR: Installer file was not created!
    echo ERROR: Output file missing: %INSTALLER_NAME% >> "%BUILD_LOG%"
    echo.
    echo Expected: %INSTALLER_NAME%
    echo.
    echo This suggests the build failed silently.
    echo Check constructor output above for errors.
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
echo   3. Check installation logs (install_log_v1.5.3.txt)
echo   4. Distribute to users
echo.
echo To test installer:
echo   - Run %INSTALLER_NAME%
echo   - Follow installation prompts
echo   - Launch GUI from desktop shortcut
echo   - Process a test video
echo.
echo ===============================================================================

pause
exit /b 0
