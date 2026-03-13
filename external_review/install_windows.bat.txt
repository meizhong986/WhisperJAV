@echo off
REM ==============================================================================
REM WhisperJAV Windows Installation Script - Thin Wrapper
REM ==============================================================================
REM
REM ARCHITECTURAL NOTE:
REM -------------------
REM This is a THIN WRAPPER that delegates to install.py.
REM
REM WHY THIN WRAPPER:
REM Before v2.0: This was 950+ lines with duplicated GPU detection, retry logic,
REM              timeout handling, etc. Changes had to be made in 3 places.
REM After v2.0:  All logic is in the unified installer module (Python).
REM              This script just forwards arguments to install.py.
REM
REM WHAT THIS SCRIPT DOES:
REM 1. Verify we're in the right directory (has install.py)
REM 2. Forward all arguments to python install.py
REM 3. Return the exit code from install.py
REM
REM WHY KEEP SHELL SCRIPT AT ALL:
REM - Users expect to double-click .bat files on Windows
REM - Easier than explaining "open cmd, cd to directory, run python install.py"
REM - Legacy muscle memory from previous versions
REM
REM For the full implementation, see:
REM   - install.py (root) - Main installation script
REM   - whisperjav/installer/ - Unified installer module
REM
REM Options (forwarded to install.py):
REM   --cpu-only              Install CPU-only PyTorch
REM   --cuda118               Install PyTorch for CUDA 11.8
REM   --cuda128               Install PyTorch for CUDA 12.8 (default)
REM   --no-speech-enhancement Skip speech enhancement packages
REM   --minimal               Minimal install (transcription only)
REM   --dev                   Install in development/editable mode
REM   --local-llm             Install local LLM support (prebuilt wheel)
REM   --local-llm-build       Install local LLM support (build from source)
REM   --help                  Show help message
REM
REM Author: Senior Architect
REM Date: 2026-01-26
REM ==============================================================================

setlocal EnableDelayedExpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"

REM Navigate to repository root (parent of installer/)
cd /d "%SCRIPT_DIR%.."

REM Verify install.py exists
if not exist "install.py" (
    echo ============================================================
    echo   ERROR: install.py not found
    echo ============================================================
    echo.
    echo   This script must be run from the WhisperJAV repository.
    echo   Expected location: whisperjav/installer/install_windows.bat
    echo.
    echo   To install WhisperJAV:
    echo     git clone https://github.com/meizhong986/whisperjav.git
    echo     cd whisperjav
    echo     installer\install_windows.bat
    echo.
    exit /b 1
)

REM ==============================================================================
REM Forward to install.py
REM ==============================================================================
REM
REM WHY FORWARD ALL ARGUMENTS:
REM - install.py handles argument parsing with argparse
REM - All validation, help text, etc. is maintained in one place
REM - No need to duplicate argument parsing in batch script
REM
echo ============================================================
echo   WhisperJAV Installation (via install.py)
echo ============================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10-3.12 and add it to PATH
    exit /b 1
)

REM Run install.py with all arguments
python install.py %*
set "EXIT_CODE=%ERRORLEVEL%"

REM ==============================================================================
REM Return exit code from install.py
REM ==============================================================================
REM
REM WHY PRESERVE EXIT CODE:
REM - Allows CI/CD systems to detect installation failures
REM - Allows other scripts to check if installation succeeded
REM
if %EXIT_CODE% NEQ 0 (
    echo.
    echo ============================================================
    echo   Installation failed with exit code %EXIT_CODE%
    echo ============================================================
    echo   Check the output above for error details.
    echo   For help: python install.py --help
    echo.
)

exit /b %EXIT_CODE%
