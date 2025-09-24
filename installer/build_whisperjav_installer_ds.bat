@echo off
REM Build WhisperJAV Installer

echo Building WhisperJAV Installer...
echo.

REM Check if constructor is installed by trying to run it with --help
constructor --help >nul 2>nul
if errorlevel 1 (
    echo ERROR: Constructor not found or not working!
    echo Please install with: conda install constructor -c conda-forge
    echo.
    echo If constructor is installed but not found, try:
    echo 1. Activate your conda environment: conda activate base
    echo 2. Ensure conda is in your PATH
    pause
    exit /b 1
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist _build rmdir /s /q _build
if exist *.exe del *.exe

REM Build installer
constructor . -v 

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete!
for /f "tokens=2 delims=: " %%A in ('findstr /b /c:"version:" construct.yaml') do set INSTVER=%%A
set INSTVER=%INSTVER: =%
echo Installer created: WhisperJAV-%INSTVER%-Windows-x86_64.exe
echo.
pause