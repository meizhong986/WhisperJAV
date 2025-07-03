@echo off
REM Build WhisperJAV Installer

echo Building WhisperJAV Installer...
echo.

REM Check if constructor is installed
where constructor >nul 2>nul
if errorlevel 1 (
    echo ERROR: Constructor not found!
    echo Please install with: conda install constructor -c conda-forge
    pause
    exit /b 1
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist _build rmdir /s /q _build
if exist *.exe del *.exe

REM Build installer
constructor . --clean

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete!
echo Installer created: WhisperJAV-1.1.2-Windows-x86_64-Setup.exe
echo.
pause
