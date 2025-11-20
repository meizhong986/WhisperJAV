@echo off
REM ========================================
REM Build WhisperJAV wheel for installer
REM ========================================
REM
REM This script builds a wheel from the local source code
REM to be bundled with the installer
REM

echo Building WhisperJAV wheel from local source...
echo.

REM Go to repository root
cd ..

REM Clean old builds
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build

REM Build wheel
python -m pip install --upgrade build
python -m build --wheel

REM Copy wheel to installer directory
if not exist "dist\whisperjav-*.whl" (
    echo ERROR: Wheel build failed!
    exit /b 1
)

REM Find the wheel file and copy it (keep original filename)
for %%f in (dist\whisperjav-*.whl) do (
    copy "%%f" "installer\"
    echo.
    echo Wheel copied to: installer\%%~nxf
    echo Original wheel: %%f
)

cd installer
echo.
echo Wheel build complete!
exit /b 0
