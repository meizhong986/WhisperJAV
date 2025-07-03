@echo off
title WhisperJAV
cd /d "%~dp0"
set PATH=%~dp0;%~dp0Scripts;%~dp0Library\bin;%PATH%
set PYTHONPATH=%~dp0

echo Starting WhisperJAV GUI...
echo.
echo If this is your first run, models will be downloaded to your local .cache folder
echo.

python.exe -m whisperjav.gui.whisperjav_gui %*

if errorlevel 1 (
    echo.
    echo ========================================
    echo WhisperJAV encountered an error.
    echo Check the error message above.
    echo ========================================
    pause
)