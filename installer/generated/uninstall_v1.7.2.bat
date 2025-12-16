@echo off
REM ===============================================================================
REM WhisperJAV v1.7.2 Uninstaller
REM ===============================================================================
REM
REM This script completely removes WhisperJAV v1.7.2 from your system:
REM - Installation directory (all files)
REM - Desktop shortcut
REM - Optionally: User configuration
REM - Optionally: Cached AI models (~3GB)
REM
REM This script should be run from the installation directory.
REM

SETLOCAL EnableDelayedExpansion

echo ===============================================================================
echo                      WhisperJAV v1.7.2 Uninstaller
echo ===============================================================================
echo.
echo This will PERMANENTLY DELETE WhisperJAV v1.7.2 and all its files.
echo.

REM Get the installation directory (where this script is located)
set INSTALL_DIR=%~dp0
set INSTALL_DIR=%INSTALL_DIR:~0,-1%

echo Installation Directory:
echo   %INSTALL_DIR%
echo.

REM ===== Confirmation Prompt =====
echo WARNING: This action cannot be undone!
echo.
set /p CONFIRM="Are you sure you want to uninstall WhisperJAV v1.7.2? (yes/no): "

if /i not "%CONFIRM%"=="yes" (
    echo.
    echo Uninstallation canceled.
    echo.
    pause
    exit /b 0
)

echo.
echo ===============================================================================
echo Starting uninstallation...
echo ===============================================================================
echo.

REM Create uninstall log
set UNINSTALL_LOG=%TEMP%\whisperjav_uninstall_v1.7.2.txt
echo WhisperJAV v1.7.2 Uninstallation Log > "%UNINSTALL_LOG%"
echo Uninstall started: %DATE% %TIME% >> "%UNINSTALL_LOG%"
echo Installation directory: %INSTALL_DIR% >> "%UNINSTALL_LOG%"
echo. >> "%UNINSTALL_LOG%"

REM ===== Step 1: Remove Desktop Shortcut =====
echo [Step 1/5] Removing desktop shortcut...
echo [Step 1/5] Desktop shortcut >> "%UNINSTALL_LOG%"

set SHORTCUT_PATH=%USERPROFILE%\Desktop\WhisperJAV v1.7.2.lnk

if exist "%SHORTCUT_PATH%" (
    del "%SHORTCUT_PATH%"
    if exist "%SHORTCUT_PATH%" (
        echo   WARNING: Could not delete shortcut
        echo   WARNING: Shortcut deletion failed >> "%UNINSTALL_LOG%"
    ) else (
        echo   - Desktop shortcut removed
        echo   - Desktop shortcut removed >> "%UNINSTALL_LOG%"
    )
) else (
    echo   - Desktop shortcut not found (already removed)
    echo   - Desktop shortcut not found >> "%UNINSTALL_LOG%"
)
echo.

REM ===== Step 2: Optional - Remove User Configuration =====
echo [Step 2/5] User configuration...
echo [Step 2/5] User configuration >> "%UNINSTALL_LOG%"

set CONFIG_FILE=%INSTALL_DIR%\whisperjav_config.json

if exist "%CONFIG_FILE%" (
    echo.
    echo   Found user configuration file: whisperjav_config.json
    echo   This contains your settings and preferences.
    echo.
    set /p DEL_CONFIG="   Delete configuration file? (y/N): "

    if /i "!DEL_CONFIG!"=="y" (
        del "%CONFIG_FILE%"
        echo   - Configuration file deleted
        echo   - Configuration file deleted >> "%UNINSTALL_LOG%"
    ) else (
        echo   - Configuration file kept
        echo   - Configuration file kept (user choice) >> "%UNINSTALL_LOG%"
    )
) else (
    echo   - No configuration file found
    echo   - No configuration file >> "%UNINSTALL_LOG%"
)
echo.

REM ===== Step 3: Optional - Remove Cached Models =====
echo [Step 3/5] Cached AI models...
echo [Step 3/5] Cached AI models >> "%UNINSTALL_LOG%"

set MODEL_CACHE=%USERPROFILE%\.cache\whisper

if exist "%MODEL_CACHE%" (
    REM Calculate cache size
    for /f "tokens=3" %%a in ('dir "%MODEL_CACHE%" /s /-c 2^>nul ^| findstr /c:"File(s)"') do set MODEL_SIZE=%%a
    set /a MODEL_SIZE_MB=!MODEL_SIZE! / 1048576

    echo.
    echo   Found cached AI models in: %MODEL_CACHE%
    echo   Cache size: ~!MODEL_SIZE_MB! MB
    echo.
    echo   Deleting these will free up disk space, but models will need
    echo   to be re-downloaded if you reinstall WhisperJAV (~3GB, 5-10 min).
    echo.
    set /p DEL_MODELS="   Delete cached models? (y/N): "

    if /i "!DEL_MODELS!"=="y" (
        echo   - Deleting model cache... (this may take a minute)
        rmdir /s /q "%MODEL_CACHE%"
        if exist "%MODEL_CACHE%" (
            echo   WARNING: Could not fully delete model cache
            echo   WARNING: Model cache deletion failed >> "%UNINSTALL_LOG%"
        ) else (
            echo   - Model cache deleted (~!MODEL_SIZE_MB! MB freed)
            echo   - Model cache deleted (~!MODEL_SIZE_MB! MB) >> "%UNINSTALL_LOG%"
        )
    ) else (
        echo   - Model cache kept
        echo   - Model cache kept (user choice) >> "%UNINSTALL_LOG%"
    )
) else (
    echo   - No model cache found
    echo   - No model cache found >> "%UNINSTALL_LOG%"
)
echo.

REM ===== Step 4: Remove Installation Directory =====
echo [Step 4/5] Removing installation directory...
echo [Step 4/5] Installation directory >> "%UNINSTALL_LOG%"

REM We need to delete the installation directory, but this script is running from it!
REM Strategy: Create a temporary cleanup script that runs after this script exits

set CLEANUP_SCRIPT=%TEMP%\whisperjav_cleanup_v1.7.2.bat

echo @echo off > "%CLEANUP_SCRIPT%"
echo REM Temporary cleanup script for WhisperJAV v1.7.2 >> "%CLEANUP_SCRIPT%"
echo echo Removing installation directory... >> "%CLEANUP_SCRIPT%"
echo timeout /t 2 /nobreak ^>nul >> "%CLEANUP_SCRIPT%"
echo rmdir /s /q "%INSTALL_DIR%" >> "%CLEANUP_SCRIPT%"
echo if exist "%INSTALL_DIR%" ( >> "%CLEANUP_SCRIPT%"
echo     echo ERROR: Could not delete installation directory >> "%CLEANUP_SCRIPT%"
echo     echo %INSTALL_DIR% >> "%CLEANUP_SCRIPT%"
echo     echo. >> "%CLEANUP_SCRIPT%"
echo     echo You may need to delete it manually. >> "%CLEANUP_SCRIPT%"
echo     pause >> "%CLEANUP_SCRIPT%"
echo ) else ( >> "%CLEANUP_SCRIPT%"
echo     echo Installation directory removed successfully! >> "%CLEANUP_SCRIPT%"
echo     echo. >> "%CLEANUP_SCRIPT%"
echo     echo WhisperJAV v1.7.2 has been uninstalled. >> "%CLEANUP_SCRIPT%"
echo     timeout /t 3 /nobreak ^>nul >> "%CLEANUP_SCRIPT%"
echo ) >> "%CLEANUP_SCRIPT%"
echo del "%%~f0" >> "%CLEANUP_SCRIPT%"

echo   - Cleanup script created: %CLEANUP_SCRIPT%
echo   - Cleanup script created >> "%UNINSTALL_LOG%"
echo.

REM ===== Step 5: Final Summary =====
echo [Step 5/5] Uninstallation summary
echo [Step 5/5] Uninstallation summary >> "%UNINSTALL_LOG%"

echo.
echo ===============================================================================
echo                      Uninstallation Complete!
echo ===============================================================================
echo.
echo The following items have been removed:
echo   - Desktop shortcut: WhisperJAV v1.7.2.lnk
if /i "%DEL_CONFIG%"=="y" echo   - User configuration file
if /i "%DEL_MODELS%"=="y" echo   - Cached AI models (~!MODEL_SIZE_MB! MB)
echo.
echo The installation directory will be deleted when you close this window:
echo   %INSTALL_DIR%
echo.
echo Uninstall log saved to:
echo   %UNINSTALL_LOG%
echo.
echo ===============================================================================

REM Log completion
echo. >> "%UNINSTALL_LOG%"
echo Uninstallation completed: %DATE% %TIME% >> "%UNINSTALL_LOG%"

REM Launch cleanup script and exit
echo.
echo Press any key to finish uninstallation and close this window...
pause >nul

start "" "%CLEANUP_SCRIPT%"
exit
