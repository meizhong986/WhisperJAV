@echo off
REM ========================================
REM WhisperJAV Desktop Shortcut Creator
REM Manual/Diagnostic Version
REM ========================================

echo WhisperJAV Desktop Shortcut Creator
echo =====================================
echo.

REM Prompt for installation directory
set /p INSTALL_DIR="Enter WhisperJAV installation directory (e.g., C:\Users\YourName\AppData\Local\WhisperJAV): "

if not exist "%INSTALL_DIR%" (
    echo ERROR: Directory not found: %INSTALL_DIR%
    pause
    exit /b 1
)

echo.
echo Using installation directory: %INSTALL_DIR%
echo.

REM Setup paths
set SHORTCUT_NAME=WhisperJAV v1.5.1.lnk
set TARGET_EXE=%INSTALL_DIR%\WhisperJAV-GUI.exe
set PYTHONW=%INSTALL_DIR%\pythonw.exe
set TARGET_ARGS=-m whisperjav.webview_gui.main
set ICON=%INSTALL_DIR%\whisperjav_icon.ico

REM Check what files exist
echo Checking files:
if exist "%TARGET_EXE%" (
    echo [OK] WhisperJAV-GUI.exe found
    set USE_LAUNCHER_EXE=1
) else (
    echo [MISSING] WhisperJAV-GUI.exe not found
    set USE_LAUNCHER_EXE=0
)

if exist "%PYTHONW%" (
    echo [OK] pythonw.exe found
) else (
    echo [MISSING] pythonw.exe not found
)

if exist "%ICON%" (
    echo [OK] whisperjav_icon.ico found
) else (
    echo [WARNING] whisperjav_icon.ico not found - shortcut will use default icon
    set ICON=
)

echo.
echo Creating desktop shortcut...
echo.

if "%USE_LAUNCHER_EXE%"=="1" (
    REM Preferred: Point to WhisperJAV-GUI.exe
    echo Target: %TARGET_EXE%
    echo Icon: %ICON%
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desktop = [Environment]::GetFolderPath('Desktop'); $ShortcutPath = Join-Path $Desktop '%SHORTCUT_NAME%'; Write-Host 'Creating shortcut:' $ShortcutPath; $Shortcut = $WshShell.CreateShortcut($ShortcutPath); $Shortcut.TargetPath = '%TARGET_EXE%'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; if ('%ICON%' -ne '') { $Shortcut.IconLocation = '%ICON%' }; $Shortcut.Description = 'WhisperJAV v1.5.1 - Japanese AV Subtitle Generator'; $Shortcut.Save(); Write-Host 'Shortcut created successfully!'"
) else (
    REM Fallback: Point to pythonw.exe with module arguments
    echo Target: %PYTHONW% %TARGET_ARGS%
    echo Icon: %ICON%
    echo.

    if not exist "%PYTHONW%" (
        echo ERROR: Neither WhisperJAV-GUI.exe nor pythonw.exe found!
        echo Cannot create shortcut.
        pause
        exit /b 1
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desktop = [Environment]::GetFolderPath('Desktop'); $ShortcutPath = Join-Path $Desktop '%SHORTCUT_NAME%'; Write-Host 'Creating shortcut:' $ShortcutPath; $Shortcut = $WshShell.CreateShortcut($ShortcutPath); $Shortcut.TargetPath = '%PYTHONW%'; $Shortcut.Arguments = '%TARGET_ARGS%'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; if ('%ICON%' -ne '') { $Shortcut.IconLocation = '%ICON%' }; $Shortcut.Description = 'WhisperJAV v1.5.1 - Japanese AV Subtitle Generator'; $Shortcut.Save(); Write-Host 'Shortcut created successfully!'"
)

if %errorlevel% neq 0 (
    echo.
    echo ERROR: PowerShell command failed!
    echo Error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

REM Verify shortcut was created
set DESKTOP_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%
if exist "%DESKTOP_PATH%" (
    echo.
    echo ========================================
    echo SUCCESS: Desktop shortcut created!
    echo ========================================
    echo Location: %DESKTOP_PATH%
    echo.
) else (
    echo.
    echo ========================================
    echo WARNING: Shortcut may not have been created
    echo ========================================
    echo Expected location: %DESKTOP_PATH%
    echo.
)

pause
