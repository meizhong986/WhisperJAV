@echo off
REM ========================================
REM WhisperJAV v1.7.4 Desktop Shortcut Creator
REM ========================================
REM
REM This script creates a Windows desktop shortcut that:
REM - Launches pythonw.exe (no console window)
REM - Runs whisperjav.webview_gui.main module
REM - Uses the WhisperJAV icon
REM - Sets working directory to installation root
REM
REM Uses PowerShell to create .lnk file via WScript.Shell COM object
REM

echo Creating WhisperJAV v1.7.4 desktop shortcut...

REM Get the directory where this script is located (install root)
set SCRIPT_DIR=%~dp0

REM Shortcut configuration
set SHORTCUT_NAME=WhisperJAV v1.7.4.lnk
set TARGET_EXE=%SCRIPT_DIR%WhisperJAV-GUI.exe
set PYTHONW=%SCRIPT_DIR%pythonw.exe
set TARGET_ARGS=-m whisperjav.webview_gui.main
set ICON=%SCRIPT_DIR%whisperjav_icon.ico

REM Verify WhisperJAV-GUI.exe exists (preferred target)
if exist "%TARGET_EXE%" (
    echo Target: WhisperJAV-GUI.exe (standalone launcher)
    set USE_LAUNCHER_EXE=1
) else (
    echo WARNING: WhisperJAV-GUI.exe not found, using pythonw fallback
    set USE_LAUNCHER_EXE=0

    REM Verify pythonw.exe exists as fallback
    if not exist "%PYTHONW%" (
        echo WARNING: pythonw.exe not found at: %PYTHONW%
        echo Trying python.exe as fallback...
        set PYTHONW=%SCRIPT_DIR%python.exe
    )

    if not exist "%PYTHONW%" (
        echo ERROR: Neither WhisperJAV-GUI.exe nor python.exe found!
        echo Cannot create shortcut.
        exit /b 1
    )
)

REM Verify icon exists
if not exist "%ICON%" (
    echo WARNING: Icon file not found at: %ICON%
    echo Shortcut will be created without custom icon.
    set ICON=
)

REM Create shortcut using PowerShell or VBScript fallback
echo Icon: %ICON%
echo Working Directory: %SCRIPT_DIR%
echo.

REM Try PowerShell first (modern Windows systems)
where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using PowerShell to create shortcut...
    if "%USE_LAUNCHER_EXE%"=="1" (
        REM Preferred: Point to WhisperJAV-GUI.exe
        echo Shortcut target: %TARGET_EXE%
        echo.
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desktop = [Environment]::GetFolderPath('Desktop'); $ShortcutPath = Join-Path $Desktop '%SHORTCUT_NAME%'; $Shortcut = $WshShell.CreateShortcut($ShortcutPath); $Shortcut.TargetPath = '%TARGET_EXE%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; if ('%ICON%' -ne '') { $Shortcut.IconLocation = '%ICON%' }; $Shortcut.Description = 'WhisperJAV v1.7.4 Japanese AV Subtitle Generator'; $Shortcut.Save()"
    ) else (
        REM Fallback: Point to pythonw.exe with module arguments
        echo Shortcut target: %PYTHONW% %TARGET_ARGS%
        echo.
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desktop = [Environment]::GetFolderPath('Desktop'); $ShortcutPath = Join-Path $Desktop '%SHORTCUT_NAME%'; $Shortcut = $WshShell.CreateShortcut($ShortcutPath); $Shortcut.TargetPath = '%PYTHONW%'; $Shortcut.Arguments = '%TARGET_ARGS%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; if ('%ICON%' -ne '') { $Shortcut.IconLocation = '%ICON%' }; $Shortcut.Description = 'WhisperJAV v1.7.4 Japanese AV Subtitle Generator'; $Shortcut.Save()"
    )
) else (
    REM PowerShell not available - use VBScript fallback (works on all Windows)
    echo PowerShell not found, using VBScript fallback...

    REM Create temporary VBScript
    set VBS_SCRIPT=%TEMP%\create_whisperjav_shortcut.vbs

    if "%USE_LAUNCHER_EXE%"=="1" (
        REM Preferred: Point to WhisperJAV-GUI.exe
        echo Shortcut target: %TARGET_EXE%
        echo.
        (
            echo Set WshShell = CreateObject^("WScript.Shell"^)
            echo DesktopPath = WshShell.SpecialFolders^("Desktop"^)
            echo Set Shortcut = WshShell.CreateShortcut^(DesktopPath ^& "\%SHORTCUT_NAME%"^)
            echo Shortcut.TargetPath = "%TARGET_EXE%"
            echo Shortcut.WorkingDirectory = "%SCRIPT_DIR%"
            echo Shortcut.Description = "WhisperJAV v1.7.4 - Japanese AV Subtitle Generator"
        ) > "%VBS_SCRIPT%"
        REM Add icon line if icon exists
        if not "%ICON%"=="" (
            echo Shortcut.IconLocation = "%ICON%" >> "%VBS_SCRIPT%"
        )
        REM Complete VBScript
        echo Shortcut.Save >> "%VBS_SCRIPT%"
    ) else (
        REM Fallback: Point to pythonw.exe with module arguments
        echo Shortcut target: %PYTHONW% %TARGET_ARGS%
        echo.
        (
            echo Set WshShell = CreateObject^("WScript.Shell"^)
            echo DesktopPath = WshShell.SpecialFolders^("Desktop"^)
            echo Set Shortcut = WshShell.CreateShortcut^(DesktopPath ^& "\%SHORTCUT_NAME%"^)
            echo Shortcut.TargetPath = "%PYTHONW%"
            echo Shortcut.Arguments = "%TARGET_ARGS%"
            echo Shortcut.WorkingDirectory = "%SCRIPT_DIR%"
            echo Shortcut.Description = "WhisperJAV v1.7.4 - Japanese AV Subtitle Generator"
        ) > "%VBS_SCRIPT%"
        REM Add icon line if icon exists
        if not "%ICON%"=="" (
            echo Shortcut.IconLocation = "%ICON%" >> "%VBS_SCRIPT%"
        )
        REM Complete VBScript
        echo Shortcut.Save >> "%VBS_SCRIPT%"
    )

    REM Execute VBScript
    cscript //NoLogo "%VBS_SCRIPT%"

    REM Cleanup
    del "%VBS_SCRIPT%" >nul 2>&1
)

REM Verify shortcut was created
set DESKTOP_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%
if exist "%DESKTOP_PATH%" (
    echo.
    echo ========================================
    echo Desktop shortcut created successfully!
    echo ========================================
    echo Location: %DESKTOP_PATH%
    echo.
    exit /b 0
) else (
    echo.
    echo ========================================
    echo WARNING: Failed to create desktop shortcut
    echo ========================================
    echo You can manually create a shortcut to:
    echo   %PYTHONW% %TARGET_ARGS%
    echo.
    exit /b 1
)
