@echo off
echo Creating desktop shortcut...

set SCRIPT_DIR=%~dp0
set SHORTCUT_NAME=WhisperJAV.lnk
set TARGET=%SCRIPT_DIR%WhisperJAV.exe
set ICON=%SCRIPT_DIR%whisperjav_icon.ico

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\%SHORTCUT_NAME%'); $Shortcut.TargetPath = '%TARGET%'; $Shortcut.IconLocation = '%ICON%'; $Shortcut.Save()"

if exist "%USERPROFILE%\Desktop\%SHORTCUT_NAME%" (
    echo Desktop shortcut created successfully!
) else (
    echo Failed to create desktop shortcut.
)
