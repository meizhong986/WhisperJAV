@echo off
echo Creating desktop shortcut...

set SCRIPT_DIR=%~dp0
set SHORTCUT_NAME=WhisperJAV.lnk
set PYTHONW=%SCRIPT_DIR%pythonw.exe
set TARGET_ARGS=-m whisperjav.gui.whisperjav_gui
set ICON=%SCRIPT_DIR%whisperjav_icon.ico

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\%SHORTCUT_NAME%'); $Shortcut.TargetPath = '%PYTHONW%'; $Shortcut.Arguments = '%TARGET_ARGS%'; $Shortcut.IconLocation = '%ICON%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $Shortcut.Save()"

if exist "%USERPROFILE%\Desktop\%SHORTCUT_NAME%" (
    echo Desktop shortcut created successfully!
) else (
    echo Failed to create desktop shortcut.
)
