@echo off
REM Build WhisperJAV PyWebView GUI Executable
REM
REM This script builds a standalone executable for the PyWebView-based GUI
REM using PyInstaller. The output is a single-folder distribution.
REM
REM Prerequisites:
REM   - PyInstaller installed: pip install pyinstaller
REM   - PyWebView installed: pip install pywebview
REM   - All dependencies installed: pip install -e .[gui]
REM
REM Output:
REM   dist/whisperjav-gui/whisperjav-gui.exe

echo ========================================
echo WhisperJAV PyWebView GUI Builder
echo ========================================
echo.

REM Check if PyInstaller is installed
pyinstaller --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller not found!
    echo.
    echo Please install PyInstaller:
    echo   pip install pyinstaller
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking PyInstaller... OK
echo.

REM Check if spec file exists
if not exist "%~dp0whisperjav-gui.spec" (
    echo ERROR: Spec file not found!
    echo Expected: %~dp0whisperjav-gui.spec
    echo.
    pause
    exit /b 1
)

echo [2/4] Checking spec file... OK
echo.

REM Clean previous builds
echo [3/4] Cleaning previous builds...
if exist "%~dp0build" rmdir /s /q "%~dp0build"
if exist "%~dp0dist\whisperjav-gui" rmdir /s /q "%~dp0dist\whisperjav-gui"
echo      Cleanup complete.
echo.

REM Build executable using spec file
echo [4/4] Building executable...
echo      This may take several minutes...
echo.

pyinstaller "%~dp0whisperjav-gui.spec" --clean --noconfirm

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Build failed!
    echo ========================================
    echo.
    echo Check the error messages above for details.
    echo Common issues:
    echo   - Missing dependencies (install with: pip install -e .[gui])
    echo   - Import errors (ensure whisperjav module is importable)
    echo   - Asset files not found (check whisperjav/webview_gui/assets/)
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location:
echo   %~dp0dist\whisperjav-gui\whisperjav-gui.exe
echo.
echo To test the executable:
echo   cd "%~dp0dist\whisperjav-gui"
echo   whisperjav-gui.exe
echo.
echo Distribution folder contents:
dir /b "%~dp0dist\whisperjav-gui"
echo.
pause
