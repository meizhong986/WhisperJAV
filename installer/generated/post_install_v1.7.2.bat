@echo off
REM ========================================
REM WhisperJAV v1.7.2 Post-Install Wrapper
REM ========================================
REM
REM This batch file is called by conda-constructor after the base environment
REM is created. Constructor places this script in pkgs/ and extra_files in the
REM parent directory (installation root).
REM
REM This script launches the Python post-install script that handles:
REM - Preflight checks (disk space, network, WebView2)
REM - PyTorch installation (CUDA or CPU)
REM - Python dependencies installation
REM - WhisperJAV application installation
REM - Desktop shortcut creation
REM

ECHO ========================================
ECHO WhisperJAV v1.7.2 Post-Install
ECHO ========================================
ECHO.

REM Verify Python executable exists in parent directory
if not exist "%~dp0..\python.exe" (
    ECHO ERROR: Cannot find python.exe
    ECHO Expected location: %~dp0..\python.exe
    ECHO Installation may be corrupted.
    ECHO.
    pause
    exit /b 1
)

REM Verify Python post-install script exists in parent directory
if not exist "%~dp0..\post_install_v1.7.2.py" (
    ECHO ERROR: Cannot find post_install_v1.7.2.py
    ECHO Expected location: %~dp0..\post_install_v1.7.2.py
    ECHO Installation files may be missing.
    ECHO.
    pause
    exit /b 1
)

ECHO Using Python: "%~dp0..\python.exe"
ECHO Script location: "%~dp0..\post_install_v1.7.2.py"
ECHO.

REM Run the Python post-install script
REM Both python.exe and the .py script are in the parent directory (../)
"%~dp0..\python.exe" "%~dp0..\post_install_v1.7.2.py"

REM Check exit code
if %errorlevel% neq 0 (
    ECHO.
    ECHO ========================================
    ECHO ERROR: Post-install script failed!
    ECHO ========================================
    ECHO Exit code: %errorlevel%
    ECHO Check install_log_v1.7.2.txt for details
    ECHO.
    pause
    exit /b %errorlevel%
)

ECHO.
ECHO ========================================
ECHO Post-Install completed successfully!
ECHO ========================================
ECHO.

exit /b 0
