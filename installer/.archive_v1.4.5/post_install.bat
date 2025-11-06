@echo off
ECHO Running Python Post-Install Script...

REM This command uses the python.exe from the root of the installation directory
REM to run the post_install.py script, which is also in the root.
"%~dp0..\python.exe" "%~dp0..\post_install.py"

REM Check the exit code of the python script. If it's not 0, there was an error.
if %errorlevel% neq 0 (
  ECHO ERROR: Python post-install script failed with error code %errorlevel% >&2
  exit /b %errorlevel%
)

ECHO Python Post-Install Script finished successfully.
exit /b 0