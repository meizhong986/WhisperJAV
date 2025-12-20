@echo off
REM ============================================================================
REM Single Parameter Test - Find which param prevents the crash
REM ============================================================================
REM
REM Run ONE test at a time. If it passes, that change prevents the crash.
REM If it crashes (exit code -1073741676), try the next one.
REM
REM Usage:
REM   diagnostic_single_param_test.bat minimal    (safest - try first)
REM   diagnostic_single_param_test.bat beam1      (greedy decoding)
REM   diagnostic_single_param_test.bat no_word_ts (disable word timestamps)
REM   diagnostic_single_param_test.bat baseline   (will crash)
REM   diagnostic_single_param_test.bat --list     (show all tests)
REM
REM ============================================================================

echo.
echo ============================================================================
echo  Single Parameter Crash Test
echo ============================================================================

cd /d %~dp0\..

if "%1"=="" (
    echo.
    echo Usage: diagnostic_single_param_test.bat ^<test_name^>
    echo.
    echo Recommended order:
    echo   1. minimal     - Start here ^(safest^)
    echo   2. beam1       - Try greedy decoding
    echo   3. no_word_ts  - Try without word timestamps
    echo   4. beam5       - Try different beam size
    echo   5. no_speech_06 - Try higher no_speech threshold
    echo   6. baseline    - Will crash ^(confirms bug^)
    echo.
    echo Use --list to see all available tests
    echo.
    pause
    exit /b 1
)

python tests\diagnostic_single_param_test.py %1

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================================
if %EXIT_CODE%==0 (
    echo  PASSED! Exit code: 0
    echo  This parameter change PREVENTS the crash!
) else if %EXIT_CODE%==-1073741676 (
    echo  CRASHED! Exit code: -1073741676 ^(STATUS_INTEGER_DIVIDE_BY_ZERO^)
    echo  This parameter change does NOT prevent the crash.
) else (
    echo  Exit code: %EXIT_CODE%
)
echo ============================================================================
echo.
pause
