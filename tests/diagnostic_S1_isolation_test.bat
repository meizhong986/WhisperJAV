@echo off
REM ============================================================================
REM DIAGNOSTIC TEST: S1 Isolation Test (VRAM Fragmentation Hypothesis)
REM ============================================================================
REM
REM PURPOSE: Test if crash occurs in complete isolation (fresh process, no prior
REM          GPU usage) to rule out VRAM fragmentation as the cause.
REM
REM HYPOTHESIS BEING TESTED:
REM   H-FRAG: VRAM fragmentation from prior del/gc/empty_cache causes crash
REM
REM IF CRASH OCCURS IN ISOLATION:
REM   -> Fragmentation hypothesis RULED OUT
REM   -> Bug is in faster-whisper/ctranslate2 with specific parameters
REM
REM IF NO CRASH IN ISOLATION:
REM   -> Fragmentation hypothesis SUPPORTED
REM   -> Need to investigate memory management
REM
REM CONFIGURATION:
REM   - Pipeline: balanced
REM   - Sensitivity: aggressive
REM   - Single test, fresh process
REM   - No prior GPU operations
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM CONFIGURATION - ABSOLUTE PATHS
REM ============================================================================
set "TEST_FILE=C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav"
set "BASE_DIR=C:\BIN\git\whisperJav_V1_Minami_Edition"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=%BASE_DIR%\test_results\diagnostic_isolation_%TIMESTAMP%"

REM WhisperJAV command
set "WJ_CMD=whisperjav"

echo.
echo ============================================================================
echo  DIAGNOSTIC TEST: S1 Isolation Test (VRAM Fragmentation Hypothesis)
echo ============================================================================
echo.
echo  Test File: %TEST_FILE%
echo  Output:    %OUTPUT_DIR%
echo  Timestamp: %TIMESTAMP%
echo.
echo ============================================================================
echo  PURPOSE:
echo    This test runs S1 (balanced+aggressive) on SONE-966 in COMPLETE ISOLATION.
echo    No prior GPU operations. Fresh process. First and only test.
echo.
echo  INTERPRETATION:
echo    - If CRASH occurs:   Fragmentation hypothesis RULED OUT
echo                         Bug is in faster-whisper/ctranslate2
echo    - If NO CRASH:       Fragmentation hypothesis SUPPORTED
echo                         Prior GPU usage may cause the issue
echo ============================================================================
echo.

REM Verify test file exists
if not exist "%TEST_FILE%" (
    echo ERROR: Test file not found!
    echo Expected: %TEST_FILE%
    echo.
    echo Please verify the path is correct.
    exit /b 1
)

echo [OK] Test file found: %TEST_FILE%
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo [OK] Output directory created: %OUTPUT_DIR%
echo.

REM Create temp directory
set "TEMP_DIR=%OUTPUT_DIR%\temp"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

REM Log file
set "LOG_FILE=%OUTPUT_DIR%\isolation_test.log"
set "SUMMARY_FILE=%OUTPUT_DIR%\isolation_test_summary.txt"

REM Write summary header
echo DIAGNOSTIC TEST: S1 Isolation Test > "%SUMMARY_FILE%"
echo ============================================ >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo Test File: %TEST_FILE% >> "%SUMMARY_FILE%"
echo Started: %date% %time% >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo Configuration: >> "%SUMMARY_FILE%"
echo   Pipeline: balanced >> "%SUMMARY_FILE%"
echo   Sensitivity: aggressive >> "%SUMMARY_FILE%"
echo   Mode: ISOLATION (fresh process, no prior GPU ops) >> "%SUMMARY_FILE%"
echo   Debug: enabled >> "%SUMMARY_FILE%"
echo   Keep-temp: enabled >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"

echo ############################################################################
echo  RUNNING ISOLATION TEST
echo ############################################################################
echo.
echo  This is a FRESH PROCESS with NO prior GPU operations.
echo  Log file: %LOG_FILE%
echo.
echo  Running whisperjav...
echo.

REM Record start time
set "START_TIME=%time%"

REM ============================================================================
REM THE ACTUAL TEST - Single S1 run in isolation
REM ============================================================================
call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --sensitivity aggressive ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
set "END_TIME=%time%"

echo.
echo ############################################################################
echo  TEST COMPLETE
echo ############################################################################
echo.
echo  Start time: %START_TIME%
echo  End time:   %END_TIME%
echo  Exit code:  %EXIT_CODE%
echo.

REM Record result
echo RESULT: >> "%SUMMARY_FILE%"
echo -------------------------------------------- >> "%SUMMARY_FILE%"
echo Start time: %START_TIME% >> "%SUMMARY_FILE%"
echo End time:   %END_TIME% >> "%SUMMARY_FILE%"
echo Exit code:  %EXIT_CODE% >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"

REM Interpret exit code
if %EXIT_CODE% EQU 0 (
    echo  STATUS: PASSED
    echo.
    echo  ============================================================
    echo  INTERPRETATION: NO CRASH in isolation
    echo  ============================================================
    echo  The test PASSED when run in isolation (fresh process).
    echo.
    echo  This SUPPORTS the VRAM fragmentation hypothesis:
    echo    - In isolation, no fragmentation exists
    echo    - Prior GPU operations may cause fragmentation
    echo    - The crash in the full test suite may be due to
    echo      accumulated VRAM fragmentation from earlier tests
    echo.
    echo  NEXT STEPS:
    echo    - Run this test AFTER running other GPU-heavy tests
    echo    - Compare results to see if prior GPU usage matters
    echo  ============================================================
    echo. >> "%SUMMARY_FILE%"
    echo STATUS: PASSED >> "%SUMMARY_FILE%"
    echo. >> "%SUMMARY_FILE%"
    echo INTERPRETATION: >> "%SUMMARY_FILE%"
    echo   NO CRASH in isolation - SUPPORTS fragmentation hypothesis >> "%SUMMARY_FILE%"
    echo   Prior GPU operations may cause the crash >> "%SUMMARY_FILE%"
) else if %EXIT_CODE% EQU 3221225620 (
    echo  STATUS: CRASHED (exit code 3221225620 = STATUS_INTEGER_DIVIDE_BY_ZERO)
    echo.
    echo  ============================================================
    echo  INTERPRETATION: CRASH occurred even in isolation
    echo  ============================================================
    echo  The test CRASHED even in a fresh process with no prior GPU usage.
    echo.
    echo  This RULES OUT the VRAM fragmentation hypothesis:
    echo    - Fragmentation cannot be the cause
    echo    - The bug is in faster-whisper/ctranslate2 code
    echo    - Triggered by specific parameter + audio combination
    echo.
    echo  ROOT CAUSE is likely:
    echo    - beam_size=3 with word_timestamps on this specific audio
    echo    - low no_speech_threshold=0.22 causing edge case
    echo    - Bug in ctranslate2 timestamp/probability calculation
    echo.
    echo  NEXT STEPS:
    echo    - Test with beam_size=1
    echo    - Test with higher no_speech_threshold
    echo    - Report bug to faster-whisper maintainers
    echo  ============================================================
    echo. >> "%SUMMARY_FILE%"
    echo STATUS: CRASHED (exit code 3221225620) >> "%SUMMARY_FILE%"
    echo. >> "%SUMMARY_FILE%"
    echo INTERPRETATION: >> "%SUMMARY_FILE%"
    echo   CRASH in isolation - RULES OUT fragmentation hypothesis >> "%SUMMARY_FILE%"
    echo   Bug is in faster-whisper/ctranslate2 with specific parameters >> "%SUMMARY_FILE%"
) else (
    echo  STATUS: FAILED (exit code %EXIT_CODE%)
    echo.
    echo  ============================================================
    echo  INTERPRETATION: Unexpected failure
    echo  ============================================================
    echo  The test failed with an unexpected exit code.
    echo  Review the log file for details.
    echo  ============================================================
    echo. >> "%SUMMARY_FILE%"
    echo STATUS: FAILED (exit code %EXIT_CODE%) >> "%SUMMARY_FILE%"
    echo. >> "%SUMMARY_FILE%"
    echo INTERPRETATION: >> "%SUMMARY_FILE%"
    echo   Unexpected failure - review log file >> "%SUMMARY_FILE%"
)

echo.
echo  Completed: %date% %time% >> "%SUMMARY_FILE%"

REM Check for output SRT
echo.
echo  Output files:
set "SRT_FOUND=NO"
for %%s in ("%OUTPUT_DIR%\*.srt") do (
    set "SRT_FOUND=YES"
    echo    - %%~nxs
)
if "%SRT_FOUND%"=="NO" (
    echo    - No SRT output generated
)

REM Extract key lines from log
echo.
echo  Key log entries:
echo  ----------------------------------------
findstr /I /C:"ERROR" "%LOG_FILE%" 2>nul
findstr /I /C:"worker died" "%LOG_FILE%" 2>nul
findstr /I /C:"exit code" "%LOG_FILE%" 2>nul
findstr /I /C:"Failed:" "%LOG_FILE%" 2>nul
echo  ----------------------------------------

echo.
echo  Full log: %LOG_FILE%
echo  Summary:  %SUMMARY_FILE%
echo.
echo ============================================================================

endlocal
exit /b %EXIT_CODE%
