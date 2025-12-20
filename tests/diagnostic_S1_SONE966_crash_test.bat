@echo off
REM ============================================================================
REM DIAGNOSTIC TEST: S1 Crash Reproducibility on SONE-966
REM ============================================================================
REM
REM PURPOSE: Verify if the crash in balanced+aggressive pipeline is reproducible
REM          when processing SONE-966 audio file.
REM
REM HYPOTHESIS BEING TESTED:
REM   H1: Crash is deterministic and triggered by balanced+aggressive+SONE-966
REM   H5: Crash is reproducible across multiple runs
REM
REM CONFIGURATION:
REM   - Pipeline: balanced (same as E1 Pass 1 which crashed)
REM   - Sensitivity: aggressive (same as E1 Pass 1)
REM   - Debug mode: ENABLED
REM   - Iterations: 5 runs
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "TEST_FILE=.\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=.\test_results\diagnostic_S1_crash_%TIMESTAMP%"
set "NUM_ITERATIONS=5"

REM WhisperJAV command
set "WJ_CMD=whisperjav"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================================
echo  DIAGNOSTIC TEST: S1 Crash Reproducibility
echo ============================================================================
echo  Target File: %TEST_FILE%
echo  Output Dir:  %OUTPUT_DIR%
echo  Iterations:  %NUM_ITERATIONS%
echo  Timestamp:   %TIMESTAMP%
echo ============================================================================
echo.
echo  Configuration under test:
echo    Pipeline:    balanced
echo    Sensitivity: aggressive
echo    Debug:       ENABLED
echo    Keep-temp:   ENABLED
echo.
echo  This test will run %NUM_ITERATIONS% iterations to verify crash reproducibility.
echo ============================================================================
echo.

REM Check if test file exists
if not exist "%TEST_FILE%" (
    echo ERROR: Test file not found: %TEST_FILE%
    echo Please ensure the SONE-966 audio file exists.
    exit /b 1
)

REM Initialize counters
set "PASS_COUNT=0"
set "FAIL_COUNT=0"
set "CRASH_COUNT=0"

REM Create summary file header
set "SUMMARY_FILE=%OUTPUT_DIR%\crash_test_summary.txt"
echo DIAGNOSTIC TEST: S1 Crash Reproducibility > "%SUMMARY_FILE%"
echo ============================================ >> "%SUMMARY_FILE%"
echo Test File: %TEST_FILE% >> "%SUMMARY_FILE%"
echo Started: %date% %time% >> "%SUMMARY_FILE%"
echo Iterations: %NUM_ITERATIONS% >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo Configuration: >> "%SUMMARY_FILE%"
echo   Pipeline: balanced >> "%SUMMARY_FILE%"
echo   Sensitivity: aggressive >> "%SUMMARY_FILE%"
echo   Debug: enabled >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo RESULTS: >> "%SUMMARY_FILE%"
echo -------------------------------------------- >> "%SUMMARY_FILE%"

REM Run iterations
for /L %%i in (1,1,%NUM_ITERATIONS%) do (
    echo.
    echo ############################################################################
    echo  ITERATION %%i of %NUM_ITERATIONS%
    echo ############################################################################
    echo.

    set "RUN_ID=run_%%i"
    set "RUN_TEMP_DIR=%OUTPUT_DIR%\!RUN_ID!_temp"
    set "RUN_LOG_FILE=%OUTPUT_DIR%\!RUN_ID!.log"

    if not exist "!RUN_TEMP_DIR!" mkdir "!RUN_TEMP_DIR!"

    echo   Log file: !RUN_LOG_FILE!
    echo   Temp dir: !RUN_TEMP_DIR!
    echo   Running test... (output captured to log file)
    echo.

    REM Record start time
    set "START_TIME=!time!"

    REM Run the test
    call %WJ_CMD% "%TEST_FILE%" ^
        --mode balanced ^
        --sensitivity aggressive ^
        --debug ^
        --keep-temp ^
        --temp-dir "!RUN_TEMP_DIR!" ^
        --output-dir "%OUTPUT_DIR%" > "!RUN_LOG_FILE!" 2>&1

    set "EXIT_CODE=!ERRORLEVEL!"
    set "END_TIME=!time!"

    REM Analyze result
    if !EXIT_CODE! EQU 0 (
        echo   Result: PASSED ^(exit code: 0^)
        set /a PASS_COUNT+=1
        echo   Iteration %%i: PASSED ^(exit code: 0^) - !START_TIME! to !END_TIME! >> "%SUMMARY_FILE%"
    ) else (
        if !EXIT_CODE! EQU 3221225620 (
            echo   Result: CRASHED ^(exit code: 3221225620 = STATUS_INTEGER_DIVIDE_BY_ZERO^)
            set /a CRASH_COUNT+=1
            echo   Iteration %%i: CRASHED ^(exit code: 3221225620 - DIVIDE_BY_ZERO^) - !START_TIME! to !END_TIME! >> "%SUMMARY_FILE%"
        ) else (
            echo   Result: FAILED ^(exit code: !EXIT_CODE!^)
            set /a FAIL_COUNT+=1
            echo   Iteration %%i: FAILED ^(exit code: !EXIT_CODE!^) - !START_TIME! to !END_TIME! >> "%SUMMARY_FILE%"
        )
    )

    REM Check for output SRT
    set "SRT_FOUND=NO"
    for %%s in ("%OUTPUT_DIR%\SONE-966*.srt") do (
        set "SRT_FOUND=YES"
        echo   Output SRT: %%~nxs
        REM Rename to include run number
        if exist "%%s" (
            copy "%%s" "%OUTPUT_DIR%\!RUN_ID!_output.srt" >nul 2>&1
            del "%%s" >nul 2>&1
        )
    )
    if "!SRT_FOUND!"=="NO" (
        echo   Output SRT: NONE ^(no output generated^)
    )

    echo.
    echo   Iteration %%i complete. Pausing 3 seconds before next run...
    timeout /t 3 /nobreak >nul
)

REM Calculate totals
set /a TOTAL_RUNS=%PASS_COUNT%+%FAIL_COUNT%+%CRASH_COUNT%

echo.
echo ############################################################################
echo  DIAGNOSTIC TEST COMPLETE
echo ############################################################################
echo.
echo  Results Summary:
echo    Total runs:     %TOTAL_RUNS%
echo    Passed:         %PASS_COUNT%
echo    Failed:         %FAIL_COUNT%
echo    Crashed:        %CRASH_COUNT%
echo.

REM Append summary
echo. >> "%SUMMARY_FILE%"
echo ============================================ >> "%SUMMARY_FILE%"
echo SUMMARY >> "%SUMMARY_FILE%"
echo ============================================ >> "%SUMMARY_FILE%"
echo Total runs:  %TOTAL_RUNS% >> "%SUMMARY_FILE%"
echo Passed:      %PASS_COUNT% >> "%SUMMARY_FILE%"
echo Failed:      %FAIL_COUNT% >> "%SUMMARY_FILE%"
echo Crashed:     %CRASH_COUNT% >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo Completed: %date% %time% >> "%SUMMARY_FILE%"

REM Interpretation
echo.
if %CRASH_COUNT% EQU %NUM_ITERATIONS% (
    echo  CONCLUSION: Crash is 100%% REPRODUCIBLE
    echo              Hypothesis H1 CONFIRMED: balanced+aggressive+SONE-966 triggers crash
    echo  CONCLUSION: Crash is 100%% REPRODUCIBLE >> "%SUMMARY_FILE%"
    echo              Hypothesis H1 CONFIRMED >> "%SUMMARY_FILE%"
) else if %CRASH_COUNT% GTR 0 (
    echo  CONCLUSION: Crash is INTERMITTENT ^(%CRASH_COUNT%/%NUM_ITERATIONS% runs^)
    echo              May indicate race condition or memory-dependent bug
    echo  CONCLUSION: Crash is INTERMITTENT ^(%CRASH_COUNT%/%NUM_ITERATIONS%^) >> "%SUMMARY_FILE%"
) else if %PASS_COUNT% EQU %NUM_ITERATIONS% (
    echo  CONCLUSION: All runs PASSED - crash not reproduced
    echo              May need different conditions to trigger
    echo  CONCLUSION: All runs PASSED - crash not reproduced >> "%SUMMARY_FILE%"
) else (
    echo  CONCLUSION: Mixed results - further investigation needed
    echo  CONCLUSION: Mixed results >> "%SUMMARY_FILE%"
)

echo.
echo ############################################################################
echo  SCANNING LOG FILES FOR ERRORS
echo ############################################################################
echo.

set "ERROR_REPORT=%OUTPUT_DIR%\error_analysis.txt"
echo Error Analysis Report > "%ERROR_REPORT%"
echo ==================== >> "%ERROR_REPORT%"
echo. >> "%ERROR_REPORT%"

for /L %%i in (1,1,%NUM_ITERATIONS%) do (
    set "RUN_LOG=%OUTPUT_DIR%\run_%%i.log"
    if exist "!RUN_LOG!" (
        echo Analyzing run_%%i.log...
        echo. >> "%ERROR_REPORT%"
        echo ============================================ >> "%ERROR_REPORT%"
        echo RUN %%i >> "%ERROR_REPORT%"
        echo ============================================ >> "%ERROR_REPORT%"

        REM Check for specific error patterns
        findstr /I /C:"exit code" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
        findstr /I /C:"ERROR" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
        findstr /I /C:"worker died" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
        findstr /I /C:"crash" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
        findstr /I /C:"divide" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
        findstr /I /C:"Failed:" "!RUN_LOG!" >> "%ERROR_REPORT%" 2>&1
    )
)

echo.
echo  Error analysis saved to: %ERROR_REPORT%
echo  Full summary saved to: %SUMMARY_FILE%
echo  All logs saved to: %OUTPUT_DIR%
echo.
echo ============================================================================

endlocal
exit /b 0
