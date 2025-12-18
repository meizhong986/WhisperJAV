@echo off
REM ============================================================================
REM WhisperJAV Single-Pipeline Automated Test Suite
REM ============================================================================
REM Runs single-pass (NON-ENSEMBLE) tests for each supported media file.
REM Scenarios:
REM   1) balanced  | aggressive
REM   2) fidelity  | conservative
REM   3) fast      | balanced
REM   4) direct-to-english + balanced | balanced
REM   5) transformers (balanced)
REM
REM Requirements:
REM   - DEBUG and INFO output enabled: uses --log-level DEBUG and --debug
REM   - No AI translate tests (--translate is NOT used)
REM
REM Output per test:
REM   - Logs: <output>\<test>_<basename>.log
REM   - SRT copied to stable filename: <output>\<test>_<basename>.srt
REM   - Per-test temp dir: <output>\<test>_<basename>_temp
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "TEST_MEDIA_DIR=.\test_media\173_acceptance_test"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=.\test_results\single_%TIMESTAMP%"

REM Always run whisperjav inside conda env WJ
set "WJ_CMD=conda run -n WJ whisperjav"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================================
echo  WhisperJAV Single-Pipeline Automated Test Suite
echo ============================================================================
echo  Test Media: %TEST_MEDIA_DIR%
echo  Output Dir: %OUTPUT_DIR%
echo  Timestamp:  %TIMESTAMP%
echo ============================================================================
echo.

set "FOUND_MEDIA=0"
set "OVERALL_FAIL=0"

for %%f in ("%TEST_MEDIA_DIR%\*.*") do call :MAYBE_RUN "%%~ff"

if "%FOUND_MEDIA%"=="0" (
    echo ERROR: No audio/video files found in %TEST_MEDIA_DIR%
    echo Please add a media file: wav/mp3/mp4/mkv/... to the test_media folder.
    endlocal
    exit /b 1
)

echo.
echo ============================================================================
echo  ALL MEDIA COMPLETE
echo ============================================================================
echo Results saved to: %OUTPUT_DIR%
echo.
echo Output files:
dir /b "%OUTPUT_DIR%"
echo.

if "%OVERALL_FAIL%"=="1" (
    echo One or more tests FAILED.
    endlocal
    exit /b 1
)

echo All tests PASSED.
endlocal
exit /b 0

REM ---------------------------------------------------------------------------
REM Only run suite for supported media extensions
REM ---------------------------------------------------------------------------
:MAYBE_RUN
set "CANDIDATE=%~1"
set "EXT=%~x1"

set "MATCHED=0"
for %%e in (.wav .mp3 .m4a .aac .flac .ogg .opus .wma .mp4 .mkv .mov .avi .webm) do (
    if /I "%EXT%"=="%%e" set "MATCHED=1"
)

if "%MATCHED%"=="1" (
    set "FOUND_MEDIA=1"
    call :RUN_SUITE "%CANDIDATE%"
)

exit /b 0

REM ---------------------------------------------------------------------------
REM Run all single-pipeline tests for a single file
REM ---------------------------------------------------------------------------
:RUN_SUITE
set "TEST_FILE=%~1"
set "TEST_BASENAME=%~n1"

set "T1_RESULT="
set "T2_RESULT="
set "T3_RESULT="
set "T4_RESULT="
set "T5_RESULT="

echo.
echo ----------------------------------------------------------------------------
echo Using test file: %TEST_FILE%
echo Basename: %TEST_BASENAME%
echo ----------------------------------------------------------------------------

REM Helper: remove any prior whisperjav outputs for this basename in OUTPUT_DIR
call :CLEAN_BASE_OUTPUTS "%TEST_BASENAME%"

REM 1) balanced | aggressive
set "TEST_NAME=single_balanced_aggressive"
call :RUN_ONE "%TEST_FILE%" "%TEST_BASENAME%" "%TEST_NAME%" "--mode balanced --sensitivity aggressive --subs-language native"
set "T1_RESULT=%ERRORLEVEL%"
if not "%T1_RESULT%"=="0" set "OVERALL_FAIL=1"

REM 2) fidelity | conservative
set "TEST_NAME=single_fidelity_conservative"
call :RUN_ONE "%TEST_FILE%" "%TEST_BASENAME%" "%TEST_NAME%" "--mode fidelity --sensitivity conservative --subs-language native"
set "T2_RESULT=%ERRORLEVEL%"
if not "%T2_RESULT%"=="0" set "OVERALL_FAIL=1"

REM 3) fast | balanced
set "TEST_NAME=single_fast_balanced"
call :RUN_ONE "%TEST_FILE%" "%TEST_BASENAME%" "%TEST_NAME%" "--mode fast --sensitivity balanced --subs-language native"
set "T3_RESULT=%ERRORLEVEL%"
if not "%T3_RESULT%"=="0" set "OVERALL_FAIL=1"

REM 4) direct to english balanced | balanced
set "TEST_NAME=single_direct_to_english_balanced"
call :RUN_ONE "%TEST_FILE%" "%TEST_BASENAME%" "%TEST_NAME%" "--mode balanced --sensitivity balanced --subs-language direct-to-english"
set "T4_RESULT=%ERRORLEVEL%"
if not "%T4_RESULT%"=="0" set "OVERALL_FAIL=1"

REM 5) transformers
set "TEST_NAME=single_transformers_balanced"
call :RUN_ONE "%TEST_FILE%" "%TEST_BASENAME%" "%TEST_NAME%" "--mode transformers --sensitivity balanced --subs-language native"
set "T5_RESULT=%ERRORLEVEL%"
if not "%T5_RESULT%"=="0" set "OVERALL_FAIL=1"

echo.
echo ============================================================================
echo  FILE SUMMARY: %TEST_BASENAME%
echo ============================================================================
echo   1) balanced|aggressive:            %T1_RESULT%
echo   2) fidelity|conservative:         %T2_RESULT%
echo   3) fast|balanced:                 %T3_RESULT%
echo   4) direct-to-english balanced:    %T4_RESULT%
echo   5) transformers:                  %T5_RESULT%
echo.

exit /b 0

REM ---------------------------------------------------------------------------
REM Run one whisperjav invocation with common settings
REM Args:
REM   %1 = file
REM   %2 = basename
REM   %3 = test name
REM   %4 = extra args string
REM ---------------------------------------------------------------------------
:RUN_ONE
set "ONE_FILE=%~1"
set "ONE_BASE=%~2"
set "ONE_TEST=%~3"
set "ONE_ARGS=%~4"

set "TEMP_DIR=%OUTPUT_DIR%\%ONE_TEST%_%ONE_BASE%_temp"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

set "LOG_FILE=%OUTPUT_DIR%\%ONE_TEST%_%ONE_BASE%.log"

echo.
echo ============================================================================
echo %ONE_TEST%  (%ONE_BASE%)
echo ============================================================================

call :CLEAN_BASE_OUTPUTS "%ONE_BASE%"

call %WJ_CMD% "%ONE_FILE%" --log-level DEBUG --debug --log-file "%LOG_FILE%" --temp-dir "%TEMP_DIR%" --output-dir "%OUTPUT_DIR%" %ONE_ARGS%
set "RC=%ERRORLEVEL%"

call :COPY_SRT "%ONE_BASE%" "%ONE_TEST%"

if "%RC%"=="0" (
    echo %ONE_TEST%: PASSED
) else (
    echo %ONE_TEST%: FAILED (exit code: %RC%)
)

exit /b %RC%

REM ---------------------------------------------------------------------------
REM Copy latest output SRT for a basename into stable per-test name.
REM Expects the run to have produced exactly one: <basename>.*.whisperjav.srt
REM ---------------------------------------------------------------------------
:COPY_SRT
set "BASE=%~1"
set "LABEL=%~2"

set "FINAL_SRT="
for %%s in ("%OUTPUT_DIR%\%BASE%.*.whisperjav.srt") do set "FINAL_SRT=%%~fs"

if defined FINAL_SRT (
    copy "%FINAL_SRT%" "%OUTPUT_DIR%\%LABEL%_%BASE%.srt" >nul
    echo SRT saved as: %LABEL%_%BASE%.srt
) else (
    echo WARNING: No whisperjav SRT found for %BASE% in %OUTPUT_DIR%
)

exit /b 0

REM ---------------------------------------------------------------------------
REM Delete any existing whisperjav output SRTs for this basename in OUTPUT_DIR
REM ---------------------------------------------------------------------------
:CLEAN_BASE_OUTPUTS
del /q "%OUTPUT_DIR%\%~1.*.whisperjav.srt" 2>nul
exit /b 0
