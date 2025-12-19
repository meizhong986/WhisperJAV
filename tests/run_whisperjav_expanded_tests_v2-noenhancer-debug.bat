@echo off
REM ============================================================================
REM WhisperJAV Expanded Test Suite - No Enhancer + Debug Edition
REM ============================================================================
REM
REM PURPOSE: Comprehensive test coverage WITHOUT speech enhancers, with full
REM          debug logging enabled and console output captured to files.
REM
REM CONFIGURATION:
REM   - Speech enhancers: DISABLED (all set to none)
REM   - Debug mode: ENABLED (--debug flag)
REM   - Keep temp files: ENABLED (--keep-temp flag)
REM   - Console output: Captured to log files for review
REM
REM INCLUDED:
REM   - Pipelines: balanced, faster, fast, transformers
REM   - Segmenters: silero-v4.0, silero-v3.1, whisper-vad, none
REM   - Enhancers: none (all disabled)
REM
REM TEST PLAN (8 tests, maximally diverse):
REM   E1-E4: Ensemble (2-pass) - 4 varieties with distinct configurations
REM   S1-S3: Single-pass - 3 varieties covering different pipelines
REM   T1:    Direct-to-English - Whisper translation mode
REM
REM OUTPUT:
REM   - SRT files: {test_id}_{basename}.srt (test_id prefix for easy sorting)
REM   - Log files: {test_id}_{basename}.log (console debug output)
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "TEST_MEDIA_DIR=.\test_media\173_acceptance_test"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=.\test_results\noenhancer_debug_%TIMESTAMP%"

REM WhisperJAV command
set "WJ_CMD=whisperjav"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================================
echo  WhisperJAV Expanded Test Suite - No Enhancer + Debug Edition
echo ============================================================================
echo  Test Media: %TEST_MEDIA_DIR%
echo  Output Dir: %OUTPUT_DIR%
echo  Timestamp:  %TIMESTAMP%
echo ============================================================================
echo.
echo  Configuration:
echo    - Speech Enhancers: DISABLED (none)
echo    - Debug Mode: ENABLED
echo    - Keep Temp Files: ENABLED
echo    - Console Logging: Captured to .log files
echo ============================================================================
echo.
echo  Test Plan:
echo    E1: Ensemble balanced+agg / faster+cons  [silero40/31]
echo    E2: Ensemble fast+bal / transformers+bal [whispervad/none]
echo    E3: Ensemble transformers+agg / balanced+cons [s31/whispervad]
echo    E4: Ensemble faster+bal / fast+agg       [none/silero40]
echo    S1: Single balanced+aggressive           [default config]
echo    S2: Single faster+conservative           [speed-focused]
echo    S3: Single transformers+balanced         [HF pipeline]
echo    T1: Direct-to-English fast+balanced      [Whisper translate]
echo ============================================================================
echo.

goto :MAIN

REM ---------------------------------------------------------------------------
REM Test runner for a single file - runs all 8 test varieties
REM ---------------------------------------------------------------------------
:RUN_SUITE
set "TEST_FILE=%~1"
set "TEST_BASENAME=%~n1"

REM Initialize result variables
set "E1_RESULT=" & set "E2_RESULT=" & set "E3_RESULT=" & set "E4_RESULT="
set "S1_RESULT=" & set "S2_RESULT=" & set "S3_RESULT="
set "T1_RESULT="

echo.
echo ############################################################################
echo  Processing: %TEST_BASENAME%
echo ############################################################################
echo.

REM ============================================================================
REM ENSEMBLE TEST 1: balanced+aggressive / faster+conservative
REM   Pass1: balanced, aggressive, silero-v4.0, no enhancer
REM   Pass2: faster, conservative, silero-v3.1, no enhancer
REM   Merge: pass1_primary (default - pass1 as base, fill gaps from pass2)
REM ============================================================================
echo.
echo [E1] Ensemble: balanced+agg / faster+cons - silero40/s31, no enhancer
echo ----------------------------------------------------------------------------
set "TEST_NAME=E1_ens_bal-agg_faster-cons_s40-s31_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline balanced ^
    --pass1-sensitivity aggressive ^
    --pass1-speech-segmenter silero-v4.0 ^
    --pass1-speech-enhancer none ^
    --pass2-pipeline faster ^
    --pass2-sensitivity conservative ^
    --pass2-speech-segmenter silero-v3.1 ^
    --pass2-speech-enhancer none ^
    --merge-strategy pass1_primary ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "E1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "E1" "%E1_RESULT%"

REM ============================================================================
REM ENSEMBLE TEST 2: fast+balanced / transformers+balanced
REM   Pass1: fast, balanced, whisper-vad, no enhancer
REM   Pass2: transformers, balanced, none, no enhancer
REM   Merge: smart_merge (AI-like quality prioritization)
REM ============================================================================
echo.
echo [E2] Ensemble: fast+bal / transformers+bal - whispervad/none, no enhancer
echo ----------------------------------------------------------------------------
set "TEST_NAME=E2_ens_fast-bal_trans-bal_wvad-none_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline fast ^
    --pass1-sensitivity balanced ^
    --pass1-speech-segmenter whisper-vad ^
    --pass1-speech-enhancer none ^
    --pass2-pipeline transformers ^
    --pass2-sensitivity balanced ^
    --pass2-speech-segmenter none ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "E2_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "E2" "%E2_RESULT%"

REM ============================================================================
REM ENSEMBLE TEST 3: transformers+aggressive / balanced+conservative
REM   Pass1: transformers, aggressive, silero-v3.1, no enhancer
REM   Pass2: balanced, conservative, whisper-vad, no enhancer
REM   Merge: pass2_primary (pass2 as base, fill gaps from pass1)
REM ============================================================================
echo.
echo [E3] Ensemble: transformers+agg / balanced+cons - s31/wvad, no enhancer
echo ----------------------------------------------------------------------------
set "TEST_NAME=E3_ens_trans-agg_bal-cons_s31-wvad_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline transformers ^
    --pass1-sensitivity aggressive ^
    --pass1-speech-segmenter silero-v3.1 ^
    --pass1-speech-enhancer none ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity conservative ^
    --pass2-speech-segmenter whisper-vad ^
    --pass2-speech-enhancer none ^
    --merge-strategy pass2_primary ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "E3_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "E3" "%E3_RESULT%"

REM ============================================================================
REM ENSEMBLE TEST 4: faster+balanced / fast+aggressive
REM   Pass1: faster, balanced, none (no segmenter), no enhancer
REM   Pass2: fast, aggressive, silero-v4.0, no enhancer
REM   Merge: pass1_overlap (pass1 base with 30% overlap allowance)
REM ============================================================================
echo.
echo [E4] Ensemble: faster+bal / fast+agg - none/s40, no enhancer
echo ----------------------------------------------------------------------------
set "TEST_NAME=E4_ens_faster-bal_fast-agg_none-s40_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline faster ^
    --pass1-sensitivity balanced ^
    --pass1-speech-segmenter none ^
    --pass1-speech-enhancer none ^
    --pass2-pipeline fast ^
    --pass2-sensitivity aggressive ^
    --pass2-speech-segmenter silero-v4.0 ^
    --pass2-speech-enhancer none ^
    --merge-strategy pass1_overlap ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "E4_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "E4" "%E4_RESULT%"

REM ============================================================================
REM SINGLE-PASS TESTS (Non-Ensemble)
REM ============================================================================

REM ----------------------------------------------------------------------------
REM SINGLE TEST 1: balanced + aggressive (most detail capture)
REM ----------------------------------------------------------------------------
echo.
echo [S1] Single: balanced + aggressive (max detail, no enhancer)
echo ----------------------------------------------------------------------------
set "TEST_NAME=S1_single_balanced_aggressive_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --sensitivity aggressive ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "S1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "S1" "%S1_RESULT%"

REM ----------------------------------------------------------------------------
REM SINGLE TEST 2: faster + conservative (speed-focused, minimal false positives)
REM ----------------------------------------------------------------------------
echo.
echo [S2] Single: faster + conservative (speed-focused, no enhancer)
echo ----------------------------------------------------------------------------
set "TEST_NAME=S2_single_faster_conservative_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --mode faster ^
    --sensitivity conservative ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "S2_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "S2" "%S2_RESULT%"

REM ----------------------------------------------------------------------------
REM SINGLE TEST 3: transformers + balanced (HuggingFace pipeline)
REM ----------------------------------------------------------------------------
echo.
echo [S3] Single: transformers + balanced (HF pipeline, no enhancer)
echo ----------------------------------------------------------------------------
set "TEST_NAME=S3_single_transformers_balanced_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --mode transformers ^
    --sensitivity balanced ^
    --hf-scene auditok ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "S3_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "S3" "%S3_RESULT%"

REM ============================================================================
REM DIRECT-TO-ENGLISH TEST (Whisper translate task)
REM ============================================================================
echo.
echo [T1] Direct-to-English: fast + balanced (Whisper translate, no enhancer)
echo ----------------------------------------------------------------------------
set "TEST_NAME=T1_direct_english_fast_balanced_noenhance"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
set "TEST_TEMP_DIR=%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
set "TEST_LOG_FILE=%OUTPUT_DIR%\%TEST_RUN_ID%.log"
if not exist "%TEST_TEMP_DIR%" mkdir "%TEST_TEMP_DIR%"

echo   Logging to: %TEST_LOG_FILE%
echo   Running test... (output captured to log file)

call %WJ_CMD% "%TEST_FILE%" ^
    --mode fast ^
    --sensitivity balanced ^
    --language japanese ^
    --subs-language direct-to-english ^
    --debug ^
    --keep-temp ^
    --temp-dir "%TEST_TEMP_DIR%" ^
    --output-dir "%OUTPUT_DIR%" > "%TEST_LOG_FILE%" 2>&1
set "T1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"
call :REPORT_RESULT "T1" "%T1_RESULT%"

REM ============================================================================
REM Summary for this media file
REM ============================================================================
echo.
echo ============================================================================
echo  SUMMARY: %TEST_BASENAME%
echo ============================================================================
echo.
echo  Ensemble Tests (2-pass, no enhancer):
echo    E1 [bal+agg/faster+cons, s40/s31]:   %E1_RESULT%
echo    E2 [fast+bal/trans+bal, wvad/none]:  %E2_RESULT%
echo    E3 [trans+agg/bal+cons, s31/wvad]:   %E3_RESULT%
echo    E4 [faster+bal/fast+agg, none/s40]:  %E4_RESULT%
echo.
echo  Single-Pass Tests (no enhancer):
echo    S1 [balanced + aggressive]:    %S1_RESULT%
echo    S2 [faster + conservative]:    %S2_RESULT%
echo    S3 [transformers + balanced]:  %S3_RESULT%
echo.
echo  Translation Test (no enhancer):
echo    T1 [direct-to-english]:        %T1_RESULT%
echo.
echo ============================================================================

exit /b 0

REM ===========================================================================
REM MAIN: Process all media files in test directory
REM ===========================================================================
:MAIN
set "FOUND_MEDIA=0"
set "TOTAL_TESTS=0"
set "PASSED_TESTS=0"

for %%f in ("%TEST_MEDIA_DIR%\*.*") do call :MAYBE_RUN "%%~ff"

if "%FOUND_MEDIA%"=="0" (
    echo.
    echo ERROR: No audio/video files found in %TEST_MEDIA_DIR%
    echo Please add a media file ^(wav/mp3/mp4/mkv/...^) to the test_media folder.
    endlocal
    exit /b 1
)

echo.
echo ############################################################################
echo  ALL TESTS COMPLETE
echo ############################################################################
echo.
echo  Results saved to: %OUTPUT_DIR%
echo.
echo  Output SRT files:
dir /b "%OUTPUT_DIR%\*.srt" 2>nul || echo   (no SRT files found)
echo.
echo  Debug log files:
dir /b "%OUTPUT_DIR%\*.log" 2>nul || echo   (no log files found)
echo.

REM ============================================================================
REM ERROR SCANNING: Search all log files for error indicators
REM ============================================================================
echo ############################################################################
echo  SCANNING LOG FILES FOR ERRORS AND ISSUES
echo ############################################################################
echo.

set "ERROR_REPORT=%OUTPUT_DIR%\error_scan_report.txt"
set "ERRORS_FOUND=0"

echo WhisperJAV Test Suite - Error Scan Report > "%ERROR_REPORT%"
echo Generated: %date% %time% >> "%ERROR_REPORT%"
echo ============================================== >> "%ERROR_REPORT%"
echo. >> "%ERROR_REPORT%"

REM Define error patterns to search for
set "ERROR_PATTERNS=error exception traceback failed crash fatal critical aborted killed segfault oom"

echo Scanning for: %ERROR_PATTERNS%
echo.

for %%L in ("%OUTPUT_DIR%\*.log") do (
    set "LOG_FILE=%%~nxL"
    set "HAS_ISSUES=0"

    REM Search for each error pattern (case-insensitive)
    for %%P in (%ERROR_PATTERNS%) do (
        findstr /I /C:"%%P" "%%L" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            if !HAS_ISSUES! EQU 0 (
                echo   [!] Issues found in: !LOG_FILE!
                echo. >> "%ERROR_REPORT%"
                echo ============================================== >> "%ERROR_REPORT%"
                echo FILE: !LOG_FILE! >> "%ERROR_REPORT%"
                echo ============================================== >> "%ERROR_REPORT%"
                set "HAS_ISSUES=1"
                set "ERRORS_FOUND=1"
            )
            echo     - Found: %%P
            echo. >> "%ERROR_REPORT%"
            echo --- Matches for "%%P": --- >> "%ERROR_REPORT%"
            findstr /I /N /C:"%%P" "%%L" >> "%ERROR_REPORT%" 2>&1
        )
    )

    if !HAS_ISSUES! EQU 0 (
        echo   [OK] No issues in: !LOG_FILE!
    )
)

echo.
echo ============================================== >> "%ERROR_REPORT%"
echo END OF REPORT >> "%ERROR_REPORT%"
echo ============================================== >> "%ERROR_REPORT%"

if %ERRORS_FOUND% EQU 1 (
    echo.
    echo ############################################################################
    echo  WARNING: ERRORS OR ISSUES DETECTED IN LOG FILES
    echo ############################################################################
    echo.
    echo  Review the detailed report at:
    echo    %ERROR_REPORT%
    echo.
    echo  Or review individual log files in:
    echo    %OUTPUT_DIR%
    echo.
) else (
    echo.
    echo ############################################################################
    echo  SUCCESS: NO ERROR INDICATORS FOUND IN LOG FILES
    echo ############################################################################
    echo.
    echo  All tests completed without detected errors.
    echo  Full report saved to: %ERROR_REPORT%
    echo.
)

endlocal
exit /b 0

REM ===========================================================================
REM MAYBE_RUN: Check if file is a supported media type and run tests
REM ===========================================================================
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

REM ===========================================================================
REM COPY_SRT: Copy output SRT to stable per-test filename
REM   Priority: merged.whisperjav.srt > pass1.srt > *.whisperjav.srt
REM ===========================================================================
:COPY_SRT
set "CS_BASENAME=%~1"
set "CS_TESTNAME=%~2"
set "CS_FINAL_SRT="

REM Try merged output first (ensemble)
for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.merged.whisperjav.srt") do set "CS_FINAL_SRT=%%~fs"
REM Try pass1 fallback (ensemble artifacts)
if not defined CS_FINAL_SRT for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.pass1.srt") do set "CS_FINAL_SRT=%%~fs"
REM Try single-pass output
if not defined CS_FINAL_SRT for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.whisperjav.srt") do set "CS_FINAL_SRT=%%~fs"

if defined CS_FINAL_SRT (
    copy "!CS_FINAL_SRT!" "%OUTPUT_DIR%\%CS_TESTNAME%_%CS_BASENAME%.srt" >nul
    echo   Output: %CS_TESTNAME%_%CS_BASENAME%.srt
) else (
    echo   WARNING: No output SRT found for %CS_BASENAME%
)

exit /b 0

REM ===========================================================================
REM REPORT_RESULT: Display pass/fail status for a test
REM ===========================================================================
:REPORT_RESULT
set "RR_TEST=%~1"
set "RR_CODE=%~2"

if "%RR_CODE%"=="0" (
    echo   Result: PASSED
) else (
    echo   Result: FAILED ^(exit code: %RR_CODE%^)
)

exit /b 0
