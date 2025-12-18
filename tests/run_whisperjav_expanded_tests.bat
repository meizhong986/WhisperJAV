@echo off
REM ============================================================================
REM WhisperJAV Expanded Automated Test Suite (No debug/log switches)
REM ============================================================================
REM - Runs the existing two-pass ensemble suite (without --debug/--log-level/--log-file)
REM - Adds additional NON-ENSEMBLE tests across sensitivity levels
REM - Adds a Whisper direct-to-English test (translate task via Whisper)
REM - Does NOT test AI translation (--translate)
REM
REM Output per test:
REM   - SRT copied to a stable per-test filename: <test>_<basename>.srt
REM   - Per-test temp dir: <output>\<test>_<basename>_temp
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "TEST_MEDIA_DIR=.\test_media\173_acceptance_test"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=.\test_results\expanded_%TIMESTAMP%"

REM Always run whisperjav inside conda env WJ
set "WJ_CMD=conda run -n WJ whisperjav"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================================
echo  WhisperJAV Expanded Automated Test Suite
echo ============================================================================
echo  Test Media: %TEST_MEDIA_DIR%
echo  Output Dir: %OUTPUT_DIR%
echo  Timestamp:  %TIMESTAMP%
echo ============================================================================
echo.

goto :MAIN

REM ---------------------------------------------------------------------------
REM Test runner for a single file
REM ---------------------------------------------------------------------------
:RUN_SUITE
set "TEST_FILE=%~1"
set "TEST_BASENAME=%~n1"

set "E1_RESULT="
set "E2_RESULT="
set "E3_RESULT="
set "E4_RESULT="

set "S1_RESULT="
set "S2_RESULT="
set "S3_RESULT="
set "S4_RESULT="
set "S5_RESULT="
set "T1_RESULT="

echo.
echo Using test file: %TEST_FILE%
echo Basename: %TEST_BASENAME%
echo.

REM ============================================================================
REM ENSEMBLE TESTS (no --debug / no --log-level / no --log-file)
REM ============================================================================

REM ENSEMBLE 1
echo.
echo ============================================================================
echo ENSEMBLE 1: Pass1=balanced, Pass2=fidelity
echo            Segmenter: silero-v4.0, Enhancer: none
echo ============================================================================
echo.
set "TEST_NAME=ens1_balanced_fidelity_silero4_none"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline balanced ^
    --pass1-sensitivity balanced ^
    --pass1-speech-segmenter silero-v4.0 ^
    --pass1-speech-enhancer none ^
    --pass2-pipeline fidelity ^
    --pass2-sensitivity conservative ^
    --pass2-speech-segmenter silero-v4.0 ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "E1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM ENSEMBLE 2
echo.
echo ============================================================================
echo ENSEMBLE 2: Pass1=balanced, Pass2=balanced
echo            Segmenter: silero-v3.1, Enhancer: clearvoice
echo ============================================================================
echo.
set "TEST_NAME=ens2_balanced_balanced_silero31_clearvoice"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline balanced ^
    --pass1-sensitivity aggressive ^
    --pass1-speech-segmenter silero-v3.1 ^
    --pass1-speech-enhancer clearvoice ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity balanced ^
    --pass2-speech-segmenter silero-v3.1 ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "E2_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM ENSEMBLE 3
echo.
echo ============================================================================
echo ENSEMBLE 3: Pass1=fidelity, Pass2=balanced
echo            Segmenter: ten, Enhancer: ffmpeg-dsp
echo ============================================================================
echo.
set "TEST_NAME=ens3_fidelity_balanced_ten_ffmpeg"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline fidelity ^
    --pass1-sensitivity balanced ^
    --pass1-speech-segmenter ten ^
    --pass1-speech-enhancer ffmpeg-dsp ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity aggressive ^
    --pass2-speech-segmenter ten ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "E3_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM ENSEMBLE 4
echo.
echo ============================================================================
echo ENSEMBLE 4: Pass1=transformers, Pass2=balanced
echo            Segmenter: whisper-vad, Enhancer: zipenhancer
echo ============================================================================
echo.
set "TEST_NAME=ens4_transformers_balanced_whispervad_zip"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline transformers ^
    --pass1-sensitivity balanced ^
    --pass1-speech-segmenter whisper-vad ^
    --pass1-speech-enhancer zipenhancer ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity balanced ^
    --pass2-speech-segmenter whisper-vad ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "E4_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM ============================================================================
REM SINGLE-PASS (NON-ENSEMBLE) TESTS
REM NOTE: Uses --mode + --sensitivity; avoids --translate (AI)
REM ============================================================================

REM SINGLE 1: balanced + conservative
echo.
echo ============================================================================
echo SINGLE 1: mode=balanced, sensitivity=conservative
echo ============================================================================
echo.
set "TEST_NAME=single_balanced_conservative"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --sensitivity conservative ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "S1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM SINGLE 2: balanced + balanced
echo.
echo ============================================================================
echo SINGLE 2: mode=balanced, sensitivity=balanced
echo ============================================================================
echo.
set "TEST_NAME=single_balanced_balanced"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --sensitivity balanced ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "S2_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM SINGLE 3: balanced + aggressive
echo.
echo ============================================================================
echo SINGLE 3: mode=balanced, sensitivity=aggressive
echo ============================================================================
echo.
set "TEST_NAME=single_balanced_aggressive"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --sensitivity aggressive ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "S3_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM SINGLE 4: fidelity + balanced
echo.
echo ============================================================================
echo SINGLE 4: mode=fidelity, sensitivity=balanced
echo ============================================================================
echo.
set "TEST_NAME=single_fidelity_balanced"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode fidelity ^
    --sensitivity balanced ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "S4_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM SINGLE 5: transformers + balanced
echo.
echo ============================================================================
echo SINGLE 5: mode=transformers, sensitivity=balanced
echo ============================================================================
echo.
set "TEST_NAME=single_transformers_balanced"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode transformers ^
    --sensitivity balanced ^
    --hf-scene none ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "S5_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM DIRECT-TO-ENGLISH (Whisper translate)
echo.
echo ============================================================================
echo TRANSLATE 1: direct-to-English via Whisper (task=translate)
echo ============================================================================
echo.
set "TEST_NAME=translate_direct_to_english"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
call %WJ_CMD% "%TEST_FILE%" ^
    --mode balanced ^
    --language japanese ^
    --task translate ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"
set "T1_RESULT=%ERRORLEVEL%"
call :COPY_SRT "%TEST_BASENAME%" "%TEST_NAME%"

REM ============================================================================
REM Summary for this media file
REM ============================================================================
echo.
echo ============================================================================
echo  FILE COMPLETE: %TEST_BASENAME%
echo ============================================================================
echo Ensemble:
echo   E1: %E1_RESULT%
echo   E2: %E2_RESULT%
echo   E3: %E3_RESULT%
echo   E4: %E4_RESULT%
echo Single-pass:
echo   S1: %S1_RESULT%
echo   S2: %S2_RESULT%
echo   S3: %S3_RESULT%
echo   S4: %S4_RESULT%
echo   S5: %S5_RESULT%
echo Translate:
echo   T1: %T1_RESULT%
echo.

exit /b 0

:MAIN
REM Run the full suite for each supported media file in TEST_MEDIA_DIR
set "FOUND_MEDIA=0"
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

endlocal
exit /b 0

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
REM Helper: copy output SRT for a basename to a stable per-test filename
REM Priorities:
REM   1) merged.whisperjav.srt (ensemble)
REM   2) pass1.srt (ensemble fallback artifacts)
REM   3) *.whisperjav.srt (single-pass output)
REM ---------------------------------------------------------------------------
:COPY_SRT
set "CS_BASENAME=%~1"
set "CS_TESTNAME=%~2"
set "CS_FINAL_SRT="

for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.merged.whisperjav.srt") do set "CS_FINAL_SRT=%%~fs"
if not defined CS_FINAL_SRT for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.pass1.srt") do set "CS_FINAL_SRT=%%~fs"
if not defined CS_FINAL_SRT for %%s in ("%OUTPUT_DIR%\%CS_BASENAME%.*.whisperjav.srt") do set "CS_FINAL_SRT=%%~fs"

if defined CS_FINAL_SRT (
    copy "!CS_FINAL_SRT!" "%OUTPUT_DIR%\%CS_TESTNAME%_%CS_BASENAME%.srt" >nul
    echo SRT saved as: %CS_TESTNAME%_%CS_BASENAME%.srt
) else (
    echo WARNING: No output SRT found to copy for %CS_BASENAME%
)

exit /b 0
