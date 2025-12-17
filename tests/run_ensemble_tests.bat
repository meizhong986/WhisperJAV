@echo off
REM ============================================================================
REM WhisperJAV Two-Pass Ensemble Test Suite
REM ============================================================================
REM Tests various combinations of speech segmenters, enhancers, and models
REM Each test outputs:
REM   - Debug logs (console + log file)
REM   - Parameter dump (JSON, before processing) - separate diagnostic run
REM   - Parameter trace (JSONL, during processing)
REM   - Output SRT file (renamed with test identifier)
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "TEST_MEDIA_DIR=.\test_media\173_acceptance_test"
set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "OUTPUT_DIR=.\test_results\ensemble_%TIMESTAMP%"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================================
echo  WhisperJAV Two-Pass Ensemble Test Suite
echo ============================================================================
echo  Test Media: %TEST_MEDIA_DIR%
echo  Output Dir: %OUTPUT_DIR%
echo  Timestamp:  %TIMESTAMP%
echo ============================================================================
echo.

goto :MAIN

:RUN_SUITE
set "TEST_FILE=%~1"
set "TEST_BASENAME=%~n1"
set "TEST1_RESULT="
set "TEST2_RESULT="
set "TEST3_RESULT="
set "TEST4_RESULT="

echo.
echo Using test file: %TEST_FILE%
echo Basename: %TEST_BASENAME%
echo.

REM ============================================================================
REM TEST 1: Balanced + Fidelity, Silero v4.0, No Enhancer
REM ============================================================================
echo.
echo ============================================================================
echo TEST 1: Pass1=balanced, Pass2=fidelity
echo         Segmenter: silero-v4.0, Enhancer: none
echo ============================================================================
echo.

set "TEST_NAME=test1_balanced_fidelity_silero4_none"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"

REM Step 1a: Dump parameters (diagnostic only, exits without processing)
echo [Step 1a] Dumping parameters...
whisperjav "%TEST_FILE%" ^
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
    --dump-params "%OUTPUT_DIR%\%TEST_RUN_ID%_dump.json"

REM Step 1b: Run actual transcription
echo [Step 1b] Running transcription...
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
whisperjav "%TEST_FILE%" ^
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
    --debug ^
    --trace-params "%OUTPUT_DIR%\%TEST_RUN_ID%_trace.jsonl" ^
    --log-file "%OUTPUT_DIR%\%TEST_RUN_ID%.log" ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"

set "TEST1_RESULT=%ERRORLEVEL%"

REM Copy merged/pass1 output to stable per-test filename
set "FINAL_SRT="
for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.merged.whisperjav.srt") do set "FINAL_SRT=%%~fs"
if not defined FINAL_SRT for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.pass1.srt") do set "FINAL_SRT=%%~fs"
if defined FINAL_SRT (
    copy "!FINAL_SRT!" "%OUTPUT_DIR%\%TEST_NAME%_%TEST_BASENAME%.srt" >nul
    echo SRT saved as: %TEST_NAME%_%TEST_BASENAME%.srt
)

if %TEST1_RESULT% EQU 0 (
    echo TEST 1: PASSED
) else (
    echo TEST 1: FAILED (exit code: %TEST1_RESULT%)
)

REM ============================================================================
REM TEST 2: Balanced + Balanced, Silero v3.1, ClearVoice 16kHz Enhancer
REM NOTE: Using FRCRN_SE_16K instead of MossFormer2_SE_48K due to VRAM
REM       fragmentation issues on 8GB GPUs. See docs/known_issues/
REM ============================================================================
echo.
echo ============================================================================
echo TEST 2: Pass1=balanced, Pass2=balanced
echo         Segmenter: silero-v3.1, Enhancer: clearvoice (16kHz)
echo ============================================================================
echo.

set "TEST_NAME=test2_balanced_balanced_silero31_clearvoice16k"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"

REM Step 2a: Dump parameters
echo [Step 2a] Dumping parameters...
whisperjav "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline balanced ^
    --pass1-sensitivity aggressive ^
    --pass1-speech-segmenter silero-v3.1 ^
    --pass1-speech-enhancer clearvoice:FRCRN_SE_16K ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity balanced ^
    --pass2-speech-segmenter silero-v3.1 ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --dump-params "%OUTPUT_DIR%\%TEST_RUN_ID%_dump.json"

REM Step 2b: Run actual transcription
echo [Step 2b] Running transcription...
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
whisperjav "%TEST_FILE%" ^
    --ensemble ^
    --pass1-pipeline balanced ^
    --pass1-sensitivity aggressive ^
    --pass1-speech-segmenter silero-v3.1 ^
    --pass1-speech-enhancer clearvoice:FRCRN_SE_16K ^
    --pass2-pipeline balanced ^
    --pass2-sensitivity balanced ^
    --pass2-speech-segmenter silero-v3.1 ^
    --pass2-speech-enhancer none ^
    --merge-strategy smart_merge ^
    --debug ^
    --trace-params "%OUTPUT_DIR%\%TEST_RUN_ID%_trace.jsonl" ^
    --log-file "%OUTPUT_DIR%\%TEST_RUN_ID%.log" ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"

set "TEST2_RESULT=%ERRORLEVEL%"

REM Copy merged/pass1 output to stable per-test filename
set "FINAL_SRT="
for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.merged.whisperjav.srt") do set "FINAL_SRT=%%~fs"
if not defined FINAL_SRT for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.pass1.srt") do set "FINAL_SRT=%%~fs"
if defined FINAL_SRT (
    copy "!FINAL_SRT!" "%OUTPUT_DIR%\%TEST_NAME%_%TEST_BASENAME%.srt" >nul
    echo SRT saved as: %TEST_NAME%_%TEST_BASENAME%.srt
)

if %TEST2_RESULT% EQU 0 (
    echo TEST 2: PASSED
) else (
    echo TEST 2: FAILED (exit code: %TEST2_RESULT%)
)

REM ============================================================================
REM TEST 3: Fidelity + Balanced, TEN Segmenter, FFmpeg-DSP Enhancer
REM ============================================================================
echo.
echo ============================================================================
echo TEST 3: Pass1=fidelity, Pass2=balanced
echo         Segmenter: ten, Enhancer: ffmpeg-dsp
echo ============================================================================
echo.

set "TEST_NAME=test3_fidelity_balanced_ten_ffmpeg"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"

REM Step 3a: Dump parameters
echo [Step 3a] Dumping parameters...
whisperjav "%TEST_FILE%" ^
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
    --dump-params "%OUTPUT_DIR%\%TEST_RUN_ID%_dump.json"

REM Step 3b: Run actual transcription
echo [Step 3b] Running transcription...
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
whisperjav "%TEST_FILE%" ^
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
    --debug ^
    --trace-params "%OUTPUT_DIR%\%TEST_RUN_ID%_trace.jsonl" ^
    --log-file "%OUTPUT_DIR%\%TEST_RUN_ID%.log" ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"

set "TEST3_RESULT=%ERRORLEVEL%"

REM Copy merged/pass1 output to stable per-test filename
set "FINAL_SRT="
for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.merged.whisperjav.srt") do set "FINAL_SRT=%%~fs"
if not defined FINAL_SRT for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.pass1.srt") do set "FINAL_SRT=%%~fs"
if defined FINAL_SRT (
    copy "!FINAL_SRT!" "%OUTPUT_DIR%\%TEST_NAME%_%TEST_BASENAME%.srt" >nul
    echo SRT saved as: %TEST_NAME%_%TEST_BASENAME%.srt
)

if %TEST3_RESULT% EQU 0 (
    echo TEST 3: PASSED
) else (
    echo TEST 3: FAILED (exit code: %TEST3_RESULT%)
)

REM ============================================================================
REM TEST 4: Transformers + Balanced, Whisper-VAD, ZipEnhancer
REM ============================================================================
echo.
echo ============================================================================
echo TEST 4: Pass1=transformers, Pass2=balanced
echo         Segmenter: whisper-vad, Enhancer: zipenhancer
echo ============================================================================
echo.

set "TEST_NAME=test4_transformers_balanced_whispervad_zip"
set "TEST_RUN_ID=%TEST_NAME%_%TEST_BASENAME%"

REM Step 4a: Dump parameters
echo [Step 4a] Dumping parameters...
whisperjav "%TEST_FILE%" ^
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
    --dump-params "%OUTPUT_DIR%\%TEST_RUN_ID%_dump.json"

REM Step 4b: Run actual transcription
echo [Step 4b] Running transcription...
if not exist "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" mkdir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp"
whisperjav "%TEST_FILE%" ^
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
    --debug ^
    --trace-params "%OUTPUT_DIR%\%TEST_RUN_ID%_trace.jsonl" ^
    --log-file "%OUTPUT_DIR%\%TEST_RUN_ID%.log" ^
    --temp-dir "%OUTPUT_DIR%\%TEST_RUN_ID%_temp" ^
    --output-dir "%OUTPUT_DIR%"

set "TEST4_RESULT=%ERRORLEVEL%"

REM Copy merged/pass1 output to stable per-test filename
set "FINAL_SRT="
for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.merged.whisperjav.srt") do set "FINAL_SRT=%%~fs"
if not defined FINAL_SRT for %%s in ("%OUTPUT_DIR%\%TEST_BASENAME%.*.pass1.srt") do set "FINAL_SRT=%%~fs"
if defined FINAL_SRT (
    copy "!FINAL_SRT!" "%OUTPUT_DIR%\%TEST_NAME%_%TEST_BASENAME%.srt" >nul
    echo SRT saved as: %TEST_NAME%_%TEST_BASENAME%.srt
)

if %TEST4_RESULT% EQU 0 (
    echo TEST 4: PASSED
) else (
    echo TEST 4: FAILED (exit code: %TEST4_RESULT%)
)

REM ============================================================================
REM Summary
REM ============================================================================
echo.
echo ============================================================================
echo  TEST SUITE COMPLETE
echo ============================================================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo.
echo Test Results:
echo   TEST 1 (balanced+fidelity, silero-v4.0, none):       %TEST1_RESULT%
echo   TEST 2 (balanced+balanced, silero-v3.1, clearvoice-16k): %TEST2_RESULT%
echo   TEST 3 (fidelity+balanced, ten, ffmpeg-dsp):         %TEST3_RESULT%
echo   TEST 4 (transformers+balanced, whisper-vad, zip):    %TEST4_RESULT%
echo.
echo Files per test:
echo   - *_dump.json          : Resolved parameters (before processing)
echo   - *_trace.jsonl        : Parameter snapshots (during processing)
echo   - *.log                : Debug log file
echo   - *_%TEST_BASENAME%.srt : Output subtitle (with test identifier)
echo.
echo To review parameters:
echo   - Open *_dump.json for pre-resolved config
echo   - Open *_trace.jsonl for real-time parameter flow
echo   - Search *.log for "Final faster-whisper parameters"
echo.

echo.
echo Output files:
dir /b "%OUTPUT_DIR%"

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
