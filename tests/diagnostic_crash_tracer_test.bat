@echo off
REM ============================================================================
REM Test crash tracer integration
REM ============================================================================
REM
REM This runs WhisperJAV with crash tracing enabled to capture state
REM during the ctranslate2 crash for analysis.
REM
REM Output will be in: crash_traces/crash_trace_*.jsonl
REM
REM ============================================================================

echo.
echo ============================================================================
echo  Crash Tracer Test
echo ============================================================================
echo.
echo  This will run WhisperJAV with --crash-trace enabled.
echo  Trace files will be written to: crash_traces\
echo.

cd /d C:\BIN\git\whisperJav_V1_Minami_Edition

REM Run with crash tracing on the test file
python -m whisperjav.main ^
    "test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav" ^
    --mode balanced ^
    --sensitivity aggressive ^
    --crash-trace ^
    --debug ^
    --keep-temp

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================================
echo  Test completed. Exit code: %EXIT_CODE%
echo ============================================================================

if %EXIT_CODE% neq 0 (
    echo.
    echo  Check crash_traces\ folder for trace files.
    echo.
    if exist crash_traces\*.crash_dump.json (
        echo  Found crash dump files:
        dir /b crash_traces\*.crash_dump.json
    )
)

echo.
echo  Trace files:
if exist crash_traces\*.jsonl (
    dir /b crash_traces\*.jsonl
) else (
    echo  (none found)
)

echo.
pause
