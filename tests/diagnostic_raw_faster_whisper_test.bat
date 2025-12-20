@echo off
REM ============================================================================
REM Raw Faster-Whisper Crash Reproduction Test
REM ============================================================================
REM
REM This test uses raw faster-whisper WITHOUT WhisperJAV to determine if the
REM crash is in faster-whisper/ctranslate2 or in WhisperJAV's audio handling.
REM
REM Expected output:
REM   - If crashes with raw faster-whisper: bug is in faster-whisper/ctranslate2
REM   - If passes with raw faster-whisper: bug is in WhisperJAV's handling
REM
REM ============================================================================

echo.
echo ============================================================================
echo  Raw Faster-Whisper Crash Reproduction Test
echo ============================================================================
echo.

cd /d C:\BIN\git\whisperJav_V1_Minami_Edition

REM Clear old trace files
if exist raw_fw_trace.log del raw_fw_trace.log
if exist raw_fw_results.json del raw_fw_results.json

echo Running test...
echo.

python tests\diagnostic_raw_faster_whisper_test.py

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================================
echo  Test completed. Exit code: %EXIT_CODE%
echo ============================================================================

if %EXIT_CODE% neq 0 (
    echo.
    echo  CRASH DETECTED - Exit code %EXIT_CODE%
    if %EXIT_CODE% equ -1073741676 (
        echo  STATUS_INTEGER_DIVIDE_BY_ZERO confirmed
    )
    echo.
    echo  Check raw_fw_trace.log for last recorded state
)

echo.
echo  Trace log:
if exist raw_fw_trace.log (
    type raw_fw_trace.log
) else (
    echo  (no trace log found)
)

echo.
pause
