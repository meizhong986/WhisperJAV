@echo off
REM ============================================================================
REM DIAGNOSTIC: Task Override Mutation Test
REM ============================================================================
REM
REM Tests whether mutating whisper_params AFTER model initialization causes
REM crashes due to stale references in ctranslate2 C++ layer.
REM
REM The hypothesis:
REM   - WhisperJAV keeps ASR alive via _IMMORTAL_ASR_REFERENCE
REM   - whisper_params dict is passed to ctranslate2 at model init
REM   - Task override mutates whisper_params['task'] AFTER init
REM   - ctranslate2 may hold reference to original dict contents
REM   - Mutation causes memory corruption -> STATUS_INTEGER_DIVIDE_BY_ZERO
REM
REM ============================================================================

echo.
echo ============================================================================
echo  DIAGNOSTIC: Task Override Mutation Test
echo ============================================================================
echo.
echo  Testing if shared state mutation causes ctranslate2 crashes.
echo.

cd /d %~dp0\..

python tests\diagnostic_task_override_mutation_test.py %*

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================================
echo  Test completed. Exit code: %EXIT_CODE%
echo ============================================================================

if %EXIT_CODE% neq 0 (
    echo.
    echo  Non-zero exit may indicate a crash - check the hypothesis!
)

echo.
pause
