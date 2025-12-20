# Session State - Resume After PC Restart

## Date: 2025-12-19

## What We Were Doing

Forensic investigation of test suite crashes. No coding - deep analysis mode.

---

## Key Findings So Far

### Issue B: Worker Crash (PRIMARY FOCUS)

- **Exit code:** 3221225620 (STATUS_INTEGER_DIVIDE_BY_ZERO)
- **When:** E1 test (balanced + aggressive) on SONE-966 audio file
- **Pass 1 crashed, Pass 2 succeeded** on same file

### Root Cause Hypotheses

| ID | Hypothesis | Status |
|----|-----------|--------|
| H1 | Crash triggered by balanced+aggressive+SONE-966 combination | Testing |
| H2 | Low no_speech_threshold (0.22) causes edge case | Untested |
| H3 | beam_size=3 with word_timestamps creates vulnerable code path | Untested |
| H-FRAG | VRAM fragmentation from prior del/gc/empty_cache | **TESTING NOW** |

### Evidence Against Fragmentation (but not conclusive)

- Pass 2 succeeded immediately after Pass 1 crashed
- Pass 2 used larger model (large-v3 vs large-v2)
- Same test passed on other audio files

### Evidence That Could Support Fragmentation

- 4.5GB VRAM was in use before isolation test
- Need clean GPU state to rule out

---

## Test Files Created

1. `tests\diagnostic_S1_isolation_test.bat` - **RUN THIS AFTER RESTART**
   - Runs S1 (balanced+aggressive) on SONE-966 in isolation
   - Fresh process, no prior GPU ops
   - Will determine if fragmentation is the cause

2. `tests\diagnostic_S1_SONE966_crash_test.bat`
   - Runs S1 five times to test reproducibility
   - Run after isolation test if needed

3. `tests\run_whisperjav_expanded_tests_v2-noenhancer-debug.bat`
   - Main test suite (fixed - removed invalid --speech-enhancer flag)

---

## After Restart - Do This

1. **Open fresh Anaconda Prompt**

2. **Activate environment:**
   ```
   conda activate WJ
   ```

3. **Navigate to project:**
   ```
   cd C:\BIN\git\whisperJav_V1_Minami_Edition
   ```

4. **Verify GPU is clean:**
   ```
   nvidia-smi
   ```
   - Should show ~500MB-1.5GB memory usage (display only)
   - If higher, something is wrong

5. **Run isolation test:**
   ```
   .\tests\diagnostic_S1_isolation_test.bat
   ```

6. **Share results with Claude** - Copy the summary output

---

## Interpretation of Isolation Test Results

### If CRASH occurs (exit code 3221225620):
- **Fragmentation hypothesis RULED OUT**
- Bug is in faster-whisper/ctranslate2
- Next: Test with beam_size=1 or higher no_speech_threshold

### If NO CRASH (exit code 0):
- **Fragmentation hypothesis SUPPORTED**
- Prior GPU operations cause the issue
- Next: Investigate memory management in whisperjav

---

## Test Audio File Location

```
C:\BIN\git\whisperJav_V1_Minami_Edition\test_media\173_acceptance_test\SONE-966-15_sec_test-966-00_01_45-00_01_59.wav
```

---

## Other Issues Found (Lower Priority)

### Issue A: UnicodeEncodeError in logging
- Japanese characters fail to encode on Windows console
- Fixed arrows (→ to ->), but Japanese text still fails
- Needs UTF-8 logging configuration
- **NOT blocking - tests still pass**

---

## To Resume Conversation with Claude

After restart, tell Claude:

> "I restarted my PC. Please read the file `tests\SESSION_STATE_RESUME.md` to restore context. Then I will share the isolation test results."

---
