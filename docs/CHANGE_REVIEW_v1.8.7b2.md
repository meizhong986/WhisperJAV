# Change Review Document — v1.8.7b2 Proposed Fixes

> Date: 2026-03-09
> Author: Senior Architect (Claude)
> Status: PENDING REVIEW — No changes should be merged without reviewer approval
> Scope: Groups B, C, D from the v1.8.7b1 Issue Analysis

---

## Table of Contents

1. [Group B — Custom Provider API Key TypeError (#143)](#group-b)
2. [Group D — MPS Beam Search IndexError (#198)](#group-d)
3. [Group C — Repetition Hallucination (#209)](#group-c)
4. [Groups Not Yet Addressed](#not-addressed)

---

<a name="group-b"></a>
## 1. Group B — Custom Provider API Key TypeError (#143)

### i1: Problem Description

**Issue**: #143 (OdinShiva)
**Platform**: Windows, Ollama running locally with `qwen3.5:4b`
**GUI Configuration**: Provider = "Custom (OpenAI-compatible)", API Key field = empty, Endpoint = `http://localhost:11434/v1`, Model = `qwen3.5:4b`

The user clicks "Test Connection" in the GUI. Instead of testing the connection or gracefully reporting that no key is needed, the GUI shows:

```
Connection to custom failed: str expected, not NoneType
```

Ollama, LM Studio, and other local OpenAI-compatible servers do NOT require API keys. The user did everything correctly but cannot use the translation feature.

### i2: Final Diagnostics

The error `str expected, not NoneType` is a Python `TypeError` raised by `os.getenv(None)`. I traced the call path:

1. User clicks "Test Connection" in GUI
2. JavaScript calls `pywebview.api.test_provider_connection('custom', null)`
3. Python method `test_provider_connection()` in `api.py:3115`
4. Line 3145 (BEFORE my change): `key = api_key or os.getenv(config.get('env_var', ''))`
5. `config.get('env_var', '')` → returns `None` (NOT the default `''`, because the key EXISTS in the dict with value `None`)
6. `os.getenv(None)` → raises `TypeError: str expected, not NoneType`
7. Caught by `except Exception as e:` at line 3184, returned as `{"success": False, "error": "str expected, not NoneType"}`
8. GUI formats this as `Connection to custom failed: str expected, not NoneType`

**Additional finding**: The same bug exists in `cli.py:467`:
```python
api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.getenv(provider_config['env_var'])
```
For custom provider, `provider_config['env_var']` is `None`, so `os.getenv(None)` would crash here too if no `--api-key` is provided. This path is triggered when the GUI launches translation via subprocess (which goes through `cli.py`).

### i3: Final Root Cause

**Architectural cause**: The `PROVIDER_CONFIGS` dict in `providers.py` defines `custom.env_var = None` (line 55). This is semantically correct — custom providers don't have a standard env var. But TWO code paths use `os.getenv(provider_config['env_var'])` without guarding against `None`:

1. `api.py:3145` (GUI connection test) — `config.get('env_var', '')` doesn't protect because the key EXISTS with value `None`
2. `cli.py:467` (CLI translation) — direct dict access `provider_config['env_var']` passes `None` to `os.getenv()`

The `service.py` path (line 50-75) does NOT have this bug — its `_resolve_api_key()` function properly checks `if env_var:` before calling `os.getenv()`. And `service.py:290-291` explicitly handles `local`/`custom` with `resolved_api_key = api_key or ''`.

PySubtrans's `CustomClient` (line 27-28 in its source) already handles empty API keys gracefully:
```python
if self.api_key:
    self.headers['Authorization'] = f"Bearer {self.api_key}"
```
So passing `api_key=''` to PySubtrans works correctly — no Authorization header is sent, which is what Ollama expects.

### i4: Changes Decided

**Change 1 — `api.py` connection test**: Add `custom` to the early-return group alongside `local`. Both are local server providers that don't need API key validation. The "Test Connection" button for custom providers returns success immediately, same as local.

**Concern with this approach**: The early return means "Test Connection" doesn't actually ping the Ollama/LM Studio server. The endpoint URL is available in the GUI but the `test_provider_connection` method doesn't receive it as a parameter — it only receives `provider` and `api_key`. Changing the method signature would require frontend JS changes too. The connection will be validated when the user actually starts translation.

**Change 2 — `api.py` env_var safety**: Make the `os.getenv()` call safe for any provider with `None` env_var, not just custom. This is defensive coding for future providers.

**Previously missing change (NOW FIXED)**: `cli.py:464-470` had the same `os.getenv(None)` bug for custom provider. See File 2 below.

### i5: Code Changed

**File 1: `whisperjav/webview_gui/api.py`** — method `test_provider_connection()` (~line 3136)

BEFORE:
```python
# Local LLM provider doesn't need API key testing
if provider == 'local':
    return {
        "success": True,
        "message": "Local LLM provider - no API key required",
        "local": True
    }

# Get API key
key = api_key or os.getenv(config.get('env_var', ''))
if not key:
    return {
        "success": False,
        "error": f"No API key. Set {config.get('env_var')} or provide key."
    }
```

AFTER:
```python
# Local/Custom providers don't require API keys
if provider in ('local', 'custom'):
    return {
        "success": True,
        "message": f"{provider.capitalize()} provider - no API key required",
        "local": provider == 'local'
    }

# Get API key
env_var = config.get('env_var') or ''
key = api_key or (os.getenv(env_var) if env_var else None)
if not key:
    return {
        "success": False,
        "error": f"No API key. Set {config.get('env_var')} or provide key."
    }
```

**File 2: `whisperjav/translate/cli.py`** (~line 464)

BEFORE:
```python
# Get API key (not needed for local provider)
if provider_name == 'local':
    api_key = None
else:
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.getenv(provider_config['env_var'])
    if not api_key:
        print(f"Error: API key not found. Set {provider_config['env_var']} or use --api-key", file=sys.stderr)
        sys.exit(1)
```

AFTER:
```python
# Get API key (not needed for local/custom providers — Ollama, LM Studio, etc.)
if provider_name in ('local', 'custom'):
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else ''
else:
    env_var = provider_config.get('env_var')
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else (os.getenv(env_var) if env_var else None)
    if not api_key:
        print(f"Error: API key not found. Set {env_var} or use --api-key", file=sys.stderr)
        sys.exit(1)
```

Changes:
- `custom` added alongside `local` — both skip the API key requirement
- For custom, if `--api-key` is provided it's used; otherwise defaults to `''` (which PySubtrans CustomClient handles correctly — no Authorization header sent)
- The `else` branch now guards `os.getenv()` with `if env_var` to prevent `os.getenv(None)` for any future provider with `env_var=None`

### i6: Impact of Changes

**Positive impacts**:
- Users can select "Custom" provider with empty API key and the GUI no longer shows a cryptic TypeError
- Ollama, LM Studio, LocalAI, and other keyless servers are unblocked for translation

**Negative impacts / concerns**:
- "Test Connection" button for custom provider now always returns success without actually testing connectivity. This may confuse users who have the wrong endpoint URL — they'll see "success" in the GUI but fail when translation starts. However, this matches the existing behavior for `local` provider.
- The `"local": provider == 'local'` field in the return dict is a minor inconsistency — for custom provider, `local` will be `False`, but the message says "no API key required". The frontend JS may use this field. (Low risk — the field is mainly used for UI display.)

**Reviewer questions**:
1. Should the custom provider's "Test Connection" actually attempt to reach the endpoint? This would require passing the endpoint URL to `test_provider_connection()` and updating the JS call — a larger change.
2. For the `cli.py` change, `custom` with `--api-key` provided still passes the key through to PySubtrans. Is this correct for services like OpenRouter-compatible proxies that DO need keys but use the "custom" provider?

---

<a name="group-d"></a>
## 2. Group D — MPS Beam Search IndexError (#198)

### i1: Problem Description

**Issue**: #198 (francetoastVN)
**Platform**: macOS M1 (Apple Silicon), Transformers mode
**Model**: `kotoba-tech/kotoba-whisper-bilingual-v1.0`
**Settings**: MPS device, float16 dtype, batch size 8, 2-hour movie

The user runs transcription in Transformers mode. Audio extracts successfully (7348.9s), model loads on MPS successfully, but transcription crashes 37 seconds later with:

```
IndexError: index 1077827584 is out of bounds for dimension 0 with size 40
```

The traceback originates in HuggingFace Transformers library:
```
transformers/models/whisper/generation_whisper.py:1172
  in split_by_batch_index:
    return [v[beam_idx].cpu() for (v, beam_idx) in zip(values, beam_indices[batch_idx][: len(values)])]
```

### i2: Final Diagnostics

The index value `1077827584` = `0x40404040` in hexadecimal. This is a classic "fill pattern" indicating **uninitialized GPU memory**. The MPS (Metal Performance Shaders) backend on Apple Silicon sometimes returns garbage values in beam search index tensors.

Key observations from the log:
- Audio extracts fine → not a file/format issue
- Model loads on MPS → not a model compatibility issue
- Crash occurs on first chunk's beam search postprocessing → the error happens AFTER inference, during output tensor indexing
- The garbage index (`0x40404040`) is consistent with uninitialized Metal GPU buffer reads
- The error is in `transformers` library code, not WhisperJAV code

This is a **known class of MPS backend issues in PyTorch**. MPS is still maturing and has various edge cases with memory management, particularly with long audio files that produce large beam search tensors.

### i3: Final Root Cause

**Upstream library bug** in HuggingFace Transformers' Whisper implementation on MPS.

The `generate_with_fallback()` → `_postprocess_outputs()` → `split_by_batch_index()` call chain reads `beam_indices` from MPS tensor output. On Apple Silicon, the `beam_indices` tensor sometimes contains uninitialized memory values instead of valid beam indices. When the code tries to index into a tensor of size 40 using index 1077827584, it crashes with `IndexError`.

This is NOT a WhisperJAV bug. WhisperJAV correctly configures the model, device, and dtype. The failure is inside the `transformers` library's beam search postprocessing step.

### i4: Changes Decided

Add a defensive `except IndexError` handler in `transformers_asr.py` that:
1. Detects if the error occurred on MPS device
2. Logs a clear warning explaining this is an upstream bug
3. Changes device to CPU and dtype to float32
4. Unloads and reloads the model on CPU
5. Retries the transcription
6. If the error occurs on a non-MPS device, re-raises it (since that would be a genuine bug)

This follows the **existing pattern** already established in the same method: the `except torch.cuda.OutOfMemoryError` handler (lines 293-309) does the same unload/reload/retry approach for CUDA OOM errors.

### i5: Code Changed

**File: `whisperjav/modules/transformers_asr.py`** — method `transcribe()`, after the existing CUDA OOM handler (~line 311)

BEFORE (nothing after the CUDA OOM handler):
```python
        except torch.cuda.OutOfMemoryError:
            # ... existing CUDA OOM handling ...
            logger.info(f"Retry successful with batch_size={self.batch_size} (was {original_batch})")

        process_time = time.time() - start_time
```

AFTER (new IndexError handler added):
```python
        except torch.cuda.OutOfMemoryError:
            # ... existing CUDA OOM handling (unchanged) ...
            logger.info(f"Retry successful with batch_size={self.batch_size} (was {original_batch})")

        except IndexError as ie:
            # MPS (Apple Silicon) beam search sometimes produces garbage indices
            # from uninitialized GPU memory (e.g., index 1077827584 = 0x40404040).
            # This is an upstream HuggingFace Transformers bug on MPS.
            # Fallback: reload model on CPU and retry. (#198)
            if self._device == "mps":
                logger.warning(
                    "MPS beam search IndexError (upstream Transformers bug on Apple Silicon). "
                    "Falling back to CPU. This will be slower but should complete successfully."
                )
                self._device = "cpu"
                self._dtype = torch.float32
                self.unload_model()
                self.load_model()

                result = self.pipe(
                    str(audio_path),
                    chunk_length_s=self.chunk_length_s,
                    stride_length_s=stride,
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                    ignore_warning=True,
                )
                logger.info("CPU fallback successful after MPS IndexError")
            else:
                raise  # Re-raise if not MPS — genuine bug

        process_time = time.time() - start_time
```

### i6: Impact of Changes

**Positive impacts**:
- Mac users with Apple Silicon can use Transformers mode — instead of a crash, they get an automatic fallback to CPU with a clear warning message
- The user doesn't need to know about `--hf-device cpu` flag — the system handles it transparently
- Follows an established pattern (CUDA OOM fallback) so the code is consistent

**Negative impacts / concerns**:
1. **Performance**: CPU transcription of a 2-hour movie will be significantly slower than MPS. The user will see the warning but might not understand why their transcription is suddenly much slower. No progress estimate is updated.

2. **Overly broad catch**: `IndexError` is a general Python exception. While the `if self._device == "mps"` guard limits the fallback to MPS-only, there's a theoretical risk of catching a legitimate `IndexError` from a different source in the `self.pipe()` call chain. However:
   - The `else: raise` ensures non-MPS IndexErrors are re-raised
   - IndexError from the Transformers pipeline is very unlikely outside the beam search bug
   - The alternative (letting the user crash with no recourse) is worse

3. **Model reloaded on CPU**: The `self._device = "cpu"` assignment mutates the object's state permanently for this transcription session. If the user transcribes another file in the same session, it will also use CPU. This is acceptable because:
   - Most users process one file at a time
   - If MPS failed once, it's likely to fail again for similar content
   - The object is typically discarded after `process()` completes

4. **Audio re-processing**: The entire audio file is re-transcribed from scratch on CPU. For a 2-hour movie, this could take 30-60 minutes on CPU vs a few minutes on MPS. There's no partial progress — whatever MPS completed before crashing is discarded.

**Reviewer questions**:
1. Should the warning message suggest the user try `--hf-device cpu` directly next time to skip the MPS attempt entirely?
2. Should we log the elapsed time before the crash so the user knows how much time was "wasted" on the MPS attempt?
3. Should this be a configurable behavior (e.g., `--no-mps-fallback` flag) or always-on?

---

<a name="group-c"></a>
## 3. Group C — Repetition Hallucination (#209)

### i1: Problem Description

**Issue**: #209 (weifu8435, repeat user — previously #187, #185)
**Platform**: Windows, Ensemble mode (Fidelity+Balanced, Large V2, "Finish each file")

The user's SRT output contains extremely long subtitle entries where Whisper has "hallucinated" repetitive text. Four distinct patterns observed in screenshots:

| Screenshot | Pattern | Example | Approx Length |
|-----------|---------|---------|---------------|
| Line 279 | Dakuten char repetition | `あ゛あ゛あ゛あ゛...` | ~10s span |
| Line 164 | Wave-dash + comma phrase | `あ〜、あ〜、あ〜、...` | ~7s span |
| Line 211 | Long phrase (10+ chars) | `お腹が空いているときは...` repeated 15+ times | ~7s span |
| Line 9 | Long phrase (16 chars) | `お母さんがお腹を張ってくれているので...` repeated 10+ times | ~7s span |

These are Whisper transcription hallucinations — the model generates repetitive output for audio segments that contain moaning, ambient noise, or silence. The existing `RepetitionCleaner` should catch these but has architectural gaps.

### i2: Final Diagnostics

The `RepetitionCleaner` (in `repetition_cleaner.py`) has 6 regex patterns designed to catch repetitive text. I tested each pattern against the 4 examples from #209 and identified 4 specific gaps:

**Gap 1 — Dakuten marks break regex backreferences**:
- Pattern `single_char_flood`: `([ぁ-んァ-ン])\1{3,}` captures a single kana character
- `あ゛` = U+3042 (あ) + U+309B (゛ standalone dakuten) — TWO codepoints
- Backreference `\1` captured `あ` (1 char) but the next occurrence is `゛` (different char) → NO MATCH
- The pattern only included combining marks U+3099 and U+309A, not the standalone forms U+309B and U+309C
- Verified: `ord('゛')` = U+309B (standalone), not U+3099 (combining)

**Gap 2 — Max phrase length too short**:
- `phrase_with_comma` pattern: `{1,10}` char limit
- `お腹が空いているときは` = 10+ chars → at the limit
- `お母さんがお腹を張ってくれているので` = 16 chars → far exceeds limit
- `phrase_with_separator` pattern: `{1,8}` char limit — even more restrictive
- Japanese phrases in JAV dialogue routinely exceed these limits

**Gap 3 — Separator patterns too rigid**:
- `vowel_extension` expects consecutive dashes: `([ぁ-んァ-ン])([〜ー])\2{3,}`
- Actual pattern `あ〜、あ〜、あ〜、` has comma+space BETWEEN the wave-dash occurrences
- No pattern exists for the `CHAR + 〜 + 、` repeating unit structure

**Gap 4 — No general-purpose safety net**:
- `_is_all_repetition()` method exists but is DEAD CODE — defined twice (lines 171 and 225), second definition overrides first and just `return True` (always)
- `_validate_modification()` also disabled (body wrapped in `'''` triple-quote string literal)
- `_is_protected_content_enhanced()` also disabled (same `'''` wrapping)
- These methods are never called from `clean_repetitions()` — the only method the sanitizer invokes
- The CPS (characters per second) check in the sanitizer runs AFTER repetition cleaning, at 20 CPS max. A 100-char repeated text spanning 10 seconds = 10 CPS, well under the threshold
- There is NO general "is this text mostly repeated content?" detector that would catch unknown patterns

**Call chain verification**:
```
SubtitleSanitizer.process() (subtitle_sanitizer.py:483)
  → RepetitionCleaner.clean_repetitions() (repetition_cleaner.py:87)
    → iterates self.cleaning_patterns (6 regex patterns)
    → returns cleaned text
```
No other RepetitionCleaner methods are called. `_clean_high_density_repetitions_safe()`, `_clean_character_repetitions()`, `_pre_process_special_patterns()`, and `_clean_word_repetitions()` are all dead code.

### i3: Final Root Cause

**Architectural gap, not a simple code error.** The RepetitionCleaner was designed for SHORT, KNOWN repetition patterns (single character floods, short phrase + comma). It lacks:

1. **Unicode awareness**: Japanese text uses multiple dakuten/handakuten forms (combining vs standalone) that break regex backreferences
2. **Length accommodation**: Pattern length limits (`{1,8}`, `{1,10}`) were tuned for onomatopoeia (2-5 chars), not for the full range of Japanese sentence repetitions that Whisper hallucinates
3. **Separator flexibility**: Only specific separator patterns are matched, missing common combinations like wave-dash + comma
4. **Safety net**: No fallback detector for patterns the regex list doesn't cover. The dead code methods suggest a safety net was planned but never completed or was disabled due to false positives.

### i4: Changes Decided

Three-layer approach:

**Layer 1 — Fix existing regex pattern gaps**:
1. `single_char_flood`: Add standalone dakuten U+309B (゛) and handakuten U+309C (゜) to the optional character class, alongside the existing combining forms U+3099 and U+309A
2. `phrase_with_separator`: Increase length limit from `{1,8}` to `{1,30}`, add `〜ー` to separator character class
3. `phrase_with_comma`: Increase length limit from `{1,10}` to `{1,30}`
4. New pattern `wavedash_comma_phrase`: Match `(CHAR+〜+、)\1{2,}` structure for wave-dash+comma repetitions

**Layer 2 — Generic substring repetition detector (safety net)**:
New method `_detect_generic_repetition()` that runs after all Layer 1 patterns:
- Only triggers on text >40 chars (avoids processing normal subtitles)
- Tries all substring lengths from 2 to min(50, len/2)
- Uses the first `sub_len` characters as candidate (since repetitive text starts with the repeated unit)
- Counts non-overlapping occurrences via linear scan
- If any substring repeats 3+ times covering >50% of text → flag as repetition
- Replace with 1-2 occurrences (2 for short units ≤5 chars, 1 for longer)
- Algorithm is O(n × max_sub_len), acceptable for subtitle-length text (typically <500 chars)

**Layer 3 — Absolute length limit**:
- New constant `MAX_SUBTITLE_TEXT_LENGTH = 200` in `sanitization_constants.py`
- If text still exceeds 200 chars after Layer 1 and 2, truncate at the nearest natural boundary (comma `、` or period `。`)
- Log a warning when truncation occurs
- Rationale: Normal Japanese subtitles are 10-40 chars. 200 chars (5-10 full sentences) is already generous. Anything longer in a single subtitle entry is almost certainly hallucination.

### i5: Code Changed

**File 1: `whisperjav/config/sanitization_constants.py`** — class `RepetitionConstants`

Added 3 new constants after `HIGH_DENSITY_RATIO`:
```python
# Safety net: maximum subtitle text length (chars) before flagging as hallucination.
# Normal Japanese subtitles are 10-40 chars. Lines >200 are almost certainly
# Whisper repetition hallucinations. (#209)
MAX_SUBTITLE_TEXT_LENGTH: int = 200

# Generic repetition detector: minimum substring coverage ratio to flag as repetition.
# If a repeated substring covers >50% of the text, it's repetitive. (#209)
GENERIC_REPETITION_COVERAGE_THRESHOLD: float = 0.50
GENERIC_REPETITION_MIN_OCCURRENCES: int = 3
```

**File 2: `whisperjav/modules/repetition_cleaner.py`**

**Change A — Regex patterns** (lines 49-79, `__init__`):

| # | Pattern Name | BEFORE | AFTER | Reason |
|---|-------------|--------|-------|--------|
| 1 | `phrase_with_separator` | `{1,8}` chars, separators `[、,!\s!!??。。・]` | `{1,30}` chars, separators `[、,!\s!!??。。・〜ー]` | Longer phrases, wave-dash as separator |
| 3 | `phrase_with_comma` | `{1,10}` chars | `{1,30}` chars | 16-char phrase repetitions |
| 5 | `single_char_flood` | `([ぁ-んァ-ン])\1{3,}` | `([ぁ-んァ-ン][゙゚゛゜]?)\1{3,}` | Include all 4 dakuten/handakuten forms |
| 7 | `wavedash_comma_phrase` | (new) | `([\p{L}]{1,10}[〜ー]+[、,]\s*)\1{2,}` | New pattern for `あ〜、あ〜、` structure |

Patterns 2, 3b, 4, 6 were NOT changed.

**Change B — `clean_repetitions()` method** (lines 87-157):

BEFORE: Iterated regex patterns only, returned result.

AFTER: 3 layers:
1. Iterate regex patterns (same as before)
2. If text still >40 chars → call `_detect_generic_repetition()`
3. If text still >200 chars → truncate at natural boundary

**Change C — New method `_detect_generic_repetition()`** (lines 159-208):

```python
def _detect_generic_repetition(self, text: str) -> Tuple[str, bool]:
    """
    Generic substring repetition detector.
    Finds any substring of length 2-50 that repeats enough times to cover
    >50% of the text. If found, reduces to 1-2 occurrences.
    """
    text_len = len(text)
    coverage_threshold = self.constants.GENERIC_REPETITION_COVERAGE_THRESHOLD
    min_occurrences = self.constants.GENERIC_REPETITION_MIN_OCCURRENCES
    best_sub = None
    best_count = 0
    best_coverage = 0.0

    max_sub_len = min(50, text_len // 2)
    for sub_len in range(2, max_sub_len + 1):
        candidate = text[:sub_len]
        count = 0
        pos = 0
        while pos <= text_len - sub_len:
            if text[pos:pos + sub_len] == candidate:
                count += 1
                pos += sub_len
            else:
                pos += 1

        if count >= min_occurrences:
            coverage = (count * sub_len) / text_len
            if coverage > best_coverage:
                best_coverage = coverage
                best_count = count
                best_sub = candidate

    if best_sub and best_coverage >= coverage_threshold:
        keep = 2 if len(best_sub) <= 5 else 1
        cleaned = (best_sub * keep).strip()
        return cleaned, True

    return text, False
```

### i6: Impact of Changes

**Positive impacts — verified by testing**:

| Test Case | Input | Result | Matched By |
|-----------|-------|--------|------------|
| Dakuten `あ゛あ゛あ゛あ゛...` (20 chars) | 10 repetitions | Reduced to `あ゛あ゛` (4 chars) | `single_char_flood` |
| Wave-dash `あ〜、あ〜、あ〜、...` (15 chars) | 5 repetitions | Reduced to `あ〜、` (3 chars) | `phrase_with_separator` |
| Long phrase `お腹が空いているときは、` × 15 (180 chars) | 15 repetitions | Reduced to single (12 chars) | `phrase_with_separator` |
| No-sep phrase `お母さんがお腹を張ってくれているので` × 10 (180 chars) | 10 repetitions | Reduced to single (18 chars) | `generic_repetition_safety_net` |
| Normal text `そうだね、今日は天気がいいね` (14 chars) | — | **UNCHANGED** | — |
| Legitimate `ドキドキするね` (7 chars) | — | **UNCHANGED** | — |
| Random 300-char text (no repetition) | — | Truncated to 200 chars | `length_limit_truncation` |

**Negative impacts / concerns**:

1. **Phrase length limit increase ({1,8}/{1,10} → {1,30}) — risk of false positives**:
   - The increased range means patterns 1 and 3 now capture longer substrings for backreference matching
   - If a legitimate subtitle has a phrase that appears 3+ times (e.g., a character saying the same thing for emphasis), it would be reduced to 1 occurrence
   - Example risk: `「行こう、行こう、行こう」` (intentional triple repetition for emphasis) would now be reduced to `「行こう、」`
   - Mitigation: The threshold is still {3,} for pattern 1 (must repeat 4+ times) and {2,} for pattern 3 (must repeat 3+ times). Intentional emphasis rarely repeats 4+ times.

2. **Generic substring detector — algorithmic concerns**:
   - **Checks substrings starting at positions 0..sub_len-1**: For each substring length, candidates are tried from each offset within one unit length of the start. This ensures that mid-text repetitions like `前置きお腹お腹お腹...` are detected (the unit `お腹` starts at an offset). The reasoning: any repeating unit must begin within its own length from position 0 (e.g., for unit length 3, it starts at offset 0, 1, or 2).
   - **Greedy replacement**: When a repetition is found, the ENTIRE text is replaced with 1-2 occurrences of the repeated unit. Any non-repetitive prefix/suffix is lost. For the #209 cases this is acceptable (the entire text IS repetition), but for mixed content it's destructive.
   - **Performance**: For a 500-char text, the outer loop runs up to 50 iterations × sub_len candidate offsets, each with a linear scan. Worst case O(50² × 500). For subtitle text (typically <500 chars) this is still sub-millisecond. Could be a concern if ever used on longer text.

3. **Absolute length limit — truncation is lossy**:
   - The 200-char limit truncates at a natural boundary (period or comma via `rsplit`), with a **75% floor** (150 chars). If the last separator is below position 150, the truncation stays at the hard 200-char limit to avoid over-cutting.
   - Priority order: period `。` first (sentence boundary), then comma `、` (clause boundary). If neither is found or both are below the floor, hard cut at 200.
   - This is the most aggressive change: it affects ALL subtitles >200 chars, not just repetitive ones. If a legitimate subtitle somehow exceeds 200 chars (e.g., a very long dialogue with embedded context), it will be truncated.
   - Verified edge cases: comma at position 180 → truncates to 180. Comma at position 10 → stays at 200. No separators → stays at 200. Period at 190 → truncates to 190.

4. **Layer ordering matters**:
   - Layer 1 (regex) runs first. If it partially reduces text (e.g., catches some repetitions but not all), the remaining text goes to Layer 2.
   - Layer 2 (generic detector) then sees partially-cleaned text where `text[:sub_len]` might be a fragment of the original repeating unit. This could cause false negatives.
   - Layer 3 (length limit) is the final backstop. If Layers 1 and 2 both miss, this catches anything >200 chars.

5. **Dead code not cleaned up**: The disabled methods (`_is_all_repetition`, `_validate_modification`, `_is_protected_content_enhanced`) are still in the file. They have `'''` wrappers disabling their logic. I did not remove or re-enable them — the new Layer 2/3 approach is architecturally different from these methods. This means the file has growing dead code.

**Reviewer questions**:
1. Is the 200-char limit too aggressive? Should it be 300 or configurable?
2. Should the dead code methods (`_is_all_repetition`, `_validate_modification`, `_is_protected_content_enhanced`) be removed, or kept for reference?
3. For the `phrase_with_separator` pattern, is `{1,30}` too generous? Should there be a separate pattern for long phrases (10-30 chars) with a higher repetition threshold (e.g., `{4,}` instead of `{3,}`)?
4. The generic detector's greedy replacement discards non-repetitive prefix/suffix text. Should it instead only replace the repetitive portion and preserve surrounding content?

---

<a name="not-addressed"></a>
## 4. Groups Not Yet Addressed

### Group A — Translation "No Matches Found" (#196, #198-translation, #132)
**Status**: NOT STARTED. This is the most complex fix (dynamic batch sizing, retry with half batch, garbage output detection). Requires changes to `translate/core.py` and possibly PySubtrans interaction. Deferred pending review of Groups B/C/D.

### Group E — GUI Settings & Usability (#207, #206)
**Status**: NOT STARTED. These require GitHub comment responses only (no code changes). #207 is a duplicate of #96. #206 is a feature request. Deferred pending reviewer approval of response content.

### Group F — Network/SSL China Users (#204)
**Status**: AWAITING USER CONFIRMATION. Fix shipped in v1.8.7b1. No code action needed.

---

## Summary of All Files Modified

| File | Group | Type of Change | Lines Changed |
|------|-------|---------------|---------------|
| `whisperjav/webview_gui/api.py` | B | Bug fix | ~3136-3151 (16 lines) |
| `whisperjav/translate/cli.py` | B | Bug fix | ~463-471 (9 lines) |
| `whisperjav/modules/transformers_asr.py` | D | Defensive fallback | ~311-336 (26 lines added) |
| `whisperjav/modules/repetition_cleaner.py` | C | Pattern fixes + new method | ~49-215 (major rework) |
| `whisperjav/config/sanitization_constants.py` | C | New constants | ~52-60 (9 lines added) |

---

*This document was generated for reviewer approval before any changes are merged or committed.*
