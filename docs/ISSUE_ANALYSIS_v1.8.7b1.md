# Cross-Cutting Issue Analysis — v1.8.7b1

> Date: 2026-03-09 | Scope: All open issues + new bug reports on closed issues
> Method: Full evidence review (screenshots, logs, attachments, comment history)

---

## Group A: Translation "No Matches Found" — THE CRITICAL BUG

**Issues**: #196 (zhstark, Ubuntu 5090), #198 (francetoastVN, Mac M1), #132 (TinyRick1489, Kaggle)
**Symptom**: `Error translating scene X batch Y: No matches found in translation text using patterns: [...]`
**Severity**: HIGH — affects all platforms, all local LLM users
**Status**: v1.8.7b1 fix was PARTIAL — auto-batch works but the root cause persists

### Evidence Examined

**#196 — zhstark (Ubuntu, 5090 32GB, v1.8.7b1):**
- 3 screenshots showing full terminal output
- Version confirmed: WhisperJAV 1.8.7b1
- GPU: 33 layers CUDA, 221.7 tps (extremely fast)
- Batch auto-reduced 30→11 (**auto-batch fix IS working**)
- Output token limit: 2692 (JAV/CJK-tuned cap **IS applied**)
- Still hits: `WARNING: Hit API token limit, retrying batch without context...`
- Then: `Error translating scene X batch Y: No matches found`
- User tried `--max-batch-size 10` manually — same error
- Target language: Chinese, tone: pornify

**#198 — francetoastVN (macOS M1, v1.8.7b1):**
- Full terminal log pasted in comment
- Server starts fine, 27.4 tps CPU only
- 637 lines in 58 scenes, batch size 10
- Immediately fails scene 1 batch 1: same "No matches found" error
- Target language: English, tone: standard

**#132 — TinyRick1489 (Kaggle, pre-v1.8.7b1):**
- Full 945-line debug log examined
- Kaggle environment, gemma-9b, 29.5 tps, 43 layers CUDA
- `'stream': False` ← **streaming NOT enabled** (pre-v1.8.7b1)
- Batch 1: First request **times out** (ReadTimeout after 300s), retries, then succeeds
- Batch 1: 9/10 lines matched (1 missing — format issue: no space after `Original>`)
- Batch 1: **Failed validation → retranslation requested** (1 unmatched line triggers full retry)
- Batch 2: "Hit API token limit" → LLM response is **a wall of `###########`** (thousands of `#` chars)
- Result: 9/23 lines translated successfully, 14 untranslated
- Total time: 1163 seconds (~19 minutes) for 23 lines

### Root Cause Analysis

The translation pipeline fails through a **cascading failure chain**:

```
1. Prompt too large for context window
   ↓
2. LLM hits token limit mid-generation
   ↓
3. Retry-without-context: removes context but keeps same batch size
   ↓
4. LLM generates malformed output (garbled format, wall of ###, truncated)
   ↓
5. PySubtrans regex patterns don't match the garbage output
   ↓
6. "No matches found" error → batch fails → lines untranslated
```

**Why v1.8.7b1 fix is incomplete:**

The v1.8.7b0/b1 fix addressed 3 things:
1. **max_tokens cap** (2692 for JAV/CJK) — ✅ Applied, visible in logs
2. **Forced streaming** — ✅ Applied in v1.8.7b1 code, but...
3. **Auto-batch reduction** — ✅ Working (30→11), but...

The problem: **Even batch size 11 can exceed context** when:
- Target language is Chinese (longer system prompt with CJK-tuned instructions)
- Tone is "pornify" (additional instructions inflate prompt)
- Japanese source lines are long (JAV dialogue can be verbose)
- Previous batch's summary/scene context is included (conversation history grows)

The auto-batch calculation uses a **static formula** that doesn't account for:
- Variable instruction length per tone
- Conversation history growth (accumulated context from prior batches)
- Target language output expansion (Japanese→Chinese is ~1.5x token expansion)

**Additionally, the retry-without-context path is fundamentally broken:**
When the retry fires, it removes conversation history but keeps the same batch size. If the batch was already too large for the context, removing history alone may not be enough. And when the LLM IS forced to generate within a too-tight context, it produces garbage (`###...`) that PySubtrans can't parse.

### What's Different About #132

#132 (Kaggle) has two ADDITIONAL problems:
1. **Streaming is False** — user is on pre-v1.8.7b1 version. Non-streaming means the full response must fit in memory before being returned, and timeouts are more likely.
2. **ReadTimeout** — the first request times out after 300s. At 29.5 tps with batch size 17, generating a full translation of 10 lines could take 30-60s. But the first request times out, suggesting the model is loaded but slow on first inference (cold start on Kaggle).
3. **Validation stringency** — even when 9/10 lines match, the 1 unmatched line causes a "failed validation" → full retranslation. This wastes resources and triggers the cascade.

### Recommendations

**Immediate fix (v1.8.7b2):**
1. **Dynamic batch sizing**: Calculate max batch based on ACTUAL prompt token count, not a static formula. Use tiktoken or llama_cpp tokenizer to count tokens before sending.
2. **Graceful degradation on retry**: If retry-without-context still fails, try with HALF the batch size instead of the same size.
3. **Garbage output detection**: Before passing to PySubtrans regex, check if the response is mostly non-alphanumeric chars (wall of `#`). If so, skip the regex and report the actual failure.
4. **Partial success acceptance**: If 9/10 lines match, accept the 9 and move on. Don't force a full retranslation over 1 missing line.

**Medium-term (v1.8.8):**
5. **Per-tone context budgeting**: Different tones (standard, pornify) need different batch sizes because instructions vary in length.
6. **Adaptive batch reduction**: If a batch fails, automatically retry with batch_size//2 before giving up.

---

## Group B: Custom Provider API Key Handling

**Issues**: #143 (OdinShiva)
**Symptom**: `"Connection to custom failed: str expected, not NoneType"`
**Severity**: MEDIUM — blocks Ollama/LM Studio users (growing user segment)
**Status**: Not fixed

### Evidence Examined

- Screenshot 1: GUI showing Custom (OpenAI-compatible) provider, API Key empty, endpoint `http://localhost:11434/v1`, model `qwen3.5:4b`, "Failed" badge
- Screenshot 2: Ollama interface running with `qwen3.5:4b` loaded

### Root Cause

The OpenAI Python SDK requires `api_key` to be a string. When the GUI field is empty, `None` is passed to `openai.OpenAI(api_key=None)`. Ollama, LM Studio, and other local servers don't require API keys, but the SDK constructor validates the type.

### Recommendation

**Tiny fix**: When provider is "custom" and API key is empty/None, default to `"not-needed"` or `"ollama"`. This is a 1-line fix in the GUI API or the translation backend.

**Files likely involved**: `whisperjav/translate/core.py` or `whisperjav/webview_gui/api.py` — wherever the custom provider's OpenAI client is constructed.

---

## Group C: Whisper Repetition Hallucination

**Issues**: #209 (weifu8435)
**Symptom**: SRT output contains extremely long subtitle entries with repeated phrases
**Severity**: HIGH — visible output quality degradation
**Status**: Root cause identified, fix planned

### Evidence Examined

- 5 SRT screenshots showing 4 distinct repetition patterns
- 1 GUI settings screenshot (Ensemble mode, Fidelity+Balanced, Large V2)
- 6 user comments

### Root Cause

Four architectural gaps in `whisperjav/modules/repetition_cleaner.py`:
1. **Dakuten marks** (U+309B combining diacritics) break regex backreferences
2. **Max phrase length 10 chars** — Japanese phrases routinely exceed this
3. **Rigid separator patterns** — miss comma+wave-dash combinations
4. **No safety net** — no general-purpose "is this text mostly repetitions?" detector

Full analysis in tracker file, Cluster 9.

### Recommendation

3-layer fix:
1. **Generic substring repetition detector** (safety net for any repeated pattern)
2. **Fix specific regex gaps** (dakuten, length limit 10→30, separator flexibility)
3. **Absolute length limit** (~200 chars for Japanese subtitle = almost certainly hallucination)

---

## Group D: MPS/Apple Silicon Runtime Crash

**Issues**: #198 (francetoastVN — Transformers mode)
**Symptom**: `IndexError: index 1077827584 is out of bounds for dimension 0 with size 40`
**Severity**: MEDIUM — blocks Transformers mode on Apple Silicon
**Status**: Upstream library bug, not WhisperJAV code

### Evidence Examined

- Full 166-line traceback from `TransformerTest.txt`
- Error occurs in `transformers/models/whisper/generation_whisper.py:1172` → `split_by_batch_index`
- Using MPS device, kotoba-whisper-bilingual-v1.0, float16, batch 8
- The index value `1077827584` (= 0x40404040) looks like uninitialized memory

### Root Cause

This is a **HuggingFace Transformers library bug on MPS**. The beam search postprocessing step reads `beam_indices` from MPS tensor output, and on Apple Silicon, the indices sometimes contain garbage values (uninitialized GPU memory). This is a known class of MPS backend issues in PyTorch.

Key observations:
- Audio extracts fine (7348.9s = ~2 hour movie)
- Model loads on MPS successfully
- Transcription begins but crashes on first chunk's beam search postprocessing
- The garbage index (0x40404040) is a classic "fill pattern" indicating uninitialized memory

### Recommendation

**Workaround**: Tell user to try `--hf-device cpu` to bypass MPS. Slower but will work.

**Defensive fix**: In `transformers_asr.py`, catch `IndexError` during transcription and retry with `device="cpu"` + log a warning. This follows the existing pattern (MPS→CPU fallback) already used in speech enhancement backends.

**Upstream**: File issue with HuggingFace Transformers for beam search on MPS with long audio files.

---

## Group E: GUI Settings & Usability

**Issues**: #207 (q864310563), #96 (sky9639), #206 (techguru0)
**Symptom**: Settings lost on restart; incompatible options not blocked
**Severity**: MEDIUM — most frequently reported UX frustration (4 duplicates of #96 now)
**Status**: Partial implementation (translation+ensemble saved, pipeline tab not)

### Evidence

- #207 is the 4th duplicate of #96 — Chinese user on .exe installer
- #206 is a separate feature request for option compatibility checking
- #96 has been open since early v1.8.x cycle

### Root Cause

Pipeline tab settings (model, sensitivity, scene detector, speech enhancer, speech segmenter) are not persisted to disk. Only translation tab and ensemble tab settings are saved (implemented in hotfix2). This is by design (v1.9 scope) but the frequency of reports suggests it should be prioritized.

### Recommendation

**#207**: Respond explaining what IS saved vs what isn't. Reference #96.
**#206**: Acknowledge as valid v1.9+ feature request.
**#96**: Consider partial implementation in v1.8.8 — save pipeline tab's dropdown values to the existing settings file. This would address the most common complaint with modest effort.

---

## Group F: Network/SSL (China Users)

**Issues**: #204 (yangming2027)
**Symptom**: SSL/certificate errors when using VPN proxies
**Severity**: HIGH for Chinese user base
**Status**: Fix shipped in v1.8.7b1, awaiting user confirmation

### No New Evidence

Last comment was meizhong986 pointing user to v1.8.7b1 release. No response yet from yangming2027 or weifu8435 confirming the fix works.

### Recommendation

Wait for user confirmation. No code action needed.

---

## Cross-Group Patterns

### Pattern 1: Translation Pipeline is the Weakest Link

Groups A, B, and part of D all involve the translation subsystem. Three different failure modes:
- **Token limit cascade** (A) — the most impactful
- **API key validation** (B) — simple fix
- **Output format parsing** (A) — PySubtrans regex is brittle

The translation pipeline is architecturally fragile because it depends on:
1. LLM producing output in an exact format (PySubtrans regex patterns)
2. Token counts staying within context windows (hard to predict)
3. API clients being configured correctly (no validation on inputs)

**Recommendation**: The translation pipeline needs a **resilience layer** similar to what we built for model downloads (3-step fallback). Specifically: graceful degradation when LLM output is malformed, adaptive batch sizing, and input validation.

### Pattern 2: Upstream Library Bugs Manifest on Non-Primary Platforms

- MPS beam search crash (Group D) — HuggingFace Transformers bug on Apple Silicon
- MPS was only recently added (v1.8.7b0) — early adoption issues expected

**Recommendation**: For all MPS code paths, add try/except with CPU fallback. MPS is still maturing.

### Pattern 3: Output Quality vs. Output Quantity

Groups A and C are inverses:
- **Group A**: Translation produces NO output (parsing failure)
- **Group C**: Transcription produces TOO MUCH output (hallucination/repetition)

Both are output quality issues that need safety nets rather than specific pattern fixes.

### Pattern 4: Settings/UX Issues Are Persistent and Growing

4 duplicates of #96 signals this is the #1 UX irritant. Users discover it independently, report it, and are told "v1.9." The frequency suggests accelerating at least the pipeline tab persistence.

---

## Priority Ranking (All Groups)

| Rank | Group | Issue(s) | Impact | Effort | Recommendation |
|------|-------|----------|--------|--------|----------------|
| 1 | A | #196, #198 | All LLM translation users | Medium | Dynamic batch sizing + graceful retry |
| 2 | C | #209 | All transcription users | Medium | Repetition safety net |
| 3 | B | #143 | Ollama/LM Studio users | Tiny | Default dummy API key |
| 4 | D | #198 | Mac Transformers users | Small | MPS→CPU fallback on IndexError |
| 5 | E | #207 | All GUI users | Small | Pipeline tab persistence |
| 6 | F | #204 | China users | None | Await confirmation |

---

*Analysis based on complete evidence review: 14 screenshots, 3 log files (1111 total lines), 50+ comments across 10 issues.*
