# Group A Deep Dive — "No Matches Found" Translation Failure

> Analysis date: 2026-03-09 | Issues: #196, #198 (LLM), #132
> Status: Root cause identified, fix plan drafted, pending implementation

---

## 1. Evidence Inventory

| Source | Type | Key Finding |
|--------|------|------------|
| **#196** zhstark comment + 3 screenshots | Terminal output, v1.8.7b1 | Batch 30→11 (auto-cap works), max_tokens=2692 (applied), 221.7 tps GPU 5090. Still `finish_reason='length'` → retry → "No matches". Even `--max-batch-size 10` fails. **Chinese + pornify tone.** |
| **#198** francetoastVN LLM comment | Terminal paste, v1.8.7b1 | 637 lines/58 scenes, batch 10, 27.4 tps CPU. `max_tokens=2692`, `streaming=True`. Fails **immediately on scene 1 batch 1**. Same cascade. **English + standard tone.** |
| **#132** TinyRick1489 debug_log.txt (945 lines) | Full debug log, pre-v1.8.7b1 | 23 lines/1 scene. Batch 1 (10 lines): first attempt succeeds (574 prompt, 321 completion, 895 total). 9/10 matched — #9 missed due to minor formatting glitch (`Original>それで` vs `Original> それで`). Validation fails → retranslation → **914 prompt tokens + 7278 completion = 8192 (full context)** → output = wall of `#<You###Here###` garbage. Batch 2 (13 lines): `Hit API token limit` → same garbage → "No matches". **9/23 translated, 14 lost.** |
| **PySubtrans source** | Library code | Full retry logic, prompt accumulation, garbage-to-parser flow traced (SubtitleTranslator.py, CustomClient.py, TranslationParser.py, TranslationPrompt.py) |
| **WhisperJAV source** | Integration code | `cap_batch_size_for_context`, `compute_max_output_tokens`, provider config wiring (core.py, cli.py, local_backend.py) |
| Instruction files | standard.txt=3054B/444w, pornify.txt=1058B/152w | Pornify is actually **smaller** than standard — NOT the cause |

---

## 2. Verified Failure Chains (Three Distinct Modes)

### Failure Mode A — Retranslation Context Explosion (pre-v1.8.7b1)

Observed in: #132 batch 1 retry

```
1. Initial translation succeeds (9/10 lines)
2. One line mismatched (minor format glitch: "Original>それで" vs "Original> それで")
3. PySubtrans validation: 1 line missing → RequestRetranslation()
4. GenerateRetryPrompt() APPENDS to conversation (5 messages, 914 prompt tokens)
5. No max_tokens cap → model fills remaining 7278 tokens with degenerate output
6. Output: wall of "#<You###Here###..." (classic LLM degeneration)
7. TranslationParser.FindMatches() → zero regex matches
8. "No matches found" error
```

**Status**: Partially addressed by v1.8.7b1 (max_tokens cap). But...

### Failure Mode B — finish_reason='length' on First Attempt (v1.8.7b1)

Observed in: #196 (zhstark), #198 LLM (francetoastVN)

```
1. Batch properly sized (10-11 lines)
2. max_tokens=2692 properly set
3. Model starts generating → enters degenerate state → fills ALL 2692 tokens
4. Output is garbage (no #N\nTranslation> format)
5. finish_reason='length' (hit max_tokens, not natural stop)
6. PySubtrans: reached_token_limit → retry WITHOUT context
7. Retry: fresh prompt, same batch size, same max_tokens
8. Model AGAIN degenerates → 2692 tokens of garbage
9. No retry after retry → goes to parser → "No matches found"
```

This is the **critical failure** — the v1.8.7b1 fix doesn't help because the model degenerates on the FIRST attempt.

### Failure Mode C — Prompt Exceeds Context Budget (v1.8.7b1)

Observed in: #132 batch 2 (13 lines)

```
1. Batch has 13 lines (exceeds the 10 that fit in 8K)
2. prompt_tokens + max_tokens > n_ctx
3. llama-cpp truncates or mishandles → garbage output
4. "Hit API token limit" → retry → still fails
```

**Status**: Addressed by batch cap (11 for 8K). But this only works for the default `--max-batch-size`.

---

## 3. Root Cause — Architectural

The surface error is "No matches found in translation text." The tracker's preliminary diagnosis was "cascading token overflow." After examining all evidence, the actual root cause is deeper:

**The fundamental problem is: WhisperJAV depends on PySubtrans, which depends on the LLM producing structured output in a specific format. Small quantized local LLMs (8B-9B GGUF) are inherently unreliable at following structured output formats, and the system has no defense against model degeneration.**

### Architectural Gap Inventory

#### Gap 1: max_tokens is too generous (allows degenerate output)
- `compute_max_output_tokens(batch=10, n_ctx=8192)` returns 2692
- Expected output for 10 lines is ~800 tokens
- That's 3.4x the expected output — far too much headroom
- A degenerate model fills all 2692 tokens with garbage

#### Gap 2: No garbage detection before parsing
- Output goes directly from `CustomClient` → `Translation` → `TranslationParser`
- No intermediate check for "is this output remotely valid?"
- By the time PySubtrans tries regex matching, it's too late — the batch is "failed"
- WhisperJAV can't intervene because **PySubtrans owns the translation loop**

#### Gap 3: Retry doesn't reduce batch size
- `SubtitleTranslator` line 243-249: retry on `reached_token_limit` strips context but keeps same batch size
- If the model degenerates with 10 lines, it will also degenerate with 10 lines minus context
- The PySubtrans code has a `TODO` at line 245: `# TODO: better to split the batch into smaller chunks`
- WhisperJAV can't override this behavior without modifying PySubtrans

#### Gap 4: No repetition_penalty or min_p in LLM requests
- The request body only sets `temperature` and `max_tokens`
- No `repetition_penalty` (prevents repetitive degeneration)
- No `min_p` (prunes low-probability tokens)
- These are standard parameters for preventing LLM degeneration

#### Gap 5: WhisperJAV has no control over PySubtrans's internal retry loop
- `project.TranslateSubtitles(translator)` is a single call
- PySubtrans iterates batches internally
- WhisperJAV can't intercept individual batch failures
- The only events WhisperJAV hooks are logging-related

#### Gap 6: Streaming doesn't help with degenerate output
- v1.8.7b1 forces streaming for local LLM (prevents timeout)
- But streaming doesn't prevent degeneration — it just delivers garbage incrementally

---

## 4. Why v1.8.7b1 Fix Is Incomplete

The v1.8.7b1 fix addressed three things:
1. **max_tokens cap** — prevents filling the entire context with output ✓
2. **Auto-batch sizing** — caps batch to fit 8K context ✓
3. **Streaming** — prevents HTTP timeouts ✓

But it missed the core issue: **the model itself degenerates**. The fix assumed that if you limit the output tokens and right-size the batch, the model will produce valid output. This is true for commercial APIs (GPT/Claude/Gemini) but NOT for quantized 8B GGUF models.

---

## 5. Cross-Platform Analysis

| Reporter | Platform | Model | GPU | Speed | Issue |
|----------|----------|-------|-----|-------|-------|
| zhstark | Ubuntu CLI | unknown GGUF | 5090 32GB | 221.7 tps | Degenerate first attempt |
| francetoastVN | macOS M1 | llama-8b Q4 | CPU only | 27.4 tps | Degenerate first attempt |
| TinyRick1489 | Kaggle Linux | gemma-9b Q4 | T4 GPU | 29.5 tps | Degenerate on retry |

**All reporters are on Unix-like systems** (Ubuntu, macOS, Kaggle Linux). No Windows reporter has filed this issue. This may be coincidence (local LLM users skew Unix) or may indicate a platform-specific factor in llama-cpp-python behavior.

The degeneration is **model-specific** — quantized 8B/9B models are the common factor across all platforms.

---

## 6. #132 Additional Considerations

Issue #132 (TinyRick1489, Kaggle) has a complex history. Many earlier comments are about notebook/installation issues (numpy/numba, soundfile, pip, tag mismatch) — all resolved in previous versions. The translation-related issue is the same Group A root cause but with additional Kaggle-specific factors:

1. **Pre-v1.8.7b1 code** — no max_tokens, no streaming, no auto-batch
2. **Kaggle T4 GPU** — works well for inference (29.5 tps) but shared environment
3. **The actual translation failure** is identical: model degeneration on retry → garbage output
4. Owner previously closed as "known Colab limitation" — but user says "it used to work previously"

---

## 7. Fix Plan — Architectural

Given that WhisperJAV wraps PySubtrans and can't easily modify its internal retry loop, the fix must work at two levels:

### Level 1: Prevent degeneration (WhisperJAV side)

| Fix | Where | Impact |
|-----|-------|--------|
| **Tighter max_tokens**: 1.3x expected instead of 2x | `core.py:compute_max_output_tokens()` | Reduces garbage headroom from ~1900 extra tokens to ~240 |
| **Add `repetition_penalty=1.1`** to request body | provider_options in cli.py or CustomClient | Discourages repetitive degeneration at the model level |
| **Reduce batch size**: 7 for 8K (not 11) | `core.py:cap_batch_size_for_context()` | Smaller batches = less chance of degeneration |

### Level 2: Detect and recover from degeneration (WhisperJAV side)

| Fix | Where | Impact |
|-----|-------|--------|
| **Garbage detection**: Check if LLM output is >50% non-text characters | New helper in `core.py` | Meaningful error instead of cryptic regex failure |
| **Progressive batch reduction**: If translation fails, retry entire file with half batch | `cli.py` retry loop | Recovers from batch-size-related failures |

### Level 3: PySubtrans collaboration (deeper)

| Fix | Where | Impact |
|-----|-------|--------|
| **Batch splitting on token limit** | PySubtrans `SubtitleTranslator` line 245 (the existing TODO) | Halves batch on `reached_token_limit` instead of just stripping context |
| **Garbage detection in TranslationParser** | `TranslationParser.ProcessTranslation()` | Detect garbage before regex, return actionable error |
| **Pass-through of `repetition_penalty`, `min_p`** | `CustomClient._generate_request_body()` | Allow LLM control parameters for local models |
| **Partial match acceptance** | `SubtitleTranslator.ProcessBatchTranslation()` | If 9/10 lines match, accept 9 instead of retranslating |

---

## 8. Unknowns (Need Verification)

1. **What model is zhstark using?** — Screenshots show GPU info but not model name
2. **Does `repetition_penalty` in llama-cpp actually prevent this degeneration?** — Hypothesis, not verified
3. **What does llama-cpp-python do when prompt + max_tokens > n_ctx?** — Truncation behavior unknown
4. **Can WhisperJAV hook into PySubtrans events to intercept batch failures?** — `batch_translated` exists but fires AFTER
5. **Would grammar-constrained output in llama-cpp help?** — Forces valid format, not tested

## 9. Recommended Investigation Order

Before writing code:
1. Verify degeneration hypothesis with local `--debug` test
2. Test `repetition_penalty=1.1` manually
3. Test tighter max_tokens (~1200 for batch=10)
4. Check llama-cpp context handling edge cases
5. Evaluate PySubtrans collaboration opportunity (see section below)

---

## 10. PySubtrans Collaboration Opportunity

PySubtrans is maintained by machinewrapped on GitHub. The library has a `TODO` at SubtitleTranslator.py:245 acknowledging the need to split batches on token limit. Several fixes would benefit both projects:

- Batch splitting on `reached_token_limit` (the existing TODO)
- Garbage output detection before regex parsing
- Pass-through of model control parameters (`repetition_penalty`, `min_p`)
- Partial match acceptance (9/10 matched = accept 9)

These are generic improvements that help ANY local LLM user of PySubtrans, not just WhisperJAV.

---

*This analysis is a living document. Update as investigation progresses.*
