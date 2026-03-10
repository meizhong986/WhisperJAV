# Draft Issue for machinewrapped/llm-subtrans

> For review before posting. NOT to be posted without owner approval.

---

**Title:** Local LLM degeneration causes "No matches found" — batch splitting on token limit

**Labels:** bug, enhancement

---

Hi! I'm Claude, sending this on behalf of [WhisperJAV](https://github.com/meizhong986/WhisperJAV), which uses PySubtrans (v1.5.7) for subtitle translation. First — thank you for this library, it's excellent.

I wanted to share some findings that may relate to your #393 ("Error when prompt is too long"). We've been investigating recurring "No matches found" failures when using local LLMs (llama-cpp-python with 8B/9B GGUF models), and have detailed debug logs showing what happens.

### What we're seeing

With `CustomClient` + local llama-cpp server (8K context), smaller quantized models sometimes degenerate — they stop following the `#N / Translation>` format and fill their output budget with repetitive garbage. We've captured a full debug log showing this in action:

**Batch 1, first attempt** — works fine:
```
prompt_tokens: 574, completion_tokens: 321, total: 895
finish_reason: stop
→ 9/10 lines matched correctly
```

**Batch 1, retranslation** (triggered because 1 line had a minor formatting mismatch):
```
prompt_tokens: 914, completion_tokens: 7278, total: 8192
finish_reason: length
→ Output: "<##You############<####<#<##Here######..." (thousands of garbage characters)
→ "No matches found in translation text using patterns: [...]"
```

The model filled the entire remaining context (7278 tokens) with degenerate output. The `TranslationParser` then can't find any `#N` patterns in the garbage, and the batch is lost.

This happens across platforms (Ubuntu/macOS/Kaggle) with different GGUF models (Llama-3.1-8B, Gemma-9b). It appears to be a general property of smaller quantized models under certain prompt conditions.

### The specific flow

In `SubtitleTranslator.TranslateBatch()`:

1. `reached_token_limit` is True → the retry at line 243-249 strips context but keeps the same batch size
2. The retry often also degenerates → goes to `ProcessBatchTranslation` → `TranslationParser` → "No matches found"
3. The batch is lost with no further recovery attempt

I noticed the TODO at line 245: `# TODO: better to split the batch into smaller chunks` — which is exactly the scenario we're hitting.

### Suggestions (just ideas, happy to contribute a PR if any are welcome)

1. **Split batch on token limit**: When `reached_token_limit` is True, halve the batch and retry both halves separately, rather than retrying the same size without context. This addresses the TODO at line 245.

2. **Garbage detection before regex**: A quick check in `TranslationParser.ProcessTranslation()` — if the response text doesn't contain a single `#` followed by a digit, skip the regex and return a more specific error (e.g., "LLM response does not appear to contain structured translations"). This would help users diagnose the issue faster.

3. **Pass-through of `repetition_penalty`**: `CustomClient._generate_request_body()` currently passes `temperature` and `max_tokens`. Local LLM servers (llama-cpp, vllm, ollama) also support `repetition_penalty` and `min_p`, which can prevent the degenerate cycling we're seeing. Exposing these as optional settings would help.

4. **Partial match acceptance**: In the retranslation path, if the initial translation matched 9/10 lines but failed validation for 1 missing line, accepting the 9 successful translations rather than discarding them all and retranslating could improve resilience.

### Debug log

I can share the full 945-line debug log if useful. It shows the complete request/response bodies, token counts, and the exact garbage output for each batch.

Happy to discuss further or put together a PR for any of these. Thanks again for the great work on this project.
