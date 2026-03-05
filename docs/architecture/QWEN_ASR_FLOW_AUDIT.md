# Qwen-ASR Flow Audit: Finding Gaps

**Created**: 2026-02-01
**Issue**: User reports identical output before and after parameter changes
**Status**: Under Investigation

---

## 1. Complete Flow Trace

### CLI → Pipeline → QwenASR → qwen-asr

```
User runs: whisperjav video.mp4 --mode qwen --qwen-max-tokens 4096

1. CLI PARSING (main.py:419-448)
   ├── argparse parses --qwen-* arguments
   ├── args.qwen_max_tokens = 4096 ✓ VERIFIED
   └── args.qwen_batch_size = 1 ✓ VERIFIED

2. PIPELINE CREATION (main.py:751-774)
   ├── TransformersPipeline(
   │     asr_backend="qwen",
   │     qwen_max_tokens=getattr(args, 'qwen_max_tokens', 4096),
   │     ...
   │   )
   └── Parameters passed correctly ✓ VERIFIED

3. PIPELINE INIT (transformers_pipeline.py:55-250)
   ├── self.qwen_config stores parameters
   ├── self._asr_config = {
   │     'max_new_tokens': qwen_max_tokens,  # 4096
   │     'batch_size': qwen_batch_size,      # 1
   │     ...
   │   }
   └── Config stored correctly ✓ VERIFIED

4. ASR CREATION (transformers_pipeline.py:553-556)
   ├── from whisperjav.modules.qwen_asr import QwenASR
   ├── asr = QwenASR(**self._asr_config)
   └── QwenASR instantiated with config ✓ VERIFIED

5. QWEN ASR INIT (qwen_asr.py:97-140)
   ├── self.max_new_tokens = max_new_tokens  # 4096
   ├── self.batch_size = batch_size          # 1
   └── Parameters stored ✓ VERIFIED

6. MODEL LOADING (qwen_asr.py:333-395)
   ├── model_kwargs = {
   │     "max_inference_batch_size": self.batch_size,  # 1
   │     "max_new_tokens": self.max_new_tokens,        # 4096
   │     ...
   │   }
   ├── Qwen3ASRModel.from_pretrained(self.model_id, **model_kwargs)
   └── Parameters passed to qwen-asr ✓ VERIFIED (diagnostic showed 4096/1)

7. QWEN-ASR INTERNAL (qwen_asr library)
   ├── __init__ stores: self.max_new_tokens = max_new_tokens
   ├── _infer_asr_transformers uses: model.generate(..., max_new_tokens=self.max_new_tokens)
   └── Parameters SHOULD be used ✓ VERIFIED (source code shows usage)
```

---

## 2. Potential Gaps Identified

### GAP-001: stable_whisper.transcribe_any() Behavior

**Location**: qwen_asr.py:625-639

```python
result = stable_whisper.transcribe_any(
    inference_func=qwen_inference,
    audio=str(audio_path),
    audio_type='str',
    regroup=True,      # <-- Uses default regrouping algorithm
    vad=False,
    demucs=False,
    suppress_silence=True,
    suppress_word_ts=True,
    verbose=False,
)
```

**Potential Issue**:
- `regroup=True` uses stable-ts's default regrouping algorithm ('da')
- This algorithm groups words into sentences based on timing/silence
- Even if qwen-asr produces MORE words with higher token limit, the regrouping might produce IDENTICAL sentence-level segments if the timing is similar

**Impact**: Could explain identical output
**Test**: Compare word-level output before regrouping

---

### GAP-002: qwen_inference() Closure

**Location**: qwen_asr.py:538-612

```python
def qwen_inference(audio: str, **kwargs) -> List[List[dict]]:
    results = qwen_model.transcribe(
        audio=str(audio),
        context=qwen_context,
        language=qwen_language,
        return_time_stamps=True,
    )
    # Converts to word list for stable-ts
```

**Potential Issue**:
- The closure captures `qwen_model` at function definition time
- If stable_whisper.transcribe_any() is caching something, old inference could be used

**Impact**: Unlikely but possible
**Test**: Add logging inside qwen_inference()

---

### GAP-003: HuggingFace Model Caching

**Location**: External (HuggingFace Hub cache)

**Potential Issue**:
- HuggingFace caches downloaded models in ~/.cache/huggingface/
- Model WEIGHTS are cached, but generation parameters like max_new_tokens are NOT baked into weights
- Each from_pretrained() call uses the parameters passed at that time

**Impact**: Should NOT cause identical output
**Conclusion**: Not the issue

---

### GAP-004: Audio Too Short

**Potential Issue**:
- If test audio is < 60 seconds, even 256 tokens might be sufficient
- Japanese speech: ~400 chars/min = ~800 tokens/min
- 60 seconds = ~800 tokens needed
- 256 tokens would truncate at ~20 seconds

**Impact**: If audio > 20 seconds but truncated before, should see difference now
**Test**: User should test with audio > 2 minutes

---

### GAP-005: Deterministic Model Output

**Potential Issue**:
- Same audio + same model weights = same output
- If the audio was never actually truncated (was short enough), output would be identical

**Impact**: Expected behavior for short audio
**Test**: User should verify audio length and check if truncation actually occurred before

---

## 3. Diagnostic Logging Recommendations

### Add to qwen_asr.py:

```python
# In qwen_inference() closure:
logger.info(f"[DIAG] qwen_inference called for: {audio}")
logger.info(f"[DIAG] Using max_new_tokens={qwen_model.max_new_tokens}")

# After qwen_model.transcribe():
logger.info(f"[DIAG] Raw result text length: {len(result.text)}")
logger.info(f"[DIAG] Word count: {len(result.time_stamps) if result.time_stamps else 0}")
```

### Add before stable_whisper.transcribe_any():

```python
logger.info(f"[DIAG] Calling transcribe_any with audio: {audio_path}")
logger.info(f"[DIAG] regroup={True}, suppress_silence={True}")
```

---

## 4. Test Scenarios

### Test 1: Verify Truncation Was Happening Before

1. Find audio file > 2 minutes
2. Run with old code (max_new_tokens=256)
3. Check if output ends abruptly mid-sentence
4. If yes, truncation was happening

### Test 2: Compare Word-Level Output

1. Modify qwen_inference() to log word count
2. Run same audio with both settings
3. Compare word counts

### Test 3: Bypass stable-ts Regrouping

1. Modify code to set `regroup=False`
2. Compare raw word-level output

---

## 5. Most Likely Explanation

Given that all parameter passing is verified correct, the most likely explanations are:

1. **Audio was short enough** - The test audio was < 20 seconds, so 256 tokens was sufficient
2. **stable-ts regrouping normalizes** - Even with more words, sentence boundaries are the same
3. **No actual truncation before** - The "before" test also had sufficient tokens

---

## 6. Next Steps

1. User should specify the duration of the test audio
2. User should run with `--log-level DEBUG` to see diagnostic messages
3. User should compare a LONG audio file (> 5 minutes) with and without scene detection
4. Add diagnostic logging to qwen_inference() closure
