# Qwen-ASR Integration: Architectural Remediation Plan

**Created**: 2026-01-31
**Status**: Active
**Owner**: Senior Architect

---

## Executive Summary

The current Qwen-ASR integration in WhisperJAV has several architectural issues that cause:
1. Truncated transcriptions for audio > 1 minute
2. Missing functionality available in qwen-asr
3. Potential conflicts between stable-ts and qwen-asr chunking
4. Suboptimal performance (missing flash attention)

This document provides a comprehensive remediation plan.

---

## 1. Use Case Requirements

### Target Use Cases

| Use Case | Audio Duration | Description |
|----------|----------------|-------------|
| UC-1 | 1-3 minutes | Single scene from scene detection |
| UC-2 | 3-10 minutes | Long scene or disabled scene detection |
| UC-3 | 10-30 minutes | Full short video without scene detection |
| UC-4 | 30-120 minutes | Full video (must use scene detection) |

### Design Constraints

1. **Memory**: Single GPU (8-24GB VRAM typical)
2. **Accuracy**: Prioritize accuracy over speed
3. **Robustness**: Handle edge cases gracefully
4. **Compatibility**: Maintain backward compatibility with CLI

---

## 2. Issues Registry

### ISSUE-001: max_new_tokens Too Low [CRITICAL]

**Current State**:
- Default: 256 tokens
- Transformers backend qwen-asr default: 512
- vLLM backend qwen-asr default: 4096
- Evaluation benchmark: 1024

**Impact**:
- Truncated transcriptions after ~1 minute of audio
- "Fatigue" pattern where output degrades
- Missing content in final subtitles

**Root Cause**:
Japanese speech rate: ~300-400 characters/minute
Tokenization ratio: ~1.5-2 tokens per character
10 minutes = 3000-4000 characters = 4500-8000 tokens

**Solution**:
- Calculate based on target duration (10 min)
- Conservative estimate: 10 min × 400 char/min × 2 tokens/char = 8000 tokens
- Set default to 4096 (safe for 10 min, memory efficient)
- Expose as user-configurable parameter

**Files to Change**:
- `whisperjav/modules/qwen_asr.py`: DEFAULT_MAX_NEW_TOKENS = 4096
- `whisperjav/main.py`: --qwen-max-tokens default = 4096
- `whisperjav/pipelines/transformers_pipeline.py`: qwen_max_tokens = 4096
- `whisperjav/config/v4/ecosystems/qwen/*.yaml`: Update defaults

---

### ISSUE-002: ForcedAligner 3-Minute Limit [HIGH]

**Current State**:
- qwen-asr constant: MAX_FORCE_ALIGN_INPUT_SECONDS = 180 (3 min)
- WhisperJAV: Not aware of this limit

**Impact**:
- For audio > 3 minutes, qwen-asr internally chunks
- Results may have discontinuities at chunk boundaries
- Timestamp accuracy may degrade at boundaries

**Root Cause**:
qwen-asr splits audio into 180-second chunks for ForcedAligner processing.
WhisperJAV's scene detection typically produces 30-180 second scenes.
When scene > 180s, internal chunking activates.

**Solution Options**:

Option A: Rely on qwen-asr internal chunking
- Pros: Simple, uses tested code
- Cons: No control, boundary issues possible

Option B: Pre-chunk in WhisperJAV to 180s
- Pros: Explicit control, can add overlap
- Cons: Duplicates qwen-asr logic, complexity

Option C: Document limitation, recommend scene detection
- Pros: Simple, uses existing infrastructure
- Cons: User education needed

**Recommended**: Option A + Documentation
- Trust qwen-asr's internal chunking (it merges results)
- Document that scene detection is recommended for videos > 3 min
- Add warning log when audio > 180s without scene detection

**Files to Change**:
- `whisperjav/modules/qwen_asr.py`: Add warning for long audio
- `docs/`: Document the 3-minute ForcedAligner limit
- CLI help text: Recommend scene detection for long videos

---

### ISSUE-003: context Parameter Not Exposed [MEDIUM] ✅ COMPLETED

**Current State**: ~~Not exposed~~ → **Now exposed via --qwen-context CLI argument**

**Implementation (2026-01-31)**:
- Added `context` parameter to QwenASR class (`whisperjav/modules/qwen_asr.py`)
- Added `--qwen-context` CLI argument (`whisperjav/main.py`)
- Added `qwen_context` to TransformersPipeline (`whisperjav/pipelines/transformers_pipeline.py`)
- Context is passed through to qwen-asr transcribe() calls
- Added unit tests for context parameter

**Usage**:
```bash
whisperjav video.mp4 --mode qwen --qwen-context "Adult video transcription with speaker names"
```

**Files Changed**:
- `whisperjav/modules/qwen_asr.py`: Added context parameter
- `whisperjav/main.py`: Added --qwen-context argument
- `whisperjav/pipelines/transformers_pipeline.py`: Pass context through
- `tests/test_qwen_asr.py`: Added TestQwenASRContextParameter test class

---

### ISSUE-004: attn_implementation Not Exposed [MEDIUM] ✅ COMPLETED

**Current State**: ~~Not exposed~~ → **Now exposed via --qwen-attn CLI argument with auto-detection**

**Implementation (2026-01-31)**:
- Added `attn_implementation` parameter to QwenASR class with "auto" default
- Added `_detect_attn_implementation()` method that:
  - Uses explicit value if not "auto"
  - Auto-detects flash-attn availability on CUDA
  - Falls back to "sdpa" on CPU or when flash-attn unavailable
- Added `--qwen-attn` CLI argument with choices: auto, sdpa, flash_attention_2, eager
- Added `qwen_attn` to TransformersPipeline
- Added unit tests for attention implementation detection

**Usage**:
```bash
# Auto-detect (default - uses flash_attention_2 if available)
whisperjav video.mp4 --mode qwen

# Force specific implementation
whisperjav video.mp4 --mode qwen --qwen-attn flash_attention_2
whisperjav video.mp4 --mode qwen --qwen-attn sdpa
```

**Files Changed**:
- `whisperjav/modules/qwen_asr.py`: Added attn_implementation parameter and _detect_attn_implementation()
- `whisperjav/main.py`: Added --qwen-attn argument
- `whisperjav/pipelines/transformers_pipeline.py`: Added qwen_attn parameter
- `tests/test_qwen_asr.py`: Added TestQwenASRAttentionImplementation test class

---

### ISSUE-005: forced_aligner_kwargs Not Exposed [LOW] ✅ DOCUMENTED

**Current State**: Documented in QwenASR class docstring

**Implementation (2026-01-31)**:
- Added comprehensive documentation to QwenASR class docstring
- Explains that aligner uses same device/dtype as main model by default
- Shows Python API example for advanced users who need separate aligner config
- Not exposed via CLI (edge case, use Python API if needed)

**Documentation Added** (`whisperjav/modules/qwen_asr.py`):
```
Aligner Configuration (ISSUE-005):
    - ForcedAligner uses same device/dtype as main model by default
    - To configure aligner separately, use forced_aligner_kwargs in qwen-asr
```

---

### ISSUE-006: Naming Inconsistencies [LOW] ✅ DOCUMENTED

**Current State**: Parameter mapping documented in QwenASR class docstring

**Implementation (2026-01-31)**:
- Added parameter mapping table to QwenASR class docstring
- Clear documentation of WhisperJAV → qwen-asr name mappings

**Documentation Added** (`whisperjav/modules/qwen_asr.py`):
```
Parameter Mapping (ISSUE-005/006):
    WhisperJAV Name              qwen-asr Name
    ─────────────────────────────────────────────────────────
    model_id                  → pretrained_model_name_or_path
    device                    → device_map
    batch_size                → max_inference_batch_size
    aligner_id + use_aligner  → forced_aligner (combined)
    dtype                     → dtype (torch.dtype)
    attn_implementation       → attn_implementation
```

---

### ISSUE-007: stable-ts vs qwen-asr Chunking Interaction [INVESTIGATION] ✅ DOCUMENTED

**Investigation Complete (2026-01-31)**:

**Finding 1: stable_whisper.transcribe_any() does NOT chunk audio**
- It passes the full audio path to the inference callback
- All chunking is handled by qwen-asr internally

**Finding 2: qwen-asr chunking behavior**
- With ForcedAligner: Chunks audio > 180 seconds (3 min)
- Without ForcedAligner: Handles audio up to 1200 seconds (20 min)
- Results are merged internally by qwen-asr

**Finding 3: No double-chunking conflict**
- WhisperJAV's stable-ts integration does not add chunking
- Only qwen-asr's internal chunking applies

**Recommendation**: Use WhisperJAV scene detection for videos > 3 minutes
to get explicit control over chunk boundaries and avoid internal chunking.

**Documentation Added** (`whisperjav/modules/qwen_asr.py`):
```
Chunking Behavior (ISSUE-007):
    - stable_whisper.transcribe_any() does NOT chunk audio internally
    - qwen-asr handles chunking for audio > 180s (with aligner)
    - Recommendation: Use scene detection for videos > 3 minutes
```

---

### ISSUE-008: Error Handling for Long Audio [MEDIUM]

**Current State**:
- OOM handling exists (retry with smaller batch)
- No handling for token limit exceeded
- No handling for ForcedAligner timeout

**Impact**:
- Silent failures for long audio
- Users don't know why transcription is incomplete

**Solution**:
- Add token count estimation before transcription
- Warn if estimated tokens > max_new_tokens
- Add timeout handling for ForcedAligner
- Provide actionable error messages

**Files to Change**:
- `whisperjav/modules/qwen_asr.py`: Add estimation and warnings

---

### ISSUE-009: Progress Reporting for Long Audio [LOW] ✅ COMPLETED

**Implementation (2026-01-31)**:
- Added audio duration detection before transcription starts
- Enhanced progress message shows duration (e.g., "Transcribing 5.0min audio: file.wav")
- Added warning log for audio > 3 minutes: "Processing X min audio - this may take several minutes..."
- User is now informed about expected processing time for long audio

**Files Changed**:
- `whisperjav/modules/qwen_asr.py`: Enhanced transcribe() progress messages

---

### ISSUE-010: Memory Management for Long Audio [MEDIUM] ✅ COMPLETED

**Implementation (2026-01-31)**:
- Added gc.collect() after each transcription to release Python objects
- Added GPU memory logging after transcription (allocated/reserved)
- Memory monitoring helps diagnose OOM issues

**Files Changed**:
- `whisperjav/modules/qwen_asr.py`: Added memory management after transcription

**Code Added**:
```python
# Memory management after transcription (ISSUE-010)
gc.collect()
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.debug(f"GPU memory after transcription: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

---

## 3. Implementation Priority

### Phase 1: Critical Fixes (Immediate)

| Issue | Priority | Effort | Risk |
|-------|----------|--------|------|
| ISSUE-001: max_new_tokens | CRITICAL | Low | Low |
| ISSUE-002: 3-min limit docs | HIGH | Low | Low |
| ISSUE-008: Error handling | MEDIUM | Medium | Low |

### Phase 2: Feature Enhancements ✅ COMPLETED (2026-01-31)

| Issue | Priority | Effort | Risk | Status |
|-------|----------|--------|------|--------|
| ISSUE-003: context param | MEDIUM | Low | Low | ✅ Done |
| ISSUE-004: attn_implementation | MEDIUM | Medium | Medium | ✅ Done |
| ISSUE-009: Progress reporting | LOW | Low | Low | Pending |

### Phase 3: Investigation & Polish ✅ COMPLETED (2026-01-31)

| Issue | Priority | Effort | Risk | Status |
|-------|----------|--------|------|--------|
| ISSUE-007: Chunking investigation | HIGH | Medium | Unknown | ✅ Documented |
| ISSUE-005: aligner kwargs | LOW | Low | Low | ✅ Documented |
| ISSUE-006: Naming | LOW | Low | Medium | ✅ Documented |
| ISSUE-009: Progress reporting | LOW | Low | Low | ✅ Done |
| ISSUE-010: Memory management | MEDIUM | Medium | Low | ✅ Done |

---

## 4. Detailed Implementation Plan

### 4.1 Phase 1 Implementation

#### Step 1: Fix max_new_tokens (ISSUE-001)

**Calculation for 10-minute target**:
- Japanese speech: 400 characters/minute (conservative)
- 10 minutes = 4000 characters
- Tokenization overhead: 2x (conservative for Japanese)
- Required: 8000 tokens
- Recommended default: 4096 (balance of coverage and memory)
- Allow user override up to 8192

**Changes**:

1. `whisperjav/modules/qwen_asr.py`:
```python
DEFAULT_MAX_NEW_TOKENS = 4096  # Supports ~5-10 min audio
```

2. `whisperjav/main.py`:
```python
qwen_group.add_argument("--qwen-max-tokens", type=int, default=4096,
                       help="Maximum tokens (default: 4096, supports ~10 min audio)")
```

3. `whisperjav/pipelines/transformers_pipeline.py`:
```python
qwen_max_tokens: int = 4096,
```

4. YAML configs: Update all instances

#### Step 2: Document 3-Minute Limit (ISSUE-002)

**Changes**:

1. `whisperjav/modules/qwen_asr.py`:
```python
# In transcribe():
audio_duration = self._get_audio_duration(audio_path)
if audio_duration > 180 and self.use_aligner:
    logger.warning(
        f"Audio duration ({audio_duration:.0f}s) exceeds ForcedAligner limit (180s). "
        "For best results, enable scene detection (--qwen-scene auditok). "
        "Transcription will continue with qwen-asr internal chunking."
    )
```

2. CLI help: Add recommendation for scene detection

#### Step 3: Improve Error Handling (ISSUE-008)

**Changes**:

1. Add token estimation:
```python
def _estimate_tokens(self, audio_duration: float) -> int:
    """Estimate tokens needed for audio duration."""
    # Japanese: ~400 chars/min, ~2 tokens/char
    chars_per_minute = 400
    tokens_per_char = 2
    return int(audio_duration / 60 * chars_per_minute * tokens_per_char)

def transcribe(self, audio_path, ...):
    duration = self._get_audio_duration(audio_path)
    estimated_tokens = self._estimate_tokens(duration)

    if estimated_tokens > self.max_new_tokens:
        logger.warning(
            f"Audio may need ~{estimated_tokens} tokens but max_new_tokens={self.max_new_tokens}. "
            f"Consider increasing --qwen-max-tokens or enabling scene detection."
        )
```

---

## 5. Validation Plan

### Test Cases

| Test | Duration | Expected Result |
|------|----------|-----------------|
| TC-1 | 30 sec | Full transcription, accurate timestamps |
| TC-2 | 2 min | Full transcription, no truncation |
| TC-3 | 5 min | Full transcription, warning if no scene detection |
| TC-4 | 10 min | Full transcription with scene detection |
| TC-5 | 10 min no scene | Warning issued, best-effort transcription |

### Acceptance Criteria

1. 10-minute audio produces complete transcription (no truncation)
2. Warnings are logged for audio > 3 min without scene detection
3. Token estimation matches actual usage (±20%)
4. No OOM for audio up to 10 minutes on 8GB VRAM

---

## 6. Rollout Plan

1. **Development**: Implement Phase 1 changes
2. **Testing**: Run test cases TC-1 through TC-5
3. **Documentation**: Update README, CLI help
4. **Release**: Include in next version

---

## Appendix: qwen-asr Constants Reference

```python
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200      # 20 min (without timestamps)
MAX_FORCE_ALIGN_INPUT_SECONDS = 180  # 3 min (with timestamps)
MIN_ASR_INPUT_SECONDS = 0.5
```

## Appendix: Token Estimation Formula

```
tokens = (audio_duration_minutes) × (chars_per_minute) × (tokens_per_char)

Conservative Japanese:
tokens = duration_min × 400 × 2 = duration_min × 800

Example:
10 min audio → 10 × 800 = 8000 tokens
```
