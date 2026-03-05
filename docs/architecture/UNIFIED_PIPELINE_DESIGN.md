# Unified Pipeline Design: Mix-and-Match Architecture

**Created**: 2026-02-01
**Status**: Planning
**Goal**: Enable modular pipeline where users can mix-and-match components

---

## 1. Problem Statement

### Current State

**TransformersPipeline/Qwen Mode Limitations:**
- Scene detection: Only `auditok`, `silero` (missing `semantic`)
- Speech segmentation: None (relies on HF/qwen-asr internal handling)
- Speech enhancement: Basic support, default `none`
- No CLI exposure for segmenter selection

**BalancedPipeline Has:**
- Full scene detection (auditok, silero, semantic)
- Speech segmentation via Silero VAD
- Speech enhancement with JIT cleanup pattern
- Rich metadata contracts

**Ensemble Mode Has:**
- Full mix-and-match via `pass_config`
- `scene_detector`, `speech_segmenter`, `speech_enhancer` overrides
- Works for any pipeline

### User's Vision

A unified flow where any mode can use any component:

```
Scene Detection → Speech Segmenter → Speech Enhancer → ASR → Post-processing → Translator
     ↓                   ↓                  ↓            ↓
  auditok            silero             clearvoice     qwen
  silero              ten              bs-roformer      hf
  semantic           nemo               zipenhancer   faster
   none           whisper-vad              none       openai
                     none
```

---

## 2. Available Components Inventory

### Scene Detectors
| Name | Description | Where Implemented |
|------|-------------|-------------------|
| `auditok` | Energy-based, two-pass | `DynamicSceneDetector` |
| `silero` | VAD-based | `DynamicSceneDetector` |
| `semantic` | Sentence boundaries | `DynamicSceneDetector` |
| `none` | No scene splitting | All pipelines |

### Speech Segmenters (VAD)
| Name | Description | Where Implemented |
|------|-------------|-------------------|
| `silero` / `silero-v4.0` | Silero VAD v4.0 | `SpeechSegmenterFactory` |
| `silero-v3.1` | Silero VAD v3.1 | `SpeechSegmenterFactory` |
| `nemo` / `nemo-lite` | NeMo speech segmentation | `SpeechSegmenterFactory` |
| `whisper-vad` | Whisper-based VAD | `SpeechSegmenterFactory` |
| `ten` | TEN VAD | `SpeechSegmenterFactory` |
| `none` | Skip segmentation | `SpeechSegmenterFactory` |

### Speech Enhancers
| Name | Description | Where Implemented |
|------|-------------|-------------------|
| `none` | Passthrough (48kHz→16kHz resample) | `SpeechEnhancerFactory` |
| `clearvoice` | ClearerVoice denoising | `SpeechEnhancerFactory` |
| `bs-roformer` | BS-RoFormer vocal isolation | `SpeechEnhancerFactory` |
| `zipenhancer` | Lightweight SOTA enhancement | `SpeechEnhancerFactory` |
| `ffmpeg-dsp` | Audio filters | `SpeechEnhancerFactory` |

### ASR Backends
| Name | Description | Where Implemented |
|------|-------------|-------------------|
| `faster-whisper` | FasterWhisper (ctranslate2) | BalancedPipeline |
| `openai-whisper` | OpenAI Whisper | FastPipeline, FidelityPipeline |
| `hf-transformers` | HuggingFace Transformers | TransformersPipeline |
| `qwen` | Qwen3-ASR | TransformersPipeline (v1.8.3+) |
| `kotoba` | Kotoba Faster Whisper | KotobaFasterWhisperPipeline |

---

## 3. Design Approach: Minimal Changes, Maximum Flexibility

### Principle: Use What Already Exists

The ensemble mode (`pass_worker.py`) already has the logic to:
1. Override scene detector
2. Override speech segmenter
3. Override speech enhancer
4. Route parameters to correct config sections

**Key Insight**: TransformersPipeline already has `DynamicSceneDetector` and `SpeechEnhancer` integration. What's missing is:
1. CLI exposure for all options
2. Speech segmenter integration
3. Semantic scene detection in choices

---

## 4. Implementation Plan

### Phase 1: Expose Missing CLI Options (Low Effort) ✅ COMPLETED

**Goal**: Allow `--mode qwen` to use all scene detectors and enhancers

**Status**: Implemented 2026-02-01

**Changes Made:**

1. **main.py** - Added missing choices to Qwen arguments:
   - `--qwen-scene`: Added `semantic` to choices (now: none, auditok, silero, semantic)
   - `--qwen-enhancer`: New argument (choices: none, clearvoice, bs-roformer, zipenhancer, ffmpeg-dsp)
   - `--qwen-enhancer-model`: New argument for enhancer model variant

2. **transformers_pipeline.py** - Added qwen_enhancer support:
   - Added `qwen_enhancer` and `qwen_enhancer_model` constructor parameters
   - Updated `_enhancer_config` to use qwen_enhancer when `asr_backend="qwen"`

**Files Changed:**
- `whisperjav/main.py`: Lines 442-454, 779-780
- `whisperjav/pipelines/transformers_pipeline.py`: Lines 92-93, 131-134, 207-216

**Usage Example:**
```bash
# Qwen mode with semantic scene detection and clearvoice enhancement
whisperjav video.mp4 --mode qwen --qwen-scene semantic --qwen-enhancer clearvoice

# Qwen mode with BS-RoFormer vocal isolation
whisperjav video.mp4 --mode qwen --qwen-enhancer bs-roformer
```

---

### Phase 2: Add Speech Segmenter to TransformersPipeline (Medium Effort) ✅ COMPLETED

**Goal**: Allow Qwen/HF mode to use Silero/TEN/nemo VAD for speech segmentation

**Status**: Implemented 2026-02-01

**Architecture Decision**: Option B (post-filter)

```
Audio → Scene Detection → Enhancement → ASR (full audio) → [VAD Filter] → Output
```

- Simpler to implement
- HF/Qwen ASR already handles long audio
- VAD used to filter hallucinations in non-speech regions

**Implementation:**

1. **main.py** - Added `--qwen-segmenter` argument with choices:
   - none, silero, silero-v4.0, silero-v3.1, nemo, nemo-lite, whisper-vad, ten

2. **transformers_pipeline.py** - Added post-ASR VAD filtering:
   - Added `qwen_segmenter` constructor parameter
   - Added `_filter_segments_by_vad()` method
   - Filters segments with < 30% overlap with VAD speech regions
   - Proper VRAM cleanup after segmenter use
   - Added metadata tracking for post_asr_vad_filter

**Files Changed:**
- `whisperjav/main.py`: Lines 455-458, 785
- `whisperjav/pipelines/transformers_pipeline.py`: Lines 94, 136, 222, 344-432, 721-723, 773-775, 543

**Usage Example:**
```bash
# Qwen mode with post-ASR VAD filtering to remove hallucinations
whisperjav video.mp4 --mode qwen --qwen-segmenter silero

# Full pipeline: semantic scenes + enhancement + VAD filter
whisperjav video.mp4 --mode qwen --qwen-scene semantic --qwen-enhancer clearvoice --qwen-segmenter silero-v4.0
```

---

### Phase 3: Unified CLI Interface (Future)

**Goal**: Single interface that works for all modes

```bash
# Current (mode-specific args):
whisperjav video.mp4 --mode qwen --qwen-scene auditok --qwen-enhancer clearvoice

# Future (unified args):
whisperjav video.mp4 --mode qwen --scene auditok --enhancer clearvoice --segmenter silero
```

**Changes:**
- Add generic `--scene`, `--enhancer`, `--segmenter` arguments
- Mode-specific args become aliases/overrides
- Reduces cognitive load for users

---

## 5. Minimal Implementation (Phase 1 Only)

For immediate fix, implement only Phase 1:

### Changes Required

**File 1: `whisperjav/main.py`**

```python
# Line ~442: Fix --qwen-scene choices
qwen_group.add_argument("--qwen-scene", type=str, default="none",
                       choices=["none", "auditok", "silero", "semantic"],
                       help="Scene detection method (default: none)")

# Add after line ~444:
qwen_group.add_argument("--qwen-enhancer", type=str, default="none",
                       choices=["none", "clearvoice", "bs-roformer", "zipenhancer", "ffmpeg-dsp"],
                       help="Speech enhancement backend (default: none)")
qwen_group.add_argument("--qwen-enhancer-model", type=str, default=None,
                       help="Speech enhancer model variant")

# Line ~751-774: Pass enhancer params to pipeline
qwen_enhancer=getattr(args, 'qwen_enhancer', 'none'),
qwen_enhancer_model=getattr(args, 'qwen_enhancer_model', None),
```

**File 2: `whisperjav/pipelines/transformers_pipeline.py`**

```python
# __init__ signature: Add qwen_enhancer, qwen_enhancer_model parameters

# _enhancer_config: Use qwen_enhancer when asr_backend="qwen"
```

---

## 6. Priority Matrix

| Feature | Effort | Impact | Priority | Status |
|---------|--------|--------|----------|--------|
| Add `semantic` to --qwen-scene | Low | Medium | P1 | ✅ Done |
| Add --qwen-enhancer argument | Low | High | P1 | ✅ Done |
| Wire enhancer to TransformersPipeline | Low | High | P1 | ✅ Done |
| Add --qwen-segmenter argument | Medium | Medium | P2 | ✅ Done |
| Implement post-ASR VAD filtering | Medium | Medium | P2 | ✅ Done |
| Unified --scene/--enhancer/--segmenter | High | High | P3 | Pending |

---

## 7. Relationship to Ensemble Mode

The ensemble mode (`EnsembleOrchestrator`) already supports all this via `pass_config`:

```python
pass1_config = {
    'pipeline': 'qwen',              # Use Qwen mode
    'scene_detector': 'semantic',    # Full scene detector options
    'speech_enhancer': 'clearvoice', # Full enhancer options
    'speech_segmenter': 'silero',    # Full segmenter options
}
```

**Key Insight**: Ensemble mode's `pass_worker.py` (lines 394-642) shows exactly how to route these parameters. We can reuse this logic for single-pass modes.

---

## 8. Summary

### What's Already There
- `DynamicSceneDetector` with auditok/silero/semantic
- `SpeechEnhancerFactory` with all enhancers
- `SpeechSegmenterFactory` with all segmenters
- TransformersPipeline with full enhancement support (Phase 1 ✅)
- Ensemble mode with full mix-and-match

### What's Now Available (Phase 1 + Phase 2 ✅)
- Semantic scene detection in Qwen mode (`--qwen-scene semantic`)
- Speech enhancer in Qwen mode (`--qwen-enhancer clearvoice|bs-roformer|zipenhancer|ffmpeg-dsp`)
- Enhancer model variants (`--qwen-enhancer-model`)
- Post-ASR VAD filtering (`--qwen-segmenter silero|nemo|whisper-vad|ten`)

### What's Still Missing
- Unified CLI interface across all modes (Phase 3)

### Recommended Next Steps
1. ~~**Phase 1** (Immediate): Add missing CLI args, wire to pipeline~~ ✅ Done
2. ~~**Phase 2** (Next): Add speech segmenter with post-ASR filtering~~ ✅ Done
3. **Phase 3** (Future): Unified CLI interface

This approach:
- Uses existing components (no new code needed)
- Minimal changes (just CLI args and parameter routing)
- No over-engineering
- Consistent with ensemble mode's design
