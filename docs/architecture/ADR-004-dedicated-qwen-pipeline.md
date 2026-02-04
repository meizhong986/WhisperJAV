# ADR-004: Dedicated QwenPipeline Architecture

**Created**: 2026-02-03
**Updated**: 2026-02-04 (v1.0 — Context biasing feature added)
**Status**: DRAFT v1.0 - Under Review
**Author**: Senior Architect
**Reviewed By**: External Architect Review (2026-02-04), User Design Review (2026-02-04)
**Supersedes**: Parts of ADR-003 (Qwen3-ASR Integration)

---

## Glossary

| Term | Definition |
|------|------------|
| **SRT** | SubRip Subtitle format - standard subtitle file format with timestamps |
| **ASR** | Automatic Speech Recognition - converting audio to text |
| **VAD** | Voice Activity Detection - identifying speech vs silence in audio |
| **VRAM** | Video RAM - GPU memory used for model inference |
| **JIT** | Just-In-Time - loading/unloading models on demand to manage memory |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Context](#2-background-and-context)
3. [Problem Statement](#3-problem-statement)
4. [Architectural Analysis](#4-architectural-analysis)
5. [Decision](#5-decision)
6. [Detailed Design](#6-detailed-design)
7. [Module Contracts](#7-module-contracts)
8. [Configuration Architecture](#8-configuration-architecture)
9. [CLI Arguments Strategy](#9-cli-arguments-strategy)
10. [Impact on Existing Code](#10-impact-on-existing-code)
11. [Implementation Plan](#11-implementation-plan)
12. [Risk Assessment](#12-risk-assessment)
13. [Acceptance Criteria](#13-acceptance-criteria)
14. [Error Handling Strategy](#14-error-handling-strategy)
15. [Logging Standards](#15-logging-standards)
16. [Future Work](#16-future-work)
17. [Appendices](#appendices)

---

## 1. Executive Summary

### Problem
The current Qwen3-ASR integration embeds Qwen as a backend switch (`asr_backend="qwen"`) inside `TransformersPipeline`. This creates:
- 45+ parameters in a single constructor
- Conditional branching throughout the codebase
- Log inconsistencies and debugging complexity
- Complex parameter translation through multiple layers

### Decision
Create a **dedicated `QwenPipeline` class** that:
- Lives in its own file (`whisperjav/pipelines/qwen_pipeline.py`)
- Has only Qwen-specific parameters (~20 vs 45+)
- Uses the existing V4 YAML configuration system
- Follows the "redundancy over reuse" principle per original requirements

### Impact
- **New file**: `qwen_pipeline.py` (~500-600 lines, mostly orchestration)
- **Modified**: `pass_worker.py` (routing change)
- **Modified**: `main.py` (routing change)
- **Cleanup**: Remove Qwen backend code from `TransformersPipeline`
- **Fix**: `JapanesePostProcessor` internal step 6 (preserve tiny fragments)
- **Feature**: Context biasing with cross-scene propagation and file loading
- **Modified**: `qwen_asr.py` (add per-call `context` parameter to `transcribe()`)

---

## 2. Background and Context

### 2.1 Original Feature Request

The original feature request for Qwen3-ASR integration specified:

> **"I prefer redundancy over reuse"** - The user explicitly requested that Qwen should have its own dedicated pipeline rather than being integrated into an existing pipeline.

This instruction was not followed. Instead, Qwen was implemented as a backend switch within `TransformersPipeline`.

### 2.2 What is Qwen3-ASR?

Qwen3-ASR is a fundamentally different ASR paradigm from Whisper-based models:

| Aspect | Whisper Models | Qwen3-ASR |
|--------|---------------|-----------|
| Output | Timestamped segments | Raw text only |
| Timestamps | Built-in | Requires ForcedAligner |
| Chunking | External (stable-ts) | Internal (qwen-asr handles) |
| Languages | Requires language code | Auto-detects |
| Max Duration | Model-dependent | 1200s (ASR) / 180s (Aligner) |

This is not a configuration difference—it's a fundamentally different **data flow**.

### 2.3 Existing Documentation

The following documents provide context:

| Document | Purpose | Key Contents |
|----------|---------|--------------|
| `ADR-003-qwen3-asr-integration.md` | Original integration design | Initial architecture decision for Qwen3-ASR support; documents the (flawed) decision to embed Qwen in TransformersPipeline |
| `QWEN_ASR_REMEDIATION_PLAN.md` | Issue registry and fixes | Tracks known issues with current implementation, including language code normalization and log inconsistencies |
| `QWEN_ASR_FLOW_AUDIT.md` | Parameter flow tracing | Detailed trace of how parameters flow from GUI → CLI → Pipeline → Module; useful for understanding the translation layers |
| `UNIFIED_PIPELINE_DESIGN.md` | Mix-and-match component vision | Future architecture vision for composable pipelines; this ADR aligns with that direction |

### 2.4 Existing V4 Configuration

Qwen already has V4 YAML configuration:

```
config/v4/ecosystems/qwen/
├── ecosystem.yaml          # Provider definition
└── models/
    └── qwen3-asr-1.7b.yaml # Model parameters
```

This infrastructure should be leveraged by the dedicated pipeline.

---

## 3. Problem Statement

### 3.1 Current Architecture (Flawed)

```
┌─────────────────────────────────────────────────────────────┐
│                  pass_worker.py                              │
│                                                              │
│   PIPELINE_CLASSES = {                                       │
│     "balanced": BalancedPipeline,                           │
│     "transformers": TransformersPipeline,                   │
│     "qwen": TransformersPipeline,  ◄─── SAME CLASS!         │
│   }                                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TransformersPipeline                            │
│                                                              │
│   __init__(                                                  │
│     asr_backend: str = "hf",   ◄─── Backend switch          │
│     # HF parameters (15+)                                    │
│     hf_model_id, hf_chunk_length, hf_stride, hf_batch_size, │
│     hf_scene, hf_beam_size, hf_temperature, hf_attn, ...    │
│     # Qwen parameters (15+)                                  │
│     qwen_model_id, qwen_device, qwen_dtype, qwen_batch_size,│
│     qwen_max_tokens, qwen_language, qwen_timestamps, ...    │
│   )                                                          │
│                                                              │
│   # Conditional branching throughout:                        │
│   if self.asr_backend == "qwen":                            │
│       # Qwen-specific logic                                  │
│   else:                                                      │
│       # HF-specific logic                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Specific Issues

#### Issue 1: Parameter Explosion
`TransformersPipeline.__init__` has **45+ parameters** because it must handle both backends.

#### Issue 2: Log Inconsistency
```python
# In TransformersPipeline.process() - the model logging code ALWAYS
# reads from hf_config regardless of which backend is active:
logger.debug("process() started... (model=%s)", self.hf_config.get("model_id"))

# This is a bug: hf_config is ALWAYS populated (with HF defaults),
# even when asr_backend="qwen". The code should conditionally read
# from qwen_config when using Qwen backend.
```
When running Qwen mode, logs incorrectly show `kotoba-whisper-bilingual-v1.0` (HF default) instead of `Qwen/Qwen3-ASR-1.7B`.

#### Issue 3: Configuration Confusion
```
Resolved configuration for pipeline='balanced', sensitivity='aggressive'
```
This log appears even in ensemble mode when using Qwen pipeline, because `args.mode` defaults to "balanced" and the log runs unconditionally.

#### Issue 4: Complex Parameter Translation
```
GUI (app.js)
    → api.py (_build_twopass_args)
    → main.py (builds pass_config)
    → pass_worker.py (translates to pipeline params)
    → TransformersPipeline (switches on asr_backend)
    → QwenASR or TransformersASR
```
Multiple translation layers create debugging nightmares.

### 3.3 Additional Issue: JapanesePostProcessor

The `JapanesePostProcessor` has a related issue:

**JapanesePostProcessor internal step 6 (Merge Tiny Fragments)** currently **merges** segments that are too short:
```python
# Current behavior (WRONG for JAV):
if is_tiny:
    merge_with_previous_or_next()
```

**User requirement**: For JAV content, tiny fragments should be **preserved** because they may be moans or meaningful short utterances.

---

## 4. Architectural Analysis

### 4.1 Comparison: Reuse vs Redundancy

| Aspect | Current (Reuse) | Proposed (Redundancy) |
|--------|-----------------|----------------------|
| Code duplication | None | Some orchestration |
| Complexity | High (conditionals) | Low (single path) |
| Debugging | Hard (which backend?) | Easy (always Qwen) |
| Testing | Must test both paths | Isolated testing |
| Maintenance | Changes affect both | Isolated changes |
| Parameter count | 45+ | ~20 |

### 4.2 What Gets Duplicated

Only **orchestration logic** is duplicated:
- Step sequencing (extract → detect → enhance → transcribe → stitch → post-process)
- VRAM management (JIT load/unload pattern)
- Progress reporting
- Error handling

**NOT duplicated** (shared modules):
- `AudioExtractor`
- `DynamicSceneDetector`
- `SpeechEnhancer` (via factory)
- `SpeechSegmenter` (via factory)
- `QwenASR`
- `JapanesePostProcessor`
- `SRTStitcher`
- `SRTPostProcessor`

### 4.3 Pipeline Comparison: Fidelity vs Qwen

`FidelityPipeline` is a good template—clean, single-purpose:

| Aspect | FidelityPipeline | QwenPipeline (Proposed) |
|--------|-----------------|-------------------------|
| ASR Module | WhisperProASR | QwenASR |
| Scene Detection | DynamicSceneDetector | DynamicSceneDetector |
| Speech Enhancement | SpeechEnhancer (factory) | SpeechEnhancer (factory) |
| Speech Segmenter | Integrated in ASR | Pre-ASR VAD / Post-ASR filter (optional) |
| Post-processing | SRTPostProcessor | JapanesePostProcessor + SRTPostProcessor |
| Config System | V3 resolved_config | V4 YAML (recommended) |

---

## 5. Decision

### 5.1 Primary Decision

**Create a dedicated `QwenPipeline` class** following the "redundancy over reuse" principle.

### 5.2 Supporting Decisions

| Decision | Rationale |
|----------|-----------|
| Use V4 YAML config | Already exists, follows current architecture direction |
| Reuse existing CLI args | `--qwen-*` arguments are well-designed and complete |
| Revert TransformersPipeline | Remove Qwen backend code, simplify to HF-only |
| Fix JapanesePostProcessor | Internal step 6 should preserve (not merge) tiny fragments |
| Add Speech Segmenter to flow | Currently missing from architecture diagrams |

---

## 6. Detailed Design

### 6.1 Complete Data Flow Diagram

**CORRECTED FLOW** (Speech Segmenter AFTER Enhancer, BEFORE ASR):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QwenPipeline.process(media_info)                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: media_info = {path: "video.mkv", basename: "video", ...}

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 1: AUDIO EXTRACTION                                                ║
    ║                                                                          ║
    ║   video.mkv ──► AudioExtractor ──► video_extracted.wav (48kHz)           ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  Any media file (video or audio)                              ║
    ║     Output: WAV @ 48kHz (v1.7.4+ contract for enhancement compatibility) ║
    ║   Module: AudioExtractor (existing)                                      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 2: SCENE DETECTION (Optional, default: none)                       ║
    ║                                                                          ║
    ║   video_extracted.wav ──► DynamicSceneDetector                           ║
    ║                                    │                                     ║
    ║                                    ▼                                     ║
    ║   scenes/ ├── video_scene_0000.wav (0.0s - 45.2s)                        ║
    ║           ├── video_scene_0001.wav (45.2s - 112.8s)                      ║
    ║           └── ...                                                        ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  Full WAV file                                                ║
    ║     Output: List[(scene_path, start_sec, end_sec, duration_sec)]         ║
    ║     Config: method (none, auditok, silero, semantic)                     ║
    ║   Module: DynamicSceneDetector (existing)                                ║
    ║                                                                          ║
    ║   When method="none" (scene detection disabled):                         ║
    ║     - Pipeline treats full extracted WAV as a single scene               ║
    ║     - scene_paths = [(extracted_wav, 0.0, duration, duration)]            ║
    ║     - DynamicSceneDetector is NOT instantiated                            ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 3: SPEECH ENHANCEMENT (Optional, default: none)                    ║
    ║          *** EXCLUSIVE VRAM BLOCK 1 ***                                  ║
    ║                                                                          ║
    ║   For each scene:                                                        ║
    ║     scene_XXXX.wav (48kHz) ──► Enhancer ──► scene_XXXX_enh.wav (16kHz)   ║
    ║                                                                          ║
    ║   Then: enhancer.cleanup(); del enhancer; torch.cuda.empty_cache()       ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  Scene WAV @ 48kHz                                            ║
    ║     Output: Enhanced WAV @ 16kHz (resampled for ASR)                     ║
    ║     Config: backend (none, clearvoice, bs-roformer)                      ║
    ║   Module: SpeechEnhancer via factory (existing)                          ║
    ║   Helper: enhance_scenes() from pipeline_helper.py                       ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 4: SPEECH SEGMENTATION / VAD (Optional, default: none)             ║
    ║                                                                          ║
    ║   Purpose: Identify speech regions in CLEAN (enhanced) audio             ║
    ║            VAD on enhanced audio = more accurate detection               ║
    ║                                                                          ║
    ║   For each scene:                                                        ║
    ║     scene_XXXX_enh.wav (16kHz) ──► SpeechSegmenter ──► speech_regions    ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  Enhanced scene WAV @ 16kHz                                   ║
    ║     Output: SegmentationResult with List[SpeechSegment]                  ║
    ║             Each SpeechSegment has: start_sec, end_sec, confidence       ║
    ║     Config: backend (none, silero, ten, nemo, whisper-vad)               ║
    ║   Module: SpeechSegmenter via factory (existing)                         ║
    ║                                                                          ║
    ║   Passthrough Behavior (backend="none"):                                 ║
    ║     - Returns full audio as single speech region                         ║
    ║     - ASR processes entire scene (current default behavior)              ║
    ║                                                                          ║
    ║   Active VAD Behavior (backend="silero" etc):                            ║
    ║     - Returns list of speech regions within scene                        ║
    ║     - Used as POST-ASR filter (same pattern as TransformersPipeline):   ║
    ║       1. VAD runs on enhanced audio, produces speech_regions            ║
    ║       2. ASR transcribes full scene (QwenASR handles its own chunking) ║
    ║       3. Pipeline filters ASR segments: keep only those overlapping     ║
    ║          with speech_regions (min_overlap_ratio=0.3)                    ║
    ║     - This approach is preferred because QwenASR manages its own        ║
    ║       internal chunking and alignment — splitting audio into small      ║
    ║       speech clips would break its alignment mechanism                  ║
    ║     - Reduces hallucinations in silence/music/noise regions             ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 5: ASR TRANSCRIPTION                                               ║
    ║          *** EXCLUSIVE VRAM BLOCK 2 ***                                  ║
    ║                                                                          ║
    ║   Load QwenASR model once (deferred loading after Phase 3 cleanup)       ║
    ║                                                                          ║
    ║   ┌────────────────────────────────────────────────────────────────────┐ ║
    ║   │                INSIDE qwen_asr.py (QwenASR.transcribe)             │ ║
    ║   │                                                                    │ ║
    ║   │   For each scene (or speech region if VAD active):                 │ ║
    ║   │                                                                    │ ║
    ║   │   ┌────────────────────────────────────────────────────────────┐   │ ║
    ║   │   │ STEP 5a: ASR Transcription + Timestamp Alignment           │   │ ║
    ║   │   │   (Integrated by Qwen3-ASR package)                        │   │ ║
    ║   │   │                                                            │   │ ║
    ║   │   │   audio ──► qwen_model.transcribe(return_time_stamps=True) │   │ ║
    ║   │   │                         │                                  │   │ ║
    ║   │   │                         ▼                                  │   │ ║
    ║   │   │   raw_text + ForcedAligner timestamps                      │   │ ║
    ║   │   └────────────────────────────────────────────────────────────┘   │ ║
    ║   │                         │                                          │ ║
    ║   │                         ▼                                          │ ║
    ║   │   ┌────────────────────────────────────────────────────────────┐   │ ║
    ║   │   │ STEP 5b: WhisperResult Creation                            │   │ ║
    ║   │   │   (via stable_whisper.transcribe_any)                      │   │ ║
    ║   │   │                                                            │   │ ║
    ║   │   │   word_timestamps ──► transcribe_any() ──► WhisperResult   │   │ ║
    ║   │   │                       (with regroup=True)                  │   │ ║
    ║   │   └────────────────────────────────────────────────────────────┘   │ ║
    ║   │                         │                                          │ ║
    ║   │                         ▼                                          │ ║
    ║   │   ┌────────────────────────────────────────────────────────────┐   │ ║
    ║   │   │ STEP 5c: Sentencer (JapanesePostProcessor)                 │   │ ║
    ║   │   │   (Called inside qwen_asr.py, separate module with contract)│   │ ║
    ║   │   │                                                            │   │ ║
    ║   │   │   WhisperResult ──► JapanesePostProcessor.process()        │   │ ║
    ║   │   │                              │                             │   │ ║
    ║   │   │                              ▼                             │   │ ║
    ║   │   │   WhisperResult (regrouped into natural sentences)         │   │ ║
    ║   │   └────────────────────────────────────────────────────────────┘   │ ║
    ║   │                                                                    │ ║
    ║   │   Returns: WhisperResult with sentence-level segments              │ ║
    ║   └────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                          ║
    ║   Then: asr.cleanup(); del asr; torch.cuda.empty_cache()                 ║
    ║                                                                          ║
    ║   Phase 5 Output: List[WhisperResult] (one per scene)                    ║
    ║   Module: QwenASR (existing) + JapanesePostProcessor (existing)          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 6: MICRO-SUBS (Scene SRT Generation)                               ║
    ║          *** DONE BY PIPELINE, NOT qwen_asr.py ***                       ║
    ║                                                                          ║
    ║   For each scene's WhisperResult:                                        ║
    ║     WhisperResult ──► .to_srt_vtt(path) ──► scene_XXXX.srt              ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  WhisperResult from Phase 5                                   ║
    ║     Output: Scene SRT file (timestamps relative to scene start)          ║
    ║   Method: WhisperResult.to_srt_vtt() (stable_whisper built-in)           ║
    ║   Status: EXISTING - same pattern as StableTSASR._save_to_srt()          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 7: STITCHING                                                       ║
    ║                                                                          ║
    ║   scene_0000.srt (offset: 0.0s)    ┐                                     ║
    ║   scene_0001.srt (offset: 45.2s)   │──► SRTStitcher                      ║
    ║   scene_0002.srt (offset: 112.8s)  ┘         │                           ║
    ║                                              ▼                           ║
    ║                                      video_stitched.srt                  ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  List[(scene_srt_path, offset_seconds)]                       ║
    ║     Output: Combined SRT with adjusted timestamps                        ║
    ║   Module: SRTStitcher (existing)                                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║ PHASE 8: SANITISER (SRT Post-Processing)                                 ║
    ║                                                                          ║
    ║   video_stitched.srt ──► SRTPostProcessor                                ║
    ║                                │                                         ║
    ║                                ▼                                         ║
    ║                      video.ja.whisperjav.srt                             ║
    ║                                                                          ║
    ║   Operations:                                                            ║
    ║     - Hallucination removal (pattern-based)                              ║
    ║     - CPS filtering (remove impossibly fast subtitles)                   ║
    ║     - Duration adjustments                                               ║
    ║     - Final cleanup                                                      ║
    ║                                                                          ║
    ║   Contract:                                                              ║
    ║     Input:  Stitched SRT file                                            ║
    ║     Output: Sanitized SRT + processing statistics                        ║
    ║   Module: SRTPostProcessor (existing)                                    ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    Output: master_metadata dict with paths, stats, quality metrics
```

### 6.2 Module Boundary Summary

| Module | Location | Called By |
|--------|----------|-----------|
| AudioExtractor | `modules/audio_extraction.py` | Pipeline |
| DynamicSceneDetector | `modules/scene_detection.py` | Pipeline |
| SpeechEnhancer | `modules/speech_enhancement/` | Pipeline via `enhance_scenes()` |
| SpeechSegmenter | `modules/speech_segmentation/` | Pipeline via factory |
| QwenASR | `modules/qwen_asr.py` | Pipeline |
| JapanesePostProcessor | `modules/japanese_postprocessor.py` | Inside qwen_asr.py |
| WhisperResult.to_srt_vtt() | stable_whisper (built-in) | Pipeline calls on QwenASR result |
| SRTStitcher | `modules/srt_stitching.py` | Pipeline |
| SRTPostProcessor | `modules/srt_postprocessing.py` | Pipeline |

### 6.3 QwenPipeline Class Structure

```python
class QwenPipeline(BasePipeline):
    """
    Dedicated pipeline for Qwen3-ASR transcription.

    BasePipeline (ABC) requires:
    - __init__ must call super().__init__(output_dir, temp_dir, keep_temp_files, ...)
    - process(media_info: Dict) -> Dict          [abstract]
    - get_mode_name() -> str                     [abstract]
    - cleanup_temp_files(media_basename: str)     [inherited, can override]

    BasePipeline.__init__ also accepts:
    - save_metadata_json: bool = False
    - adaptive_classification: bool = False
    - adaptive_audio_enhancement: bool = False
    - smart_postprocessing: bool = False
    - **kwargs  (logged as debug, not an error)

    This pipeline orchestrates the 8-phase flow:
    1. Audio Extraction
    2. Scene Detection
    3. Speech Enhancement
    4. Speech Segmentation (VAD)
    5. ASR Transcription (includes internal sentencing)
    6. Micro-subs (Scene SRT generation)
    7. Stitching
    8. Sanitisation

    Follows the "redundancy over reuse" principle - no backend switching,
    no conditionals for other ASR types.
    """

    def __init__(
        self,
        # Standard pipeline params
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        progress_display = None,

        # Scene detection (Phase 2)
        # NOTE: CLI default is "none" (--qwen-scene), but pass_worker defaults to "semantic".
        # QwenPipeline should default to "none" to match CLI, unless overridden by config.
        scene_detector: str = "none",      # none, auditok, silero, semantic

        # Speech enhancement (Phase 3)
        speech_enhancer: str = "none",      # none, clearvoice, bs-roformer
        speech_enhancer_model: Optional[str] = None,

        # Speech segmentation / VAD (Phase 4)
        speech_segmenter: str = "none",     # none, silero, ten, nemo, whisper-vad

        # Qwen ASR (Phase 5)
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: Optional[str] = None,     # None = auto-detect
        timestamps: str = "word",           # word, none
        use_aligner: bool = True,
        aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        context: str = "",
        context_file: Optional[str] = None,   # Path to glossary/context text file
        attn_implementation: str = "auto",

        # Japanese post-processing (inside Phase 5, called by qwen_asr.py)
        japanese_postprocess: bool = True,
        postprocess_preset: str = "high_moan",  # default, high_moan, narrative

        # Output
        subs_language: str = "native",

        **kwargs
    ):
        """Initialize QwenPipeline with Qwen-specific configuration."""
        pass

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through the Qwen pipeline.

        Phases:
        1. Audio extraction (48kHz)
        2. Scene detection (coarse splitting)
        3. Speech enhancement (48kHz → 16kHz, exclusive VRAM block 1)
        4. Speech segmentation (VAD on clean audio)
        5. ASR transcription (exclusive VRAM block 2)
           - Includes: ASR + Aligner + WhisperResult + Sentencer
        6. Micro-subs (scene SRT generation from WhisperResult)
        7. Stitching (combine scene SRTs)
        8. Sanitisation (final cleanup)

        Returns:
            Master metadata dictionary
        """
        pass

    def _save_scene_srt(self, result: 'WhisperResult', output_path: Path) -> None:
        """Save WhisperResult to SRT file using stable_whisper's built-in method."""
        result.to_srt_vtt(str(output_path), word_level=False,
                          segment_level=True, strip=True)

    def get_mode_name(self) -> str:
        return "qwen"
```

### 6.4 QwenASR Internal Structure (Reference)

The `QwenASR.transcribe()` method internally performs multiple steps. These are documented here for clarity but do NOT need refactoring - they are already well-integrated:

```python
# Inside qwen_asr.py - QwenASR.transcribe() internal flow:

def transcribe(self, audio_path, ...) -> WhisperResult:
    """
    STEP 5a: ASR + Timestamp Alignment (integrated by qwen-asr package)
    """
    results = self.model.transcribe(
        audio=str(audio_path),
        return_time_stamps=True,  # Triggers ForcedAligner internally
    )
    # Returns: raw_text + word timestamps from ForcedAligner

    """
    STEP 5b: WhisperResult Creation (via stable_whisper)
    """
    result = stable_whisper.transcribe_any(
        inference_func=qwen_inference,  # Wraps the above
        regroup=True,  # Groups words into segments
    )
    # Returns: WhisperResult with word-level timestamps

    """
    STEP 5c: Sentencer (JapanesePostProcessor)
    """
    if self._postprocessor and detected_language == "Japanese":
        result = self._postprocessor.process(
            result,
            preset=self.postprocess_preset,
        )
    # Returns: WhisperResult with sentence-level segments

    return result  # WhisperResult ready for SRT conversion
```

**Key Point**: JapanesePostProcessor is a **separate module with its own contract**, but it is **called inside** `qwen_asr.py`. This is good modular design - no refactoring needed.

### 6.5 Parameter Count Comparison

| Parameter Category | TransformersPipeline (Current) | QwenPipeline (Proposed) |
|-------------------|-------------------------------|------------------------|
| Standard | 4 | 4 |
| Scene Detection | 1 | 1 |
| Speech Enhancement | 4 | 2 |
| Speech Segmentation | 1 | 1 |
| HF ASR | 12 | 0 |
| Qwen ASR | 12 | 10 |
| Japanese Post-proc | 2 | 2 |
| Output | 1 | 1 |
| Context / File Loading | 0 | 1 |
| **TOTAL** | **~45** | **~22** |

### 6.6 Context Biasing Architecture

#### 6.6.1 What Context Is

Qwen3-ASR's `context` parameter provides **contextual biasing** — it influences ASR
decoding toward specific terms without retraining. Technically, the context string
becomes the **system prompt** in the model's ChatML conversation template. The model
was explicitly trained to treat this as **background knowledge**, not as instructions.
From the Qwen3-ASR technical report:

> "The model learns to utilize the context tokens inside the system prompt as
> background knowledge, allowing the user to obtain customized ASR results."

> "We train the model to be an ASR-only model that does not follow natural-language
> instructions in the prompt."

This means: provide terms and reference text, not instructions like "please transcribe
carefully." The model ignores instruction-style content.

#### 6.6.2 Two Capabilities

**1. User Glossary (static context)**

Domain-specific terms the user provides before processing. Examples:
- Performer names (resolves kanji ambiguity from homophones)
- Genre-specific vocabulary
- Technical terms, product names

Already partially works via `--qwen-context "term1, term2"`. Enhancement: load from
a text file via `--qwen-context-file path/to/glossary.txt`.

**2. Cross-Scene Propagation (dynamic context)**

When processing multiple scenes, the trailing text from scene N's transcription is
appended to the context for scene N+1. This improves:
- **Speaker consistency**: Same speaker's name/style carries across scenes
- **Topic continuity**: Domain terms recognized in scene N bias scene N+1
- **Proper noun carryover**: Names established early propagate forward

Context is **stateless per call** in the qwen-asr library — each `model.transcribe()`
call receives its own independent context string. Cross-scene propagation must be
managed by the pipeline.

#### 6.6.3 Where the Logic Lives

```
QwenASR module:
  - Receives context as a string parameter per transcribe() call
  - Passes it through to model.transcribe(context=...)
  - No knowledge of glossaries, files, or propagation
  - Stateless: each call is independent

QwenPipeline orchestrator:
  - Owns context assembly
  - Loads user glossary (from string or file)
  - Manages cross-scene propagation state
  - Combines: user_glossary + previous_scene_tail → per-call context
  - Passes assembled context to QwenASR.transcribe(context=...)
```

This is a clean separation: QwenASR is a stateless transcription service; the pipeline
is the stateful orchestrator.

#### 6.6.4 QwenASR Interface Change

Currently `transcribe()` always uses `self.context` from `__init__`:

```python
# Current (per-instance only):
def transcribe(self, audio_path, ...) -> WhisperResult:
    # Always uses self.context set at construction time
    results = self.model.transcribe(audio=str(audio_path), context=self.context, ...)
```

Proposed: add optional per-call `context` override:

```python
# Proposed (per-call override, backward-compatible):
def transcribe(self, audio_path, ..., context=None) -> WhisperResult:
    effective_context = context if context is not None else self.context
    results = self.model.transcribe(audio=str(audio_path), context=effective_context, ...)
```

Backward-compatible: callers that don't pass `context` get existing behavior.

#### 6.6.5 Pipeline Implementation Pattern

```python
# In QwenPipeline.process(), inside the scene loop:

user_context = self.context  # From --qwen-context or --qwen-context-file
previous_tail = ""

for scene_idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scenes):

    # Assemble context: user glossary + previous scene tail
    if previous_tail and user_context:
        context = f"{user_context}\n{previous_tail}"
    elif previous_tail:
        context = previous_tail
    else:
        context = user_context

    # Transcribe with per-scene context
    result = asr.transcribe(scene_path, context=context)

    # Extract tail for next scene (last 2 segments' text)
    if result.segments:
        previous_tail = " ".join(seg.text for seg in result.segments[-2:])
    # If 0 segments (silence/music), previous_tail carries forward from earlier scene
```

This is ~10 lines of pipeline logic. No additional modules or classes needed.

#### 6.6.6 Context File Loading

```python
# In QwenPipeline.__init__:
if context_file:
    self.context = Path(context_file).read_text(encoding='utf-8').strip()
elif context:
    self.context = context
else:
    self.context = ""
```

Users maintain a plain text file with performer names, glossary terms, or reference
paragraphs. One file per project/genre, reusable across multiple videos.

#### 6.6.7 Token Budget

| Content | Typical Size | Notes |
|---------|-------------|-------|
| User glossary | 50-200 terms (~500 tokens) | Performer names, domain terms |
| Previous scene tail | 1-2 sentences (~50 tokens) | Last 2 segments from WhisperResult |
| **Total context** | **~550 tokens** | Well within 10,000 token API limit |

The model's total position embedding budget is 128K tokens. Audio at 12.5 tokens/sec
for a 20-minute scene ≈ 15K audio tokens. Context + audio + output fits comfortably.
No elaborate truncation logic needed for practical use cases. If a user provides an
absurdly long context file, log a warning when it exceeds 5000 tokens (rough estimate:
>3500 words for Japanese text).

#### 6.6.8 Important Constraints

- Context biases **ASR transcription only**. It has no effect on ForcedAligner —
  alignment operates on the text output, not on the context.
- The model is robust to "noise" in context — irrelevant terms rarely hurt accuracy.
  But extremely long context (>5000 tokens) can cause hallucination of context terms
  that aren't in the audio.
- Context is per-call in non-streaming mode. In streaming mode (vLLM backend, not used
  by WhisperJAV), context is fixed for the entire stream.
- Cross-scene propagation is only meaningful when scene detection is active. When scene
  detection is "none" (single scene), the entire audio is one call and there is nothing
  to propagate.

#### 6.6.9 Why Not a ContextManager Class

The agent input proposes a `ContextManager` with Strategy Pattern (3 extraction
strategy classes), weighted glossary tracking, scene type hints, and segment history.
This is over-engineering for several specific reasons:

1. **Context assembly is string concatenation.** The pipeline combines user glossary +
   previous tail. This is ~8 lines of code, not an architecture that needs pluggable
   strategies.
2. **"Weighted glossary" tracking** assumes terms appearing more frequently in past
   transcriptions should be prioritized. This is an unverified heuristic that adds
   stateful complexity for no proven benefit. The model already handles relevance
   filtering internally.
3. **Scene type hints** (adding "状況: 室内シーン" to context) assume the model uses
   scene descriptions to improve accuracy. The documentation explicitly states the
   model ignores instruction-like content — it's trained for background knowledge bias,
   not situational awareness.
4. **Three extraction strategies** (SentenceBasedExtractor, TokenBasedExtractor,
   HybridExtractor) for a task that is: "take the text of the last 2 segments." This
   does not need an ABC hierarchy with a factory and a strategy registry.

If context assembly becomes genuinely complex based on real-world usage feedback, a
helper function can be extracted at that time. Do not pre-build abstractions for
hypothetical requirements.

---

## 7. Module Contracts

> **Note**: Contracts are organized by pipeline phase order (1-8).
> All modules listed here are EXISTING - no new modules need to be created.

### 7.1 AudioExtractor (Phase 1)

```python
Contract:
    __init__(sample_rate: int = 16000, channels: str = "mono",
             audio_codec: str = "pcm_s16le", ffmpeg_path: Optional[str] = None)

    extract(input_file: Union[str, Path], output_path: Union[str, Path])
        -> Tuple[Path, float]

    Input:  Any media file (video or audio)
    Output: (wav_path, duration_seconds)

    Guarantees:
    - Output WAV at configured sample rate
    - Pipeline must pass sample_rate=48000 for v1.7.4+ enhancement contract
    - Mono channel
    - PCM format

    Location: whisperjav/modules/audio_extraction.py
    Status: EXISTING - no changes needed
```

### 7.2 DynamicSceneDetector (Phase 2)

```python
Contract:
    __init__(method: str = "auditok", max_duration: float = 29.0,
             min_duration: float = 0.3, ...)

    detect_scenes(audio_path: Path, output_dir: Path, media_basename: str)
        -> List[Tuple[Path, float, float, float]]

    get_detection_metadata() -> Dict
        Returns: {scenes_detected, coarse_boundaries, vad_segments, vad_method, vad_params}

    Input:  Full-length WAV file
    Output: List of (scene_path, start_sec, end_sec, duration_sec)

    Config:
    - method: "auditok" | "silero" | "semantic"
      (Note: Pipeline handles "none" by skipping scene detection entirely,
       treating full audio as a single scene)

    Guarantees:
    - Scenes are non-overlapping
    - Scenes cover full audio duration
    - Each scene is a valid WAV file
    - Scene metadata available via get_detection_metadata() after detection

    Location: whisperjav/modules/scene_detection.py (DynamicSceneDetector class)
    Status: EXISTING - no changes needed
```

### 7.3 SpeechEnhancer (Phase 3)

```python
Contract:
    enhance(audio: np.ndarray | Path, sample_rate: int) -> EnhancementResult
    cleanup() -> None
    get_preferred_sample_rate() -> int
    get_output_sample_rate() -> int

    Input:  Audio @ 48kHz (from scene detection)
    Output: EnhancementResult with enhanced audio @ 16kHz

    Config:
    - backend: "none" | "clearvoice" | "bs-roformer" | "ffmpeg-dsp"
    - model: Optional model variant

    Guarantees:
    - Output always at 16kHz (contract with downstream VAD/ASR)
    - cleanup() releases all GPU memory
    - "none" backend still resamples 48kHz → 16kHz (passthrough with resample)
    - On failure: returns original audio with success=False (graceful degradation)

    Location: whisperjav/modules/speech_enhancement/
    Factory: SpeechEnhancerFactory.create(backend, config)
    Helper: enhance_scenes() in pipeline_helper.py
    Status: EXISTING - no changes needed
```

### 7.4 SpeechSegmenter (Phase 4)

```python
Contract:
    segment(audio: Union[np.ndarray, Path, str], sample_rate: int = 16000, **kwargs)
        -> SegmentationResult
    cleanup() -> None

    Input:  Enhanced audio @ 16kHz (from Phase 3) — file path or numpy array
    Output: SegmentationResult containing:
            - segments: List[SpeechSegment] with start_sec, end_sec, confidence
            - groups: List[List[SpeechSegment]] for chunk-based ASR
            - speech_coverage_ratio: float (computed property, not stored field)

    Config:
    - backend: "none" | "silero" | "silero-v4.0" | "ten" | "nemo" | "whisper-vad"

    Backend Behaviors:
    - "none": Returns full audio as single speech region (passthrough)
    - "silero": Silero VAD v4.0, GPU-accelerated, returns speech regions
    - "ten": TEN VAD (requires ten_vad package)
    - "nemo": NeMo speech segmentation (requires nemo_toolkit)
    - "whisper-vad": faster-whisper built-in VAD

    Usage:
    - Runs AFTER enhancement (VAD on clean audio = better accuracy)
    - Runs BEFORE ASR (identifies regions to transcribe)
    - When backend="none": ASR processes full scene (current default)
    - When backend="silero" etc: ASR can skip non-speech regions

    Location: whisperjav/modules/speech_segmentation/
    Factory: SpeechSegmenterFactory.create(backend, config)
    Status: EXISTING - no changes needed
```

### 7.5 QwenASR (Phase 5)

```python
Contract:
    __init__(model_id: str, device: str = "auto", dtype: str = "auto",
             batch_size: int = 1, max_new_tokens: int = 4096,
             language: Optional[str] = None, task: str = "transcribe",
             timestamps: str = "word", use_aligner: bool = True,
             aligner_id: str = "...", context: str = "",
             attn_implementation: str = "auto",
             japanese_postprocess: bool = True,
             postprocess_preset: str = "high_moan")

    transcribe(audio_path: Union[str, Path],
               progress_callback: Optional[Callable[[float, str], None]] = None,
               artifacts_dir: Optional[Union[str, Path]] = None,
               context: Optional[str] = None)
        -> stable_whisper.WhisperResult

    cleanup() -> None   # Alias for unload_model()

    Input:  Audio file path (16kHz recommended; qwen-asr library handles internally)
    Output: WhisperResult with word-level timestamps (if aligner enabled)

    Config:
    - model_id: HuggingFace model ID
    - language: Language code or None for auto-detect (returns detected language)
    - timestamps: "word" (enables aligner) | "none" (disables aligner)
    - max_new_tokens: Token limit for generation
    - batch_size: Number of chunks to process in parallel
    - attn_implementation: Attention backend (sdpa, flash_attention_2, eager, auto)
    - japanese_postprocess: Enable/disable JapanesePostProcessor (Step 5c)
    - postprocess_preset: Preset for JapanesePostProcessor

    Context Parameter (NEW - see Section 6.6):
    - context: Optional per-call context string for contextual biasing
    - When None: uses self.context from __init__ (backward-compatible)
    - When provided: overrides self.context for this specific call
    - Enables cross-scene context propagation (pipeline passes different
      context per scene without reconstructing the QwenASR instance)
    - The underlying qwen-asr library already supports per-call context;
      this change exposes it through WhisperJAV's QwenASR wrapper

    Guarantees:
    - Word timestamps when timestamps="word" (use_aligner=True)
    - Handles audio > 180s internally (splits into chunks, emits warning)
    - cleanup() releases all GPU memory (calls unload_model())
    - Returns empty WhisperResult (0 segments) for silent/failed audio
    - OOM recovery: Automatically retries with reduced batch_size
    - progress_callback called with (progress_float, status_message) during transcription
    - Context biases ASR only; has no effect on ForcedAligner

    Note on Sample Rate:
    - QwenASR.transcribe() accepts an audio file path
    - The qwen-asr library handles sample rate conversion internally
    - Pipeline convention: provide 16kHz audio (output of Phase 3)
    - Phase 3 (SpeechEnhancer) guarantees 16kHz output even with backend="none"
```

### 7.6 JapanesePostProcessor (Inside Phase 5)

```python
Contract:
    process(result: WhisperResult, preset: str, language: Optional[str],
            skip_if_not_japanese: bool = True)
        -> WhisperResult

    Input:  WhisperResult with word timestamps
    Output: WhisperResult with regrouped sentence segments (modified in-place)

    Config:
    - preset: "default" | "high_moan" | "narrative"
    - skip_if_not_japanese: When True (default in QwenASR), only applies
      post-processing if detected language is Japanese. Important for
      auto-detect mode where non-Japanese audio should pass through unchanged.

    Guarantees:
    - Preserves all original text content
    - Produces natural sentence boundaries
    - Skips processing for non-Japanese content when skip_if_not_japanese=True

    Location: whisperjav/modules/japanese_postprocessor.py
    Called From: qwen_asr.py (lines 860-870)
    Status: EXISTING - behavioral change required (see below)

    ┌────────────────────────────────────────────────────────────────────────┐
    │ ⚠️  BEHAVIORAL CHANGE REQUIRED (not current behavior)                  │
    │                                                                        │
    │ NOTE: "Step 6" below refers to JapanesePostProcessor's INTERNAL step   │
    │ numbering, NOT pipeline Phase 6 (which is Scene SRT Generation).       │
    │                                                                        │
    │ Current JPP Internal Step 6 Behavior (WRONG for JAV):                  │
    │   - Merges segments shorter than min_segment_duration                  │
    │   - Merges segments with fewer than min_segment_chars                  │
    │   - This DESTROYS meaningful short utterances (moans, gasps)           │
    │                                                                        │
    │ Required JPP Internal Step 6 Behavior (to be implemented):             │
    │   - PRESERVE tiny fragments for JAV content                            │
    │   - Only merge truly empty or whitespace-only segments                 │
    │   - Consider adding jav_preserve_tiny preset parameter                 │
    │                                                                        │
    │ Implementation Options:                                                │
    │   Option A: Add preserve_tiny_fragments: bool to PresetParameters      │
    │   Option B: Create new preset "jav_moan" with tiny fragment protection │
    │   Option C: Modify high_moan preset to skip internal step 6 entirely   │
    │                                                                        │
    │ See: japanese_postprocessor.py lines 542-612 (_merge_tiny_fragments)   │
    └────────────────────────────────────────────────────────────────────────┘
```

### 7.7 Scene SRT Generation (Phase 6 - Micro-subs)

```python
Contract:
    WhisperResult.to_srt_vtt(filepath, segment_level=True, word_level=False, strip=True)

    Input:  WhisperResult from Phase 5 (QwenASR.transcribe() return value)
    Output: SRT file on disk (timestamps relative to scene start)

    Guarantees:
    - Valid SRT format with sequential numbering
    - Segment-level timestamps (not word-level)
    - Strips leading/trailing whitespace from segment text
    - Minimum duration enforcement (min_dur=0.02s default)

    Existing Patterns (3 approaches used by different pipelines):

    Pattern A - WhisperResult.to_srt_vtt() [stable_whisper built-in]:
        Used by: StableTSASR._save_to_srt() (stable_ts_asr.py:628-638)
        Input:   WhisperResult (returned by QwenASR, StableTSASR)
        Method:  result.to_srt_vtt(str(srt_path), word_level=False,
                                    segment_level=True, strip=True)

    Pattern B - ASR.transcribe_to_srt() [ASR handles SRT writing]:
        Used by: FidelityPipeline, BalancedPipeline
        Input:   Audio path (ASR transcribes + writes SRT internally)
        Method:  asr.transcribe_to_srt(scene_path, scene_srt_path)

    Pattern C - Pipeline._segments_to_srt() [pipeline helper]:
        Used by: TransformersPipeline
        Input:   List[Dict] with 'text', 'start', 'end' keys
        Method:  srt_content = self._segments_to_srt(segments)

    Recommended for QwenPipeline: Pattern A
        - QwenASR.transcribe() already returns WhisperResult
        - to_srt_vtt() is a method ON WhisperResult (stable_whisper built-in)
        - Same pattern already proven in stable_ts_asr.py
        - No new code needed — just call the existing method

    Location: stable_whisper.result.WhisperResult.to_srt_vtt()
              (bound from stable_whisper.text_output.result_to_srt_vtt)
    Status: EXISTING - no new code needed

    Usage Pattern (in QwenPipeline.process):
        for scene_idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scenes):
            result = asr.transcribe(scene_path)  # Phase 5 → WhisperResult
            scene_srt_path = scene_srts_dir / f"{basename}_scene_{scene_idx:04d}.srt"
            result.to_srt_vtt(str(scene_srt_path),
                              word_level=False,
                              segment_level=True,
                              strip=True)
            scene_srts.append((scene_srt_path, start_sec))  # For stitching

    Note:
    - Scene SRTs have timestamps relative to scene start (0.0)
    - Stitcher (Phase 7) adds scene offset to produce absolute timestamps
```

### 7.8 SRTStitcher (Phase 7)

```python
Contract:
    stitch(scene_srt_info: List[Tuple[Path, float]], output_path: Path) -> int

    Input:  List of (scene_srt_path, offset_seconds)
    Output: Combined SRT file path, returns subtitle count

    Guarantees:
    - Timestamps adjusted by scene offsets (relative → absolute)
    - Subtitles re-indexed sequentially (1, 2, 3, ...)
    - Maintains scene order
    - Empty scenes (0 subtitles) contribute nothing to output

    Location: whisperjav/modules/srt_stitching.py
    Status: EXISTING - no changes needed

    Usage Pattern:
        stitcher = SRTStitcher()
        scene_srt_info = [
            (temp_dir / "video_scene_0000.srt", 0.0),      # Scene 1 starts at 0.0s
            (temp_dir / "video_scene_0001.srt", 45.2),    # Scene 2 starts at 45.2s
            (temp_dir / "video_scene_0002.srt", 112.8),   # Scene 3 starts at 112.8s
        ]
        subtitle_count = stitcher.stitch(scene_srt_info, output_path)
```

### 7.9 SRTPostProcessor (Phase 8 - Sanitiser)

```python
Contract:
    __init__(language: str, **post_proc_opts)

    process(srt_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None)
        -> Tuple[Path, Dict]

    Input:  Stitched SRT file (from Phase 7)
    Output: (sanitized_srt_path, statistics_dict)

    Operations:
    - Hallucination removal (pattern matching for known AI hallucinations)
    - CPS filtering (remove subtitles with impossibly fast character-per-second rate)
    - Duration adjustments (extend/clamp subtitle display times)
    - Empty subtitle removal (subtitles with no text content)
    - Overlap resolution (fix overlapping timestamps)

    Statistics Dict Contains (actual keys from srt_postprocessing.py):
    - total_subtitles: int          # Total subtitles in input file
    - removed_hallucinations: int   # Hallucinations removed
    - removed_repetitions: int      # Repetitions removed
    - duration_adjustments: int     # Timing adjustments applied
    - empty_removed: int            # Empty subtitles removed
    - cps_filtered: int             # CPS-based filtering (English only, optional key)

    Location: whisperjav/modules/srt_postprocessing.py
    Status: EXISTING - no changes needed

    Usage Pattern:
        processor = SRTPostProcessor(language=lang_code)
        final_path, stats = processor.process(
            stitched_srt_path,
            output_dir / f"{basename}.{language}.whisperjav.srt"
        )
        logger.info(f"Sanitization: {stats.get('total_subtitles', 0)} subtitles, "
                     f"{stats.get('removed_hallucinations', 0)} hallucinations removed")
```

---

## 8. Configuration Architecture

### 8.1 Recommendation: Use V4 YAML

The V4 YAML configuration system already exists for Qwen:

```
config/v4/ecosystems/qwen/
├── ecosystem.yaml          # Provider: QwenASR
└── models/
    └── qwen3-asr-1.7b.yaml # Default parameters
```

**Benefits:**
- YAML is source of truth (no code changes to add models)
- Patchable without redistribution
- GUI auto-generation from hints
- Sensitivity presets (conservative, balanced, aggressive)

### 8.2 V4 Integration Strategy

QwenPipeline should:

1. **Accept both CLI args and V4 config** for flexibility
2. **CLI args override V4 defaults** (standard precedence)
3. **V4 provides defaults and GUI hints**

```python
# In QwenPipeline.__init__:
from whisperjav.config.v4 import ConfigManager

# V4 YAML provides model-level defaults; CLI args always take precedence.
# Since __init__ parameters have defaults, V4 is only consulted when
# explicitly requested (e.g., loading a named model config):
v4_config = ConfigManager().get_model_config("qwen3-asr-1.7b")
# Pipeline uses CLI args directly; V4 is the source of truth for
# model discovery and GUI hint generation, not runtime overrides.
```

### 8.3 V4 YAML Updates (Deferred to Future Work)

> **Note**: The V4 YAML schema extensions described below are **deferred** to avoid scope creep.
> Phase 1 implementation should use **hardcoded defaults** or **direct CLI argument passthrough**.
> See [Section 16: Future Work](#16-future-work) for the full schema extension specification.

For Phase 1, the pipeline will:
1. Accept CLI arguments directly (no V4 config lookup for new parameters)
2. Use existing V4 YAML for model-level parameters (model_id, dtype, etc.)
3. Hardcode defaults for new parameters (scene=none, enhancer=none, etc.)

---

## 9. CLI Arguments Strategy

### 9.1 Existing CLI Arguments (Reuse)

The existing `--qwen-*` arguments are well-designed and should be **reused unchanged**,
with one new argument added for context file loading.

**Total: 17 arguments** (16 existing + 1 new):

| # | Argument | Type | Default | Purpose |
|---|----------|------|---------|---------|
| 1 | `--qwen-model-id` | str | Qwen/Qwen3-ASR-1.7B | Model identifier |
| 2 | `--qwen-device` | str | auto | Compute device |
| 3 | `--qwen-dtype` | str | auto | Data type |
| 4 | `--qwen-batch-size` | int | 1 | Inference batch size |
| 5 | `--qwen-max-tokens` | int | 4096 | Max new tokens |
| 6 | `--qwen-language` | str | None | Language (None=auto) |
| 7 | `--qwen-timestamps` | str | word | Timestamp mode (word=use aligner, none=no aligner) |
| 8 | `--qwen-aligner` | str | Qwen/Qwen3-ForcedAligner-0.6B | Aligner model |
| 9 | `--qwen-scene` | str | none | Scene detection method |
| 10 | `--qwen-context` | str | "" | Context string for contextual biasing (see Section 6.6) |
| 11 | `--qwen-context-file` | str | None | **NEW**: Path to text file with glossary/context (see Section 6.6) |
| 12 | `--qwen-attn` | str | auto | Attention implementation |
| 13 | `--qwen-enhancer` | str | none | Speech enhancement backend |
| 14 | `--qwen-enhancer-model` | str | None | Enhancer model variant |
| 15 | `--qwen-segmenter` | str | none | Speech segmentation / VAD backend (see Section 6.1 Phase 4) |
| 16 | `--qwen-japanese-postprocess` | bool | True | Enable Japanese post-processing |
| 17 | `--qwen-postprocess-preset` | str | high_moan | Japanese post-proc preset |

> **`--qwen-context` vs `--qwen-context-file`**: Both provide static context for
> contextual biasing. `--qwen-context` is for short inline strings (e.g., a few performer
> names). `--qwen-context-file` loads from a text file (e.g., a maintained glossary).
> If both are provided, they are concatenated. See Section 6.6 for full details.

> **Note on `use_aligner`**: There is no `--qwen-use-aligner` CLI argument. The aligner is
> enabled/disabled implicitly via `--qwen-timestamps`: `word` enables ForcedAligner, `none` disables it.
> QwenASR.__init__ derives `use_aligner = (timestamps == "word")` internally.

### 9.2 Routing Changes in main.py

```python
# Current (in main.py):
elif args.mode == "qwen":
    # Creates TransformersPipeline with asr_backend="qwen"
    pipeline = TransformersPipeline(asr_backend="qwen", ...)

# Proposed:
elif args.mode == "qwen":
    # Creates dedicated QwenPipeline
    pipeline = QwenPipeline(
        model_id=args.qwen_model_id,
        device=args.qwen_device,
        # ... direct mapping, no asr_backend switch
    )
```

### 9.3 Ensemble Mode Routing

```python
# In pass_worker.py:

# Current:
PIPELINE_CLASSES = {
    "qwen": TransformersPipeline,  # with asr_backend="qwen"
}

# Proposed:
PIPELINE_CLASSES = {
    "qwen": QwenPipeline,  # dedicated pipeline
}
```

---

## 10. Impact on Existing Code

### 10.1 New File

| File | Lines (Est.) | Description |
|------|--------------|-------------|
| `whisperjav/pipelines/qwen_pipeline.py` | ~500-600 | Dedicated Qwen pipeline |

**Estimate Basis**: Comparable dedicated pipelines:
- `fidelity_pipeline.py`: ~280 lines (7 phases, but simpler error handling)
- `transformers_pipeline.py`: ~900 lines (dual-backend, over-complicated)
- Expected: ~500-600 lines with proper error handling, progress reporting, and VRAM management

### 10.2 Modified Files

| File | Change Type | Description |
|------|-------------|-------------|
| `pass_worker.py` | Routing | Change `"qwen": QwenPipeline` |
| `main.py` | Routing + CLI | Create `QwenPipeline` for `--mode qwen`; add `--qwen-context-file` arg |
| `japanese_postprocessor.py` | Bug fix | Internal step 6: preserve tiny fragments |
| `qwen_asr.py` | Interface | Add optional `context` parameter to `transcribe()` (see Section 6.6.4) |

### 10.3 Cleanup (After Migration)

| File | Change Type | Description |
|------|-------------|-------------|
| `transformers_pipeline.py` | Remove code | Remove `asr_backend`, `qwen_*` params |

### 10.4 No Changes Required

| File | Reason |
|------|--------|
| CLI arguments | 16 existing args reused unchanged; 1 new arg added |
| V4 YAML | Already exists, minor additions only |
| GUI | Context text area and file picker to be added (standard widget addition) |

---

## 11. Implementation Plan

### Phase 1: Create QwenPipeline (Primary Task)

**Scope**: New file, no breaking changes

1. Create `whisperjav/pipelines/qwen_pipeline.py`
2. Implement `__init__` with Qwen-only parameters (~22 parameters)
3. Implement `process()` following the 8-phase flow:
   - Phase 1: Audio extraction (48kHz)
   - Phase 2: Scene detection (optional, default: none)
   - Phase 3: Speech enhancement (optional, VRAM Block 1)
   - Phase 4: Speech segmentation / VAD (optional)
   - Phase 5: ASR transcription (VRAM Block 2, includes sentencer)
     - With context biasing: user glossary + cross-scene propagation (Section 6.6)
   - Phase 6: Scene SRT generation (micro-subs)
   - Phase 7: SRT stitching
   - Phase 8: Sanitisation
4. Add per-call `context` parameter to `QwenASR.transcribe()` (Section 6.6.4)
5. Add `--qwen-context-file` CLI argument to `main.py`
6. Use existing modules (no new logic beyond context assembly)
7. Add unit tests

**Deliverables**:
- `qwen_pipeline.py` (~500-600 lines)
- Modified `qwen_asr.py` (per-call context override)
- Modified `main.py` (new CLI argument)
- `tests/test_qwen_pipeline.py`

**Note**: Phase 1 can be developed and tested in isolation before Phase 2 routing changes.

### Phase 2: Update Routing

**Scope**: Minimal changes to existing files
**Depends On**: Phase 1 must be complete and unit-tested

1. Update `pass_worker.py`: `"qwen": QwenPipeline`
2. Update `main.py`: Route `--mode qwen` to `QwenPipeline`
3. Verify CLI → Pipeline parameter flow
4. Verify ensemble mode → QwenPipeline flow

**Integration Testing Required**:
- Test `whisperjav video.mp4 --mode qwen` end-to-end
- Test ensemble mode: `--pass1-pipeline qwen --pass2-pipeline balanced`
- Compare output with old implementation (behavioral equivalence)

**Deliverables**:
- Modified `pass_worker.py`
- Modified `main.py`
- Integration test results documenting behavioral equivalence

### Phase 3: Fix JapanesePostProcessor

**Scope**: Behavioral change, separate from pipeline work
**Can Run In Parallel**: This phase is independent of Phases 1-2

**Problem**: Current `_merge_tiny_fragments()` (lines 542-612) merges short segments, destroying meaningful short utterances in JAV content.

**Implementation Options**:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Add `preserve_tiny_fragments: bool` to PresetParameters | Flexible, per-preset control | More parameters |
| B | Create new preset `"jav_preserve"` | Clean separation | Another preset to maintain |
| C | Modify `high_moan` preset to skip JPP internal step 6 | Simple | Reduces `high_moan` flexibility |

**Recommended**: Option A - Add parameter to PresetParameters with `preserve_tiny_fragments=True` default for `high_moan` preset.

**Tasks**:
1. Add `preserve_tiny_fragments: bool = False` to `PresetParameters` dataclass
2. Modify `_merge_tiny_fragments()` to check this flag and skip if True
3. Set `preserve_tiny_fragments=True` in `high_moan` preset defaults
4. Review Phase 1 aizuchi list: Consider if "あっ" should be preserved (could be moan)
5. Add tests for tiny fragment preservation behavior

**Deliverables**:
- Modified `japanese_postprocessor.py`
- `tests/test_japanese_postprocessor.py` updates

### Phase 4: Cleanup TransformersPipeline

**Scope**: Remove Qwen code, simplify

1. Remove `asr_backend` parameter
2. Remove all `qwen_*` parameters
3. Remove Qwen-specific conditionals
4. Update docstrings

**Deliverables**:
- Simplified `transformers_pipeline.py` (HF-only)

### Phase 5: Documentation

**Scope**: Update docs to reflect new architecture

1. Update CLAUDE.md pipeline descriptions
2. Update CLI help text
3. Archive this ADR with decision
4. Update QWEN_ASR_REMEDIATION_PLAN with completion status

---

## 12. Risk Assessment

### 12.1 Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Regression in CLI mode | High | Low | Comprehensive testing before/after |
| Regression in ensemble mode | High | Medium | Test all pipeline combinations with comparison tests |
| V4 config integration issues | Medium | Low | V4 already works, defer extensions to Future Work |
| Orchestration bugs | Medium | Medium | Copy proven patterns from FidelityPipeline |
| VRAM management issues | High | Low | Use identical JIT pattern |
| **GUI integration regression** | High | Medium | GUI builds CLI args via `_build_twopass_args()` - must test GUI Qwen mode |
| **Ensemble parameter preparation** | High | Medium | `prepare_qwen_params()` in pass_worker.py has significant logic for building Qwen params |
| **Backward compatibility** | Medium | Low | Users with saved presets or scripts using `--mode qwen` must get identical behavior |

### 12.2 New Risks Identified in Review

#### 12.2.1 GUI Integration Risk

The GUI (`webview_gui/api.py`) builds CLI arguments via `_build_twopass_args()`. If the routing changes in `pass_worker.py` require different argument mapping, the GUI will break silently.

**Mitigation**: Add GUI regression test that:
1. Launches GUI with Qwen mode selected
2. Verifies generated CLI arguments are correct
3. Runs full transcription and compares output

#### 12.2.2 Ensemble Mode Parameter Preparation Risk

The ensemble mode (`pass_worker.py:803-862`) has significant logic for building Qwen pipelines via `prepare_qwen_params()`. Changing `"qwen": QwenPipeline` requires updating all parameter preparation.

**Mitigation**:
1. Create integration test for ensemble mode with `--pass1-pipeline qwen --pass2-pipeline balanced`
2. Run comparison test: same file processed with old vs new implementation
3. Diff outputs for behavioral equivalence

### 12.3 Rollback Strategy

Each phase is independent and reversible:

| Phase | Rollback Method | Risk |
|-------|-----------------|------|
| Phase 1 | Delete `qwen_pipeline.py`, remove from `__init__.py` | None (new file) |
| Phase 2 | Revert routing changes in pass_worker.py and main.py | Low |
| Phase 3 | Revert JapanesePostProcessor changes | Low |
| Phase 4 | Can be deferred indefinitely (cleanup only) | None |
| Phase 5 | Documentation can be updated independently | None |

---

## 13. Acceptance Criteria

### 13.1 Functional Requirements

- [ ] `whisperjav video.mp4 --mode qwen` works identically to current behavior
- [ ] Ensemble mode with `--pass1-pipeline qwen` works correctly
- [ ] All 16 `--qwen-*` CLI arguments function correctly
- [ ] Scene detection (auditok, silero, semantic) works with Qwen
- [ ] Speech enhancement (none, clearvoice, bs-roformer) works with Qwen
- [ ] Speech segmenter (none, silero, ten) works with Qwen
- [ ] Japanese post-processing produces correct output
- [ ] Tiny fragments (moans) are preserved, not merged

### 13.2 Regression Tests (NEW)

- [ ] **GUI Qwen mode**: GUI-initiated Qwen transcription produces identical output to CLI
- [ ] **Ensemble mode combination**: `--pass1-pipeline qwen --pass2-pipeline balanced` works correctly
- [ ] **Behavioral equivalence**: Process test file with old and new implementation, diff outputs match
- [ ] **Empty result handling**: Pipeline handles 0-segment scenes gracefully (no crash)
- [ ] **OOM recovery**: Pipeline handles GPU OOM by reducing batch size (existing behavior preserved)

### 13.3 Non-Functional Requirements

- [ ] QwenPipeline has ~21 parameters (not 45+)
- [ ] No `if asr_backend == "qwen"` conditionals in TransformersPipeline
- [ ] Logs show correct model name (Qwen/Qwen3-ASR-1.7B)
- [ ] VRAM usage matches current behavior (JIT load/unload)
- [ ] Processing time within 10% of current behavior

### 13.4 Documentation Requirements

- [ ] CLAUDE.md updated with QwenPipeline description in "Processing Pipelines" section
- [ ] CLI help text accurate for all 16 `--qwen-*` arguments
- [ ] This ADR archived with final decision
- [ ] ADR-003 sections marked as superseded by this ADR

---

## 14. Error Handling Strategy

### 14.1 Error Levels

The pipeline distinguishes between **scene-level errors** and **pipeline-level errors**:

| Error Level | Behavior | Example |
|-------------|----------|---------|
| **Scene-level** | Log warning, continue with other scenes | OOM on scene 3 of 10 |
| **Pipeline-level** | Fail entire job, return error metadata | Model load failure |

### 14.2 Scene-Level Error Handling

When processing multiple scenes, a failure in one scene should NOT fail the entire job:

```python
# Pseudocode for scene processing
for scene in scenes:
    try:
        result = process_scene(scene)
        results.append(result)
    except SceneProcessingError as e:
        logger.warning(f"Scene {scene.id} failed: {e}, skipping")
        failed_scenes.append(scene.id)
        continue

# Return partial results with metadata about failures
return {
    "srt_path": stitch(results),
    "failed_scenes": failed_scenes,
    "total_scenes": len(scenes),
    "success_rate": len(results) / len(scenes)
}
```

### 14.3 Empty Result Handling

When QwenASR returns 0 segments for a scene:

1. **Do NOT treat as error** - Scene may contain only music/silence
2. **Log as info** - `"Scene {id} produced 0 segments (may be non-speech audio)"`
3. **Include in stitch** - Empty scene contributes nothing to final SRT
4. **Track in metadata** - `"empty_scenes": [scene_ids]`

### 14.4 OOM Recovery

The existing `qwen_asr.py` OOM recovery (lines 874-909) should be preserved:

1. Catch `torch.cuda.OutOfMemoryError`
2. Log warning with current batch size
3. Reduce batch size by half
4. Retry with reduced batch size
5. If batch_size=1 still fails, propagate as scene-level error

### 14.5 Language Detection Propagation

When `qwen_language=None` (auto-detect):

1. QwenASR detects language during transcription
2. Detected language returned in result metadata
3. Pipeline uses detected language for SRT filename: `{basename}.{detected_lang}.whisperjav.srt`
4. If detection fails, fallback to `"ja"` (primary use case)

---

## 15. Logging Standards

### 15.1 Log Format Requirements

All pipeline logs should follow this format:

```
[QwenPipeline PID {pid}] {phase}: {message}
```

Examples:
```
[QwenPipeline PID 12345] Phase 1: Extracting audio from video.mkv
[QwenPipeline PID 12345] Phase 2: Detected 5 scenes (method=semantic)
[QwenPipeline PID 12345] Phase 5: Transcribing scene 3/5 (model=Qwen/Qwen3-ASR-1.7B)
[QwenPipeline PID 12345] Complete: Generated video.ja.whisperjav.srt (47 subtitles)
```

### 15.2 Model Name Logging (Bug Fix)

The current bug shows wrong model name. The fix:

```python
# WRONG (current TransformersPipeline):
logger.debug("model=%s", self.hf_config.get("model_id"))  # Always HF model

# CORRECT (new QwenPipeline):
logger.debug("model=%s", self.model_id)  # Direct attribute, always correct
```

### 15.3 Progress Logging

For long-running phases, log progress at reasonable intervals:

| Phase | Progress Logging |
|-------|------------------|
| Phase 1 (Extract) | Start/complete only |
| Phase 2 (Scenes) | Number of scenes detected |
| Phase 3 (Enhance) | Per-scene progress (VRAM Block 1) |
| Phase 4 (VAD/Segment) | Per-scene progress (if enabled) |
| Phase 5 (ASR) | Per-scene progress with ETA (VRAM Block 2) |
| Phase 6 (Micro-subs) | Per-scene SRT generation |
| Phase 7 (Stitch) | Start/complete only |
| Phase 8 (Sanitise) | Start/complete with stats |

---

## 16. Future Work

### 16.1 V4 YAML Schema Extensions (Deferred)

The following V4 YAML extensions were identified but deferred to avoid scope creep:

```yaml
# Proposed additions to qwen3-asr-1.7b.yaml (NOT for Phase 1)
spec:
  # ... existing ...

  # Scene detection configuration
  scene.method: semantic           # none, auditok, silero, semantic
  scene.min_duration: 10.0         # Minimum scene duration (seconds)
  scene.max_duration: 300.0        # Maximum scene duration (seconds)

  # Speech enhancement configuration
  enhancer.backend: none           # none, clearvoice, bs-roformer
  enhancer.model: null             # Model variant (backend-specific)

  # Speech segmentation / VAD configuration
  segmenter.backend: none          # none, silero, ten, nemo, whisper-vad
  segmenter.threshold: 0.5         # VAD threshold (backend-specific)

  # Japanese post-processing configuration
  postprocess.enabled: true        # Enable/disable post-processing
  postprocess.preset: high_moan    # Preset name
  postprocess.preserve_tiny: true  # Preserve tiny fragments

gui:
  scene.method:
    widget: dropdown
    label: "Scene Detection"
    options:
      - {value: "none", label: "Disabled"}
      - {value: "auditok", label: "Auditok (silence-based)"}
      - {value: "silero", label: "Silero VAD"}
      - {value: "semantic", label: "Semantic (recommended)"}

  enhancer.backend:
    widget: dropdown
    label: "Speech Enhancement"
    options:
      - {value: "none", label: "Disabled"}
      - {value: "clearvoice", label: "ClearVoice (denoising)"}
      - {value: "bs-roformer", label: "BS-RoFormer (vocal isolation)"}

  segmenter.backend:
    widget: dropdown
    label: "Speech Segmentation (VAD)"
    options:
      - {value: "none", label: "Disabled"}
      - {value: "silero", label: "Silero VAD"}
      - {value: "ten", label: "TEN VAD"}
      - {value: "nemo", label: "NeMo"}
```

**Implementation Requirements** (when ready):
1. Extend V4 schema in `config/v4/schema.py`
2. Update ConfigManager to read new paths
3. Add GUI hints processing for new widgets
4. Update model YAML files with new fields

### 16.2 Integration Points Diagram (For Future Reference)

```
                              ┌─────────────────────┐
                              │   User Entry Points │
                              └─────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   CLI (main.py) │      │ GUI (api.py)    │      │ Pass Worker     │
    │                 │      │                 │      │ (ensemble mode) │
    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
             │                        │                        │
             │  --mode qwen           │  Qwen tab              │  --pass1-pipeline qwen
             │                        │  selected              │
             ▼                        ▼                        ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                        Pipeline Routing Layer                         │
    │                                                                       │
    │   main.py:                     pass_worker.py:                        │
    │     if args.mode == "qwen":      PIPELINE_CLASSES = {                 │
    │       pipeline = QwenPipeline()    "qwen": QwenPipeline,              │
    │                                    ...                                │
    │                                  }                                    │
    └──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                           QwenPipeline                                │
    │                                                                       │
    │   8-Phase Flow:                                                       │
    │   1. AudioExtractor → 2. SceneDetector → 3. SpeechEnhancer →         │
    │   4. SpeechSegmenter → 5. QwenASR (+ JapanesePostProcessor) →        │
    │   6. Scene SRT Generation → 7. SRTStitcher → 8. SRTPostProcessor     │
    └──────────────────────────────────────────────────────────────────────┘
```

### 16.3 Performance Baselines (To Be Established)

After implementation, establish performance baselines:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| 1-hour video processing time | < 15 min (GPU) | Benchmark suite |
| VRAM peak usage | < 8 GB | nvidia-smi monitoring |
| Subtitle quality (WER) | < 15% | Manual comparison |
| Scene detection accuracy | > 90% | Ground truth comparison |

---

## Appendices

### Appendix A: File Inventory

**New Files:**
```
whisperjav/pipelines/qwen_pipeline.py     # New dedicated pipeline (~500-600 lines)
tests/test_qwen_pipeline.py               # Unit tests
tests/test_qwen_pipeline_integration.py   # Integration tests (GUI, ensemble)
```

**Modified Files:**
```
whisperjav/ensemble/pass_worker.py        # Routing change + prepare_qwen_params update
whisperjav/main.py                        # Routing change + --qwen-context-file CLI arg
whisperjav/modules/qwen_asr.py            # Add per-call context parameter to transcribe()
whisperjav/modules/japanese_postprocessor.py  # JPP internal step 6 behavioral change
whisperjav/pipelines/__init__.py          # Export QwenPipeline
```

**Cleanup (Phase 4):**
```
whisperjav/pipelines/transformers_pipeline.py  # Remove Qwen code (~200 lines removed)
```

### Appendix B: Existing V4 YAML Reference

**ecosystem.yaml:**
```yaml
schemaVersion: v1
kind: Ecosystem
metadata:
  name: qwen
  displayName: "Qwen3-ASR"
provider:
  module: whisperjav.modules.qwen_asr
  class: QwenASR
```

**qwen3-asr-1.7b.yaml:**
```yaml
schemaVersion: v1
kind: Model
metadata:
  name: qwen3-asr-1.7b
  ecosystem: qwen
spec:
  model.id: "Qwen/Qwen3-ASR-1.7B"
  model.device: auto
  model.dtype: bfloat16
  model.max_new_tokens: 4096
  decode.language: null
  timestamps.mode: word
  aligner.enabled: true
  aligner.id: "Qwen/Qwen3-ForcedAligner-0.6B"
```

### Appendix C: Speech Segmenter Backends

| Backend | Description | Availability |
|---------|-------------|--------------|
| `none` | Passthrough (no filtering) | Always |
| `silero` / `silero-v4.0` | Silero VAD v4.0 | Always (torch required) |
| `silero-v3.1` | Silero VAD v3.1 | Always |
| `ten` | TEN VAD | Requires ten_vad |
| `nemo` / `nemo-lite` | NeMo speech segmentation | Requires nemo_toolkit |
| `whisper-vad` | Whisper-based VAD | Always (faster-whisper required) |

### Appendix D: Related Documents

| Document | Location |
|----------|----------|
| ADR-003: Qwen3-ASR Integration | `docs/architecture/ADR-003-qwen3-asr-integration.md` |
| Qwen ASR Remediation Plan | `docs/architecture/QWEN_ASR_REMEDIATION_PLAN.md` |
| Qwen ASR Flow Audit | `docs/architecture/QWEN_ASR_FLOW_AUDIT.md` |
| Unified Pipeline Design | `docs/architecture/UNIFIED_PIPELINE_DESIGN.md` |
| V4 Config README | `whisperjav/config/v4/README.md` |

### Appendix E: VRAM Management Contract

The JIT (Just-In-Time) load/unload pattern is critical for managing GPU memory across pipeline phases:

```python
# VRAM Management Contract

class VRAMBlock:
    """
    Represents an exclusive VRAM block where only one model should be loaded.

    Contract:
    - Before entering a VRAM block, previous block's model MUST be cleaned up
    - cleanup() MUST release all GPU memory (verified by torch.cuda.memory_allocated())
    - Models within the same block CAN coexist (e.g., ASR + Aligner in Phase 5)
    """
    pass

# Example implementation pattern (from FidelityPipeline):

def process(self, media_info):
    # Phase 3: Enhancement (VRAM Block 1)
    enhancer = SpeechEnhancerFactory.create(backend=self.enhancer_backend)
    for scene in scenes:
        enhancer.enhance(scene.audio_path, scene.enhanced_path)

    # MANDATORY: Clean up before Phase 5
    enhancer.cleanup()
    del enhancer
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 5: ASR (VRAM Block 2)
    asr = QwenASR(model_id=self.model_id, ...)
    for scene in scenes:
        result = asr.transcribe(scene.enhanced_path)
        # ...

    # MANDATORY: Clean up after Phase 5
    asr.cleanup()
    del asr
    torch.cuda.empty_cache()
    gc.collect()
```

**Cleanup Method Contract:**
```python
def cleanup(self) -> None:
    """
    Release all GPU resources held by this module.

    Guarantees:
    - After cleanup(), torch.cuda.memory_allocated() should return to pre-load levels
    - Module is unusable after cleanup() (subsequent calls raise RuntimeError)
    - cleanup() is idempotent (safe to call multiple times)
    """
    if self._model is not None:
        del self._model
        self._model = None
    if self._processor is not None:
        del self._processor
        self._processor = None
    torch.cuda.empty_cache()
    gc.collect()
```

### Appendix F: Review History

| Date | Reviewer | Status | Notes |
|------|----------|--------|-------|
| 2026-02-03 | Author | Draft | Initial creation |
| 2026-02-04 | External Architect | Reviewed | 8 issues identified, recommendations provided |
| 2026-02-04 | Author | Revised | All 8 issues addressed, new sections added |
| 2026-02-04 | User Design Review | Revised | Data flow correction, module contracts completed |
| 2026-02-04 | User Audit | Revised | Scene SRT generation contract corrected (was incorrectly marked NEW) |
| 2026-02-04 | Full Audit | Final | 7 critical, 6 significant, 5 minor issues fixed (see Appendix G) |

**Issues Addressed in First Revision:**
1. ADR-004-01: Clarified JapanesePostProcessor internal step 6 is a proposed behavioral change
2. ADR-004-02: Separated ASR result from SRT generation in Phase 4 diagram
3. ADR-004-03: Moved V4 YAML additions to Future Work section
4. ADR-004-04: Documented aligner derivation from timestamps mode (total: 16 CLI arguments)
5. ADR-004-05: Updated line estimate from ~400 to ~500-600
6. ADR-004-06: Added GUI and ensemble mode regression risks and tests
7. ADR-004-07: Clarified sample rate contract and Phase 3 responsibility
8. ADR-004-08: Deferred V4 config additions to avoid scope creep

**Issues Addressed in Final Revision (User Design Review):**
1. CRITICAL FIX: Speech segmenter moved from BEFORE enhancer to AFTER enhancer
   - Rationale: VAD on clean (enhanced) audio = more accurate detection
   - Flow corrected: extract → scene → enhance → segment → ASR → ...
2. Clarified QwenASR internal structure (Steps 5a/5b/5c already integrated)
3. Confirmed JapanesePostProcessor is separate module called inside qwen_asr.py
4. Added Phase 6 (Scene SRT Generation / Micro-subs) as pipeline responsibility
5. Corrected all phase numbering to reflect 8-phase flow
6. Added complete contracts for all 8 phases with module locations
7. Updated Section 6.2 Module Boundary Summary table
8. Added Section 6.3 QwenPipeline Class Structure
9. Added Section 6.4 QwenASR Internal Structure documentation
10. Fixed duplicate section numbering in Section 7 (was two 7.5 and two 7.8)

**New Sections Added:**
- Glossary (acronym definitions)
- Section 14: Error Handling Strategy
- Section 15: Logging Standards
- Section 16: Future Work
- Appendix E: VRAM Management Contract
- Appendix F: Review History

**Issues Addressed in Audit (User Audit):**
1. Section 7.7 (Scene SRT Generation) was incorrectly marked as "Status: NEW"
   - Scene SRT generation is an EXISTING pattern used by ALL pipelines
   - Corrected to use WhisperResult.to_srt_vtt() (stable_whisper built-in)
   - Same pattern already proven in stable_ts_asr.py:628-638
   - Documented all 3 existing patterns (A/B/C) across pipelines
2. Section 6.1 Phase 6 diagram updated to reference existing method
3. Section 6.2 Module Boundary corrected (was "_segments_to_srt() new method")
4. Section 6.3 QwenPipeline class updated: _save_scene_srt() wraps to_srt_vtt()

**Sections Substantially Updated in Final Revision:**
- Section 6.1: Complete 8-phase data flow diagram with corrected flow
- Section 6.2: Module Boundary Summary (new)
- Section 6.3: QwenPipeline Class Structure (new)
- Section 6.4: QwenASR Internal Structure (new)
- Section 7: All module contracts completed and correctly numbered (7.1-7.9)
- Section 11: Implementation Plan updated with 8-phase detail
- Section 15.3: Progress Logging table updated for 8 phases
- Section 16.2: Integration Points diagram updated

### Appendix G: Full Audit Results (2026-02-04)

Cross-referenced all contracts, phase references, CLI arguments, and method signatures
against the actual codebase. Issues found and fixed:

**Critical (7 items — all fixed):**

| ID | Section | Issue | Fix |
|----|---------|-------|-----|
| C1 | Appendix E | VRAM example said "Phase 4: ASR" — ASR is Phase 5 | Fixed phase numbers |
| C2 | 15.1 | Logging example said "Phase 4: Transcribing" | Fixed to Phase 5 |
| C3 | 4.3 | Table said "Post-ASR filter" for segmenter | Fixed to "Pre-ASR VAD / Post-ASR filter" |
| C4 | 9.1 | `--qwen-segmenter` described as "Post-ASR VAD filter" | Fixed description |
| C5 | 9.1 | Listed `--qwen-use-aligner` as CLI arg #8 — does not exist | Removed; count corrected to 16 |
| C6 | 7.9 | Stats dict keys were invented (didn't match actual code) | Fixed to actual keys from srt_postprocessing.py |
| C7 | 6.3 vs 9.1 | scene_detector default "semantic" vs CLI default "none" | Aligned to "none" with comment |

**Significant (6 items — all fixed; S7 was false alarm):**

| ID | Section | Issue | Fix |
|----|---------|-------|-----|
| S1 | 7.1-7.9 | Parameter names wrong (media_path vs input_file, etc.) | All contracts updated to match actual code |
| S2 | 7.5 | QwenASR.transcribe() missing progress_callback | Added to contract |
| S3 | 7.6 | JapanesePostProcessor missing skip_if_not_japanese | Added to contract |
| S4 | 3.3, 7.6 | "Phase 6" naming collision (JPP internal vs pipeline) | Disambiguated to "JPP internal step 6" |
| S5 | 6.1 | Phase 4→5 interaction underspecified | Added post-ASR filter mechanism spec |
| S6 | 8.2 | V4 fallback code was unreachable (model_id has default) | Rewrote example |

**Minor (5 items — all fixed):**

| ID | Section | Issue | Fix |
|----|---------|-------|-----|
| M1 | 7.x | Type annotations oversimplified (Path vs Union[str, Path]) | Updated in contracts |
| M2 | 7.4 | speech_coverage_ratio listed as field, is actually @property | Added "(computed property)" note |
| M3 | 6.3 | BasePipeline interface not documented | Added required methods and init params |
| M4 | 6.1 | scene detection="none" edge case not documented | Added behavior spec to Phase 2 box |
| M5 | - | DynamicSceneDetector defaults to "auditok" not noted | Documented in 7.2 contract |

**Observations (noted, no changes needed):**

| ID | Topic | Note |
|----|-------|------|
| O1 | VRAM | Segmenter (Phase 4) is lightweight (~2MB), no VRAM block needed | No action needed |
| O2 | Design | 8-phase flow is architecturally sound, good separation of concerns | No action |
| O3 | Testing | No CPU-only testing strategy documented | Deferred to implementation |

---

**End of Document**

*This document has been fully audited against the codebase and is ready for implementation.*
