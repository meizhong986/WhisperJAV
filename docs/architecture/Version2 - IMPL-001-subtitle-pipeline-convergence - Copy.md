# IMPL-001: Subtitle Pipeline Convergence — Implementation Plan

| Field | Value |
|---|---|
| **Status** | DRAFT v2 — Revised after reviewer input |
| **Date** | 2026-02-19 (v2), 2026-02-18 (v1) |
| **Authors** | MK (product vision & strategic direction), Claude (architecture & implementation plan) |
| **Depends on** | ADR-006, Audit documents S1–S5, Reviewer input (reviewer_input_to_IMPL001.txt) |
| **Goal** | Make the Decoupled Subtitle Pipeline the fully-featured, sole production path for Qwen ASR, then delete the legacy coupled modes |
| **Strategy** | **Stabilize & Strangulate** — Make the decoupled pipeline feature-complete and superior, rendering legacy modes obsolete, then remove them |

---

## Revision History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-02-18 | Initial draft with 6-phase plan |
| v2 | 2026-02-19 | Strategic pivot per reviewer input: dropped BRK-1 (physically impossible — coupled modes have no intervention point between generation and alignment); consolidated from 6 phases to 4; reordered to prioritize decoupled pipeline feature-completeness over legacy fixes |

---

## Table of Contents

1. [Context for External Reviewers](#1-context-for-external-reviewers)
2. [Strategic Direction](#2-strategic-direction)
3. [Current State Summary](#3-current-state-summary)
4. [Target State](#4-target-state)
5. [Phase Overview](#5-phase-overview)
6. [Phase 1: Parity — Make Decoupled Pipeline Feature-Complete](#6-phase-1-parity)
7. [Phase 2: Superiority — Step-Down Retry in Orchestrator](#7-phase-2-superiority)
8. [Phase 3: Expansion — Generic Pipeline Entry Point](#8-phase-3-expansion)
9. [Phase 4: Cleanup — Delete Coupled Modes](#9-phase-4-cleanup)
10. [Future: New Backends and Workflows](#10-future-new-backends-and-workflows)
11. [Testing Strategy](#11-testing-strategy)
12. [Risk Register](#12-risk-register)
13. [Decision Log](#13-decision-log)
14. [Scope Boundaries](#14-scope-boundaries)
15. [Dependencies and Prerequisites](#15-dependencies-and-prerequisites)

---

## 1. Context for External Reviewers

### 1.1 What Is WhisperJAV?

WhisperJAV is a subtitle generation tool optimized for Japanese content. It takes video/audio input and produces SRT subtitle files using AI-powered speech recognition.

### 1.2 The Problem Being Solved

Many excellent ASR models produce high-quality text transcription but **do not output timestamps**. Without timestamps, you can't make SRT subtitle files. These models need auxiliary capabilities — temporal framing, forced alignment, timestamp hardening — to produce reliable subtitles.

The goal is to build a **reusable, modular pipeline** that works with any such model, using Qwen3-ASR as the first (reference) implementation.

### 1.3 What Exists Today

A module called `subtitle_pipeline` (`whisperjav/modules/subtitle_pipeline/`) implements the vision via:
- **4 protocol domains**: TemporalFramer, TextGenerator, TextCleaner, TextAligner
- **Factory pattern** per domain for pluggable backends
- **DecoupledSubtitlePipeline orchestrator** composing all components
- **Shared hardening** for timestamp resolution and boundary clamping

However, only **one of three** pipeline input modes (Assembly) uses this architecture. The other two modes (CONTEXT_AWARE, VAD_SLICING) still operate as inline procedural code.

### 1.4 Key Documents

| Document | Location | Content |
|---|---|---|
| ADR-006 | `docs/architecture/ADR-006-decoupled-subtitle-pipeline.md` | Architectural vision and audit findings |
| S1–S5 | `docs/audit/S1_*.md` through `S5_*.md` | Audit trail: module studies, gap analysis, vision feasibility |
| Reviewer Input | `docs/architecture/reviewer_input_to_IMPL001.txt` | Strategic correction from product owner |

---

## 2. Strategic Direction

### 2.1 The Key Insight: BRK-1 Is Physically Impossible

The initial plan (v1) proposed fixing BRK-1: "Add mid-pipeline text cleaning to coupled modes." This was based on the assumption that text cleaning could be injected between text generation and alignment.

**This is physically impossible.** In coupled modes, `QwenASR.transcribe()` runs text generation AND forced alignment as a single atomic operation. There is no intervention point between them. The aligner receives uncleaned, potentially hallucinated text as part of the same model call. By the time we get a result back, the alignment (and any collapse caused by hallucinations) has already happened.

Cleaning the result text AFTER the combined call is cosmetic — it doesn't prevent the aligner collapse that hallucinated text causes. The only architecture that CAN prevent this is the **decoupled pipeline**, where text generation and alignment are separate VRAM-exclusive phases with text cleaning between them.

### 2.2 The "Stabilize & Strangulate" Strategy

**Old strategy** (v1): "Fix the legacy modes so they work better while we build the new one."

**Corrected strategy** (v2): "Accept the legacy modes are architecturally constrained. Make the Decoupled Pipeline the feature-complete default immediately, and retire legacy modes as soon as it's superior."

This means:
1. **No effort on legacy mode fixes** — they are "use at your own risk"
2. **All effort on the decoupled pipeline** — make it reach parity, then surpass legacy modes
3. **Delete legacy code** once the decoupled pipeline is demonstrably superior
4. **Users who want quality** must use the decoupled mode — this is by design, not a limitation

### 2.3 Why This Is Better

| Dimension | Old Strategy (v1) | Corrected Strategy (v2) |
|---|---|---|
| Engineering effort | Writes code that gets deleted (BRK-1 fix) | Every line of code serves the target state |
| User messaging | "All modes work reasonably well" | "Use assembly mode for best quality" (honest) |
| Time to target state | 6 phases | 4 phases |
| Risk | Patching legacy modes may introduce new bugs | Legacy modes are frozen — no new risk |
| Vision alignment | Partially — fixes legacy alongside building new | Fully — all effort goes to the vision |

---

## 3. Current State Summary

### 3.1 Architecture Diagram (Current)

```
                      QwenPipeline
                    (Phases 1-4, 6-9 shared)
                           │
          ┌────────────────┼────────────────────┐
          │                │                     │
    ┌─────┴──────┐   ┌────┴─────┐   ┌──────────┴──────────┐
    │  ASSEMBLY   │   │ CONTEXT  │   │     VAD_SLICING      │
    │  (60 LOC)   │   │ _AWARE   │   │     (434 LOC)        │
    │  Delegates  │   │ (130 LOC)│   │  Inline procedural   │
    │  to orch.   │   │ Inline   │   │  w/ step-down retry  │
    └─────┬───────┘   └────┬─────┘   └──────────┬───────────┘
          │                │                     │
    ┌─────┴─────────┐     │                     │
    │subtitle_pipeline│    │                     │
    │ (orchestrator) ├─────┤─────────────────────┤
    │ + hardening    │  (shared: hardening, reconstruction, types)
    └───────────────┘
```

### 3.2 What the Decoupled Pipeline Has That Coupled Modes Don't

| Feature | Decoupled (Assembly) | Coupled Modes | Impact |
|---|---|---|---|
| Mid-pipeline text cleaning | YES — between generation and alignment | **IMPOSSIBLE** — single atomic call | Prevents hallucination-induced aligner collapse |
| VRAM-exclusive phases | YES — only one model on GPU at a time | NO — both models loaded simultaneously | Higher batch potential, less OOM risk |
| Batch processing | YES — batch generate, batch align | NO — per-scene sequential | Throughput |
| Protocol-based composition | YES — any component swappable | NO — hardcoded to QwenASR | Extensibility |

### 3.3 What Coupled Modes Have That Decoupled Lacks (Gaps to Close)

| Feature | Coupled Modes | Decoupled | Gap ID |
|---|---|---|---|
| Context propagation (cast names, terminology) | YES | **NO** — orchestrator doesn't pass context to generator | BRK-4 |
| Flexible framing (VAD-based dialogue chunks) | YES (VAD_SLICING) | **NO** — hardcoded to FullSceneFramer | GAP-5 |
| Step-down retry (re-frame on collapse) | YES (VAD_SLICING Tier 1→2) | **NO** | GAP-2 |

**Once these three gaps are closed, the decoupled pipeline is superior in every dimension.** That's the goal of Phases 1-2.

---

## 4. Target State

### 4.1 Architecture Diagram (Target)

```
                      QwenPipeline
                    (Phases 1-4, 6-9 shared)
                           │
                    ┌──────┴──────┐
                    │ subtitle_   │
                    │ pipeline    │
                    │ orchestrator│
                    │             │
                    │ + step-down │
                    │ + context   │
                    │ + any framer│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                   │
   ┌────┴────┐      ┌─────┴─────┐     ┌──────┴──────┐
   │full-scene│      │vad-grouped│     │  future:    │
   │ framer   │      │ framer    │     │  whisper,   │
   │          │      │+ step-down│     │  srt-source │
   └──────────┘      └───────────┘     └─────────────┘
```

### 4.2 Key Properties

1. **One orchestrator** for all configurations — no inline mode-specific code
2. **Step-down retry** in orchestrator — available to any framer that supports reframing
3. **Context propagation** works for all configurations
4. **Mid-pipeline text cleaning** applies to all configurations (architectural advantage)
5. **New models** added by registering one TextGenerator backend — orchestrator unchanged
6. **Coupled mode code deleted** — ~600 LOC removed from qwen_pipeline.py

---

## 5. Phase Overview

```
Phase 1 (Parity) ──→ Phase 2 (Superiority) ──→ Phase 3 (Expansion) ──→ Phase 4 (Cleanup)
```

| Phase | Objective | Milestone | Effort |
|---|---|---|---|
| **1: Parity** | Close BRK-4 + GAP-5. Decoupled pipeline matches coupled mode feature set. | Users can do everything in assembly mode that they could in coupled modes (except step-down) | Small-Medium |
| **2: Superiority** | Close GAP-2. Decoupled pipeline is superior to coupled modes in every way. | Assembly + vad-grouped + step-down > VAD_SLICING. No reason to use legacy modes. | Medium |
| **3: Expansion** | Generic entry point. Any-model deployment. | `DecoupledPipeline` class + CLI/YAML config. New models = register one backend. | Medium |
| **4: Cleanup** | Delete coupled mode code. | `CONTEXT_AWARE` and `VAD_SLICING` removed. ~600 LOC deleted. | Small |

**Each phase is independently valuable and independently shippable.**

---

## 6. Phase 1: Parity

### 6.1 Objective

Make the decoupled pipeline (Assembly mode) a complete replacement for daily use by closing the two functional gaps that currently force users toward coupled modes.

### 6.2 Prerequisites

None. This is the starting point.

### 6.3 Task 1.1: Fix Context Propagation in Orchestrator (BRK-4)

**Why**: The orchestrator's `generate_batch()` call doesn't pass user context (cast names, terminology hints) to the generator. Users who provide `--context "cast: Yua Mikami"` get this in coupled modes but NOT in assembly mode.

**Files to change**:
- `whisperjav/modules/subtitle_pipeline/orchestrator.py`
- `whisperjav/pipelines/qwen_pipeline.py`

**Changes to orchestrator.py**:

1. Add `context: str = ""` parameter to `__init__()` (after `language`):

```python
def __init__(
    self,
    framer: TemporalFramer,
    generator: TextGenerator,
    cleaner: TextCleaner,
    aligner: Optional[TextAligner],
    hardening_config: HardeningConfig,
    artifacts_dir: Optional[Path] = None,
    language: str = "ja",
    context: str = "",          # NEW — user context for ASR
):
    ...
    self.context = context
```

2. In `_step2_4_generate_and_clean()`, pass context to generation calls. Around line 300 (batch generation):

```python
# Build per-frame contexts (same context for all frames)
gen_contexts = [self.context] * len(gen_audio_paths) if self.context else None

gen_results = self.generator.generate_batch(
    audio_paths=gen_audio_paths,
    language=self.language,
    contexts=gen_contexts,                    # NEW
    audio_durations=[frames[i].duration for i in gen_indices],
)
```

3. In the per-frame fallback path (around line 316):

```python
result = self.generator.generate(
    audio_path=gen_audio_paths[i],
    language=self.language,
    context=self.context if self.context else None,    # NEW
)
```

**Changes to qwen_pipeline.py**:

In `_build_subtitle_pipeline()` (around line 406), pass context:

```python
return DecoupledSubtitlePipeline(
    framer=framer,
    generator=generator,
    cleaner=cleaner,
    aligner=aligner,
    hardening_config=HardeningConfig(timestamp_mode=new_ts_mode),
    language=cfg.get("language", "ja"),
    context=cfg.get("context", ""),    # NEW
)
```

**Acceptance criteria**:
- [ ] User-provided `--context` reaches the TextGenerator in assembly mode
- [ ] Context is passed to both batch and per-frame generation paths
- [ ] Empty context (`""`) produces no errors or unnecessary parameter passing
- [ ] Coupled modes' context propagation is unchanged

**Test**: Construct orchestrator with `context="test"`. Mock generator. Assert `generate_batch()` receives `contexts=["test", ...]`.

### 6.4 Task 1.2: Wire Framer Selection to Assembly Mode (GAP-5)

**Why**: Assembly mode hardcodes `FullSceneFramer`. The orchestrator supports any framer, but QwenPipeline doesn't expose this. Users who want VAD-based framing are forced to use VAD_SLICING (coupled mode). Exposing the `vad-grouped` framer in assembly mode immediately makes the coupled VAD_SLICING mode redundant (except for step-down, addressed in Phase 2).

**Files to change**:
- `whisperjav/pipelines/qwen_pipeline.py` — constructor + `_build_subtitle_pipeline()`
- `whisperjav/main.py` — CLI argument

**Changes to qwen_pipeline.py**:

1. Add `qwen_framer: str = "full-scene"` parameter to constructor:

```python
def __init__(
    self,
    ...
    qwen_framer: str = "full-scene",    # NEW — "full-scene", "vad-grouped", "srt-source", "manual"
    ...
):
    ...
    self.framer_backend = qwen_framer
```

2. In `_build_subtitle_pipeline()`, replace the hardcoded framer creation:

```python
# BEFORE:
framer = TemporalFramerFactory.create("full-scene")

# AFTER:
framer_kwargs = {}
if self.framer_backend == "vad-grouped":
    framer_kwargs = {
        "segmenter_backend": self.segmenter_backend,
        "max_group_duration_s": self.segmenter_max_group_duration,
        "segmenter_config": self.segmenter_config,
    }
elif self.framer_backend == "srt-source":
    if not hasattr(self, 'framer_srt_path') or not self.framer_srt_path:
        raise ValueError("--qwen-framer srt-source requires --qwen-framer-srt-path")
    framer_kwargs = {"srt_path": self.framer_srt_path}
framer = TemporalFramerFactory.create(self.framer_backend, **framer_kwargs)
```

**Changes to main.py**:

Add to `qwen_group`:

```python
qwen_group.add_argument("--qwen-framer", type=str, default="full-scene",
                        choices=["full-scene", "vad-grouped", "srt-source", "manual"],
                        help="Temporal framing for assembly mode: "
                             "'full-scene' (default), 'vad-grouped' (VAD dialogue chunks)")
qwen_group.add_argument("--qwen-framer-srt-path", type=str, default=None,
                        help="SRT file path for srt-source framer")
```

Wire to pipeline construction:

```python
"qwen_framer": getattr(args, 'qwen_framer', 'full-scene'),
```

**Acceptance criteria**:
- [ ] `--qwen-input-mode assembly --qwen-framer full-scene` produces identical results to current assembly
- [ ] `--qwen-input-mode assembly --qwen-framer vad-grouped` runs the orchestrator with VadGroupedFramer
- [ ] VadGroupedFramer respects `--speech-segmenter`, `--qwen-max-group-duration` settings
- [ ] `--qwen-framer srt-source --qwen-framer-srt-path path.srt` works

**Test**: Run same audio through `assembly + vad-grouped` and `vad_slicing`. Compare segment count and timing coverage (not identical — different code paths — but comparable).

### 6.5 Task 1.3: Minor Cleanups

These are small, low-risk changes done alongside the main tasks:

| Item | File | Change |
|---|---|---|
| CON-1: Duplicate TimestampMode | `qwen_pipeline.py` | Delete local enum (lines 85-117), import from `subtitle_pipeline.types` |
| CON-3: Inline VRAM cleanup | `qwen_pipeline.py` | Replace `gc.collect() + torch.cuda.empty_cache()` with `safe_cuda_cleanup()` |
| ANO-2: Fallback mode mismatch | `qwen_pipeline.py:215` | Change `InputMode.CONTEXT_AWARE` → `InputMode.VAD_SLICING` |
| ANO-5: Stale docstring | `qwen_pipeline.py:9` | "8-Phase" → "9-Phase" |
| ANO-1: Future version refs | `qwen_pipeline.py` | Remove misleading `v1.8.7+`, `v1.8.9+`, `v1.8.10+` annotations |

### 6.6 Phase 1 Milestone

After Phase 1:
- Assembly mode has context propagation (feature parity with coupled modes)
- Assembly mode can use VAD-grouped framing (functional replacement for VAD_SLICING)
- Clean codebase (no duplicate enums, consistent VRAM cleanup)
- **Users can switch from VAD_SLICING to assembly + vad-grouped for everything except step-down retry**

---

## 7. Phase 2: Superiority

### 7.1 Objective

Add step-down retry to the orchestrator. After this phase, the decoupled pipeline is **functionally superior** to all legacy modes. There is zero reason to use coupled modes.

### 7.2 Prerequisites

Phase 1 complete (framer flexibility is required for step-down — the retry re-frames with tighter parameters).

### 7.3 How Step-Down Works Today (VAD_SLICING)

```
1. Group speech segments into Tier 1 groups (e.g., 6s max)
2. Transcribe each group (coupled ASR)
3. If sentinel detects COLLAPSED: defer recovery, queue raw segments
4. Re-group collapsed segments into Tier 2 groups (tighter)
5. Transcribe Tier 2 groups
6. If still collapsed: apply proportional recovery (last resort)
```

### 7.4 How Step-Down Should Work in the Orchestrator

The orchestrator processes all scenes in batch. Step-down is a **second-pass retry** for collapsed scenes:

```
Pass 1 (normal):
  generator.load() → generate ALL scenes → generator.unload()
  cleaner.clean_batch()
  aligner.load() → align ALL scenes → aligner.unload()
  sentinel: identify collapsed scenes → collect indices

Pass 2 (step-down, only for collapsed scenes):
  Re-frame collapsed scenes with tighter framer parameters
  generator.load() → generate retried frames → generator.unload()
  cleaner.clean_batch()
  aligner.load() → align retried frames → aligner.unload()
  sentinel: final assessment
  If still collapsed: proportional recovery (last resort)

Replace Pass 1 results for retried scenes with Pass 2 results
```

**Why batched retry**: This preserves the VRAM-efficient batch load/unload pattern. Per-scene VRAM loading would be too slow.

### 7.5 Task 2.1: Add StepDownConfig Type

**File**: `whisperjav/modules/subtitle_pipeline/types.py`

```python
@dataclass
class StepDownConfig:
    """Configuration for step-down retry on alignment collapse."""
    enabled: bool = False
    fallback_max_group_s: float = 6.0   # Tighter grouping for retry
    max_retries: int = 1                # Number of retry passes
```

### 7.6 Task 2.2: Add `reframe()` to VadGroupedFramer

**File**: `whisperjav/modules/subtitle_pipeline/framers/vad_grouped.py`

```python
def reframe(self, audio: np.ndarray, sample_rate: int,
            max_group_duration_s: float, **kwargs) -> FramingResult:
    """Re-frame with tighter grouping parameters (for step-down retry)."""
    original = self.max_group_duration_s
    self.max_group_duration_s = max_group_duration_s
    try:
        return self.frame(audio, sample_rate, **kwargs)
    finally:
        self.max_group_duration_s = original
```

### 7.7 Task 2.3: Implement Batched Step-Down in Orchestrator

**File**: `whisperjav/modules/subtitle_pipeline/orchestrator.py`

This is the most significant change in the plan. The orchestrator needs:

1. **Constructor**: Accept `stepdown_config: Optional[StepDownConfig] = None`

2. **`process_scenes()` restructure**: After the initial pass (steps 1-9), check for collapsed scenes and run a retry pass:

```python
def process_scenes(self, scene_audio_paths, scene_durations, scene_speech_regions=None):
    # --- Pass 1: Normal processing ---
    results = self._run_pass(scene_audio_paths, scene_durations, scene_speech_regions)

    # --- Pass 2: Step-down retry for collapsed scenes ---
    if self.stepdown_config and self.stepdown_config.enabled and hasattr(self.framer, 'reframe'):
        collapsed_indices = [
            i for i, (result, diag) in enumerate(results)
            if diag.get("sentinel_status") == "COLLAPSED" or (
                isinstance(diag.get("sentinel"), dict) and
                diag["sentinel"].get("status") == "COLLAPSED"
            )
        ]

        if collapsed_indices:
            logger.info(
                "[DecoupledPipeline] Step-down: %d/%d scenes collapsed, retrying with tighter framing",
                len(collapsed_indices), len(results),
            )
            retry_results = self._run_stepdown_pass(
                collapsed_indices, scene_audio_paths, scene_durations, scene_speech_regions,
            )
            # Replace original results with retry results
            for idx, retry_result in zip(collapsed_indices, retry_results):
                results[idx] = retry_result

    return results
```

3. **`_run_pass()`**: Extract the current `process_scenes()` body (steps 1-9) into this method. This is a refactor, not new logic.

4. **`_run_stepdown_pass()`**: New method that re-frames collapsed scenes with tighter parameters:

```python
def _run_stepdown_pass(self, collapsed_indices, scene_audio_paths, scene_durations, scene_speech_regions):
    """Re-process collapsed scenes with tighter framing."""
    # Re-frame only collapsed scenes
    retry_audio_paths = [scene_audio_paths[i] for i in collapsed_indices]
    retry_durations = [scene_durations[i] for i in collapsed_indices]
    retry_speech_regions = ...  # extract for collapsed indices

    # Use tighter framing
    # The framer.reframe() method provides this
    # Run the full pass (generate → clean → align → sentinel → harden) on re-framed scenes
    return self._run_pass(
        retry_audio_paths, retry_durations, retry_speech_regions,
        framer_override_max_group=self.stepdown_config.fallback_max_group_s,
    )
```

**Design note**: `_run_pass()` needs a parameter to override the framer's grouping for retry. The cleanest approach is to have `_run_pass()` accept an optional `framer_kwargs` dict that it passes to the framer's `frame()` or `reframe()` call.

**Estimated complexity**: ~100-150 lines of new orchestrator code, plus refactoring existing `process_scenes()` into `_run_pass()`.

### 7.8 Task 2.4: Wire Step-Down to QwenPipeline

**File**: `whisperjav/pipelines/qwen_pipeline.py`

In `_build_subtitle_pipeline()`:

```python
from whisperjav.modules.subtitle_pipeline.types import StepDownConfig

stepdown_cfg = StepDownConfig(
    enabled=self.stepdown_enabled,
    fallback_max_group_s=self.stepdown_fallback_group,
)

return DecoupledSubtitlePipeline(
    ...,
    stepdown_config=stepdown_cfg,
)
```

### 7.9 Task 2.5: Define Canonical Diagnostics Schema

**File**: `whisperjav/modules/subtitle_pipeline/types.py`

Add a `SceneDiagnostics` dataclass that all paths produce:

```python
@dataclass
class SceneDiagnostics:
    """Canonical per-scene diagnostics (v2.0.0)."""
    schema_version: str = "2.0.0"
    scene_index: int = 0
    scene_duration_sec: float = 0.0
    input_mode: str = ""
    framer_backend: str = ""
    frame_count: int = 0
    word_count: int = 0
    segment_count: int = 0
    sentinel_status: str = "N/A"
    sentinel_triggers: list = field(default_factory=list)
    sentinel_recovery: Optional[dict] = None
    timing_aligner_native: int = 0
    timing_interpolated: int = 0
    timing_vad_fallback: int = 0
    timing_total_segments: int = 0
    hardening_clamped: int = 0
    hardening_sorted: bool = False
    stepdown: Optional[dict] = None
    vad_regions: Optional[list] = None
    group_details: Optional[list] = None
    error: Optional[str] = None
```

Update orchestrator step 9 to build `SceneDiagnostics` instead of ad-hoc dict.

### 7.10 Phase 2 Milestone

After Phase 2:
- Orchestrator has step-down retry (batched, VRAM-efficient)
- Assembly + vad-grouped + step-down ≥ VAD_SLICING in every dimension
- Assembly + full-scene ≥ CONTEXT_AWARE in every dimension
- **Coupled modes have zero unique value. They can be deleted.**
- Canonical diagnostics schema for all analytics

---

## 8. Phase 3: Expansion

### 8.1 Objective

Create a generic, configuration-driven pipeline entry point that enables any-model deployment. This is the architectural enabler for the broader vision (transformers models, vLLM backends).

### 8.2 Prerequisites

Phase 2 recommended (step-down makes the pipeline complete). Phase 1 is sufficient if Phase 2 is deferred.

### 8.3 Task 3.1: Create DecoupledPipeline Class

**New file**: `whisperjav/pipelines/decoupled_pipeline.py`

```python
class DecoupledPipeline(BasePipeline):
    """
    Configuration-driven pipeline using the DecoupledSubtitlePipeline orchestrator.

    Phases 1-4 and 6-9 are standard (shared with QwenPipeline).
    Phase 5 delegates to the orchestrator with user-selected component backends.

    This is the forward-looking entry point for all ASR models.
    QwenPipeline remains for backward compatibility.
    """

    def __init__(
        self,
        # Component backend selection
        generator_backend: str = "qwen3",
        framer_backend: str = "full-scene",
        cleaner_backend: str = "qwen3",
        aligner_backend: str = "qwen3",

        # Component-specific configs (passed to factory.create())
        generator_config: Optional[dict] = None,
        framer_config: Optional[dict] = None,
        cleaner_config: Optional[dict] = None,
        aligner_config: Optional[dict] = None,

        # Pipeline-level config
        timestamp_mode: str = "aligner_interpolation",
        stepdown_enabled: bool = False,
        stepdown_fallback_group_s: float = 6.0,
        context: str = "",
        language: str = "ja",

        # Standard pipeline params
        **kwargs,
    ):
        ...
```

**Design principle**: `DecoupledPipeline` is model-agnostic. It doesn't know about Qwen3, VAD_SLICING, or CONTEXT_AWARE. The "mode" is determined by the combination of components selected.

**Equivalent configurations**:

| Old Mode | DecoupledPipeline Config |
|---|---|
| `assembly` | `generator=qwen3, framer=full-scene, aligner=qwen3` |
| `assembly + vad` | `generator=qwen3, framer=vad-grouped, aligner=qwen3` |
| Future Whisper-guided | `generator=qwen3, framer=whisper-segment, aligner=none` |
| Future vLLM | `generator=vllm, framer=vad-grouped, aligner=none` |

### 8.4 Task 3.2: Wire to CLI

**File**: `whisperjav/main.py`

```python
parser.add_argument("--pipeline", type=str, default=None,
                    choices=["decoupled"],
                    help="Use the generic decoupled pipeline with component selection")
parser.add_argument("--generator", type=str, default="qwen3",
                    help="TextGenerator backend (default: qwen3)")
parser.add_argument("--framer", type=str, default="full-scene",
                    help="TemporalFramer backend (default: full-scene)")
parser.add_argument("--aligner", type=str, default="qwen3",
                    help="TextAligner backend (default: qwen3, 'none' to skip)")
parser.add_argument("--cleaner", type=str, default="qwen3",
                    help="TextCleaner backend (default: qwen3)")
```

### 8.5 Task 3.3: YAML Configuration Support (Optional)

**File**: `whisperjav/config/v4/ecosystems/pipelines/decoupled.yaml`

```yaml
pipeline:
  type: decoupled
  generator:
    backend: qwen3
    model_id: "Qwen/Qwen3-ASR-1.7B"
  framer:
    backend: vad-grouped
    max_group_duration_s: 6.0
  cleaner:
    backend: qwen3
  aligner:
    backend: qwen3
  stepdown:
    enabled: true
    fallback_max_group_s: 6.0
```

### 8.6 Phase 3 Milestone

After Phase 3:
- Any model can be deployed by writing one TextGenerator class and registering it in the factory
- Users select components via CLI or YAML — no code changes needed
- `QwenPipeline` still works for backward compatibility
- `DecoupledPipeline` is the recommended path for new work

---

## 9. Phase 4: Cleanup

### 9.1 Objective

Delete the coupled mode inline code. This is the "strangulate" step — the legacy code is now redundant and can be safely removed.

### 9.2 Prerequisites

Phase 2 complete (decoupled pipeline is superior). Validation confirms output quality is equal or better.

### 9.3 Task 4.1: Map Legacy Input Modes to Orchestrator Configs

**File**: `whisperjav/pipelines/qwen_pipeline.py`

Change `__init__` so all modes construct the orchestrator:

```python
# ALL modes now use the orchestrator
if self.input_mode == InputMode.ASSEMBLY:
    framer_name = self.framer_backend
elif self.input_mode == InputMode.VAD_SLICING:
    framer_name = "vad-grouped"
    logger.warning(
        "input_mode='vad_slicing' is deprecated. "
        "Use input_mode='assembly' with --qwen-framer=vad-grouped instead."
    )
elif self.input_mode == InputMode.CONTEXT_AWARE:
    framer_name = "full-scene"
    logger.warning(
        "input_mode='context_aware' is deprecated. "
        "Use input_mode='assembly' with --qwen-framer=full-scene instead."
    )

self._subtitle_pipeline = self._build_subtitle_pipeline(framer_name=framer_name)
```

### 9.4 Task 4.2: Delete Inline Coupled Mode Code

**File**: `whisperjav/pipelines/qwen_pipeline.py`

Delete:
- The entire `else` branch in Phase 5 (lines ~725-1000) — coupled mode processing
- `_transcribe_speech_regions()` method (lines 1171-1605)
- `_offset_result_timestamps()` static method (lines 1607-1628)
- All imports only used by coupled mode code paths

**Expected reduction**: ~600 lines removed from qwen_pipeline.py.

### 9.5 Task 4.3: Update Ensemble Defaults

**File**: `whisperjav/ensemble/pass_worker.py`

Update default `qwen_input_mode` from `"vad_slicing"` to `"assembly"` (with `framer=vad-grouped` to preserve behavior).

### 9.6 Task 4.4: Update GUI

**File**: `whisperjav/webview_gui/assets/app.js`, `index.html`

Update input mode selectors. Options become framer selections instead of input mode selections:
- Remove "context_aware" and "vad_slicing" from dropdown
- Add framer selection (full-scene, vad-grouped) as the primary UX

### 9.7 Phase 4 Milestone

After Phase 4:
- `qwen_pipeline.py` is ~1000 LOC instead of ~1630
- One code path for Phase 5 — the orchestrator
- All improvements (step-down, diagnostics, new framers) automatically apply to all users
- Legacy mode names still work (with deprecation warning) for backward compatibility

---

## 10. Future: New Backends and Workflows

These are independent items that can be built after Phase 3 (generic entry point). Each is small and self-contained.

### 10.1 WhisperSegmentFramer

**New file**: `whisperjav/modules/subtitle_pipeline/framers/whisper_segment.py`

Runs faster-whisper, extracts segment boundaries as TemporalFrames. Enables "Whisper-Guided Qwen" — Whisper's reliable timing + Qwen's superior Japanese text + no aligner needed.

### 10.2 TransformersTextGenerator

**New file**: `whisperjav/modules/subtitle_pipeline/generators/transformers_generic.py`

Generic HuggingFace ASR pipeline adapter. Any model on HuggingFace that does ASR can be used.

### 10.3 VLLMTextGenerator

**New file**: `whisperjav/modules/subtitle_pipeline/generators/vllm_client.py`

HTTP client for vLLM-served audio-language models. `load()`/`unload()` are no-ops (server manages VRAM).

### 10.4 Cross-Model Benchmarking Utility

CLI tool that runs multiple generator/aligner configurations on the same audio, compares against ground-truth SRT, reports per-scene accuracy metrics.

---

## 11. Testing Strategy

### 11.1 Per-Phase Testing

| Phase | Tests | What to Validate |
|---|---|---|
| 1 | Unit + integration | Context reaches generator; vad-grouped framer produces reasonable output |
| 2 | Integration | Step-down triggers on collapsed scenes; retried results are better; VRAM lifecycle preserved |
| 3 | Integration | DecoupledPipeline constructs correctly from config; produces valid SRT |
| 4 | Regression | No quality degradation after coupled mode removal; legacy mode names still work with warning |

### 11.2 Equivalence Validation (Critical for Phase 4)

Before deleting coupled mode code, run a comparison:
1. Process 3-5 audio files through `vad_slicing` mode (old)
2. Process same files through `assembly + vad-grouped + step-down` (new)
3. Compare: segment count, timing coverage, text quality, collapse rate
4. New pipeline must be equal or better on all metrics

### 11.3 Continuous

- `pytest tests/` passes at every phase boundary
- `ruff check whisperjav/` passes
- Manual smoke test with known audio file

---

## 12. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Phase 2 step-down restructures orchestrator too aggressively | Regression in normal (non-step-down) path | Medium | Extract `_run_pass()` as pure refactor first, then add retry |
| Phase 4 deletion causes quality regression | Users see worse results | Medium | Extensive equivalence testing before deletion |
| VadGroupedFramer in assembly produces different results than VAD_SLICING | User confusion during transition | Low | Expected — different code paths. Validate "comparable, not identical" |
| Ensemble system depends on coupled mode internals | Ensemble breaks after Phase 4 | Low | Audit `pass_worker.py` during Phase 4 |

---

## 13. Decision Log

| # | Decision | Phase | Options | Recommendation | Status |
|---|---|---|---|---|---|
| D1 | Step-down retry approach | 2 | Per-scene VRAM load vs batched retry | Batched retry (preserves efficiency) | OPEN |
| D2 | DecoupledPipeline vs extending QwenPipeline | 3 | New class vs refactor existing | New class (backward compat) | OPEN |
| D3 | Deprecation period for input modes | 4 | Immediate removal vs deprecation warning | One-release deprecation warning | OPEN |
| D4 | CLI UX for component selection | 3 | Flat args vs YAML only | Both (flat for common, YAML for complex) | OPEN |

---

## 14. Scope Boundaries

### In Scope
- Orchestrator feature completeness (context, framer flexibility, step-down, diagnostics)
- Generic pipeline entry point
- Coupled mode deprecation and deletion
- CLI changes for new parameters
- Ensemble default updates

### Out of Scope
- Changes to Phases 1-4 (audio extraction, scene detection, enhancement, VAD)
- Changes to Phases 6-9 (SRT generation, stitching, sanitization, analytics)
- GUI frontend changes (deferred to Phase 4 or later)
- Non-Qwen pipelines (faster, fast, balanced, pro, stable)
- `qwen_asr.py` internals (only external API used)
- Installer or build system changes

---

## 15. Dependencies and Prerequisites

### External Dependencies
- No new packages for Phases 1-4
- Future backends (Phase 6+): `transformers` (existing optional dep), `httpx` (new for vLLM)

### Phase Dependencies

```
Phase 1 ──→ Phase 2 ──→ Phase 3
                │              (can start after Phase 1)
                ↓
             Phase 4
          (requires Phase 2)
```

- **Phase 1**: No dependencies (starting point)
- **Phase 2**: Depends on Phase 1 (framer flexibility needed for step-down)
- **Phase 3**: Depends on Phase 1 minimum; Phase 2 recommended
- **Phase 4**: Depends on Phase 2 (must be superior before deleting legacy)

---

*End of IMPL-001 v2*
