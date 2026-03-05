# IMPL-001: Subtitle Pipeline Convergence — Implementation Plan

| Field | Value |
|---|---|
| **Status** | DRAFT v4 |
| **Date** | 2026-02-19 |
| **Authors** | MK (product vision & strategic direction), Claude (architecture & implementation plan) |
| **Depends on** | ADR-006, Audit documents S1–S5, Reviewer inputs (1st, 2nd, 3rd) |
| **Goal** | Build the Decoupled Subtitle Pipeline into the sole, vision-complete production path for all ASR models — then delete the legacy coupled code |
| **Strategy** | **Stabilize & Strangulate** — All engineering effort serves the decoupled pipeline. Zero effort on legacy fixes. Legacy modes are frozen, then killed. |

---

## Revision History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-02-18 | Initial 6-phase plan. Included BRK-1 fix (coupled mode text cleaning). |
| v2 | 2026-02-19 | Partial revision: removed BRK-1, reordered to 4 phases. **REJECTED** — still framed as "catching up to legacy"; strangulation not formalized; Generic Entry Point delayed; deprecation stance too gentle. |
| v3 | 2026-02-19 | Strategic rewrite. Vision-first framing. 3 phases. Formalized strangulation milestones. **REJECTED** — introduced parallel workstreams (operational risk); step-down lacked user opt-out control. |
| v4 | 2026-02-19 | Serialized to 4 sequential phases. Generic Entry Point (low risk, vision enabler) before Step-Down (complex, gates strangulation). Step-down is explicitly optional with CLI opt-out. |

### What Changed v3 → v4

| Dimension | v3 (Rejected) | v4 (Current) |
|---|---|---|
| **Execution model** | Phase 2 had parallel workstreams (2A + 2B) | **Strictly sequential.** No parallel execution. Each phase completes before the next begins. |
| **Phase count** | 3 phases | **4 phases** — serialized for stability |
| **Generic Entry Point** | Phase 2B (parallel with step-down) | **Phase 2** (before step-down). Low-risk wrapper that unlocks the vision immediately. |
| **Step-Down** | Phase 2A (parallel with entry point) | **Phase 3** (after entry point). Complex orchestrator surgery — done when the vision wrapper is already shipping. |
| **Step-Down opt-out** | Implied via `StepDownConfig.enabled` | **Explicit CLI flag** (`--no-step-down` / `--step-down-attempts 0`). Orchestrator logs warning on collapse when disabled. User controls the quality-vs-speed tradeoff. |

---

## Table of Contents

1. [Context for External Reviewers](#1-context-for-external-reviewers)
2. [Strategic Direction](#2-strategic-direction)
3. [Current State](#3-current-state)
4. [Target State: The Vision Delivered](#4-target-state-the-vision-delivered)
5. [Phase Overview & Strangulation Milestones](#5-phase-overview--strangulation-milestones)
6. [Phase 1: Assembly Parity](#6-phase-1-assembly-parity)
7. [Phase 2: Vision Entry Point](#7-phase-2-vision-entry-point)
8. [Phase 3: Superiority](#8-phase-3-superiority)
9. [Phase 4: Strangulation](#9-phase-4-strangulation)
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

### 1.2 The Product Vision

Many excellent ASR models produce high-quality text transcription but **do not output timestamps**. Without timestamps, you can't make SRT subtitle files. These models need auxiliary capabilities — temporal framing, forced alignment, timestamp hardening — to produce reliable subtitles.

The vision is a **reusable, modular pipeline** that works with **any such model**:

1. Local transformers models (HuggingFace)
2. Remote vLLM-served models (OpenAI-compatible API)
3. Any future inference pattern

Qwen3-ASR is the first (reference) implementation. The architecture must make adding the next model trivial: write one `TextGenerator` class (~100-200 lines), register it in a factory (one line), done.

### 1.3 What Exists Today

A module called `subtitle_pipeline` (`whisperjav/modules/subtitle_pipeline/`) implements this vision via:
- **4 protocol domains**: TemporalFramer, TextGenerator, TextCleaner, TextAligner
- **Factory pattern** per domain for pluggable backends
- **DecoupledSubtitlePipeline orchestrator** composing all components
- **Shared hardening** for timestamp resolution and boundary clamping

However, only **one of three** pipeline input modes (Assembly) uses this architecture. The other two modes (CONTEXT_AWARE, VAD_SLICING) still operate as inline procedural code (~564 LOC) inside `qwen_pipeline.py`. This plan eliminates that duplication.

### 1.4 Key Documents

| Document | Location | Content |
|---|---|---|
| ADR-006 | `docs/architecture/ADR-006-decoupled-subtitle-pipeline.md` | Architectural vision, protocol domains, orchestrator design |
| S1–S5 | `docs/audit/S1_*.md` through `S5_*.md` | Audit trail: module studies, gap analysis, vision feasibility |
| Reviewer Input 1 | `docs/architecture/reviewer_input_to_IMPL001.txt` | Strategic correction: BRK-1 is physically impossible |
| Reviewer Input 2 | `docs/architecture/reviewer_2nd_input_to_IMPL001.txt` | Strategic correction: formalize strangulation, no deprecation grace period |
| Reviewer Input 3 | `docs/architecture/reviewer_3rd_input_to_IMPL001.txt` | Operational correction: no parallel execution, step-down must be user-optional |

---

## 2. Strategic Direction

### 2.1 The Integrated Aligner Limitation

In coupled modes, `QwenASR.transcribe()` runs text generation AND forced alignment as a **single atomic operation**. There is no intervention point between them. The aligner receives uncleaned, potentially hallucinated text as part of the same model call. By the time we get a result back, the alignment collapse caused by hallucinations has already happened.

**Cleaning the result text after the combined call is cosmetic** — it doesn't prevent the collapse. The only architecture that CAN prevent this is the **decoupled pipeline**, where text generation and alignment are separate VRAM-exclusive phases with text cleaning between them.

This means: **fixing coupled modes is physically impossible.** Any engineering time spent on them is wasted.

### 2.2 The "Stabilize & Strangulate" Strategy

| Principle | Action |
|---|---|
| **Zero effort on legacy** | Coupled modes (CONTEXT_AWARE, VAD_SLICING) receive no fixes, no improvements, no attention. They are frozen as-is. |
| **All effort on the vision** | Every line of code serves the decoupled pipeline — the architecture that delivers the product vision. |
| **Assembly is the default** | As soon as the decoupled pipeline reaches functional parity (Phase 1), Assembly becomes the default mode. Users who stay on legacy modes do so at their own risk. |
| **No deprecation grace period** | Coupled modes are not "deprecated with a transition period." They are replaced by a superior architecture and then deleted. |
| **Delete, don't maintain** | Once the decoupled pipeline is superior (Phase 3), the coupled code is deleted (Phase 4). No keeping it around "just in case." |

### 2.3 Why This Is the Right Strategy

The decoupled pipeline isn't just a refactor — it's the **only architecture that can deliver the product vision**:

| Capability | Decoupled Pipeline | Coupled Modes |
|---|---|---|
| Mid-pipeline text cleaning (prevents aligner collapse) | YES | **IMPOSSIBLE** |
| VRAM-exclusive phases (one model on GPU at a time) | YES | NO |
| Batch processing (throughput) | YES | NO |
| New model deployment (plug in any TextGenerator) | YES | NO — hardcoded to QwenASR |
| vLLM backend support | YES — `load()`/`unload()` become no-ops | NO |
| Step-down retry (quality recovery) | Will have (Phase 3, optional) | Only VAD_SLICING has it |
| Framer flexibility (VAD-grouped, SRT-source, etc.) | YES — any framer via factory | NO — hardcoded framing logic |

Engineering time spent on coupled modes is time NOT spent making this vision real. The coupled modes cannot evolve toward the vision — they are architecturally dead ends.

---

## 3. Current State

### 3.1 Architecture Diagram

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

### 3.2 Three Features the Vision Requires

These are **features the vision demands** that the decoupled pipeline must have regardless of whether legacy modes exist:

| Vision Requirement | Current Gap | Gap ID |
|---|---|---|
| **Context propagation** — user-provided cast names, terminology, domain hints must reach the text generator | Orchestrator constructor has no `context` parameter; `_step2_4_generate_and_clean()` doesn't pass context to `generate_batch()` | BRK-4 |
| **Flexible temporal framing** — different content needs different framing strategies (full-scene for clean audio, VAD-grouped for noisy/multi-speaker) | `_build_subtitle_pipeline()` hardcodes `FullSceneFramer` (line 365). Orchestrator supports any framer, but the wiring doesn't expose the choice. | GAP-5 |
| **Step-down retry (optional)** — when forced alignment collapses, optionally re-process with tighter framing. Users may prefer speed over retry, so this must be opt-out. | Orchestrator has no retry logic. This capability exists only in VAD_SLICING's inline code (~434 LOC). | GAP-2 |

**Once these three features are built into the orchestrator, the decoupled pipeline delivers the full vision. The coupled modes have zero remaining value.**

---

## 4. Target State: The Vision Delivered

### 4.1 Architecture Diagram

```
                      QwenPipeline                DecoupledPipeline
                   (backward compat)              (vision entry point)
                           │                            │
                           └────────────┬───────────────┘
                                        │
                                 ┌──────┴──────┐
                                 │ subtitle_   │
                                 │ pipeline    │
                                 │ orchestrator│
                                 │             │
                                 │ + context   │
                                 │ + step-down │
                                 │   (optional)│
                                 │ + any framer│
                                 └──────┬──────┘
                                        │
                  ┌─────────────────────┼──────────────────────┐
                  │                     │                       │
           ┌──────┴──────┐      ┌──────┴──────┐      ┌────────┴────────┐
           │ full-scene   │      │ vad-grouped  │      │ future:         │
           │ framer       │      │ framer       │      │ whisper-segment │
           │              │      │ + step-down  │      │ srt-source      │
           └──────────────┘      └──────────────┘      └─────────────────┘
```

### 4.2 What "Vision Delivered" Means

1. **One orchestrator** for all configurations — zero inline mode-specific ASR code
2. **Context propagation** works for all configurations
3. **Step-down retry** in orchestrator — optional, user-controlled via `--no-step-down`
4. **Mid-pipeline text cleaning** applies to all configurations (the architectural advantage that makes this worth doing)
5. **Generic entry point** (`DecoupledPipeline`) — user selects components via CLI/YAML, no code changes
6. **New models** added by registering one TextGenerator backend — orchestrator unchanged
7. **Coupled mode code deleted** — ~600 LOC removed from qwen_pipeline.py
8. **Adding the next ASR model**:
   - Write one `TextGenerator` class (~100-200 lines)
   - Optionally write one `TextCleaner` for model-specific artifacts
   - Register both in their respective factories (one line each)
   - User selects the new model via CLI/GUI
   - No orchestrator changes. No hardening changes. No framer changes.

---

## 5. Phase Overview & Strangulation Milestones

```
Phase 1 (Parity) ──→ Phase 2 (Vision Entry Point) ──→ Phase 3 (Superiority) ──→ Phase 4 (Strangulation)
   [sequential]           [sequential]                    [sequential]              [sequential]
```

| Phase | Objective | Risk | Effort |
|---|---|---|---|
| **1: Assembly Parity** | Context propagation + framer flexibility. Assembly becomes the default. | Low | Small-Medium |
| **2: Vision Entry Point** | Create `DecoupledPipeline` class. Any-model deployment unlocked. | Low (wiring, no orchestrator surgery) | Small-Medium |
| **3: Superiority** | Step-down retry (optional, user-controlled). Assembly surpasses VAD_SLICING. | Medium (orchestrator refactor) | Medium |
| **4: Strangulation** | Delete coupled mode code. One architecture. | Low (deletion + validation) | Small |

**Execution model: strictly sequential.** Each phase completes, ships, and stabilizes before the next begins. No parallel workstreams.

**Each phase is independently valuable and independently shippable.**

### Strangulation Milestones

These are non-negotiable deliverables at each phase boundary.

| Phase Exit | Strangulation Milestone | Action |
|---|---|---|
| **Phase 1** | **Assembly is the default mode.** | Change `qwen_input_mode` default from `"vad_slicing"` to `"assembly"` in constructor, CLI, and GUI. Coupled modes still work but are no longer the default path. |
| **Phase 2** | **Vision is unlocked.** | `DecoupledPipeline` exists. New models can be deployed by writing one TextGenerator + registering it. The architecture's promise is real, not theoretical. |
| **Phase 3** | **Coupled modes have zero unique value.** | Assembly + vad-grouped + step-down ≥ VAD_SLICING in every dimension. Assembly + full-scene ≥ CONTEXT_AWARE. No reason to use legacy modes. |
| **Phase 4** | **Coupled mode code is deleted.** | Remove `CONTEXT_AWARE` and `VAD_SLICING` inline code (~600 LOC). Old mode names in CLI map to assembly + appropriate framer config. |

### Why This Phase Order

| Order | Rationale |
|---|---|
| **Phase 2 (Entry Point) before Phase 3 (Step-Down)** | The Generic Entry Point is a low-risk wiring exercise that unlocks the product vision immediately. Step-Down is complex orchestrator surgery. If Step-Down takes longer than expected, the vision's value is already shipping. |
| **Phase 3 (Step-Down) before Phase 4 (Strangulation)** | Step-Down is the last feature that makes coupled modes redundant. Strangulation is gated on Assembly being provably superior — which requires step-down. |
| **No parallel phases** | Parallel workstreams introduce integration conflicts, merge overhead, and debugging complexity. Sequential execution is safer and more predictable for a stability-critical codebase. |

---

## 6. Phase 1: Assembly Parity

### 6.1 Objective

Build context propagation and framer flexibility into the decoupled pipeline — two features the vision requires. After this phase, Assembly can do everything coupled modes can do except step-down retry. **Assembly becomes the default mode.**

### 6.2 Prerequisites

None. This is the starting point.

### 6.3 Task 1.1: Fix Context Propagation in Orchestrator (BRK-4)

**Why (vision framing)**: Context propagation is essential for real-world ASR quality. Every model benefits from hints about expected content. The `TextGenerator` protocol already accepts `context` — the orchestrator just doesn't pass it through. This is a gap in the vision's infrastructure.

**Files to change**:
- `whisperjav/modules/subtitle_pipeline/orchestrator.py`
- `whisperjav/pipelines/qwen_pipeline.py`

**Changes to orchestrator.py**:

1. Add `context: str = ""` parameter to `__init__()` (after `language`, line 68):

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
    context: str = "",          # User context for ASR (cast names, terminology)
):
    ...
    self.context = context
```

2. In `_step2_4_generate_and_clean()`, pass context to generation calls.

Batch generation path (around line 300):

```python
gen_contexts = [self.context] * len(gen_audio_paths) if self.context else None

gen_results = self.generator.generate_batch(
    audio_paths=gen_audio_paths,
    language=self.language,
    contexts=gen_contexts,                    # NEW
    audio_durations=[frames[i].duration for i in gen_indices],
)
```

Per-frame fallback path (around line 316):

```python
result = self.generator.generate(
    audio_path=gen_audio_paths[i],
    language=self.language,
    context=self.context if self.context else None,    # NEW
)
```

**Changes to qwen_pipeline.py**:

In `_build_subtitle_pipeline()` (line 406), pass context:

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

**Test**: Construct orchestrator with `context="test"`. Mock generator. Assert `generate_batch()` receives `contexts=["test", ...]`.

### 6.4 Task 1.2: Wire Framer Selection to Assembly Mode (GAP-5)

**Why (vision framing)**: The vision requires flexible temporal framing — different content needs different strategies. The orchestrator already supports any framer via the TemporalFramer protocol, but `_build_subtitle_pipeline()` hardcodes `FullSceneFramer` (line 365). Exposing framer selection allows the decoupled pipeline to handle the same use cases as VAD_SLICING, immediately making that legacy mode redundant for daily use (except step-down).

**Files to change**:
- `whisperjav/pipelines/qwen_pipeline.py` — constructor + `_build_subtitle_pipeline()`
- `whisperjav/main.py` — CLI argument

**Changes to qwen_pipeline.py**:

1. Add `qwen_framer: str = "full-scene"` parameter to constructor:

```python
def __init__(
    self,
    ...
    qwen_framer: str = "full-scene",    # "full-scene", "vad-grouped", "srt-source", "manual"
    ...
):
    ...
    self.framer_backend = qwen_framer
```

2. In `_build_subtitle_pipeline()`, replace the hardcoded framer creation (line 365):

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

### 6.5 Task 1.3: Switch Default Mode to Assembly

**Why**: This is the first strangulation milestone. Assembly is now functionally equivalent to coupled modes for daily use. Making it the default steers all users toward the architecture that can actually deliver quality results (mid-pipeline text cleaning prevents aligner collapse).

**Files to change**:
- `whisperjav/pipelines/qwen_pipeline.py` — constructor default + fallback
- `whisperjav/main.py` — CLI default
- `whisperjav/webview_gui/assets/app.js` — GUI default

**Changes**:

1. Constructor default (line ~150):
```python
# BEFORE:
qwen_input_mode: str = "vad_slicing"

# AFTER:
qwen_input_mode: str = "assembly"
```

2. Fallback mode (line 215):
```python
# BEFORE:
self.input_mode = InputMode.CONTEXT_AWARE

# AFTER:
self.input_mode = InputMode.ASSEMBLY
```

3. CLI default in `main.py`:
```python
# BEFORE:
default="vad_slicing"

# AFTER:
default="assembly"
```

4. GUI default in `app.js`: Update the Qwen input mode selector default.

5. Add warning when coupled modes are explicitly selected:
```python
if self.input_mode in (InputMode.CONTEXT_AWARE, InputMode.VAD_SLICING):
    logger.warning(
        "input_mode='%s' uses legacy coupled architecture. "
        "For best quality, use input_mode='assembly' with --qwen-framer. "
        "Coupled modes will be removed in a future release.",
        self.input_mode.value,
    )
```

**Acceptance criteria**:
- [ ] Running `whisperjav video.mp4 --mode qwen` without `--qwen-input-mode` uses Assembly
- [ ] Running with `--qwen-input-mode vad_slicing` still works but logs a warning
- [ ] GUI defaults to Assembly

### 6.6 Task 1.4: Minor Cleanups

Small, low-risk changes done alongside the main tasks:

| Item | File | Change |
|---|---|---|
| CON-1: Duplicate TimestampMode | `qwen_pipeline.py` | Delete local enum (lines 85-117), import from `subtitle_pipeline.types` |
| CON-3: Inline VRAM cleanup | `qwen_pipeline.py` | Replace `gc.collect() + torch.cuda.empty_cache()` (lines 569-575, 981-986) with `safe_cuda_cleanup()` |
| ANO-2: Fallback mode mismatch | `qwen_pipeline.py:215` | Change fallback to `InputMode.ASSEMBLY` (done in Task 1.3) |
| ANO-5: Stale docstring | `qwen_pipeline.py:9` | "8-Phase" → "9-Phase" |
| ANO-1: Future version refs | `qwen_pipeline.py` | Remove misleading `v1.8.7+`, `v1.8.9+`, `v1.8.10+` annotations |

### 6.7 Phase 1 Exit Criteria

- [ ] Context propagation works in orchestrator (BRK-4 closed)
- [ ] Framer selection exposed in assembly mode (GAP-5 closed)
- [ ] Assembly is the default mode in constructor, CLI, and GUI
- [ ] Coupled modes log deprecation warning when explicitly selected
- [ ] Minor cleanups complete (CON-1, CON-3, ANO-1, ANO-2, ANO-5)
- [ ] `pytest tests/` passes
- [ ] `ruff check whisperjav/` passes

**Strangulation milestone achieved**: Assembly is the default. Users must opt-in to legacy modes.

---

## 7. Phase 2: Vision Entry Point

### 7.1 Objective

Create the `DecoupledPipeline` class — the configuration-driven, model-agnostic pipeline entry point. This is the product vision's core deliverable: the ability to deploy any ASR model by selecting components, without writing pipeline code.

This is a **low-risk wiring phase**. It creates a new pipeline class that delegates to the existing orchestrator. No orchestrator internals are modified.

### 7.2 Prerequisites

Phase 1 complete (context propagation + framer selection must be in the orchestrator for the entry point to be feature-complete).

### 7.3 Task 2.1: Create DecoupledPipeline Class

**New file**: `whisperjav/pipelines/decoupled_pipeline.py`

```python
class DecoupledPipeline(BasePipeline):
    """
    Configuration-driven pipeline using the DecoupledSubtitlePipeline orchestrator.

    Phases 1-4 and 6-9 are standard (shared with QwenPipeline).
    Phase 5 delegates to the orchestrator with user-selected component backends.

    This is the forward-looking entry point for all ASR models.
    QwenPipeline remains for Qwen3-specific backward compatibility.
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

**Design principle**: `DecoupledPipeline` is model-agnostic. It doesn't know about Qwen3, VAD_SLICING, or CONTEXT_AWARE. The "behavior" is determined by the combination of components selected.

**Equivalent configurations**:

| Use Case | DecoupledPipeline Config |
|---|---|
| Current assembly default | `generator=qwen3, framer=full-scene, aligner=qwen3` |
| Assembly + VAD framing | `generator=qwen3, framer=vad-grouped, aligner=qwen3` |
| Fast mode (no aligner) | `generator=qwen3, framer=vad-grouped, aligner=none` |
| Future: Whisper-guided Qwen | `generator=qwen3, framer=whisper-segment, aligner=none` |
| Future: vLLM model | `generator=vllm, framer=vad-grouped, aligner=none` |
| Future: HuggingFace model | `generator=transformers, framer=full-scene, aligner=none` |

### 7.4 Task 2.2: Wire to CLI

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

### 7.5 Task 2.3: YAML Configuration Support (Optional)

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

### 7.6 Phase 2 Exit Criteria

- [ ] `DecoupledPipeline` constructs correctly from component backend names
- [ ] CLI `--pipeline decoupled --generator qwen3 --framer vad-grouped` produces valid SRT
- [ ] `DecoupledPipeline` with `aligner=none` works (aligner-free workflow)
- [ ] Registering a new generator backend in factory makes it available via CLI without code changes
- [ ] `pytest tests/` passes
- [ ] `ruff check whisperjav/` passes

**Strangulation milestone achieved**: The vision is unlocked. New models can be deployed by writing one TextGenerator and registering it.

---

## 8. Phase 3: Superiority

### 8.1 Objective

Add **optional** step-down retry to the orchestrator. This is the complex orchestrator surgery that makes Assembly provably superior to VAD_SLICING. After this phase, coupled modes have zero unique value.

**Step-down is user-controlled.** Users who prefer speed over quality, or who trust proportional recovery, can opt out via `--no-step-down`. The orchestrator respects this choice.

### 8.2 Prerequisites

Phase 1 complete (framer flexibility is required — step-down re-frames with tighter parameters). Phase 2 is NOT a prerequisite — step-down works through QwenPipeline, not through DecoupledPipeline.

### 8.3 How Step-Down Works Today (VAD_SLICING)

```
1. Group speech segments into Tier 1 groups (e.g., 6s max)
2. Transcribe each group (coupled ASR — text + alignment as one atomic call)
3. If sentinel detects COLLAPSED: defer recovery, queue raw segments
4. Re-group collapsed segments into Tier 2 groups (tighter)
5. Transcribe Tier 2 groups
6. If still collapsed: apply proportional recovery (last resort)
```

This logic lives in `_transcribe_speech_regions()` (lines 1171-1605, 434 LOC of inline code). It is valuable but trapped inside legacy code that is architecturally constrained.

### 8.4 How Step-Down Works in the Orchestrator

The orchestrator processes all scenes in batch. Step-down is a **second-pass retry** for collapsed scenes:

```
Pass 1 (normal):
  framer.frame() all scenes
  generator.load() → generate ALL scenes → generator.unload()
  cleaner.clean_batch()
  aligner.load() → align ALL scenes → aligner.unload()
  sentinel: identify collapsed scenes → collect indices

IF step-down is enabled:
  Pass 2 (step-down, only for collapsed scenes):
    Re-frame collapsed scenes with tighter framer parameters
    generator.load() → generate retried frames → generator.unload()
    cleaner.clean_batch()
    aligner.load() → align retried frames → aligner.unload()
    sentinel: final assessment
    If still collapsed: proportional recovery (last resort)
    Replace Pass 1 results for retried scenes with Pass 2 results

IF step-down is disabled:
  Log warning: "N scenes collapsed — step-down retry disabled by user"
  Apply proportional recovery directly (standard sentinel fallback)
```

**Why batched retry**: This preserves the VRAM-efficient batch load/unload pattern. Per-scene VRAM loading would be too slow.

### 8.5 Task 3.1: Add StepDownConfig Type

**File**: `whisperjav/modules/subtitle_pipeline/types.py`

```python
@dataclass
class StepDownConfig:
    """Configuration for step-down retry on alignment collapse.

    Step-down is OPTIONAL. When disabled (enabled=False), the orchestrator
    skips the retry pass and falls through to proportional recovery directly.
    Users control this via --no-step-down or --step-down-attempts 0.
    """
    enabled: bool = True              # User can opt out
    fallback_max_group_s: float = 6.0  # Tighter grouping for retry
    max_retries: int = 1               # Number of retry passes
```

### 8.6 Task 3.2: Add `reframe()` to VadGroupedFramer

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

### 8.7 Task 3.3: Implement Batched Step-Down in Orchestrator

**File**: `whisperjav/modules/subtitle_pipeline/orchestrator.py`

This is the most significant change in the plan. The orchestrator needs:

1. **Constructor**: Accept `stepdown_config: Optional[StepDownConfig] = None`

2. **Refactor `process_scenes()` into `_run_pass()`**: Extract the current body (steps 1-9) into `_run_pass()`. This is a pure refactor — no new logic. `process_scenes()` becomes the outer driver that calls `_run_pass()` and then optionally runs step-down.

3. **Add step-down logic to `process_scenes()`**:

```python
def process_scenes(self, scene_audio_paths, scene_durations, scene_speech_regions=None):
    # --- Pass 1: Normal processing ---
    results = self._run_pass(scene_audio_paths, scene_durations, scene_speech_regions)

    # --- Identify collapsed scenes ---
    collapsed_indices = [
        i for i, (result, diag) in enumerate(results)
        if self._is_collapsed(diag)
    ]

    if not collapsed_indices:
        return results

    # --- Step-down decision ---
    if (self.stepdown_config
            and self.stepdown_config.enabled
            and hasattr(self.framer, 'reframe')):
        # Step-down enabled: retry with tighter framing
        logger.info(
            "[DecoupledPipeline] Step-down: %d/%d scenes collapsed, "
            "retrying with tighter framing (%.1fs max group)",
            len(collapsed_indices), len(results),
            self.stepdown_config.fallback_max_group_s,
        )
        retry_results = self._run_stepdown_pass(
            collapsed_indices, scene_audio_paths,
            scene_durations, scene_speech_regions,
        )
        for idx, retry_result in zip(collapsed_indices, retry_results):
            results[idx] = retry_result
    else:
        # Step-down disabled or not available: log and proceed
        reason = (
            "disabled by user configuration"
            if self.stepdown_config and not self.stepdown_config.enabled
            else "framer does not support reframing"
            if not hasattr(self.framer, 'reframe')
            else "not configured"
        )
        logger.warning(
            "[DecoupledPipeline] %d/%d scenes collapsed but step-down retry is %s. "
            "Falling back to proportional recovery.",
            len(collapsed_indices), len(results), reason,
        )
        # Proportional recovery happens in _step9 sentinel logic (already applied)

    return results
```

4. **`_run_stepdown_pass()`**: New method that re-frames collapsed scenes:

```python
def _run_stepdown_pass(self, collapsed_indices, scene_audio_paths,
                       scene_durations, scene_speech_regions):
    """Re-process collapsed scenes with tighter framing."""
    retry_audio_paths = [scene_audio_paths[i] for i in collapsed_indices]
    retry_durations = [scene_durations[i] for i in collapsed_indices]
    retry_speech_regions = (
        [scene_speech_regions[i] for i in collapsed_indices]
        if scene_speech_regions else None
    )
    return self._run_pass(
        retry_audio_paths, retry_durations, retry_speech_regions,
        framer_override_max_group=self.stepdown_config.fallback_max_group_s,
    )
```

5. **`_run_pass()`**: Accepts optional `framer_override_max_group` parameter. When present, calls `framer.reframe(audio, sr, max_group_duration_s=override)` instead of `framer.frame(audio, sr)`.

**Implementation order** (minimize regression risk):
1. Extract `_run_pass()` as pure refactor — `process_scenes()` calls it, behavior identical
2. Add `stepdown_config` to constructor (no-op when not configured)
3. Add step-down logic with enabled/disabled branching
4. Test both paths (enabled + disabled)

**Estimated complexity**: ~100-150 lines of new orchestrator code, plus refactoring existing `process_scenes()` into `_run_pass()`.

### 8.8 Task 3.4: Wire Step-Down to QwenPipeline and CLI

**Files to change**:
- `whisperjav/pipelines/qwen_pipeline.py`
- `whisperjav/main.py`

**Changes to qwen_pipeline.py**:

In `_build_subtitle_pipeline()`:

```python
from whisperjav.modules.subtitle_pipeline.types import StepDownConfig

stepdown_cfg = StepDownConfig(
    enabled=self.stepdown_enabled,
    fallback_max_group_s=self.stepdown_fallback_group,
    max_retries=self.stepdown_max_retries,
)

return DecoupledSubtitlePipeline(
    ...,
    stepdown_config=stepdown_cfg,
)
```

**Changes to main.py** — explicit opt-out controls:

```python
qwen_group.add_argument("--no-step-down", action="store_true", default=False,
                        help="Disable step-down retry on alignment collapse. "
                             "Collapsed scenes use proportional recovery directly.")
qwen_group.add_argument("--step-down-attempts", type=int, default=1,
                        help="Number of step-down retry attempts (0 = disabled, default: 1)")
```

Wire to pipeline construction:

```python
# --no-step-down overrides --step-down-attempts
stepdown_enabled = not getattr(args, 'no_step_down', False)
if getattr(args, 'step_down_attempts', 1) == 0:
    stepdown_enabled = False

pipeline_kwargs["stepdown_enabled"] = stepdown_enabled
pipeline_kwargs["stepdown_max_retries"] = getattr(args, 'step_down_attempts', 1)
```

**User experience**:

```bash
# Default: step-down enabled (1 attempt)
whisperjav video.mp4 --mode qwen --qwen-framer vad-grouped

# Opt out: prefer speed, skip retry
whisperjav video.mp4 --mode qwen --qwen-framer vad-grouped --no-step-down

# Equivalent opt-out via attempts=0
whisperjav video.mp4 --mode qwen --qwen-framer vad-grouped --step-down-attempts 0
```

### 8.9 Task 3.5: Define Canonical Diagnostics Schema

**File**: `whisperjav/modules/subtitle_pipeline/types.py`

```python
@dataclass
class SceneDiagnostics:
    """Canonical per-scene diagnostics."""
    schema_version: str = "2.0.0"
    scene_index: int = 0
    scene_duration_sec: float = 0.0
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
    stepdown: Optional[dict] = None    # {"attempted": bool, "enabled": bool, "improved": bool}
    vad_regions: Optional[list] = None
    group_details: Optional[list] = None
    error: Optional[str] = None
```

The `stepdown` field records whether retry was attempted, whether it was enabled, and whether it improved the result — giving users and developers clear visibility into the quality-recovery decision.

### 8.10 Phase 3 Exit Criteria

- [ ] Step-down retry works in orchestrator when enabled
- [ ] Step-down is correctly skipped when `--no-step-down` or `--step-down-attempts 0` is passed
- [ ] Orchestrator logs clear warning when collapse detected but step-down disabled
- [ ] Proportional recovery applies as fallback when step-down is disabled
- [ ] VRAM lifecycle preserved (batch load/unload, not per-scene)
- [ ] Assembly + vad-grouped + step-down ≥ VAD_SLICING on test audio
- [ ] Assembly + full-scene ≥ CONTEXT_AWARE on test audio
- [ ] Canonical `SceneDiagnostics` schema in use
- [ ] `pytest tests/` passes
- [ ] `ruff check whisperjav/` passes

**Strangulation milestone achieved**: Coupled modes have zero unique value. Assembly is superior in every dimension.

---

## 9. Phase 4: Strangulation

### 9.1 Objective

Delete the coupled mode code. This is not "cleanup" — this is the completion of the strategic objective. The legacy code is dead weight: it cannot deliver the product vision, it confuses users with a false choice, and it doubles the maintenance surface of Phase 5.

### 9.2 Prerequisites

Phase 3 complete (Assembly is provably superior). Equivalence validation confirms output quality is equal or better.

### 9.3 Task 4.1: Map Legacy Input Mode Names to Assembly Configs

**File**: `whisperjav/pipelines/qwen_pipeline.py`

All input modes now construct the orchestrator. Legacy mode names become convenience aliases:

```python
# ALL modes use the orchestrator — legacy names map to framer configs
if self.input_mode == InputMode.ASSEMBLY:
    framer_name = self.framer_backend
elif self.input_mode == InputMode.VAD_SLICING:
    framer_name = "vad-grouped"
    logger.warning(
        "input_mode='vad_slicing' is removed. "
        "Automatically using assembly mode with vad-grouped framer."
    )
elif self.input_mode == InputMode.CONTEXT_AWARE:
    framer_name = "full-scene"
    logger.warning(
        "input_mode='context_aware' is removed. "
        "Automatically using assembly mode with full-scene framer."
    )

self._subtitle_pipeline = self._build_subtitle_pipeline(framer_name=framer_name)
```

### 9.4 Task 4.2: Delete Inline Coupled Mode Code

**File**: `whisperjav/pipelines/qwen_pipeline.py`

Delete:
- The entire `else` branch in Phase 5 (lines ~725-1000) — coupled mode processing
- `_transcribe_speech_regions()` method (lines 1171-1605) — 434 LOC
- `_offset_result_timestamps()` static method (lines 1607-1628)
- All imports only used by coupled mode code paths

**Expected reduction**: ~600 lines removed from qwen_pipeline.py.

### 9.5 Task 4.3: Update Ensemble Defaults

**File**: `whisperjav/ensemble/pass_worker.py`

Update default `qwen_input_mode` from `"vad_slicing"` to `"assembly"` (with `framer=vad-grouped` to preserve behavior).

### 9.6 Task 4.4: Update GUI

**Files**: `whisperjav/webview_gui/assets/app.js`, `index.html`

- Remove "context_aware" and "vad_slicing" from input mode dropdown
- Replace with framer selection (full-scene, vad-grouped) as the primary UX
- The concept of "input modes" is gone — users choose a framer strategy

### 9.7 Phase 4 Exit Criteria

- [ ] All Phase 5 code paths go through the orchestrator
- [ ] `qwen_pipeline.py` is ~1000 LOC (down from ~1630)
- [ ] Legacy mode names still accepted in CLI (mapped to assembly configs, with warning)
- [ ] Ensemble uses assembly mode
- [ ] GUI offers framer selection instead of input mode selection
- [ ] `pytest tests/` passes
- [ ] `ruff check whisperjav/` passes

**Strangulation milestone achieved**: Coupled mode code no longer exists. One code path. One architecture.

---

## 10. Future: New Backends and Workflows

These are independent items that can be built after Phase 2 (generic entry point exists). Each is small and self-contained. They represent the product vision being realized.

### 10.1 WhisperSegmentFramer

**New file**: `whisperjav/modules/subtitle_pipeline/framers/whisper_segment.py`

Runs faster-whisper, extracts segment boundaries as TemporalFrames. Enables "Whisper-Guided Qwen" — Whisper's reliable timing + Qwen's superior Japanese text + no aligner needed.

### 10.2 TransformersTextGenerator

**New file**: `whisperjav/modules/subtitle_pipeline/generators/transformers_generic.py`

Generic HuggingFace ASR pipeline adapter. Any model on HuggingFace that does ASR can be used. `load()` loads the model to GPU, `generate()` runs inference, `unload()` frees VRAM.

### 10.3 VLLMTextGenerator

**New file**: `whisperjav/modules/subtitle_pipeline/generators/vllm_client.py`

HTTP client for vLLM-served audio-language models. `load()`/`unload()` are no-ops (server manages VRAM). `generate()` sends HTTP request to vLLM server.

### 10.4 Cross-Model Benchmarking Utility

CLI tool that runs multiple generator/aligner configurations on the same audio, compares against ground-truth SRT, reports per-scene accuracy metrics.

---

## 11. Testing Strategy

### 11.1 Per-Phase Testing

| Phase | Tests | What to Validate |
|---|---|---|
| 1 | Unit + integration | Context reaches generator; vad-grouped framer produces reasonable output; default mode is assembly |
| 2 | Integration | DecoupledPipeline constructs correctly from config; produces valid SRT; aligner=none works |
| 3 | Integration | Step-down triggers on collapsed scenes when enabled; skipped when disabled with clear warning; VRAM lifecycle preserved |
| 4 | Regression | No quality degradation after coupled mode removal; legacy mode names map correctly |

### 11.2 Equivalence Validation (Critical Gate for Phase 4)

Before deleting coupled mode code, run a comparison:
1. Process 3-5 audio files through `vad_slicing` mode (old)
2. Process same files through `assembly + vad-grouped + step-down` (new)
3. Compare: segment count, timing coverage, text quality, collapse rate
4. New pipeline must be equal or better on all metrics
5. **This validation gates Phase 4. No deletion until equivalence is demonstrated.**

### 11.3 Step-Down Opt-Out Testing (Phase 3)

Explicitly test the disabled path:
1. Run with `--no-step-down` on audio known to produce alignment collapse
2. Verify: warning is logged, proportional recovery is applied, no retry attempt
3. Run same audio without `--no-step-down`
4. Verify: step-down retry is attempted, result is equal or better

### 11.4 Continuous

- `pytest tests/` passes at every phase boundary
- `ruff check whisperjav/` passes
- Manual smoke test with known audio file

---

## 12. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Phase 3 step-down restructures orchestrator too aggressively | Regression in normal (non-step-down) path | Medium | Extract `_run_pass()` as pure refactor first (no logic change), verify tests pass, then add retry logic |
| R2 | Phase 4 deletion causes quality regression | Users see worse results | Medium | Extensive equivalence testing before deletion (Section 11.2). Phase 4 is gated on validation. |
| R3 | VadGroupedFramer in assembly produces different results than VAD_SLICING | User confusion during transition | Low | Expected — different code paths. Validate "comparable or better, not identical" |
| R4 | Ensemble system depends on coupled mode internals | Ensemble breaks after Phase 4 | Low | Audit `pass_worker.py` during Phase 4 |
| **R5** | **Effort spent on legacy coupled modes delays migration to the vision** | **Strategic: delays the product vision, wastes engineering time on dead-end code** | **High (if not managed)** | **This plan's core mitigation: zero effort on legacy fixes. All effort serves the decoupled pipeline. Phase 1 strangulation milestone (default switch) formalizes the commitment.** |

---

## 13. Decision Log

| # | Decision | Phase | Chosen | Rationale |
|---|---|---|---|---|
| D1 | Step-down retry approach | 3 | Batched retry | Preserves VRAM-efficient batch load/unload pattern. Per-scene loading too slow. |
| D2 | Generic entry point: new class vs extending QwenPipeline | 2 | New `DecoupledPipeline` class | Model-agnostic by design. QwenPipeline remains for backward compat but new work goes through DecoupledPipeline. |
| D3 | Deprecation period for coupled modes | 1 → 4 | **No grace period.** Default switches in Phase 1. Code deleted in Phase 4. | Legacy modes are architecturally constrained (integrated aligner limitation). Extended coexistence wastes maintenance effort and confuses users. |
| D4 | CLI UX for component selection | 2 | Both flat args and YAML | Flat args for common use (`--generator qwen3 --framer vad-grouped`). YAML for complex configurations. |
| D5 | Whether to keep CONTEXT_AWARE mode | 4 | **Delete it.** | CONTEXT_AWARE = assembly + full-scene framer. No unique capability. Legacy name maps to assembly config. |
| D6 | When Assembly becomes default | 1 | **Phase 1 exit.** | The decoupled pipeline is the only architecture that prevents aligner collapse (mid-pipeline cleaning). Users should be on it by default as early as possible. |
| D7 | Step-down user control | 3 | **Opt-out via `--no-step-down` / `--step-down-attempts 0`.** | Users may prefer speed over quality, or trust proportional recovery. The system must respect user choice. When disabled, orchestrator logs a clear warning on collapse. |
| D8 | Phase execution model | All | **Strictly sequential. No parallel workstreams.** | Parallel execution introduces integration conflicts, merge overhead, and debugging complexity. Sequential is safer for a stability-critical codebase. |
| D9 | Phase 2 vs Phase 3 order | 2, 3 | **Generic Entry Point first, Step-Down second.** | Entry Point is low-risk wiring that unlocks the vision immediately. Step-Down is complex orchestrator surgery. If Step-Down takes longer than expected, the vision's value is already shipping. |

---

## 14. Scope Boundaries

### In Scope
- Orchestrator feature completeness (context, framer flexibility, step-down with opt-out, diagnostics)
- Generic pipeline entry point (`DecoupledPipeline` class)
- Default mode switch to Assembly (Phase 1)
- CLI controls for step-down opt-out (Phase 3)
- Coupled mode code deletion (Phase 4)
- CLI changes for new parameters
- Ensemble default updates
- GUI updates (Phase 4)

### Out of Scope
- Changes to Phases 1-4 (audio extraction, scene detection, enhancement, VAD)
- Changes to Phases 6-9 (SRT generation, stitching, sanitization, analytics)
- Non-Qwen pipelines (faster, fast, balanced, pro, stable)
- `qwen_asr.py` internals (only external API used)
- Installer or build system changes
- New model backends (TransformersGenerator, VLLMGenerator) — these are "Future" items enabled by Phase 2

---

## 15. Dependencies and Prerequisites

### External Dependencies
- No new packages for Phases 1-4
- Future backends: `transformers` (existing optional dep), `httpx` (new for vLLM)

### Phase Dependencies

```
Phase 1 (Parity) ──→ Phase 2 (Entry Point) ──→ Phase 3 (Superiority) ──→ Phase 4 (Strangulation)
                     [low risk, vision win]     [complex, gates deletion]   [deletion + validation]
```

- **Phase 1**: No dependencies (starting point)
- **Phase 2**: Depends on Phase 1 (context propagation + framer selection needed for complete entry point)
- **Phase 3**: Depends on Phase 1 (framer flexibility needed for step-down). Does NOT depend on Phase 2 architecturally, but is sequenced after it per the no-parallel-execution mandate.
- **Phase 4**: Depends on Phase 3 (must be provably superior before deleting legacy)

**All phases are strictly sequential. No phase begins until the previous phase is complete, shipped, and stabilized.**

---

*End of IMPL-001 v4*
