# ADR-006 Implementation Plan: Decoupled Subtitle Pipeline

| Field       | Value                                        |
|-------------|----------------------------------------------|
| **Parent**  | ADR-006-decoupled-subtitle-pipeline.md       |
| **Status**  | COMPLETE (all 7 phases)                      |
| **Date**    | 2026-02-17                                   |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 0 — Foundation](#2-phase-0--foundation)
3. [Phase 1 — Qwen3 Adapters](#3-phase-1--qwen3-adapters)
4. [Phase 2 — TemporalFramer Backends](#4-phase-2--temporalframer-backends)
5. [Phase 3 — Orchestrator](#5-phase-3--orchestrator)
6. [Phase 4 — Integration](#6-phase-4--integration)
7. [Phase 5 — Audit Bug Fixes](#7-phase-5--audit-bug-fixes)
8. [Phase 6 — Dead Code Removal](#8-phase-6--dead-code-removal)
9. [Dependency Graph](#9-dependency-graph)
10. [Risk Assessment](#10-risk-assessment)

---

## 1. Overview

### 1.1 Guiding Principles

1. **Additive before subtractive** — Phases 0-3 create new code only. No existing code is
   modified. The current pipeline continues working throughout. Existing code is modified
   only in Phases 4-6.
2. **Each phase is independently verifiable** — Every phase has explicit verification
   criteria that can be checked before proceeding.
3. **Extract, don't rewrite** — Where possible, extract existing tested logic into new
   locations rather than reimplementing from scratch.
4. **Backward compatible** — The existing CLI/GUI/ensemble interfaces continue working
   identically. The new architecture is wired in transparently.

### 1.2 Phase Summary

| Phase | Name | Creates/Modifies | Risk | Dependencies |
|-------|------|-----------------|------|--------------|
| 0 | Foundation | 5 new files | Low | None | **COMPLETE** |
| 1 | Qwen3 Adapters | 4 new files | Low | Phase 0 | **COMPLETE** |
| 2 | TemporalFramer Backends | 7 new files | Low | Phase 0 | **COMPLETE** |
| 3 | Orchestrator | 1 new file | Medium | Phases 0, 1, 2 | **COMPLETE** |
| 4 | Integration | 2 modified files | Medium | Phase 3 | **COMPLETE** |
| 5 | Audit Bug Fixes | 2 modified files | Low | Phase 4 | **COMPLETE** |
| 6 | Dead Code Removal | 1 modified file | Low | Phase 5 | **COMPLETE** |

### 1.3 Files Created (new)

```
whisperjav/modules/subtitle_pipeline/
├── __init__.py                    (Phase 0)
├── types.py                      (Phase 0)
├── protocols.py                  (Phase 0)
├── hardening.py                  (Phase 0)
├── reconstruction.py             (Phase 0)
├── framers/
│   ├── __init__.py               (Phase 2)
│   ├── factory.py                (Phase 2)
│   ├── full_scene.py             (Phase 2)
│   ├── vad_grouped.py            (Phase 2)
│   ├── srt_source.py             (Phase 2)
│   └── manual.py                 (Phase 2)
├── generators/
│   ├── __init__.py               (Phase 1)
│   ├── factory.py                (Phase 1)
│   └── qwen3.py                  (Phase 1)
├── aligners/
│   ├── __init__.py               (Phase 1)
│   ├── factory.py                (Phase 1)
│   ├── qwen3.py                  (Phase 1)
│   └── none.py                   (Phase 1)
├── cleaners/
│   ├── __init__.py               (Phase 1)
│   ├── factory.py                (Phase 1)
│   ├── qwen3.py                  (Phase 1)
│   └── passthrough.py            (Phase 1)
└── orchestrator.py               (Phase 3)
```

**Total: 22 new files** (many are small: `__init__.py` files, factory registries,
thin adapters).

### 1.4 Files Modified (existing)

| File | Phase | Nature of Change |
|------|-------|------------------|
| `whisperjav/pipelines/qwen_pipeline.py` | 4, 6 | Wire orchestrator into assembly mode; remove dead code |
| `whisperjav/modules/qwen_asr.py` | 5 | Fix C1 (OOM stale closure), fix C2 (disable JapanesePostProcessor) |

---

## 2. Phase 0 — Foundation

**Goal**: Create the shared types, protocol definitions, and extracted utility functions
that all other phases depend on. No existing code is modified.

### 2.0 File: `__init__.py`

**Path**: `whisperjav/modules/subtitle_pipeline/__init__.py`

```python
"""
Decoupled Subtitle Pipeline — model-agnostic subtitle generation.

See ADR-006 for architectural documentation.
"""
```

Exports the key public types and protocols for downstream consumers.

### 2.1 File: `types.py`

**Path**: `whisperjav/modules/subtitle_pipeline/types.py`

**Contents** — all dataclasses, enums, and type aliases:

```python
@dataclass
class WordTimestamp:
    word: str
    start: float    # seconds, scene-relative
    end: float      # seconds, scene-relative

@dataclass
class TemporalFrame:
    start: float                        # seconds, scene-relative
    end: float                          # seconds, scene-relative
    text: Optional[str] = None          # pre-existing text (SRT, Whisper draft)
    confidence: Optional[float] = None
    source: str = ""                    # backend that produced this frame

@dataclass
class FramingResult:
    frames: List[TemporalFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranscriptionResult:
    text: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlignmentResult:
    words: List[WordTimestamp]
    metadata: Dict[str, Any] = field(default_factory=dict)

class TimestampMode(str, Enum):
    """Timestamp resolution strategy for hardening."""
    ALIGNER_WITH_INTERPOLATION = "aligner_interpolation"
    ALIGNER_WITH_VAD_FALLBACK = "aligner_vad_fallback"
    ALIGNER_ONLY = "aligner_only"
    VAD_ONLY = "vad_only"

@dataclass
class HardeningConfig:
    timestamp_mode: TimestampMode = TimestampMode.ALIGNER_WITH_INTERPOLATION
    scene_duration_sec: float = 0.0
    speech_regions: Optional[List[Tuple[float, float]]] = None

@dataclass
class HardeningDiagnostics:
    segment_count: int = 0
    interpolated_count: int = 0
    fallback_count: int = 0
    clamped_count: int = 0
    sorted: bool = False
```

**Design notes**:
- `TimestampMode` is a NEW enum in this module (not imported from `qwen_pipeline.py`).
  The old enum stays in `qwen_pipeline.py` during Phase 0-3. In Phase 4 integration,
  `qwen_pipeline.py` will import from here instead.
- All timestamps are in **scene-relative** coordinates (consistent with ADR-006 Section
  5.2). Group-relative coordinates are an internal concern of the `vad-grouped` framer.

### 2.2 File: `protocols.py`

**Path**: `whisperjav/modules/subtitle_pipeline/protocols.py`

**Contents** — all four Protocol definitions:

```python
@runtime_checkable
class TemporalFramer(Protocol):
    def frame(self, audio: np.ndarray, sample_rate: int,
              **kwargs) -> FramingResult: ...
    def cleanup(self) -> None: ...

@runtime_checkable
class TextGenerator(Protocol):
    def generate(self, audio_path: Path, language: str = "ja",
                 context: Optional[str] = None,
                 **kwargs) -> TranscriptionResult: ...
    def generate_batch(self, audio_paths: List[Path], language: str = "ja",
                       contexts: Optional[List[str]] = None,
                       **kwargs) -> List[TranscriptionResult]: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...

@runtime_checkable
class TextCleaner(Protocol):
    def clean(self, text: str, **kwargs) -> str: ...
    def clean_batch(self, texts: List[str], **kwargs) -> List[str]: ...

@runtime_checkable
class TextAligner(Protocol):
    def align(self, audio_path: Path, text: str, language: str = "ja",
              **kwargs) -> AlignmentResult: ...
    def align_batch(self, audio_paths: List[Path], texts: List[str],
                    language: str = "ja",
                    **kwargs) -> List[AlignmentResult]: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...
```

**Design notes**:
- `TextCleaner` has no `load()`/`unload()` — it's a lightweight text transform, no model.
- `TemporalFramer.frame()` takes `np.ndarray` audio (scene-level, already loaded). It does
  not manage file I/O — the orchestrator handles loading scene audio.
- `TextGenerator` and `TextAligner` take `Path` to audio files — they manage their own
  audio loading internally (consistent with existing QwenASR API which expects file paths).

### 2.3 File: `hardening.py`

**Path**: `whisperjav/modules/subtitle_pipeline/hardening.py`

**Purpose**: Extract and unify the timestamp resolution + boundary clamping + sorting logic
currently scattered across `qwen_pipeline.py`.

**Contents** — standalone functions (no class):

```python
def harden_scene_result(
    result: WhisperResult,
    config: HardeningConfig,
) -> HardeningDiagnostics:
    """
    Post-reconstruction hardening — shared by all pipeline paths.
    Mutates result in-place. Returns diagnostics.
    """
```

**Internal functions** (extracted from `qwen_pipeline.py`, converted from `@staticmethod`
to module-level):

| Function | Source | Lines | Extraction Notes |
|----------|--------|-------|------------------|
| `_apply_timestamp_interpolation(result, duration)` | `QwenPipeline._apply_timestamp_interpolation` | 1978-2109 | Direct extraction. Already a `@staticmethod` with no `self` dependencies. 132 lines. |
| `_apply_vad_timestamp_fallback(result, duration)` | `QwenPipeline._apply_vad_timestamp_fallback` | 1888-1950 | Direct extraction. Already a `@staticmethod`. 63 lines. |
| `_apply_vad_only_timestamps(result, duration)` | `QwenPipeline._apply_vad_only_timestamps` | 1952-1975 | Extract AND FIX audit H1: proportional distribution by character count instead of blanket `[0, group_duration]`. ~30 lines rewritten. |
| `_clamp_timestamps(result, max_duration)` | Inline at lines 1220-1227 and 1234-1242 | — | Extract the clamping pattern into a reusable function. Single implementation, called once (scene-relative). ~15 lines. |
| `_sort_segments_chronologically(result)` | Inline at line 1336 and 1367-1368 | — | Extract `result.segments.sort(key=lambda s: s.start)` plus word-level sort within segments. ~10 lines. |

**`harden_scene_result()` orchestration**:

```
1. Timestamp resolution (based on config.timestamp_mode):
   - ALIGNER_WITH_INTERPOLATION → _apply_timestamp_interpolation()
   - ALIGNER_WITH_VAD_FALLBACK  → _apply_vad_timestamp_fallback()
   - ALIGNER_ONLY               → no-op (keep whatever aligner produced)
   - VAD_ONLY                   → _apply_vad_only_timestamps()  [FIXED]
2. Boundary clamping → _clamp_timestamps(result, config.scene_duration_sec)
3. Chronological sort → _sort_segments_chronologically(result)
4. Diagnostics → count segments, record what was applied
```

**Verification criteria**:
- Unit test: Create a WhisperResult with known null-timestamp segments, run each
  timestamp mode, verify output timestamps are correct.
- Unit test: Create segments with timestamps > scene_duration, verify clamping.
- Unit test: Create out-of-order segments, verify sorting.
- Unit test: H1 regression test — `VAD_ONLY` with 3 segments produces 3 DIFFERENT
  time ranges (not all identical).

### 2.4 File: `reconstruction.py`

**Path**: `whisperjav/modules/subtitle_pipeline/reconstruction.py`

**Purpose**: Extract `_reconstruct_from_words()` from `qwen_pipeline.py:1846-1882` into a
standalone function.

**Contents**:

```python
def reconstruct_from_words(
    words: List[Dict[str, Any]],
    audio_path: Union[str, Path],
    suppress_silence: bool = True,
) -> WhisperResult:
    """
    Reconstruct a WhisperResult from word dicts via stable-ts transcribe_any.

    Args:
        words: List of {'word': str, 'start': float, 'end': float} dicts.
        audio_path: Path to audio file (needed by transcribe_any for duration).
        suppress_silence: Whether stable-ts should adjust timestamps based on
            silence detection. Set False for sentinel-recovered words to avoid
            undoing the recovery. (Addresses audit H3.)
    """
```

**Key design decision (audit H3 resolution)**:

The `suppress_silence` parameter defaults to `True` (existing behavior for normal words)
but the orchestrator passes `False` for sentinel-recovered results. This is the design
decision documented in ADR-006 Section 10, resolving H3 ("suppress_silence may alter
sentinel recovery timestamps").

**Verification criteria**:
- Unit test: Provide known word dicts, verify WhisperResult has correct segments after
  stable-ts regrouping.
- Unit test: Verify `suppress_silence=False` does NOT modify pre-computed timestamps.

---

## 3. Phase 1 — Qwen3 Adapters

**Goal**: Wrap existing `QwenASR` methods behind the new protocols. No new functionality —
pure structural extraction. No existing code modified.

### 3.1 Generators

#### 3.1.1 File: `generators/__init__.py`

Exports `TextGeneratorFactory`.

#### 3.1.2 File: `generators/factory.py`

```python
class TextGeneratorFactory:
    _registry = {
        "qwen3": "whisperjav.modules.subtitle_pipeline.generators.qwen3.Qwen3TextGenerator",
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> TextGenerator: ...

    @classmethod
    def available(cls) -> List[str]: ...
```

Follows the same lazy-import factory pattern as `SpeechSegmenterFactory`.

#### 3.1.3 File: `generators/qwen3.py`

**Class**: `Qwen3TextGenerator`

```python
class Qwen3TextGenerator:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: str = "Japanese",
        repetition_penalty: float = 1.1,
        max_tokens_per_audio_second: float = 20.0,
        attn_implementation: str = "auto",
    ): ...
```

**Method mapping**:

| Protocol Method | Implementation |
|----------------|----------------|
| `load()` | Creates `QwenASR` instance with stored config, calls `asr.load_model_text_only()` |
| `unload()` | Calls `asr.unload_model()`, runs `safe_cuda_cleanup()`, sets `asr = None` |
| `generate(audio_path, language, context)` | Calls `asr.transcribe_text_only(audio_path, context, language)`, wraps result in `TranscriptionResult` |
| `generate_batch(audio_paths, language, contexts)` | Calls `asr.transcribe_text_only(audio_paths, contexts, language)` (batch API), wraps results |
| `cleanup()` | Calls `unload()` if loaded |

**Lifecycle**: `load()` and `unload()` are explicit. The adapter does NOT keep a persistent
QwenASR instance — it creates one in `load()` and destroys it in `unload()`. This prevents
the stale closure problem (audit C1) by design.

**Batch size**: Constructor accepts `batch_size` independently. This resolves audit M3 —
the generator can be configured with a higher batch_size than the coupled mode since it
only loads the text-generation model (lower VRAM).

**VRAM cleanup**: `unload()` uses `safe_cuda_cleanup()` from `whisperjav.utils.gpu_utils`,
resolving audit M5 (redundant inline cleanup).

### 3.2 Aligners

#### 3.2.1 File: `aligners/__init__.py`

Exports `TextAlignerFactory`.

#### 3.2.2 File: `aligners/factory.py`

```python
class TextAlignerFactory:
    _registry = {
        "qwen3": "...aligners.qwen3.Qwen3ForcedAlignerAdapter",
        "none": "...aligners.none.NoneAligner",
    }
```

#### 3.2.3 File: `aligners/qwen3.py`

**Class**: `Qwen3ForcedAlignerAdapter`

```python
class Qwen3ForcedAlignerAdapter:
    def __init__(
        self,
        aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "Japanese",
    ): ...
```

**Method mapping**:

| Protocol Method | Implementation |
|----------------|----------------|
| `load()` | Creates `QwenASR` instance, calls `asr.load_aligner_only()` |
| `unload()` | Calls `asr.unload_model()`, runs `safe_cuda_cleanup()`, sets `asr = None` |
| `align(audio_path, text, language)` | Calls `asr.align_standalone(audio_path, text, language)`, runs `merge_master_with_timestamps()`, wraps in `AlignmentResult` |
| `align_batch(audio_paths, texts, language)` | Calls `asr.align_standalone(audio_paths, texts, language)` (batch), merges each, wraps results |
| `cleanup()` | Calls `unload()` if loaded |

**Critical detail — merge_master_with_timestamps()**:

The `merge_master_with_timestamps()` function (module-level in `qwen_asr.py`) reconciles
ASR text (with punctuation) and aligner timestamps (without punctuation). This function
is **Qwen3-specific** — it understands Qwen3's particular text format.

The adapter calls it internally after `align_standalone()` returns raw aligner output. The
`AlignmentResult.words` contains the merged word dicts. Downstream consumers don't need to
know about the merge step.

#### 3.2.4 File: `aligners/none.py`

**Class**: `NoneAligner`

A no-op aligner that returns empty alignment results. Used in aligner-free workflows where
timestamps come entirely from the TemporalFramer.

```python
class NoneAligner:
    def align(self, audio_path, text, language="ja", **kwargs):
        return AlignmentResult(words=[], metadata={"skipped": True})

    def align_batch(self, audio_paths, texts, language="ja", **kwargs):
        return [self.align(p, t, language) for p, t in zip(audio_paths, texts)]

    def load(self): pass
    def unload(self): pass
    def cleanup(self): pass
```

### 3.3 Cleaners

#### 3.3.1 File: `cleaners/__init__.py`

Exports `TextCleanerFactory`.

#### 3.3.2 File: `cleaners/factory.py`

```python
class TextCleanerFactory:
    _registry = {
        "qwen3": "...cleaners.qwen3.Qwen3TextCleaner",
        "passthrough": "...cleaners.passthrough.PassthroughCleaner",
    }
```

#### 3.3.3 File: `cleaners/qwen3.py`

**Class**: `Qwen3TextCleaner`

Thin wrapper around the existing `AssemblyTextCleaner`. Does NOT reimplement — delegates.

```python
class Qwen3TextCleaner:
    def __init__(self, config: Optional[AssemblyCleanerConfig] = None):
        self._cleaner = AssemblyTextCleaner(config)

    def clean(self, text: str, **kwargs) -> str:
        return self._cleaner.clean(text)

    def clean_batch(self, texts: List[str], **kwargs) -> List[str]:
        return self._cleaner.clean_batch(texts)
```

#### 3.3.4 File: `cleaners/passthrough.py`

**Class**: `PassthroughCleaner`

No-op cleaner for models whose output needs no cleaning.

```python
class PassthroughCleaner:
    def clean(self, text, **kwargs): return text
    def clean_batch(self, texts, **kwargs): return texts
```

### 3.4 Phase 1 Verification

- Unit test: `Qwen3TextGenerator` instantiation with various configs (no model load).
- Unit test: `Qwen3ForcedAlignerAdapter` instantiation.
- Unit test: `NoneAligner` returns correct empty results.
- Unit test: `Qwen3TextCleaner` delegates to `AssemblyTextCleaner` correctly.
- Unit test: `PassthroughCleaner` returns input unchanged.
- Unit test: All classes satisfy their respective protocol via `isinstance()` check.
- Integration test (requires GPU + model): `Qwen3TextGenerator.load()` → `.generate()` →
  `.unload()` → verify text output matches `QwenASR.transcribe_text_only()` output.

---

## 4. Phase 2 — TemporalFramer Backends

**Goal**: Create all TemporalFramer implementations. No existing code modified.

### 4.1 Factory

#### 4.1.1 File: `framers/__init__.py`

Exports `TemporalFramerFactory`.

#### 4.1.2 File: `framers/factory.py`

```python
class TemporalFramerFactory:
    _registry = {
        "full-scene": "...framers.full_scene.FullSceneFramer",
        "vad-grouped": "...framers.vad_grouped.VadGroupedFramer",
        "srt-source": "...framers.srt_source.SrtSourceFramer",
        "manual": "...framers.manual.ManualFramer",
    }
```

**Note**: `whisper-segment` is NOT included in the initial implementation. It requires
loading a Whisper model which adds complexity (VRAM management, model download). It will
be added in a future phase when the core architecture is proven. The factory registry is
extensible — adding it later is a single file + registry entry.

### 4.2 File: `framers/full_scene.py`

**Class**: `FullSceneFramer`

The simplest framer — returns one frame spanning the entire scene.

```python
class FullSceneFramer:
    def frame(self, audio, sample_rate, **kwargs):
        duration = len(audio) / sample_rate
        return FramingResult(
            frames=[TemporalFrame(start=0.0, end=duration, source="full-scene")],
            metadata={"strategy": "full-scene", "frame_count": 1},
        )

    def cleanup(self): pass
```

This is what current ASSEMBLY and CONTEXT_AWARE modes do implicitly — they process the
full scene as one unit.

### 4.3 File: `framers/vad_grouped.py`

**Class**: `VadGroupedFramer`

Wraps the existing `SpeechSegmenterFactory` to produce dialogue-level temporal frames.
This is what current VAD_SLICING does — run VAD, group speech regions into chunks.

```python
class VadGroupedFramer:
    def __init__(
        self,
        segmenter_backend: str = "ten",
        max_group_duration_s: float = 29.0,
        chunk_threshold_s: float = 1.0,
        min_frame_duration_s: float = 0.1,
        segmenter_config: Optional[Dict] = None,
    ): ...

    def frame(self, audio, sample_rate, **kwargs):
        """
        1. Create SpeechSegmenter via SpeechSegmenterFactory
        2. Detect speech regions → SegmentationResult
        3. Use SegmentationResult.groups (pre-grouped by segmenter)
        4. Filter groups below min_frame_duration_s
        5. Convert each group to TemporalFrame
        6. Return FramingResult with speech_regions in metadata
        """
```

**Key detail — speech_regions in metadata**:

The `FramingResult.metadata["speech_regions"]` stores the raw speech regions (per frame)
so the orchestrator can pass them to the sentinel for Strategy C (VAD-guided) recovery.
This is how the current VAD_SLICING passes speech regions to the sentinel.

**Design note — min_frame_duration_s**:

This implements the min duration guard from VAD_SLICING (audit comparison row M). Groups
shorter than 0.1s are skipped, preventing wasteful transcription of near-empty clips.

### 4.4 File: `framers/srt_source.py`

**Class**: `SrtSourceFramer`

Parses an existing SRT file and uses its timestamps as temporal frames.

```python
class SrtSourceFramer:
    def __init__(
        self,
        srt_path: Union[str, Path],
        keep_text: bool = False,
        min_frame_duration_s: float = 0.1,
    ): ...

    def frame(self, audio, sample_rate, **kwargs):
        """
        1. Parse SRT file using pysrt (already a dependency)
        2. For each subtitle entry:
           - Create TemporalFrame(start, end, text=entry.text if keep_text)
        3. Filter entries below min_frame_duration_s
        4. Return FramingResult
        """
```

**When `keep_text=True`**: The frame's `text` field contains the SRT entry's text. The
orchestrator can use this as context for the TextGenerator, or skip generation entirely
if only re-timing is needed.

### 4.5 File: `framers/manual.py`

**Class**: `ManualFramer`

Accepts user-provided timestamps directly. Useful for testing and debugging.

```python
class ManualFramer:
    def __init__(self, timestamps: List[Tuple[float, float]]): ...

    def frame(self, audio, sample_rate, **kwargs):
        return FramingResult(
            frames=[TemporalFrame(s, e, source="manual") for s, e in self._timestamps],
            metadata={"strategy": "manual"},
        )

    def cleanup(self): pass
```

### 4.6 Phase 2 Verification

- Unit test: `FullSceneFramer` returns exactly one frame spanning audio duration.
- Unit test: `VadGroupedFramer` with mock audio produces correct frame groupings.
- Unit test: `SrtSourceFramer` correctly parses a sample SRT file.
- Unit test: `ManualFramer` returns exact timestamps provided.
- Unit test: `VadGroupedFramer` filters out frames below `min_frame_duration_s`.
- Unit test: All framers satisfy the `TemporalFramer` protocol via `isinstance()` check.
- Integration test: `VadGroupedFramer` with real TEN segmenter produces frames that match
  current VAD_SLICING group boundaries.

---

## 5. Phase 3 — Orchestrator

**Goal**: Build the `DecoupledSubtitlePipeline` that composes all protocol components into
a working pipeline. This is the most complex phase.

### 5.1 File: `orchestrator.py`

**Path**: `whisperjav/modules/subtitle_pipeline/orchestrator.py`

**Class**: `DecoupledSubtitlePipeline`

```python
class DecoupledSubtitlePipeline:
    def __init__(
        self,
        framer: TemporalFramer,
        generator: TextGenerator,
        cleaner: TextCleaner,
        aligner: Optional[TextAligner],   # None = aligner-free workflow
        hardening_config: HardeningConfig,
        artifacts_dir: Optional[Path] = None,  # debug artifact output
    ): ...

    def process_scenes(
        self,
        scene_audio_paths: List[Path],
        scene_durations: List[float],
        scene_speech_regions: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> List[Tuple[Optional[WhisperResult], Dict[str, Any]]]:
        """
        Process all scenes through the decoupled pipeline.

        Returns list of (result, diagnostics) per scene.
        Result is None if scene processing failed.
        """
```

### 5.2 Internal Flow — Detailed

The `process_scenes()` method implements the 9-step flow from ADR-006 Section 9.2. Below
is the detailed logic for each step.

#### Step 1: Temporal Framing

```
For each scene:
    Load scene audio as np.ndarray (via librosa/soundfile)
    frames = framer.frame(audio, sample_rate)
    Store: scene_frames[scene_idx] = frames.frames
    Store: frame_speech_regions from frames.metadata (if available)
```

**If framer produces text** (e.g., `srt-source` with `keep_text=True`): Store frame texts.
These will be used in Step 3 instead of running the generator.

**Audio slicing**: For each frame, slice the scene audio to `[frame.start, frame.end]` and
write a temporary WAV file. The TextGenerator and TextAligner expect file paths. Store
temp file paths for cleanup.

#### Step 2-4: Text Generation + Cleaning

```
generator.load()

For each scene:
    For each frame in scene_frames[scene_idx]:
        IF frame.text is not None:
            raw_text = frame.text  (use framer-provided text)
        ELSE:
            raw_text = generator.generate(frame_audio_path, language, context)
        Store raw_text

generator.unload()
safe_cuda_cleanup()

For each scene:
    clean_texts = cleaner.clean_batch(raw_texts)
```

**Error handling for generation**:
- If `generate()` raises an exception for a single frame, log the error and set the frame's
  text to empty string. The frame will produce an empty subtitle (which the SRT stitcher
  will skip). Other frames continue processing.
- If `generate_batch()` raises (batch-level failure), fall back to per-frame `generate()`
  with individual error handling.

**Batch vs per-frame generation**:
- If the generator supports batch (Qwen3 does), use `generate_batch()` per scene for
  efficiency.
- If there's only one frame per scene (full-scene framer), `generate()` and
  `generate_batch()` are equivalent.

#### Step 5-8: Alignment (if aligner is not None)

```
IF aligner is not None:
    aligner.load()

    For each scene:
        For each frame:
            alignment = aligner.align(frame_audio_path, clean_text, language)
            Store alignment.words

    aligner.unload()
    safe_cuda_cleanup()
```

**If aligner is None** (aligner-free workflow):
- Skip Steps 5-8.
- In Step 9, word timestamps are derived from frame boundaries: each frame becomes one
  segment with `start=frame.start`, `end=frame.end`, `text=clean_text`.

#### Step 9: Per-Scene Reconstruction + Sentinel + Hardening

```
For each scene:
    TRY:
        IF aligner was used:
            # Merge aligned words from all frames in this scene
            all_words = []
            for frame_idx, frame in enumerate(scene_frames):
                frame_words = alignments[scene_idx][frame_idx].words
                # Offset word timestamps from frame-relative to scene-relative
                for w in frame_words:
                    w.start += frame.start
                    w.end += frame.start
                all_words.extend(frame_words)

            # Sentinel detection
            assessment = assess_alignment_quality(all_words, scene_duration)

            # Recovery if collapsed
            IF assessment["status"] == "COLLAPSED":
                speech_regions = scene_speech_regions[scene_idx] (if available)
                corrected_words = redistribute_collapsed_words(
                    all_words, scene_duration, speech_regions
                )
                # Reconstruct with suppress_silence=False (H3 fix)
                result = reconstruct_from_words(
                    corrected_words, scene_audio_path, suppress_silence=False
                )
            ELSE:
                result = reconstruct_from_words(
                    all_words, scene_audio_path, suppress_silence=True
                )

        ELSE (aligner-free):
            # Build word dicts directly from frames
            words = []
            for frame, text in zip(frames, clean_texts):
                words.append({
                    "word": text,
                    "start": frame.start,
                    "end": frame.end,
                })
            result = reconstruct_from_words(words, scene_audio_path)

        # Hardening (shared, all paths)
        hardening_diag = harden_scene_result(result, hardening_config)

        # Emit diagnostics
        diagnostics = {
            "scene_idx": scene_idx,
            "frame_count": len(frames),
            "sentinel_status": assessment["status"] if aligner else "N/A",
            "hardening": asdict(hardening_diag),
            "segment_count": len(result.segments) if result else 0,
        }

        results.append((result, diagnostics))

    EXCEPT Exception as e:
        logger.error(f"Scene {scene_idx} failed: {e}")
        results.append((None, {"scene_idx": scene_idx, "error": str(e)}))
```

#### Temp File Cleanup

After all scenes are processed:
```
For each temp WAV file created in Step 1:
    Delete temp file (unless keep_temp_files is True)
```

### 5.3 Sentinel Integration — One Place

The sentinel functions (`assess_alignment_quality`, `redistribute_collapsed_words`) from
`alignment_sentinel.py` are called in Step 9 only. They are NOT called inside any adapter
or framer — they are an orchestrator concern.

**Strategy selection**:
- If `scene_speech_regions` is provided (from VadGroupedFramer metadata or passed by
  caller): Use Strategy C (VAD-guided redistribution).
- If not: Use Strategy B (proportional from anchor).

This is consistent with current behavior and ADR-006 Appendix A.

### 5.4 Sentinel Stats Accumulation

The orchestrator maintains a `sentinel_stats` dict that accumulates across all scenes:

```python
self.sentinel_stats = {
    "total_scenes": 0,
    "collapsed_scenes": 0,
    "recovered_scenes": 0,
    "recovery_strategies": {"vad_guided": 0, "proportional": 0},
}
```

This is returned as part of the overall processing result, consistent with current
`self._sentinel_stats` in `qwen_pipeline.py`.

### 5.5 Debug Artifacts

When `artifacts_dir` is set (via `--debug` or `--keep-temp`), the orchestrator saves:
- Per-scene: `{basename}_scene{idx}_raw.txt` (raw transcription)
- Per-scene: `{basename}_scene{idx}_clean.txt` (cleaned transcription)
- Per-scene: `{basename}_scene{idx}_merged.json` (merged word dicts with timestamps)
- Per-scene: `{basename}_scene{idx}_diag.json` (diagnostics)

This provides richer mid-pipeline visibility (audit comparison row R — assembly advantage).

### 5.6 Phase 3 Verification

- Unit test with mock components: Verify 9-step flow runs correctly with mock generator,
  cleaner, aligner, framer.
- Unit test: Verify aligner-free workflow (aligner=None) produces correct results.
- Unit test: Verify sentinel detection and recovery are called correctly.
- Unit test: Verify VRAM swap (unload before load) happens in correct order.
- Unit test: Verify per-scene error handling (one scene fails, others continue).
- Unit test: Verify diagnostics contain correct fields.
- Integration test (requires GPU + model): Run full pipeline with Qwen3TextGenerator +
  Qwen3ForcedAlignerAdapter + FullSceneFramer on real audio. Compare output with current
  `_phase5_assembly()` output.

---

## 6. Phase 4 — Integration

**Goal**: Wire the orchestrator into `QwenPipeline` for assembly mode. This is the first
phase that modifies existing code.

### 6.1 Modify: `qwen_pipeline.py`

#### 6.1.1 Add Import

```python
from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import (
    TimestampMode as NewTimestampMode, HardeningConfig,
)
from whisperjav.modules.subtitle_pipeline.generators.factory import TextGeneratorFactory
from whisperjav.modules.subtitle_pipeline.aligners.factory import TextAlignerFactory
from whisperjav.modules.subtitle_pipeline.cleaners.factory import TextCleanerFactory
from whisperjav.modules.subtitle_pipeline.framers.factory import TemporalFramerFactory
```

#### 6.1.2 Modify `__init__()` — Construct Orchestrator

In the `__init__()` method, after existing parameter processing, construct the orchestrator
**for assembly mode only** (other modes continue as-is initially):

```python
if self.input_mode == InputMode.ASSEMBLY:
    framer = TemporalFramerFactory.create("full-scene")
    generator_kwargs = {
        "model_id": self.model_id,
        "device": self.device,
        "dtype": self.dtype,
        "batch_size": self.batch_size,
        "max_new_tokens": self.max_new_tokens,
        "language": self.language,
        "repetition_penalty": self._asr_config.get("repetition_penalty", 1.1),
        "max_tokens_per_audio_second": self._asr_config.get(
            "max_tokens_per_audio_second", 20.0
        ),
        "attn_implementation": self.attn_implementation,
    }
    generator = TextGeneratorFactory.create("qwen3", **generator_kwargs)
    cleaner = TextCleanerFactory.create("qwen3")
    aligner = TextAlignerFactory.create("qwen3",
        aligner_id=self.aligner_id,
        device=self.device,
        dtype=self.dtype,
        language=self.language,
    )
    self._subtitle_pipeline = DecoupledSubtitlePipeline(
        framer=framer,
        generator=generator,
        cleaner=cleaner,
        aligner=aligner,
        hardening_config=HardeningConfig(
            timestamp_mode=NewTimestampMode(self.timestamp_mode.value),
        ),
        artifacts_dir=self.temp_dir / "raw_subs" if self.save_metadata_json else None,
    )
```

#### 6.1.3 Modify `process()` — Phase 5 Assembly Path

Replace the `_phase5_assembly()` call with the orchestrator:

```python
# Phase 5: ASR Transcription (Assembly mode via orchestrator)
if self.input_mode == InputMode.ASSEMBLY:
    scene_results = self._subtitle_pipeline.process_scenes(
        scene_audio_paths=scene_paths,
        scene_durations=scene_durations,
        scene_speech_regions=speech_regions_per_scene,
    )
    # Unpack results into the format expected by Phase 6
    for scene_idx, (result, diag) in enumerate(scene_results):
        if result is not None:
            # ... existing Phase 6 SRT generation code ...
```

The Phase 6/7/8 code remains unchanged — it receives `WhisperResult` objects, same as
before.

#### 6.1.4 Cleanup

Add orchestrator cleanup to `QwenPipeline.cleanup()`:

```python
if hasattr(self, '_subtitle_pipeline') and self._subtitle_pipeline:
    self._subtitle_pipeline.cleanup()
    self._subtitle_pipeline = None
```

### 6.2 Backward Compatibility

- **CLI/GUI unchanged**: Same parameters, same flags. The `--qwen-input-mode assembly`
  flag triggers the orchestrator path. Other modes continue using existing code.
- **Ensemble unchanged**: `pass_worker.py` constructs `QwenPipeline` with the same
  parameters. The orchestrator is constructed internally.
- **Output format unchanged**: `WhisperResult` → SRT → same output files.

### 6.3 Phase 4 Verification

- Smoke test: Run `whisperjav video.mp4 --mode qwen --qwen-input-mode assembly` and verify
  output SRT is produced.
- Regression test: Process the same audio file with old code (before Phase 4) and new code
  (after Phase 4). Compare output SRTs — they should be substantively similar (exact match
  not expected due to H1/H2/H3 fixes, but text content should match).
- Verify: `--timestamp-mode` flag now works for assembly mode (was silently ignored before).
- Verify: Debug artifacts are created when `--debug` is passed.
- Verify: Sentinel stats are correctly accumulated and reported.
- Verify: `whisperjav-gui` ensemble mode with Qwen assembly pass works correctly.

---

## 7. Phase 5 — Audit Bug Fixes

**Goal**: Fix the remaining audit findings that weren't resolved as side-effects of the
architecture change.

### 7.1 Modify: `qwen_asr.py`

#### 7.1.1 Fix C1: OOM Retry Stale Closure

**Location**: `qwen_asr.py:784, 977-997`

**Current bug**: The `qwen_inference` closure at line 784 captures `qwen_model` by
reference. The OOM retry at line 983-984 unloads and reloads the model, but the closure
still references the old model.

**Fix**: In the `except torch.cuda.OutOfMemoryError` block, reassign `qwen_model` after
reloading:

```python
# After self.load_model() at line 984:
qwen_model = self.model  # Update closure's reference to new model
```

This ensures the retry uses the new model with the halved batch_size.

**Alternative fix (stronger)**: Have `qwen_inference` reference `self.model` instead of
the local `qwen_model` capture. This eliminates the closure capture entirely. However, this
changes the closure semantics (it becomes late-binding), which needs careful analysis of
whether anything else mutates `self.model` during the transcribe_any call.

**Recommendation**: Use the simple reassignment fix. It's targeted, minimal, and verifiable.

#### 7.1.2 Fix C2 (revised): Disable JapanesePostProcessor for Qwen Pipeline

**Location**: `qwen_asr.py:964-973`

**Current bug**: Coupled modes (CONTEXT_AWARE, VAD_SLICING) apply the Whisper-era
`JapanesePostProcessor` to Qwen3-ASR output via `self._postprocessor.process()`.

**Fix**: The `QwenPipeline` already passes `japanese_postprocess=True/False` to `QwenASR`.
The fix is in `QwenPipeline.__init__()` — set it to `False`:

```python
# In qwen_pipeline.py, QwenPipeline.__init__(), _asr_config construction:
"japanese_postprocess": False,  # Qwen3 uses AssemblyTextCleaner, not JapanesePostProcessor
```

This disables the Whisper-era post-processor for ALL Qwen input modes (assembly already
skips it; this fixes context_aware and vad_slicing).

**Verification**: Run VAD_SLICING mode and verify `JapanesePostProcessor.process()` is NOT
called (add a debug log or trace).

### 7.2 Fix M1: Stale Comment

**Location**: `qwen_pipeline.py:217`

Change `300s` → `180s` in comment.

### 7.3 Phase 5 Verification

- Unit test: C1 regression — Mock OOM scenario, verify retry uses new model reference.
- Integration test: Run CONTEXT_AWARE and VAD_SLICING modes, verify output does NOT
  contain JapanesePostProcessor artifacts (e.g., specific merge patterns).
- Code review: Verify M1 comment is corrected.

---

## 8. Phase 6 — Dead Code Removal

**Goal**: Clean up dead code identified in the audit. Low risk — all verified as unreachable.

### 8.1 Modify: `qwen_pipeline.py`

| Audit ID | What to Remove | Lines | Notes |
|----------|---------------|-------|-------|
| L1 | `_apply_vad_filter()` method | 2116-2177 | Deprecated, never called. 62 lines. |
| L2 | `cross_scene_context` branches | 249, 593-599, 782-784, 1489-1491 | Flag hardcoded `False`. All branches dead. |
| L2 | `self.cross_scene_context` attribute | 249 | Remove the attribute itself. |
| L2 | `previous_tail` accumulation | 593-601 | Dead code inside dead branch. |

**Do NOT remove** in Phase 6:
- The old `_phase5_assembly()` method — keep for reference/fallback until the orchestrator
  is proven stable over several releases. Mark as `# DEPRECATED: use DecoupledSubtitlePipeline`.
- The old `_transcribe_speech_regions()` / `_transcribe_group()` — these are still used by
  VAD_SLICING and CONTEXT_AWARE modes.
- The `TimestampMode` enum in `qwen_pipeline.py` — still referenced by the old code paths.

### 8.2 Phase 6 Verification

- `ruff check whisperjav/pipelines/qwen_pipeline.py` — no lint errors.
- Run full test suite — no regressions.
- `git diff --stat` — verify only removals, no functional changes.

---

## 9. Dependency Graph

```
Phase 0 (Foundation)
  ├── types.py
  ├── protocols.py
  ├── hardening.py      ← depends on types.py
  └── reconstruction.py ← depends on types.py
        │
        ├──────────────────────────┐
        ▼                          ▼
Phase 1 (Adapters)           Phase 2 (Framers)
  ├── generators/qwen3.py     ├── full_scene.py
  ├── aligners/qwen3.py       ├── vad_grouped.py
  ├── aligners/none.py        ├── srt_source.py
  ├── cleaners/qwen3.py       └── manual.py
  └── cleaners/passthrough.py
        │                          │
        └──────────┬───────────────┘
                   ▼
Phase 3 (Orchestrator)
  └── orchestrator.py
        │
        ▼
Phase 4 (Integration)        ← FIRST phase that modifies existing code
  └── qwen_pipeline.py (modified)
        │
        ▼
Phase 5 (Bug Fixes)
  ├── qwen_asr.py (modified: C1, C2)
  └── qwen_pipeline.py (modified: M1 comment)
        │
        ▼
Phase 6 (Dead Code Removal)
  └── qwen_pipeline.py (modified: remove L1, L2)
```

**Phases 1 and 2 are independent** — they can be implemented in either order or even
concurrently. Both depend only on Phase 0.

---

## 10. Risk Assessment

### 10.1 Per-Phase Risk

| Phase | Risk Level | Key Risk | Mitigation |
|-------|-----------|----------|------------|
| 0 | **Low** | Hardening logic extraction introduces subtle behavioral differences | Extract as literal copy first, add H1 fix separately with explicit test |
| 1 | **Low** | Adapters introduce lifecycle bugs (load/unload ordering) | Each adapter manages its own QwenASR instance — no shared state |
| 2 | **Low** | VadGroupedFramer diverges from current VAD_SLICING grouping | Integration test: compare frame boundaries with current group boundaries |
| 3 | **Medium** | Orchestrator VRAM management fails on edge cases | Explicit try/finally for unload(); defensive cleanup in error paths |
| 3 | **Medium** | Sentinel integration produces different results than current code | Side-by-side comparison test with current pipeline on same audio |
| 4 | **Medium** | Integration breaks existing ensemble/CLI workflows | Smoke test every entry point (CLI, GUI, ensemble) after integration |
| 5 | **Low** | C1 fix may have secondary effects on OOM recovery | The fix is targeted (single variable reassignment); existing OOM tests still pass |
| 6 | **Low** | Dead code removal accidentally removes live code | Each removal verified by audit as unreachable; run full test suite |

### 10.2 Rollback Strategy

- **Phases 0-3**: Pure additive code. Rollback = delete the new package directory.
- **Phase 4**: Rollback = revert the two modified sections in `qwen_pipeline.py`. The old
  `_phase5_assembly()` method is preserved (not deleted), so reactivating it is trivial.
- **Phases 5-6**: Each is a small, independent commit. Standard `git revert`.

### 10.3 What Is Explicitly Out of Scope

| Item | Why Deferred | When |
|------|-------------|------|
| `whisper-segment` TemporalFramer | Requires Whisper model VRAM management; proves architecture first | After core architecture is stable |
| `VLLMGenerator` | vLLM on Windows is not stable (per ADR-005) | When vLLM Windows support matures |
| `CTCAligner` | Requires wav2vec2 model and CTC segmentation tooling | After Qwen3 aligner alternatives are explored |
| `HybridAligner` | Composition pattern; needs at least 2 working aligners | After CTCAligner is implemented |
| `WhisperTextGenerator` | `whisper-segment` framer covers the main Whisper-as-timing use case | After core pipeline is proven |
| `TransformersGenerator` | Generic HuggingFace adapter; low priority vs Qwen3 | After core pipeline is proven |
| `VADProportionalAligner` | Aligner-free path in orchestrator Step 9 covers the core logic inline; dedicated backend adds marginal value | After core architecture is stable |
| Migrating CONTEXT_AWARE to orchestrator | Coupled mode has different semantics; needs design work | Future ADR |
| Migrating VAD_SLICING to orchestrator | Step-down logic doesn't fit cleanly; needs design work | Future ADR |
| Diagnostic/Benchmarking utility | Depends on stable protocol contracts | After Phase 4 integration proven |

---

## Appendix A: Commit Strategy

Each phase should be a single commit (or a small number of logically grouped commits if the
phase is large). Suggested commit messages:

```
Phase 0: "Add subtitle_pipeline foundation: types, protocols, hardening, reconstruction"
Phase 1: "Add Qwen3 protocol adapters: generator, aligner, cleaner"
Phase 2: "Add TemporalFramer backends: full-scene, vad-grouped, srt-source, manual"
Phase 3: "Add DecoupledSubtitlePipeline orchestrator"
Phase 4: "Wire orchestrator into QwenPipeline assembly mode"
Phase 5: "Fix C1 (OOM stale closure) and C2 (wrong post-processor) in Qwen ASR"
Phase 6: "Remove dead code: _apply_vad_filter, cross_scene_context branches"
```

## Appendix B: Lines of Code Estimates

| Phase | New Lines | Modified Lines | Deleted Lines |
|-------|-----------|---------------|---------------|
| 0 | ~450 | 0 | 0 |
| 1 | ~350 | 0 | 0 |
| 2 | ~300 | 0 | 0 |
| 3 | ~400 | 0 | 0 |
| 4 | ~60 | ~30 (replaced) | 0 |
| 5 | ~5 | ~10 | 0 |
| 6 | 0 | 0 | ~100 |
| **Total** | **~1565** | **~40** | **~100** |

The bulk of the work is new code (Phases 0-3). Existing code modifications are minimal
(Phases 4-6).

## Appendix C: Test Files

```
tests/
├── test_subtitle_pipeline/
│   ├── test_types.py              (Phase 0)
│   ├── test_hardening.py          (Phase 0)
│   ├── test_reconstruction.py     (Phase 0)
│   ├── test_generators.py         (Phase 1)
│   ├── test_aligners.py           (Phase 1)
│   ├── test_cleaners.py           (Phase 1)
│   ├── test_framers.py            (Phase 2)
│   ├── test_orchestrator.py       (Phase 3)
│   └── test_integration.py        (Phase 4)
```

---

*End of Implementation Plan*
