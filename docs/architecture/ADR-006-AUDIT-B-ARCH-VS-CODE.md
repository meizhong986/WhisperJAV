# ADR-006 Audit B: Architecture vs. Implementation Code Gap Analysis

| Field       | Value                                          |
|-------------|------------------------------------------------|
| **Date**    | 2026-02-17                                     |
| **Scope**   | ADR-006 Phases 0-6 implementation fidelity     |
| **Auditor** | Claude (code-level review)                     |

---

## Executive Summary

The ADR-006 implementation is **faithful to the architectural vision** across all seven phases. The protocol definitions, data types, orchestrator flow, backend implementations, and integration wiring match the ADR specification with high fidelity. The four protocol domains (TemporalFramer, TextGenerator, TextCleaner, TextAligner) are correctly defined, the orchestrator implements the 9-step flow as specified, and all audit bug fixes (C1, C2, M1) and dead code removals (L1, L2) are confirmed applied.

The deviations found are predominantly **additive enhancements** — extra convenience properties on data types, additional constructor parameters for flexibility, and richer metadata in backend implementations — rather than contradictions or omissions. There are two findings of moderate concern: (1) the alignment step uses per-frame `align()` calls instead of `align_batch()`, which deviates from the ADR's batch-oriented description and forfeits batch efficiency for the Qwen3 aligner; (2) the hardening config's `speech_regions` field on `HardeningConfig` is never populated at the orchestrator level, meaning speech regions flow only to the sentinel, not to the hardening stage.

No regressions were found in coupled mode code paths (CONTEXT_AWARE, VAD_SLICING). The old `_phase5_assembly()` method was fully removed (implementation plan said to keep it as deprecated fallback, but this is a low-risk cleanup decision). All explicitly deferred items (whisper-segment framer, VLLMGenerator, CTCAligner, HybridAligner, VADProportionalAligner, coupled mode migration) are confirmed absent as expected.

---

## Methodology

1. Read the ADR (`ADR-006-decoupled-subtitle-pipeline.md`) in full — all 14 sections.
2. Read the Implementation Plan (`ADR-006-IMPLEMENTATION-PLAN.md`) for detailed specifications.
3. For each architectural element (protocols, types, orchestrator, backends, integration), read the corresponding implementation file(s) line-by-line.
4. Field-by-field, method-by-method comparison between ADR specification and actual code.
5. Traced integration wiring in `qwen_pipeline.py` from `__init__` through `process()` to `process_scenes()`.
6. Verified bug fixes (C1, C2, M1) and dead code removal (L1, L2) in modified files.
7. Verified coupled mode code paths are untouched.

**Files audited** (22 new + 2 modified):
- `whisperjav/modules/subtitle_pipeline/__init__.py`
- `whisperjav/modules/subtitle_pipeline/types.py`
- `whisperjav/modules/subtitle_pipeline/protocols.py`
- `whisperjav/modules/subtitle_pipeline/hardening.py`
- `whisperjav/modules/subtitle_pipeline/reconstruction.py`
- `whisperjav/modules/subtitle_pipeline/orchestrator.py`
- `whisperjav/modules/subtitle_pipeline/framers/{__init__, factory, full_scene, vad_grouped, srt_source, manual}.py`
- `whisperjav/modules/subtitle_pipeline/generators/{__init__, factory, qwen3}.py`
- `whisperjav/modules/subtitle_pipeline/aligners/{__init__, factory, qwen3, none}.py`
- `whisperjav/modules/subtitle_pipeline/cleaners/{__init__, factory, qwen3, passthrough}.py`
- `whisperjav/pipelines/qwen_pipeline.py` (integration point, bug fixes, dead code removal)
- `whisperjav/modules/qwen_asr.py` (C1 fix verification)

---

## 1. Protocol Audit

### 1.1 TemporalFramer

**ADR Section 7.1 specification:**
```python
@runtime_checkable
class TemporalFramer(Protocol):
    def frame(self, audio: np.ndarray, sample_rate: int, **kwargs) -> FramingResult: ...
    def cleanup(self) -> None: ...
```

**Actual code** (`protocols.py:28-60`):
```python
@runtime_checkable
class TemporalFramer(Protocol):
    def frame(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> FramingResult: ...
    def cleanup(self) -> None: ...
```

**Verdict: MATCH.** The only difference is the explicit `Any` annotation on `**kwargs`, which is a refinement, not a deviation. Method signatures, return types, and `@runtime_checkable` decorator all match.

### 1.2 TextGenerator

**ADR Section 7.2 specification:**
```python
@runtime_checkable
class TextGenerator(Protocol):
    def generate(self, audio_path: Path, language: str = "ja",
                 context: Optional[str] = None, **kwargs) -> TranscriptionResult: ...
    def generate_batch(self, audio_paths: List[Path], language: str = "ja",
                       contexts: Optional[List[str]] = None, **kwargs) -> List[TranscriptionResult]: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...
```

**Actual code** (`protocols.py:68-107`):
All five methods present with identical signatures (modulo `**kwargs: Any` annotation). Return types match.

**Verdict: MATCH.**

### 1.3 TextCleaner

**ADR Section 7.3 specification:**
```python
@runtime_checkable
class TextCleaner(Protocol):
    def clean(self, text: str, **kwargs) -> str: ...
    def clean_batch(self, texts: List[str], **kwargs) -> List[str]: ...
```

**Actual code** (`protocols.py:115-130`):
Both methods present with identical signatures. No `load()`/`unload()` as ADR specifies ("lightweight text transform, no model").

**Verdict: MATCH.**

### 1.4 TextAligner

**ADR Section 7.4 specification:**
```python
@runtime_checkable
class TextAligner(Protocol):
    def align(self, audio_path: Path, text: str, language: str = "ja",
              **kwargs) -> AlignmentResult: ...
    def align_batch(self, audio_paths: List[Path], texts: List[str],
                    language: str = "ja", **kwargs) -> List[AlignmentResult]: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def cleanup(self) -> None: ...
```

**Actual code** (`protocols.py:138-179`):
All five methods present with identical signatures.

**Verdict: MATCH.**

---

## 2. Data Type Audit

### 2.1 WordTimestamp

**ADR Section 7.4 / 8.1:**
```python
@dataclass
class WordTimestamp:
    word: str
    start: float  # seconds, scene-relative
    end: float    # seconds, scene-relative
```

**Actual code** (`types.py:18-24`): Identical.

**Verdict: MATCH.**

### 2.2 TemporalFrame

**ADR Section 7.1:**
```python
@dataclass
class TemporalFrame:
    start: float                        # seconds, scene-relative
    end: float                          # seconds, scene-relative
    text: Optional[str] = None
    confidence: Optional[float] = None
    source: str = ""
```

**Actual code** (`types.py:33-51`): All five ADR fields present with correct types and defaults. **Addition**: A `duration` property (`max(0.0, self.end - self.start)`).

**Verdict: MATCH with benign addition.** The `duration` property is a pure convenience accessor — no state, no side effects.

### 2.3 FramingResult

**ADR Section 7.1:**
```python
@dataclass
class FramingResult:
    frames: List[TemporalFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Actual code** (`types.py:55-67`): Both fields present. Uses `list[TemporalFrame]` (Python 3.9+ lowercase) vs `List[TemporalFrame]` — functionally identical. **Additions**: `frame_count` and `total_duration` properties.

**Verdict: MATCH with benign additions.**

### 2.4 TranscriptionResult

**ADR Section 7.2:**
```python
@dataclass
class TranscriptionResult:
    text: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Actual code** (`types.py:76-81`): Identical.

**Verdict: MATCH.**

### 2.5 AlignmentResult

**ADR Section 7.4:**
```python
@dataclass
class AlignmentResult:
    words: List[WordTimestamp]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Actual code** (`types.py:89-109`): Both fields present. **Additions**: `word_count`, `char_count`, and `span_sec` properties.

**Verdict: MATCH with benign additions.**

### 2.6 TimestampMode

**ADR Section 9.3 / Implementation Plan 2.1:**
```python
class TimestampMode(str, Enum):
    ALIGNER_WITH_INTERPOLATION = "aligner_interpolation"
    ALIGNER_WITH_VAD_FALLBACK = "aligner_vad_fallback"
    ALIGNER_ONLY = "aligner_only"
    VAD_ONLY = "vad_only"
```

**Actual code** (`types.py:117-139`): Identical enum values.

**Verdict: MATCH.**

### 2.7 HardeningConfig

**ADR / Implementation Plan 2.1:**
```python
@dataclass
class HardeningConfig:
    timestamp_mode: TimestampMode = TimestampMode.ALIGNER_WITH_INTERPOLATION
    scene_duration_sec: float = 0.0
    speech_regions: Optional[List[Tuple[float, float]]] = None
```

**Actual code** (`types.py:148-153`): Identical.

**Verdict: MATCH.**

### 2.8 HardeningDiagnostics

**ADR / Implementation Plan 2.1:**
```python
@dataclass
class HardeningDiagnostics:
    segment_count: int = 0
    interpolated_count: int = 0
    fallback_count: int = 0
    clamped_count: int = 0
    sorted: bool = False
```

**Actual code** (`types.py:157-165`): All five ADR fields present. **Addition**: `timestamp_mode: str = ""`.

**Verdict: MATCH with benign addition.** The extra field records which mode was applied — useful for diagnostics.

---

## 3. Orchestrator Audit

### 3.1 Constructor

**ADR Section 9.1:**
```python
class DecoupledSubtitlePipeline:
    def __init__(
        self,
        framer: TemporalFramer,
        generator: TextGenerator,
        cleaner: TextCleaner,
        aligner: Optional[TextAligner],
        hardening_config: HardeningConfig,
        artifacts_dir: Optional[Path] = None,
    ): ...
```

**Actual code** (`orchestrator.py:60-68`): All six ADR parameters present. **Addition**: `language: str = "ja"`.

**Verdict: MATCH with addition.** The `language` parameter avoids passing language through kwargs at every `generate()`/`align()` call. Sensible enhancement.

### 3.2 process_scenes() Signature

**ADR Section 9.1:**
```python
def process_scenes(
    self,
    scene_paths: List[Path],
    scene_durations: List[float],
    speech_regions_per_scene: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[Optional[WhisperResult], Dict]]:
```

**Actual code** (`orchestrator.py:102-107`):
```python
def process_scenes(
    self,
    scene_audio_paths: list[Path],
    scene_durations: list[float],
    scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
) -> list[tuple[Any, dict[str, Any]]]:
```

**Deviations:**
- Parameter names: `scene_paths` → `scene_audio_paths`, `speech_regions_per_scene` → `scene_speech_regions`. Cosmetic naming change.
- Return type: `Optional[WhisperResult]` → `Any`. Avoids hard type dependency on `stable_whisper` at annotation level. Functionally identical at runtime.

**Verdict: MATCH (cosmetic naming differences).**

### 3.3 Internal Flow (ADR Section 9.2)

**ADR 9-step flow:**
1. `framer.frame(scene_audio)` → temporal_frames per scene
2. `generator.load()`
3. `generator.generate_batch(frame_audios)` → raw_texts
4. `generator.unload()` (VRAM swap)
5. `cleaner.clean_batch(raw_texts)` → clean_texts
6. IF aligner is not None: `aligner.load()`
7. `aligner.align_batch(frame_audios, clean_texts)` → alignments
8. `aligner.unload()` (VRAM swap)
9. Per scene: merge + sentinel + reconstruct + harden + diagnostics

**Actual implementation flow** (`orchestrator.py:130-156`):

| ADR Step | Implementation Method | Faithful? |
|----------|----------------------|-----------|
| 1 | `_step1_frame_and_slice()` | Yes |
| 2 | `_step2_4_generate_and_clean()` → `generator.load()` | Yes |
| 3 | `_step2_4_generate_and_clean()` → `generate_batch()` per scene | Yes |
| 4 | `_step2_4_generate_and_clean()` → `generator.unload()` + `_safe_cuda_cleanup()` | Yes |
| 5 | `_step2_4_generate_and_clean()` → `cleaner.clean_batch()` per scene | Yes |
| 6-8 | `_step5_7_align()` → `aligner.load()` / per-frame `align()` / `aligner.unload()` | **Partial** (see F-03) |
| 9 | `_step9_reconstruct_and_harden()` | Yes |

**Key deviation (F-03):** The ADR says Step 7 uses `aligner.align_batch()`. The actual code (`orchestrator.py:369-389`) calls `aligner.align()` per-frame in a loop, NOT `aligner.align_batch()`. This is functionally correct but forfeits any batch optimization the aligner backend might offer.

### 3.4 VRAM Lifecycle Management

**ADR specifies:**
```
generator.load() → generate → generator.unload() → safe_cuda_cleanup()
→ aligner.load() → align → aligner.unload() → safe_cuda_cleanup()
```

**Actual code:**
- `_step2_4_generate_and_clean()`: `generator.load()` at line 255, `generator.unload()` + `_safe_cuda_cleanup()` at lines 317-318, wrapped in `try/finally`.
- `_step5_7_align()`: `aligner.load()` at line 359, `aligner.unload()` + `_safe_cuda_cleanup()` at lines 412-413, wrapped in `try/finally`.

**Verdict: MATCH.** Both load/unload cycles wrapped in `try/finally` as the implementation plan specified.

### 3.5 Sentinel Integration

**ADR Section 9.4:** Sentinel operates in Step 9 only, called once with word dicts from alignment.

**Actual code** (`orchestrator.py:436-439, 454-474`):
- Imports `assess_alignment_quality` and `redistribute_collapsed_words` from `alignment_sentinel`
- Calls `assess_alignment_quality(all_words, duration)` on merged word dicts
- On COLLAPSED: gets speech regions, calls `redistribute_collapsed_words(all_words, duration, regions)`
- Reconstructs with `suppress_silence=False` for recovered words (H3 fix)
- Reconstructs with `suppress_silence=True` for normal words

**Verdict: MATCH.** Sentinel integration is exactly as specified in ADR Section 9.4 and Appendix A.

### 3.6 Sentinel Stats Accumulation

**ADR / Implementation Plan Section 5.4:**
```python
self.sentinel_stats = {
    "total_scenes": 0,
    "collapsed_scenes": 0,
    "recovered_scenes": 0,
    "recovery_strategies": {"vad_guided": 0, "proportional": 0},
}
```

**Actual code** (`orchestrator.py:92-97`): Identical structure and key names.

**Verdict: MATCH.**

### 3.7 Debug Artifacts

**ADR / Implementation Plan Section 5.5:**
- `{basename}_scene{idx}_raw.txt`
- `{basename}_scene{idx}_clean.txt`
- `{basename}_scene{idx}_merged.json`
- `{basename}_scene{idx}_diag.json`

**Actual code** (`orchestrator.py:644-656`):
- `scene{idx:03d}_raw.txt` (at line 313)
- `scene{idx:03d}_clean.txt` (at line 328)
- `scene{idx:03d}_aligned.json` (at line 405) — named "aligned" not "merged"
- `scene{idx:03d}_diag.json` (at line 511)

**Deviation:** Naming convention uses `scene{idx:03d}_` prefix instead of `{basename}_scene{idx}_`. Also "merged" → "aligned". These are cosmetic differences.

**Verdict: MATCH (cosmetic naming differences).**

### 3.8 Error Handling

**ADR / Implementation Plan Section 5.2:**
- Per-scene `try/except` in Step 9
- Batch generation failure falls back to per-frame
- Failed frame → empty string
- Failed scene → `(None, error_diagnostics)`

**Actual code:**
- Per-scene `try/except` at `orchestrator.py:451,518-525`
- Batch fallback to per-frame at `orchestrator.py:284-305`
- Failed frame → `""` at `orchestrator.py:305`
- Failed scene → `(None, {"scene_idx": idx, "error": str(e)})` at `orchestrator.py:525`

**Verdict: MATCH.**

---

## 4. Backend Audit

### 4.1 Framers

#### 4.1.1 TemporalFramerFactory

**ADR / Implementation Plan Section 4.1.2:**
```python
_registry = {
    "full-scene": "...FullSceneFramer",
    "vad-grouped": "...VadGroupedFramer",
    "srt-source": "...SrtSourceFramer",
    "manual": "...ManualFramer",
}
```

**Actual code** (`framers/factory.py:14-19`): Identical four entries.

**Not implemented (as documented):** `whisper-segment` — explicitly deferred in Implementation Plan Section 4.1 Note.

**Verdict: MATCH.**

#### 4.1.2 FullSceneFramer

**ADR / Implementation Plan Section 4.2:**
- Returns one frame `[0, duration]` with `source="full-scene"`
- Metadata includes `strategy: "full-scene"` and `frame_count: 1`
- `cleanup()` is no-op

**Actual code** (`framers/full_scene.py:19-62`):
- Returns one frame `[0, duration]` with `source="full-scene"` (line 48-50)
- Metadata: `strategy: "full-scene"`, `frame_count: 1`, **plus** `audio_duration_sec` (line 54-58)
- `cleanup()` is no-op (line 61-62)

**Verdict: MATCH with benign metadata addition.**

#### 4.1.3 VadGroupedFramer

**ADR / Implementation Plan Section 4.3:**
- Constructor: `segmenter_backend`, `max_group_duration_s`, `chunk_threshold_s`, `min_frame_duration_s`, `segmenter_config`
- `frame()`: creates segmenter, detects speech regions, groups, filters, converts to TemporalFrames
- `speech_regions` stored in metadata
- `cleanup()` releases segmenter

**Actual code** (`framers/vad_grouped.py:25-170`):
- Constructor parameters: all five match (lines 36-43)
- Lazy segmenter creation via `_ensure_segmenter()` (lines 61-75)
- Groups from `seg_result.groups`, filtered by min duration (lines 107-118)
- Speech regions in `metadata["speech_regions"]` (line 161)
- `cleanup()` releases segmenter (lines 166-170)
- Rich metadata including `speech_coverage_ratio`, `segmenter_params`, etc.

**Verdict: MATCH.** Implementation is more detailed than spec (richer metadata, logging) but structurally identical.

#### 4.1.4 SrtSourceFramer

**ADR / Implementation Plan Section 4.4:**
- Constructor: `srt_path`, `keep_text`, `min_frame_duration_s`
- Uses `pysrt` to parse SRT
- Each entry → TemporalFrame with optional text
- Filters by min duration

**Actual code** (`framers/srt_source.py:21-133`):
- Constructor: all three parameters match (lines 30-35)
- Uses `pysrt` with lazy import (line 72)
- Clamps to audio duration (lines 94-95)
- Filters by min duration (line 98)
- `source="srt-source"` (line 109)

**Verdict: MATCH.** The code adds audio boundary clamping not mentioned in the ADR — a defensive improvement.

#### 4.1.5 ManualFramer

**ADR / Implementation Plan Section 4.5:**
```python
class ManualFramer:
    def __init__(self, timestamps: List[Tuple[float, float]]): ...
```

**Actual code** (`framers/manual.py:18-92`):
```python
class ManualFramer:
    def __init__(self, timestamps: list[tuple[float, float]], texts: list[str] | None = None): ...
```

**Deviation:** Constructor has additional `texts` parameter for pre-populating frame text. Validation ensures `len(texts) == len(timestamps)` if provided.

**Verdict: MATCH with addition.** The `texts` parameter is an optional enhancement enabling "inject known timestamps + text" testing workflows.

### 4.2 Generators

#### 4.2.1 TextGeneratorFactory

**Actual code** (`generators/factory.py:14-16`): Registry has `"qwen3"` only. Matches plan.

**Not implemented (as documented):** TransformersGenerator, VLLMGenerator, WhisperTextGenerator — explicitly deferred.

**Verdict: MATCH.**

#### 4.2.2 Qwen3TextGenerator

**ADR / Implementation Plan Section 3.1.3:**
- Constructor: `model_id`, `device`, `dtype`, `batch_size`, `max_new_tokens`, `language`, `repetition_penalty`, `max_tokens_per_audio_second`, `attn_implementation`
- `load()`: Creates QwenASR, calls `load_model_text_only()`
- `unload()`: Calls `unload_model()`, `safe_cuda_cleanup()`, sets `asr = None`
- `generate()`: Delegates to `generate_batch()`
- `generate_batch()`: Calls `asr.transcribe_text_only()`, wraps in `TranscriptionResult`
- `cleanup()`: Calls `unload()`

**Actual code** (`generators/qwen3.py:24-199`):

| Method | ADR Spec | Code | Match? |
|--------|----------|------|--------|
| `__init__` params | 9 params | 9 identical params (line 33-43) | Yes |
| `load()` | Create QwenASR, `load_model_text_only()` | Lines 78-106, includes `use_aligner=False` | Yes |
| `unload()` | `unload_model()` + `safe_cuda_cleanup()` | Lines 108-125 | Yes |
| `generate()` | Delegates to `generate_batch()` | Lines 127-151 | Yes |
| `generate_batch()` | `asr.transcribe_text_only()` → `TranscriptionResult` | Lines 153-195 | Yes |
| `cleanup()` | Calls `unload()` | Lines 197-199 | Yes |

**Addition:** `is_loaded` property (line 73-76). Benign.

**Verdict: MATCH.**

### 4.3 Aligners

#### 4.3.1 TextAlignerFactory

**Actual code** (`aligners/factory.py:14-17`): Registry has `"qwen3"` and `"none"`. Matches plan.

**Not implemented (as documented):** CTCAligner, VADProportionalAligner, HybridAligner — explicitly deferred.

**Verdict: MATCH.**

#### 4.3.2 Qwen3ForcedAlignerAdapter

**ADR / Implementation Plan Section 3.2.3:**
- Constructor: `aligner_id`, `device`, `dtype`, `language`
- `load()`: Creates QwenASR, `load_aligner_only()`
- `unload()`: `unload_model()` + `safe_cuda_cleanup()`
- `align()`: Delegates to `align_batch()`
- `align_batch()`: `asr.align_standalone()` → `merge_master_with_timestamps()` → `AlignmentResult`

**Actual code** (`aligners/qwen3.py:23-215`):

| Method | ADR Spec | Code | Match? |
|--------|----------|------|--------|
| `__init__` params | 4 params | 4 identical params (lines 37-42) | Yes |
| `load()` | Create QwenASR, `load_aligner_only()` | Lines 66-88 | Yes |
| `unload()` | `unload_model()` + `safe_cuda_cleanup()` | Lines 90-103 | Yes |
| `align()` | Delegates to `align_batch()` | Lines 105-129 | Yes |
| `align_batch()` | `align_standalone()` → `merge_master_with_timestamps()` → `AlignmentResult` | Lines 131-211 | Yes |
| `cleanup()` | Calls `unload()` | Lines 213-215 | Yes |

**Critical detail verified:** `merge_master_with_timestamps()` is called inside `align_batch()` (line 187) as specified. The reconciliation of punctuated text + aligner timestamps is encapsulated within the adapter.

**Verdict: MATCH.**

#### 4.3.3 NoneAligner

**ADR / Implementation Plan Section 3.2.4:**
```python
class NoneAligner:
    def align(...): return AlignmentResult(words=[], metadata={"skipped": True})
    def load/unload/cleanup: pass
```

**Actual code** (`aligners/none.py:14-49`): Exact match.

**Verdict: MATCH.**

### 4.4 Cleaners

#### 4.4.1 TextCleanerFactory

**Actual code** (`cleaners/factory.py:14-17`): Registry has `"qwen3"` and `"passthrough"`. Matches plan.

**Verdict: MATCH.**

#### 4.4.2 Qwen3TextCleaner

**ADR / Implementation Plan Section 3.3.3:**
```python
class Qwen3TextCleaner:
    def __init__(self, config: Optional[AssemblyCleanerConfig] = None):
        self._cleaner = AssemblyTextCleaner(config)
    def clean(self, text, **kwargs) -> str:
        return self._cleaner.clean(text)
    def clean_batch(self, texts, **kwargs) -> List[str]:
        return self._cleaner.clean_batch(texts)
```

**Actual code** (`cleaners/qwen3.py:17-76`):

**Deviations:**
1. Constructor signature: `config: Optional[Any] = None, language: str = "ja"` instead of `config: Optional[AssemblyCleanerConfig] = None`. The `Any` type avoids importing `AssemblyCleanerConfig` at class definition time (lazy import inside body). The `language` parameter is passed through to `AssemblyTextCleaner`.
2. The `clean()` and `clean_batch()` methods handle the tuple return (`cleaned, stats`) from `AssemblyTextCleaner`, returning only the cleaned text per protocol contract. Stats are stored in `self._last_stats` for optional diagnostic access.
3. Config validation: if `config` is not `AssemblyCleanerConfig`, falls back to defaults with a warning.

**Verdict: MATCH with enhancements.** The deviations are defensive improvements (lazy import, config validation, stats preservation) that don't violate the protocol contract.

#### 4.4.3 PassthroughCleaner

**ADR / Implementation Plan Section 3.3.4:**
```python
class PassthroughCleaner:
    def clean(self, text, **kwargs): return text
    def clean_batch(self, texts, **kwargs): return texts
```

**Actual code** (`cleaners/passthrough.py:11-20`): `clean_batch` returns `list(texts)` (defensive copy). Otherwise identical.

**Verdict: MATCH.**

---

## 5. Integration Audit

### 5.1 Pipeline Construction (QwenPipeline.__init__)

**ADR / Implementation Plan Section 6.1:**
- Orchestrator constructed in `__init__()` for ASSEMBLY mode only
- Uses factory classes to create components from `_asr_config`

**Actual code** (`qwen_pipeline.py:288-292, 341-413`):
- `self._subtitle_pipeline = None` initialized, then built if `input_mode == InputMode.ASSEMBLY` (line 291-292)
- `_build_subtitle_pipeline()` uses all four factories (lines 349-413)
- Component configuration pulled from `self._asr_config` (line 362)

**Verified wiring:**
- TemporalFramer: `"full-scene"` (line 365)
- TextGenerator: `"qwen3"` with 9 config parameters from `_asr_config` (lines 368-379)
- TextCleaner: `"qwen3"` with `AssemblyCleanerConfig` when enabled, `"passthrough"` when disabled (lines 382-390)
- TextAligner: `"qwen3"` when `use_aligner=True`, else `None` (lines 393-401)
- HardeningConfig: `timestamp_mode` mapped from old enum to new enum (lines 403-411)

**Verdict: MATCH.** Implementation follows the plan's Section 6.1.2 precisely.

### 5.2 Phase 5 Assembly Path (QwenPipeline.process)

**ADR / Implementation Plan Section 6.1.3:**
- Replace `_phase5_assembly()` call with orchestrator
- Unpack results to Phase 6 format

**Actual code** (`qwen_pipeline.py:641-697`):
- Assembly mode detected (line 641)
- `artifacts_dir` set (line 651)
- Scene paths extracted (lines 654-655)
- Speech regions converted from `{idx: SegmentationResult}` to `List[List[Tuple]]` (lines 658-668)
- `process_scenes()` called (lines 671-674)
- Results unpacked to `(result, scene_idx)` tuples (lines 678-679)
- Sentinel stats mapped to existing format (lines 682-686)
- Per-scene diagnostics saved (lines 689-697)

**Verdict: MATCH.** The conversion of speech regions from the dict-based Phase 4 format to the list-based orchestrator format is a necessary adapter step.

### 5.3 Cleanup

**ADR / Implementation Plan Section 6.1.4:**
```python
if hasattr(self, '_subtitle_pipeline') and self._subtitle_pipeline:
    self._subtitle_pipeline.cleanup()
    self._subtitle_pipeline = None
```

**Actual code** (`qwen_pipeline.py:301-312`):
```python
if self._subtitle_pipeline is not None:
    try:
        self._subtitle_pipeline.cleanup()
    except Exception as e:
        logger.warning("Subtitle pipeline cleanup failed (non-fatal): %s", e)
    self._subtitle_pipeline = None
super().cleanup()
```

**Verdict: MATCH with improvement.** The `try/except` is more robust than the plan specified.

### 5.4 Old _phase5_assembly() Removal

**Implementation Plan Section 8.1:**
> Do NOT remove the old `_phase5_assembly()` method — keep for reference/fallback until the orchestrator is proven stable.

**Actual code:** `_phase5_assembly()` is **completely removed** (grep confirms no matches). This is a deviation from the plan's conservative approach, but a reasonable one if the orchestrator has been validated.

**Verdict: DEVIATION (low risk).** The fallback path no longer exists. If the orchestrator fails, there is no way to revert to the old assembly code without git revert.

---

## 6. Workflow Audit

### 6.1 Workflows Currently Exercisable

**ADR Section 12 promises five new workflows:**

| # | Workflow | Possible Now? | How? |
|---|---------|---------------|------|
| 12.1 | Whisper-Guided Qwen | **No** | Requires `whisper-segment` framer (deferred) |
| 12.2 | SRT Re-Transcription | **Yes** | `SrtSourceFramer(srt_path, keep_text=False)` + `Qwen3TextGenerator` + `NoneAligner` or `None` |
| 12.3 | Two-Pass Refinement | **Yes** | Pass 1 → SRT → Pass 2 with `SrtSourceFramer(pass1.srt)` |
| 12.4 | Aligner-Free Fast Mode | **Yes** | `VadGroupedFramer` + `Qwen3TextGenerator` + `aligner=None` |
| 12.5 | Cross-Model Benchmarking | **Partial** | Only `qwen3` generator is implemented; needs additional generators for comparison |

**Assessment:** Three of five workflows are structurally possible. The missing `whisper-segment` framer was explicitly deferred. Cross-model benchmarking is structurally supported but requires additional generator backends.

**Important caveat:** Workflows 12.2-12.4 are **only possible via programmatic API** — there are no CLI flags or GUI options to select non-default framers, generators, or aligner configurations. The current CLI integration (`qwen_pipeline.py`) always constructs `full-scene` framer and `qwen3` aligner for assembly mode. The composability exists at the code level but is not yet exposed to users.

### 6.2 Workflow: Assembly via CLI

**Before ADR-006:** `whisperjav video.mp4 --mode qwen --qwen-input-mode assembly`
**After ADR-006:** Same command, now routes through `DecoupledSubtitlePipeline.process_scenes()`.

**Verdict:** The primary workflow (assembly mode via CLI) is fully functional.

---

## 7. Error Handling & VRAM Audit

### 7.1 safe_cuda_cleanup() Usage

**ADR / Audit M5:** "Orchestrator calls `safe_cuda_cleanup()` once after each `unload()`."

**Actual code:**
- After generator unload: `orchestrator.py:318` — `self._safe_cuda_cleanup()`
- After aligner unload: `orchestrator.py:413` — `self._safe_cuda_cleanup()`
- In generator adapter unload: `generators/qwen3.py:121-123` — `safe_cuda_cleanup()`
- In aligner adapter unload: `aligners/qwen3.py:99-101` — `safe_cuda_cleanup()`

**Finding (F-07):** `safe_cuda_cleanup()` is called TWICE per unload — once inside the adapter's `unload()` and once by the orchestrator after calling `unload()`. This is not harmful (double cleanup is idempotent) but wastes ~2ms per call.

### 7.2 OOM Recovery

**ADR / Audit C1:** "Eliminated — orchestrator manages model lifecycle externally via `generator.load()`/`unload()`. No closures capture model references."

**Actual code:** The `Qwen3TextGenerator.load()` creates a fresh `QwenASR` instance each time (line 91-103). The `unload()` sets `self._asr = None` (line 119). No closure captures.

**However:** The QwenASR instance used inside `generate_batch()` still has its own internal OOM handling (in `qwen_asr.py:977-985`). The C1 fix (reassigning `qwen_model` after reload, line 985) is applied there as a defense-in-depth measure.

**Verdict: MATCH.** The architecture eliminates the stale closure problem at the orchestrator level, and the C1 fix provides defense-in-depth at the QwenASR level.

### 7.3 try/finally for VRAM Lifecycle

**Actual code:**
- Generator: `try` at line 257, `finally` at line 315-318 — always unloads even on exception.
- Aligner: `try` at line 360, `finally` at line 411-413 — always unloads even on exception.

**Verdict: MATCH.**

---

## 8. Regression Audit

### 8.1 Coupled Modes (CONTEXT_AWARE, VAD_SLICING)

**ADR Section 13:** "What Doesn't Change" — coupled modes continue through old code paths.

**Verification:**
- `qwen_pipeline.py:641`: `if self.input_mode == InputMode.ASSEMBLY:` — only assembly routes to orchestrator
- `qwen_pipeline.py:698-934`: `else:` block contains full CONTEXT_AWARE and VAD_SLICING code paths
- `_transcribe_speech_regions()` still present at line 1116
- All `_transcribe_group()` logic preserved

**C2 fix impact on coupled modes:** `japanese_postprocess` is now hardcoded `False` in `_asr_config` (line 272). This means CONTEXT_AWARE and VAD_SLICING no longer apply `JapanesePostProcessor` — this is intentional per ADR Section 5.1 ("Coupled modes are incorrect to apply it").

**Verdict: NO REGRESSION.** Coupled modes fully preserved. C2 fix intentionally changes their behavior (removing incorrect post-processing).

### 8.2 Dead Code Removal Verification

**L1 (`_apply_vad_filter`):** Confirmed absent from `qwen_pipeline.py`. No grep matches.
**L2 (`cross_scene_context` branches):** Confirmed absent. No grep matches.

**Verdict: CLEAN.** Dead code properly removed.

### 8.3 M1 Fix (300s comment)

Three occurrences of "ForcedAligner's ... limit" in `qwen_pipeline.py` (lines 218, 474, 492) now correctly say "180s".

**Verdict: FIXED.**

---

## 9. Consolidated Findings Register

| ID | Category | Severity | ADR Reference | Code Reference | Description |
|----|----------|----------|---------------|----------------|-------------|
| F-01 | Orchestrator | **MEDIUM** | Section 9.2, Step 7 | `orchestrator.py:369-389` | Alignment uses per-frame `align()` instead of `align_batch()`. Forfeits batch efficiency. ADR flow says "aligner.align_batch(frame_audios, clean_texts)". |
| F-02 | Integration | **MEDIUM** | Section 9.3 | `orchestrator.py:491-496` | `HardeningConfig.speech_regions` is never populated at the orchestrator level. The field exists on the type but the orchestrator creates `HardeningConfig` with `speech_regions=self.hardening_config.speech_regions` which is always `None` (never set by `_build_subtitle_pipeline()`). Speech regions flow only to sentinel recovery, not to hardening. |
| F-03 | Integration | **LOW** | Implementation Plan 8.1 | `qwen_pipeline.py` (absence) | Old `_phase5_assembly()` method completely removed instead of kept as deprecated fallback. No rollback path without git revert. |
| F-04 | Type | **LOW** | Section 8 | `types.py:48-51, 62-67, 96-109, 165` | Extra convenience properties (`duration`, `frame_count`, `total_duration`, `word_count`, `char_count`, `span_sec`, `timestamp_mode`) not in ADR. All benign additions. |
| F-05 | Backend | **LOW** | Section 5.4, Plan 4.5 | `framers/manual.py:29` | `ManualFramer` has `texts` parameter not in ADR. Optional enhancement for testing. |
| F-06 | Backend | **LOW** | Plan 3.3.3 | `cleaners/qwen3.py:26` | `Qwen3TextCleaner` constructor takes `config: Optional[Any]` + `language: str` instead of `config: Optional[AssemblyCleanerConfig]`. Defensive type handling improvement. |
| F-07 | VRAM | **LOW** | Section 9, M5 | `orchestrator.py:318,413` + `generators/qwen3.py:123`, `aligners/qwen3.py:101` | `safe_cuda_cleanup()` called twice per unload cycle — once inside adapter and once by orchestrator. Idempotent but redundant. |
| F-08 | Orchestrator | **LOW** | Section 9.1 | `orchestrator.py:68` | Extra `language: str = "ja"` parameter in `__init__` not in ADR. Avoids passing language through kwargs at every call. |
| F-09 | Artifacts | **LOW** | Plan 5.5 | `orchestrator.py:405` | Debug artifact named `aligned.json` instead of `merged.json` as in plan. Cosmetic. |
| F-10 | Workflows | **INFO** | Section 12 | N/A | Workflows 12.2-12.4 are only accessible via programmatic API, not CLI/GUI. No CLI flags for framer/aligner selection. |

**Severity counts:**

| Severity | Count |
|----------|-------|
| MEDIUM | 2 |
| LOW | 7 |
| INFO | 1 |
| **Total** | **10** |

---

## 10. Recommendations

### Priority 1 (Address soon)

**F-01: Use `align_batch()` in orchestrator Step 5-7.** The per-frame `align()` loop in `_step5_7_align()` misses an optimization opportunity. For the `full-scene` framer (one frame per scene), this is moot. But for `vad-grouped` framer (multiple frames per scene), batching alignment calls would be significantly more efficient. Recommendation: Change the per-frame loop to collect all frame audio paths + texts for a scene, call `align_batch()` once per scene, then unpack results. The adapter's `align_batch()` already exists and works.

**F-02: Populate `HardeningConfig.speech_regions` for VAD_ONLY mode.** Currently, the `speech_regions` field on `HardeningConfig` is never populated. This means the `_apply_vad_only_timestamps()` function does not have access to actual speech regions — it distributes proportionally across the entire scene duration, not across speech regions specifically. For `TimestampMode.VAD_ONLY`, the hardening stage should use speech regions if available. Recommendation: Pass `scene_speech_regions[scene_idx]` through to `HardeningConfig` in `_step9_reconstruct_and_harden()`.

### Priority 2 (Low urgency)

**F-07: Remove double `safe_cuda_cleanup()`.** Either remove the call from the adapter's `unload()` methods (and document that the caller is responsible) or remove it from the orchestrator (and document that adapters self-clean). The current approach works but is architecturally unclear about responsibility.

**F-03: Document the removal of `_phase5_assembly()`.** If the old method is intentionally removed (not just forgotten), add a comment in the commit or CHANGELOG noting that the orchestrator is now the only assembly path with no fallback.

### Priority 3 (Future work)

**F-10: Expose workflow composability via CLI/GUI.** The ADR's workflow promises (12.1-12.5) require either CLI flags (e.g., `--framer srt-source --framer-srt-path existing.srt --aligner none`) or a configuration file mechanism. Currently, users can only access the default assembly configuration (full-scene framer + qwen3 generator + qwen3 aligner).

---

*End of ADR-006 Audit B*
