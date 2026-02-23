# A3: Roadmap & Path Forward — subtitle-pipeline as Model-Agnostic Abstract Pipeline

**Date**: 2026-02-19
**Status**: Design Document (synthesized from A1 Vision Conformance Audit + A2 IMPL-001 Conformance Audit)
**Scope**: `whisperjav/modules/subtitle_pipeline/` and its integration surface

### Revision History

| Date | Change |
|------|--------|
| 2026-02-19 | Initial version from A1/A2/A3 audits |
| 2026-02-19 | **Clarifications applied** — see below |

**Applied Clarifications:**
- **C1**: WhisperSegmentFramer (BUILD-1) explicitly deferred to a later phase. Not a near-term priority.
- **C2**: The Qwen3 ForcedAligner is **model-agnostic** — it works with any model's text output, not just Qwen3. This removes the urgency for a CTC aligner (BUILD-3) as a prerequisite for model independence.
- **C5**: CLI/GUI composability exposure elevated to **high priority** (Tier 1). The pipeline infrastructure exists but is invisible to users.
- **Execution order restructured**: Tier 0 (step-down foundation) + Tier 1 (composability UX, formerly Tier 4) are the priorities. Middle tiers (new generators, aligners, framers) deprioritized.

---

## 1. State of the Art

### What Works (Proven in Production)

The subtitle-pipeline is a structurally complete, operationally proven system that has survived the full IMPL-001 lifecycle: design, implementation, parity validation, and strangulation of the legacy coupled code. The following components are production-hardened:

**Architecture (all proven):**
- 4 runtime-checkable protocols: `TemporalFramer`, `TextGenerator`, `TextCleaner`, `TextAligner` (`protocols.py`)
- 4 lazy-import factory registries with class caching (`generators/factory.py`, `framers/factory.py`, `cleaners/factory.py`, `aligners/factory.py`)
- `DecoupledSubtitlePipeline` orchestrator with 9-step flow, explicit VRAM lifecycle, two-pass step-down retry, and per-scene diagnostics (`orchestrator.py`)
- `DecoupledPipeline` entry point: configuration-driven, 9-phase outer pipeline with YAML config loading (`decoupled_pipeline.py`)
- Shared hardening layer with 4 timestamp resolution modes (`hardening.py`)
- `reconstruct_from_words()` via stable-ts `transcribe_any()` with Japanese regrouping (`reconstruction.py`)
- Alignment sentinel with V2 detection (5 triggers: coverage, CPS, span, zero_position, degenerate) and 2 recovery strategies (VAD-guided, proportional) (`alignment_sentinel.py`)
- `SceneDiagnostics` v2.0.0 canonical diagnostics schema (`types.py`)
- Step-down retry with `StepDownConfig` and framer `reframe()` support (`orchestrator.py`, `vad_grouped.py`)

**Backends (all proven):**
- Framers: `FullSceneFramer`, `VadGroupedFramer` (with `reframe()`), `SrtSourceFramer`, `ManualFramer`
- Generators: `Qwen3TextGenerator` (text-only mode, batch support, dynamic token budgets)
- Cleaners: `Qwen3TextCleaner` (wraps AssemblyTextCleaner), `PassthroughCleaner`
- Aligners: `Qwen3ForcedAlignerAdapter` (model-agnostic forced aligner — works with any model's text output, not Qwen-specific), `NoneAligner`

**Integration (all proven):**
- `QwenPipeline` uses the orchestrator as its sole ASR code path (post-Phase 4 strangulation, 861 LOC)
- `DecoupledPipeline` as the generic entry point with `--pipeline decoupled` CLI flag
- Legacy `InputMode` enum values (`CONTEXT_AWARE`, `VAD_SLICING`) map to assembly with appropriate framer overrides
- YAML pipeline configuration via `load_pipeline_config()`

### What's Theoretical (Designed But Not Built)

- **Non-Qwen backends**: No TextGenerator or TextAligner exists beyond Qwen3. The protocol contracts are validated only by one implementation.
- **Cross-model composability**: The `SrtSourceFramer` enables a "Whisper-framed, Qwen-generated" workflow in principle, but nobody has tested Whisper SRT timestamps feeding into Qwen3 text generation.
- **CLI/GUI exposure** (**HIGH PRIORITY GAP**): The DecoupledPipeline's component selection (`--generator`, `--framer`, `--cleaner`, `--aligner`) is accessible but not prominently surfaced. Power users cannot easily discover or use the composability. This is the primary barrier to adoption and must be addressed early — see Tier 1 in Section 8.
- **Multi-model ensemble at the generator level**: Running 2+ TextGenerators and selecting the best output per scene.
- **Step-down on FullSceneFramer**: The `FullSceneFramer` has no `reframe()`, so step-down is only available when `VadGroupedFramer` is the primary framer.

### What's Absent (Not Even Designed)

- **Whisper-segment framer** (GAP-1, **deferred**): The cross-model workflow — use Whisper's segment timestamps as temporal frames for a different model's text generation. Explicitly deferred to a later phase.
- **CTC aligner**: wav2vec2 or similar CTC models that produce character-level timestamps. Note: the existing Qwen3 ForcedAligner is model-agnostic (works with any model's text), so a CTC aligner is an alternative option rather than a blocking dependency for model independence.
- **HTTP/API-based generators**: Connecting to external ASR services (vLLM server, Gemini audio API, GPT-4o audio) as TextGenerator backends.
- **Benchmarking/evaluation framework**: Comparing different component combinations against ground-truth SRT files.
- **Dynamic component selection**: Choosing framer/generator/aligner at runtime based on content analysis (e.g., "this scene has heavy BGM, use enhanced audio + aggressive VAD").

---

## 2. The Model-Agnostic Vision

### The Core Proposition

The subtitle-pipeline's value is NOT that it can run an ASR model. Any script can do that. Its value is the **infrastructure that wraps model output**:

1. **Temporal Framing** — deciding WHEN in the audio to listen (scene detection + VAD + grouping)
2. **VRAM Lifecycle** — loading/unloading models across GPU memory constraints (generator unload before aligner load)
3. **Alignment Sentinel** — detecting when timestamps are garbage and recovering
4. **Reconstruction** — converting raw words into properly structured, Japanese-regrouped WhisperResult
5. **Hardening** — 4 timestamp resolution strategies, boundary clamping, chronological sorting
6. **Step-down Retry** — automatically re-framing and retrying when scenes fail
7. **Diagnostics** — per-scene structured telemetry for debugging and quality assessment

A new ASR model gets ALL of this infrastructure for free by implementing a single protocol (`TextGenerator`: `load()`, `unload()`, `generate()`, `generate_batch()`, `cleanup()`). That is approximately 100-200 lines of adapter code. Zero orchestrator changes. Zero hardening changes. Zero framer changes.

### Model Categories and Their Characteristics

The vision targets "any model that produces text of transcription but no timestamps." In practice, the model landscape is broader and the pipeline should accommodate various output modalities:

#### Category A: Text-Only Output (Primary Target)

Models that produce transcription text with no temporal information whatsoever. These are the pipeline's sweet spot — they need the full framing + generation + alignment flow.

| Model Type | Example | VRAM | Latency | Quality for JAV |
|------------|---------|------|---------|-----------------|
| Qwen3-ASR (current) | `Qwen/Qwen3-ASR-1.7B` | ~3.5GB | 2-4s/scene | Excellent (trained on conversational Japanese) |
| HuggingFace Transformers | `kotoba-tech/kotoba-whisper-v2.2` | 1-4GB | 1-3s/scene | Good (general Japanese ASR) |
| Local LLM (audio input) | Qwen2-Audio, SALMONN | 4-16GB | 3-10s/scene | Unknown (general audio understanding) |
| Anime/media ASR | Hypothetical fine-tuned model | 1-4GB | 1-3s/scene | Potentially excellent (trained on similar content) |

**Integration pattern**: Implement `TextGenerator` protocol. Pair with `Qwen3ForcedAlignerAdapter` for timestamps — the Qwen3 ForcedAligner is model-agnostic and works with any model's text output, so no aligner swap is needed.

#### Category B: Text + Timestamps (Existing Timestamps Usable)

Models that produce both text and timestamps. The pipeline can either use their timestamps directly (aligner-free) or discard timestamps and re-align with a dedicated aligner.

| Model Type | Example | Use Case |
|------------|---------|----------|
| Whisper variants | `openai/whisper-large-v3-turbo` | Framing source (WhisperSegmentFramer) OR generator with `NoneAligner` |
| Faster-Whisper | Community builds | Fast draft for framing, Qwen for final text |
| CTC models | wav2vec2-large-xlsr-japanese | Character-level timestamps, useful as aligner |

**Integration patterns**:
1. As `TemporalFramer`: Use model's segment timestamps as frames (WhisperSegmentFramer — GAP-1)
2. As `TextGenerator` + `NoneAligner`: Trust the model's own timestamps, skip alignment
3. As `TextAligner`: CTC models produce character-level alignment suitable as a `TextAligner` backend (optional — the existing Qwen3 ForcedAligner already works with any model's text)

#### Category C: API/HTTP-Based Models (Cloud or Self-Hosted)

Models accessed via HTTP API. Text output, no local VRAM management needed (or separate VRAM server).

| Model Type | Example | VRAM Management |
|------------|---------|-----------------|
| vLLM-served models | Self-hosted Qwen3-ASR via vLLM | Server manages VRAM; `load()`/`unload()` become no-ops |
| Gemini audio | `gemini-2.0-flash` with audio input | No VRAM; API key auth; rate limiting |
| GPT-4o audio | `gpt-4o-audio-preview` | No VRAM; API key auth; high latency |
| OpenRouter | Various models via unified API | No VRAM; key + model selection |

**Integration pattern**: Implement `TextGenerator` where `load()`/`unload()` are no-ops (or manage connection pooling). The orchestrator's VRAM swap logic gracefully handles no-op lifecycle methods.

#### Category D: Ensemble / Multi-Model

Running multiple generators and selecting or merging the best output. Not a single model but a composition pattern.

| Pattern | Description |
|---------|-------------|
| Best-of-N | Run N generators, score outputs (by length, coherence, consistency), pick best |
| Voting | Run N generators, keep text that appears in majority |
| Draft-and-refine | Fast model generates draft, expensive model refines |
| Parallel specialization | Model A for speech, Model B for onomatopoeia, merge |

**Integration pattern**: Implement a `CompositeTextGenerator` that wraps multiple `TextGenerator` instances and implements selection/merging logic. The orchestrator sees a single `TextGenerator`.

### Why subtitle-pipeline vs. Running a Model Directly

A user can always run `model.transcribe(audio)` and get text. The subtitle-pipeline adds:

| Infrastructure | Value | Who Benefits |
|---------------|-------|-------------|
| Temporal framing | Breaks 2-hour video into manageable chunks within model context window | All users |
| Scene detection integration | Intelligent splitting at silence/semantic boundaries | All users |
| VRAM lifecycle | Runs 3.5GB generator + 1.2GB aligner on 8GB GPU via sequential loading | Users with limited VRAM (8-12GB) |
| Alignment sentinel | Catches broken timestamps that would produce unwatchable subtitles | All users (automatic quality gate) |
| Step-down retry | Recovers from failed scenes instead of losing content | Users with long/complex videos |
| Hardening | 4 timestamp strategies for different quality/speed tradeoffs | Power users |
| Japanese reconstruction | stable-ts regrouping with particle-based boundaries | Japanese subtitle users |
| Diagnostics | Per-scene telemetry for debugging | Power users, developers |

Without subtitle-pipeline, each model integration would need to independently solve all of these problems. The pipeline is a force multiplier.

---

## 3. Immediate Wins (0-2 weeks effort)

### WIN-1: FullSceneFramer.reframe() — Enable Step-Down for Default Framer

**Gap closed**: GAP-6 (step-down disabled for full-scene framer)
**Effort**: 2-4 hours (including tests)
**Impact**: High — currently the most common framer (`full-scene` is the default for assembly mode) cannot participate in step-down retry. Users with long scenes (30-120s) that fail alignment get no recovery.

**Implementation**:
Add a `reframe()` method to `FullSceneFramer` in `whisperjav/modules/subtitle_pipeline/framers/full_scene.py` that delegates to `VadGroupedFramer`:

```python
def reframe(
    self,
    audio: np.ndarray,
    sample_rate: int,
    max_group_duration_s: float,
    **kwargs: Any,
) -> FramingResult:
    """Step-down: split the full scene into VAD-grouped frames."""
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer
    fallback = VadGroupedFramer(max_group_duration_s=max_group_duration_s)
    try:
        return fallback.frame(audio, sample_rate, **kwargs)
    finally:
        fallback.cleanup()
```

**Files**: `whisperjav/modules/subtitle_pipeline/framers/full_scene.py` (~20 lines added)
**Who benefits**: Every user running assembly mode with step-down enabled. Scene failures that currently produce `None` results get a second chance with finer temporal framing.

### WIN-2: PassthroughCleaner Registration Completeness

**Gap closed**: Ensuring every factory has a "none/passthrough" option
**Effort**: 1 hour
**Impact**: Low-medium — enables aligner-free + cleaner-free workflows for testing and for models that produce clean output.

**Status**: Already done. `PassthroughCleaner` is registered as `"passthrough"` in the cleaner factory and `NoneAligner` is registered as `"none"` in the aligner factory. No action needed.

### WIN-3: Enable Step-Down by Default in QwenPipeline

**Gap closed**: A2 deviation D2 (YAML defaults step-down to false)
**Effort**: 1-2 hours (config change + validation testing)
**Impact**: Medium — step-down is the pipeline's primary resilience mechanism, but it defaults to off. Once WIN-1 lands (FullSceneFramer.reframe()), enabling step-down by default becomes safe.

**Dependency**: WIN-1 must land first (otherwise step-down for the default framer is a no-op).

**Files**:
- `whisperjav/pipelines/qwen_pipeline.py`: Change `stepdown_enabled` default to `True` (constructor, ~1 line)
- `whisperjav/config/v4/ecosystems/pipelines/decoupled.yaml`: Change `stepdown.enabled: true`
- `whisperjav/pipelines/decoupled_pipeline.py`: Change `stepdown_enabled` default to `True` (constructor, ~1 line)

**Who benefits**: All users. Scenes that fail on the first pass automatically get a second attempt with tighter framing, instead of silently producing empty subtitles.

### WIN-4: GenericTextCleaner — Model-Agnostic Baseline Cleaner

**Effort**: 4-8 hours
**Impact**: Medium — when adding a new TextGenerator (e.g., Transformers-based), users need a cleaner that handles universal ASR artifacts without being tied to Qwen3's specific patterns.

**Implementation**: Create `whisperjav/modules/subtitle_pipeline/cleaners/generic.py` with a `GenericTextCleaner` that handles:
- Whitespace normalization
- Repeated character sequences (e.g., "ああああああ" beyond reasonable interjection length)
- Basic hallucination patterns common across ASR models (repeated phrases)
- Language-specific rules loaded from config

Register as `"generic"` in `cleaners/factory.py`.

**Files**:
- `whisperjav/modules/subtitle_pipeline/cleaners/generic.py` (~80-120 lines)
- `whisperjav/modules/subtitle_pipeline/cleaners/factory.py` (1 line addition to registry)

**Who benefits**: Anyone integrating a new ASR model. Without this, they must either use the Qwen3-specific cleaner (which may not match their model's artifacts) or the passthrough cleaner (which provides no cleaning at all).

---

## 4. Strategic Builds (2-8 weeks effort)

### BUILD-1: WhisperSegmentFramer — The Flagship Cross-Model Workflow [DEFERRED]

> **Status**: Explicitly deferred to a later phase. Will be added when the composability UX and foundation are mature.

**Gap closed**: GAP-1 (missing whisper-segment framer) + GAP-2 (non-Qwen generator pathway validation)
**Effort**: 2-3 weeks
**Impact**: Very high — this is the single most compelling composability workflow. Use Whisper (fast, well-understood) to identify WHERE dialogue is, then use a stronger model (Qwen3, or a future model) to transcribe WHAT was said.

**Why it matters for JAV users**: Whisper is excellent at detecting speech boundaries in noisy audio (moaning, BGM, ambient sound) but mediocre at transcribing conversational Japanese dialogue. Qwen3-ASR is excellent at Japanese text but processes full scenes without precision on where speech starts/ends. Combining them gives the best of both worlds.

**Architecture**:

```
Audio → Scene Detection → WhisperSegmentFramer → frames[]
    → Qwen3TextGenerator → texts[]
    → Qwen3TextCleaner → cleaned[]
    → Qwen3ForcedAligner → words[]
    → Sentinel → Reconstruct → Harden → SRT
```

**Implementation**:

1. `whisperjav/modules/subtitle_pipeline/framers/whisper_segment.py` (~150-200 lines):
   - Constructor takes Whisper model size (tiny/base/small/medium/large)
   - `frame()`: Run Whisper transcription on scene audio, extract segment boundaries as `TemporalFrame` objects
   - Optionally carry Whisper's text in `frame.text` (for comparison or as context to the generator)
   - `cleanup()`: Unload Whisper model
   - `reframe()`: Re-run with a smaller Whisper model or with VAD-only fallback

2. `whisperjav/modules/subtitle_pipeline/framers/factory.py` (1 line: register `"whisper-segment"`)

3. VRAM design question: Whisper needs to be loaded to produce frames, then unloaded before the generator loads. The orchestrator's current flow handles this correctly because framing happens in Step 1 (before generator.load() in Steps 2-4). The framer can load Whisper in `frame()` and unload it in `cleanup()`, OR keep it loaded and rely on the natural lifecycle. Since framing happens once per pass, load-in-frame/unload-in-cleanup is the clean pattern.

**VRAM budget** (worst case with WhisperSegmentFramer + Qwen3):
- Step 1: Whisper small (~1GB) for framing
- Step 1 cleanup: Whisper unloaded
- Steps 2-4: Qwen3 text-only (~3.5GB) for generation
- Steps 2-4 cleanup: Qwen3 unloaded
- Steps 5-7: Qwen3 ForcedAligner (~1.2GB) for alignment
- Peak VRAM: ~3.5GB (sequential, never concurrent)

**Validation**: Process the same video with:
1. `full-scene` framer + Qwen3 (current default)
2. `whisper-segment` framer + Qwen3 (new)
3. Compare SRT quality, timing accuracy, and processing time

**Files**:
- `whisperjav/modules/subtitle_pipeline/framers/whisper_segment.py` (~200 lines)
- `whisperjav/modules/subtitle_pipeline/framers/factory.py` (1 line)
- Tests: `tests/test_whisper_segment_framer.py`

**Who benefits**: Users with BGM-heavy content, multi-speaker scenes, or content where Whisper's speech detection is better than raw scene detection + full-scene framing.

### BUILD-2: TransformersTextGenerator — HuggingFace Model Support

**Gap closed**: GAP-2 (no non-Qwen TextGenerator)
**Effort**: 1-2 weeks
**Impact**: High — validates the "model-agnostic" claim. If a second generator works without orchestrator changes, the architecture is proven.

**Why it matters**: HuggingFace hosts dozens of Japanese ASR models. Users should be able to swap in any `AutoModelForSpeechSeq2Seq` model without writing adapter code. This also enables community experimentation — users can try fine-tuned models for specific content types.

**Implementation**:

`whisperjav/modules/subtitle_pipeline/generators/transformers_asr.py` (~150-200 lines):

```python
class TransformersTextGenerator:
    """TextGenerator backed by any HuggingFace AutoModelForSpeechSeq2Seq."""

    def __init__(
        self,
        model_id: str,  # e.g., "openai/whisper-large-v3-turbo"
        device: str = "auto",
        dtype: str = "auto",
        language: str = "ja",
        task: str = "transcribe",
        batch_size: int = 1,
        max_new_tokens: int = 448,
    ): ...

    def load(self) -> None:
        """Load model + processor via AutoModelForSpeechSeq2Seq."""

    def unload(self) -> None:
        """Delete model + processor, clear CUDA cache."""

    def generate(self, audio_path, language, context, **kw) -> TranscriptionResult: ...
    def generate_batch(self, audio_paths, language, contexts, **kw) -> list[TranscriptionResult]: ...
    def cleanup(self) -> None: ...
```

Key implementation details:
- Audio loading via `librosa` or `soundfile` (already in dependencies)
- Processor handles feature extraction (log-mel spectrogram)
- Generation kwargs passed through for model-specific tuning
- Batch support via `pipeline()` or manual batching
- `generate_batch()` respects `batch_size` for GPU memory management

**Registry**: Add `"transformers"` to `generators/factory.py`

**Validation question Q1 answer confirmed**: ~150-200 lines. Orchestrator needs zero changes.

**Files**:
- `whisperjav/modules/subtitle_pipeline/generators/transformers_asr.py` (~150-200 lines)
- `whisperjav/modules/subtitle_pipeline/generators/factory.py` (1 line)
- Tests: `tests/test_transformers_generator.py`

**Who benefits**: Users who want to try different ASR models. Community contributors who have fine-tuned models for specific content. Researchers comparing model quality.

### BUILD-3: CTC Forced Aligner — Alternative Timestamp Alignment [DEPRIORITIZED]

**Effort**: 2-3 weeks
**Impact**: Medium — provides a TextAligner alternative. Not blocking for model independence since the Qwen3 ForcedAligner is already model-agnostic.

**Clarification (C2)**: The Qwen3 ForcedAligner, despite its name, is model-agnostic — it works with any model's text output, not just Qwen3-generated text. The `merge_master_with_timestamps()` logic reconciles punctuated text with unpunctuated aligner tokens, which is a universal text normalization problem, not a Qwen-specific one. A CTC-based aligner is therefore a nice-to-have alternative (different alignment algorithm, different failure modes) rather than a prerequisite for model independence.

**Implementation**:

`whisperjav/modules/subtitle_pipeline/aligners/ctc.py` (~200-250 lines):

```python
class CTCForcedAligner:
    """TextAligner using CTC-based forced alignment (wav2vec2/MMS)."""

    def __init__(
        self,
        model_id: str = "facebook/mms-1b-all",  # or wav2vec2-large-xlsr-japanese
        device: str = "auto",
        dtype: str = "auto",
    ): ...

    def load(self) -> None:
        """Load wav2vec2 model + processor."""

    def align(self, audio_path, text, language, **kw) -> AlignmentResult:
        """CTC forced alignment: text → character positions → word boundaries."""

    def align_batch(self, audio_paths, texts, language, **kw) -> list[AlignmentResult]: ...
```

The CTC alignment algorithm:
1. Tokenize text into characters (for Japanese: each character is a token)
2. Run audio through wav2vec2 to get frame-level CTC probabilities
3. Viterbi alignment: find optimal path mapping characters to frames
4. Group character timestamps into word boundaries

**Registry**: Add `"ctc"` to `aligners/factory.py`

**Files**:
- `whisperjav/modules/subtitle_pipeline/aligners/ctc.py` (~200-250 lines)
- `whisperjav/modules/subtitle_pipeline/aligners/factory.py` (1 line)
- Tests: `tests/test_ctc_aligner.py`

**Who benefits**: Users who want an alternative alignment algorithm when the Qwen3 ForcedAligner collapses on specific content. Users who prefer a lighter-weight aligner without the Qwen3 model dependency (smaller VRAM footprint).

### BUILD-4: HTTPTextGenerator — vLLM and Cloud API Support

**Effort**: 1-2 weeks
**Impact**: Medium-high — unlocks cloud ASR and self-hosted vLLM inference servers.

**Implementation**:

`whisperjav/modules/subtitle_pipeline/generators/http_generator.py` (~150-200 lines):

```python
class HTTPTextGenerator:
    """TextGenerator that calls an HTTP ASR endpoint."""

    def __init__(
        self,
        endpoint_url: str,  # e.g., "http://localhost:8000/v1/audio/transcriptions"
        api_key: Optional[str] = None,
        model: Optional[str] = None,  # for multi-model servers
        timeout: float = 60.0,
        max_concurrent: int = 4,
    ): ...

    def load(self) -> None:
        """No-op or connection pool warmup."""

    def unload(self) -> None:
        """No-op or connection pool cleanup."""

    def generate(self, audio_path, language, context, **kw) -> TranscriptionResult:
        """POST audio file to endpoint, parse response."""

    def generate_batch(self, audio_paths, language, contexts, **kw) -> list[TranscriptionResult]:
        """Concurrent HTTP requests (asyncio or ThreadPoolExecutor)."""
```

Supports two API patterns:
1. **OpenAI-compatible** (`/v1/audio/transcriptions`): Works with vLLM, Whisper servers, any OpenAI-API-compatible endpoint
2. **Raw endpoint**: POST audio binary, receive text response (configurable via `response_format`)

The VRAM lifecycle is interesting here: `load()`/`unload()` are no-ops because the server manages its own VRAM. This means the orchestrator's VRAM swap pattern (unload generator before loading aligner) still works correctly — it just does nothing on the generator side.

**Registry**: Add `"http"` to `generators/factory.py`

**Files**:
- `whisperjav/modules/subtitle_pipeline/generators/http_generator.py` (~150-200 lines)
- `whisperjav/modules/subtitle_pipeline/generators/factory.py` (1 line)

**Who benefits**: Users with vLLM servers running Qwen3-ASR or other audio models. Users who want to use cloud ASR (Gemini, GPT-4o audio) for transcription. Users with high-VRAM servers separate from their editing workstation.

### BUILD-5: Equivalence Validation Framework

**Gap closed**: A2 deferred item (Phase 4 equivalence validation status unclear)
**Effort**: 1-2 weeks
**Impact**: Medium — critical for confidence in the strangulation (did we actually match the old code's output?) and for future model comparisons.

**Implementation**:

A lightweight benchmarking tool that compares SRT output against a reference:

`whisperjav/tools/subtitle_benchmark.py` (~200-300 lines):

```python
class SubtitleBenchmark:
    """Compare pipeline output against reference SRT."""

    def compare(self, generated_srt: Path, reference_srt: Path) -> BenchmarkResult:
        """
        Metrics:
        - Text similarity (character-level edit distance, normalized)
        - Timing accuracy (mean absolute error of start/end timestamps)
        - Coverage (% of reference lines that have a matching generated line)
        - Over-generation (generated lines with no reference match)
        """

    def batch_compare(self, pairs: list[tuple[Path, Path]]) -> BatchResult:
        """Aggregate metrics across multiple files."""
```

CLI interface:
```bash
whisperjav-benchmark --generated output.srt --reference ground_truth.srt
whisperjav-benchmark --generated-dir output/ --reference-dir ground_truth/ --report report.json
```

**Files**:
- `whisperjav/tools/subtitle_benchmark.py` (~200-300 lines)
- `whisperjav/tools/__init__.py`
- Entry point in `pyproject.toml`: `whisperjav-benchmark`

**Who benefits**: Developers validating model changes. Users comparing different pipeline configurations. Quality assurance for releases.

---

## 5. Speculative / Future (Research-Level)

### SPEC-1: CompositeTextGenerator — Multi-Model Ensemble at Generator Level

**Concept**: Run 2+ TextGenerators on the same frames and select the best output per scene.

```python
class CompositeTextGenerator:
    """Runs multiple generators and selects the best output per scene."""

    def __init__(
        self,
        generators: list[TextGenerator],
        selector: OutputSelector,  # "longest", "voting", "llm_judge"
    ): ...
```

**Challenges**:
- VRAM: Cannot load 2 generators simultaneously on consumer GPUs. Must serialize: load A → generate → unload A → load B → generate → unload B → select. This doubles VRAM swap overhead.
- Selection heuristics: "Longest output" is a surprisingly good heuristic for Japanese ASR (models that hallucinate tend to produce shorter outputs). "LLM judge" would use a small LLM to score coherence. "Voting" requires 3+ generators.
- Latency: 2x or more processing time. Only worthwhile for high-quality batch processing.

**Research needed**: Is the quality improvement worth the 2x latency? For which content types? Can we predict which scenes benefit from ensemble and only run multiple generators on those?

### SPEC-2: ContentAdaptiveFramer — Dynamic Strategy Selection

**Concept**: Analyze scene audio characteristics (SNR, speech ratio, BGM presence) and select the optimal framer per scene.

```python
class ContentAdaptiveFramer:
    """Selects framer strategy based on audio analysis."""

    def frame(self, audio, sample_rate, **kw) -> FramingResult:
        features = self._analyze(audio, sample_rate)
        if features.speech_ratio < 0.1:
            return self._skip_frame()  # No speech detected
        elif features.snr < 10:
            return self._vad_framer.frame(audio, sample_rate)  # Noisy: use VAD
        else:
            return self._full_scene_framer.frame(audio, sample_rate)  # Clean: full scene
```

**Why this matters for JAV**: JAV audio varies dramatically within a single video — quiet dialogue scenes, loud BGM montages, ambient sounds, etc. A single framing strategy is a compromise. Adaptive framing could skip non-speech scenes entirely (saving 158 seconds of wasted GPU time, as observed by the user) and use tighter framing for noisy scenes.

**Research needed**: What audio features reliably predict optimal framer strategy? Can this be done cheaply (in Step 1, before GPU models load)?

### SPEC-3: Streaming/Incremental Pipeline

**Concept**: Process scenes as they become available from scene detection, rather than waiting for all scenes to be detected first.

**Why**: For long videos (2+ hours), the current pipeline detects ALL scenes first, then processes them all. Streaming would allow the user to see early results while later scenes are still being detected.

**Challenges**: The orchestrator's batch-oriented VRAM lifecycle (load generator → generate ALL scenes → unload) conflicts with streaming. Would need per-scene or per-batch lifecycle management with configurable batch sizes.

**Verdict**: Architecturally possible but requires significant orchestrator changes. Deferred until there is user demand for it.

### SPEC-4: Speaker Diarization Integration

**Concept**: Add a speaker diarization stage between VAD and ASR, then tag subtitle lines with speaker identifiers.

**Why for JAV**: Dialogue between multiple characters benefits from speaker identification. "Who is speaking?" is a common user question.

**Integration point**: This would be a new protocol (`SpeakerDiarizer`) between framing and generation, or a metadata annotation on `TemporalFrame`. The orchestrator would need to pass speaker IDs through to reconstruction.

**Research needed**: How well do diarization models work on JAV audio (overlapping speech, varying recording quality)? What's the VRAM cost?

### SPEC-5: Iterative Refinement Pipeline

**Concept**: Run the full pipeline, identify low-confidence or sentinel-flagged scenes, then re-process only those scenes with different parameters (longer context window, different model, manual framing).

**Why**: The sentinel already identifies problematic scenes. Instead of just recovering with proportional redistribution, we could re-attempt with a fundamentally different strategy.

**Architecture**: This is a meta-pipeline that wraps the existing pipeline:
```
Pass 1: Default config → identify flagged scenes
Pass 2: Flagged scenes only, different generator/framer → merge
Pass 3: (Optional) Human review of remaining issues
```

The step-down mechanism (WIN-1/WIN-3) is a lightweight version of this concept. Full iterative refinement would support arbitrary strategy changes between passes.

---

## 6. Validation Questions — Concrete Roadmap Items

### Q1: "Can I add a HuggingFace anime ASR model?"

**A1 audit answer**: ~150-200 lines for a new TextGenerator. Orchestrator needs zero changes. Effort: 4-8 hours.

**Roadmap item**: BUILD-2 (TransformersTextGenerator)

**Concrete steps**:
1. Implement `TransformersTextGenerator` in `generators/transformers_asr.py` (~150-200 lines)
2. Register `"transformers"` in `generators/factory.py` (1 line)
3. User runs: `whisperjav video.mp4 --pipeline decoupled --generator transformers --generator-config '{"model_id": "user/anime-asr-v1"}'`
4. Or via YAML config:
```yaml
kind: Pipeline
pipeline:
  generator:
    backend: transformers
    model_id: "user/anime-asr-v1"
    device: auto
  framer: full-scene
  cleaner: passthrough  # or generic (WIN-4)
  aligner: qwen3         # Qwen3 ForcedAligner is model-agnostic — works with any model's text
```

**Effort estimate**: 4-8 hours for the generator. 0 hours for orchestrator changes. 2-4 hours for integration testing. Total: 1-2 days.

### Q2: "Can users mix and match components?"

**A1 audit answer**: Python API is excellent. CLI/GUI surface exists via `--pipeline decoupled`. No runtime swapping (by design).

**Roadmap item**: Three layers of composability exposure.

**Layer 1 — Already works (CLI)**:
```bash
whisperjav video.mp4 --pipeline decoupled \
  --generator qwen3 \
  --framer full-scene \
  --cleaner passthrough \
  --aligner none
```

**Layer 2 — YAML config (already works)**:
```yaml
kind: Pipeline
pipeline:
  generator:
    backend: qwen3
    model_id: "Qwen/Qwen3-ASR-1.7B"
  framer:
    backend: vad-grouped
    max_group_duration_s: 15.0
  cleaner: passthrough
  aligner: qwen3
  stepdown:
    enabled: true
    fallback_group_s: 6.0
```

**Layer 3 — GUI exposure (not built, from COMPOSABILITY_DESIGN.md)**:
- The 5-tab Customize modal design exists in the COMPOSABILITY_DESIGN.md document
- Implementation requires GUI frontend changes (~500 lines JS)
- This is a UI project, not an architecture project

**Effort estimate**: Layer 1 and 2 already work. Layer 3 is 2-3 weeks of GUI work (Phase 2-3 of COMPOSABILITY_DESIGN.md).

> **Priority (C5)**: CLI/GUI composability exposure is elevated to **high priority** (Tier 1). The infrastructure exists but is invisible to users. Exposing it is the highest-leverage UX investment and should proceed immediately after the Tier 0 foundation.

### Q3: "Can I do dual-VAD fusion (Silero + TEN)?"

**A1 audit answer**: Possible as a CompositeFramer backend without touching orchestrator. ~100 lines. Factory needs parameterized composition support.

**Roadmap item**: A `CompositeFramer` that merges speech regions from multiple VAD backends.

**Implementation**:

`whisperjav/modules/subtitle_pipeline/framers/composite.py` (~100-150 lines):

```python
class CompositeFramer:
    """Merges frames from multiple TemporalFramers."""

    def __init__(
        self,
        framers: list[TemporalFramer],
        strategy: str = "intersection",  # "intersection", "union", "voting"
    ): ...

    def frame(self, audio, sample_rate, **kw) -> FramingResult:
        all_results = [f.frame(audio, sample_rate, **kw) for f in self.framers]
        return self._merge(all_results)
```

Merge strategies:
- **Intersection**: Keep only time regions where ALL framers agree there is speech. Conservative — reduces false positives.
- **Union**: Keep time regions where ANY framer detects speech. Aggressive — reduces missed speech.
- **Voting**: Keep regions where majority of framers agree. Balanced.

**Factory challenge**: The current factory pattern (`TemporalFramerFactory.create(name, **kwargs)`) creates a single backend. A composite framer needs a list of backends. Options:
1. Special-case in factory: `create("composite", framers=["vad-grouped:ten", "vad-grouped:silero"])` with string-based nested creation
2. Python API only: Users construct `CompositeFramer` directly and pass it to the orchestrator
3. YAML config support: `framer: {backend: composite, framers: [{backend: vad-grouped, segmenter_backend: ten}, {backend: vad-grouped, segmenter_backend: silero}]}`

**Effort estimate**: 100-150 lines for CompositeFramer. 50-100 lines for factory support. 2-3 days total.

---

## 7. Risk Assessment

### Risk 1: stable_whisper Dependency Lock-In

**What**: `reconstruction.py` depends on `stable_whisper.transcribe_any()` to convert word dicts into `WhisperResult` with Japanese regrouping. Every downstream consumer (SRT generation, stitching, analytics) expects `WhisperResult`.

**Impact**: If we want to support output formats other than `WhisperResult` (e.g., for models that produce their own structured output), we cannot — the pipeline is locked to stable-ts's data model.

**Mitigation**: This is an acknowledged design choice (GAP-4). The `WhisperResult` is a well-understood, feature-rich data structure with SRT/VTT export, word-level access, and regrouping capabilities. Replacing it would require rewriting all downstream consumers. The correct approach is to keep `WhisperResult` as the pipeline's internal representation and provide export adapters if needed.

**Likelihood of becoming a problem**: Low. stable-ts is actively maintained (our fork), and `WhisperResult` is a good fit for subtitle data.

### Risk 2: Protocol Validation with Only One Implementation

**What**: All 4 protocols (`TextGenerator`, `TextAligner`, `TextCleaner`, `TemporalFramer`) have real implementations, but only one "heavy" implementation each (Qwen3). The protocol contracts have only been validated against Qwen3's behavior.

**Impact**: A new backend might discover protocol ambiguities. For example: Does `generate_batch()` guarantee ordering? What happens if `generate()` returns empty text for a non-silent scene? What does `align()` do with text that contains characters the aligner cannot handle?

**Mitigation**: BUILD-2 (TransformersTextGenerator) is the critical validation. If a second generator works without orchestrator changes, the protocols are proven. If it requires changes, we learn where the contracts need tightening.

**Recommendation**: BUILD-2 remains important for protocol validation, but is lower urgency than composability UX (Tier 1). Note that the aligner protocol is already validated as model-agnostic (C2) — only the generator protocol needs a second implementation to be stress-tested.

### Risk 3: VRAM Accounting for Multi-Model Workflows

**What**: The orchestrator's VRAM lifecycle assumes a simple pattern: generator load/unload, then aligner load/unload. Multi-model workflows (WhisperSegmentFramer + Qwen3Generator + CTC Aligner) add a third VRAM occupant (the framer's Whisper model).

**Impact**: On 8GB GPUs, Whisper small (1GB) + Qwen3 text-only (3.5GB) + Qwen3 ForcedAligner (1.2GB) = 5.7GB if all loaded simultaneously. If framing doesn't unload before generation loads, OOM.

**Mitigation**: The current architecture handles this correctly because framing (Step 1) completes before generation (Steps 2-4) begins. As long as framers unload their models in `cleanup()` (called between pipeline passes) or immediately after `frame()` returns, VRAM is safe. The WhisperSegmentFramer implementation (BUILD-1) must follow this pattern.

**Recommendation**: Add a VRAM budget assertion in the orchestrator's debug mode that logs peak VRAM at each step transition.

### Risk 4: Japanese Regrouping Quality Across Models

**What**: `reconstruct_from_words()` uses stable-ts's regrouping with Japanese-aware rules (particle boundaries, etc.). This works well with Qwen3's output because Qwen3 produces natural Japanese text. Other models may produce text that trips up regrouping — e.g., missing particles, incorrect character types, or mixed-language output.

**Impact**: Bad regrouping = subtitles that break in the middle of sentences, or subtitles that are too long (no breaks at all).

**Mitigation**: The `TextCleaner` stage exists specifically to normalize model-specific artifacts before reconstruction. A good cleaner for a new model should produce text that regrouping handles well. Additionally, regrouping is configurable via stable-ts parameters — the pipeline could expose regrouping config as an option.

### Risk 5: Community Adoption vs. Maintenance Burden

**What**: Each new backend (generator, framer, cleaner, aligner) is a maintenance obligation. If we build 5 generators for 5 model families, we have 5 things to keep working as upstream libraries update.

**Impact**: Backends break silently when upstream dependencies change (e.g., transformers API changes, wav2vec2 model format changes).

**Mitigation**:
1. Keep backends thin — adapters, not implementations. The less logic in a backend, the less can break.
2. Lazy imports — backends that aren't used don't need their dependencies installed.
3. Community ownership — backends contributed by users can be maintained by users.
4. Validation tests — `python -m whisperjav.installer.validation` already catches configuration drift; extend to backend smoke tests.

### Risk 6: The "Second System Effect"

**What**: The temptation to over-engineer the pipeline for hypothetical future needs, adding complexity that serves no current user.

**Impact**: Increased maintenance burden, harder onboarding for contributors, slower iteration.

**Mitigation**: Every roadmap item in this document includes a "Who benefits" section. If the answer is "nobody currently," it should not be built. The speculative items (Section 5) are explicitly labeled as research-level and should only be pursued when there is concrete demand.

---

## 8. Recommended Execution Order

> **Priority Decision**: Tier 0 (step-down foundation) and composability exposure (formerly Tier 4) are the immediate priorities. The Qwen3 ForcedAligner's model-agnosticism (C2) removes the urgency for alternative aligners, and the WhisperSegmentFramer (C1) is explicitly deferred. The middle tiers (new generators, new aligners, new framers) are valuable but not blocking — the existing pipeline with a single generator + model-agnostic aligner already serves users well. What's missing is **discoverability**: users can't find or use what already exists.

### Tier 0: Foundation (Week 1)

These items are prerequisites that unblock everything else and should be done first.

| Item | Effort | Unblocks |
|------|--------|----------|
| **WIN-1**: FullSceneFramer.reframe() | 2-4 hours | WIN-3 (step-down by default) |
| **WIN-3**: Enable step-down by default | 1-2 hours | General pipeline resilience |

**Rationale**: Step-down is the pipeline's primary resilience mechanism. With `FullSceneFramer.reframe()` and step-down enabled by default, every user gets automatic failure recovery. This is the highest impact-to-effort ratio item on the entire roadmap.

### Tier 1: Composability Exposure — CLI & GUI (Weeks 2-5) [ELEVATED PRIORITY]

> **Elevated from Tier 4** per clarification C5. The pipeline infrastructure already works — framers, generators, cleaners, aligners, YAML config, CLI flags. But users cannot discover or use it. This is the primary adoption barrier and the highest-leverage UX investment.

| Item | Effort | Depends On |
|------|--------|------------|
| COMPOSABILITY_DESIGN.md Phase 1 (Backend Schema + New Pipeline Parameters) | ~200 lines | Tier 0 |
| COMPOSABILITY_DESIGN.md Phase 5 (CLI Help Regrouping) | ~50 lines | Phase 1 |
| COMPOSABILITY_DESIGN.md Phases 2-3 (GUI Tab Restructuring + Mode Adaptation) | ~600 lines JS | Phase 1 |
| COMPOSABILITY_DESIGN.md Phase 4 (Profiles) | ~300 lines | Phases 1-3 |
| Q3: CompositeFramer (dual-VAD fusion) | 2-3 days | None (standalone framer) |
| **Option C: Composable Reconstructor** | 1-2 days | Phase 1 (backend schema) |

**Rationale**: Even with a single generator (Qwen3), there are 4 framers, 2 cleaners, 2 aligners, YAML pipeline configs, step-down tuning, hardening modes, and sentinel parameters. All of these exist today but are invisible. Surfacing them through good CLI help, GUI component pickers, and profile presets turns the pipeline from a developer tool into a user-facing product. This does NOT require waiting for BUILD-2 or BUILD-1 — the existing components already offer meaningful user choices.

#### Option C: Composable Reconstructor Protocol (Future)

> **Status**: Deferred. Currently handled by inline branching in the orchestrator
> (Branch A → `REGROUP_JAV`, Branch B/vad_only → `REGROUP_VAD_ONLY`).
> Elevate to implementation when a 3rd regrouping strategy is needed.

**Problem**: Reconstruction (word dicts → WhisperResult with subtitle-level segments)
is currently a hardcoded function (`reconstruct_from_words()`) called directly by the
orchestrator. The orchestrator selects a regroup string based on `TimestampMode`, but
this is an if/else chain, not a composable component.  As more regrouping strategies
emerge (e.g., pre-segmented inference for aligner-free modes, or LLM-guided subtitle
boundaries), the orchestrator's Branch B will accumulate mode-specific logic.

**Design**: Add `Reconstructor` as the 5th protocol alongside Framer, Generator,
Cleaner, and Aligner.

```python
@runtime_checkable
class Reconstructor(Protocol):
    """Converts word-level dicts into a WhisperResult with subtitle segments."""

    def reconstruct(
        self,
        words: list[dict[str, Any]],
        audio_path: Path,
    ) -> "stable_whisper.WhisperResult": ...

    def cleanup(self) -> None: ...
```

**Backends** (factory-registered):

| Backend | Regroup | Use Case |
|---------|---------|----------|
| `RegroupJavReconstructor` | `REGROUP_JAV` (gap + merge + caps) | Branch A (aligned, real timestamps) |
| `RegroupVadOnlyReconstructor` | `REGROUP_VAD_ONLY` (punct + caps only) | Branch B/vad_only (synthetic timestamps) |
| `PreSegmentedReconstructor` | `regroup=False`, pre-segmented inference | Future: one segment per pseudo-word |
| `PassthroughReconstructor` | `regroup=False` | Debugging / raw output |

**Orchestrator change**: Replace the current regroup branching with:
```python
result = self.reconstructor.reconstruct(words, audio_path)
```

**Config wiring**: Factory selects backend based on `timestamp_mode` default,
overridable via CLI `--reconstructor` and GUI dropdown.

**Why deferred**: The current 2-way branch (`REGROUP_JAV` vs `REGROUP_VAD_ONLY`)
is manageable with inline logic.  Protocol extraction is warranted when:
- A 3rd strategy is needed (pre-segmented, LLM-guided, etc.)
- Users want to select reconstruction strategy independently of timestamp mode
- The orchestrator's Branch B grows beyond ~20 lines of mode-specific logic

**What users gain immediately**:
- Framer selection (full-scene vs vad-grouped vs srt-source vs manual) visible in GUI
- Cleaner bypass (passthrough) for models that produce clean output
- Aligner bypass (none) for text-only workflows
- Step-down tuning (fallback_group_s, max retries) exposed as sliders
- YAML profile presets: "quality", "speed", "noisy-audio"

### Tier 2: Architecture Validation (Weeks 5-8) [DEPRIORITIZED]

Prove the architecture with a second generator. Important but not blocking for current users.

| Item | Effort | Unblocks |
|------|--------|----------|
| **WIN-4**: GenericTextCleaner | 4-8 hours | BUILD-2 (new generators need a cleaner) |
| **BUILD-2**: TransformersTextGenerator | 1-2 weeks | Validates "model-agnostic" claim |

**Rationale**: If BUILD-2 works without orchestrator changes, the architecture is proven and the model-agnostic vision is real. If it requires orchestrator changes, we learn early and cheaply what needs fixing. This validates the protocol contracts with a second implementation. Deprioritized because (a) the aligner is already model-agnostic (C2), so only the generator protocol needs validation, and (b) composability UX (Tier 1) has a higher immediate impact.

### Tier 3: Ecosystem Expansion (Weeks 8-12) [DEPRIORITIZED]

Broaden model support and add evaluation tooling. These are independent items that can be pursued based on demand.

| Item | Effort | Depends On | Notes |
|------|--------|------------|-------|
| **BUILD-5**: Equivalence Validation Framework | 1-2 weeks | None (standalone tool) | Useful for any config comparison |
| **BUILD-4**: HTTPTextGenerator | 1-2 weeks | BUILD-2 (validates generator pattern) | Unlocks vLLM + cloud ASR |
| **BUILD-3**: CTC Forced Aligner | 2-3 weeks | BUILD-2 | Nice-to-have alternative; Qwen3 aligner already model-agnostic (C2) |

**Rationale**: BUILD-5 is the most immediately useful (standalone, no deps). BUILD-4 unlocks cloud/server workflows. BUILD-3 provides an alternative alignment algorithm but is NOT required for model independence since the existing Qwen3 ForcedAligner works with any model's text (C2).

### Tier 4: Cross-Model Framing (No Fixed Timeline) [DEFERRED]

> **Deferred** per clarification C1. Will be revisited when the composability UX is mature and there is concrete demand for cross-model workflows.

| Item | Effort | Depends On |
|------|--------|------------|
| **BUILD-1**: WhisperSegmentFramer | 2-3 weeks | Tier 1 (composability UX to expose it) |

**Rationale**: This is the "killer feature" of the pipeline architecture — combining Whisper's speech detection with Qwen3's Japanese transcription. However, it requires significant effort (2-3 weeks) and benefits most when the composability UX (Tier 1) is already in place so users can actually select it. Explicitly deferred to focus on foundation + UX first.

### Tier 5: Speculative (No Timeline)

Only pursue when there is concrete user demand:

| Item | Trigger |
|------|---------|
| SPEC-1: CompositeTextGenerator | User reports wanting to compare model outputs |
| SPEC-2: ContentAdaptiveFramer | Sufficient data on which framers work best for which content |
| SPEC-3: Streaming Pipeline | User demand for live preview of long video processing |
| SPEC-4: Speaker Diarization | User request for speaker tagging |
| SPEC-5: Iterative Refinement | Step-down proves insufficient for quality recovery |

---

## Appendix A: Current Module Inventory

### Files in `whisperjav/modules/subtitle_pipeline/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 61 | Public API re-exports |
| `protocols.py` | 180 | 4 protocol definitions (runtime-checkable) |
| `types.py` | 211 | Data types: TemporalFrame, TranscriptionResult, AlignmentResult, configs, diagnostics |
| `orchestrator.py` | 1007 | DecoupledSubtitlePipeline: 9-step flow, VRAM lifecycle, step-down, diagnostics |
| `hardening.py` | 426 | 4 timestamp resolution modes, boundary clamping, chronological sort |
| `reconstruction.py` | 94 | Word dicts → WhisperResult via stable-ts transcribe_any() |
| `framers/factory.py` | 72 | TemporalFramerFactory (4 backends registered) |
| `framers/full_scene.py` | 63 | FullSceneFramer (1 frame = entire scene) |
| `framers/vad_grouped.py` | 206 | VadGroupedFramer (VAD + grouping, has reframe()) |
| `framers/srt_source.py` | 134 | SrtSourceFramer (SRT timestamps as frames) |
| `framers/manual.py` | 93 | ManualFramer (user-provided timestamps) |
| `generators/factory.py` | 69 | TextGeneratorFactory (1 backend: qwen3) |
| `generators/qwen3.py` | 200 | Qwen3TextGenerator (text-only mode adapter) |
| `cleaners/factory.py` | 70 | TextCleanerFactory (2 backends: qwen3, passthrough) |
| `cleaners/qwen3.py` | 94 | Qwen3TextCleaner (wraps AssemblyTextCleaner) |
| `cleaners/passthrough.py` | 18 | PassthroughCleaner (no-op) |
| `aligners/factory.py` | 70 | TextAlignerFactory (2 backends: qwen3, none) |
| `aligners/qwen3.py` | 222 | Qwen3ForcedAlignerAdapter (with merge_master_with_timestamps) |
| `aligners/none.py` | 50 | NoneAligner (no-op for aligner-free workflows) |

**Total**: ~3,339 lines across 19 files (excluding `__init__.py` files)

### Integration Surface

| File | Role |
|------|------|
| `whisperjav/pipelines/decoupled_pipeline.py` (793 lines) | Generic entry point: 9-phase outer pipeline with YAML config |
| `whisperjav/pipelines/qwen_pipeline.py` (861 lines) | Qwen-specific entry point: uses orchestrator for assembly mode |
| `whisperjav/modules/alignment_sentinel.py` (417 lines) | Standalone sentinel: collapse detection + 2 recovery strategies |

## Appendix B: Factory Registry State

### Current Registrations

```
TextGeneratorFactory:
  "qwen3" → Qwen3TextGenerator

TemporalFramerFactory:
  "full-scene" → FullSceneFramer
  "vad-grouped" → VadGroupedFramer
  "srt-source" → SrtSourceFramer
  "manual" → ManualFramer

TextCleanerFactory:
  "qwen3" → Qwen3TextCleaner
  "passthrough" → PassthroughCleaner

TextAlignerFactory:
  "qwen3" → Qwen3ForcedAlignerAdapter
  "none" → NoneAligner
```

### After Full Roadmap Execution

```
TextGeneratorFactory:
  "qwen3" → Qwen3TextGenerator
  "transformers" → TransformersTextGenerator        (BUILD-2)
  "http" → HTTPTextGenerator                        (BUILD-4)

TemporalFramerFactory:
  "full-scene" → FullSceneFramer                    (+ reframe(), WIN-1)
  "vad-grouped" → VadGroupedFramer
  "srt-source" → SrtSourceFramer
  "manual" → ManualFramer
  "whisper-segment" → WhisperSegmentFramer          (BUILD-1)
  "composite" → CompositeFramer                     (Q3)

TextCleanerFactory:
  "qwen3" → Qwen3TextCleaner
  "passthrough" → PassthroughCleaner
  "generic" → GenericTextCleaner                    (WIN-4)

TextAlignerFactory:
  "qwen3" → Qwen3ForcedAlignerAdapter              (model-agnostic — works with any model's text)
  "none" → NoneAligner
  "ctc" → CTCForcedAligner                         (BUILD-3, optional alternative)
```
