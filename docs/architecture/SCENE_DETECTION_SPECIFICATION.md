# Scene Detection System — Specification & Developer Guide

**Version**: 1.1 (post-audit corrections)
**Last Updated**: 2026-02-15
**Package**: `whisperjav.modules.scene_detection_backends`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Protocol Contract](#3-protocol-contract)
4. [Data Types](#4-data-types)
5. [Factory](#5-factory)
6. [Backends](#6-backends)
7. [Parameters Reference](#7-parameters-reference)
8. [Developer Guide: Integrating Scene Detection](#8-developer-guide-integrating-scene-detection)
9. [Consumer Adoption Map](#9-consumer-adoption-map)
10. [Error Handling](#10-error-handling)
11. [Utilities](#11-utilities)
12. [Migration from DynamicSceneDetector](#12-migration-from-dynamicscenedetector)

---

## 1. Overview

The scene detection system splits audio files into temporal segments ("scenes") for
downstream ASR processing. It answers the question: **"Where should we split this
audio so each chunk is digestible by the ASR model?"**

### Design Goals

- **Protocol-driven**: All backends implement `SceneDetector` (`@runtime_checkable`)
- **Factory-created**: `SceneDetectorFactory` handles lazy loading and routing
- **Uniform output**: Every backend returns `SceneDetectionResult` containing `SceneInfo` objects
- **Backward compatible**: `to_legacy_tuples()` produces the same `List[Tuple[Path, float, float, float]]` that the deprecated `DynamicSceneDetector` returned
- **Resource-safe**: `cleanup()` releases models and cached state

### Package Location

```
whisperjav/modules/scene_detection_backends/
├── __init__.py            # Public API re-exports
├── base.py                # Protocol, SceneInfo, SceneDetectionResult, SceneDetectionError
├── factory.py             # SceneDetectorFactory (registry + lazy loading)
├── utils.py               # load_audio_unified, save_scene_wav, brute_force_split
├── auditok_backend.py     # AuditokSceneDetector + AuditokSceneConfig
├── silero_backend.py      # SileroSceneDetector + SileroSceneConfig (extends Auditok)
├── semantic_backend.py    # SemanticSceneDetector (wraps SemanticClusteringAdapter)
├── semantic_adapter.py    # SemanticClusteringAdapter (external engine bridge)
└── none_backend.py        # NullSceneDetector (passthrough)
```

---

## 2. Architecture

### High-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Pipeline Consumer                         │
│  (qwen_pipeline, balanced_pipeline, fidelity_pipeline, etc.)     │
│                                                                  │
│  1. scene_opts = features.get("scene_detection", {})             │
│  2. detector = SceneDetectorFactory                              │
│       .create_from_legacy_kwargs(**scene_opts)                   │
│  3. result = detector.detect_scenes(audio, out_dir, basename)    │
│  4. scene_paths = result.to_legacy_tuples()                      │
│  5. metadata = result.to_metadata_dict()                         │
│  6. detector.cleanup()                                           │
└─────────────────┬────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SceneDetectorFactory                           │
│                                                                  │
│  Registry: { "auditok" → AuditokSceneDetector,                   │
│              "silero"  → SileroSceneDetector,                     │
│              "semantic"→ SemanticSceneDetector,                   │
│              "none"    → NullSceneDetector }                      │
│                                                                  │
│  create_from_legacy_kwargs(**kw):                                │
│    method = kw.pop("method", "auditok")                          │
│    return create(method, **kw)                                   │
└─────────────────┬────────────────────────────────────────────────┘
                  │ Lazy import + instantiate
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│              SceneDetector Protocol Implementations               │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ AuditokScene     │  │ SileroScene      │                     │
│  │ Detector         │←─│ Detector         │  (inherits)          │
│  │                  │  │                  │                      │
│  │ Pass1: auditok   │  │ Pass1: auditok   │                     │
│  │ Pass2: auditok   │  │ Pass2: Silero VAD│                     │
│  └──────────────────┘  └──────────────────┘                     │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ SemanticScene    │  │ NullScene        │                     │
│  │ Detector         │  │ Detector         │                     │
│  │                  │  │                  │                      │
│  │ Single-pass MFCC │  │ Full audio as    │                     │
│  │ clustering       │  │ single scene     │                     │
│  └──────────────────┘  └──────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Shared Output Contract                         │
│                                                                  │
│  SceneDetectionResult                                            │
│  ├── scenes: List[SceneInfo]                                     │
│  │   ├── start_sec, end_sec, scene_path, detection_pass          │
│  │   └── .to_legacy_tuple() → (Path, float, float, float)       │
│  ├── method: str                                                 │
│  ├── audio_duration_sec, parameters, processing_time_sec         │
│  ├── coarse_boundaries: Optional[List[Dict]]                     │
│  ├── vad_segments: Optional[List[Dict]]                          │
│  ├── .to_legacy_tuples() → List[(Path, float, float, float)]    │
│  └── .to_metadata_dict() → Dict (for pipeline metadata)         │
└──────────────────────────────────────────────────────────────────┘
```

### Two-Pass Architecture (Auditok & Silero)

```
Audio File
    │
    ▼
┌──────────────────────────────────────────┐
│         Pass 1: Coarse Chapters          │
│                                          │
│  Strategy: Find long silences            │
│  Parameters: pass1_max_silence (1.8s),   │
│              pass1_energy_threshold (38)  │
│  Result: ~5-20 "story line" regions      │
│  Max duration: 2700s (45 min per region) │
└────────────────┬─────────────────────────┘
                 │
    ┌────────────┼─────────────┐
    ▼            ▼             ▼
┌────────┐ ┌────────┐   ┌────────┐
│Region 1│ │Region 2│...│Region N│
└───┬────┘ └───┬────┘   └───┬────┘
    │          │             │
    ▼          ▼             ▼
┌──────────────────────────────────────────┐
│     Per-Region Decision                  │
│                                          │
│  IF min_dur ≤ duration ≤ max_dur:        │
│     → Save directly (detection_pass=1)   │
│  ELSE IF duration > max_dur:             │
│     → Pass 2: Fine split                 │
│  ELSE IF duration < min_dur:             │
│     → Skip (too short)                   │
└────────────────┬─────────────────────────┘
                 │ (oversized regions only)
                 ▼
┌──────────────────────────────────────────┐
│          Pass 2: Fine Chunking           │
│                                          │
│  Auditok: shorter silence, higher energy │
│    pass2_max_silence=0.94s, threshold=50 │
│  Silero: VAD speech boundaries           │
│    silero_threshold=0.08, min_silence=   │
│    1500ms                                │
│                                          │
│  IF no sub-regions found:                │
│     → Brute-force split (fixed chunks)   │
│                                          │
│  Result: scenes (detection_pass=2)       │
└──────────────────────────────────────────┘
```

### Semantic Architecture (Single-Pass)

```
Audio File
    │
    ▼
┌──────────────────────────────────────────┐
│    Semantic Audio Clustering Engine       │
│                                          │
│  1. Extract MFCC features                │
│  2. Agglomerative clustering             │
│  3. Boundary detection + snapping        │
│  4. Min/max duration enforcement         │
│                                          │
│  Natural range: 20s-420s scenes          │
│  detection_pass = 0 (single-pass)        │
│                                          │
│  Each SceneInfo.metadata includes:       │
│    { "context": {...}, "asr_prompt": ""} │
└──────────────────────────────────────────┘
```

---

## 3. Protocol Contract

Defined in `base.py`. All backends **MUST** implement this interface.

```python
@runtime_checkable
class SceneDetector(Protocol):
    @property
    def name(self) -> str:
        """Unique backend identifier (e.g., 'auditok', 'silero', 'semantic', 'none')"""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for GUI display."""
        ...

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        **kwargs,
    ) -> SceneDetectionResult:
        """
        Detect scenes in audio and extract WAV files.

        Args:
            audio_path:     Path to input audio (.wav)
            output_dir:     Directory to write scene_NNNN.wav files
            media_basename: Prefix for output filenames

        Returns:
            SceneDetectionResult (never None)

        Raises:
            SceneDetectionError: On fatal failure (audio load, backend crash)
        """
        ...

    def cleanup(self) -> None:
        """Release loaded models and cached state."""
        ...
```

### Contract Guarantees

| Guarantee | Detail |
|-----------|--------|
| **Return type** | Always `SceneDetectionResult` (never `None`) |
| **Empty scenes** | Valid result — means no speech detected, not an error |
| **WAV files** | Each `SceneInfo.scene_path` points to a PCM16 WAV file in `output_dir` |
| **Filenames** | `{media_basename}_scene_{NNNN:04d}.wav` (0-indexed) |
| **Time coords** | `start_sec` and `end_sec` are relative to the original audio |
| **Immutability** | `detect_scenes()` clears prior state; each call is independent |
| **Cleanup** | `cleanup()` is idempotent and safe to call multiple times |
| **Errors** | `SceneDetectionError` for fatal failures only |

---

## 4. Data Types

### SceneInfo

```python
@dataclass
class SceneInfo:
    start_sec: float                            # Start time (seconds, absolute)
    end_sec: float                              # End time (seconds, absolute)
    scene_path: Optional[Path] = None           # Path to extracted WAV
    detection_pass: int = 0                     # 0=single-pass, 1=coarse/direct, 2=fine/split
    metadata: Dict[str, Any] = field(...)       # Backend-specific

    @property
    def duration_sec(self) -> float: ...        # end_sec - start_sec

    def to_dict(self) -> Dict: ...              # JSON-serializable
    def to_legacy_tuple(self) -> Tuple[Path, float, float, float]: ...
```

**`detection_pass` values:**

| Value | Meaning | Used By |
|-------|---------|---------|
| `0` | Single-pass detection (no two-pass) | Semantic, None |
| `1` | Pass 1 direct save (region fits within max_duration) | Auditok, Silero |
| `2` | Pass 2 fine split (region was oversized) | Auditok, Silero |

### SceneDetectionResult

```python
@dataclass
class SceneDetectionResult:
    scenes: List[SceneInfo]                     # All detected scenes
    method: str                                 # Backend name ("auditok", "silero", etc.)
    audio_duration_sec: float = 0.0             # Total audio duration
    parameters: Dict[str, Any] = field(...)     # Detection parameters used
    processing_time_sec: float = 0.0            # Wall clock time
    coarse_boundaries: Optional[List[Dict]] = None  # Pass 1 data (two-pass only)
    vad_segments: Optional[List[Dict]] = None       # VAD data (Silero only)

    @property
    def num_scenes(self) -> int: ...
    @property
    def total_scene_duration_sec(self) -> float: ...
    @property
    def coverage_ratio(self) -> float: ...

    def to_legacy_tuples(self) -> List[Tuple[Path, float, float, float]]: ...
    def to_metadata_dict(self) -> Dict: ...
```

**`to_metadata_dict()` output schema:**

```python
{
    "scenes_detected": [
        {
            "scene_index": 0,
            "start_time_seconds": 0.0,
            "end_time_seconds": 28.5,
            "duration_seconds": 28.5,
            "detection_pass": 1,
            "filename": "video_scene_0000.wav",
            "path": "/tmp/scenes/video_scene_0000.wav"
        },
        # ...
    ],
    "coarse_boundaries": [ ... ] or None,
    "vad_segments": [ ... ] or None,
    "vad_method": "auditok",       # Same as result.method
    "vad_params": { ... } or None  # Same as result.parameters
}
```

### SceneDetectionError

```python
class SceneDetectionError(Exception):
    """Fatal detection failure (audio load error, backend crash, etc.)."""
```

**Critical distinction:**
- `SceneDetectionError` → Detection **failed**. Cannot continue.
- Empty `result.scenes` → Detection **succeeded**. No speech found. Valid result.

---

## 5. Factory

### SceneDetectorFactory

Located in `factory.py`. Static methods only (no instances).

#### Creating a Detector

```python
from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

# Method 1: Direct creation (typed)
detector = SceneDetectorFactory.create("auditok", max_duration=29.0)

# Method 2: Legacy kwargs bridge (used by all pipelines)
detector = SceneDetectorFactory.create_from_legacy_kwargs(
    method="silero",
    max_duration_s=29.0,
    min_duration_s=0.3,
    silero_threshold=0.08,
)
```

#### `create_from_legacy_kwargs(**kwargs)` — How It Works

1. Pops `method` key (default: `"auditok"`)
2. Passes remaining kwargs to `create(method, **remaining_kwargs)`
3. Backend constructor accepts or silently ignores unrecognized kwargs

This is the **zero-risk migration bridge** — same kwargs that `DynamicSceneDetector` accepted.

#### Backend Registry

| Name | Class | Dependencies | Always Available |
|------|-------|-------------|-----------------|
| `"auditok"` | `AuditokSceneDetector` | `auditok` | Yes (core dep) |
| `"silero"` | `SileroSceneDetector` | `torch` | Yes (core dep) |
| `"semantic"` | `SemanticSceneDetector` | `scikit-learn` | No (optional) |
| `"none"` | `NullSceneDetector` | — | Yes |

#### Introspection Methods

```python
# List all registered backend names
SceneDetectorFactory.list_backends()
# → ["auditok", "silero", "semantic", "none"]

# Check if a backend's dependencies are installed
available, hint = SceneDetectorFactory.is_backend_available("semantic")
# → (True, "") or (False, "pip install scikit-learn")

# Get all backends with availability status
SceneDetectorFactory.get_available_backends()
# → [{"name": "auditok", "display_name": "Auditok (Silence-Based)",
#      "available": True, "install_hint": ""}, ...]
```

---

## 6. Backends

### 6.1 AuditokSceneDetector

**Strategy**: Two-pass silence-based detection.
**Config**: `AuditokSceneConfig` (dataclass)
**Scene range**: `min_duration` to `max_duration` (default 0.3s–29.0s)

Pass 1 uses auditok to find coarse chapter boundaries via long silences.
Pass 2 uses auditok again with tighter parameters to chunk oversized chapters.
Optional assistive processing (bandpass filter + DRC) for Pass 2 on challenging audio.

### 6.2 SileroSceneDetector

**Strategy**: Two-pass — auditok Pass 1 + Silero VAD Pass 2.
**Config**: `SileroSceneConfig` (extends `AuditokSceneConfig`)
**Inherits**: `AuditokSceneDetector` (overrides `_detect_pass2()` only)
**Scene range**: Same as auditok config

Pass 2 uses Silero VAD for speech boundary detection instead of energy-based auditok.
Silero model is loaded once at construction and reused for all regions.
Per-region resampling to 16kHz is done as needed.
**No assistive processing** — Silero is trained on natural audio; bandpass/DRC would
change signal distribution outside its training data.

### 6.3 SemanticSceneDetector

**Strategy**: Single-pass content-aware clustering.
**Config**: `SemanticClusteringConfig` (via adapter)
**Scene range**: 20.0s–420.0s (not 0.3s–29.0s like auditok/silero)
**Dependencies**: scikit-learn (optional)

Uses MFCC features + agglomerative clustering for texture-based segmentation.
Wraps the existing `SemanticClusteringAdapter` for Protocol conformance.
Each `SceneInfo.metadata` includes `context` and `asr_prompt` from the engine.

### 6.4 NullSceneDetector

**Strategy**: Passthrough — full audio as one scene.
**Config**: None (ignores all kwargs)
**Scene range**: Entire audio duration

Used when scene detection should be bypassed (e.g., Qwen pipeline with `method="none"`).
Loads the full audio, saves it as a single WAV, returns one `SceneInfo`.

---

## 7. Parameters Reference

### AuditokSceneConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Core** | | | |
| `max_duration` | float | 29.0 | Maximum scene duration (seconds) |
| `min_duration` | float | 0.3 | Minimum scene duration (seconds) |
| **Pass 1 (Coarse)** | | | |
| `pass1_min_duration` | float | 0.3 | Min region duration |
| `pass1_max_duration` | float | 2700.0 | Max region duration (45 min) |
| `pass1_max_silence` | float | 1.8 | Silence length to split on (seconds) |
| `pass1_energy_threshold` | int | 38 | Energy threshold for speech (dB) |
| **Pass 2 (Fine)** | | | |
| `pass2_min_duration` | float | 0.3 | Min sub-region duration |
| `pass2_max_duration` | float | *derived* | Max sub-region (default: `max(max_duration - 1.0, min_duration)`) |
| `pass2_max_silence` | float | 0.94 | Silence length to split on (seconds) |
| `pass2_energy_threshold` | int | 50 | Energy threshold for speech (dB) |
| **Assist Processing** | | | |
| `assist_processing` | bool | False | Enable bandpass + DRC for Pass 2 |
| `bandpass_low_hz` | int | 200 | Bandpass filter low cutoff (Hz) |
| `bandpass_high_hz` | int | 4000 | Bandpass filter high cutoff (Hz) |
| `drc_threshold_db` | float | -24.0 | DRC threshold (dB) |
| `drc_ratio` | float | 4.0 | DRC compression ratio |
| `drc_attack_ms` | float | 5.0 | DRC attack time (ms) |
| `drc_release_ms` | float | 100.0 | DRC release time (ms) |
| `skip_assist_on_loud_dbfs` | float | -5.0 | Skip assist if louder than this |
| **Fallback** | | | |
| `brute_force_fallback` | bool | True | Use fixed-chunk splitting if Pass 2 fails |
| `brute_force_chunk_s` | float | *derived* | Chunk size (default: max_duration) |
| **Edge** | | | |
| `pad_edges_s` | float | 0.0 | Padding at segment edges (seconds) |
| **Output** | | | |
| `verbose_summary` | bool | True | Log detection summary |
| `force_mono` | bool | True | Convert stereo to mono |

### SileroSceneConfig (extends AuditokSceneConfig)

All auditok parameters above, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `silero_threshold` | float | 0.08 | Speech probability threshold |
| `silero_neg_threshold` | float | 0.15 | Deactivation threshold |
| `silero_min_silence_ms` | int | 1500 | Min silence to split (ms) |
| `silero_min_speech_ms` | int | 100 | Min speech duration (ms) |
| `silero_max_speech_s` | float | 600.0 | Max speech segment (seconds) |
| `silero_min_silence_at_max` | int | 500 | Min silence at max speech (ms) |
| `silero_speech_pad_ms` | int | 200 | Padding around speech (ms) |

**Note**: `assist_processing` is forced to `False` in `SileroSceneConfig.__post_init__()`.

### SemanticClusteringConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_duration` | float | 20.0 | Minimum segment duration |
| `max_duration` | float | 420.0 | Maximum segment duration |
| `snap_window` | float | 5.0 | Boundary snapping window (seconds) |
| `clustering_threshold` | float | 18.0 | Agglomerative clustering distance threshold |
| `sample_rate` | int | 16000 | Processing sample rate |
| `preserve_original_sr` | bool | True | Save WAVs at original sample rate |
| `visualize` | bool | False | Generate visualization PNG |
| `preset` | str | "default" | Preset name |

### Legacy Kwargs Alias Support

The factory bridge (`create_from_legacy_kwargs`) and `AuditokSceneDetector._build_config_from_kwargs`
accept both `_s` suffixed and un-suffixed parameter names:

```python
# These are equivalent:
SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", max_duration_s=29.0)
SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", max_duration=29.0)
```

The `_s` suffix takes precedence when both are present.

Additionally, the bare top-level aliases `max_silence` and `energy_threshold` (without
the `pass1_`/`pass2_` prefix) are supported as fallbacks for `pass1_max_silence` and
`pass1_energy_threshold` respectively. This preserves backward compatibility with
the oldest config format:

```python
# These are equivalent:
SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", max_silence=1.8)
SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", pass1_max_silence=1.8)
```

---

## 8. Developer Guide: Integrating Scene Detection

### Minimal Integration (5 steps)

```python
from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

# 1. Create detector from config
scene_opts = resolved_config["features"].get("scene_detection", {})
detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_opts)

# 2. Run detection
scenes_dir = temp_dir / "scenes"
scenes_dir.mkdir(exist_ok=True)
result = detector.detect_scenes(audio_path, scenes_dir, media_basename)

# 3. Get legacy tuples for downstream ASR
scene_paths = result.to_legacy_tuples()
# scene_paths: [(Path, start_sec, end_sec, duration_sec), ...]

# 4. Optionally extract metadata
metadata = result.to_metadata_dict()

# 5. Cleanup when done
detector.cleanup()
```

### With Overrides (e.g., Qwen safe_chunking)

```python
scene_kwargs = {"method": self.scene_method}

if self.safe_chunking:
    scene_kwargs["min_duration"] = 30   # wider scenes
    scene_kwargs["max_duration"] = 120  # ForcedAligner limit headroom

detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_kwargs)
result = detector.detect_scenes(audio_path, scenes_dir, media_basename)
scene_paths = result.to_legacy_tuples()
detector.cleanup()
```

### Accessing Rich Metadata

```python
result = detector.detect_scenes(audio_path, scenes_dir, basename)

# Per-scene data
for scene in result.scenes:
    print(f"Scene: {scene.start_sec:.2f}s - {scene.end_sec:.2f}s "
          f"(pass={scene.detection_pass})")
    if scene.metadata.get("split_method") == "brute_force":
        print("  ⚠ Brute-force split (no natural boundary found)")

# Aggregate stats
print(f"Coverage: {result.coverage_ratio:.1%}")
print(f"Processing time: {result.processing_time_sec:.1f}s")

# Coarse boundaries (two-pass methods only)
if result.coarse_boundaries:
    for cb in result.coarse_boundaries:
        print(f"Chapter: {cb['start_time_seconds']:.1f}s - "
              f"{cb['end_time_seconds']:.1f}s")

# VAD segments (Silero only)
if result.vad_segments:
    for seg in result.vad_segments:
        print(f"Speech: {seg['start_sec']:.3f}s - {seg['end_sec']:.3f}s")
```

### Lifecycle Management

Scene detectors should be:
1. **Created once** at pipeline construction time (or once per process call for Qwen)
2. **Used** via `detect_scenes()` for each file
3. **Cleaned up** via `cleanup()` when the pipeline is done

The base pipeline (`base_pipeline.py`) includes automatic cleanup with defensive guards:

```python
# base_pipeline.py cleanup() — called automatically
if hasattr(self, 'scene_detector') and self.scene_detector:
    try:
        if hasattr(self.scene_detector, 'cleanup'):
            self.scene_detector.cleanup()
        self.scene_detector = None
    except Exception as e:
        logger.warning(f"Scene detector cleanup failed (non-fatal): {e}")
```

For one-shot usage (like Qwen), call `cleanup()` explicitly after use.

### Writing a New Backend

1. Create `whisperjav/modules/scene_detection_backends/my_backend.py`
2. Implement the `SceneDetector` Protocol:

```python
from pathlib import Path
from .base import SceneDetectionResult, SceneInfo, SceneDetectionError

class MySceneDetector:
    def __init__(self, **kwargs):
        # Accept/ignore kwargs for factory compatibility
        ...

    @property
    def name(self) -> str:
        return "my_method"

    @property
    def display_name(self) -> str:
        return "My Method (Description)"

    def detect_scenes(self, audio_path, output_dir, media_basename, **kwargs):
        # 1. Load audio
        # 2. Detect boundaries
        # 3. Save WAV files via save_scene_wav()
        # 4. Return SceneDetectionResult with SceneInfo list
        ...

    def cleanup(self):
        ...
```

3. Register in `factory.py`:

```python
_BACKEND_REGISTRY["my_method"] = (
    "whisperjav.modules.scene_detection_backends.my_backend.MySceneDetector"
)
```

4. Add dependency info:

```python
_BACKEND_DEPENDENCIES["my_method"] = {
    "packages": ["my_dep"],
    "install_hint": "pip install my_dep",
    "always_available": False,
}
```

---

## 9. Consumer Adoption Map

### How Each Pipeline Creates and Uses Scene Detection

#### QwenPipeline (`qwen_pipeline.py`)

```
Construction:  scene_method set from CLI / config
Usage (Phase 2):
  kwargs = {"method": self.scene_method}
  + safe_chunking overrides:
    ASSEMBLY mode:    min_duration=30, max_duration=120
    Other modes:      min_duration=12, max_duration=90
  detector = SceneDetectorFactory.create_from_legacy_kwargs(**kwargs)
  result = detector.detect_scenes(audio, scenes_dir, basename)
  scene_paths = result.to_legacy_tuples()
  detector.cleanup()  ← explicit, one-shot
Metadata:  method, scenes_detected count, time_sec only
Cleanup:   Manual (detector created/destroyed per process call)
```

**Qwen Input Modes and Scene Detection:**

| Input Mode | Scene Method | max_duration | min_duration | Notes |
|------------|-------------|-------------|-------------|-------|
| ASSEMBLY | varies | 120s (safe) | 30s (safe) | ForcedAligner 180s limit |
| CONTEXT_AWARE | varies | 90s (safe) | 12s (safe) | Coupled ASR+Aligner |
| VAD_SLICING | varies | 90s (safe) | 12s (safe) | VAD segments up to 29s |
| Any (no safe_chunking) | varies | backend default | backend default | Uses config as-is |

#### BalancedPipeline (`balanced_pipeline.py`)

```
Construction:
  scene_opts = features.get("scene_detection", {})
  self.scene_detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_opts)
Usage (Step 2):
  result = self.scene_detector.detect_scenes(audio, scenes_dir, basename)
  scene_paths = result.to_legacy_tuples()
  meta = result.to_metadata_dict()
  master_metadata["scenes_detected"] = meta["scenes_detected"]
  + coarse_boundaries, vad_segments if present
Metadata:  Full metadata via to_metadata_dict()
Cleanup:   Via base_pipeline.cleanup() (automatic)
```

#### FidelityPipeline (`fidelity_pipeline.py`)

```
Construction:
  scene_opts = features.get("scene_detection", {})
  self.scene_detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_opts)
Usage:
  result = self.scene_detector.detect_scenes(audio, scenes_dir, basename)
  scene_paths = result.to_legacy_tuples()
Metadata:  Manual per-scene metadata from legacy tuples (path, start, end, duration).
           Does NOT use to_metadata_dict().
Cleanup:   Via base_pipeline.cleanup() (automatic)
```

#### FastPipeline (`fast_pipeline.py`)

```
Construction:
  scene_opts = features.get("scene_detection", {})
  self.scene_detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_opts)
Usage:
  result = self.scene_detector.detect_scenes(audio, scenes_dir, basename)
  scene_paths = result.to_legacy_tuples()
Metadata:  Manual per-scene metadata from legacy tuples.
           Uses self.scene_detector.name in tracer.
           Does NOT use to_metadata_dict().
Cleanup:   Via base_pipeline.cleanup() (automatic)
```

#### TransformersPipeline (`transformers_pipeline.py`)

```
Construction:
  IF scene_method != "none":
    self.scene_detector = SceneDetectorFactory.create_from_legacy_kwargs(
        method=self.scene_method
    )
  ELSE:
    self.scene_detector = None  ← guards against "none" at code level
Usage:
  IF self.scene_detector:
    result = self.scene_detector.detect_scenes(audio, scenes_dir, basename)
    scene_paths = result.to_legacy_tuples()
    → per-scene enhancement + ASR
  ELSE:
    → full-audio enhancement + ASR (different code path)
Metadata:  Manual per-scene metadata from legacy tuples.
           Does NOT use to_metadata_dict().
Cleanup:   Via base_pipeline.cleanup() (automatic)
```

**Notes on consumer adoption patterns:**
- **Transformers** is the only consumer that keeps the `scene_method != "none"` guard.
  This is correct because it has genuinely different code paths for scene-based vs
  full-audio processing (different enhancement calls, different ASR flow).
- **BalancedPipeline** is the only consumer that calls `to_metadata_dict()` for rich
  structured metadata. All others manually build per-scene metadata from the legacy tuples.
- **QwenPipeline** is the only consumer that calls `cleanup()` explicitly (one-shot usage).
  All others rely on `base_pipeline.cleanup()` for automatic cleanup.

#### KotobaFasterWhisperPipeline (`kotoba_faster_whisper_pipeline.py`)

```
Construction:
  scene_opts["method"] = self.scene_method
  self.scene_detector = SceneDetectorFactory.create_from_legacy_kwargs(**scene_opts)
Usage:
  result = self.scene_detector.detect_scenes(audio, scenes_dir, basename)
  scene_paths = result.to_legacy_tuples()
Metadata:  Minimal
Cleanup:   Via base_pipeline.cleanup() (automatic)
```

#### EnsemblePassWorker (`ensemble/pass_worker.py`)

```
Does NOT directly create scene detectors.
Prepares resolved_config with scene_detection options, then
delegates to a child pipeline (balanced, fidelity, etc.) which
creates its own detector.
Uses SCENE_DETECTION_PARAMS set to route params to
resolved_config["features"]["scene_detection"].
```

---

## 10. Error Handling

### Detection Failures

```python
try:
    result = detector.detect_scenes(audio, out_dir, basename)
except SceneDetectionError as e:
    # Fatal: audio load failed, backend crashed
    logger.error(f"Scene detection failed: {e}")
    raise  # or fall back to NullSceneDetector
```

### Empty Results (Not an Error)

```python
result = detector.detect_scenes(audio, out_dir, basename)
if not result.scenes:
    # No speech found — valid result
    logger.warning("No scenes detected (audio may be silent)")
    # Pipeline typically skips ASR for this file
```

### Backend Not Available

```python
try:
    detector = SceneDetectorFactory.create("semantic")
except ImportError as e:
    # scikit-learn not installed
    logger.warning(f"Semantic backend unavailable: {e}")
    detector = SceneDetectorFactory.create("auditok")  # fallback
```

### Unknown Backend

```python
try:
    detector = SceneDetectorFactory.create("unknown_method")
except ValueError as e:
    # "Unknown scene detector: 'unknown_method'. Available: [...]"
    ...
```

---

## 11. Utilities

Located in `utils.py`, shared across all backends.

### load_audio_unified

```python
def load_audio_unified(
    audio_path: Path,
    target_sr: Optional[int] = None,  # None = preserve original
    force_mono: bool = True,
) -> Tuple[np.ndarray, int]:
```

Unified audio loading. Uses `soundfile` for reading, `librosa` for resampling.
Handles stereo→mono conversion, actionable error messages.

### save_scene_wav

```python
def save_scene_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    scene_idx: int,
    output_dir: Path,
    media_basename: str,
) -> Path:
```

Saves PCM16 WAV with standardized naming: `{basename}_scene_{idx:04d}.wav`.

### brute_force_split

```python
def brute_force_split(
    start_sec: float,
    end_sec: float,
    chunk_duration: float,
    min_duration: float = 0.3,
) -> List[SceneInfo]:
```

Fixed-duration fallback splitting when silence-based detection fails.
Returns `SceneInfo` objects with `scene_path=None` (caller saves WAVs).

---

## 12. Migration from DynamicSceneDetector

`DynamicSceneDetector` (in `scene_detection.py`) is **deprecated** and emits a
`FutureWarning` on instantiation. All pipelines have been migrated to
`SceneDetectorFactory`.

### Before (Legacy)

```python
from whisperjav.modules.scene_detection import DynamicSceneDetector

detector = DynamicSceneDetector(method="auditok", **scene_opts)
scene_tuples = detector.detect_scenes(audio, out_dir, basename)
# scene_tuples: List[Tuple[Path, float, float, float]]
```

### After (Current)

```python
from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

detector = SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", **scene_opts)
result = detector.detect_scenes(audio, out_dir, basename)
scene_tuples = result.to_legacy_tuples()  # Same type as before
detector.cleanup()
```

### Key Differences

1. **Return type**: `SceneDetectionResult` (rich) vs `List[Tuple]` (flat)
2. **Metadata**: `result.to_metadata_dict()` vs `detector.get_detection_metadata()`
3. **Cleanup**: Explicit `cleanup()` call vs implicit (no cleanup)
4. **Error type**: `SceneDetectionError` vs generic exceptions
5. **"none" handling**: `NullSceneDetector` via factory vs ad-hoc `if` guards in each pipeline

---

*End of specification.*
