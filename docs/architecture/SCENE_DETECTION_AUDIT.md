# Scene Detection Audit: DynamicSceneDetector

**Date**: 2026-02-05
**Auditor**: Claude Opus 4.5
**Files Audited**:
- `whisperjav/modules/scene_detection.py` (lines 791-1508)
- `whisperjav/modules/scene_detection_backends/semantic_adapter.py` (lines 1-504)
- `whisperjav/vendor/semantic_audio_clustering.py` (lines 1-769)

---

## Executive Summary

This audit examines the `DynamicSceneDetector` class and its three scene detection methods: **auditok**, **silero**, and **semantic**. The audit reveals significant architectural inconsistencies between methods, particularly in how `min_duration` and `max_duration` constraints are enforced.

### Key Findings

| Finding | Severity | Description |
|---------|----------|-------------|
| **min_duration FILTERING vs MERGING** | HIGH | Auditok/Silero FILTER short segments; Semantic MERGES them |
| **max_duration SPLITTING inconsistency** | MEDIUM | Auditok uses auditok's max_dur; Semantic uses silence-snap splitting |
| **Parameter contract broken** | HIGH | Semantic adapter ignores `self.min_duration`/`self.max_duration` from DynamicSceneDetector |
| **Timestamp mismatch** | MEDIUM | Semantic returns asr_processing timestamps (with padding); others return exact |
| **Silero resamples per-region** | LOW | Memory-efficient but CPU-intensive for large files |

---

## 1. Overall Architecture

```
                                    +------------------------+
                                    | DynamicSceneDetector   |
                                    | (scene_detection.py)   |
                                    | Line 791               |
                                    +------------------------+
                                              |
                            method selection (line 1203-1237)
                                              |
          +-----------------------------------+-----------------------------------+
          |                                   |                                   |
    method="auditok"                   method="silero"                   method="semantic"
          |                                   |                                   |
          v                                   v                                   v
  +----------------+               +------------------+               +------------------------+
  | auditok.split  |               | auditok Pass 1   |               | SemanticClusteringAdapter |
  | (Pass 1 + 2)   |               | + Silero Pass 2  |               | (semantic_adapter.py)     |
  +----------------+               +------------------+               +------------------------+
                                                                                |
                                                                                v
                                                                      +------------------------+
                                                                      | process_movie_v7()     |
                                                                      | (semantic_audio_       |
                                                                      |  clustering.py)        |
                                                                      +------------------------+
```

---

## 2. Auditok Method Flow

### 2.1 Block Diagram

```
detect_scenes() [line 1183]
        |
        v
load_audio_unified() [line 1245] --- No resampling (native SR)
        |
        v
+-----------------------------------------------+
|              PASS 1: Coarse Splitting         |
|    auditok.split() [line 1285]                |
|    params: pass1_min_dur, pass1_max_dur,      |
|            pass1_max_silence, pass1_energy    |
+-----------------------------------------------+
        |
        v
For each story_line region:
        |
        +---> Duration <= max_duration AND >= min_duration?
        |         |
        |        YES --> SAVE directly [line 1357]
        |         |       (detection_pass=1)
        |         |
        |        NO (too long)
        |         |
        |         v
        |    +-------------------------------------------+
        |    |           PASS 2: Fine Splitting         |
        |    | Optional: _apply_assistive_processing()  |
        |    |           [line 1382]                    |
        |    |                                          |
        |    | auditok.split() [line 1395]              |
        |    | params: pass2_min_dur, pass2_max_dur,    |
        |    |         pass2_max_silence, pass2_energy  |
        |    +-------------------------------------------+
        |              |
        |              v
        |         sub_regions found?
        |              |
        |      YES     |      NO
        |       |      |       |
        |       v      |       v
        |   FILTER by  |   BRUTE FORCE [line 1419]
        |   min_dur    |   chunk_len = max_duration
        |   [line 1404]|   num_chunks = ceil(region_dur/chunk_len)
        |       |      |       |
        |       v      |       v
        |   SAVE each  |   SAVE each chunk
        |   (pass=2)   |   (pass=2)
        +-------+------+-------+
                |
                v
        final_scene_tuples[]
```

### 2.2 Parameter Flow

```
DynamicSceneDetector.__init__() [lines 800-929]
        |
        +---> self.max_duration = 29.0 (default)
        +---> self.min_duration = 0.3 (default)
        +---> self.pass1_max_duration = 2700.0 (default)
        +---> self.pass1_min_duration = 0.3 (default)
        +---> self.pass1_max_silence = 1.8 (default)
        +---> self.pass1_energy_threshold = 38 (default)
        +---> self.pass2_max_duration = max_duration - 1.0 = 28.0
        +---> self.pass2_min_duration = 0.3 (default)
        +---> self.pass2_max_silence = 0.94 (default)
        +---> self.pass2_energy_threshold = 50 (default)

detect_scenes():
        |
        +---> Pass 1 params [line 1275-1284]:
        |         min_dur = self.pass1_min_duration  (auditok param)
        |         max_dur = self.pass1_max_duration  (auditok param)
        |         max_silence = min(duration*0.95, self.pass1_max_silence)
        |         energy_threshold = self.pass1_energy_threshold
        |
        +---> Direct save check [line 1353]:
        |         duration <= self.max_duration AND >= self.min_duration
        |
        +---> Pass 2 params [line 1385-1394]:
                  min_dur = self.pass2_min_duration  (auditok param)
                  max_dur = self.pass2_max_duration  (auditok param)
                  max_silence = min(region_duration*0.95, self.pass2_max_silence)
                  energy_threshold = self.pass2_energy_threshold
```

### 2.3 Min/Max Duration Enforcement

**min_duration**:
- Pass 1: `auditok.split(min_dur=pass1_min_duration)` - auditok drops segments < min_dur
- Pass 2: `auditok.split(min_dur=pass2_min_duration)` - auditok drops segments < min_dur
- Post-filter: `if sub_dur < self.min_duration: continue` [line 1404] - FILTERING only
- **NO MERGING** of short segments

**max_duration**:
- Pass 1: `auditok.split(max_dur=pass1_max_duration)` - auditok splits at max_dur
- Direct save: Only if `region_duration <= self.max_duration` [line 1353]
- Pass 2: `auditok.split(max_dur=pass2_max_duration)` - auditok splits at max_dur
- Brute force: `chunk_len = self.brute_force_chunk_s` (defaults to max_duration) [line 1419]

---

## 3. Silero Method Flow

### 3.1 Block Diagram

```
detect_scenes() [line 1183]
        |
        v
_init_silero_vad() [line 931-968] --- Called in __init__ if method="silero"
        |
        +---> load_silero_vad() from silero_vad package
        +---> Store silero_* parameters from config

load_audio_unified() [line 1245] --- No resampling (native SR)
        |
        v
+-----------------------------------------------+
|              PASS 1: Coarse Splitting         |
|    auditok.split() [line 1285]                |
|    (SAME AS AUDITOK METHOD)                   |
+-----------------------------------------------+
        |
        v
For each story_line region:
        |
        +---> Duration <= max_duration AND >= min_duration?
        |         |
        |        YES --> SAVE directly [line 1357]
        |         |       (detection_pass=1)
        |         |
        |        NO (too long)
        |         |
        |         v
        |    +-------------------------------------------+
        |    |         PASS 2: Silero VAD               |
        |    | _detect_pass2_silero() [line 1015-1084]  |
        |    +-------------------------------------------+
        |              |
        |              v
        |    Resample region to 16kHz [line 1031-1034]
        |              |
        |              v
        |    get_speech_timestamps() [line 1043-1055]
        |    params: threshold, neg_threshold,
        |            min_silence_duration_ms,
        |            min_speech_duration_ms,
        |            max_speech_duration_s,
        |            speech_pad_ms
        |              |
        |              v
        |    Convert to SileroRegion objects [line 1070-1073]
        |              |
        |              v
        |         sub_regions found?
        |              |
        |      YES     |      NO
        |       |      |       |
        |       v      |       v
        |   FILTER by  |   BRUTE FORCE [line 1419]
        |   min_dur    |   (SAME AS AUDITOK)
        |   [line 1404]|
        |       |      |
        |       v      |
        |   SAVE each  |
        |   (pass=2)   |
        +-------+------+
                |
                v
        final_scene_tuples[]
```

### 3.2 Parameter Flow

```
DynamicSceneDetector.__init__() [lines 800-929]
        |
        +---> (All auditok params same as above)
        +---> _init_silero_vad(kwargs) [line 915]
                  |
                  +---> self.silero_threshold = 0.08 (default)
                  +---> self.silero_neg_threshold = 0.15 (default)
                  +---> self.silero_min_silence_ms = 1500 (default)
                  +---> self.silero_min_speech_ms = 100 (default)
                  +---> self.silero_max_speech_s = 600 (default)
                  +---> self.silero_min_silence_at_max = 500 (default)
                  +---> self.silero_speech_pad_ms = 200 (default)

_detect_pass2_silero() [line 1015]:
        |
        +---> get_speech_timestamps() params [line 1043-1055]:
                  threshold = self.silero_threshold
                  neg_threshold = self.silero_neg_threshold
                  min_silence_duration_ms = self.silero_min_silence_ms
                  min_speech_duration_ms = self.silero_min_speech_ms
                  max_speech_duration_s = self.silero_max_speech_s
                  min_silence_at_max_speech = self.silero_min_silence_at_max
                  speech_pad_ms = self.silero_speech_pad_ms
```

### 3.3 Min/Max Duration Enforcement

**min_duration**:
- Pass 1: Same as Auditok (auditok.split drops short segments)
- Pass 2: `min_speech_duration_ms` controls Silero's minimum (defaults to 100ms)
- Post-filter: `if sub_dur < self.min_duration: continue` [line 1404] - FILTERING only
- **NO MERGING** of short segments
- **NOTE**: `silero_min_speech_ms` (100) != `self.min_duration` (300) - DISCONNECT

**max_duration**:
- Pass 1: Same as Auditok (auditok.split)
- Pass 2: `max_speech_duration_s` controls Silero's maximum (defaults to 600s)
- Brute force fallback: Same as Auditok
- **NOTE**: `silero_max_speech_s` (600) != `self.max_duration` (29) - DISCONNECT

---

## 4. Semantic Method Flow

### 4.1 Block Diagram

```
detect_scenes() [line 1183]
        |
        v
method == "semantic" [line 1215]
        |
        v
_semantic_adapter.detect_scenes() [line 1223]
        |
        v
+-------------------------------------------+
| SemanticClusteringAdapter.detect_scenes() |
| (semantic_adapter.py line 228)            |
+-------------------------------------------+
        |
        v
_ensure_engine() [line 188] --- Lazy load vendor module
        |
        v
Build engine_config [line 286-292]:
    min_duration = self.config.min_duration  (from adapter config)
    max_duration = self.config.max_duration  (from adapter config)
    snap_window = self.config.snap_window
    clustering_threshold = self.config.clustering_threshold
        |
        v
+-------------------------------------------+
| process_movie_v7() [vendor line 566]      |
+-------------------------------------------+
        |
        v
StreamFeatureExtractor.extract() [line 193]
        |
        +---> MFCC, delta, RMS, ZCR, spectral contrast, chroma
        |
        v
AdaptiveClassifier.calibrate() [line 461]
        |
        v
+-------------------------------------------+
| SemanticSegmenter.segment() [line 277]    |
+-------------------------------------------+
        |
        v
1. Pre-clustering Smoothing [line 282]
        |
        v
2. AgglomerativeClustering [line 291]
        |
        v
3. Raw Boundaries from cluster labels [line 294-299]
        |
        v
4. _snap_to_silence() [line 303, 316-341]
   - Snaps boundaries to lowest energy within snap_window
        |
        v
5. _smart_merge() [line 306, 343-393]
   - MERGES segments < min_dur with most similar neighbor
   - Respects max_dur constraint during merge
        |
        v
6. _forced_cleanup() [line 309, 395-413]
   - Final pass to merge any remaining short segments
        |
        v
7. _ensure_timeline_coverage() [line 312, 415-445]
   - Fills gaps with filler segments
   - Ensures 0.0 -> duration coverage
        |
        v
Return raw_segments (list of dicts)
        |
        v
+-------------------------------------------+
| Back to Adapter                           |
+-------------------------------------------+
        |
        v
_transform_and_split() [line 325, 358-455]
        |
        +---> Load source audio [line 371-375]
        +---> For each segment in semantic_metadata:
        |         Use asr_processing timestamps (WITH PADDING)
        |         Extract audio slice
        |         Save as WAV
        |         Populate scenes_detected[]
        |
        v
Return scene_tuples to DynamicSceneDetector
```

### 4.2 Parameter Flow

```
DynamicSceneDetector.__init__() [lines 800-929]
        |
        +---> self.max_duration = 29.0 (default)   <-- NOT PASSED TO SEMANTIC
        +---> self.min_duration = 0.3 (default)    <-- NOT PASSED TO SEMANTIC
        |
        +---> _init_semantic_adapter(kwargs) [line 920]
                  |
                  v
              SemanticClusteringConfig() [line 986-994]:
                  min_duration = config.get("min_duration", 20.0)   <-- DIFFERENT DEFAULT!
                  max_duration = config.get("max_duration", 420.0)  <-- DIFFERENT DEFAULT!
                  snap_window = config.get("snap_window", 5.0)
                  clustering_threshold = config.get("clustering_threshold", 18.0)

SemanticClusteringAdapter._ensure_engine() [line 188]:
        |
        v
process_movie_v7() receives SegmentationConfig with:
        min_duration = 20.0 (from adapter config, NOT from DynamicSceneDetector)
        max_duration = 420.0 (from adapter config, NOT from DynamicSceneDetector)
```

### 4.3 Min/Max Duration Enforcement

**min_duration**:
- SemanticSegmenter._smart_merge() [line 359-362]:
  ```python
  if seg["dur"] >= self.min_dur:
      i += 1
      continue
  # else: MERGE with most similar neighbor
  ```
- SemanticSegmenter._forced_cleanup() [line 401-412]:
  - Final pass merges remaining short segments
- **MERGING** of short segments (NOT filtering)

**max_duration**:
- _smart_merge() respects max_dur during merging [lines 371, 374, 376, 377]:
  ```python
  if (left["dur"] + seg["dur"]) <= self.max_dur: merge_target = -1
  ```
- _snap_to_silence() [line 316-341]:
  - Snaps to lowest energy, but does NOT split long segments
- **NO EXPLICIT max_duration SPLITTING** after clustering
- Segments CAN exceed max_duration if clustering produces them

---

## 5. Comparison Matrix

| Aspect | Auditok | Silero | Semantic |
|--------|---------|--------|----------|
| **Pass 1** | auditok.split | auditok.split | MFCC clustering |
| **Pass 2** | auditok.split | Silero VAD | None (single-pass) |
| **min_duration default** | 0.3s | 0.3s | 20.0s |
| **max_duration default** | 29.0s | 29.0s | 420.0s |
| **min_duration enforcement** | FILTER | FILTER | MERGE |
| **max_duration enforcement** | auditok splits | Silero + brute force | Merge constraint only |
| **Short segment handling** | Discard | Discard | Merge to neighbor |
| **Timestamp type returned** | Exact | Exact | asr_processing (padded) |
| **Resampling** | None | Per-region to 16kHz | Per-file to 16kHz |
| **Parameter source** | DynamicSceneDetector | DynamicSceneDetector | SemanticClusteringConfig |

---

## 6. Gaps, Bugs, and Technical Debt

### 6.1 CRITICAL: Parameter Contract Broken (Semantic)

**Location**: `scene_detection.py` lines 986-994, `semantic_adapter.py` line 52-63

**Issue**: When `DynamicSceneDetector` is instantiated with `min_duration=0.3, max_duration=29.0`, these values are **NOT** passed to the semantic adapter. The adapter uses its own defaults:
- `min_duration=20.0` (vs 0.3)
- `max_duration=420.0` (vs 29.0)

**Impact**: Users expecting 30-second max scenes get 7-minute scenes with semantic method.

**Root Cause**: `_init_semantic_adapter()` reads from `kwargs` which doesn't include `self.min_duration` or `self.max_duration`:

```python
# scene_detection.py line 986-994
adapter_config = SemanticClusteringConfig(
    min_duration=float(config.get("min_duration", 20.0)),  # kwargs, not self
    max_duration=float(config.get("max_duration", 420.0)), # kwargs, not self
    ...
)
```

### 6.2 HIGH: Silero VAD Duration Parameters Disconnect

**Location**: `scene_detection.py` lines 940-946

**Issue**: Silero VAD has its own duration parameters that don't match the class-level ones:
- `silero_min_speech_ms=100` vs `self.min_duration=300` (ms vs s mismatch too!)
- `silero_max_speech_s=600` vs `self.max_duration=29`

**Impact**: Silero can return segments as short as 100ms or as long as 600s, which are then only filtered (not merged) at line 1404.

### 6.3 MEDIUM: Timestamp Semantics Inconsistency (Semantic)

**Location**: `semantic_adapter.py` lines 401-406, `semantic_audio_clustering.py` lines 122-144

**Issue**: Semantic method returns `asr_processing` timestamps which include 0.2s padding:

```python
# semantic_audio_clustering.py line 123-127
pad = 0.2
safe_start = max(0.0, self.start - pad)
safe_end = self.end + pad
```

Meanwhile, `_transform_and_split()` uses these padded timestamps [line 403-406]:
```python
asr_ts = segment.get("asr_processing", segment.get("timestamps", {}))
start_sec = asr_ts.get("start", 0.0)
end_sec = asr_ts.get("end", 0.0)
```

**Impact**: Scene files are longer than reported timestamps. Could cause subtitle timing drift if downstream expects exact timestamps.

### 6.4 MEDIUM: No max_duration Enforcement in Semantic Post-Clustering

**Location**: `semantic_audio_clustering.py` lines 306-313

**Issue**: After clustering, `_smart_merge()` only prevents merges that would exceed `max_dur`. It does NOT split segments that already exceed `max_dur` from clustering.

**Impact**: If clustering produces a 500s segment, it stays 500s. The `max_duration` parameter is only a merge constraint, not a hard limit.

### 6.5 LOW: Magic Numbers in Semantic Adapter

**Location**: `semantic_adapter.py` line 52-63, `semantic_audio_clustering.py` lines 70-91

**Issue**: Multiple hardcoded constants without configuration:
- `pad = 0.2` (padding for asr_processing)
- `fps = 31` (frames per second for feature extraction)
- `smoothing_window = 15`
- Various percentile thresholds in classifier

### 6.6 LOW: Silero Per-Region Resampling

**Location**: `scene_detection.py` lines 1029-1034

**Issue**: Silero resamples each region to 16kHz individually. For files with many long regions, this is repeated work.

**Code**:
```python
if sr != 16000:
    region_duration = len(region_audio) / sr
    logger.debug(f"Resampling {region_duration:.1f}s region from {sr}Hz to 16kHz for Silero")
    audio_16k = librosa.resample(region_audio, orig_sr=sr, target_sr=16000)
```

**Impact**: CPU overhead for files with native SR != 16kHz. Documented as intentional trade-off for memory.

### 6.7 MEDIUM: Brute Force Fallback Ignores Scene Boundaries

**Location**: `scene_detection.py` lines 1419-1435

**Issue**: When Pass 2 returns no sub-regions, brute force splits at fixed `chunk_len` intervals with no regard for audio content.

```python
chunk_len = self.brute_force_chunk_s
num_chunks = int(np.ceil(region_duration / max(chunk_len, self.min_duration)))
for i in range(num_chunks):
    sub_start = region_start_sec + i * chunk_len
    sub_end = min(region_start_sec + (i + 1) * chunk_len, region_end_sec)
```

**Impact**: Can cut mid-sentence. Better approaches exist (e.g., silence detection fallback).

### 6.8 LOW: Inconsistent Logging Patterns

**Locations**: Multiple files

**Issue**: Three different logging patterns:
1. `scene_detection.py`: Uses `from whisperjav.utils.logger import logger`
2. `semantic_adapter.py`: Same, but also accepts `logger_instance` parameter
3. `semantic_audio_clustering.py`: Uses `print()` as default, optional `logger` param

---

## 7. Recommendations

### 7.1 Fix Parameter Contract (Priority: HIGH)

```python
# In _init_semantic_adapter(), use self.min_duration and self.max_duration:
adapter_config = SemanticClusteringConfig(
    min_duration=float(config.get("min_duration", self.min_duration)),  # Use self
    max_duration=float(config.get("max_duration", self.max_duration)),  # Use self
    ...
)
```

### 7.2 Align Silero Duration Parameters (Priority: HIGH)

Either:
1. Scale `self.min_duration` (seconds) to `silero_min_speech_ms` (milliseconds)
2. Or document clearly that Silero uses different duration constraints

### 7.3 Add max_duration Splitting to Semantic (Priority: MEDIUM)

Add a `_split_long_segments()` method in `SemanticSegmenter` that runs after `_smart_merge()` to split any segment exceeding `max_duration` at lowest-energy points.

### 7.4 Standardize Timestamp Semantics (Priority: MEDIUM)

Document clearly which timestamp type each method returns, or add a configuration option to control padding.

### 7.5 Improve Brute Force Fallback (Priority: LOW)

Use simple energy-based silence detection for brute force splits instead of fixed intervals.

---

## 8. Appendix: Code Reference Quick Links

| Function/Method | File | Line |
|-----------------|------|------|
| `DynamicSceneDetector.__init__` | scene_detection.py | 800 |
| `DynamicSceneDetector.detect_scenes` | scene_detection.py | 1183 |
| `DynamicSceneDetector._init_silero_vad` | scene_detection.py | 931 |
| `DynamicSceneDetector._init_semantic_adapter` | scene_detection.py | 970 |
| `DynamicSceneDetector._detect_pass2_silero` | scene_detection.py | 1015 |
| `SemanticClusteringAdapter.__init__` | semantic_adapter.py | 130 |
| `SemanticClusteringAdapter.detect_scenes` | semantic_adapter.py | 228 |
| `SemanticClusteringAdapter._transform_and_split` | semantic_adapter.py | 358 |
| `process_movie_v7` | semantic_audio_clustering.py | 566 |
| `SemanticSegmenter.segment` | semantic_audio_clustering.py | 277 |
| `SemanticSegmenter._smart_merge` | semantic_audio_clustering.py | 343 |
| `SemanticSegmenter._snap_to_silence` | semantic_audio_clustering.py | 316 |

---

*End of Audit*
