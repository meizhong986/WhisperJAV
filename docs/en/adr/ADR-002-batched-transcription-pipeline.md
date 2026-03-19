# ADR-002: Batched Transcription Pipeline Architecture

**Status:** Proposed
**Date:** 2024-12-11
**Decision Makers:** WhisperJAV Development Team
**Target Release:** v1.8.0+

---

## Context

WhisperJAV's current transcription pipeline processes scenes **sequentially**, where each scene goes through the full pipeline (speech segmentation → transcription → SRT) before the next scene begins.

### Current Architecture (Sequential)

```
Scene 1: Load → Speech Segmentation → Transcribe Groups → Write SRT
Scene 2: Load → Speech Segmentation → Transcribe Groups → Write SRT
Scene 3: Load → Speech Segmentation → Transcribe Groups → Write SRT
...
Final: Stitch all scene SRTs together
```

### Observations from Testing

When processing a 5-minute audio file with 3 scenes:
- Each scene triggers a **separate speech segmentation pass**
- The Whisper model remains loaded but processes **one VAD group at a time**
- Progress reporting shows scene-level updates, not segment-level
- Total segments not known until all scenes are processed

### Key Question
> "Would a batched architecture—where all scenes are segmented first, then all segments transcribed together—provide better performance, progress reporting, and resource utilization?"

---

## Problem Statement

The sequential architecture has several limitations:

| Issue | Impact |
|-------|--------|
| **Repeated I/O** | Each scene loaded separately, potential cache misses |
| **No global progress** | Cannot show "X of Y total segments" until all scenes processed |
| **Memory fragmentation** | VAD model and Whisper model both in VRAM throughout |
| **Limited parallelization** | Scene processing is inherently serial |
| **Unpredictable ETA** | Cannot estimate completion without knowing total work |

---

## Proposed Solution: Two-Stage Batched Processing

### Architecture Overview

```
Stage 1: Speech Segmentation (Batch All Scenes)
┌─────────────────────────────────────────────────────────────────┐
│ Scene 1 ─→ VAD ─→ [Group 1, Group 2]                           │
│ Scene 2 ─→ VAD ─→ [Group 1]                                    │
│ Scene 3 ─→ VAD ─→ [Group 1, Group 2, Group 3]                  │
│                                                                  │
│ Output: Global VAD Group Registry (6 groups total)              │
│         with provenance tracking (scene_idx, group_idx)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                    [Release VAD model from VRAM]
                              │
                              ▼
Stage 2: Batched Transcription
┌─────────────────────────────────────────────────────────────────┐
│ Load Whisper model once                                         │
│                                                                  │
│ for group in global_registry:                                   │
│     transcribe(group.audio_data)                                │
│     track_provenance(group.scene_idx, group.group_idx)          │
│                                                                  │
│ Output: Transcription results with scene/group mapping          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Stage 3: SRT Reconstruction
┌─────────────────────────────────────────────────────────────────┐
│ Group results by scene_idx                                      │
│ Apply timestamp offsets per scene                               │
│ Write scene SRTs → Stitch → Final output                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

```python
@dataclass
class VADGroupEntry:
    """Entry in the global VAD group registry."""
    scene_idx: int           # Which scene this group belongs to
    group_idx: int           # Group index within the scene
    audio_data: np.ndarray   # Audio samples for this group
    sample_rate: int         # Audio sample rate
    start_sec: float         # Start time within scene
    end_sec: float           # End time within scene
    scene_offset_sec: float  # Scene start offset in original audio
    segments: List[SpeechSegment]  # Individual speech segments

class VADGroupRegistry:
    """Global registry of all VAD groups across all scenes."""
    entries: List[VADGroupEntry]

    @property
    def total_groups(self) -> int: ...

    @property
    def total_segments(self) -> int: ...

    def groups_by_scene(self, scene_idx: int) -> List[VADGroupEntry]: ...
```

### API Changes

```python
# New method in BalancedPipeline
def process_batched(self, scene_paths: List[Tuple[Path, float, float, float]]) -> Dict:
    """Process all scenes using two-stage batched architecture."""

    # Stage 1: Batch speech segmentation
    registry = self._batch_segment_scenes(scene_paths)

    # Release VAD model
    self.asr.release_vad_model()

    # Stage 2: Batch transcription
    results = self.asr.transcribe_registry(registry)

    # Stage 3: Reconstruct SRTs
    return self._reconstruct_scene_srts(results, scene_paths)

# New method in FasterWhisperProASR
def transcribe_registry(self, registry: VADGroupRegistry) -> List[TranscriptionResult]:
    """Transcribe all groups in the registry."""
    ...
```

---

## Alternatives Considered

### Alternative 1: Keep Sequential, Add Caching
**Approach:** Cache VAD results, reuse if scene already processed.

**Rejected because:**
- Still sequential processing
- Caching adds complexity without addressing core issues
- Progress reporting still scene-bound

### Alternative 2: Parallel Scene Processing
**Approach:** Process multiple scenes in parallel using multiprocessing.

**Rejected because:**
- VRAM constraints limit parallelism (Whisper models are large)
- Complexity of managing multiple GPU contexts
- Diminishing returns on consumer hardware

### Alternative 3: Streaming Architecture
**Approach:** Stream audio through pipeline, process segments as they arrive.

**Rejected because:**
- Major rewrite of pipeline architecture
- Incompatible with scene detection paradigm
- Overkill for batch processing use case

---

## Consequences

### Positive

| Benefit | Description |
|---------|-------------|
| **Global progress** | Know total segments upfront: "Transcribing 45/120 segments" |
| **Better ETA** | Accurate time estimates based on segments processed |
| **Memory optimization** | Release VAD model before loading Whisper |
| **Reduced I/O** | Load audio once per scene, keep in memory |
| **Parallelization ready** | Stage 1 could be parallelized in future |
| **Predictable behavior** | Same processing regardless of scene count |

### Negative

| Drawback | Mitigation |
|----------|------------|
| **Higher peak memory** | Limit registry size, process in chunks if needed |
| **Delayed first output** | Show progress during Stage 1 to maintain UX |
| **Architecture change** | Keep sequential as fallback, gradual migration |
| **Testing complexity** | Comprehensive test coverage for new data flow |

### Risks

1. **Memory exhaustion** on very long files (100+ scenes)
   - Mitigation: Chunk processing with configurable batch size

2. **Regression in edge cases** (single scene, no speech detected)
   - Mitigation: Thorough test coverage, fallback to sequential

3. **Progress UI changes** required for two-stage model
   - Mitigation: Abstract progress interface, update incrementally

---

## Implementation Plan

### Phase 1: Foundation (v1.8.0)
- [ ] Implement `VADGroupRegistry` data structure
- [ ] Add `batch_segment_scenes()` to BalancedPipeline
- [ ] Add `transcribe_registry()` to FasterWhisperProASR
- [ ] Update progress reporting for two-stage model
- [ ] Add feature flag to enable/disable batched processing

### Phase 2: Optimization (v1.8.x)
- [ ] Implement chunked processing for large files
- [ ] Add memory profiling and limits
- [ ] Optimize audio data handling (memory views vs copies)
- [ ] Performance benchmarking vs sequential

### Phase 3: Migration (v1.9.0)
- [ ] Make batched processing the default
- [ ] Deprecate sequential-only code paths
- [ ] Update documentation and user guides

---

## Metrics for Success

| Metric | Target |
|--------|--------|
| Memory usage | < 20% increase in peak usage |
| Processing time | No regression (ideally 5-10% improvement) |
| Progress accuracy | ETA within 10% of actual for files > 1 minute |
| Test coverage | > 90% for new code paths |

---

## References

- Current implementation: `whisperjav/pipelines/balanced_pipeline.py`
- Speech segmentation: `whisperjav/modules/speech_segmentation/`
- ASR module: `whisperjav/modules/faster_whisper_pro_asr.py`
- Related: ADR-001 (YAML configuration architecture)

---

## Appendix: Current Code Flow

### BalancedPipeline.process() - Scene Loop (lines 278-356)
```python
for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
    scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
    # ... progress update ...
    self.asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
    # ... collect results ...
```

### FasterWhisperProASR.transcribe() - VAD Group Loop (lines 426-430)
```python
all_segments = []
for i, vad_group in enumerate(vad_segments, 1):
    segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group)
    all_segments.extend(segments)
```

### Speech Segmentation Output Format
```
"Speech segmentation complete: 26 segments in 8 groups"
- Segments: Individual speech regions detected by VAD
- Groups: Segments clustered by time proximity (gap < 4.0s)
```
