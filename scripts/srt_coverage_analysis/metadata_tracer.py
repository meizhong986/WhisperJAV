"""
Metadata Tracer Module
======================

Trace missing/partial segments through WhisperJAV pipeline metadata
to determine root cause of coverage gaps.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .srt_parser import Segment


@dataclass
class SceneInfo:
    """Information about a scene from metadata."""
    scene_index: int
    filename: str
    start_time: float
    end_time: float
    duration: float
    transcribed: bool
    srt_path: Optional[str] = None


@dataclass
class TraceResult:
    """Result of tracing a segment through pipeline metadata."""
    ref_segment: Segment
    root_cause: str
    details: str
    containing_scenes: List[SceneInfo]
    coverage_percent: float = 0.0

    @property
    def is_not_in_scene(self) -> bool:
        """Check if segment was not captured by any scene."""
        return self.root_cause == "NOT_IN_SCENE"

    @property
    def is_scene_not_transcribed(self) -> bool:
        """Check if scene was detected but not transcribed."""
        return self.root_cause == "SCENE_NOT_TRANSCRIBED"

    @property
    def is_filtered_or_failed(self) -> bool:
        """Check if segment was transcribed but filtered/failed."""
        return self.root_cause == "FILTERED_OR_FAILED"


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load WhisperJAV metadata JSON file.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        Parsed metadata dictionary

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    metadata_file = Path(metadata_path)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


def extract_scenes(metadata: Dict[str, Any]) -> List[SceneInfo]:
    """
    Extract scene information from metadata.

    Args:
        metadata: Parsed metadata dictionary

    Returns:
        List of SceneInfo objects
    """
    scenes_data = metadata.get('scenes_detected', [])
    scenes = []

    for scene_data in scenes_data:
        scene = SceneInfo(
            scene_index=scene_data.get('scene_index', -1),
            filename=scene_data.get('filename', ''),
            start_time=scene_data.get('start_time_seconds', 0.0),
            end_time=scene_data.get('end_time_seconds', 0.0),
            duration=scene_data.get('duration_seconds', 0.0),
            transcribed=scene_data.get('transcribed', False),
            srt_path=scene_data.get('srt_path')
        )
        scenes.append(scene)

    return scenes


def find_containing_scenes(
    segment: Segment,
    scenes: List[SceneInfo]
) -> List[SceneInfo]:
    """
    Find which scene(s) contain a given segment.

    Uses segment midpoint to determine scene membership.

    Args:
        segment: The segment to locate
        scenes: List of available scenes

    Returns:
        List of scenes that contain this segment
    """
    seg_midpoint = segment.midpoint
    containing_scenes = []

    for scene in scenes:
        if scene.start_time <= seg_midpoint <= scene.end_time:
            containing_scenes.append(scene)

    return containing_scenes


def trace_in_metadata(
    segment: Segment,
    metadata: Dict[str, Any],
    coverage_percent: float = 0.0
) -> TraceResult:
    """
    Trace a missing/partial segment through pipeline metadata
    to determine root cause.

    Root Cause Categories:
        NOT_IN_SCENE: Segment not captured by any scene chunk
        SCENE_NOT_TRANSCRIBED: Scene detected but not transcribed
        FILTERED_OR_FAILED: Scene transcribed but segment missing
                            (likely VAD filtered, transcription failed,
                             or sanitization filtered)

    Args:
        segment: Reference segment to trace
        metadata: WhisperJAV pipeline metadata
        coverage_percent: Coverage percentage from analysis

    Returns:
        TraceResult with root cause and details
    """
    scenes = extract_scenes(metadata)

    # Find which scene(s) contain this segment
    containing_scenes = find_containing_scenes(segment, scenes)

    # Case 1: Segment not in any scene
    if not containing_scenes:
        return TraceResult(
            ref_segment=segment,
            root_cause="NOT_IN_SCENE",
            details=(
                f"Segment at {segment.start:.2f}s-{segment.end:.2f}s "
                f"(midpoint: {segment.midpoint:.2f}s) was not captured "
                f"by any scene chunk. Scene detection may have missed this dialogue."
            ),
            containing_scenes=[],
            coverage_percent=coverage_percent
        )

    # Case 2: Scene(s) found but not transcribed
    untranscribed_scenes = [s for s in containing_scenes if not s.transcribed]
    if untranscribed_scenes:
        scene_indices = [s.scene_index for s in untranscribed_scenes]
        return TraceResult(
            ref_segment=segment,
            root_cause="SCENE_NOT_TRANSCRIBED",
            details=(
                f"Segment belongs to scene(s) {scene_indices} which were "
                f"detected but not transcribed. Check ASR processing logs."
            ),
            containing_scenes=containing_scenes,
            coverage_percent=coverage_percent
        )

    # Case 3: Scene was transcribed but segment still missing/partial
    # Could be VAD filtered, transcription failed, or sanitization filtered
    scene_indices = [s.scene_index for s in containing_scenes]
    return TraceResult(
        ref_segment=segment,
        root_cause="FILTERED_OR_FAILED",
        details=(
            f"Segment belongs to scene(s) {scene_indices} which were "
            f"transcribed successfully. Segment may have been:\n"
            f"  - Filtered by VAD (marked as silence)\n"
            f"  - Failed transcription (ASR produced no output)\n"
            f"  - Removed by sanitization (hallucination/logprob filtering)\n"
            f"Coverage: {coverage_percent:.1f}%"
        ),
        containing_scenes=containing_scenes,
        coverage_percent=coverage_percent
    )


def trace_all_segments(
    segments_to_trace: List[Segment],
    coverage_results: List,
    metadata: Dict[str, Any]
) -> List[TraceResult]:
    """
    Trace multiple segments through metadata.

    Args:
        segments_to_trace: List of segments that need tracing
        coverage_results: Coverage results (to get coverage percentages)
        metadata: WhisperJAV pipeline metadata

    Returns:
        List of TraceResult objects
    """
    results = []

    # Create a mapping from segment index to coverage result
    coverage_map = {r.ref_segment.index: r for r in coverage_results}

    for segment in segments_to_trace:
        coverage_result = coverage_map.get(segment.index)
        coverage_percent = (
            coverage_result.coverage_percent if coverage_result else 0.0
        )

        trace_result = trace_in_metadata(segment, metadata, coverage_percent)
        results.append(trace_result)

    return results


def analyze_root_causes(trace_results: List[TraceResult]) -> Dict[str, Any]:
    """
    Analyze distribution of root causes.

    Args:
        trace_results: List of TraceResult objects

    Returns:
        Dictionary with root cause statistics
    """
    total = len(trace_results)
    not_in_scene = sum(1 for r in trace_results if r.is_not_in_scene)
    not_transcribed = sum(1 for r in trace_results if r.is_scene_not_transcribed)
    filtered_or_failed = sum(1 for r in trace_results if r.is_filtered_or_failed)

    return {
        'total_traced': total,
        'not_in_scene': not_in_scene,
        'scene_not_transcribed': not_transcribed,
        'filtered_or_failed': filtered_or_failed,
        'not_in_scene_percent': (not_in_scene / total * 100) if total > 0 else 0.0,
        'scene_not_transcribed_percent': (not_transcribed / total * 100) if total > 0 else 0.0,
        'filtered_or_failed_percent': (filtered_or_failed / total * 100) if total > 0 else 0.0,
    }


if __name__ == "__main__":
    # Test module
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metadata_tracer.py <metadata.json>")
        sys.exit(1)

    metadata_path = sys.argv[1]

    print(f"Loading metadata: {metadata_path}")
    metadata = load_metadata(metadata_path)

    scenes = extract_scenes(metadata)
    print(f"\nExtracted {len(scenes)} scenes:")
    for scene in scenes:
        status = "✓ transcribed" if scene.transcribed else "✗ not transcribed"
        print(f"  Scene {scene.scene_index}: {scene.start_time:.2f}s - {scene.end_time:.2f}s ({status})")

    # Test tracing with a sample segment
    from .srt_parser import Segment

    test_segment = Segment(
        index=1,
        start=10.0,
        end=15.0,
        text="Test segment"
    )

    print(f"\nTracing test segment: {test_segment}")
    result = trace_in_metadata(test_segment, metadata)
    print(f"  Root cause: {result.root_cause}")
    print(f"  Details: {result.details}")
    print(f"  Containing scenes: {len(result.containing_scenes)}")
