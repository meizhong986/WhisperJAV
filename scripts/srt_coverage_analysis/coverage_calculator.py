"""
Coverage Calculator Module
===========================

Calculate temporal coverage of reference segments by test segments
using pure interval overlap algorithm.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from .srt_parser import Segment


@dataclass
class OverlappingSegment:
    """Information about a test segment that overlaps with a reference segment."""
    segment: Segment
    overlap_start: float
    overlap_end: float
    overlap_duration: float


@dataclass
class CoverageResult:
    """Result of coverage analysis for a single reference segment."""
    ref_segment: Segment
    coverage_percent: float
    status: str  # "COVERED", "PARTIAL", or "MISSING"
    total_overlap_seconds: float
    overlapping_segments: List[OverlappingSegment] = field(default_factory=list)

    @property
    def is_covered(self) -> bool:
        """Check if segment is sufficiently covered."""
        return self.status == "COVERED"

    @property
    def is_partial(self) -> bool:
        """Check if segment has partial coverage."""
        return self.status == "PARTIAL"

    @property
    def is_missing(self) -> bool:
        """Check if segment is completely missing."""
        return self.status == "MISSING"

    @property
    def needs_review(self) -> bool:
        """Check if segment needs manual review (partial or missing)."""
        return self.status in ("PARTIAL", "MISSING")


def calculate_coverage(
    ref_segment: Segment,
    test_segments: List[Segment],
    coverage_threshold: float = 0.60
) -> CoverageResult:
    """
    Calculate what percentage of a reference segment is covered by test segments.

    Uses pure interval overlap - no tolerance, no text matching.
    Simply calculates what fraction of the reference segment's time interval
    overlaps with any test segment intervals.

    Args:
        ref_segment: Reference segment to check coverage for
        test_segments: List of all test segments
        coverage_threshold: Minimum coverage fraction to consider "covered" (default 60%)

    Returns:
        CoverageResult object with coverage statistics

    Algorithm:
        For each test segment:
            1. Calculate intersection interval: [max(ref_start, test_start), min(ref_end, test_end)]
            2. If intersection is valid (start < end), add to total overlap
        Coverage = total_overlap / ref_duration
    """
    ref_start = ref_segment.start
    ref_end = ref_segment.end
    ref_duration = ref_segment.duration

    # Handle edge case of zero-duration segments
    if ref_duration == 0:
        return CoverageResult(
            ref_segment=ref_segment,
            coverage_percent=0.0,
            status="MISSING",
            total_overlap_seconds=0.0,
            overlapping_segments=[]
        )

    # Collect all overlapping intervals
    total_overlap = 0.0
    overlapping_test_segs = []

    for test_seg in test_segments:
        # Calculate intersection interval
        overlap_start = max(ref_start, test_seg.start)
        overlap_end = min(ref_end, test_seg.end)

        # Check if there's actual overlap
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            total_overlap += overlap_duration

            overlapping_test_segs.append(
                OverlappingSegment(
                    segment=test_seg,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                    overlap_duration=overlap_duration
                )
            )

    # Calculate coverage percentage
    coverage_percent = (total_overlap / ref_duration) * 100.0

    # Determine status based on threshold
    if coverage_percent >= (coverage_threshold * 100):
        status = "COVERED"
    elif coverage_percent > 0:
        status = "PARTIAL"
    else:
        status = "MISSING"

    return CoverageResult(
        ref_segment=ref_segment,
        coverage_percent=coverage_percent,
        status=status,
        total_overlap_seconds=total_overlap,
        overlapping_segments=overlapping_test_segs
    )


def analyze_all_segments(
    reference_segments: List[Segment],
    test_segments: List[Segment],
    coverage_threshold: float = 0.60
) -> List[CoverageResult]:
    """
    Analyze coverage for all reference segments.

    Args:
        reference_segments: List of reference segments
        test_segments: List of test segments
        coverage_threshold: Minimum coverage fraction (default 60%)

    Returns:
        List of CoverageResult objects, one per reference segment
    """
    results = []

    for ref_seg in reference_segments:
        result = calculate_coverage(ref_seg, test_segments, coverage_threshold)
        results.append(result)

    return results


def calculate_summary_statistics(results: List[CoverageResult]) -> Dict[str, Any]:
    """
    Calculate summary statistics from coverage results.

    Args:
        results: List of CoverageResult objects

    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    covered = sum(1 for r in results if r.is_covered)
    partial = sum(1 for r in results if r.is_partial)
    missing = sum(1 for r in results if r.is_missing)

    total_ref_duration = sum(r.ref_segment.duration for r in results)
    total_covered_duration = sum(
        r.total_overlap_seconds for r in results if r.is_covered
    )

    avg_coverage = (
        sum(r.coverage_percent for r in results) / total
        if total > 0 else 0.0
    )

    return {
        'total_segments': total,
        'covered_segments': covered,
        'partial_segments': partial,
        'missing_segments': missing,
        'covered_percent': (covered / total * 100) if total > 0 else 0.0,
        'partial_percent': (partial / total * 100) if total > 0 else 0.0,
        'missing_percent': (missing / total * 100) if total > 0 else 0.0,
        'average_coverage_percent': avg_coverage,
        'total_reference_duration': total_ref_duration,
        'total_covered_duration': total_covered_duration,
    }


if __name__ == "__main__":
    # Test module with sample data
    from .srt_parser import parse_srt_file
    import sys

    if len(sys.argv) < 3:
        print("Usage: python coverage_calculator.py <reference.srt> <test.srt>")
        sys.exit(1)

    ref_path = sys.argv[1]
    test_path = sys.argv[2]

    print(f"Loading reference: {ref_path}")
    ref_segments = parse_srt_file(ref_path)
    print(f"  Loaded {len(ref_segments)} segments")

    print(f"\nLoading test: {test_path}")
    test_segments = parse_srt_file(test_path)
    print(f"  Loaded {len(test_segments)} segments")

    print("\nCalculating coverage...")
    results = analyze_all_segments(ref_segments, test_segments)

    stats = calculate_summary_statistics(results)
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        if 'percent' in key:
            print(f"  {key}: {value:.2f}%")
        elif 'duration' in key:
            print(f"  {key}: {value:.2f}s")
        else:
            print(f"  {key}: {value}")

    # Show some examples
    missing = [r for r in results if r.is_missing]
    partial = [r for r in results if r.is_partial]

    if missing:
        print(f"\nExample missing segment:")
        m = missing[0]
        print(f"  {m.ref_segment}")
        print(f"  Coverage: {m.coverage_percent:.1f}%")

    if partial:
        print(f"\nExample partial segment:")
        p = partial[0]
        print(f"  {p.ref_segment}")
        print(f"  Coverage: {p.coverage_percent:.1f}%")
        print(f"  Overlapping with {len(p.overlapping_segments)} test segments")
