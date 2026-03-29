"""
Console report and JSON export module for the pipeline analysis test suite.

Formats and displays analysis results. No WhisperJAV imports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AnalysisResult,
    AudioInfo,
    BackendRunResult,
    ComparisonResult,
    SegmentInfo,
)


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------


def print_summary_table(
    analyses: Dict[str, AnalysisResult],
    audio_duration_sec: float,
) -> None:
    """Print formatted summary table of per-backend metrics.

    Scene detection backends are printed first, then speech segmentation.
    """
    if not analyses:
        print("No results to display.")
        return

    # Separate by type
    scene_analyses = {
        k: v for k, v in analyses.items() if v.backend_type == "scene_detection"
    }
    seg_analyses = {
        k: v
        for k, v in analyses.items()
        if v.backend_type == "speech_segmentation"
    }

    header = (
        f"{'Backend':<25} {'Type':<8} {'Segs':>6} {'Coverage':>9} "
        f"{'Mean':>7} {'Median':>7} {'Min':>6} {'Max':>7} {'Time':>7}"
    )
    separator = "-" * len(header)

    print()
    print("=" * len(header))
    print("RESULTS SUMMARY")
    print("=" * len(header))
    print(f"Audio duration: {audio_duration_sec:.2f}s")
    print()
    print(header)
    print(separator)

    def _print_row(key: str, a: AnalysisResult) -> None:
        label = a.backend_name
        btype = "scene" if a.backend_type == "scene_detection" else "seg"
        print(
            f"{label:<25} {btype:<8} {a.num_segments:>6} "
            f"{a.coverage_ratio:>8.1%} "
            f"{a.mean_segment_duration:>6.1f}s "
            f"{a.median_segment_duration:>6.1f}s "
            f"{a.min_segment_duration:>5.1f}s "
            f"{a.max_segment_duration:>6.1f}s "
            f"{a.processing_time_sec:>6.2f}s"
        )

    if scene_analyses:
        for k, a in scene_analyses.items():
            _print_row(k, a)

    if scene_analyses and seg_analyses:
        print(separator)

    if seg_analyses:
        for k, a in seg_analyses.items():
            _print_row(k, a)

    print("=" * len(header))


def print_ground_truth_report(
    analyses: Dict[str, AnalysisResult],
) -> None:
    """Print precision/recall/F1 for backends with ground truth metrics.

    Scene detection and speech segmentation results are printed in separate
    sections since they operate at different abstraction levels.
    """
    gt_analyses = {
        k: v for k, v in analyses.items() if v.gt_recall is not None
    }
    if not gt_analyses:
        return

    # Separate by type
    scene_gt = {
        k: v for k, v in gt_analyses.items()
        if v.backend_type == "scene_detection"
    }
    seg_gt = {
        k: v for k, v in gt_analyses.items()
        if v.backend_type == "speech_segmentation"
    }

    header = (
        f"{'Backend':<25} {'Recall':>8} {'Precision':>10} "
        f"{'F1':>6} {'Matched':>9}"
    )

    def _print_gt_rows(
        section_analyses: Dict[str, AnalysisResult],
    ) -> None:
        for key, a in section_analyses.items():
            matched_str = (
                f"{a.gt_matched_count}/{a.gt_total_count}"
                if a.gt_matched_count is not None
                else "-"
            )
            print(
                f"{a.backend_name:<25} "
                f"{a.gt_recall:>7.1%} "
                f"{a.gt_precision:>9.1%} "
                f"{a.gt_f1:>6.3f} "
                f"{matched_str:>9}"
            )

    print()
    print("=" * 70)
    print("GROUND TRUTH COMPARISON")
    print("=" * 70)

    if scene_gt:
        print("  Scene Detection:")
        print(header)
        print("-" * 70)
        _print_gt_rows(scene_gt)

    if scene_gt and seg_gt:
        print()
        print("  Speech Segmentation:")
        print(header)
        print("-" * 70)
        _print_gt_rows(seg_gt)
    elif seg_gt:
        print(header)
        print("-" * 70)
        _print_gt_rows(seg_gt)

    print("=" * 70)
    print(
        f"  Matching: >=50% temporal overlap counts as a match"
    )


def print_comparison_matrix(
    comparisons: List[ComparisonResult],
) -> None:
    """Print pairwise IoU comparison table (within-type only)."""
    if not comparisons:
        return

    print()
    print("=" * 60)
    print("PAIRWISE IoU COMPARISON")
    print("=" * 60)

    header = f"{'Backend A':<20} {'Backend B':<20} {'IoU':>6} {'Cov Diff':>9}"
    print(header)
    print("-" * 60)

    for c in comparisons:
        print(
            f"{c.backend_a:<20} {c.backend_b:<20} "
            f"{c.iou_score:>5.1%} "
            f"{c.coverage_diff:>8.1%}"
        )

    print("=" * 60)
    print("  (comparisons are within same backend type only)")


def print_parameters_detail(
    results: Dict[str, BackendRunResult],
) -> None:
    """Print detailed parameter presets for each backend."""
    print()
    print("=" * 70)
    print("BACKEND PARAMETERS")
    print("=" * 70)

    for key, result in results.items():
        if not result.available:
            error_msg = result.error or ""
            if "Unknown backend" in error_msg:
                print(f"\n  {result.display_name}: UNKNOWN BACKEND (not recognized)")
            else:
                print(f"\n  {result.display_name}: NOT INSTALLED — {error_msg}")
            continue

        if not result.success:
            print(f"\n  {result.display_name}: FAILED — {result.error}")
            continue

        print(f"\n  {result.display_name} ({result.backend_type}):")
        if result.method:
            print(f"    Method: {result.method}")

        if result.parameters:
            print("    Parameters:")
            for pkey, value in sorted(result.parameters.items()):
                print(f"      {pkey}: {value}")
        else:
            print("    Parameters: (none captured)")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def _segment_to_dict(seg: SegmentInfo) -> Dict[str, Any]:
    """Convert SegmentInfo to JSON-serializable dict."""
    d: Dict[str, Any] = {
        "start_sec": round(seg.start_sec, 4),
        "end_sec": round(seg.end_sec, 4),
        "duration_sec": round(seg.duration_sec, 4),
    }
    if seg.confidence != 1.0:
        d["confidence"] = round(seg.confidence, 4)
    if seg.metadata:
        # Filter to JSON-serializable values
        safe_meta = {}
        for k, v in seg.metadata.items():
            try:
                json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)
        if safe_meta:
            d["metadata"] = safe_meta
    return d


def export_json(
    results: Dict[str, BackendRunResult],
    analyses: Dict[str, AnalysisResult],
    comparisons: List[ComparisonResult],
    audio_info: AudioInfo,
    ground_truth: Optional[List[SegmentInfo]],
    sensitivity: Optional[str],
    output_path: Path,
) -> None:
    """Export complete analysis to JSON file.

    Args:
        results: All backend run results
        analyses: Computed metrics per backend
        comparisons: Pairwise comparison results
        audio_info: Audio metadata
        ground_truth: Ground truth segments (None if not provided)
        sensitivity: Sensitivity preset used
        output_path: Path to write JSON file
    """
    output: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "media_file": str(audio_info.source_media),
            "audio_duration_sec": round(audio_info.duration_sec, 4),
            "sample_rate": audio_info.sample_rate,
            "sensitivity": sensitivity,
            "suite_version": "1.0.0",
        },
        "ground_truth": None,
        "backends": {},
        "analyses": {},
        "comparisons": [],
    }

    # Ground truth
    if ground_truth:
        output["ground_truth"] = {
            "num_segments": len(ground_truth),
            "total_coverage_sec": round(
                sum(s.duration_sec for s in ground_truth), 4
            ),
            "segments": [_segment_to_dict(s) for s in ground_truth],
        }

    # Backend results
    for key, result in results.items():
        output["backends"][key] = {
            "backend_name": result.backend_name,
            "backend_type": result.backend_type,
            "display_name": result.display_name,
            "available": result.available,
            "success": result.success,
            "error": result.error,
            "method": result.method,
            "processing_time_sec": round(result.processing_time_sec, 4),
            "num_segments": len(result.segments),
            "parameters": result.parameters or {},
            "segments": [_segment_to_dict(s) for s in result.segments],
        }

    # Analysis metrics
    for key, analysis in analyses.items():
        a_dict: Dict[str, Any] = {
            "backend_name": analysis.backend_name,
            "backend_type": analysis.backend_type,
            "num_segments": analysis.num_segments,
            "total_coverage_sec": round(analysis.total_coverage_sec, 4),
            "coverage_ratio": round(analysis.coverage_ratio, 4),
            "mean_segment_duration": round(analysis.mean_segment_duration, 4),
            "median_segment_duration": round(
                analysis.median_segment_duration, 4
            ),
            "min_segment_duration": round(analysis.min_segment_duration, 4),
            "max_segment_duration": round(analysis.max_segment_duration, 4),
            "std_segment_duration": round(analysis.std_segment_duration, 4),
            "processing_time_sec": round(analysis.processing_time_sec, 4),
        }
        if analysis.gt_recall is not None:
            a_dict["ground_truth"] = {
                "recall": round(analysis.gt_recall, 4),
                "precision": round(analysis.gt_precision, 4)
                if analysis.gt_precision is not None
                else None,
                "f1": round(analysis.gt_f1, 4)
                if analysis.gt_f1 is not None
                else None,
                "matched_count": analysis.gt_matched_count,
                "total_count": analysis.gt_total_count,
            }
        output["analyses"][key] = a_dict

    # Pairwise comparisons
    for c in comparisons:
        output["comparisons"].append(
            {
                "backend_a": c.backend_a,
                "backend_b": c.backend_b,
                "iou_score": round(c.iou_score, 4),
                "coverage_diff": round(c.coverage_diff, 4),
                "segment_count_diff": c.segment_count_diff,
            }
        )

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to: {output_path}")
