#!/usr/bin/env python3
"""
Pipeline Analysis Test Suite — Entry Point

Interactive tool for running scene detection and speech segmentation backends
side by side, computing comparison metrics, and visualizing results.

Usage:
    python -m tests.pipeline_analysis path/to/media.mp4
    python -m tests.pipeline_analysis video.mp4 --scene-backends auditok silero semantic
    python -m tests.pipeline_analysis video.mp4 --seg-backends silero-v6.2 ten
    python -m tests.pipeline_analysis video.mp4 --srt ground-truth.srt --sensitivity aggressive

Controls (interactive player):
    SPACE: Play/Stop toggle
    R: Reset to beginning
    Click: Seek to clicked position
    Q/ESC: Quit
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure parent directory is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.pipeline_analysis import analyzer, presenter, runner, visualizer
from tests.pipeline_analysis.models import (
    AnalysisResult,
    BackendRunResult,
    ComparisonResult,
    SegmentInfo,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the test suite."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure whisperjav logger
    wj_logger = logging.getLogger("whisperjav")
    wj_logger.setLevel(level)
    wj_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    wj_logger.addHandler(handler)

    # Configure our own logger
    pa_logger = logging.getLogger("tests.pipeline_analysis")
    pa_logger.setLevel(level)
    pa_logger.handlers.clear()
    pa_logger.addHandler(handler)


def print_available_backends() -> None:
    """Print all available backends for user reference."""
    print()
    print("Available scene detection backends:")
    for info in runner.get_available_scene_backends():
        status = "OK" if info.get("available", False) else "not installed"
        print(f"  {info['name']:<15} ({status})")

    print()
    print("Available speech segmentation backends:")
    for info in runner.get_available_seg_backends():
        status = "OK" if info.get("available", False) else "not installed"
        print(f"  {info['name']:<15} ({status})")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline Analysis Test Suite — compare scene detection "
        "and speech segmentation backends on real media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default backends (auditok + silero scene; silero-v4.0 + ten speech)
    python -m tests.pipeline_analysis video.mp4

    # Compare all three scene detectors
    python -m tests.pipeline_analysis video.mp4 --scene-backends auditok silero semantic

    # Compare speech segmenters with production parameters
    python -m tests.pipeline_analysis video.mp4 \\
        --seg-backends silero-v6.2 ten silero-v4.0 \\
        --sensitivity aggressive

    # With ground truth SRT for recall/precision analysis
    python -m tests.pipeline_analysis video.mp4 \\
        --srt ground-truth.srt \\
        --scene-backends semantic \\
        --seg-backends ten

    # Non-interactive (just reports + JSON export)
    python -m tests.pipeline_analysis video.mp4 --no-player

    # List available backends
    python -m tests.pipeline_analysis --list-backends

Controls (interactive player):
    SPACE: Play/Stop toggle
    R: Reset to beginning
    Click: Seek to clicked position
    Q/ESC: Quit
        """,
    )

    parser.add_argument(
        "media_file",
        type=Path,
        nargs="?",
        help="Path to media file (video or audio)",
    )

    parser.add_argument(
        "--scene-backends",
        "-sb",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Scene detection backends to test (default: auditok silero)",
    )

    parser.add_argument(
        "--seg-backends",
        "-vb",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Speech segmentation backends to test (default: silero-v4.0 ten)",
    )

    parser.add_argument(
        "--sensitivity",
        "-s",
        choices=["conservative", "balanced", "aggressive"],
        default=None,
        help="Use production VAD parameters for this sensitivity level "
        "(applies to Silero segmenters only)",
    )

    parser.add_argument(
        "--srt",
        "--ground-truth",
        type=Path,
        default=None,
        help="SRT subtitle file for ground truth comparison",
    )

    parser.add_argument(
        "--output-json",
        "-o",
        type=Path,
        default=None,
        help="JSON export path (default: auto-generated alongside media file)",
    )

    parser.add_argument(
        "--no-player",
        action="store_true",
        help="Skip interactive visualization, just run analysis and export",
    )

    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable coverage heatmap row in visualization",
    )

    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Timeout per backend in seconds (default: 120)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output",
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit",
    )

    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Ground truth matching overlap threshold 0-1 (default: 0.5)",
    )

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Handle --list-backends
    if args.list_backends:
        print_available_backends()
        sys.exit(0)

    # Validate media file
    if args.media_file is None:
        parser.error("media_file is required (use --list-backends to see options)")

    if not args.media_file.exists():
        print(f"Error: File not found: {args.media_file}")
        sys.exit(1)

    # Setup
    setup_logging(args.verbose)

    print("Pipeline Analysis Test Suite")
    print("=" * 50)
    print(f"Input: {args.media_file}")
    if args.sensitivity:
        print(f"Sensitivity: {args.sensitivity}")
    print()

    # --- Step 1: Prepare audio ---
    print("Preparing audio...")
    try:
        audio_info, audio_data = runner.prepare_audio(
            args.media_file, args.sample_rate
        )
    except Exception as e:
        print(f"Error preparing audio: {e}")
        sys.exit(1)

    print(
        f"  Audio: {audio_info.duration_sec:.2f}s @ "
        f"{audio_info.sample_rate} Hz "
        f"({audio_info.num_samples:,} samples)"
    )
    print()

    # --- Step 2: Parse ground truth ---
    ground_truth: Optional[List[SegmentInfo]] = None
    if args.srt:
        if not args.srt.exists():
            print(f"Warning: SRT file not found: {args.srt}")
        else:
            print(f"Ground truth: {args.srt}")
            ground_truth = analyzer.parse_srt_file(args.srt)
            if ground_truth:
                gt_coverage = sum(s.duration_sec for s in ground_truth)
                print(
                    f"  Loaded {len(ground_truth)} segments "
                    f"({gt_coverage:.1f}s coverage)"
                )
            else:
                print("  Warning: No segments found in SRT file")
            print()

    # --- Step 3: Run backends ---
    # Detect whether user explicitly specified each backend type
    # (vs leaving them at default None, which means "use defaults")
    user_specified_scene = args.scene_backends is not None
    user_specified_seg = args.seg_backends is not None

    results = runner.run_all_backends(
        audio_info=audio_info,
        audio_data=audio_data,
        scene_backends=args.scene_backends,
        seg_backends=args.seg_backends,
        timeout_sec=args.timeout,
        sensitivity=args.sensitivity,
        user_specified_scene=user_specified_scene,
        user_specified_seg=user_specified_seg,
    )

    # Check if anything succeeded
    successful = {k: v for k, v in results.items() if v.success}
    if not successful:
        print("No backends completed successfully. Nothing to analyze.")
        runner.cleanup_audio(audio_info)
        sys.exit(1)

    # --- Step 4: Analyze ---
    analyses: Dict[str, AnalysisResult] = {}
    for key, result in results.items():
        if not result.success:
            continue
        if ground_truth:
            analyses[key] = analyzer.compute_ground_truth_metrics(
                result,
                ground_truth,
                audio_info.duration_sec,
                overlap_threshold=args.overlap_threshold,
            )
        else:
            analyses[key] = analyzer.compute_metrics(
                result, audio_info.duration_sec
            )

    # Pairwise comparisons (within same backend type only — comparing
    # scene detection vs speech segmentation is meaningless)
    comparisons: List[ComparisonResult] = []
    successful_keys = list(successful.keys())
    for i in range(len(successful_keys)):
        for j in range(i + 1, len(successful_keys)):
            result_i = results[successful_keys[i]]
            result_j = results[successful_keys[j]]
            # Only compare backends of the same type
            if result_i.backend_type != result_j.backend_type:
                continue
            comparisons.append(
                analyzer.compute_pairwise_comparison(
                    result_i,
                    result_j,
                    audio_info.duration_sec,
                )
            )

    # --- Step 5: Present ---
    presenter.print_summary_table(analyses, audio_info.duration_sec)
    presenter.print_parameters_detail(results)

    if comparisons:
        presenter.print_comparison_matrix(comparisons)

    if ground_truth:
        presenter.print_ground_truth_report(analyses)

    # --- Step 6: Export JSON ---
    json_path = args.output_json
    if json_path is None:
        json_path = (
            args.media_file.parent
            / f"{args.media_file.stem}_pipeline_analysis.json"
        )

    presenter.export_json(
        results=results,
        analyses=analyses,
        comparisons=comparisons,
        audio_info=audio_info,
        ground_truth=ground_truth,
        sensitivity=args.sensitivity,
        output_path=json_path,
    )

    # --- Step 7: Visualize ---
    if not args.no_player:
        print("\nLaunching interactive player...")
        print("Controls: SPACE=Play/Stop, R=Reset, Click=Seek, Q=Quit")
        print()

        visualizer.create_interactive_player(
            audio_data=audio_data,
            sample_rate=audio_info.sample_rate,
            results=results,
            analyses=analyses,
            title=f"Pipeline Analysis: {args.media_file.name}",
            ground_truth=ground_truth,
            show_heatmap=not args.no_heatmap,
        )
    else:
        print("\nSkipping interactive player (--no-player)")

    # --- Cleanup ---
    runner.cleanup_audio(audio_info)
    print("\nDone!")


if __name__ == "__main__":
    main()
