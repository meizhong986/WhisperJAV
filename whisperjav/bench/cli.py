"""
CLI entry point for the Qwen pipeline benchmarking utility.

Usage:
    whisperjav-bench \\
        --ground-truth /path/to/ground_truth.srt \\
        --test /path/to/test_output_t1 "T1: vad_slicing" \\
        --test /path/to/test_output_t2 "T2: context_aware" \\
        --output /path/to/report.json \\
        --scene-detail 0 3
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisperjav-bench",
        description="Compare Qwen pipeline test outputs against ground-truth SRT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisperjav-bench \\
    --ground-truth gt.srt \\
    --test ./temp_t1 "T1: vad_slicing" \\
    --test ./temp_t2 "T2: context_aware" \\
    --test ./temp_t3 "T3: assembly"

  whisperjav-bench \\
    --ground-truth gt.srt \\
    --test ./temp_t1 "T1" \\
    --scene-detail 0 3 \\
    --output report.json
        """,
    )

    parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        type=Path,
        help="Path to ground-truth SRT file.",
    )
    parser.add_argument(
        "--test", "-t",
        action="append",
        nargs=2,
        metavar=("PATH", "LABEL"),
        required=True,
        help="Test output directory and label. Repeat for multiple tests.",
    )
    parser.add_argument(
        "--name", "-n",
        default="",
        help="Benchmark name (shown in report header).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to write JSON report (optional).",
    )
    parser.add_argument(
        "--scene-detail", "-s",
        type=int,
        nargs="*",
        default=None,
        help="Scene indices to show detailed drill-down for.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full sub provenance traceability table for each test.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Import here to keep CLI startup fast
    from whisperjav.bench.loader import load_ground_truth, load_test_result
    from whisperjav.bench.report import (
        analyze,
        print_scene_detail,
        print_summary,
        print_timing_analytics,
        print_traceability_table,
        write_json_report,
    )

    # Load ground truth
    try:
        gt_subs = load_ground_truth(args.ground_truth)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading ground truth: {e}", file=sys.stderr)
        return 1

    # Load test results
    tests = []
    for path_str, label in args.test:
        test_path = Path(path_str)
        try:
            result = load_test_result(test_path, label)
            tests.append(result)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading test '{label}': {e}", file=sys.stderr)
            return 1

    if not tests:
        print("No test results loaded.", file=sys.stderr)
        return 1

    # Analyze
    analysis = analyze(gt_subs, tests)

    # Print console summary
    print_summary(analysis, benchmark_name=args.name)

    # Timing source analytics (always shown when provenance data exists)
    print_timing_analytics(analysis)

    # Full provenance traceability table
    if args.trace:
        for t_idx in range(len(tests)):
            print_traceability_table(analysis, t_idx)

    # Scene drill-down
    if args.scene_detail is not None:
        for scene_idx in args.scene_detail:
            print_scene_detail(analysis, scene_idx, gt_subs, tests)

    # JSON report
    if args.output:
        write_json_report(analysis, args.output)
        print(f"JSON report written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
