#!/usr/bin/env python3
"""
SRT Coverage Analysis Tool
===========================

Automated tool for analyzing SRT subtitle coverage by comparing
reference subtitles against WhisperJAV test outputs.

Features:
  - Temporal coverage analysis using interval overlap
  - Root cause tracing through pipeline metadata
  - Media chunk extraction for manual review
  - Visual Gantt chart timeline (static + interactive)
  - Comprehensive HTML and JSON reports

Usage:
    python test_srt_coverage.py \\
        --media path/to/video.mp4 \\
        --reference path/to/reference.srt \\
        --test path/to/test.srt \\
        --metadata path/to/metadata.json \\
        --output results/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import analysis modules
from srt_coverage_analysis import (
    parse_srt_file,
    calculate_srt_statistics,
    analyze_all_segments,
    calculate_summary_statistics,
    load_metadata,
    trace_all_segments,
    analyze_root_causes,
    extract_multiple_chunks,
    generate_gantt_chart,
    generate_json_report,
    generate_html_report,
)


def print_banner():
    """Print tool banner."""
    print("=" * 80)
    print(" " * 20 + "SRT Coverage Analysis Tool")
    print(" " * 20 + "WhisperJAV Quality Assurance")
    print("=" * 80)
    print()


def validate_inputs(args):
    """Validate input file paths."""
    errors = []

    if not Path(args.reference).exists():
        errors.append(f"Reference SRT not found: {args.reference}")

    if not Path(args.test).exists():
        errors.append(f"Test SRT not found: {args.test}")

    if not Path(args.media).exists():
        errors.append(f"Media file not found: {args.media}")

    if args.metadata and not Path(args.metadata).exists():
        errors.append(f"Metadata file not found: {args.metadata}")

    if errors:
        print("❌ Input validation failed:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    return True


def main():
    """Main entry point for SRT coverage analysis."""
    print_banner()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze SRT subtitle coverage by comparing reference vs test outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--media',
        required=True,
        help='Path to source media file (video or audio)'
    )

    parser.add_argument(
        '--reference',
        required=True,
        help='Path to reference SRT file (ground truth)'
    )

    parser.add_argument(
        '--test',
        required=True,
        help='Path to test SRT file (WhisperJAV output to analyze)'
    )

    parser.add_argument(
        '--metadata',
        help='Path to WhisperJAV metadata JSON file (optional, enables root cause tracing)'
    )

    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=0.60,
        help='Minimum coverage fraction to consider "covered" (default: 0.60 = 60%%)'
    )

    parser.add_argument(
        '--padding',
        type=float,
        default=1.0,
        help='Padding in seconds before/after segments for media extraction (default: 1.0s)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip media chunk extraction (faster analysis)'
    )

    parser.add_argument(
        '--skip-interactive',
        action='store_true',
        help='Skip interactive timeline generation (requires Plotly)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not validate_inputs(args):
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata dictionary (will be updated with SRT stats after parsing)
    metadata = {
        'reference_srt': str(Path(args.reference).absolute()),
        'test_srt': str(Path(args.test).absolute()),
        'media_file': str(Path(args.media).absolute()),
        'metadata_file': str(Path(args.metadata).absolute()) if args.metadata else None,
        'coverage_threshold': args.coverage_threshold,
        'padding_seconds': args.padding,
        'analysis_timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir.absolute()),
    }

    try:
        # Step 1: Parse SRT files
        print("📄 Loading SRT files...")
        print(f"  Reference: {args.reference}")
        ref_segments = parse_srt_file(args.reference)
        print(f"    ✓ Loaded {len(ref_segments)} segments")

        print(f"  Test: {args.test}")
        test_segments = parse_srt_file(args.test)
        print(f"    ✓ Loaded {len(test_segments)} segments")
        print()

        # Display basic SRT statistics
        print("📊 SRT File Statistics:")
        print()

        ref_stats = calculate_srt_statistics(ref_segments)
        print(f"  Reference SRT:")
        print(f"    Total lines:        {ref_stats['total_lines']}")
        print(f"    Total duration:     {ref_stats['total_duration_formatted']} ({ref_stats['total_duration']:.2f}s)")
        print(f"    Average duration:   {ref_stats['average_duration']:.2f}s per segment")
        print(f"    Timeline span:      {ref_stats['timeline_start']:.2f}s - {ref_stats['timeline_end']:.2f}s ({ref_stats['timeline_span']:.2f}s)")
        print()

        test_stats = calculate_srt_statistics(test_segments)
        print(f"  Test SRT:")
        print(f"    Total lines:        {test_stats['total_lines']}")
        print(f"    Total duration:     {test_stats['total_duration_formatted']} ({test_stats['total_duration']:.2f}s)")
        print(f"    Average duration:   {test_stats['average_duration']:.2f}s per segment")
        print(f"    Timeline span:      {test_stats['timeline_start']:.2f}s - {test_stats['timeline_end']:.2f}s ({test_stats['timeline_span']:.2f}s)")
        print()

        # Add SRT statistics to metadata
        metadata['reference_stats'] = ref_stats
        metadata['test_stats'] = test_stats

        # Step 2: Calculate coverage
        print("📊 Calculating coverage...")
        coverage_results = analyze_all_segments(
            ref_segments,
            test_segments,
            coverage_threshold=args.coverage_threshold
        )
        summary_stats = calculate_summary_statistics(coverage_results)

        print(f"  ✓ Analyzed {summary_stats['total_segments']} reference segments")
        print(f"    - Covered: {summary_stats['covered_segments']} ({summary_stats['covered_percent']:.1f}%)")
        print(f"    - Partial: {summary_stats['partial_segments']} ({summary_stats['partial_percent']:.1f}%)")
        print(f"    - Missing: {summary_stats['missing_segments']} ({summary_stats['missing_percent']:.1f}%)")
        print()

        # Step 3: Trace through metadata (if provided)
        trace_results = []
        root_cause_stats = {}

        if args.metadata:
            print("🔍 Tracing missing/partial segments through metadata...")
            pipeline_metadata = load_metadata(args.metadata)

            segments_to_trace = [
                r.ref_segment for r in coverage_results if r.needs_review
            ]

            trace_results = trace_all_segments(
                segments_to_trace,
                coverage_results,
                pipeline_metadata
            )

            root_cause_stats = analyze_root_causes(trace_results)

            print(f"  ✓ Traced {len(trace_results)} segments")
            print(f"    - Not in scene: {root_cause_stats['not_in_scene']}")
            print(f"    - Scene not transcribed: {root_cause_stats['scene_not_transcribed']}")
            print(f"    - Filtered/Failed: {root_cause_stats['filtered_or_failed']}")
            print()
        else:
            print("⚠ No metadata provided - skipping root cause tracing")
            print()

        # Step 4: Extract media chunks
        extraction_stats = {'total': 0, 'successful': 0, 'failed': 0, 'missing_files': [], 'partial_files': []}

        if not args.skip_extraction:
            print("🎬 Extracting media chunks for review...")
            chunks_dir = output_dir / "chunks"

            segments_to_extract = [
                r.ref_segment for r in coverage_results if r.needs_review
            ]

            if segments_to_extract:
                extraction_stats = extract_multiple_chunks(
                    media_path=args.media,
                    segments=segments_to_extract,
                    coverage_results=coverage_results,
                    output_path=str(chunks_dir),
                    padding_seconds=args.padding
                )

                print(f"  ✓ Extracted {extraction_stats['successful']}/{extraction_stats['total']} chunks")
                print(f"    - Missing: {len(extraction_stats['missing_files'])}")
                print(f"    - Partial: {len(extraction_stats['partial_files'])}")
                if extraction_stats['failed'] > 0:
                    print(f"    - Failed: {extraction_stats['failed']}")
                print()
            else:
                print("  ✓ No segments need extraction (all covered)")
                print()
        else:
            print("⚠ Skipping media chunk extraction (--skip-extraction)")
            print()

        # Step 5: Generate visualizations
        print("📈 Generating timeline visualizations...")
        timeline_files = generate_gantt_chart(
            coverage_results=coverage_results,
            test_segments=test_segments,
            output_dir=str(output_dir),
            generate_static=True,
            generate_interactive=not args.skip_interactive,
            max_segments_static=50
        )

        if timeline_files['static_chart']:
            print(f"  ✓ Static timeline: {Path(timeline_files['static_chart']).name}")

        if timeline_files['interactive_chart']:
            print(f"  ✓ Interactive timeline: {Path(timeline_files['interactive_chart']).name}")

        print()

        # Step 6: Generate reports
        print("📝 Generating reports...")

        # JSON report
        json_path = output_dir / "report.json"
        generate_json_report(
            reference_segments=ref_segments,
            test_segments=test_segments,
            coverage_results=coverage_results,
            trace_results=trace_results,
            summary_stats=summary_stats,
            root_cause_stats=root_cause_stats,
            extraction_stats=extraction_stats,
            metadata=metadata,
            output_path=str(json_path)
        )
        print(f"  ✓ JSON report: {json_path.name}")

        # HTML report
        html_path = output_dir / "report.html"
        generate_html_report(
            reference_segments=ref_segments,
            test_segments=test_segments,
            coverage_results=coverage_results,
            trace_results=trace_results,
            summary_stats=summary_stats,
            root_cause_stats=root_cause_stats,
            extraction_stats=extraction_stats,
            metadata=metadata,
            timeline_png_path="timeline.png",  # Relative path
            timeline_html_path="timeline_interactive.html",  # Relative path
            output_path=str(html_path)
        )
        print(f"  ✓ HTML report: {html_path.name}")
        print()

        # Final summary
        print("=" * 80)
        print("✅ Analysis complete!")
        print("=" * 80)
        print()
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"📊 Coverage rate: {summary_stats['covered_percent']:.1f}%")
        print(f"⚠  Segments needing review: {summary_stats['partial_segments'] + summary_stats['missing_segments']}")
        print()
        print(f"👉 Open {html_path.name} in your browser to view the full report")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Analysis failed!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
