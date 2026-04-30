"""CLI entry point — orchestrates load → run → compute → report.

Exit codes:
    0 — success (>=1 backend produced a valid report)
    1 — media load failure
    2 — all backends failed (no useful output)
    3 — argument / configuration error (argparse returns its own 2, we re-map in main)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .loader import load_media_audio, parse_srt
from .metrics import compute_agreement_matrix, compute_timing_metrics
from .models import AnalysisReport, BackendReport, GtSegment
from .reporter import write_csv, write_html, write_json
from .runner import BackendRunner, VALID_SENSITIVITIES

logger = logging.getLogger("whisperjav.vad_gt")

DEFAULT_BACKENDS = ["silero-v3.1", "silero-v6.2", "ten", "whisperseg"]

EXIT_OK = 0
EXIT_MEDIA = 1
EXIT_ALL_FAILED = 2
EXIT_ARG = 3


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vad_groundtruth_analyser",
        description=(
            "Compare speech-segmenter backends on a media file. "
            "Ground truth SRT is optional; when absent, produces an "
            "inter-backend agreement matrix instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # With ground truth
  python -m tools.vad_groundtruth_analyser video.mp4 --ground-truth gt.srt

  # Without ground truth (still useful — agreement matrix + visual comparison)
  python -m tools.vad_groundtruth_analyser video.mp4

  # Pick backends and sensitivity
  python -m tools.vad_groundtruth_analyser audio.wav \\
      --backends whisperseg,silero-v6.2 --sensitivity balanced

  # CSV-only mode (for CI / batch collection)
  python -m tools.vad_groundtruth_analyser video.mp4 -g gt.srt \\
      --no-html --no-json
""",
    )

    p.add_argument("media", type=Path, help="audio or video file (extracts via ffmpeg when needed)")
    p.add_argument("-g", "--ground-truth", type=Path, default=None,
                   help="optional SRT file; every entry treated as a speech segment")
    p.add_argument("-b", "--backends", type=str, default=",".join(DEFAULT_BACKENDS),
                   metavar="LIST",
                   help=f"comma-separated backend names (default: {','.join(DEFAULT_BACKENDS)})")
    p.add_argument("-s", "--sensitivity", choices=VALID_SENSITIVITIES, default="aggressive",
                   help="sensitivity preset (default: aggressive)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output directory (default: same as media file)")
    p.add_argument("--frame-ms", type=int, default=10,
                   help="frame grid for frame-level metrics in ms (default: 10)")
    p.add_argument("--iou-match-threshold", type=float, default=0.1,
                   help="minimum IoU to count as a segment match (default: 0.1)")
    p.add_argument("--waveform-points", type=int, default=10000,
                   help="waveform downsample target for HTML (default: 10000)")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-backend timeout in seconds (default: 300)")
    p.add_argument("--no-html", action="store_true", help="skip HTML output")
    p.add_argument("--no-json", action="store_true", help="skip JSON output")
    p.add_argument("--no-csv", action="store_true", help="skip CSV output")
    p.add_argument("--title", type=str, default=None, help="custom HTML title")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _setup_logging(verbose: bool) -> None:
    root = logging.getLogger("whisperjav")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # If no handlers configured yet, add a concise stream handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        root.addHandler(h)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(report: AnalysisReport) -> None:
    print()
    print("=" * 96)
    gt_line = ""
    if report.ground_truth:
        gt_line = (f"  |  GT: {report.ground_truth['num_segments']} segments "
                   f"({report.ground_truth['coverage_ratio']*100:.1f}% coverage)")
    else:
        gt_line = "  |  GT: none (agreement matrix below)"
    print(f"VAD Ground-Truth Analyser — {Path(report.media_file).name} "
          f"({report.audio_duration_sec:.2f}s)")
    print(f"sensitivity: {report.sensitivity}{gt_line}")
    print("-" * 96)

    if report.ground_truth:
        hdr = f"{'backend':<16}{'status':<8}{'segs':>6}{'cov%':>8}" \
              f"{'F1':>8}{'P':>8}{'R':>8}{'miss%':>8}{'FA%':>8}{'time':>8}"
        print(hdr)
        for name, rep in report.backends.items():
            status = "OK" if rep.success else ("N/A" if not rep.available else "FAIL")
            if rep.success and rep.metrics:
                m = rep.metrics
                line = (
                    f"{name:<16}{status:<8}"
                    f"{rep.num_segments:>6}"
                    f"{rep.coverage_ratio*100:>7.1f}%"
                    f"{m.frame_f1:>8.3f}"
                    f"{m.frame_precision:>8.3f}"
                    f"{m.frame_recall:>8.3f}"
                    f"{m.missed_speech_pct:>7.1f}"
                    f"{m.false_alarm_pct:>7.1f}"
                    f"{rep.processing_time_sec:>7.2f}s"
                )
            elif rep.success:
                line = (
                    f"{name:<16}{status:<8}"
                    f"{rep.num_segments:>6}"
                    f"{rep.coverage_ratio*100:>7.1f}%"
                    f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}"
                    f"{rep.processing_time_sec:>7.2f}s"
                )
            else:
                line = f"{name:<16}{status:<8}{'-':>6}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}"
            print(line)
        # Highlight best F1
        okays = [(n, r.metrics.frame_f1) for n, r in report.backends.items()
                 if r.success and r.metrics is not None]
        if okays:
            best = max(okays, key=lambda x: x[1])
            print(f"\nBest F1: {best[0]} ({best[1]:.3f})")
    else:
        # GT-less summary
        hdr = f"{'backend':<16}{'status':<8}{'segs':>6}{'cov%':>8}{'time':>8}"
        print(hdr)
        for name, rep in report.backends.items():
            status = "OK" if rep.success else ("N/A" if not rep.available else "FAIL")
            if rep.success:
                print(f"{name:<16}{status:<8}{rep.num_segments:>6}"
                      f"{rep.coverage_ratio*100:>7.1f}%{rep.processing_time_sec:>7.2f}s")
            else:
                print(f"{name:<16}{status:<8}{'-':>6}{'-':>8}{'-':>8}")

        if report.agreement and len(report.agreement.backend_order) >= 2:
            print()
            print(f"Inter-backend agreement (pairwise F1, consensus={report.agreement.consensus_coverage_pct:.1f}%):")
            order = report.agreement.backend_order
            hdr2 = "            " + "".join(f"{n[:10]:>12}" for n in order)
            print(hdr2)
            for i, name in enumerate(order):
                row_str = f"{name[:10]:>10}  "
                row_str += "".join(f"{report.agreement.pair_f1[i][j]:>12.3f}" for j in range(len(order)))
                print(row_str)

    print("=" * 96)


# ---------------------------------------------------------------------------
# Core analysis (programmatic entry point)
# ---------------------------------------------------------------------------

def run_analysis(
    media_path: Path,
    gt_path: Optional[Path] = None,
    backends: Optional[List[str]] = None,
    sensitivity: str = "aggressive",
    frame_ms: int = 10,
    iou_match_threshold: float = 0.1,
    timeout_sec: int = 300,
) -> tuple[AnalysisReport, np.ndarray, List[GtSegment]]:
    """Run the full pipeline and return (report, waveform, gt_segments).

    Callers handle their own I/O (writers live in reporter.py). Designed for
    advanced-user automation — no side effects other than model loads.
    """
    media_path = Path(media_path)
    backends = list(backends) if backends else list(DEFAULT_BACKENDS)

    logger.info("Loading audio: %s", media_path)
    audio, sample_rate, duration = load_media_audio(media_path)

    gt_segments: List[GtSegment] = []
    gt_summary: Optional[Dict] = None
    if gt_path is not None:
        gt_path = Path(gt_path)
        logger.info("Loading ground truth: %s", gt_path)
        gt_segments = parse_srt(gt_path)
        if gt_segments:
            total_speech = sum(g.duration_sec for g in gt_segments)
            gt_summary = {
                "path": str(gt_path),
                "num_segments": len(gt_segments),
                "total_speech_sec": round(total_speech, 3),
                "coverage_ratio": round(total_speech / duration if duration > 0 else 0.0, 4),
            }
        else:
            logger.warning("GT SRT has zero segments; degrading to GT-less mode")
            gt_segments = []

    # Run backends sequentially (safer than parallel when sharing GPU)
    runner = BackendRunner(sensitivity=sensitivity, timeout_sec=timeout_sec)
    backend_reports: Dict[str, BackendReport] = {}
    for backend in backends:
        logger.info("-> %s", backend)
        rep = runner.run(backend, audio, sample_rate)
        backend_reports[backend] = rep

    # GT metrics per successful backend
    if gt_segments:
        for rep in backend_reports.values():
            if rep.success:
                rep.metrics = compute_timing_metrics(
                    gt_segments, rep.segments, duration,
                    frame_ms=frame_ms,
                    iou_match_threshold=iou_match_threshold,
                )

    # Agreement matrix across successful backends (always run, helps cross-check GT mode too)
    agreement = None
    n_ok = sum(1 for r in backend_reports.values() if r.success)
    if n_ok >= 2:
        agreement = compute_agreement_matrix(backend_reports, duration, frame_ms=frame_ms)

    report = AnalysisReport(
        schema_version="1.0.0",
        media_file=str(media_path),
        audio_duration_sec=round(float(duration), 3),
        sample_rate=int(sample_rate),
        sensitivity=sensitivity,
        frame_ms=int(frame_ms),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ground_truth=gt_summary,
        backends=backend_reports,
        agreement=agreement,
    )
    return report, audio, gt_segments


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_backends(raw: str) -> List[str]:
    lst = [b.strip() for b in raw.split(",") if b.strip()]
    if not lst:
        raise argparse.ArgumentTypeError("--backends must be non-empty")
    return lst


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        backends = _parse_backends(args.backends)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ARG
    if args.frame_ms <= 0 or args.frame_ms > 100:
        print("Error: --frame-ms must be in (0, 100]", file=sys.stderr)
        return EXIT_ARG
    if not (0.0 <= args.iou_match_threshold <= 1.0):
        print("Error: --iou-match-threshold must be in [0, 1]", file=sys.stderr)
        return EXIT_ARG
    if args.timeout <= 0:
        print("Error: --timeout must be > 0", file=sys.stderr)
        return EXIT_ARG

    # Media load
    try:
        report, audio, gt_segments = run_analysis(
            media_path=args.media,
            gt_path=args.ground_truth,
            backends=backends,
            sensitivity=args.sensitivity,
            frame_ms=args.frame_ms,
            iou_match_threshold=args.iou_match_threshold,
            timeout_sec=args.timeout,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_MEDIA
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"Error loading media or GT: {e}", file=sys.stderr)
        return EXIT_MEDIA

    # Summary print
    _print_summary(report)

    # Decide output paths
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.media).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.media).stem
    html_path = out_dir / f"{stem}.vad_analysis.html"
    json_path = out_dir / f"{stem}.vad_analysis.json"
    csv_path = out_dir / f"{stem}.vad_analysis.csv"

    # Write outputs
    if not args.no_json:
        write_json(report, json_path)
    if not args.no_csv:
        write_csv(report, csv_path)
    if not args.no_html:
        try:
            write_html(
                report=report,
                waveform=audio,
                path=html_path,
                title=args.title,
                waveform_points=args.waveform_points,
                gt_segments=gt_segments if gt_segments else None,
            )
        except ImportError as e:
            print(f"Warning: HTML not written ({e})", file=sys.stderr)

    # Exit code logic
    n_ok = sum(1 for r in report.backends.values() if r.success)
    if n_ok == 0:
        return EXIT_ALL_FAILED
    return EXIT_OK
