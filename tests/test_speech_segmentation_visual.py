#!/usr/bin/env python3
"""
Visual Speech Segmentation Backend Comparison Test

This script runs all available speech segmentation backends on a media file
and produces a visualization comparing their outputs.

Usage:
    python tests/test_speech_segmentation_visual.py path/to/media.mp4
    python tests/test_speech_segmentation_visual.py path/to/audio.wav --output results.png

Features:
    - Tests all available backends (silero-v4.0, silero-v3.1, nemo-lite, nemo-diarization, ten, none)
    - Extracts audio from video files automatically
    - Saves segment timestamps to JSON
    - Creates Gantt-chart style visualization with waveform
"""

import sys
import json
import time
import argparse
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for visual test - show INFO level from whisperjav
def setup_logging(verbose: bool = False):
    """Setup logging for the visual test."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure whisperjav logger
    wj_logger = logging.getLogger("whisperjav")
    wj_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    wj_logger.handlers.clear()

    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    wj_logger.addHandler(handler)

    return wj_logger

# Import directly to avoid import chain issues
from whisperjav.modules.speech_segmentation.base import (
    SpeechSegment,
    SegmentationResult,
)
from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory


@dataclass
class BackendResult:
    """Results from running a single backend."""
    name: str
    display_name: str
    available: bool
    success: bool
    error: Optional[str]
    processing_time_sec: float
    num_segments: int
    num_groups: int
    speech_coverage_ratio: float
    segments: List[Dict[str, float]]  # [{start_sec, end_sec}, ...]


def extract_audio(media_path: Path, output_path: Path, sample_rate: int = 16000) -> bool:
    """Extract audio from media file using FFmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(media_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(sample_rate),  # Sample rate
            "-ac", "1",  # Mono
            str(output_path)
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file and return (audio_data, sample_rate)."""
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, sample_rate
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")


def run_backend(
    backend_name: str,
    audio_data: np.ndarray,
    sample_rate: int,
    timeout_sec: int = 120
) -> BackendResult:
    """Run a single backend and return results.

    Args:
        backend_name: Name of the backend to test
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
        timeout_sec: Maximum time allowed for this backend (default 120s)
    """
    import threading
    import queue

    # Check availability first
    available, hint = SpeechSegmenterFactory.is_backend_available(backend_name)

    if not available:
        return BackendResult(
            name=backend_name,
            display_name=backend_name,
            available=False,
            success=False,
            error=f"Not available: {hint}",
            processing_time_sec=0.0,
            num_segments=0,
            num_groups=0,
            speech_coverage_ratio=0.0,
            segments=[]
        )

    # Use a queue to get results from thread
    result_queue = queue.Queue()

    def _run_segmentation():
        """Worker function to run segmentation in thread."""
        try:
            # Create segmenter
            segmenter = SpeechSegmenterFactory.create(backend_name)
            display_name = segmenter.display_name

            # Run segmentation
            start_time = time.time()
            result = segmenter.segment(audio_data, sample_rate=sample_rate)
            elapsed = time.time() - start_time

            # Extract segment timestamps
            segments = [
                {"start_sec": seg.start_sec, "end_sec": seg.end_sec}
                for seg in result.segments
            ]

            # Cleanup
            segmenter.cleanup()

            result_queue.put(BackendResult(
                name=backend_name,
                display_name=display_name,
                available=True,
                success=True,
                error=None,
                processing_time_sec=elapsed,
                num_segments=result.num_segments,
                num_groups=result.num_groups,
                speech_coverage_ratio=result.speech_coverage_ratio,
                segments=segments
            ))

        except Exception as e:
            result_queue.put(BackendResult(
                name=backend_name,
                display_name=backend_name,
                available=True,
                success=False,
                error=str(e),
                processing_time_sec=0.0,
                num_segments=0,
                num_groups=0,
                speech_coverage_ratio=0.0,
                segments=[]
            ))

    # Run in thread with timeout
    thread = threading.Thread(target=_run_segmentation, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        # Timeout occurred
        return BackendResult(
            name=backend_name,
            display_name=backend_name,
            available=True,
            success=False,
            error=f"Timeout after {timeout_sec}s",
            processing_time_sec=float(timeout_sec),
            num_segments=0,
            num_groups=0,
            speech_coverage_ratio=0.0,
            segments=[]
        )

    # Get result from queue
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return BackendResult(
            name=backend_name,
            display_name=backend_name,
            available=True,
            success=False,
            error="No result returned",
            processing_time_sec=0.0,
            num_segments=0,
            num_groups=0,
            speech_coverage_ratio=0.0,
            segments=[]
        )


def run_all_backends(
    audio_data: np.ndarray,
    sample_rate: int,
    backends: Optional[List[str]] = None,
    timeout_sec: int = 120
) -> Dict[str, BackendResult]:
    """Run all specified backends and return results.

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
        backends: List of backend names to test (default: practical set)
        timeout_sec: Timeout per backend in seconds
    """

    if backends is None:
        # Default: all practical backends including NeMo-Lite, Whisper VAD
        # Note: whisper-vad downloads models on first run
        backends = ["silero-v4.0", "silero-v3.1", "nemo-lite", "whisper-vad", "ten", "none"]
        print("  Note: whisper-vad (~500MB) downloads models on first run")

    results = {}

    for backend_name in backends:
        print(f"  Testing {backend_name}...", end=" ", flush=True)
        result = run_backend(backend_name, audio_data, sample_rate, timeout_sec=timeout_sec)
        results[backend_name] = result

        if result.success:
            print(f"OK ({result.num_segments} segments, {result.processing_time_sec:.2f}s)")
        elif not result.available:
            print(f"SKIP (not installed)")
        elif "Timeout" in (result.error or ""):
            print(f"TIMEOUT ({timeout_sec}s)")
        else:
            print(f"FAIL: {result.error}")

    return results


def create_visualization(
    audio_data: np.ndarray,
    sample_rate: int,
    results: Dict[str, BackendResult],
    output_path: Optional[Path] = None,
    title: str = "Speech Segmentation Backend Comparison"
) -> None:
    """Create Gantt-chart style visualization with waveform."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib is required for visualization: pip install matplotlib")
        return

    # Filter to only successful results
    successful_results = {k: v for k, v in results.items() if v.success}

    if not successful_results:
        print("No successful results to visualize")
        return

    # Calculate audio duration
    duration = len(audio_data) / sample_rate

    # Create figure with subplots
    num_backends = len(successful_results)
    fig_height = 2 + num_backends * 0.8  # Dynamic height based on backends

    fig, axes = plt.subplots(
        num_backends + 1, 1,
        figsize=(14, fig_height),
        sharex=True,
        gridspec_kw={'height_ratios': [2] + [1] * num_backends}
    )

    if num_backends == 0:
        axes = [axes]

    # Color palette for backends
    colors = {
        'silero-v4.0': '#2196F3',       # Blue
        'silero-v3.1': '#03A9F4',       # Light Blue
        'nemo': '#4CAF50',              # Green
        'nemo-lite': '#4CAF50',         # Green (fast frame VAD)
        'nemo-diarization': '#8BC34A',  # Light Green (full diarization)
        'ten': '#FF9800',               # Orange
        'none': '#9E9E9E',              # Grey
    }
    default_color = '#673AB7'  # Purple for unknown

    # Plot 1: Waveform
    ax_wave = axes[0]
    time_axis = np.linspace(0, duration, len(audio_data))
    ax_wave.plot(time_axis, audio_data, color='#424242', linewidth=0.3, alpha=0.8)
    ax_wave.fill_between(time_axis, audio_data, alpha=0.3, color='#757575')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_title(title, fontsize=12, fontweight='bold')
    ax_wave.set_xlim(0, duration)
    ax_wave.grid(True, alpha=0.3)

    # Plot speech regions on waveform (from first successful backend)
    first_result = list(successful_results.values())[0]
    for seg in first_result.segments:
        ax_wave.axvspan(seg['start_sec'], seg['end_sec'],
                       alpha=0.2, color='green', label='Speech')

    # Plot 2+: Backend results as Gantt chart
    for idx, (backend_name, result) in enumerate(successful_results.items()):
        ax = axes[idx + 1]
        color = colors.get(backend_name, default_color)

        # Draw segments as horizontal bars
        for seg in result.segments:
            start = seg['start_sec']
            width = seg['end_sec'] - seg['start_sec']
            ax.barh(0, width, left=start, height=0.6, color=color, alpha=0.8)

        # Style the axis
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

        # Add label with statistics
        stats = f"{result.display_name}\n{result.num_segments} segs | {result.speech_coverage_ratio:.1%} coverage | {result.processing_time_sec:.2f}s"
        ax.set_ylabel(stats, fontsize=8, rotation=0, ha='right', va='center', labelpad=100)

        # Add grid
        ax.grid(True, axis='x', alpha=0.3)

        # Add time ticks for last plot
        if idx == len(successful_results) - 1:
            ax.set_xlabel('Time (seconds)')

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def save_results_json(
    results: Dict[str, BackendResult],
    output_path: Path,
    media_path: Path,
    audio_duration: float
) -> None:
    """Save results to JSON file."""

    data = {
        "media_file": str(media_path),
        "audio_duration_sec": audio_duration,
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backends": {}
    }

    for name, result in results.items():
        data["backends"][name] = asdict(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def print_summary(results: Dict[str, BackendResult], duration: float) -> None:
    """Print summary table of results."""

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Backend':<20} {'Status':<10} {'Segments':<10} {'Coverage':<10} {'Time':<10}")
    print("-" * 80)

    for name, result in results.items():
        if result.success:
            status = "OK"
            segments = str(result.num_segments)
            coverage = f"{result.speech_coverage_ratio:.1%}"
            time_str = f"{result.processing_time_sec:.2f}s"
        elif not result.available:
            status = "N/A"
            segments = "-"
            coverage = "-"
            time_str = "-"
        else:
            status = "FAIL"
            segments = "-"
            coverage = "-"
            time_str = "-"

        print(f"{result.display_name:<20} {status:<10} {segments:<10} {coverage:<10} {time_str:<10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare speech segmentation backends on a media file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_speech_segmentation_visual.py video.mp4
    python tests/test_speech_segmentation_visual.py audio.wav --output comparison.png
    python tests/test_speech_segmentation_visual.py video.mp4 --backends silero-v4.0 silero-v3.1
        """
    )

    parser.add_argument(
        "media_file",
        type=Path,
        help="Path to media file (video or audio)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for visualization (PNG). If not specified, displays interactively."
    )

    parser.add_argument(
        "--json", "-j",
        type=Path,
        default=None,
        help="Output path for JSON results"
    )

    parser.add_argument(
        "--backends", "-b",
        nargs="+",
        default=None,
        help="Specific backends to test (default: all)"
    )

    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=16000,
        help="Sample rate for audio processing (default: 16000)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output"
    )

    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=120,
        help="Timeout per backend in seconds (default: 120)"
    )

    args = parser.parse_args()

    # Setup logging based on verbosity
    setup_logging(verbose=args.verbose)

    # Validate input file
    if not args.media_file.exists():
        print(f"Error: File not found: {args.media_file}")
        sys.exit(1)

    print(f"Speech Segmentation Backend Comparison Test")
    print(f"=" * 50)
    print(f"Input: {args.media_file}")
    print()

    # Determine if we need to extract audio
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    is_audio = args.media_file.suffix.lower() in audio_extensions

    if is_audio:
        audio_path = args.media_file
        print(f"Loading audio file...")
    else:
        print(f"Extracting audio from video...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = Path(tmp.name)

        if not extract_audio(args.media_file, audio_path, args.sample_rate):
            print("Error: Failed to extract audio")
            sys.exit(1)
        print(f"Audio extracted to: {audio_path}")

    # Load audio
    print(f"Loading audio data...")
    try:
        audio_data, sample_rate = load_audio(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    duration = len(audio_data) / sample_rate
    print(f"Audio duration: {duration:.2f}s ({sample_rate} Hz)")
    print()

    # Run all backends
    print("Running backends:")
    results = run_all_backends(audio_data, sample_rate, args.backends)

    # Print summary
    print_summary(results, duration)

    # Save JSON results if requested
    if args.json:
        save_results_json(results, args.json, args.media_file, duration)

    # Auto-generate JSON path if not specified
    if args.json is None:
        json_path = args.media_file.with_suffix('.segmentation_results.json')
        save_results_json(results, json_path, args.media_file, duration)

    # Create visualization
    print("\nCreating visualization...")

    # Auto-generate output path if not specified
    if args.output is None:
        output_path = args.media_file.with_suffix('.segmentation_comparison.png')
    else:
        output_path = args.output

    create_visualization(
        audio_data,
        sample_rate,
        results,
        output_path,
        title=f"Speech Segmentation: {args.media_file.name}"
    )

    # Cleanup temp file if we created one
    if not is_audio and audio_path.exists():
        audio_path.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
