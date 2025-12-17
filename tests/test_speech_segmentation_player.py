#!/usr/bin/env python3
"""
Interactive Speech Segmentation Player with Audio Playback

This script provides an interactive visualization of speech segmentation backends
with synchronized audio playback and a moving cursor line.

Usage:
    python tests/test_speech_segmentation_player.py path/to/media.mp4
    python tests/test_speech_segmentation_player.py path/to/audio.wav --backends silero-v4.0 nemo-lite

    # Use production-equivalent VAD parameters (RECOMMENDED for accurate testing):
    python tests/test_speech_segmentation_player.py path/to/media.mp4 --sensitivity balanced
    python tests/test_speech_segmentation_player.py path/to/media.mp4 --sensitivity aggressive

Controls:
    SPACE: Play/Stop toggle
    R: Reset to beginning
    Click: Seek to clicked position
    Q/ESC: Quit

Features:
    - Tests all available backends (silero-v4.0, silero-v3.1, nemo-lite, whisper-vad, ten, none)
    - Extracts audio from video files automatically
    - Real-time cursor line synchronized with audio playback
    - Click-to-seek functionality
    - --sensitivity flag to use production-equivalent VAD parameters
"""

import sys
import time
import argparse
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging for the player."""
    level = logging.DEBUG if verbose else logging.INFO
    wj_logger = logging.getLogger("whisperjav")
    wj_logger.setLevel(level)
    wj_logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    wj_logger.addHandler(handler)
    return wj_logger


# Import speech segmentation components
from whisperjav.modules.speech_segmentation.base import (
    SpeechSegment,
    SegmentationResult,
)
from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory

# Import config system for production-equivalent parameters
from whisperjav.config.transcription_tuner import TranscriptionTuner


def load_production_vad_config(sensitivity: str) -> Dict[str, Any]:
    """
    Load VAD parameters from production config (asr_config.json).

    This ensures the test uses the same parameter values that would be
    used when running whisperjav with --sensitivity <sensitivity>.

    Args:
        sensitivity: One of 'conservative', 'balanced', 'aggressive'

    Returns:
        Dict of VAD parameters matching production config
    """
    try:
        tuner = TranscriptionTuner()
        config = tuner.config

        # Get silero_vad_options for the given sensitivity
        # This is what balanced/fidelity pipelines use
        if 'silero_vad_options' in config and sensitivity in config['silero_vad_options']:
            vad_params = config['silero_vad_options'][sensitivity].copy()
            return vad_params
        else:
            print(f"  Warning: silero_vad_options/{sensitivity} not found in config")
            return {}
    except Exception as e:
        print(f"  Warning: Failed to load production config: {e}")
        return {}


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
    segments: List[Dict[str, float]]  # [{start_sec, end_sec, confidence}, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)  # Backend configuration parameters
    method: Optional[str] = None  # Method used (e.g., "nemo-lite:frame_vad")


class PlaybackState:
    """Track audio playback state."""

    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate
        self.is_playing = False
        self.position = 0.0  # Current position in seconds
        self.start_time = None  # When playback started (wall-clock time)

    def get_current_position(self) -> float:
        """Get current playback position in seconds."""
        if self.is_playing and self.start_time is not None:
            elapsed = time.time() - self.start_time
            position = min(elapsed, self.duration)
            return position
        return self.position


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


def load_audio(audio_path: Path, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """Load audio file and return (audio_data, sample_rate).

    Args:
        audio_path: Path to audio file
        target_sr: If specified, resample to this rate. If None, use native rate.

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if requested
        if target_sr is not None and target_sr != sample_rate:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
            except ImportError:
                # Fallback: simple decimation/interpolation
                ratio = target_sr / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
                sample_rate = target_sr

        return audio_data, sample_rate
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")


def downsample_for_display(audio_data: np.ndarray, max_points: int = 2000) -> np.ndarray:
    """Downsample audio for waveform display only.

    Uses max-pooling to preserve peaks for better waveform visualization.

    Args:
        audio_data: Full audio data
        max_points: Maximum number of points to display

    Returns:
        Downsampled audio suitable for plotting
    """
    if len(audio_data) <= max_points:
        return audio_data

    # Use max-pooling to preserve peaks (better waveform visualization)
    chunk_size = len(audio_data) // max_points
    n_chunks = len(audio_data) // chunk_size
    trimmed = audio_data[:n_chunks * chunk_size]
    reshaped = trimmed.reshape(n_chunks, chunk_size)

    # Take max absolute value per chunk to preserve peaks
    max_vals = np.max(np.abs(reshaped), axis=1)
    signs = np.sign(reshaped[:, chunk_size // 2])  # Use sign from middle of chunk
    return (max_vals * signs).astype(np.float32)


def parse_srt_file(srt_path: Path) -> List[Dict[str, float]]:
    """Parse SRT subtitle file and extract timestamps.

    Args:
        srt_path: Path to SRT file

    Returns:
        List of dicts with start_sec and end_sec for each subtitle
    """
    import re

    segments = []
    timestamp_pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
    )

    try:
        # Try UTF-8 first, then fallback to other encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(srt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Warning: Could not decode SRT file: {srt_path}")
            return []

        for match in timestamp_pattern.finditer(content):
            # Parse start time
            start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
            start_sec = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000

            # Parse end time
            end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
            end_sec = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

            segments.append({
                'start_sec': start_sec,
                'end_sec': end_sec
            })

        return segments

    except Exception as e:
        print(f"Warning: Failed to parse SRT file: {e}")
        return []


def run_backend(
    backend_name: str,
    audio_data: np.ndarray,
    sample_rate: int,
    timeout_sec: int = 120,
    vad_config: Optional[Dict[str, Any]] = None
) -> BackendResult:
    """Run a single backend and return results.

    Args:
        backend_name: Name of the backend to run
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
        timeout_sec: Timeout in seconds
        vad_config: Optional VAD configuration from production config.
                   If provided, these parameters are passed to the factory
                   to create the segmenter with production-equivalent settings.
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
            segments=[],
            parameters={},
            method=None
        )

    result_queue = queue.Queue()

    def _run_segmentation():
        try:
            # Create segmenter with optional production config
            # vad_config contains parameters like threshold, min_speech_duration_ms, etc.
            if vad_config:
                segmenter = SpeechSegmenterFactory.create(backend_name, config=vad_config)
            else:
                segmenter = SpeechSegmenterFactory.create(backend_name)
            display_name = segmenter.display_name

            # Get parameters BEFORE running (captures the preset values)
            parameters = {}
            if hasattr(segmenter, '_get_parameters'):
                parameters = segmenter._get_parameters()

            start_time = time.time()
            result = segmenter.segment(audio_data, sample_rate=sample_rate)
            elapsed = time.time() - start_time

            segments = [
                {"start_sec": seg.start_sec, "end_sec": seg.end_sec, "confidence": seg.confidence}
                for seg in result.segments
            ]

            # Get method from result (e.g., "nemo-lite:frame_vad")
            method = result.method if hasattr(result, 'method') else backend_name

            # Also capture parameters from result if available (may have additional info)
            if hasattr(result, 'parameters') and result.parameters:
                parameters.update(result.parameters)

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
                segments=segments,
                parameters=parameters,
                method=method
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
                segments=[],
                parameters={},
                method=None
            ))

    thread = threading.Thread(target=_run_segmentation, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
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
            segments=[],
            parameters={},
            method=None
        )

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
            segments=[],
            parameters={},
            method=None
        )


def run_all_backends(
    audio_data: np.ndarray,
    sample_rate: int,
    backends: Optional[List[str]] = None,
    timeout_sec: int = 120,
    vad_config: Optional[Dict[str, Any]] = None,
    sensitivity: Optional[str] = None
) -> Dict[str, BackendResult]:
    """Run all specified backends and return results.

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
        backends: List of backend names to test
        timeout_sec: Timeout per backend in seconds
        vad_config: Optional VAD configuration from production config
        sensitivity: Sensitivity level used (for display purposes)
    """

    if backends is None:
        # Include nemo-lite for testing (even though it may not be installed by default)
        backends = ["silero-v4.0", "silero-v3.1", "nemo-lite", "whisper-vad", "ten", "none"]
        print("  Note: whisper-vad (~500MB) and nemo-lite (~500MB) download models on first run")

    if vad_config:
        print(f"  Using production config for Silero backends (sensitivity={sensitivity or 'custom'}):")
        for key, value in sorted(vad_config.items()):
            print(f"    {key}: {value}")
        print("  Note: TEN, nemo-lite, whisper-vad use their own defaults")
        print()

    results = {}

    # VAD config from asr_config.json only applies to Silero backends
    silero_backends = {"silero-v4.0", "silero-v3.1", "silero"}

    for backend_name in backends:
        print(f"  Testing {backend_name}...", end=" ", flush=True)
        # Only pass vad_config to Silero backends (TEN and others use their own defaults)
        backend_config = vad_config if backend_name in silero_backends else None
        result = run_backend(backend_name, audio_data, sample_rate, timeout_sec=timeout_sec, vad_config=backend_config)
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


def save_results_to_json(
    results: Dict[str, BackendResult],
    output_path: Path,
    media_file: Path,
    audio_duration: float,
    sample_rate: int,
    ground_truth: List[Dict[str, float]] = None,
    sensitivity: Optional[str] = None,
    vad_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save all test results to a JSON file for later review.

    Args:
        results: Backend results from run_all_backends
        output_path: Path to save JSON file
        media_file: Original media file path
        audio_duration: Duration of audio in seconds
        sample_rate: Audio sample rate
        ground_truth: Optional ground truth segments from SRT
        sensitivity: Sensitivity level used (if any)
        vad_config: VAD configuration used (from production config)
    """
    import json
    from datetime import datetime

    # Build comprehensive output structure
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "media_file": str(media_file),
            "audio_duration_sec": audio_duration,
            "sample_rate": sample_rate,
            "test_version": "1.7.3",
            "sensitivity": sensitivity,
            "vad_config_used": vad_config or {},
        },
        "ground_truth": {
            "source": str(ground_truth) if ground_truth else None,
            "num_segments": len(ground_truth) if ground_truth else 0,
            "segments": ground_truth if ground_truth else [],
        },
        "backends": {}
    }

    # Add each backend result
    for name, result in results.items():
        backend_data = {
            "name": result.name,
            "display_name": result.display_name,
            "available": result.available,
            "success": result.success,
            "error": result.error,
            "method": result.method,
            "processing_time_sec": result.processing_time_sec,
            "num_segments": result.num_segments,
            "num_groups": result.num_groups,
            "speech_coverage_ratio": result.speech_coverage_ratio,
            "parameters": result.parameters or {},
            "segments": result.segments,
        }
        output["backends"][name] = backend_data

    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def print_parameters_summary(results: Dict[str, BackendResult]) -> None:
    """Print detailed parameter presets for each backend."""
    print("\n" + "=" * 70)
    print("BACKEND PARAMETER PRESETS")
    print("=" * 70)

    for name, result in results.items():
        if not result.available:
            print(f"\n{result.display_name or name}: NOT AVAILABLE")
            continue

        print(f"\n{result.display_name or name}:")
        print(f"  Backend name: {result.name}")
        if result.method:
            print(f"  Method used: {result.method}")
        print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")

        if result.error:
            print(f"  Error: {result.error}")

        if result.parameters:
            print("  Parameters:")
            for key, value in sorted(result.parameters.items()):
                print(f"    {key}: {value}")
        else:
            print("  Parameters: (none captured)")

        if result.success:
            print(f"  Results:")
            print(f"    Segments: {result.num_segments}")
            print(f"    Groups: {result.num_groups}")
            print(f"    Coverage: {result.speech_coverage_ratio:.1%}")
            print(f"    Time: {result.processing_time_sec:.2f}s")

    print("\n" + "=" * 70)


def add_cursor_lines(axes: List, duration: float) -> List:
    """Add vertical cursor line to all subplots.

    Args:
        axes: List of matplotlib axes
        duration: Audio duration in seconds

    Returns:
        List of cursor line objects
    """
    cursor_lines = []
    for ax in axes:
        line = ax.axvline(
            x=0,
            color='red',
            linewidth=2,
            linestyle='-',
            alpha=0.9,
            zorder=100  # Ensure cursor is on top
        )
        cursor_lines.append(line)
    return cursor_lines


def update_cursor_position(cursor_lines: List, position: float) -> None:
    """Update cursor position on all subplots.

    Args:
        cursor_lines: List of cursor line objects
        position: New position in seconds
    """
    for line in cursor_lines:
        line.set_xdata([position, position])


def play_audio(state: PlaybackState) -> None:
    """Start audio playback from current position."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is required: pip install sounddevice")
        return

    state.is_playing = True
    state.start_time = time.time() - state.position

    # Calculate start sample
    start_sample = int(state.position * state.sample_rate)
    audio_to_play = state.audio_data[start_sample:]

    # Non-blocking playback
    sd.play(audio_to_play, state.sample_rate)


def stop_audio(state: PlaybackState) -> None:
    """Stop audio playback."""
    try:
        import sounddevice as sd
        sd.stop()
    except ImportError:
        pass

    # Save current position before stopping
    if state.is_playing and state.start_time is not None:
        state.position = time.time() - state.start_time
        state.position = min(state.position, state.duration)

    state.is_playing = False
    state.start_time = None


def create_interactive_player(
    audio_data: np.ndarray,
    sample_rate: int,
    results: Dict[str, BackendResult],
    title: str = "Speech Segmentation Player",
    ground_truth: List[Dict[str, float]] = None
) -> None:
    """Create interactive visualization with audio playback.

    Args:
        audio_data: Audio data (16kHz) for both visualization and playback
        sample_rate: Sample rate of audio_data
        results: Backend segmentation results
        title: Window title
        ground_truth: Optional list of ground truth segments from SRT file

    Controls:
        SPACE: Play/Stop toggle
        R: Reset to beginning
        Click: Seek to clicked position
        Q/ESC: Quit
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return

    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is required for playback: pip install sounddevice")
        print("Continuing without audio playback...")
        sd = None

    # Filter to only successful results
    successful_results = {k: v for k, v in results.items() if v.success}

    if not successful_results:
        print("No successful results to visualize")
        return

    # Initialize playback state
    state = PlaybackState(audio_data, sample_rate)
    duration = state.duration

    # Calculate number of rows: waveform + ground truth (if provided) + backends
    has_ground_truth = ground_truth is not None and len(ground_truth) > 0
    num_backends = len(successful_results)
    num_rows = 1 + (1 if has_ground_truth else 0) + num_backends
    height_ratios = [2] + ([1] if has_ground_truth else []) + [1] * num_backends
    fig_height = 2 + (num_rows - 1) * 0.8

    fig, axes = plt.subplots(
        num_rows, 1,
        figsize=(14, fig_height),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios}
    )

    # Ensure axes is always a list
    if num_rows == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = list(axes)

    # Color palette for backends
    colors = {
        'silero-v4.0': '#2196F3',       # Blue
        'silero-v3.1': '#03A9F4',       # Light Blue
        'nemo': '#4CAF50',              # Green
        'nemo-lite': '#4CAF50',         # Green
        'whisper-vad': '#9C27B0',       # Purple
        'ten': '#FF9800',               # Orange
        'none': '#9E9E9E',              # Grey
    }
    default_color = '#673AB7'

    # Plot 1: Waveform (use downsampled data for performance)
    ax_wave = axes[0]
    display_audio = downsample_for_display(audio_data, max_points=2000)
    time_axis = np.linspace(0, duration, len(display_audio))
    ax_wave.plot(time_axis, display_audio, color='#424242', linewidth=0.5, alpha=0.8)
    ax_wave.fill_between(time_axis, display_audio, alpha=0.3, color='#757575')
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_title(title, fontsize=12, fontweight='bold')
    ax_wave.set_xlim(0, duration)
    ax_wave.grid(True, alpha=0.3)

    # Plot speech regions on waveform (from ground truth if available, else first backend)
    if has_ground_truth:
        for seg in ground_truth:
            ax_wave.axvspan(seg['start_sec'], seg['end_sec'],
                           alpha=0.2, color='gold', label='Ground Truth')
    else:
        first_result = list(successful_results.values())[0]
        for seg in first_result.segments:
            ax_wave.axvspan(seg['start_sec'], seg['end_sec'],
                           alpha=0.2, color='green', label='Speech')

    # Track current axis index
    ax_idx = 1

    # Plot Ground Truth (SRT) if provided - first track after waveform
    if has_ground_truth:
        ax_gt = axes[ax_idx]
        gt_color = '#FFD700'  # Gold

        # Calculate coverage
        total_speech = sum(seg['end_sec'] - seg['start_sec'] for seg in ground_truth)
        gt_coverage = total_speech / duration if duration > 0 else 0

        # Draw segments as horizontal bars
        for seg in ground_truth:
            start = seg['start_sec']
            width = seg['end_sec'] - seg['start_sec']
            ax_gt.barh(0, width, left=start, height=0.6, color=gt_color, alpha=0.9,
                      edgecolor='#B8860B', linewidth=0.5)

        # Style the axis
        ax_gt.set_xlim(0, duration)
        ax_gt.set_ylim(-0.5, 0.5)
        ax_gt.set_yticks([])

        # Add label
        stats = f"Ground Truth (SRT)\n{len(ground_truth)} segs | {gt_coverage:.1%}"
        ax_gt.set_ylabel(stats, fontsize=8, rotation=0, ha='right', va='center', labelpad=80)
        ax_gt.set_facecolor('#FFFACD')  # Light yellow background
        ax_gt.grid(True, axis='x', alpha=0.3)

        ax_idx += 1

    # Plot Backend results as Gantt chart
    for idx, (backend_name, result) in enumerate(successful_results.items()):
        ax = axes[ax_idx + idx]
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
        stats = f"{result.display_name}\n{result.num_segments} segs | {result.speech_coverage_ratio:.1%}"
        ax.set_ylabel(stats, fontsize=8, rotation=0, ha='right', va='center', labelpad=80)

        ax.grid(True, axis='x', alpha=0.3)

        if idx == len(successful_results) - 1:
            ax.set_xlabel('Time (seconds)')

    # Add cursor lines to all subplots
    cursor_lines = add_cursor_lines(axes, duration)

    # Add status text at bottom
    status_text = fig.text(
        0.5, 0.01,
        "Press SPACE to play/stop | R to reset | Click to seek | Q to quit",
        ha='center', va='bottom', fontsize=10,
        color='#666666', style='italic'
    )

    # Time display
    time_text = fig.text(
        0.02, 0.01,
        "0:00 / 0:00",
        ha='left', va='bottom', fontsize=10,
        color='#333333', family='monospace'
    )

    def format_time(seconds: float) -> str:
        """Format seconds as M:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    def animate(frame):
        """Update cursor position each frame."""
        if state.is_playing:
            position = state.get_current_position()

            # Check if playback finished
            if position >= duration:
                stop_audio(state)
                state.position = 0
                position = 0

            update_cursor_position(cursor_lines, position)
        else:
            position = state.position

        # Update time display
        time_text.set_text(f"{format_time(position)} / {format_time(duration)}")

        return cursor_lines + [time_text]

    def on_key(event):
        """Handle keyboard input."""
        if event.key == ' ':  # Spacebar toggles play/stop
            if state.is_playing:
                stop_audio(state)
                status_text.set_text("PAUSED | SPACE to play | R to reset | Click to seek | Q to quit")
            else:
                play_audio(state)
                status_text.set_text("PLAYING | SPACE to stop | R to reset | Click to seek | Q to quit")
            fig.canvas.draw_idle()

        elif event.key == 'r':  # Reset to beginning
            stop_audio(state)
            state.position = 0
            update_cursor_position(cursor_lines, 0)
            status_text.set_text("RESET | SPACE to play | R to reset | Click to seek | Q to quit")
            fig.canvas.draw_idle()

        elif event.key in ('q', 'escape'):  # Quit
            stop_audio(state)
            plt.close(fig)

    def on_click(event):
        """Handle mouse click to seek."""
        if event.inaxes and event.button == 1:  # Left click in axes
            new_position = event.xdata
            if new_position is not None and 0 <= new_position <= duration:
                was_playing = state.is_playing
                if was_playing:
                    stop_audio(state)

                state.position = new_position
                update_cursor_position(cursor_lines, new_position)

                if was_playing:
                    play_audio(state)

                fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Create animation (50ms interval = 20 FPS)
    ani = FuncAnimation(
        fig,
        animate,
        interval=50,
        blit=False,  # Set to False for text updates
        cache_frame_data=False
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for status text

    # Show interactive window
    plt.show()

    # Cleanup when window closes
    stop_audio(state)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive speech segmentation player with audio playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses backend defaults, NOT production parameters)
    python tests/test_speech_segmentation_player.py video.mp4

    # RECOMMENDED: Use production-equivalent VAD parameters
    python tests/test_speech_segmentation_player.py video.mp4 --sensitivity balanced
    python tests/test_speech_segmentation_player.py video.mp4 --sensitivity aggressive

    # Test specific backends
    python tests/test_speech_segmentation_player.py audio.wav --backends silero-v4.0 nemo-lite

    # Compare with ground truth SRT
    python tests/test_speech_segmentation_player.py video.mp4 --srt subtitles.srt --sensitivity balanced

    # Non-interactive (just save results)
    python tests/test_speech_segmentation_player.py video.mp4 --no-player --sensitivity balanced

Controls:
    SPACE: Play/Stop toggle
    R: Reset to beginning
    Click: Seek to clicked position
    Q/ESC: Quit
        """
    )

    parser.add_argument(
        "media_file",
        type=Path,
        help="Path to media file (video or audio)"
    )

    parser.add_argument(
        "--backends", "-b",
        nargs="+",
        default=None,
        help="Specific backends to test (default: silero-v4.0 silero-v3.1 nemo-lite whisper-vad ten none)"
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

    parser.add_argument(
        "--srt", "--ground-truth",
        type=Path,
        default=None,
        help="Path to SRT subtitle file for ground truth comparison"
    )

    parser.add_argument(
        "--output-json", "-o",
        type=Path,
        default=None,
        help="Save results to JSON file for later review"
    )

    parser.add_argument(
        "--no-player",
        action="store_true",
        help="Skip interactive player, just run tests and save results"
    )

    parser.add_argument(
        "--sensitivity", "-s",
        choices=["conservative", "balanced", "aggressive"],
        default=None,
        help="Use production VAD parameters for this sensitivity level. "
             "If not specified, uses backend defaults (which may differ from production)."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Validate input file
    if not args.media_file.exists():
        print(f"Error: File not found: {args.media_file}")
        sys.exit(1)

    print(f"Interactive Speech Segmentation Player")
    print(f"=" * 50)
    print(f"Input: {args.media_file}")

    # Parse ground truth SRT if provided
    ground_truth = None
    if args.srt:
        if not args.srt.exists():
            print(f"Warning: SRT file not found: {args.srt}")
        else:
            print(f"Ground truth: {args.srt}")
            ground_truth = parse_srt_file(args.srt)
            if ground_truth:
                print(f"  Loaded {len(ground_truth)} subtitle segments")
            else:
                print(f"  Warning: No segments found in SRT file")
    print()

    # Determine if we need to extract audio
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    is_audio = args.media_file.suffix.lower() in audio_extensions
    sample_rate = args.sample_rate  # Default 16kHz

    if is_audio:
        audio_path = args.media_file
        print(f"Loading audio file...")
    else:
        # Extract audio directly at 16kHz (for both VAD and playback)
        print(f"Extracting audio from video @ {sample_rate} Hz...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = Path(tmp.name)

        if not extract_audio(args.media_file, audio_path, sample_rate=sample_rate):
            print("Error: Failed to extract audio")
            sys.exit(1)
        print(f"Audio extracted to: {audio_path}")

    # Load audio at 16kHz for both VAD and playback (lightweight)
    print(f"Loading audio @ {sample_rate} Hz...")
    try:
        audio_data, sample_rate = load_audio(audio_path, target_sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    duration = len(audio_data) / sample_rate
    print(f"Audio: {duration:.2f}s @ {sample_rate} Hz ({len(audio_data):,} samples)")
    print()

    # Load production VAD config if sensitivity specified
    vad_config = None
    if args.sensitivity:
        print(f"Loading production config for sensitivity: {args.sensitivity}")
        vad_config = load_production_vad_config(args.sensitivity)
        if not vad_config:
            print("  Warning: Failed to load production config, using backend defaults")
        print()
    else:
        print("Note: Using backend defaults (no --sensitivity specified)")
        print("      To use production-equivalent parameters, add --sensitivity balanced")
        print()

    # Run all backends
    print("Running backends:")
    results = run_all_backends(
        audio_data, sample_rate, args.backends, args.timeout,
        vad_config=vad_config, sensitivity=args.sensitivity
    )

    # Print detailed parameter presets for each backend
    print_parameters_summary(results)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if result.success:
            print(f"  {result.display_name}: {result.num_segments} segments, {result.speech_coverage_ratio:.1%} coverage")
        elif not result.available:
            print(f"  {name}: Not available")
        else:
            print(f"  {name}: Failed - {result.error}")
    print("=" * 60)

    # Save results to JSON if requested
    if args.output_json:
        # Auto-generate filename if directory provided
        output_path = args.output_json
        if output_path.is_dir():
            output_path = output_path / f"segmentation_test_{args.media_file.stem}.json"

        save_results_to_json(
            results=results,
            output_path=output_path,
            media_file=args.media_file,
            audio_duration=duration,
            sample_rate=sample_rate,
            ground_truth=ground_truth,
            sensitivity=args.sensitivity,
            vad_config=vad_config
        )
    else:
        # Auto-save alongside media file
        auto_json_path = args.media_file.parent / f"{args.media_file.stem}_segmentation_test.json"
        save_results_to_json(
            results=results,
            output_path=auto_json_path,
            media_file=args.media_file,
            audio_duration=duration,
            sample_rate=sample_rate,
            ground_truth=ground_truth,
            sensitivity=args.sensitivity,
            vad_config=vad_config
        )

    # Launch interactive player (unless --no-player)
    if not args.no_player:
        print("\nLaunching interactive player...")
        print("Controls: SPACE=Play/Stop, R=Reset, Click=Seek, Q=Quit")
        print()

        create_interactive_player(
            audio_data,
            sample_rate,
            results,
            title=f"Speech Segmentation: {args.media_file.name}",
            ground_truth=ground_truth
        )
    else:
        print("\nSkipping interactive player (--no-player)")

    # Cleanup temp file if we created one
    if not is_audio and audio_path.exists():
        audio_path.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
