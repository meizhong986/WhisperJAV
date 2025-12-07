"""
Waveform processing utilities for visualization.

Handles downsampling large audio files for browser performance.
"""

import numpy as np
from typing import Tuple


def downsample_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    target_points: int = 10000,
    method: str = "minmax"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample waveform for efficient browser rendering.

    For a 2-hour audio at 16kHz, this reduces 115M samples to ~10K points
    while preserving the visual envelope.

    Args:
        waveform: Raw audio samples (1D array)
        sample_rate: Sample rate in Hz
        target_points: Target number of points for visualization
        method: Downsampling method ('minmax', 'rms', 'peak')

    Returns:
        Tuple of (time_array, min_envelope, max_envelope)
    """
    total_samples = len(waveform)
    duration_seconds = total_samples / sample_rate

    # Calculate samples per visualization point
    samples_per_point = max(1, total_samples // target_points)
    actual_points = total_samples // samples_per_point

    print(f"Downsampling: {total_samples:,} samples -> {actual_points:,} points ({samples_per_point:,}x reduction)")

    # Create time array
    time_array = np.linspace(0, duration_seconds, actual_points)

    if method == "minmax":
        # Min/max envelope (best for preserving peaks)
        min_envelope = np.zeros(actual_points)
        max_envelope = np.zeros(actual_points)

        for i in range(actual_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, total_samples)
            chunk = waveform[start_idx:end_idx]

            if len(chunk) > 0:
                min_envelope[i] = np.min(chunk)
                max_envelope[i] = np.max(chunk)

    elif method == "rms":
        # RMS envelope (better for overall energy)
        min_envelope = np.zeros(actual_points)
        max_envelope = np.zeros(actual_points)

        for i in range(actual_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, total_samples)
            chunk = waveform[start_idx:end_idx]

            if len(chunk) > 0:
                rms = np.sqrt(np.mean(chunk ** 2))
                min_envelope[i] = -rms
                max_envelope[i] = rms

    elif method == "peak":
        # Peak envelope (simplified, faster)
        min_envelope = np.zeros(actual_points)
        max_envelope = np.zeros(actual_points)

        for i in range(actual_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, total_samples)
            chunk = waveform[start_idx:end_idx]

            if len(chunk) > 0:
                peak = np.max(np.abs(chunk))
                min_envelope[i] = -peak
                max_envelope[i] = peak

    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax', 'rms', or 'peak'")

    return time_array, min_envelope, max_envelope


def compute_waveform_statistics(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Compute basic waveform statistics.

    Args:
        waveform: Raw audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with statistics
    """
    duration = len(waveform) / sample_rate

    # RMS level
    rms = np.sqrt(np.mean(waveform ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # Peak level
    peak = np.max(np.abs(waveform))
    peak_db = 20 * np.log10(peak + 1e-10)

    # Crest factor (peak to RMS ratio)
    crest_factor = peak / (rms + 1e-10)
    crest_factor_db = 20 * np.log10(crest_factor + 1e-10)

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "total_samples": len(waveform),
        "rms_level_db": round(rms_db, 2),
        "peak_level_db": round(peak_db, 2),
        "crest_factor_db": round(crest_factor_db, 2),
    }


def adaptive_downsample(
    waveform: np.ndarray,
    sample_rate: int,
    min_points: int = 5000,
    max_points: int = 50000,
    target_px_per_second: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptively downsample based on duration and display requirements.

    For short clips: More detail
    For long videos: Fewer points

    Args:
        waveform: Raw audio samples
        sample_rate: Sample rate in Hz
        min_points: Minimum visualization points
        max_points: Maximum visualization points
        target_px_per_second: Target points per second of audio

    Returns:
        Tuple of (time_array, min_envelope, max_envelope)
    """
    duration = len(waveform) / sample_rate

    # Calculate ideal points based on duration
    ideal_points = int(duration * target_px_per_second)

    # Clamp to bounds
    target_points = max(min_points, min(max_points, ideal_points))

    return downsample_waveform(waveform, sample_rate, target_points, method="minmax")
