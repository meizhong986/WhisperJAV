"""
Shared utilities for scene detection backends.

Contains common operations used across multiple backends:
- Audio loading (load_audio_unified)
- WAV file saving (save_scene_wav)
- Brute-force time-based splitting (brute_force_split)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from .base import SceneInfo

logger = logging.getLogger("whisperjav")


def load_audio_unified(
    audio_path: Path,
    target_sr: Optional[int] = None,
    force_mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Unified audio loading function used by all scene detectors.

    Provides consistent audio format and sample rate handling.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (None = preserve original)
        force_mono: Convert to mono if stereo

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        sf.SoundFileError: If file cannot be decoded
        MemoryError: If file is too large
        FileNotFoundError: If file doesn't exist
    """
    try:
        audio_data, sample_rate = sf.read(
            str(audio_path), dtype='float32', always_2d=False
        )

        # Convert stereo to mono efficiently if needed
        if force_mono and audio_data.ndim > 1:
            logger.debug("Converting stereo to mono")
            audio_data = np.mean(audio_data, axis=1)

        # Resample if target sample rate specified
        if target_sr is not None and sample_rate != target_sr:
            logger.debug(
                f"Resampling from {sample_rate}Hz to {target_sr}Hz"
            )
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=target_sr
            )
            sample_rate = target_sr

        logger.debug(
            f"Audio loaded: {len(audio_data)/sample_rate:.1f}s "
            f"@ {sample_rate}Hz, "
            f"{'mono' if audio_data.ndim == 1 else 'stereo'}"
        )
        return audio_data, sample_rate

    except sf.SoundFileError as e:
        file_size_mb = (
            audio_path.stat().st_size / 1024 / 1024
            if audio_path.exists() else 0
        )
        logger.error(
            f"SoundFile cannot read {audio_path}: {e}\n"
            f"  File size: {file_size_mb:.1f} MB\n"
            f"  Suggestion: Verify file integrity, try converting with ffmpeg:\n"
            f'    ffmpeg -i "{audio_path}" -acodec pcm_s16le output.wav'
        )
        raise
    except MemoryError as e:
        file_size_mb = (
            audio_path.stat().st_size / 1024 / 1024
            if audio_path.exists() else 0
        )
        logger.error(
            f"Out of memory loading {audio_path} ({file_size_mb:.1f} MB): {e}\n"
            f"  Suggestion: Close other applications or use smaller files"
        )
        raise
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        raise
    except Exception as e:
        logger.error(
            f"Failed to load audio {audio_path}: "
            f"{type(e).__name__}: {e}"
        )
        raise


def save_scene_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    scene_idx: int,
    output_dir: Path,
    media_basename: str,
) -> Path:
    """
    Save a scene audio segment as PCM16 WAV.

    Extracted from DynamicSceneDetector._save_scene() for reuse across backends.

    Args:
        audio_data: Audio data to save (1D numpy array)
        sample_rate: Sample rate of the audio
        scene_idx: Scene index (0-based, used for filename: *_scene_0001.wav)
        output_dir: Directory to save the WAV file (must already exist)
        media_basename: Base name for the file (e.g., "my_video")

    Returns:
        Path to the saved WAV file

    Raises:
        ValueError: If audio_data is empty or None
        PermissionError: If write permission denied
        OSError: If disk full or other I/O error
    """
    if audio_data is None or len(audio_data) == 0:
        raise ValueError(f"Empty audio data for scene {scene_idx}")

    scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
    scene_path = output_dir / scene_filename

    try:
        sf.write(str(scene_path), audio_data, sample_rate, subtype="PCM_16")
    except PermissionError as e:
        logger.error(
            f"Scene {scene_idx}: Permission denied writing to {scene_path}: {e}"
        )
        raise
    except OSError as e:
        logger.error(
            f"Scene {scene_idx}: OS error writing {scene_path} (disk full?): {e}"
        )
        raise

    return scene_path


def brute_force_split(
    start_sec: float,
    end_sec: float,
    chunk_duration: float,
    min_duration: float = 0.3,
) -> List[SceneInfo]:
    """
    Split a time range into fixed-duration chunks.

    Used as a fallback when silence-based splitting (Pass 2) fails to find
    sub-regions within a coarse region. Produces SceneInfo objects with
    time boundaries set but scene_path=None (caller must save WAV files).

    Args:
        start_sec: Start of the range to split (seconds)
        end_sec: End of the range to split (seconds)
        chunk_duration: Target duration for each chunk (seconds)
        min_duration: Minimum duration for a chunk to be kept (seconds)

    Returns:
        List of SceneInfo with time boundaries (scene_path not set,
        detection_pass=2, metadata includes split_method="brute_force")
    """
    scenes: List[SceneInfo] = []
    total_duration = end_sec - start_sec

    if total_duration <= 0:
        return scenes

    num_chunks = int(np.ceil(total_duration / max(chunk_duration, min_duration)))

    for i in range(num_chunks):
        sub_start = start_sec + i * chunk_duration
        sub_end = min(start_sec + (i + 1) * chunk_duration, end_sec)
        sub_dur = sub_end - sub_start

        if sub_dur < min_duration:
            continue

        scenes.append(SceneInfo(
            start_sec=sub_start,
            end_sec=sub_end,
            detection_pass=2,
            metadata={"split_method": "brute_force"},
        ))

    return scenes
