"""
Shared utilities for scene detection backends.

Contains common operations used across multiple backends:
- WAV file saving (save_scene_wav)
- Brute-force time-based splitting (brute_force_split)

Audio loading:
    Backends should use load_audio_unified() from
    whisperjav.modules.scene_detection for audio loading.
    It will be moved here in Phase 4 cleanup.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from .base import SceneInfo

logger = logging.getLogger("whisperjav")


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
