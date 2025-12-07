"""
Data loaders for WhisperJAV visualization utility.

Loads:
- WAV audio files (waveform data)
- JSON metadata (scene detection, VAD segments)
- SRT subtitle files
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class SceneInfo:
    """Scene detection metadata."""
    scene_index: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    detection_pass: Optional[int] = None  # 1=coarse, 2=fine
    filename: Optional[str] = None


@dataclass
class VadSegment:
    """VAD speech segment."""
    start_sec: float
    end_sec: float


@dataclass
class Subtitle:
    """SRT subtitle entry."""
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class VisualizationData:
    """Container for all visualization data."""
    # Audio waveform
    waveform: Optional[np.ndarray] = None
    sample_rate: int = 16000
    duration_seconds: float = 0.0

    # Scene detection
    scenes: List[SceneInfo] = field(default_factory=list)
    coarse_boundaries: List[SceneInfo] = field(default_factory=list)  # Pass 1 boundaries

    # VAD segments (when using Silero)
    vad_segments: List[VadSegment] = field(default_factory=list)
    vad_method: Optional[str] = None
    vad_params: Optional[Dict] = None

    # Subtitles
    subtitles: List[Subtitle] = field(default_factory=list)

    # Metadata
    metadata_raw: Optional[Dict] = None


def load_audio(audio_path: Path) -> tuple:
    """
    Load WAV audio file.

    Args:
        audio_path: Path to WAV file

    Returns:
        Tuple of (waveform_array, sample_rate)
    """
    import soundfile as sf

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    return audio_data, sample_rate


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load WhisperJAV master metadata JSON.

    Args:
        metadata_path: Path to _master.json file

    Returns:
        Parsed metadata dictionary
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_srt(srt_path: Path) -> List[Subtitle]:
    """
    Load SRT subtitle file.

    Args:
        srt_path: Path to .srt file

    Returns:
        List of Subtitle objects
    """
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    # First, try to read the file content directly to check accessibility
    # This handles permission issues, file locks, and network path problems
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except PermissionError as e:
        print(f"[ERROR] Cannot access SRT file (permission denied or file locked): {srt_path}")
        print(f"        Try closing any applications that may have the file open.")
        return []
    except OSError as e:
        print(f"[ERROR] OS error reading SRT file: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to read SRT file: {e}")
        return []

    # Now try pysrt with the content we already read
    try:
        import pysrt
        subs = pysrt.from_string(content)
        return [
            Subtitle(
                index=sub.index,
                start_sec=_srt_time_to_seconds(sub.start),
                end_sec=_srt_time_to_seconds(sub.end),
                text=sub.text
            )
            for sub in subs
        ]
    except Exception as e:
        print(f"Warning: Failed to parse SRT with pysrt: {e}")
        return _parse_srt_fallback_from_content(content)


def _srt_time_to_seconds(srt_time) -> float:
    """Convert pysrt time to seconds."""
    return (srt_time.hours * 3600 +
            srt_time.minutes * 60 +
            srt_time.seconds +
            srt_time.milliseconds / 1000.0)


def _parse_srt_fallback_from_content(content: str) -> List[Subtitle]:
    """Fallback SRT parser using already-loaded content."""
    import re

    subtitles = []

    # Split by double newline (subtitle blocks)
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])

            # Parse timestamp line
            time_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                lines[1]
            )
            if not time_match:
                continue

            start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])

            start_sec = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
            end_sec = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000

            text = '\n'.join(lines[2:])

            subtitles.append(Subtitle(
                index=index,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text
            ))
        except (ValueError, IndexError):
            continue

    return subtitles


def parse_scenes_from_metadata(metadata: Dict) -> List[SceneInfo]:
    """
    Extract scene detection info from metadata.

    Args:
        metadata: Parsed master metadata JSON

    Returns:
        List of SceneInfo objects
    """
    scenes = []
    scenes_data = metadata.get('scenes_detected', [])

    for scene_dict in scenes_data:
        scenes.append(SceneInfo(
            scene_index=scene_dict.get('scene_index', 0),
            start_time_seconds=scene_dict.get('start_time_seconds', 0.0),
            end_time_seconds=scene_dict.get('end_time_seconds', 0.0),
            duration_seconds=scene_dict.get('duration_seconds', 0.0),
            detection_pass=scene_dict.get('detection_pass'),
            filename=scene_dict.get('filename')
        ))

    return scenes


def parse_coarse_boundaries_from_metadata(metadata: Dict) -> List[SceneInfo]:
    """
    Extract coarse boundary info (Pass 1) from metadata.

    Args:
        metadata: Parsed master metadata JSON

    Returns:
        List of SceneInfo objects representing coarse Pass 1 boundaries
    """
    boundaries = []
    coarse_data = metadata.get('coarse_boundaries', [])

    for boundary_dict in coarse_data:
        boundaries.append(SceneInfo(
            scene_index=boundary_dict.get('scene_index', 0),
            start_time_seconds=boundary_dict.get('start_time_seconds', 0.0),
            end_time_seconds=boundary_dict.get('end_time_seconds', 0.0),
            duration_seconds=boundary_dict.get('duration_seconds', 0.0),
            detection_pass=1,  # Always Pass 1 for coarse boundaries
            filename=None
        ))

    return boundaries


def parse_vad_from_metadata(metadata: Dict) -> tuple:
    """
    Extract VAD segment info from metadata.

    Args:
        metadata: Parsed master metadata JSON

    Returns:
        Tuple of (List[VadSegment], vad_method, vad_params)
    """
    segments = []
    vad_data = metadata.get('vad_segments', [])
    vad_method = metadata.get('vad_method')
    vad_params = metadata.get('vad_params')

    if vad_data:
        for seg in vad_data:
            segments.append(VadSegment(
                start_sec=seg.get('start_sec', 0.0),
                end_sec=seg.get('end_sec', 0.0)
            ))

    return segments, vad_method, vad_params


def load_all_data(
    audio_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    srt_path: Optional[Path] = None
) -> VisualizationData:
    """
    Load all available data for visualization.

    Args:
        audio_path: Path to extracted WAV file
        metadata_path: Path to master metadata JSON
        srt_path: Path to output SRT file

    Returns:
        VisualizationData container with all loaded data
    """
    data = VisualizationData()

    # Load audio waveform
    if audio_path and audio_path.exists():
        try:
            data.waveform, data.sample_rate = load_audio(audio_path)
            data.duration_seconds = len(data.waveform) / data.sample_rate
            print(f"Loaded audio: {data.duration_seconds:.1f}s @ {data.sample_rate}Hz")
        except Exception as e:
            print(f"Warning: Failed to load audio: {e}")

    # Load metadata (scenes, VAD, coarse boundaries)
    if metadata_path and metadata_path.exists():
        try:
            data.metadata_raw = load_metadata(metadata_path)
            data.scenes = parse_scenes_from_metadata(data.metadata_raw)
            data.coarse_boundaries = parse_coarse_boundaries_from_metadata(data.metadata_raw)
            data.vad_segments, data.vad_method, data.vad_params = parse_vad_from_metadata(data.metadata_raw)
            print(f"Loaded metadata: {len(data.scenes)} scenes, {len(data.coarse_boundaries)} coarse boundaries, {len(data.vad_segments)} VAD segments")

            # Get duration from metadata if not from audio
            if data.duration_seconds == 0:
                input_info = data.metadata_raw.get('input_info', {})
                data.duration_seconds = input_info.get('audio_duration_seconds', 0.0)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")

    # Load subtitles
    if srt_path and srt_path.exists():
        try:
            data.subtitles = load_srt(srt_path)
            print(f"Loaded SRT: {len(data.subtitles)} subtitles")
        except Exception as e:
            print(f"Warning: Failed to load SRT: {e}")

    return data
