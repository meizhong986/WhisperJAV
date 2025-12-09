"""
Null/Passthrough speech segmenter.

Returns the entire audio as a single segment, effectively bypassing
speech segmentation. Useful when --no-vad or --speech-segmenter none
is specified.
"""

from typing import Union, List, Dict, Any
from pathlib import Path
import time
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from ..base import SpeechSegment, SegmentationResult


class NullSpeechSegmenter:
    """
    Passthrough segmenter that returns entire audio as a single segment.

    Used when speech segmentation should be bypassed:
    - --speech-segmenter none
    - --no-vad flag

    This allows ASR modules to process the full audio without VAD preprocessing.
    """

    def __init__(self, **kwargs):
        """
        Initialize null segmenter.

        Args:
            **kwargs: Ignored (accepted for interface compatibility)
        """
        # Accept but ignore any parameters for compatibility
        pass

    @property
    def name(self) -> str:
        return "none"

    @property
    def display_name(self) -> str:
        return "No Segmentation"

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Return entire audio as a single segment.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio

        Returns:
            SegmentationResult with single segment spanning full duration
        """
        start_time = time.time()

        # Load audio if path provided
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr
        num_samples = len(audio_data)

        # Create single segment covering entire audio
        segment = SpeechSegment(
            start_sec=0.0,
            end_sec=duration,
            start_sample=0,
            end_sample=num_samples,
            confidence=1.0,
            metadata={"bypass": True}
        )

        return SegmentationResult(
            segments=[segment],
            groups=[[segment]],
            method=self.name,
            audio_duration_sec=duration,
            parameters={"mode": "passthrough"},
            processing_time_sec=time.time() - start_time,
        )

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> tuple:
        """
        Load audio from path or return array directly.

        Args:
            audio: Audio data or path
            sample_rate: Expected sample rate

        Returns:
            Tuple of (audio_array, actual_sample_rate)
        """
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        # Load from file
        audio_path = Path(audio) if isinstance(audio, str) else audio

        if sf is None:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    def cleanup(self) -> None:
        """No resources to clean up."""
        pass

    def get_supported_sample_rates(self) -> List[int]:
        """All sample rates supported (passthrough)."""
        return [8000, 16000, 22050, 44100, 48000]

    def __repr__(self) -> str:
        return "NullSpeechSegmenter()"
