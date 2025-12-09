"""
Silero VAD speech segmentation backend.

Supports both v4.0 and v3.1 versions of Silero VAD via torch.hub.
"""

from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import logging

import numpy as np
import torch

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


def _is_silero_vad_cached(repo_or_dir: str) -> bool:
    """
    Check if a specific silero-vad version is already cached by torch.hub.

    Args:
        repo_or_dir: Repository string (e.g., "snakers4/silero-vad:v3.1")

    Returns:
        True if the exact version is cached, False otherwise
    """
    try:
        hub_dir = torch.hub.get_dir()

        if ':' in repo_or_dir:
            repo_name, version = repo_or_dir.rsplit(':', 1)
        else:
            repo_name = repo_or_dir
            version = 'master'

        repo_safe = repo_name.replace('/', '_')
        cached_repo_path = Path(hub_dir) / f"{repo_safe}_{version}"

        if cached_repo_path.exists():
            logger.debug(f"Silero VAD {version} found in cache: {cached_repo_path}")
            return True
        else:
            logger.debug(f"Silero VAD {version} NOT in cache, will download")
            return False
    except Exception as e:
        logger.warning(f"Error checking silero-vad cache: {e}")
        return False


class SileroSpeechSegmenter:
    """
    Silero VAD speech segmentation backend.

    Supports v4.0 (default) and v3.1 versions. Provides speech detection
    with configurable sensitivity and segment grouping for ASR processing.

    Example:
        segmenter = SileroSpeechSegmenter(version="v4.0", threshold=0.4)
        result = segmenter.segment(audio_path)
        for segment in result.segments:
            print(f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s")
    """

    # Repository mappings for different versions
    REPOS = {
        "v4.0": "snakers4/silero-vad:v4.0",
        "v3.1": "snakers4/silero-vad:v3.1",
        "latest": "snakers4/silero-vad",
    }

    # Default parameters per version (v4.0 is more sensitive)
    VERSION_DEFAULTS = {
        "v4.0": {
            "threshold": 0.4,
            "min_speech_duration_ms": 150,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 400,
        },
        "v3.1": {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 400,
        },
    }

    def __init__(
        self,
        version: str = "v4.0",
        threshold: Optional[float] = None,
        min_speech_duration_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
        chunk_threshold_s: float = 4.0,
        start_pad_samples: int = 3200,
        end_pad_samples: int = 20800,
        **kwargs
    ):
        """
        Initialize Silero VAD segmenter.

        Args:
            version: Silero VAD version ("v4.0", "v3.1", "latest")
            threshold: Speech probability threshold [0.0, 1.0]
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence between segments
            speech_pad_ms: Padding around detected speech
            chunk_threshold_s: Gap threshold for segment grouping (seconds)
            start_pad_samples: Samples to pad before speech start (at 16kHz)
            end_pad_samples: Samples to pad after speech end (at 16kHz)
            **kwargs: Ignored (for interface compatibility)
        """
        self.version = version if version in self.REPOS else "v4.0"
        self.repo = self.REPOS.get(self.version, self.REPOS["v4.0"])

        # Get version-specific defaults
        defaults = self.VERSION_DEFAULTS.get(self.version, self.VERSION_DEFAULTS["v4.0"])

        # Apply parameters with version-appropriate defaults
        self.threshold = threshold if threshold is not None else defaults["threshold"]
        self.min_speech_duration_ms = (
            min_speech_duration_ms if min_speech_duration_ms is not None
            else defaults["min_speech_duration_ms"]
        )
        self.min_silence_duration_ms = (
            min_silence_duration_ms if min_silence_duration_ms is not None
            else defaults["min_silence_duration_ms"]
        )
        self.speech_pad_ms = (
            speech_pad_ms if speech_pad_ms is not None
            else defaults["speech_pad_ms"]
        )
        self.chunk_threshold_s = chunk_threshold_s

        # Padding in samples (at 16kHz)
        self.start_pad_samples = start_pad_samples
        self.end_pad_samples = end_pad_samples

        # Lazy-loaded model
        self._model = None
        self._utils = None
        self._get_speech_timestamps = None

    @property
    def name(self) -> str:
        return f"silero-{self.version}"

    @property
    def display_name(self) -> str:
        return f"Silero VAD {self.version}"

    def _ensure_model(self) -> None:
        """Load VAD model if not already loaded."""
        if self._model is not None:
            return

        logger.debug(f"Loading Silero VAD model from: {self.repo}")
        try:
            is_cached = _is_silero_vad_cached(self.repo)

            self._model, self._utils = torch.hub.load(
                repo_or_dir=self.repo,
                model="silero_vad",
                force_reload=not is_cached,
                onnx=False
            )
            (self._get_speech_timestamps, _, _, _, _) = self._utils

            status = "from cache" if is_cached else "downloaded"
            logger.debug(f"Silero VAD loaded ({status}) from {self.repo}")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
            raise ImportError(f"Failed to load Silero VAD: {e}")

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio (will resample to 16kHz)
            **kwargs: Override threshold, min_speech_duration_ms, etc.

        Returns:
            SegmentationResult with detected speech segments and groups
        """
        start_time = time.time()
        self._ensure_model()

        # Load and prepare audio
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr

        # Get effective parameters (kwargs can override instance defaults)
        threshold = kwargs.get("threshold", self.threshold)
        min_speech_duration_ms = kwargs.get(
            "min_speech_duration_ms", self.min_speech_duration_ms
        )

        # Resample to 16kHz for VAD
        VAD_SR = 16000
        audio_16k = self._resample_audio(audio_data, actual_sr, VAD_SR)

        # Run VAD
        audio_tensor = torch.FloatTensor(audio_16k)

        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=VAD_SR,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms
        )

        if not speech_timestamps:
            logger.debug("No speech detected by Silero VAD")
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        # Apply padding adjustments (preserved from original implementation)
        for i in range(len(speech_timestamps)):
            speech_timestamps[i]["start"] = max(
                0, speech_timestamps[i]["start"] - self.start_pad_samples
            )
            speech_timestamps[i]["end"] = min(
                len(audio_tensor) - 16,
                speech_timestamps[i]["end"] + self.end_pad_samples
            )
            # Prevent overlap with previous segment
            if i > 0 and speech_timestamps[i]["start"] < speech_timestamps[i - 1]["end"]:
                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]

        # Convert to SpeechSegments
        segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / VAD_SR
            end_sec = ts["end"] / VAD_SR
            segments.append(SpeechSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                start_sample=ts["start"],
                end_sample=ts["end"],
                confidence=1.0,  # Silero doesn't provide per-segment confidence
                metadata={}
            ))

        # Group segments for ASR processing
        groups = self._group_segments(segments, VAD_SR)

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _group_segments(
        self,
        segments: List[SpeechSegment],
        sample_rate: int
    ) -> List[List[SpeechSegment]]:
        """
        Group segments based on time gaps for efficient ASR processing.

        Segments with gaps larger than chunk_threshold_s are split into
        separate groups.
        """
        if not segments:
            return []

        groups: List[List[SpeechSegment]] = [[]]

        for i, segment in enumerate(segments):
            if i > 0:
                prev_end = segments[i - 1].end_sec
                gap = segment.start_sec - prev_end
                if gap > self.chunk_threshold_s:
                    groups.append([])
            groups[-1].append(segment)

        return groups

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from path or return array directly.

        Args:
            audio: Audio data or path
            sample_rate: Expected sample rate for arrays

        Returns:
            Tuple of (audio_array, actual_sample_rate)
        """
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        # Load from file
        audio_path = Path(audio) if isinstance(audio, str) else audio

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    def _resample_audio(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses simple linear interpolation (same as original implementation).
        """
        if source_sr == target_sr:
            return audio

        resample_ratio = target_sr / source_sr
        resampled_length = int(len(audio) * resample_ratio)
        indices = np.linspace(0, len(audio) - 1, resampled_length).astype(int)
        return audio[indices]

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current segmentation parameters."""
        return {
            "version": self.version,
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "chunk_threshold_s": self.chunk_threshold_s,
            "start_pad_samples": self.start_pad_samples,
            "end_pad_samples": self.end_pad_samples,
        }

    def cleanup(self) -> None:
        """Release GPU memory and model resources."""
        if self._model is not None:
            del self._model
            del self._utils
            del self._get_speech_timestamps
            self._model = None
            self._utils = None
            self._get_speech_timestamps = None

            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug("Silero VAD model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates (internally resamples to 16kHz)."""
        return [8000, 16000, 22050, 44100, 48000]

    def __repr__(self) -> str:
        return (
            f"SileroSpeechSegmenter("
            f"version={self.version!r}, "
            f"threshold={self.threshold}, "
            f"min_speech_duration_ms={self.min_speech_duration_ms})"
        )
