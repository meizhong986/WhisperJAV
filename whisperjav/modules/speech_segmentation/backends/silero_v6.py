"""
Silero VAD v6.2 speech segmentation backend.

Uses the silero-vad pip package (>=6.2) which provides:
- max_speech_duration_s: Force-splits long speech chunks at internal silences
- neg_threshold: Hysteresis for stable segmentation in noisy audio
- speech_pad_ms: Internal padding (no manual sample-based padding needed)

These features solve TEN VAD's failure to detect pauses in fast Japanese
dialogue scenes with background audio.

Requires: pip install silero-vad>=6.2
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging

import numpy as np

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


class SileroV6SpeechSegmenter:
    """
    Silero VAD v6.2 speech segmentation backend.

    Uses the silero-vad pip package API (load_silero_vad, get_speech_timestamps)
    with v6.2 features: max_speech_duration_s, neg_threshold, and speech_pad_ms.

    JAV-tuned defaults:
    - threshold=0.35: Captures speech overlapping with breathing/moaning
    - speech_pad_ms=250: Capture Japanese soft onsets and trailing particles
    - min_speech_duration_ms=100: Preserve short utterances (はい, ね, うん)
    - max_speech_duration_s inherits from max_group_duration_s as safety net

    Example:
        segmenter = SileroV6SpeechSegmenter(threshold=0.35, max_group_duration_s=6.0)
        result = segmenter.segment(audio_path)
    """

    def __init__(
        self,
        threshold: float = 0.35,
        neg_threshold: Optional[float] = None,
        min_speech_duration_ms: int = 100,
        max_speech_duration_s: Optional[float] = None,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 250,
        min_silence_at_max_speech: int = 98,
        use_max_poss_sil_at_max_speech: bool = True,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize Silero VAD v6.2 segmenter.

        Args:
            threshold: Speech probability threshold [0.0, 1.0]. Higher = more
                selective. Default 0.35 (JAV-tuned: captures speech overlapping
                with breathing/moaning; v6.2 library default is 0.5).
            neg_threshold: Hysteresis threshold for speech end detection.
                None = auto-calculated as threshold - 0.15 by v6.2.
                Provides stable segmentation in noisy audio.
            min_speech_duration_ms: Minimum speech segment duration.
                Default 100ms to preserve short Japanese utterances.
            max_speech_duration_s: Force-split long speech at internal silences.
                None = inherits from max_group_duration_s (key safety net).
            min_silence_duration_ms: Minimum silence gap to split segments.
                Default 100ms (v6.2 default, detects short pauses).
            speech_pad_ms: Padding around speech segments (handled internally
                by v6.2, no manual sample-based padding needed).
                Default 250ms to capture Japanese soft onsets and trailing
                particles/breath tails that get clipped at 200ms.
            min_silence_at_max_speech: Minimum silence duration (ms) used when
                splitting at max_speech_duration_s. Default 98 (v6.2 default).
            use_max_poss_sil_at_max_speech: When splitting at max_speech_duration_s,
                use the longest possible silence. Default True (v6.2 default).
            chunk_threshold_s: Gap threshold for post-VAD segment grouping (seconds).
            max_group_duration_s: Maximum duration for a segment group (seconds).
                Default 29s to stay within Whisper's 30s context window.
                Pipeline typically overrides to 6.0 for Qwen.
            **kwargs: Absorbs factory-injected parameters (e.g., version) harmlessly.
        """
        self.threshold = float(threshold)
        self.neg_threshold = float(neg_threshold) if neg_threshold is not None else None
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.min_silence_at_max_speech = int(min_silence_at_max_speech)
        self.use_max_poss_sil_at_max_speech = bool(use_max_poss_sil_at_max_speech)

        # Grouping parameters
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = float(chunk_threshold_s)
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = float(kwargs["chunk_threshold"])
        else:
            self.chunk_threshold_s = 1.0

        self.max_group_duration_s = float(max_group_duration_s) if max_group_duration_s is not None else 29.0

        # max_speech_duration_s: inherit from max_group_duration_s if not explicit
        if max_speech_duration_s is not None:
            self.max_speech_duration_s = float(max_speech_duration_s)
        else:
            self.max_speech_duration_s = self.max_group_duration_s

        # Lazy-loaded model
        self._model = None

    @property
    def name(self) -> str:
        return "silero-v6.2"

    @property
    def display_name(self) -> str:
        return "Silero VAD v6.2"

    def _ensure_model(self) -> None:
        """Load Silero VAD model if not already loaded."""
        if self._model is not None:
            return

        try:
            from silero_vad import load_silero_vad
        except ImportError:
            raise ImportError(
                "Silero VAD v6.2 requires silero-vad>=6.2 package. Install with:\n"
                "pip install silero-vad>=6.2"
            )

        logger.debug(
            f"Loading Silero VAD v6.2 model "
            f"(threshold={self.threshold}, max_speech_duration_s={self.max_speech_duration_s})"
        )
        try:
            self._model = load_silero_vad()
            logger.debug("Silero VAD v6.2 model loaded")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD v6.2 model: {e}", exc_info=True)
            raise

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs,
    ) -> SegmentationResult:
        """
        Detect speech segments using Silero VAD v6.2.

        Args:
            audio: Audio data as numpy array, or path to audio file.
            sample_rate: Sample rate of input audio.
            **kwargs: Override parameters.

        Returns:
            SegmentationResult with detected speech segments.
        """
        import torch

        start_time = time.time()
        self._ensure_model()

        # Load and prepare audio
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr

        # Silero VAD v6.2 requires 16kHz
        if actual_sr != 16000:
            audio_data = self._resample_audio(audio_data, actual_sr, 16000)
            actual_sr = 16000

        # Convert to torch tensor (float32, range [-1, 1])
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_tensor = torch.from_numpy(audio_data)

        try:
            from silero_vad import get_speech_timestamps

            # Build kwargs for get_speech_timestamps
            vad_kwargs = {
                "threshold": self.threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "max_speech_duration_s": self.max_speech_duration_s,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "speech_pad_ms": self.speech_pad_ms,
                "return_seconds": False,  # Get samples for consistent SpeechSegment construction
                "min_silence_at_max_speech": self.min_silence_at_max_speech,
                "use_max_poss_sil_at_max_speech": self.use_max_poss_sil_at_max_speech,
            }

            # Only pass neg_threshold when explicitly set (None = v6.2 auto-calculates)
            if self.neg_threshold is not None:
                vad_kwargs["neg_threshold"] = self.neg_threshold

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=actual_sr,
                **vad_kwargs,
            )

            # Convert to SpeechSegment objects
            segments = []
            for ts in speech_timestamps:
                start_sample = ts["start"]
                end_sample = ts["end"]
                start_sec = start_sample / actual_sr
                end_sec = end_sample / actual_sr

                segments.append(SpeechSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    confidence=1.0,  # Silero VAD doesn't expose per-segment confidence
                ))

        except Exception as e:
            logger.error(f"Silero VAD v6.2 segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        # Group segments using shared grouping logic
        groups = self._group_segments(segments)

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(audio.dtype)
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), 1 / ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    def _group_segments(
        self,
        segments: List[SpeechSegment],
    ) -> List[List[SpeechSegment]]:
        """Group segments based on time gaps.

        Delegates to the standalone group_segments() function from ten.py.
        """
        from .ten import group_segments
        return group_segments(segments, self.max_group_duration_s, self.chunk_threshold_s)

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            "threshold": self.threshold,
            "neg_threshold": self.neg_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "min_silence_at_max_speech": self.min_silence_at_max_speech,
            "use_max_poss_sil_at_max_speech": self.use_max_poss_sil_at_max_speech,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("Silero VAD v6.2 model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates.

        Silero VAD v6.2 operates at 16kHz internally. Other sample rates
        will be automatically resampled to 16kHz before processing.
        """
        return [16000]

    def __repr__(self) -> str:
        return (
            f"SileroV6SpeechSegmenter("
            f"threshold={self.threshold}, "
            f"max_speech_duration_s={self.max_speech_duration_s})"
        )
