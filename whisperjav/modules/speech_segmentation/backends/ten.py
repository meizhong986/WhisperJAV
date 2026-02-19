"""
TEN Framework VAD speech segmentation backend.

Requires ten-vad package to be installed.
Install: pip install -U git+https://github.com/TEN-framework/ten-vad.git

See: https://github.com/TEN-framework/ten-vad
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging

import numpy as np

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


def group_segments(
    segments: List[SpeechSegment],
    max_group_duration_s: float = 29.0,
    chunk_threshold_s: float = 1.0,
) -> List[List[SpeechSegment]]:
    """Group speech segments by time gaps and max duration.

    Standalone function for re-grouping without re-running VAD.
    Used by adaptive step-down to create Tier 1 (30s) and Tier 2 (8s)
    groups from the same raw segments.

    Args:
        segments: List of SpeechSegment objects from VAD.
        max_group_duration_s: Maximum duration for a group in seconds.
        chunk_threshold_s: Gap threshold for splitting groups in seconds.

    Returns:
        List of groups, where each group is a list of SpeechSegment objects.
    """
    if not segments:
        return []

    groups: List[List[SpeechSegment]] = [[]]

    for i, segment in enumerate(segments):
        if i > 0:
            prev_end = segments[i - 1].end_sec
            gap = segment.start_sec - prev_end

            # Check if adding this segment would exceed max group duration
            would_exceed_max = False
            if groups[-1]:
                group_start = groups[-1][0].start_sec
                potential_duration = segment.end_sec - group_start
                would_exceed_max = potential_duration > max_group_duration_s

            # Start new group if gap too large OR would exceed max duration
            if gap > chunk_threshold_s or would_exceed_max:
                groups.append([])

        groups[-1].append(segment)

    return groups


class TenSpeechSegmenter:
    """
    TEN Framework VAD speech segmentation backend.

    Uses TEN's lightweight VAD model for fast speech detection.
    Processes audio frame-by-frame at 16kHz with configurable hop size.
    Suitable for real-time and streaming applications.

    Note: TEN VAD requires int16 audio input and operates at 16kHz only.

    Example:
        segmenter = TenSpeechSegmenter(threshold=0.20)
        result = segmenter.segment(audio_path)
    """

    def __init__(
        self,
        threshold: float = 0.20,
        hop_size: int = 256,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = None,
        start_pad_ms: int = 0,
        end_pad_ms: int = 200,
        **kwargs
    ):
        """
        Initialize TEN VAD segmenter.

        Args:
            threshold: Speech probability threshold [0.0, 1.0]. Lower values
                detect more speech (more sensitive). Default 0.20.
            hop_size: Frame size in samples (160 or 256 recommended)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence between segments (not implemented)
            chunk_threshold_s: Gap threshold for segment grouping (seconds)
            max_group_duration_s: Maximum duration for a segment group (seconds).
                Groups are split if adding a segment would exceed this limit.
                Default 29s to stay within Whisper's 30s context window.
            start_pad_ms: Milliseconds to pad before segment start (shifts start earlier).
                Default 0: TEN's 16ms frame already captures the onset, and pre-speech
                padding causes proportional timestamp drift in ASR. See #163.
            end_pad_ms: Milliseconds to pad after segment end (shifts end later)
            **kwargs: Additional parameters for backward compatibility
                - chunk_threshold: Legacy alias for chunk_threshold_s
        """
        self.threshold = float(threshold)
        self.hop_size = int(hop_size)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.start_pad_ms = int(start_pad_ms)
        self.end_pad_ms = int(end_pad_ms)

        # Handle backward compatibility: chunk_threshold (old) -> chunk_threshold_s (new)
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = float(chunk_threshold_s)
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = float(kwargs["chunk_threshold"])
        else:
            self.chunk_threshold_s = 1.0  # Default for tighter grouping

        # Maximum group duration - prevents groups from exceeding Whisper's context window
        self.max_group_duration_s = float(max_group_duration_s) if max_group_duration_s is not None else 29.0

        # Lazy-loaded model
        self._model = None

    @property
    def name(self) -> str:
        return "ten"

    @property
    def display_name(self) -> str:
        return "TEN VAD"

    def _ensure_model(self) -> None:
        """Load TEN VAD model if not already loaded."""
        if self._model is not None:
            return

        try:
            from ten_vad import TenVad
        except ImportError:
            raise ImportError(
                "TEN VAD requires ten-vad package. Install with:\n"
                "pip install -U git+https://github.com/TEN-framework/ten-vad.git"
            )

        logger.debug(f"Loading TEN VAD model (hop_size={self.hop_size}, threshold={self.threshold})")
        try:
            self._model = TenVad(hop_size=self.hop_size, threshold=self.threshold)
            logger.debug("TEN VAD model loaded")
        except Exception as e:
            logger.error(f"Failed to load TEN VAD model: {e}", exc_info=True)
            raise

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments using TEN VAD.

        TEN VAD processes audio frame-by-frame and requires:
        - 16kHz sample rate (audio will be resampled if needed)
        - int16 audio format (will be converted from float32)

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Override parameters

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        self._ensure_model()

        # Load and prepare audio
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr

        # TEN VAD requires 16kHz - resample if needed
        if actual_sr != 16000:
            audio_data = self._resample_audio(audio_data, actual_sr, 16000)
            actual_sr = 16000

        # Convert to int16 (TEN VAD requirement)
        audio_int16 = self._convert_to_int16(audio_data)

        try:
            # Process frame-by-frame and collect flags
            flags = []
            probs = []
            for i in range(0, len(audio_int16) - self.hop_size, self.hop_size):
                frame = audio_int16[i:i + self.hop_size]
                self._model.process(frame)
                flags.append(self._model.out_flags.value)
                probs.append(self._model.out_probability.value)

            # Convert frame flags to speech segments (with padding)
            segments = self._flags_to_segments(flags, probs, actual_sr, duration)

        except Exception as e:
            logger.error(f"TEN VAD segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        # Group segments
        groups = self._group_segments(segments)

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _convert_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 for TEN VAD."""
        if audio.dtype == np.int16:
            return audio

        # Normalize and convert
        if audio.dtype in (np.float32, np.float64):
            # Clamp to [-1, 1] and scale to int16 range
            audio_clipped = np.clip(audio, -1.0, 1.0)
            return (audio_clipped * 32767).astype(np.int16)
        else:
            # Assume it's already in int-like range
            return audio.astype(np.int16)

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
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
            indices = np.arange(0, len(audio), 1/ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    def _flags_to_segments(
        self,
        flags: List[int],
        probs: List[float],
        sample_rate: int,
        audio_duration: float
    ) -> List[SpeechSegment]:
        """Convert frame-level speech flags to speech segments with padding.

        Applies start_pad_ms and end_pad_ms to shift segment boundaries:
        - Start is shifted earlier by start_pad_ms
        - End is shifted later by end_pad_ms
        - Overlap between consecutive segments is prevented
        """
        raw_segments = []
        frame_duration = self.hop_size / sample_rate

        in_speech = False
        speech_start = 0.0
        speech_start_idx = 0

        for i, flag in enumerate(flags):
            time_sec = i * frame_duration

            if flag == 1 and not in_speech:
                # Speech started
                in_speech = True
                speech_start = time_sec
                speech_start_idx = i
            elif flag == 0 and in_speech:
                # Speech ended
                in_speech = False
                speech_end = time_sec

                # Calculate average confidence for this segment
                segment_probs = probs[speech_start_idx:i]
                avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 1.0

                # Apply minimum duration filter
                duration_ms = (speech_end - speech_start) * 1000
                if duration_ms >= self.min_speech_duration_ms:
                    raw_segments.append({
                        'start': speech_start,
                        'end': speech_end,
                        'confidence': avg_confidence
                    })

        # Handle speech extending to end of audio
        if in_speech:
            speech_end = len(flags) * frame_duration
            segment_probs = probs[speech_start_idx:]
            avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 1.0

            duration_ms = (speech_end - speech_start) * 1000
            if duration_ms >= self.min_speech_duration_ms:
                raw_segments.append({
                    'start': speech_start,
                    'end': speech_end,
                    'confidence': avg_confidence
                })

        # Apply padding to segments
        start_pad_sec = self.start_pad_ms / 1000.0
        end_pad_sec = self.end_pad_ms / 1000.0

        segments = []
        for i, seg in enumerate(raw_segments):
            # Apply padding: shift start earlier, shift end later
            padded_start = max(0.0, seg['start'] - start_pad_sec)
            padded_end = min(audio_duration, seg['end'] + end_pad_sec)

            # Prevent overlap with previous segment
            if i > 0 and segments:
                prev_end = segments[-1].end_sec
                if padded_start < prev_end:
                    padded_start = prev_end

            # Ensure segment is still valid after padding adjustments
            if padded_end > padded_start:
                segments.append(SpeechSegment(
                    start_sec=padded_start,
                    end_sec=padded_end,
                    start_sample=int(padded_start * sample_rate),
                    end_sample=int(padded_end * sample_rate),
                    confidence=seg['confidence'],
                    metadata={'raw_start': seg['start'], 'raw_end': seg['end']}
                ))

        return segments

    def _group_segments(
        self,
        segments: List[SpeechSegment]
    ) -> List[List[SpeechSegment]]:
        """Group segments based on time gaps.

        Delegates to the standalone group_segments() function.
        """
        return group_segments(segments, self.max_group_duration_s, self.chunk_threshold_s)

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            "threshold": self.threshold,
            "hop_size": self.hop_size,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "start_pad_ms": self.start_pad_ms,
            "end_pad_ms": self.end_pad_ms,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("TEN VAD model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates.

        Note: TEN VAD only operates at 16kHz internally. Other sample rates
        will be automatically resampled to 16kHz before processing.
        """
        return [16000]

    def __repr__(self) -> str:
        return f"TenSpeechSegmenter(threshold={self.threshold})"
