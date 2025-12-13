"""
Base classes and protocols for speech enhancement.

This module defines the core data structures and interface that all
speech enhancement backends must implement.

Design Decisions (v1.7.3):
- Single backend entry with model parameter (e.g., clearvoice with model selection)
- Graceful degradation: failures log warning and return original audio
- In-memory processing for scene-level chunks (memory-efficient)
- Sample rate negotiation via get_preferred_sample_rate() / get_output_sample_rate()
"""

from typing import Protocol, List, Dict, Any, Optional, Union, runtime_checkable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger("whisperjav")


@dataclass
class EnhancementResult:
    """
    Output from speech enhancement.

    All backends MUST return results using this structure for interoperability.

    Attributes:
        audio: Enhanced audio data (float32, mono)
        sample_rate: Sample rate of output audio
        method: Backend name with model info (e.g., "clearvoice-FRCRN_SE_48K")
        parameters: Parameters used for enhancement
        processing_time_sec: Time taken to process (seconds)
        metadata: Backend-specific metadata (SNR improvement, etc.)
        success: Whether enhancement succeeded (False = original audio returned)
        error_message: Error description if success=False
    """
    audio: np.ndarray
    sample_rate: int
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    processing_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        """Duration of the audio in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.audio) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.audio)

    def to_dict(self) -> Dict[str, Any]:
        """Export result for JSON serialization (excludes audio data)."""
        return {
            "method": self.method,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "num_samples": self.num_samples,
            "processing_time_sec": round(self.processing_time_sec, 3),
            "success": self.success,
            "error_message": self.error_message,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAILED: {self.error_message}"
        return (
            f"EnhancementResult(method={self.method}, "
            f"duration={self.duration_sec:.2f}s, "
            f"sr={self.sample_rate}, {status})"
        )


@runtime_checkable
class SpeechEnhancer(Protocol):
    """
    Protocol defining the speech enhancement interface.

    All speech enhancement backends MUST implement this protocol.

    Design Contract:
    - enhance() MUST return EnhancementResult even on failure
    - On failure, return original audio with success=False
    - cleanup() MUST be safe to call multiple times
    - get_preferred_sample_rate() returns optimal input rate
    - get_output_sample_rate() returns rate of enhanced audio
    """

    @property
    def name(self) -> str:
        """
        Unique backend identifier.

        Examples: "clearvoice", "bs-roformer", "none"
        """
        ...

    @property
    def display_name(self) -> str:
        """
        Human-readable name for GUI display.

        Examples: "ClearerVoice (48kHz)", "BS-RoFormer Vocal Isolation"
        """
        ...

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Enhance audio quality.

        Args:
            audio: Audio data as numpy array (float32, mono), or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Backend-specific parameters

        Returns:
            EnhancementResult with enhanced audio (or original on failure)

        Note:
            On failure, implementations MUST:
            1. Log a warning with details
            2. Return EnhancementResult with success=False
            3. Return original audio unchanged
            This ensures graceful degradation.
        """
        ...

    def get_preferred_sample_rate(self) -> int:
        """
        Optimal input sample rate for this backend.

        The pipeline should extract audio at this rate when this
        enhancer is selected. Common values: 16000, 44100, 48000.

        Returns:
            Preferred sample rate in Hz
        """
        ...

    def get_output_sample_rate(self) -> int:
        """
        Sample rate of output audio.

        Usually matches preferred input rate, but may differ
        for super-resolution models.

        Returns:
            Output sample rate in Hz
        """
        ...

    def cleanup(self) -> None:
        """
        Release resources (GPU memory, model handles).

        Should be called when the enhancer is no longer needed.
        Must be safe to call multiple times (idempotent).
        """
        ...

    def get_supported_models(self) -> List[str]:
        """
        Return list of supported model variants.

        Examples for ClearerVoice: ["FRCRN_SE_16K", "FRCRN_SE_48K", "MossFormer2_SE_48K"]

        Returns:
            List of model identifiers
        """
        ...


def load_audio_to_array(
    audio: Union[np.ndarray, Path, str],
    target_sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Utility function to load audio from various sources.

    Args:
        audio: Audio data as numpy array, or path to audio file
        target_sample_rate: If provided, resample to this rate

    Returns:
        Tuple of (audio_array as float32 mono, sample_rate)

    Raises:
        ImportError: If soundfile not installed
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is unsupported
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")

    if isinstance(audio, np.ndarray):
        # Already an array - assume provided sample rate or default
        audio_data = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        actual_sr = target_sample_rate or 16000

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    # Load from file
    audio_path = Path(audio) if isinstance(audio, str) else audio

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed
    if target_sample_rate and actual_sr != target_sample_rate:
        audio_data = resample_audio(audio_data, actual_sr, target_sample_rate)
        actual_sr = target_sample_rate

    return audio_data, actual_sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    High-quality audio resampling.

    Uses librosa if available, falls back to scipy.

    Args:
        audio: Audio data (float32, mono)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass

    try:
        from scipy import signal
        # Calculate resampling ratio
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return signal.resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        raise ImportError(
            "Either librosa or scipy is required for resampling. "
            "Install with: pip install librosa"
        )


def create_failed_result(
    audio: np.ndarray,
    sample_rate: int,
    method: str,
    error_message: str,
    processing_time_sec: float = 0.0
) -> EnhancementResult:
    """
    Create an EnhancementResult for failed enhancement.

    This is the standard way to handle graceful degradation.
    Returns original audio with failure metadata.

    Args:
        audio: Original audio to return unchanged
        sample_rate: Sample rate of the audio
        method: Backend name for attribution
        error_message: Description of what failed
        processing_time_sec: Time spent before failure

    Returns:
        EnhancementResult with success=False
    """
    logger.warning(f"Speech enhancement failed ({method}): {error_message}")

    return EnhancementResult(
        audio=audio,
        sample_rate=sample_rate,
        method=method,
        parameters={},
        processing_time_sec=processing_time_sec,
        metadata={"fallback": True},
        success=False,
        error_message=error_message,
    )
