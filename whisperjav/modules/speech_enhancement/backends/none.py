"""
Null/Passthrough speech enhancer.

v1.7.4+ Clean Contract:
- Receives 48kHz audio (SCENE_EXTRACTION_SR)
- Outputs 16kHz audio (TARGET_SAMPLE_RATE)
- Performs resampling only, no audio processing

This ensures consistent sample rate at the enhancement→VAD/ASR boundary
regardless of which enhancer backend is selected.
"""

from typing import Union, List, Dict, Any
from pathlib import Path
import time
import numpy as np

from ..base import (
    SpeechEnhancer,
    EnhancementResult,
    load_audio_to_array,
    resample_audio,
)

# Contract constants
TARGET_SAMPLE_RATE = 16000  # Output for VAD/ASR


class NullSpeechEnhancer:
    """
    Passthrough enhancer with sample rate conversion.

    v1.7.4+ Clean Contract:
    - Receives 48kHz audio (scene files are always 48kHz)
    - Outputs 16kHz audio (VAD/ASR always expects 16kHz)
    - No audio processing, only resampling

    Used when:
    - --speech-enhancer none is specified
    - Any enhancer backend fallback

    This ensures the pipeline always has 16kHz audio for VAD/ASR.
    """

    def __init__(self, **kwargs):
        """
        Initialize null enhancer.

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
        return "No Enhancement"

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Resample audio to 16kHz for VAD/ASR (no processing).

        v1.7.4+ Clean Contract:
        - Input: 48kHz audio (scene file standard)
        - Output: 16kHz audio (VAD/ASR standard)

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio (typically 48000)

        Returns:
            EnhancementResult with audio resampled to 16kHz
        """
        start_time = time.time()

        # Load audio if path provided
        audio_data, actual_sr = load_audio_to_array(audio, sample_rate)

        # Resample to 16kHz for VAD/ASR if needed
        resampled = actual_sr != TARGET_SAMPLE_RATE
        if resampled:
            audio_data = resample_audio(audio_data, actual_sr, TARGET_SAMPLE_RATE)

        return EnhancementResult(
            audio=audio_data,
            sample_rate=TARGET_SAMPLE_RATE,
            method=self.name,
            parameters={"mode": "passthrough", "resampled": resampled},
            processing_time_sec=time.time() - start_time,
            metadata={
                "bypass": True,
                "input_sr": actual_sr,
                "output_sr": TARGET_SAMPLE_RATE,
            },
            success=True,
            error_message=None,
        )

    def get_preferred_sample_rate(self) -> int:
        """Return 48kHz (v1.7.4+ contract: scene files are always 48kHz)."""
        return 48000

    def get_output_sample_rate(self) -> int:
        """Return 16kHz (v1.7.4+ contract: output is always 16kHz for VAD/ASR)."""
        return TARGET_SAMPLE_RATE

    def cleanup(self) -> None:
        """No resources to clean up."""
        pass

    def get_supported_models(self) -> List[str]:
        """No models for passthrough."""
        return []

    def __repr__(self) -> str:
        return "NullSpeechEnhancer()"
