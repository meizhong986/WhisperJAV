"""
Null/Passthrough speech enhancer.

Returns the input audio unchanged, effectively bypassing enhancement.
Useful when --speech-enhancer none is specified or enhancement is disabled.
"""

from typing import Union, List, Dict, Any
from pathlib import Path
import time
import numpy as np

from ..base import (
    SpeechEnhancer,
    EnhancementResult,
    load_audio_to_array,
)


class NullSpeechEnhancer:
    """
    Passthrough enhancer that returns audio unchanged.

    Used when speech enhancement should be bypassed:
    - --speech-enhancer none
    - Enhancement feature disabled

    This allows the pipeline to proceed without enhancement overhead.
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
        Return audio unchanged (passthrough).

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio

        Returns:
            EnhancementResult with original audio unchanged
        """
        start_time = time.time()

        # Load audio if path provided
        audio_data, actual_sr = load_audio_to_array(audio, sample_rate)

        return EnhancementResult(
            audio=audio_data,
            sample_rate=actual_sr,
            method=self.name,
            parameters={"mode": "passthrough"},
            processing_time_sec=time.time() - start_time,
            metadata={"bypass": True},
            success=True,
            error_message=None,
        )

    def get_preferred_sample_rate(self) -> int:
        """Return 16kHz (standard for VAD/ASR, no enhancement benefit)."""
        return 16000

    def get_output_sample_rate(self) -> int:
        """Output matches input (passthrough)."""
        return 16000

    def cleanup(self) -> None:
        """No resources to clean up."""
        pass

    def get_supported_models(self) -> List[str]:
        """No models for passthrough."""
        return []

    def __repr__(self) -> str:
        return "NullSpeechEnhancer()"
