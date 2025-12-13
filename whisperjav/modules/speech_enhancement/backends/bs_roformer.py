"""
BS-RoFormer speech enhancement backend.

Uses BS-RoFormer for vocal isolation and music source separation.
Particularly useful for extracting vocals from audio with background music.

Available models:
- vocals: Extract vocals (isolate speech/singing)
- other: Extract non-vocal audio

Installation: pip install bs-roformer-infer

Note: BS-RoFormer is primarily designed for music source separation
at 44.1kHz. For speech-only content, ClearVoice may be more appropriate.
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import time
import tempfile
import numpy as np
import logging

from ..base import (
    EnhancementResult,
    load_audio_to_array,
    create_failed_result,
    resample_audio,
)

logger = logging.getLogger("whisperjav")

# Model configurations
_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "vocals": {
        "sample_rate": 44100,
        "description": "Vocal isolation (extract speech/singing)",
        "stem": "vocals",
    },
    "other": {
        "sample_rate": 44100,
        "description": "Non-vocal audio extraction",
        "stem": "other",
    },
}

DEFAULT_MODEL = "vocals"
DEFAULT_SAMPLE_RATE = 44100


class BSRoformerSpeechEnhancer:
    """
    BS-RoFormer vocal isolation backend.

    Uses BS-RoFormer to separate vocals from background music/noise.
    Best suited for content with music or complex background audio.

    Example:
        enhancer = BSRoformerSpeechEnhancer(model="vocals")
        result = enhancer.enhance(audio_array, sample_rate=44100)
        vocals = result.audio

        # Always cleanup when done
        enhancer.cleanup()
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize BS-RoFormer enhancer.

        Args:
            model: Model/stem to extract ("vocals" or "other")
            device: Device to use ("cuda", "cpu", "auto")
            **kwargs: Additional parameters (ignored)
        """
        self._model_name = model if model in _MODEL_INFO else DEFAULT_MODEL
        self._device = device
        self._separator = None
        self._initialized = False

        if model not in _MODEL_INFO:
            logger.warning(
                f"Unknown BS-RoFormer model '{model}', using {DEFAULT_MODEL}. "
                f"Available: {list(_MODEL_INFO.keys())}"
            )

        logger.debug(f"BSRoformerSpeechEnhancer configured: model={self._model_name}")

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of BS-RoFormer model.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Import bs-roformer
            from bs_roformer import BSRoformer

            logger.info(f"Loading BS-RoFormer model for stem: {self._model_name}")

            # Determine device
            device = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # Initialize separator
            self._separator = BSRoformer(device=device)

            self._initialized = True
            logger.info(f"BS-RoFormer loaded successfully on {device}")
            return True

        except ImportError as e:
            logger.error(f"bs-roformer not installed: {e}")
            logger.error("Install with: pip install bs-roformer-infer")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize BS-RoFormer: {e}")
            return False

    @property
    def name(self) -> str:
        return "bs-roformer"

    @property
    def display_name(self) -> str:
        stem = _MODEL_INFO.get(self._model_name, {}).get("stem", "vocals")
        return f"BS-RoFormer ({stem})"

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Extract vocals using BS-RoFormer.

        Args:
            audio: Audio data as numpy array (float32, mono), or path to file
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters (ignored)

        Returns:
            EnhancementResult with extracted vocals, or original on failure

        Note:
            On failure, returns original audio with success=False
            for graceful degradation.
        """
        start_time = time.time()

        # Load audio if path provided
        try:
            audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
        except Exception as e:
            return create_failed_result(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=sample_rate,
                method=f"bs-roformer-{self._model_name}",
                error_message=f"Failed to load audio: {e}",
                processing_time_sec=time.time() - start_time,
            )

        # Initialize model if needed
        if not self._ensure_initialized():
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"bs-roformer-{self._model_name}",
                error_message="Failed to initialize BS-RoFormer model",
                processing_time_sec=time.time() - start_time,
            )

        try:
            model_sr = DEFAULT_SAMPLE_RATE  # 44100

            # Resample to model's expected rate if needed
            if actual_sr != model_sr:
                audio_for_model = resample_audio(audio_data, actual_sr, model_sr)
                logger.debug(f"Resampled {actual_sr}Hz -> {model_sr}Hz for BS-RoFormer")
            else:
                audio_for_model = audio_data

            # Process with BS-RoFormer
            separated = self._process_audio(audio_for_model, model_sr)

            processing_time = time.time() - start_time

            return EnhancementResult(
                audio=separated,
                sample_rate=model_sr,
                method=f"bs-roformer-{self._model_name}",
                parameters={
                    "model": self._model_name,
                    "stem": _MODEL_INFO[self._model_name]["stem"],
                    "input_sr": actual_sr,
                    "output_sr": model_sr,
                },
                processing_time_sec=processing_time,
                metadata={
                    "input_samples": len(audio_data),
                    "output_samples": len(separated),
                },
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning(f"BS-RoFormer separation failed: {e}")
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"bs-roformer-{self._model_name}",
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def _process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through BS-RoFormer model.

        Args:
            audio: Audio data (float32, mono, at model's sample rate)
            sample_rate: Sample rate of audio

        Returns:
            Separated audio array (vocals or other stem)
        """
        stem = _MODEL_INFO[self._model_name]["stem"]

        # BS-RoFormer expects stereo input, convert mono to stereo
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio], axis=0)  # (2, samples)
        else:
            audio_stereo = audio

        # Ensure shape is (channels, samples)
        if audio_stereo.shape[0] > audio_stereo.shape[1]:
            audio_stereo = audio_stereo.T

        # Process through separator
        # API: separator.separate(audio, sr) -> dict of stems
        result = self._separator.separate(audio_stereo, sr=sample_rate)

        # Extract the desired stem
        if isinstance(result, dict):
            if stem in result:
                separated = result[stem]
            elif "vocals" in result and stem == "vocals":
                separated = result["vocals"]
            else:
                # Take first available
                separated = list(result.values())[0]
        else:
            separated = result

        # Convert to mono if stereo
        if isinstance(separated, np.ndarray):
            if separated.ndim > 1:
                # Average channels for mono
                if separated.shape[0] <= 2:  # (channels, samples)
                    separated = np.mean(separated, axis=0)
                else:  # (samples, channels)
                    separated = np.mean(separated, axis=1)

            return separated.astype(np.float32)

        raise RuntimeError(f"Unexpected result type from BS-RoFormer: {type(result)}")

    def get_preferred_sample_rate(self) -> int:
        """Return 44100Hz (standard for music/BS-RoFormer)."""
        return DEFAULT_SAMPLE_RATE

    def get_output_sample_rate(self) -> int:
        """Return 44100Hz (same as input for BS-RoFormer)."""
        return DEFAULT_SAMPLE_RATE

    def cleanup(self) -> None:
        """Release model resources."""
        if self._separator is not None:
            try:
                del self._separator
                self._separator = None
                self._initialized = False

                # Force garbage collection for GPU memory
                import gc
                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                logger.debug("BS-RoFormer resources released")
            except Exception as e:
                logger.warning(f"Error during BS-RoFormer cleanup: {e}")

    def get_supported_models(self) -> List[str]:
        """Return list of supported model variants."""
        return list(_MODEL_INFO.keys())

    def __repr__(self) -> str:
        return f"BSRoformerSpeechEnhancer(model={self._model_name})"

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
