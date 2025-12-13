"""
ClearerVoice speech enhancement backend.

Uses the ClearVoice library (ModelScope/Alibaba) for speech enhancement
including denoising, separation, and super-resolution.

Available models:
- FRCRN_SE_16K: Speech enhancement at 16kHz
- FRCRN_SE_48K: Speech enhancement at 48kHz (default, best quality)
- MossFormer2_SE_48K: MossFormer2 enhancement at 48kHz
- MossFormer2_SS_16K: MossFormer2 separation at 16kHz

Installation: pip install clearvoice
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
)

logger = logging.getLogger("whisperjav")

# Model configurations
_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "FRCRN_SE_16K": {
        "sample_rate": 16000,
        "task": "speech_enhancement",
        "description": "FRCRN Speech Enhancement (16kHz)",
    },
    "FRCRN_SE_48K": {
        "sample_rate": 48000,
        "task": "speech_enhancement",
        "description": "FRCRN Speech Enhancement (48kHz)",
    },
    "MossFormer2_SE_48K": {
        "sample_rate": 48000,
        "task": "speech_enhancement",
        "description": "MossFormer2 Speech Enhancement (48kHz)",
    },
    "MossFormer2_SS_16K": {
        "sample_rate": 16000,
        "task": "speech_separation",
        "description": "MossFormer2 Speech Separation (16kHz)",
    },
}

DEFAULT_MODEL = "FRCRN_SE_48K"


class ClearVoiceSpeechEnhancer:
    """
    ClearerVoice speech enhancement backend.

    Provides noise reduction and speech enhancement using Alibaba's
    ClearVoice models via the clearvoice package.

    Example:
        enhancer = ClearVoiceSpeechEnhancer(model="FRCRN_SE_48K")
        result = enhancer.enhance(audio_array, sample_rate=48000)
        enhanced_audio = result.audio

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
        Initialize ClearVoice enhancer.

        Args:
            model: Model to use (default: FRCRN_SE_48K for best quality)
            device: Device to use ("cuda", "cpu", "auto")
            **kwargs: Additional parameters (ignored)
        """
        self._model_name = model if model in _MODEL_INFO else DEFAULT_MODEL
        self._device = device
        self._clearvoice = None
        self._initialized = False

        # Validate model
        if model not in _MODEL_INFO:
            logger.warning(
                f"Unknown ClearVoice model '{model}', using {DEFAULT_MODEL}. "
                f"Available: {list(_MODEL_INFO.keys())}"
            )

        logger.debug(f"ClearVoiceSpeechEnhancer configured: model={self._model_name}")

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of ClearVoice model.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        try:
            from clearvoice import ClearVoice

            model_info = _MODEL_INFO[self._model_name]
            task = model_info["task"]

            logger.info(f"Loading ClearVoice model: {self._model_name}")

            self._clearvoice = ClearVoice(
                task=task,
                model_names=[self._model_name]
            )

            self._initialized = True
            logger.info(f"ClearVoice model loaded successfully: {self._model_name}")
            return True

        except ImportError as e:
            logger.error(f"ClearVoice not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ClearVoice: {e}")
            return False

    @property
    def name(self) -> str:
        return "clearvoice"

    @property
    def display_name(self) -> str:
        model_info = _MODEL_INFO.get(self._model_name, {})
        sr = model_info.get("sample_rate", 48000) // 1000
        return f"ClearerVoice ({sr}kHz)"

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Enhance audio using ClearVoice.

        Args:
            audio: Audio data as numpy array (float32, mono), or path to file
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters (ignored)

        Returns:
            EnhancementResult with enhanced audio, or original on failure

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
                method=f"clearvoice-{self._model_name}",
                error_message=f"Failed to load audio: {e}",
                processing_time_sec=time.time() - start_time,
            )

        # Initialize model if needed
        if not self._ensure_initialized():
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"clearvoice-{self._model_name}",
                error_message="Failed to initialize ClearVoice model",
                processing_time_sec=time.time() - start_time,
            )

        try:
            # ClearVoice can accept numpy arrays directly
            # But we may need to handle sample rate mismatch
            model_sr = _MODEL_INFO[self._model_name]["sample_rate"]

            # Resample to model's expected rate if needed
            if actual_sr != model_sr:
                from ..base import resample_audio
                audio_for_model = resample_audio(audio_data, actual_sr, model_sr)
                logger.debug(f"Resampled {actual_sr}Hz -> {model_sr}Hz for ClearVoice")
            else:
                audio_for_model = audio_data

            # Process with ClearVoice
            # ClearVoice expects audio in specific format
            enhanced = self._process_audio(audio_for_model, model_sr)

            processing_time = time.time() - start_time

            return EnhancementResult(
                audio=enhanced,
                sample_rate=model_sr,
                method=f"clearvoice-{self._model_name}",
                parameters={
                    "model": self._model_name,
                    "input_sr": actual_sr,
                    "output_sr": model_sr,
                },
                processing_time_sec=processing_time,
                metadata={
                    "input_samples": len(audio_data),
                    "output_samples": len(enhanced),
                },
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning(f"ClearVoice enhancement failed: {e}")
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"clearvoice-{self._model_name}",
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def _process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through ClearVoice model.

        Args:
            audio: Audio data (float32, mono, at model's sample rate)
            sample_rate: Sample rate of audio

        Returns:
            Enhanced audio array
        """
        # ClearVoice API can work with numpy arrays
        # The exact API depends on the clearvoice version
        # We'll try the direct array approach first

        try:
            # Try direct numpy array input
            result = self._clearvoice(audio, sr=sample_rate)

            # Extract audio from result dict
            if isinstance(result, dict):
                # Result format: {'output_wav': array, ...}
                if 'output_wav' in result:
                    return result['output_wav'].astype(np.float32)
                elif self._model_name in result:
                    return result[self._model_name].astype(np.float32)
                else:
                    # Return first array value found
                    for v in result.values():
                        if isinstance(v, np.ndarray):
                            return v.astype(np.float32)

            # If result is already an array
            if isinstance(result, np.ndarray):
                return result.astype(np.float32)

            # Fallback: try via temp file
            return self._process_via_tempfile(audio, sample_rate)

        except Exception as e:
            logger.debug(f"Direct array processing failed, trying temp file: {e}")
            return self._process_via_tempfile(audio, sample_rate)

    def _process_via_tempfile(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio via temporary file (fallback method).

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Enhanced audio array
        """
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            tmp_in_path = Path(tmp_in.name)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = Path(tmp_out.name)

        try:
            # Write input to temp file
            sf.write(str(tmp_in_path), audio, sample_rate)

            # Process with ClearVoice
            result = self._clearvoice(
                str(tmp_in_path),
                output_path=str(tmp_out_path.parent),
            )

            # Read enhanced audio
            if tmp_out_path.exists():
                enhanced, _ = sf.read(str(tmp_out_path), dtype='float32')
                return enhanced

            # If output file not created, check result
            if isinstance(result, dict) and 'output_wav' in result:
                return result['output_wav'].astype(np.float32)

            raise RuntimeError("ClearVoice did not produce output")

        finally:
            # Cleanup temp files
            try:
                tmp_in_path.unlink(missing_ok=True)
                tmp_out_path.unlink(missing_ok=True)
            except Exception:
                pass

    def get_preferred_sample_rate(self) -> int:
        """Return the model's native sample rate."""
        return _MODEL_INFO.get(self._model_name, {}).get("sample_rate", 48000)

    def get_output_sample_rate(self) -> int:
        """Return the model's output sample rate (same as input for ClearVoice)."""
        return self.get_preferred_sample_rate()

    def cleanup(self) -> None:
        """Release model resources."""
        if self._clearvoice is not None:
            try:
                # Try to release GPU memory
                del self._clearvoice
                self._clearvoice = None
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

                logger.debug("ClearVoice resources released")
            except Exception as e:
                logger.warning(f"Error during ClearVoice cleanup: {e}")

    def get_supported_models(self) -> List[str]:
        """Return list of supported model variants."""
        return list(_MODEL_INFO.keys())

    def __repr__(self) -> str:
        return f"ClearVoiceSpeechEnhancer(model={self._model_name})"

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
