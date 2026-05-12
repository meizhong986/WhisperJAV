#!/usr/bin/env python3
"""
PyWhisperCpp ASR wrapper for WhisperJAV.

This module provides a whisper.cpp-based ASR backend with native Metal GPU acceleration
on Apple Silicon. It uses pywhispercpp bindings and is designed to integrate with
WhisperJAV's existing pipeline architecture.

Key advantages over faster-whisper on Mac:
- Native Metal GPU support (faster-whisper/ctranslate2 only supports CUDA)
- Automatic GPU detection and utilization
- Lower memory footprint with quantized models

Model management:
- Models are stored in ~/.cache/whisper (same as other Whisper backends)
- Automatic download from HuggingFace if model not found
- Supports ggml quantized models (q5_0, q8_0 variants)
"""

from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import numpy as np
import soundfile as sf
import logging
import os

from whisperjav.utils.logger import logger

# Default model cache directory (shared with other Whisper backends)
WHISPER_CACHE_DIR = Path.home() / ".cache" / "whisper"

# Model name mapping: WhisperJAV convention -> ggml model names
MODEL_NAME_MAP = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo": "large-v3-turbo",  # WhisperJAV 'turbo' -> ggml 'large-v3-turbo'
}

# Default quantized variants for better performance
DEFAULT_QUANTIZED = {
    "large-v2": "large-v2-q5_0",
    "large-v3": "large-v3-q5_0",
    "large-v3-turbo": "large-v3-turbo-q5_0",
}


class PyWhisperCppASR:
    """
    Whisper.cpp ASR backend using pywhispercpp bindings.

    Provides Metal GPU acceleration on Apple Silicon and CPU fallback for other platforms.
    Output format is compatible with WhisperJAV's existing ASR modules.
    """

    def __init__(
        self,
        model_config: Dict,
        params: Dict,
        task: str = "transcribe",
        tracer=None,
    ):
        """
        Initialize PyWhisperCpp ASR with WhisperJAV config structure.

        Args:
            model_config: The 'model' section from resolved config
                - model_name: Whisper model name (tiny, base, small, medium, large-v2, large-v3, turbo)
                - device: Ignored (Metal auto-detected on Mac, CPU elsewhere)
            params: The 'params' section containing decoder, vad, provider settings
                - decoder: language, beam_size, temperature, etc.
            task: ASR task ('transcribe' or 'translate')
            tracer: Optional parameter tracer (for diagnostic logging)
        """
        self.tracer = tracer
        self.task = task

        # Extract model configuration
        model_name = model_config.get("model_name", "large-v2")

        # Map WhisperJAV model name to ggml format
        ggml_model = MODEL_NAME_MAP.get(model_name, model_name)

        # Use quantized variant by default for large models (better memory/performance)
        if ggml_model in DEFAULT_QUANTIZED:
            ggml_model = DEFAULT_QUANTIZED[ggml_model]
            logger.info(f"Using quantized model variant: {ggml_model}")

        self.model_name = ggml_model
        self.original_model_name = model_name

        # Model path resolution
        self.models_dir = WHISPER_CACHE_DIR
        self._model_path = None
        self._model = None

        # Extract decoder parameters - handle both 'decoder' (legacy) and 'asr' (pywhispercpp component) structures
        decoder_params = params.get("decoder", params.get("asr", {}))
        self.language = decoder_params.get("language", "ja")
        self.beam_size = decoder_params.get("beam_size", 5)
        self.temperature = decoder_params.get("temperature", 0.0)
        self.max_len = decoder_params.get("max_len", 0)
        self.max_tokens = decoder_params.get("max_tokens", 0)

        # Whisper.cpp-specific params
        self.n_threads = decoder_params.get("n_threads", 4)

        # Suppress whisper.cpp internal logging
        self.suppress_whispercpp_logs = True

        # Store full config for metadata
        self.model_config = model_config
        self.params = params

        # Lazy loading - model initialized on first use
        logger.info(
            f"PyWhisperCppASR configured: model={self.model_name}, "
            f"language={self.language}, beam_size={self.beam_size}, task={task}"
        )

    def _get_model_path(self) -> str:
        """
        Resolve model path, downloading if necessary.

        Models are stored in ~/.cache/whisper following WhisperJAV convention.

        Returns:
            Absolute path to the ggml model file
        """
        if self._model_path:
            return self._model_path

        # Check if model exists locally
        model_filename = f"ggml-{self.model_name}.bin"
        local_path = self.models_dir / model_filename

        if local_path.exists():
            logger.info(f"Model found in cache: {local_path}")
            self._model_path = str(local_path)
            return self._model_path

        # Download model
        logger.info(f"Downloading model {self.model_name} to {self.models_dir}")
        self._model_path = self._download_model()
        return self._model_path

    def _download_model(self) -> str:
        """
        Download ggml model from HuggingFace.

        Returns:
            Absolute path to downloaded model
        """
        try:
            import pywhispercpp.utils as pwc_utils

            # Ensure cache directory exists
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Use pywhispercpp's download function with custom directory
            model_path = pwc_utils.download_model(
                model_name=self.model_name,
                download_dir=str(self.models_dir),
            )

            logger.info(f"Model downloaded to: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download model {self.model_name}: {e}")
            raise RuntimeError(
                f"Could not download whisper.cpp model '{self.model_name}'. "
                f"Check your internet connection or manually place the model in "
                f"{self.models_dir / f'ggml-{self.model_name}.bin'}"
            )

    def _ensure_model(self) -> None:
        """
        Lazy-load the whisper.cpp model.

        Metal GPU is automatically used on Apple Silicon.
        """
        if self._model is not None:
            return

        try:
            from pywhispercpp.model import Model

            model_path = self._get_model_path()

            # Prepare whisper.cpp parameters
            wcpp_params = self._prepare_whispercpp_params()

            # Initialize model (Metal auto-detected on Mac)
            logger.info(f"Initializing whisper.cpp model from: {model_path}")

            # Suppress whisper.cpp verbose output
            redirect_target = None if self.suppress_whispercpp_logs else False

            self._model = Model(
                model=model_path,
                models_dir=str(self.models_dir),
                redirect_whispercpp_logs_to=redirect_target,
                **wcpp_params,
            )

            # Log device info
            self._log_device_info()

        except ImportError:
            raise ImportError(
                "pywhispercpp is not installed. Install with: pip install pywhispercpp[coreml]"
            )
        except Exception as e:
            logger.error(f"Failed to initialize whisper.cpp model: {e}")
            raise

    def _prepare_whispercpp_params(self) -> Dict:
        """
        Map WhisperJAV config parameters to whisper.cpp format.

        Returns:
            Dict of whisper.cpp compatible parameters
        """
        params = {
            "language": self.language if self.language != "auto" else "",
            "translate": self.task == "translate",
            "n_threads": self.n_threads,
            "print_progress": False,  # Suppress internal progress
            "print_timestamps": False,
            "print_realtime": False,
            "suppress_blank": True,
            # Note: suppress_non_speech_tokens is NOT supported by whisper.cpp C++ API
        }

        # Beam search configuration
        if self.beam_size > 1:
            params["params_sampling_strategy"] = 1  # BEAM_SEARCH
            params["beam_search"] = {
                "beam_size": self.beam_size,
                "patience": -1.0,  # Use default
            }
        else:
            params["params_sampling_strategy"] = 0  # GREEDY
            params["greedy"] = {"best_of": -1}

        # Temperature
        if self.temperature > 0:
            params["temperature"] = self.temperature

        # Length limits
        if self.max_len > 0:
            params["max_len"] = self.max_len
            params["token_timestamps"] = True  # Required for max_len

        if self.max_tokens > 0:
            params["max_tokens"] = self.max_tokens

        return params

    def _log_device_info(self) -> None:
        """
        Log whisper.cpp system info (GPU acceleration status).
        """
        try:
            from pywhispercpp.model import Model

            info = Model.system_info()
            logger.info(f"whisper.cpp system: {info}")

            # Check for Metal
            if "Metal" in info and "EMBED_LIBRARY" in info:
                logger.info("Metal GPU acceleration enabled (Apple Silicon)")
            elif "COREML" in info and info.split("COREML")[1].strip().startswith("1"):
                logger.info("CoreML acceleration enabled")
            else:
                logger.info("Using CPU inference")

        except Exception:
            pass

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Optional overrides (task, language)

        Returns:
            Dict with 'segments', 'text', 'language' keys (compatible with other ASR modules)
        """
        self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Handle task override
        runtime_task = kwargs.get("task", self.task)
        runtime_language = kwargs.get("language", self.language)

        logger.info(f"Transcribing: {audio_path.name} (task={runtime_task}, lang={runtime_language})")

        # Update params for this transcription
        transcribe_params = {}
        if runtime_task != self.task:
            transcribe_params["translate"] = runtime_task == "translate"
        if runtime_language != self.language:
            transcribe_params["language"] = runtime_language if runtime_language != "auto" else ""

        # Run transcription with probability extraction for confidence scores
        transcribe_params["extract_probability"] = True
        segments = self._model.transcribe(str(audio_path), **transcribe_params)

        # Convert to WhisperJAV format
        result = self._convert_segments(segments)

        logger.info(
            f"Transcription complete: {len(result['segments'])} segments, "
            f"{len(result['text'])} chars"
        )

        return result

    def _convert_segments(self, segments: List) -> Dict:
        """
        Convert pywhispercpp Segment objects to WhisperJAV format.

        Args:
            segments: List of pywhispercpp.Segment objects

        Returns:
            Dict with 'segments', 'text', 'language'
        """
        converted_segments = []
        full_text = []

        for seg in segments:
            # t0/t1 are in centiseconds (10ms units)
            # Convert to seconds: centiseconds * 10 / 1000 = centiseconds / 100
            start_sec = seg.t0 / 100.0
            end_sec = seg.t1 / 100.0

            converted_segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": seg.text.strip(),
                "avg_logprob": float(seg.probability) if not np.isnan(seg.probability) else None,
            })

            full_text.append(seg.text.strip())

        return {
            "segments": converted_segments,
            "text": " ".join(full_text),
            "language": self.language,
        }

    def transcribe_to_srt(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """
        Transcribe audio and write to SRT file.

        Args:
            audio_path: Path to audio file
            output_path: Path to output SRT file
            **kwargs: Optional overrides

        Returns:
            Path to created SRT file
        """
        result = self.transcribe(audio_path, **kwargs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write SRT using WhisperJAV's standard format
        self._write_srt(result["segments"], output_path)

        logger.info(f"SRT written to: {output_path}")
        return output_path

    def _write_srt(self, segments: List[Dict], output_path: Path) -> None:
        """
        Write segments to SRT file in standard format.

        Args:
            segments: List of segment dicts with 'start', 'end', 'text'
            output_path: Output file path
        """
        import srt
        from datetime import timedelta

        srt_segments = []
        for i, seg in enumerate(segments):
            srt_segments.append(
                srt.Subtitle(
                    index=i + 1,
                    start=timedelta(seconds=seg["start"]),
                    end=timedelta(seconds=seg["end"]),
                    content=seg["text"],
                )
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))

    def cleanup(self) -> None:
        """
        Release model resources.

        Note: pywhispercpp handles cleanup in __del__, but explicit call
        is provided for consistency with other ASR modules.
        """
        if self._model is not None:
            # pywhispercpp's Model.__del__ calls whisper_free
            self._model = None
            logger.debug("PyWhisperCpp model released")

    def get_model_info(self) -> Dict:
        """
        Get model metadata.

        Returns:
            Dict with model information
        """
        return {
            "backend": "pywhispercpp",
            "model_name": self.model_name,
            "original_model_name": self.original_model_name,
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "models_dir": str(self.models_dir),
        }