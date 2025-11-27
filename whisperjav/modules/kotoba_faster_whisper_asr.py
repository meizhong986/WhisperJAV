#!/usr/bin/env python3
"""
Kotoba Faster-Whisper ASR Module.

Japanese-optimized Whisper transcription using kotoba-tech/kotoba-whisper-v2.0-faster
model with faster-whisper backend and internal VAD support.

Based on reference implementation with the following key features:
- Uses faster-whisper's built-in VAD (vad_filter parameter)
- Optimized default parameters for Japanese speech
- Configurable VAD on/off via vad_filter flag
"""

import gc
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Union

import srt
import torch
from faster_whisper import WhisperModel

from whisperjav.utils.logger import logger
from whisperjav.utils.device_detector import get_best_device


class KotobaFasterWhisperASR:
    """
    Kotoba Faster-Whisper ASR with internal VAD support.

    Uses the kotoba-tech/kotoba-whisper-v2.0-faster model optimized for Japanese
    speech recognition. VAD is handled internally by faster-whisper's built-in
    Silero VAD, controllable via the vad_filter parameter.
    """

    # Default model identifier
    DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"

    def __init__(self, model_config: Dict[str, Any], params: Dict[str, Any], task: str):
        """
        Initialize Kotoba Faster-Whisper ASR.

        Args:
            model_config: Model configuration dict with:
                - model_name: Model identifier (default: kotoba-tech/kotoba-whisper-v2.0-faster)
                - device: Device to use (cuda, cpu, auto)
                - compute_type: Compute precision (float16, int8, float32)
            params: ASR parameters dict (from resolved_config["params"]["asr"])
            task: Task type ("transcribe" or "translate")
        """
        # Model configuration
        self.model_name = model_config.get("model_name", self.DEFAULT_MODEL)
        self.device = model_config.get("device", get_best_device())
        self.compute_type = model_config.get("compute_type", "float16")

        # Task
        self.task = task

        # M7: Log warning for missing key params to aid debugging
        self._warn_missing_params(params)

        # VAD control (KEY FEATURE)
        self.vad_filter = params.get("vad_filter", True)
        self.vad_parameters = self._build_vad_parameters(params) if self.vad_filter else None

        # Transcription parameters
        self.language = params.get("language", "ja")
        self.beam_size = params.get("beam_size", 3)
        self.best_of = params.get("best_of", 3)
        self.patience = params.get("patience", 2.0)
        self.temperature = params.get("temperature", [0.0, 0.3])
        self.compression_ratio_threshold = params.get("compression_ratio_threshold", 2.4)
        self.logprob_threshold = params.get("logprob_threshold", -1.5)
        self.no_speech_threshold = params.get("no_speech_threshold", 0.34)
        self.condition_on_previous_text = params.get("condition_on_previous_text", True)
        self.initial_prompt = params.get("initial_prompt", None)
        self.word_timestamps = params.get("word_timestamps", False)
        self.suppress_tokens = params.get("suppress_tokens", None)
        self.suppress_blank = params.get("suppress_blank", True)
        self.without_timestamps = params.get("without_timestamps", False)
        self.repetition_penalty = params.get("repetition_penalty", 1.0)
        self.no_repeat_ngram_size = params.get("no_repeat_ngram_size", 0)
        self.log_progress = params.get("log_progress", False)

        # Initialize model
        self._initialize_model()
        self._log_initialization()

    def _build_vad_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build VAD parameters dict for faster-whisper."""
        return {
            "threshold": params.get("vad_threshold", 0.01),
            "min_speech_duration_ms": params.get("min_speech_duration_ms", 90),
            "max_speech_duration_s": params.get("max_speech_duration_s", 28.0),
            "min_silence_duration_ms": params.get("min_silence_duration_ms", 150),
            "speech_pad_ms": params.get("speech_pad_ms", 400),
        }

    def _warn_missing_params(self, params: Dict[str, Any]) -> None:
        """Log debug warnings for missing key ASR parameters to aid config debugging."""
        # Only warn if debug logging is enabled to avoid noise
        if not logger.isEnabledFor(logging.DEBUG):
            return

        # Key params that affect quality if missing (using defaults)
        key_params = [
            "beam_size", "best_of", "patience", "temperature",
            "compression_ratio_threshold", "logprob_threshold", "no_speech_threshold",
            "vad_filter", "vad_threshold"
        ]

        missing = [p for p in key_params if p not in params]
        if missing:
            logger.debug(
                f"KotobaFasterWhisperASR: Using defaults for missing params: {missing}. "
                "This is normal if using preset defaults."
            )

    def _initialize_model(self) -> None:
        """Initialize the faster-whisper model."""
        logger.info(f"Loading Kotoba model: {self.model_name}")
        logger.info(f"Device: {self.device}, Compute type: {self.compute_type}")

        # Determine compute type based on device capabilities
        # On CPU, float16 is generally not supported by CTranslate2, so we fallback to int8
        # unless float32 is explicitly requested.
        if self.device == "cuda":
            actual_compute_type = self.compute_type
        else:
            if self.compute_type == "float16":
                logger.warning("float16 compute type is not supported on CPU. Falling back to int8.")
                actual_compute_type = "int8"
            else:
                actual_compute_type = self.compute_type

        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=actual_compute_type,
                download_root=None,  # Use default HuggingFace cache
                cpu_threads=0,  # Use default
                num_workers=1,
            )
            logger.info("Kotoba model loaded successfully")
        except Exception as e:
            # C4: VRAM exhaustion fallback - try int8 if float16 fails on CUDA
            if self.device == "cuda" and actual_compute_type == "float16":
                vram_error_indicators = [
                    "out of memory",
                    "cuda out of memory",
                    "outofmemoryerror",
                    "cuda error",
                    "cublas",
                    "insufficient memory",
                ]
                error_str = str(e).lower()
                if any(indicator in error_str for indicator in vram_error_indicators):
                    logger.warning(
                        f"CUDA float16 failed (likely VRAM exhaustion): {e}. "
                        "Falling back to int8 compute type."
                    )
                    try:
                        # Clear CUDA cache before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                        self.model = WhisperModel(
                            model_size_or_path=self.model_name,
                            device=self.device,
                            compute_type="int8",
                            download_root=None,
                            cpu_threads=0,
                            num_workers=1,
                        )
                        self.compute_type = "int8"  # Update for logging/metadata
                        logger.info("Kotoba model loaded successfully with int8 fallback")
                        return
                    except Exception as fallback_e:
                        logger.error(f"int8 fallback also failed: {fallback_e}")
                        raise fallback_e

            logger.error(f"Failed to load Kotoba model: {e}")
            raise

    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- KotobaFasterWhisperASR Initialized ---")
            logger.debug(f"  Model: {self.model_name}")
            logger.debug(f"  Device: {self.device}")
            logger.debug(f"  Compute type: {self.compute_type}")
            logger.debug(f"  Task: {self.task}")
            logger.debug(f"  Language: {self.language}")
            logger.debug(f"  VAD Filter: {self.vad_filter}")
            if self.vad_filter and self.vad_parameters:
                logger.debug(f"  VAD Threshold: {self.vad_parameters.get('threshold')}")
                logger.debug(f"  Min Speech Duration: {self.vad_parameters.get('min_speech_duration_ms')}ms")
            logger.debug(f"  Beam Size: {self.beam_size}")
            logger.debug(f"  Best Of: {self.best_of}")
            logger.debug(f"  Temperature: {self.temperature}")
            logger.debug("------------------------------------------")

    def _prepare_transcribe_params(self) -> Dict[str, Any]:
        """
        Prepare parameters for model.transcribe() call.

        Parameter Name Mapping:
            Config uses 'logprob_threshold' (consistent with other components),
            but faster-whisper API expects 'log_prob_threshold' (with underscore).
            This method performs the conversion automatically.
        """
        params = {
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            # Note: Config uses 'logprob_threshold', API uses 'log_prob_threshold'
            "log_prob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "log_progress": self.log_progress,
        }

        # Add VAD parameters if VAD is enabled
        if self.vad_filter and self.vad_parameters:
            params["vad_parameters"] = self.vad_parameters

        # Optional parameters
        if self.initial_prompt is not None:
            params["initial_prompt"] = self.initial_prompt
        if self.suppress_tokens is not None:
            params["suppress_tokens"] = self.suppress_tokens
        if self.suppress_blank is not None:
            params["suppress_blank"] = self.suppress_blank
        if self.without_timestamps:
            params["without_timestamps"] = self.without_timestamps
        if self.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size > 0:
            params["no_repeat_ngram_size"] = int(self.no_repeat_ngram_size)

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters to override defaults

        Returns:
            Dict with keys:
                - segments: List of segment dicts with start, end, text
                - text: Full transcription text
                - language: Detected/specified language
        """
        audio_path = Path(audio_path)
        logger.debug(f"Transcribing: {audio_path.name}")

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return {"segments": [], "text": "", "language": self.language}

        # Prepare parameters
        params = self._prepare_transcribe_params()
        params.update(kwargs)  # Allow overrides

        logger.debug(f"VAD filter: {params.get('vad_filter', False)}")
        if params.get('vad_filter') and params.get('vad_parameters'):
            logger.debug(f"VAD parameters: {params.get('vad_parameters')}")

        try:
            # Transcribe
            segments_generator, info = self.model.transcribe(
                str(audio_path),
                **params
            )

            logger.debug(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            logger.debug(f"Audio duration: {info.duration:.2f}s")

            # Collect segments
            segments = []
            for segment in segments_generator:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                })
                logger.debug(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")

            full_text = " ".join(seg["text"] for seg in segments if seg["text"])
            logger.info(f"Transcription complete: {len(segments)} segments")

            return {
                "segments": segments,
                "text": full_text,
                "language": info.language,
                "duration": info.duration,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {"segments": [], "text": "", "language": self.language}

    def transcribe_to_srt(
        self,
        audio_path: Union[str, Path],
        output_srt_path: Union[str, Path],
        **kwargs
    ) -> Path:
        """
        Transcribe audio and save as SRT file.

        Args:
            audio_path: Path to audio file
            output_srt_path: Path for output SRT file
            **kwargs: Additional transcription parameters

        Returns:
            Path to created SRT file
        """
        audio_path = Path(audio_path)
        output_srt_path = Path(output_srt_path)

        # Transcribe
        result = self.transcribe(audio_path, **kwargs)

        # Convert to SRT
        srt_subs = []
        for idx, segment in enumerate(result.get("segments", []), 1):
            if not segment.get("text"):
                continue
            sub = srt.Subtitle(
                index=idx,
                start=datetime.timedelta(seconds=segment["start"]),
                end=datetime.timedelta(seconds=segment["end"]),
                content=segment["text"]
            )
            srt_subs.append(sub)

        # Write SRT file
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_subs))

        logger.debug(f"Saved SRT to: {output_srt_path}")
        return output_srt_path

    def cleanup(self) -> None:
        """
        Release GPU/CPU memory by unloading the model.

        Should be called when the ASR instance is no longer needed.
        Idempotent - safe to call multiple times.
        """
        # Guard against double cleanup (native code can crash on double-free)
        if not hasattr(self, 'model') or self.model is None:
            logger.debug(f"{self.__class__.__name__} already cleaned up, skipping")
            return

        logger.debug(f"Cleaning up {self.__class__.__name__} resources...")

        try:
            del self.model
        except Exception as e:
            logger.warning(f"Error deleting model: {e}")
        finally:
            self.model = None
            logger.debug("Kotoba model unloaded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

        gc.collect()
        logger.debug(f"{self.__class__.__name__} cleanup complete")
