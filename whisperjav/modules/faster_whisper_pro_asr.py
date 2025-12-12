#!/usr/bin/env python3
"""
FasterWhisperPro ASR wrapper using Speech Segmenter for speech detection.
This module uses faster-whisper direct API and is intended for the 'balanced' pipeline.

Speech segmentation is handled externally by the Speech Segmenter module.
This ASR module focuses solely on transcription.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import gc
import torch
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import srt
import datetime
import traceback
import logging

from whisperjav.utils.logger import logger
from whisperjav.utils.device_detector import get_best_device
from whisperjav.modules.segment_filters import SegmentFilterConfig, SegmentFilterHelper
from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory


class FasterWhisperProASR:
    """Faster-Whisper ASR (direct API) using Speech Segmenter for speech detection."""

    def __init__(self, model_config: Dict, params: Dict, task: str):
        """
        Initializes FasterWhisperPro ASR with structured v3 config parameters.

        Args:
            model_config: The 'model' section from the resolved config.
            params: The 'params' section containing decoder, vad, and provider settings.
            task: The ASR task to perform ('transcribe' or 'translate').
        """
        # --- V3 PARAMETER UNPACKING ---
        self.model_name = model_config.get("model_name", "large-v2")
        # Use smart device detection: CUDA → MPS → CPU
        self.device = model_config.get("device", get_best_device())
        # Default to int8 for quantized models (faster-whisper uses CTranslate2 quantized models)
        self.compute_type = model_config.get("compute_type", "int8")

        decoder_params = params["decoder"]
        vad_params = params["vad"]
        provider_params = params["provider"]

        # VAD parameters (passed to Speech Segmenter)
        self.vad_threshold = vad_params.get("threshold", 0.4)
        self.min_speech_duration_ms = vad_params.get("min_speech_duration_ms", 150)
        self.vad_chunk_threshold = vad_params.get("chunk_threshold", 4.0)

        # Speech Segmenter (MANDATORY - single owner of speech segmentation)
        # All speech segmentation goes through this contract
        speech_segmenter_config = params.get("speech_segmenter", {})
        segmenter_backend = speech_segmenter_config.get("backend", "silero-v4.0")  # Default to silero-v4.0

        try:
            self._external_segmenter = SpeechSegmenterFactory.create(
                segmenter_backend,
                config=speech_segmenter_config
            )
            logger.info(f"Speech Segmenter initialized: {self._external_segmenter.name}")
        except Exception as e:
            logger.error(f"Failed to create Speech Segmenter '{segmenter_backend}': {e}")
            raise ValueError(f"Speech Segmenter not configured - this is an architecture violation: {e}")

        # FIX: Combine all Whisper parameters into a single dictionary
        # The faster-whisper (via stable-ts) expects all parameters in one transcribe() call
        self.whisper_params = {}
        self.whisper_params.update(decoder_params)  # task, language, beam_size, etc.
        self.whisper_params.update(provider_params)  # temperature, fp16, etc.

        raw_threshold = self.whisper_params.get("logprob_threshold", -1.0)
        self.logprob_threshold = float(raw_threshold) if raw_threshold is not None else None
        raw_margin = self.whisper_params.get("logprob_margin", 0.0)
        self.logprob_margin = float(raw_margin or 0.0)
        self.drop_nonverbal_vocals = bool(self.whisper_params.get("drop_nonverbal_vocals", False))
        filter_config = SegmentFilterConfig(
            logprob_threshold=self.logprob_threshold,
            logprob_margin=self.logprob_margin,
            drop_nonverbal_vocals=self.drop_nonverbal_vocals,
        )
        self._segment_filter = SegmentFilterHelper(filter_config)

        # Helper-only parameters should not be forwarded to the backend
        self.whisper_params.pop("logprob_margin", None)
        self.whisper_params.pop("drop_nonverbal_vocals", None)

        # Ensure task is set correctly
        self.whisper_params['task'] = task
        # Language is now passed from decoder_params (set via CLI --language argument)

        # Store task for metadata
        self.task = task

        # Log task for debugging translation issues
        logger.info(f"FasterWhisperProASR initialized with task='{task}'")
        if task == 'translate':
            logger.info("Translation mode enabled - output will be in English")
        self._logged_param_snapshot = False
        self._vad_parameters = self._build_vad_parameters(vad_params)
        self._reset_runtime_statistics()
        # --- END V3 PARAMETER UNPACKING ---

        # Suppression lists for Japanese content (business logic preserved)
        self.suppress_low = ["Thank you", "視聴", "Thanks for"]
        self.suppress_high = ["視聴ありがとうございました", "ご視聴ありがとうございました",
                              "字幕作成者", "提供", "スポンサー"]

        self._initialize_models()
        self._log_sensitivity_parameters()

    def _reset_runtime_statistics(self) -> None:
        """Reset per-run counters used for final summaries."""
        self._filter_statistics = {
            'logprob_filtered': 0,
            'nonverbal_filtered': 0
        }
        # VAD segments from the last transcription (for visualization data contract)
        self._last_vad_segments: List[Dict] = []

    def reset_statistics(self) -> None:
        """Public hook for pipelines to clear accumulated statistics."""
        self._reset_runtime_statistics()

    def get_filter_statistics(self) -> Dict[str, int]:
        """Return a copy of accumulated filter statistics."""
        return dict(self._filter_statistics)

    def get_last_vad_segments(self) -> List[Dict]:
        """
        Return VAD segments from the last transcription.

        Returns:
            List of dicts with 'start_sec' and 'end_sec' keys representing
            speech regions detected by Silero VAD before transcription.

        Note:
            Timestamps are relative to the audio file that was transcribed.
            For scene-based pipelines, the caller should add the scene's
            start offset to get absolute timestamps.
        """
        return list(self._last_vad_segments)

    def _initialize_models(self):
        """Initialize Faster-Whisper model (direct API)."""
        # Note: Speech segmentation is handled by external Speech Segmenter (set in __init__)
        # Faster-Whisper direct API initialization
        logger.debug(
            f"Loading Faster-Whisper model (direct API): {self.model_name} "
            f"on device: {self.device}, compute_type: {self.compute_type}"
        )

        # DIAGNOSTIC: Log exact initialization parameters
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: Faster-Whisper Model Initialization")
        logger.debug(f"  model_size_or_path: {self.model_name}")
        logger.debug(f"  device: {self.device}")
        logger.debug(f"  compute_type: {self.compute_type}")
        logger.debug(f"  cpu_threads: 0 (default=4)")
        logger.debug(f"  num_workers: 1")
        logger.debug("=" * 60)

        try:
            self.whisper_model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=0,    # Use default (4 threads)
                num_workers=1     # Single worker (concurrency at scene level)
            )
            logger.debug(
                f"Faster-Whisper model loaded successfully "
                f"(backend: ctranslate2, direct API - no stable-ts wrapper)"
            )
            logger.debug(f"Model type: {type(self.whisper_model)}")
        except Exception as e:
            # VRAM exhaustion fallback - try int8 if float16/other fails on CUDA
            if self.device == "cuda" and self.compute_type != "int8":
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
                        f"CUDA {self.compute_type} failed (likely VRAM exhaustion): {e}. "
                        "Falling back to int8 compute type."
                    )
                    try:
                        # Clear CUDA cache before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                        self.whisper_model = WhisperModel(
                            model_size_or_path=self.model_name,
                            device=self.device,
                            compute_type="int8",
                            cpu_threads=0,
                            num_workers=1,
                        )
                        self.compute_type = "int8"  # Update for logging/metadata
                        logger.info("Faster-Whisper model loaded successfully with int8 fallback")
                        return
                    except Exception as fallback_e:
                        logger.error(f"int8 fallback also failed: {fallback_e}")
                        raise fallback_e

            logger.error(f"Failed to load Faster-Whisper model: {e}", exc_info=True)
            logger.error(f"Model name that failed: {self.model_name}")
            logger.error(f"Device: {self.device}, Compute type: {self.compute_type}")
            raise

    def _log_sensitivity_parameters(self):
        """Log the sensitivity-related parameters for debugging."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- FasterWhisperProASR Initialized with V3 Parameters ---")
            logger.debug(f"  VAD Threshold: {self.vad_threshold}")
            logger.debug(f"  Min Speech Duration: {self.min_speech_duration_ms}ms")
            logger.debug(f"  Task: {self.task}")
            logger.debug(f"  Logprob Threshold: {self.logprob_threshold}")
            logger.debug(
                "  Logprob Margin (<= %.1fs): %s",
                self._segment_filter.short_window,
                self.logprob_margin,
            )
            logger.debug(f"  Drop Nonverbal Vocals: {self.drop_nonverbal_vocals}")
            logger.debug(f"  Backend: faster-whisper (via stable-ts)")
            logger.debug(f"  Combined Whisper Params: {self.whisper_params}")
            logger.debug("----------------------------------------------------")

    def _build_vad_parameters(self, vad_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert config VAD settings into faster-whisper VadOptions format."""
        if not vad_params:
            return None

        mapping = {
            'threshold': 'threshold',  # faster-whisper VadOptions expects the raw threshold name
            'neg_threshold': 'neg_threshold',  # optional, only present if config defines it
            'min_speech_duration_ms': 'min_speech_duration_ms',
            'max_speech_duration_s': 'max_speech_duration_s',
            'min_silence_duration_ms': 'min_silence_duration_ms',
            'speech_pad_ms': 'speech_pad_ms'
        }

        vad_options: Dict[str, Any] = {}
        for source_key, target_key in mapping.items():
            if source_key in vad_params and vad_params[source_key] is not None:
                vad_options[target_key] = vad_params[source_key]

        return vad_options or None

    def _prepare_whisper_params(self) -> Dict:
        """
        Prepare final parameters for faster-whisper direct API.
        Handles parameter renaming, type conversions, and ctranslate2 compatibility.
        """
        # Start with tuner-provided parameters
        final_params = self.whisper_params.copy()

        # 1. Parameter name mapping
        if 'logprob_threshold' in final_params:
            final_params['log_prob_threshold'] = final_params.pop('logprob_threshold')
            logger.debug("Renamed logprob_threshold → log_prob_threshold")

        # 2. Type conversions for ctranslate2 compatibility

        # suppress_tokens: Must be list, not tuple or int
        if 'suppress_tokens' in final_params:
            tokens = final_params['suppress_tokens']
            if isinstance(tokens, tuple):
                final_params['suppress_tokens'] = list(tokens)
                logger.debug(f"Converted suppress_tokens from tuple to list ({len(tokens)} tokens)")
            elif isinstance(tokens, int):
                final_params['suppress_tokens'] = [tokens]
                logger.debug(f"Wrapped suppress_tokens int in list: {tokens} → [{tokens}]")
            elif not isinstance(tokens, list):
                logger.warning(f"Invalid suppress_tokens type: {type(tokens)}, removing")
                del final_params['suppress_tokens']

        # no_repeat_ngram_size: Must be int, not float
        if 'no_repeat_ngram_size' in final_params:
            try:
                original_value = final_params['no_repeat_ngram_size']
                final_params['no_repeat_ngram_size'] = int(original_value)
                logger.debug(f"Converted no_repeat_ngram_size to int: {original_value} → {final_params['no_repeat_ngram_size']}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid no_repeat_ngram_size value, removing: {final_params['no_repeat_ngram_size']}")
                del final_params['no_repeat_ngram_size']

        # temperature: Can be float, list, or tuple - normalize to tuple for sequences
        if 'temperature' in final_params:
            temp = final_params['temperature']
            if isinstance(temp, list):
                # Convert to tuple for faster-whisper (both work, but tuple is canonical)
                final_params['temperature'] = tuple(temp) if len(temp) > 1 else temp[0]
                logger.debug(f"Converted temperature list to tuple/float")
            # float and tuple are already valid

        # 3. Remove parameters not in faster-whisper API
        params_to_remove = [
            'fp16',          # Handled at model init via compute_type
            'verbose',       # Replaced by log_progress
            'vad',           # We use external VAD
            'vad_threshold', # Our VAD parameter, not faster-whisper's
            'hallucination_silence_threshold', # Custom post-processing param
        ]

        removed = []
        for param in params_to_remove:
            if param in final_params:
                del final_params[param]
                removed.append(param)

        if removed:
            logger.debug(f"Removed parameters not in faster-whisper API: {removed}")

        # 4. Set faster-whisper specific defaults
        if 'log_progress' not in final_params:
            final_params['log_progress'] = False  # Suppress progress bars

        final_params['vad_filter'] = final_params.get('vad_filter', False)

        if self._vad_parameters and 'vad_parameters' not in final_params:
            final_params['vad_parameters'] = self._vad_parameters

        # 5. Remove None values to allow library defaults to apply
        # This prevents "NoneType is not iterable" errors for params like clip_timestamps
        # where the library expects a specific type (str/list) or its own default.
        keys_to_remove = [k for k, v in final_params.items() if v is None]
        for k in keys_to_remove:
            del final_params[k]
        
        if keys_to_remove:
            logger.debug(f"Removed None values to use library defaults: {keys_to_remove}")

        logger.debug(f"Final faster-whisper parameters ({len(final_params)} params): {final_params}")

        if not self._logged_param_snapshot and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Faster-Whisper transcribe kwargs (model=%s, device=%s, compute_type=%s): %s",
                self.model_name,
                self.device,
                self.compute_type,
                final_params
            )
            self._logged_param_snapshot = True

        return final_params

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """Transcribe audio file with internal VAD processing."""
        audio_path = Path(audio_path)
        logger.debug(f"Transcribing with VAD: {audio_path.name}")

        # DIAGNOSTIC: Log audio file info
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: Audio File Loading")
        logger.debug(f"  File path: {audio_path}")
        logger.debug(f"  File exists: {audio_path.exists()}")
        if audio_path.exists():
            logger.debug(f"  File size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")

        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
            logger.debug(f"  Audio loaded successfully")
            logger.debug(f"  Sample rate: {sample_rate} Hz")
            logger.debug(f"  Original shape: {audio_data.shape}")
            logger.debug(f"  Data type: {audio_data.dtype}")

            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.debug(f"  Converted stereo to mono")
                logger.debug(f"  Final shape: {audio_data.shape}")
            logger.debug("=" * 60)
        except Exception as e:
            logger.error(f"Failed to read audio file {audio_path}: {e}")
            raise

        # Run speech segmentation through the Speech Segmenter contract
        vad_segments = self._run_speech_segmentation(audio_data, sample_rate)

        # Store segments for visualization data contract
        # Flatten grouped segments into simple list with start_sec/end_sec
        self._last_vad_segments = []
        for group in vad_segments:
            for seg in group:
                self._last_vad_segments.append({
                    "start_sec": round(seg["start_sec"], 3),
                    "end_sec": round(seg["end_sec"], 3)
                })

        # DIAGNOSTIC: Log segmentation results
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: Speech Segmentation Results")
        logger.debug(f"  Segment groups detected: {len(vad_segments)}")
        if vad_segments:
            total_segments = sum(len(group) for group in vad_segments)
            logger.debug(f"  Total segments: {total_segments}")
            for i, group in enumerate(vad_segments):
                logger.debug(f"  Group {i+1}: {len(group)} segments, "
                           f"duration {group[-1]['end_sec'] - group[0]['start_sec']:.2f}s")
            # INFO level summary for user visibility
            logger.info(f"Speech segmentation complete: {total_segments} segments in {len(vad_segments)} groups")
        logger.debug("=" * 60)

        # Handle "none" backend case - no segments means transcribe full audio
        if not vad_segments:
            # Check if this is because segmenter returned empty (no speech) or "none" backend
            if self._external_segmenter.name == "none":
                logger.info("Speech segmentation: disabled (none backend) - transcribing full audio directly")
                all_segments = self._transcribe_full_audio(audio_data, sample_rate)
            else:
                logger.debug(f"No speech detected in {audio_path.name}")
                return {"segments": [], "text": "", "language": self.whisper_params.get('language', 'ja')}
        else:
            all_segments = []
            for i, vad_group in enumerate(vad_segments, 1):
                logger.debug(f"Processing segment group {i}/{len(vad_segments)}")
                segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group)
                all_segments.extend(segments)

        return {
            "segments": all_segments,
            "text": " ".join(seg["text"] for seg in all_segments),
            "language": self.whisper_params.get('language', 'ja')
        }

    def _run_speech_segmentation(self, audio_data: np.ndarray, sample_rate: int) -> List[List[Dict]]:
        """
        Run speech segmentation through the Speech Segmenter contract.

        All speech segmentation is handled by the external Speech Segmenter module.
        This is the single entry point for segmentation in ASR.
        """
        if self._external_segmenter is None:
            raise ValueError("Speech Segmenter not configured - this is an architecture violation")

        logger.info(f"Starting speech segmentation with: {self._external_segmenter.name}")

        try:
            result = self._external_segmenter.segment(audio_data, sample_rate=sample_rate)
            logger.debug(
                f"Speech Segmenter '{self._external_segmenter.name}' found "
                f"{result.num_segments} segments in {result.num_groups} groups "
                f"({result.processing_time_sec:.2f}s)"
            )
            return result.to_legacy_format()
        except Exception as e:
            logger.error(f"Speech Segmenter failed: {e}")
            raise

    def _transcribe_full_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Transcribe full audio without speech segmentation.
        Used when Speech Segmenter backend is set to 'none'.
        """
        duration = len(audio_data) / sample_rate
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: Full Audio Transcription (VAD bypassed)")
        logger.debug(f"  Model: {self.model_name}")
        logger.debug(f"  Audio duration: {duration:.2f}s")
        logger.debug(f"  Audio shape: {audio_data.shape}")
        logger.debug(f"  Audio dtype: {audio_data.dtype}")
        logger.debug(f"  Sample rate: {sample_rate} Hz")
        logger.debug("=" * 60)

        # Get cleaned parameters from tuner
        whisper_params = self._prepare_whisper_params()

        try:
            # faster-whisper returns (generator, info) tuple
            segments_generator, transcription_info = self.whisper_model.transcribe(
                audio_data,
                **whisper_params
            )

            # Consume generator and convert to dict format
            raw_segments = []
            for segment in segments_generator:
                raw_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob
                })

            logger.debug(f"Full audio transcription produced {len(raw_segments)} segments")

            # Apply segment filtering (same as VAD path)
            filtered_segments = self._segment_filter.filter_segments(raw_segments)
            logger.debug(f"After filtering: {len(filtered_segments)} segments (removed {len(raw_segments) - len(filtered_segments)})")

            return filtered_segments

        except Exception as e:
            logger.error(f"Full audio transcription failed: {type(e).__name__}: {e}")
            raise

    def _transcribe_vad_group(self, audio_data: np.ndarray, sample_rate: int,
                             vad_group: List[Dict]) -> List[Dict]:
        """
        Transcribe a group of VAD segments using faster-whisper direct API.
        """
        if not vad_group:
            return []

        start_sec = vad_group[0]["start_sec"]
        end_sec = vad_group[-1]["end_sec"]
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        group_audio = audio_data[start_sample:end_sample]

        # Get cleaned parameters from tuner
        whisper_params = self._prepare_whisper_params()

        # DIAGNOSTIC: Log pre-transcription state
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: Pre-Transcription State")
        logger.debug(f"  Model: {self.model_name}")
        logger.debug(f"  VAD group span: {start_sec:.2f}s - {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")
        logger.debug(f"  Group audio type: {type(group_audio)}")
        logger.debug(f"  Group audio shape: {group_audio.shape}")
        logger.debug(f"  Group audio dtype: {group_audio.dtype}")
        logger.debug(f"  Sample rate: {sample_rate} Hz")
        logger.debug(f"  Audio is numpy array: {isinstance(group_audio, np.ndarray)}")
        logger.debug(f"  Audio length: {len(group_audio)} samples = {len(group_audio)/sample_rate:.2f}s")
        logger.debug(f"  Whisper params keys: {list(whisper_params.keys())}")
        logger.debug(f"  Full whisper params: {whisper_params}")
        logger.debug("=" * 60)

        try:
            # DIAGNOSTIC: Log exact transcribe call
            logger.debug("DIAGNOSTIC: Calling whisper_model.transcribe()...")
            logger.debug(f"  Input type: {type(group_audio)}")
            logger.debug(f"  Input shape: {group_audio.shape}")
            logger.debug(f"  Kwargs count: {len(whisper_params)}")

            # Log task at INFO level for translation debugging
            actual_task = whisper_params.get('task', 'transcribe')
            if actual_task == 'translate':
                logger.info(f"Transcribing with task='{actual_task}' - output should be in English")
            else:
                logger.debug(f"Transcribing with task='{actual_task}'")

            # CHANGED: faster-whisper returns (generator, info) tuple
            segments_generator, transcription_info = self.whisper_model.transcribe(
                group_audio,
                **whisper_params
            )

            logger.debug("DIAGNOSTIC: transcribe() call successful, got generator and info")
            logger.debug(f"  Info type: {type(transcription_info)}")
            logger.debug(f"  Generator type: {type(segments_generator)}")

            # CHANGED: Consume generator and convert to dict format
            raw_segments = []
            for segment in segments_generator:
                raw_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob
                })

            logger.debug(f"DIAGNOSTIC: Generator consumed successfully, got {len(raw_segments)} raw segments")
            result = {"segments": raw_segments}

            # Validate translation output - warn if translation was requested but output appears Japanese
            if actual_task == 'translate' and raw_segments:
                # Check if output contains significant Japanese characters
                sample_text = ' '.join(seg['text'] for seg in raw_segments[:5])  # First 5 segments
                japanese_char_count = sum(1 for c in sample_text if '\u3040' <= c <= '\u309f' or  # Hiragana
                                                                   '\u30a0' <= c <= '\u30ff' or  # Katakana
                                                                   '\u4e00' <= c <= '\u9fff')    # Kanji
                total_chars = len(sample_text.replace(' ', ''))
                if total_chars > 0 and japanese_char_count / total_chars > 0.3:
                    logger.warning(f"Translation mode was requested but output appears to be in Japanese "
                                   f"({japanese_char_count}/{total_chars} chars are Japanese). "
                                   f"This may indicate faster-whisper translation is not working as expected.")
                    logger.warning(f"Sample output: {sample_text[:100]}...")
                else:
                    logger.info(f"Translation output validation: appears to be English (good)")

        except Exception as e:
            # DIAGNOSTIC: Enhanced error logging
            logger.error("=" * 60)
            logger.error("DIAGNOSTIC: Transcription FAILED")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception message: {e}")
            logger.error(f"  Model: {self.model_name}")
            logger.error(f"  Audio input type: {type(group_audio)}")
            logger.error(f"  Audio shape: {group_audio.shape}")
            logger.error(f"  Params that were passed: {whisper_params}")
            logger.error("=" * 60)
            logger.error(f"Full traceback:", exc_info=True)

            # Fallback to minimal parameters
            try:
                minimal_params = {
                    'task': self.whisper_params.get('task', 'transcribe'),
                    'language': self.whisper_params.get('language', 'ja'),
                    'temperature': 0.0,
                    'beam_size': 3,  # Reduced from 5 for faster fallback
                    'log_progress': False  # CHANGED: was 'verbose': None
                }
                # Note: Removed 'fp16' - already set at init via compute_type

                logger.warning(f"Retrying with minimal parameters: {minimal_params}")

                segments_generator, info = self.whisper_model.transcribe(
                    group_audio,
                    **minimal_params
                )

                raw_segments = []
                for segment in segments_generator:
                    raw_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": segment.avg_logprob
                    })

                result = {"segments": raw_segments}
                logger.info("DIAGNOSTIC: Minimal parameters fallback succeeded")

            except Exception as e2:
                logger.error("=" * 60)
                logger.error("DIAGNOSTIC: Even MINIMAL PARAMETERS failed")
                logger.error(f"  Exception type: {type(e2).__name__}")
                logger.error(f"  Exception message: {e2}")
                logger.error(f"  Model: {self.model_name}")
                logger.error(f"  Minimal params tried: {minimal_params}")
                logger.error("=" * 60)
                logger.error(f"Full traceback:", exc_info=True)
                return []

        if not result or not result.get("segments"):
            return []

        # Adjust timestamps and apply suppression (preserved from original)
        segments = []
        for seg in result["segments"]:
            text = seg["text"].strip() if isinstance(seg["text"], str) else seg.get("text", "").strip()
            if not text:
                continue

            # Apply high suppression list filtering
            if any(suppress in text for suppress in self.suppress_high):
                logger.debug(f"Filtered segment due to suppression word: {text[:30]}...")
                continue

            # Apply logprob filtering with suppression penalties
            avg_logprob = seg.get("avg_logprob", 0.0)

            # Apply text-based suppression penalties
            for suppress_word in self.suppress_low:
                if suppress_word in text:
                    avg_logprob -= 0.15

            segment_duration = max(0.0, float(seg["end"] - seg["start"]))
            should_filter, reason, effective_threshold = self._segment_filter.should_filter(
                avg_logprob=avg_logprob,
                duration=segment_duration,
                text=text,
            )

            if should_filter:
                if reason == "logprob":
                    logger.debug(
                        "Filtered segment with logprob %.3f (threshold %.3f): %s",
                        avg_logprob,
                        effective_threshold if effective_threshold is not None else -1.0,
                        f"{text[:50]}...",
                    )
                    self._filter_statistics['logprob_filtered'] += 1
                else:
                    logger.debug(f"Filtered nonverbal segment: {text[:50]}...")
                    self._filter_statistics['nonverbal_filtered'] += 1
                continue

            adjusted_seg = {
                "start": seg["start"] + start_sec,
                "end": seg["end"] + start_sec,
                "text": text,
                "avg_logprob": avg_logprob
            }
            segments.append(adjusted_seg)

        return segments

    def transcribe_to_srt(self,
                         audio_path: Union[str, Path],
                         output_srt_path: Union[str, Path],
                         **kwargs) -> Path:
        """Transcribes audio and saves the result as an SRT file."""
        audio_path = Path(audio_path)
        output_srt_path = Path(output_srt_path)

        result = self.transcribe(audio_path, **kwargs)

        srt_subs = []
        for idx, segment in enumerate(result.get("segments", []), 1):
            sub = srt.Subtitle(
                index=idx,
                start=datetime.timedelta(seconds=segment["start"]),
                end=datetime.timedelta(seconds=segment["end"]),
                content=segment["text"]
            )
            srt_subs.append(sub)

        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_subs))

        logger.debug(f"Saved SRT to: {output_srt_path}")
        return output_srt_path

    def cleanup(self):
        """
        Release GPU/CPU memory by unloading models.

        This should be called when the ASR instance is no longer needed,
        especially in batch processing scenarios where models need to be
        swapped between passes.
        """
        import gc

        logger.debug(f"Cleaning up {self.__class__.__name__} resources...")

        # Delete Whisper model
        try:
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
                logger.debug("Whisper model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading Whisper model: {e}")

        # Note: Speech Segmenter cleanup is handled by the segmenter itself
        # No VAD model to clean up - all segmentation is external

        # Force garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")

        # NOTE: CUDA cache cleanup is handled by caller via safe_cuda_cleanup()
        # This keeps ASR modules free of subprocess-awareness logic.

        logger.debug(f"{self.__class__.__name__} cleanup complete")
