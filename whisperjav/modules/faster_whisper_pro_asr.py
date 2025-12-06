#!/usr/bin/env python3
"""
FasterWhisperPro ASR wrapper with internal VAD processing, refactored for the V3 architecture.
This module uses faster-whisper direct API and is intended for the 'balanced' pipeline.
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


def _is_silero_vad_cached(repo_or_dir: str) -> bool:
    """
    Check if a specific silero-vad version is already cached by torch.hub.

    Args:
        repo_or_dir: Repository string (e.g., "snakers4/silero-vad:v3.1")

    Returns:
        True if the exact version is cached, False otherwise
    """
    try:
        # Get torch hub cache directory
        hub_dir = torch.hub.get_dir()

        # Parse repo and version
        if ':' in repo_or_dir:
            repo_name, version = repo_or_dir.rsplit(':', 1)
        else:
            repo_name = repo_or_dir
            version = 'master'  # Default branch

        # Construct expected cache path
        # torch.hub caches repos as: hub_dir/owner_repo_branch
        repo_safe = repo_name.replace('/', '_')
        cached_repo_path = Path(hub_dir) / f"{repo_safe}_{version}"

        if cached_repo_path.exists():
            logger.debug(f"Silero VAD {version} found in cache: {cached_repo_path}")
            return True
        else:
            logger.debug(f"Silero VAD {version} NOT in cache, will download")
            return False
    except Exception as e:
        logger.warning(f"Error checking silero-vad cache: {e}")
        return False  # If we can't check, assume not cached


class FasterWhisperProASR:
    """Faster-Whisper ASR (direct API) with internal VAD, using structured v3 parameters."""

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

        # VAD parameters (kept separate as they're used by our VAD logic, not Whisper)
        self.vad_threshold = vad_params.get("threshold", 0.4)
        self.min_speech_duration_ms = vad_params.get("min_speech_duration_ms", 150)
        self.vad_chunk_threshold = vad_params.get("chunk_threshold", 4.0)

        # VAD engine repo (resolved from config, defaults to latest Silero VAD release)
        self.vad_repo = vad_params.get("vad_repo", "snakers4/silero-vad")

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

    def reset_statistics(self) -> None:
        """Public hook for pipelines to clear accumulated statistics."""
        self._reset_runtime_statistics()

    def get_filter_statistics(self) -> Dict[str, int]:
        """Return a copy of accumulated filter statistics."""
        return dict(self._filter_statistics)

    def _initialize_models(self):
        """Initialize VAD and Faster-Whisper models (direct API)."""
        logger.debug(f"Loading Silero VAD model from: {self.vad_repo}")
        try:
            # Use repo from config (resolved by TranscriptionTuner)
            # Check if already cached to avoid unnecessary download
            is_cached = _is_silero_vad_cached(self.vad_repo)

            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir=self.vad_repo,  # Config-driven, not hardcoded
                model="silero_vad",
                force_reload=not is_cached,  # Use cache if available
                onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = self.vad_utils

            status = "from cache" if is_cached else "downloaded"
            logger.debug(f"Silero VAD loaded ({status}) from {self.vad_repo}")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
            raise

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

        vad_segments = self._run_vad_on_audio(audio_data, sample_rate)

        # DIAGNOSTIC: Log VAD results
        logger.debug("=" * 60)
        logger.debug("DIAGNOSTIC: VAD Processing Results")
        logger.debug(f"  VAD groups detected: {len(vad_segments)}")
        if vad_segments:
            total_segments = sum(len(group) for group in vad_segments)
            logger.debug(f"  Total VAD segments: {total_segments}")
            for i, group in enumerate(vad_segments):
                logger.debug(f"  Group {i+1}: {len(group)} segments, "
                           f"duration {group[-1]['end_sec'] - group[0]['start_sec']:.2f}s")
        logger.debug("=" * 60)

        if not vad_segments:
            logger.debug(f"No speech detected in {audio_path.name}")
            return {"segments": [], "text": "", "language": self.whisper_params.get('language', 'ja')}

        all_segments = []
        for i, vad_group in enumerate(vad_segments, 1):
            logger.debug(f"Processing VAD group {i}/{len(vad_segments)}")
            segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group)
            all_segments.extend(segments)

        return {
            "segments": all_segments,
            "text": " ".join(seg["text"] for seg in all_segments),
            "language": self.whisper_params.get('language', 'ja')
        }

    def _run_vad_on_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[List[Dict]]:
        """
        Run VAD on audio data and return grouped segments.
        Preserves exact logic from original implementation including timestamp adjustments.
        """
        VAD_SR = 16000

        # Resample to 16kHz if needed
        if sample_rate != VAD_SR:
            resample_ratio = VAD_SR / sample_rate
            resampled_length = int(len(audio_data) * resample_ratio)
            indices = np.linspace(0, len(audio_data) - 1, resampled_length).astype(int)
            audio_16k = audio_data[indices]
        else:
            audio_16k = audio_data

        audio_tensor = torch.FloatTensor(audio_16k)

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=VAD_SR,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms
        )

        if not speech_timestamps:
            return []

        # Apply timestamp adjustments (preserved from original)
        for i in range(len(speech_timestamps)):
            speech_timestamps[i]["start"] = max(0, speech_timestamps[i]["start"] - 3200)
            speech_timestamps[i]["end"] = min(len(audio_tensor) - 16, speech_timestamps[i]["end"] + 20800)
            if i > 0 and speech_timestamps[i]["start"] < speech_timestamps[i - 1]["end"]:
                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]

        # Group segments
        groups = [[]]
        for i in range(len(speech_timestamps)):
            if (i > 0 and
                speech_timestamps[i]["start"] > speech_timestamps[i - 1]["end"] + (self.vad_chunk_threshold * VAD_SR)):
                groups.append([])
            groups[-1].append(speech_timestamps[i])

        # Add time in seconds for each segment
        for group in groups:
            for seg in group:
                seg["start_sec"] = seg["start"] / VAD_SR
                seg["end_sec"] = seg["end"] / VAD_SR

        return groups

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

        # Delete VAD model
        try:
            if hasattr(self, 'vad_model') and self.vad_model is not None:
                del self.vad_model
                self.vad_model = None
                logger.debug("VAD model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading VAD model: {e}")

        # Force garbage collection first (before CUDA cache clear)
        try:
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")

        # Clear CUDA cache if available (can sometimes cause issues on Windows)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache (non-fatal): {e}")

        logger.debug(f"{self.__class__.__name__} cleanup complete")
