#!/usr/bin/env python3
"""
WhisperPro ASR wrapper using Speech Segmenter for speech detection.
This module uses the standard whisper library and is intended for pipelines like 'fidelity'.

Speech segmentation is handled externally by the Speech Segmenter module.
This ASR module focuses solely on transcription.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import torch
import whisper
import soundfile as sf
import numpy as np
import srt
import datetime
import traceback
import logging

from whisperjav.utils.logger import logger
from whisperjav.utils.device_detector import get_best_device
from whisperjav.utils.parameter_tracer import NullTracer
from whisperjav.modules.segment_filters import SegmentFilterConfig, SegmentFilterHelper
from whisperjav.modules.vad_failover import should_force_full_transcribe
from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory


class WhisperProASR:
    """Whisper ASR using Speech Segmenter for speech detection."""
    
    def __init__(self, model_config: Dict, params: Dict, task: str, tracer=None):
        """
        Initializes WhisperPro ASR with structured v3 config parameters.

        Args:
            model_config: The 'model' section from the resolved config.
            params: The 'params' section containing decoder, vad, and provider settings.
            task: The ASR task to perform ('transcribe' or 'translate').
            tracer: Optional parameter tracer for capturing transcribe() params.
        """
        # Parameter tracer for capturing transcribe() call parameters
        self.tracer = tracer if tracer is not None else NullTracer()
        # --- V3 PARAMETER UNPACKING ---
        self.model_name = model_config.get("model_name", "large-v2")
        # Use smart device detection: CUDA → MPS → CPU
        self.device = model_config.get("device", get_best_device())
        
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

        # CRITICAL: Merge VAD params into speech segmenter config for sensitivity tuning
        # Without this, sensitivity settings (threshold, min_speech_duration_ms) are lost
        # and the segmenter uses its own defaults instead of the tuned values
        merged_segmenter_config = {**vad_params, **speech_segmenter_config}

        try:
            self._external_segmenter = SpeechSegmenterFactory.create(
                segmenter_backend,
                config=merged_segmenter_config
            )
            logger.info(f"Speech Segmenter initialized: {self._external_segmenter.name}")
        except Exception as e:
            logger.error(f"Failed to create Speech Segmenter '{segmenter_backend}': {e}")
            raise ValueError(f"Speech Segmenter not configured - this is an architecture violation: {e}")

        # FIX: Combine all Whisper parameters into a single dictionary
        # The Whisper library expects all parameters in one transcribe() call
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

        self.whisper_params.pop("logprob_margin", None)
        self.whisper_params.pop("drop_nonverbal_vocals", None)

        # Ensure task is set correctly
        self.whisper_params['task'] = task
        # Language is now passed from decoder_params (set via CLI --language argument)
        
        # Store task for metadata
        self.task = task

        # Log task for debugging translation issues
        logger.info(f"WhisperProASR initialized with task='{task}'")
        if task == 'translate':
            logger.info("Translation mode enabled - output will be in English")

        self._reset_runtime_statistics()
        # --- END V3 PARAMETER UNPACKING ---

        # Suppression lists for Japanese content (business logic preserved)
        self.suppress_low = ["Thank you", "視聴", "Thanks for"]
        self.suppress_high = ["視聴ありがとうございました", "ご視聴ありがとうございました", 
                              "字幕作成者", "提供", "スポンサー"]
        
        self._initialize_models()
        self._log_sensitivity_parameters()

    def _reset_runtime_statistics(self) -> None:
        """Reset counters that feed into final stats summaries."""
        self._filter_statistics = {
            'logprob_filtered': 0,
            'nonverbal_filtered': 0
        }

    def reset_statistics(self) -> None:
        """Public hook for pipelines to clear accumulated statistics."""
        self._reset_runtime_statistics()

    def get_filter_statistics(self) -> Dict[str, int]:
        """Expose collected statistics for downstream reporting."""
        return dict(self._filter_statistics)
        
    def _initialize_models(self):
        """Initialize Whisper model."""
        # Note: Speech segmentation is handled by external Speech Segmenter (set in __init__)
        logger.debug(f"Loading Whisper model: {self.model_name} on device: {self.device}")
        self.whisper_model = whisper.load_model(self.model_name, device=self.device)

    def _log_sensitivity_parameters(self):
        """Log the sensitivity-related parameters for debugging."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- WhisperProASR Initialized with V3 Parameters ---")
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
            logger.debug(f"  Combined Whisper Params: {self.whisper_params}")
            logger.debug("----------------------------------------------------")

    def _prepare_whisper_params(self) -> Dict:
        """
        Prepare final parameters for Whisper transcribe call.
        All tuner-provided parameters are already cleaned and properly structured.
        """
        # Start with tuner-provided parameters (already cleaned of None values)
        final_params = self.whisper_params.copy()
        
        # Handle temperature type conversion if needed
        if 'temperature' in final_params and isinstance(final_params['temperature'], list):
            final_params['temperature'] = tuple(final_params['temperature'])
        
        # Only override verbose if not already set by tuner
        if 'verbose' not in final_params:
            final_params['verbose'] = None  # Suppress progress bars
        
        logger.debug(f"Final Whisper parameters: {final_params}")
        return final_params

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """Transcribe audio file with internal VAD processing.

        Args:
            audio_path: Path to the audio file
            **kwargs: Optional overrides. Supports 'task' to override the default task.
                      If task='translate', output will be in English.
        """
        audio_path = Path(audio_path)

        # Handle task override from kwargs (important for direct-to-english support)
        if 'task' in kwargs:
            runtime_task = kwargs.pop('task')
            if runtime_task != self.task:
                logger.info(f"Task override: '{self.task}' → '{runtime_task}' (runtime)")
                self.whisper_params['task'] = runtime_task
                self.task = runtime_task

        logger.debug(f"Transcribing with VAD: {audio_path.name}")

        # Log current task at INFO level for translation debugging
        if self.task == 'translate':
            logger.info(f"Transcribing '{audio_path.name}' with task='translate' → output will be in English")
        else:
            logger.debug(f"Transcribing '{audio_path.name}' with task='{self.task}'")
        
        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
        except Exception as e:
            logger.error(f"Failed to read audio file {audio_path}: {e}")
            raise

        audio_duration = len(audio_data) / sample_rate if sample_rate else 0.0

        # Run speech segmentation through the Speech Segmenter contract
        vad_segments = self._run_speech_segmentation(audio_data, sample_rate)

        # Store segments for visualization data contract
        self._last_vad_segments = []
        for group in (vad_segments or []):
            for seg in group:
                self._last_vad_segments.append({
                    "start_sec": round(seg["start_sec"], 3),
                    "end_sec": round(seg["end_sec"], 3)
                })

        # INFO level summary for user visibility
        if vad_segments:
            total_segments = sum(len(group) for group in vad_segments)
            logger.info(f"Speech segmentation complete: {total_segments} segments in {len(vad_segments)} groups")

        fallback_triggered = False

        # Handle "none" backend case - no segments means transcribe full audio
        if not vad_segments:
            if self._external_segmenter.name == "none":
                logger.info("Speech segmentation: disabled (none backend) - transcribing full audio directly")
                all_segments = self._transcribe_full_audio(audio_data)
            elif should_force_full_transcribe(vad_segments, audio_duration):
                logger.warning(
                    "Speech segmentation produced insufficient coverage (segments=%s, duration=%.1fs). "
                    "Falling back to full-clip transcription.",
                    0,
                    audio_duration,
                )
                fallback_triggered = True
                all_segments = self._transcribe_full_audio(audio_data)
            else:
                logger.debug(f"No speech detected in {audio_path.name}")
                return {"segments": [], "text": "", "language": self.whisper_params.get('language', 'ja')}
        elif should_force_full_transcribe(vad_segments, audio_duration):
            logger.warning(
                "Speech segmentation produced insufficient coverage (segments=%s, duration=%.1fs). "
                "Falling back to full-clip transcription.",
                sum(len(group or []) for group in (vad_segments or [])),
                audio_duration,
            )
            fallback_triggered = True
            all_segments = self._transcribe_full_audio(audio_data)
        else:
            all_segments = []
            for vad_group in vad_segments:
                segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group)
                all_segments.extend(segments)

        if fallback_triggered and not all_segments:
            # Safety net: if fallback still produced nothing, log explicitly for debugging.
            logger.warning(
                "Full-clip fallback for %s emitted no segments. Returning empty result.",
                audio_path.name,
            )

        # Validate translation output - warn if translation was requested but output appears Japanese
        if self.task == 'translate' and all_segments:
            # Check if output contains significant Japanese characters
            sample_text = ' '.join(seg['text'] for seg in all_segments[:5])  # First 5 segments
            japanese_char_count = sum(1 for c in sample_text if '\u3040' <= c <= '\u309f' or  # Hiragana
                                                               '\u30a0' <= c <= '\u30ff' or  # Katakana
                                                               '\u4e00' <= c <= '\u9fff')    # Kanji
            total_chars = len(sample_text.replace(' ', ''))
            if total_chars > 0 and japanese_char_count / total_chars > 0.3:
                logger.warning(f"Translation mode was requested but output appears to be in Japanese "
                               f"({japanese_char_count}/{total_chars} chars are Japanese). "
                               f"This may indicate Whisper translation is not working as expected.")
                logger.warning(f"Sample output: {sample_text[:100]}...")
            else:
                logger.info(f"Translation output validation: appears to be English (good)")

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

    def _transcribe_vad_group(self, audio_data: np.ndarray, sample_rate: int,
                             vad_group: List[Dict]) -> List[Dict]:
        """
        Transcribe a group of VAD segments using properly structured tuner parameters.
        """
        if not vad_group:
            return []

        start_sec = vad_group[0]["start_sec"]
        end_sec = vad_group[-1]["end_sec"]
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        group_audio = audio_data[start_sample:end_sample]
        
        whisper_params = self._prepare_whisper_params()
        logging.debug("Final Parameters for Debugging: %s", whisper_params)

        result = self._run_whisper_with_fallback(group_audio, whisper_params)
        if not result or not result.get("segments"):
            return []

        return self._process_segments(result["segments"], start_sec)

    def _transcribe_full_audio(self, audio_data: np.ndarray) -> List[Dict]:
        """Transcribe the full audio clip without VAD segmentation."""
        whisper_params = self._prepare_whisper_params()
        result = self._run_whisper_with_fallback(audio_data, whisper_params)
        if not result or not result.get("segments"):
            return []
        return self._process_segments(result["segments"], 0.0)

    def _run_whisper_with_fallback(self, audio_chunk: np.ndarray, whisper_params: Dict,
                                     audio_info: Optional[Dict] = None, context: Optional[str] = None) -> Optional[Dict]:
        """Execute whisper transcription with a minimal-params retry."""
        # Trace transcribe() parameters for diagnostic visibility
        trace_audio_info = audio_info or {
            "shape": str(audio_chunk.shape),
            "dtype": str(audio_chunk.dtype),
        }
        self.tracer.emit_transcribe_params(
            params=whisper_params,
            audio_info=trace_audio_info,
            context=context or "whisper_transcribe"
        )

        try:
            return self.whisper_model.transcribe(audio_chunk, **whisper_params)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)

            minimal_params = self._minimal_whisper_params()
            try:
                logger.warning(f"Retrying transcription with minimal parameters: {minimal_params}")
                return self.whisper_model.transcribe(audio_chunk, **minimal_params)
            except Exception as e2:
                logger.error("Minimal parameter transcription also failed", exc_info=True)
                logger.error(f"Original error: {e}; fallback error: {e2}")
                return None

    def _minimal_whisper_params(self) -> Dict:
        return {
            'task': self.whisper_params.get('task', 'transcribe'),
            'language': self.whisper_params.get('language', 'ja'),
            'temperature': 0.0,
            'beam_size': 3,  # Reduced from 5 for faster fallback
            'fp16': torch.cuda.is_available(),
            'verbose': None
        }

    def _process_segments(self, raw_segments: List[Dict], start_sec: float) -> List[Dict]:
        segments: List[Dict] = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            if any(suppress in text for suppress in self.suppress_high):
                logger.debug(f"Filtered segment due to suppression word: {text[:30]}...")
                continue

            avg_logprob = seg.get("avg_logprob", 0.0)
            for suppress_word in self.suppress_low:
                if suppress_word in text:
                    avg_logprob -= 0.15

            segment_duration = max(0.0, float(seg.get("end", 0.0) - seg.get("start", 0.0)))
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

            start_value = float(seg.get("start", 0.0)) + start_sec
            end_value = float(seg.get("end", 0.0)) + start_sec
            adjusted_seg = {
                "start": start_value,
                "end": end_value,
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