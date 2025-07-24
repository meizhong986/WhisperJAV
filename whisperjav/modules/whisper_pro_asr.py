#!/usr/bin/env python3
"""
WhisperPro ASR wrapper with internal VAD processing, refactored for the V3 architecture.
This module uses the standard whisper library and is intended for pipelines like 'balanced'.
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


class WhisperProASR:
    """Whisper ASR with internal VAD, using structured v3 parameters."""
    
    def __init__(self, model_config: Dict, params: Dict, task: str):
        """
        Initializes WhisperPro ASR with structured v3 config parameters.

        Args:
            model_config: The 'model' section from the resolved config.
            params: The 'params' section containing decoder, vad, and provider settings.
            task: The ASR task to perform ('transcribe' or 'translate').
        """
        # --- V3 PARAMETER UNPACKING ---
        self.model_name = model_config.get("model_name", "large-v2")
        self.device = model_config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        
        decoder_params = params["decoder"]
        vad_params = params["vad"]
        provider_params = params["provider"]

        # VAD parameters
        self.vad_threshold = vad_params.get("threshold", 0.4)
        self.min_speech_duration_ms = vad_params.get("min_speech_duration_ms", 150)
        self.vad_chunk_threshold = vad_params.get("chunk_threshold", 4.0)

        # CORRECT: Direct assignment - trusting pre-structured parameters
        self.transcribe_options = provider_params
        self.decode_options = decoder_params
        self.decode_options['task'] = task
        self.decode_options['language'] = 'ja'
        # --- END V3 PARAMETER UNPACKING ---

        # Suppression lists for Japanese content (business logic preserved)
        self.suppress_low = ["Thank you", "視聴", "Thanks for"]
        self.suppress_high = ["視聴ありがとうございました", "ご視聴ありがとうございました", 
                              "字幕作成者", "提供", "スポンサー"]
        
        self._initialize_models()
        self._log_sensitivity_parameters()
        
    def _initialize_models(self):
        """Initialize VAD and Whisper models."""
        logger.debug("Loading Silero VAD model...")
        try:
            # Preserved exactly from original file to ensure correct VAD version
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:v3.1",
                model="silero_vad",
                force_reload=True,
                onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = self.vad_utils
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
            raise

        logger.debug(f"Loading Whisper model: {self.model_name} on device: {self.device}")
        self.whisper_model = whisper.load_model(self.model_name, device=self.device)

    def _log_sensitivity_parameters(self):
        """Log the sensitivity-related parameters for debugging."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- WhisperProASR Initialized with V3 Parameters ---")
            logger.debug(f"  VAD Threshold: {self.vad_threshold}")
            logger.debug(f"  Min Speech Duration: {self.min_speech_duration_ms}ms")
            logger.debug(f"  Decoder Params: {self.decode_options}")
            logger.debug(f"  Provider (Whisper) Params: {self.transcribe_options}")
            logger.debug("----------------------------------------------------")

    def _prepare_transcribe_options(self) -> Tuple[Dict, Dict]:
        """Prepares and validates transcription options for Whisper."""
        # CORRECT: Simple copy - trusting pre-structured inputs
        transcribe_params = self.transcribe_options.copy()
        decode_params = self.decode_options.copy()
        
        # Only minimal type conversion
        if 'temperature' in transcribe_params and isinstance(transcribe_params.get('temperature'), list):
            transcribe_params['temperature'] = tuple(transcribe_params['temperature'])
            
        decode_params['fp16'] = self.device == 'cuda'
        
        return transcribe_params, decode_params

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """Transcribe audio file with internal VAD processing."""
        audio_path = Path(audio_path)
        logger.debug(f"Transcribing with VAD: {audio_path.name}")
        
        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
        except Exception as e:
            logger.error(f"Failed to read audio file {audio_path}: {e}")
            raise

        vad_segments = self._run_vad_on_audio(audio_data, sample_rate)
        
        if not vad_segments:
            logger.debug(f"No speech detected in {audio_path.name}")
            return {"segments": [], "text": "", "language": self.decode_options.get('language', 'ja')}
        
        all_segments = []
        for vad_group in vad_segments:
            segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group)
            all_segments.extend(segments)
            
        return {
            "segments": all_segments,
            "text": " ".join(seg["text"] for seg in all_segments),
            "language": self.decode_options.get('language', 'ja')
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
        Transcribe a group of VAD segments.
        Preserves all original business logic including suppression and fallback.
        """
        if not vad_group:
            return []
            
        start_sec = vad_group[0]["start_sec"]
        end_sec = vad_group[-1]["end_sec"]
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        group_audio = audio_data[start_sample:end_sample]
        
        # Get transcription parameters
        transcribe_params, decode_params = self._prepare_transcribe_options()
        
        try:
            result = self.whisper_model.transcribe(
                group_audio,
                verbose=None,
                **transcribe_params,
                **decode_params
            )
        except Exception as e:
            logger.error(f"Transcription of a VAD group failed: {e}", exc_info=True)
            
            # Fallback to minimal parameters (preserved from original)
            try:
                minimal_params = {
                    'task': self.decode_options.get('task', 'transcribe'),
                    'language': self.decode_options.get('language', 'ja'),
                    'temperature': 0.0,
                    'beam_size': 5,
                    'fp16': torch.cuda.is_available(),
                    'verbose': None
                }
                logger.warning(f"Retrying with minimal parameters")
                result = self.whisper_model.transcribe(group_audio, **minimal_params)
            except Exception as e2:
                logger.error(f"Even minimal transcribe failed: {e2}")
                return []
                    
        if not result or not result["segments"]:
            return []
            
        # Adjust timestamps and apply suppression (preserved from original)
        segments = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            
            # Apply high suppression list filtering
            if any(suppress in text for suppress in self.suppress_high):
                logger.debug(f"Filtered segment due to suppression word: {text[:30]}...")
                continue
            
            # Apply logprob filtering with suppression penalties
            avg_logprob = seg.get("avg_logprob", 0)
            
            # Apply text-based suppression penalties
            for suppress_word in self.suppress_low:
                if suppress_word in text: 
                    avg_logprob -= 0.15
            
            # Check against logprob threshold
            logprob_filter = self.transcribe_options.get("logprob_threshold", -1.0)
            if avg_logprob < logprob_filter:
                logger.debug(f"Filtered segment with logprob {avg_logprob}: {text[:50]}...")
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