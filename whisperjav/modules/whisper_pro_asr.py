#!/usr/bin/env python3
"""WhisperPro ASR wrapper with internal VAD processing for WhisperJAV."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import torch
import whisper
import soundfile as sf
import numpy as np
import srt
import datetime
from tqdm import tqdm
import traceback

from whisperjav.utils.logger import logger


class WhisperProASR:
    """Whisper ASR with internal VAD processing for enhanced accuracy."""
    
    def __init__(self,
                 model_name: str = "turbo",
                 language: str = "ja",
                 vad_threshold: float = 0.3,
                 vad_chunk_threshold: float = 4.0,
                 device: Optional[str] = "cuda",
                 max_attempts: int = 1,
                 temperature: Union[float, Tuple[float, ...]] = (0.0, 0.4)):
        """
        Initialize WhisperPro ASR with VAD.
        
        Args:
            model_name: Whisper model size
            language: Target language
            vad_threshold: VAD activation threshold
            vad_chunk_threshold: Threshold for grouping VAD chunks (seconds)
            device: Device to use (cuda/cpu)
            max_attempts: Maximum transcription attempts per chunk
            temperature: Temperature for sampling
        """
        self.model_name = model_name
        self.language = language
        self.vad_threshold = vad_threshold
        self.vad_chunk_threshold = vad_chunk_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_attempts = max_attempts
        self.temperature = temperature
        
        # Suppression lists for Japanese content
        self.suppress_low = ["Thank you", "視聴", "Thanks for"]
        self.suppress_high = ["視聴ありがとうございました", "ご視聴ありがとうございました", 
                              "字幕作成者", "提供", "スポンサー"]
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize VAD and Whisper models."""
        # Load VAD model
        logger.info("Loading Silero VAD model...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False
        )
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.whisper_model = whisper.load_model(self.model_name, device=self.device)


    def transcribe_to_srt(self, 
                         audio_path: Union[str, Path], 
                         output_srt_path: Union[str, Path],
                         **kwargs) -> Path:
        """
        Transcribe audio and save as SRT file.
        
        Args:
            audio_path: Path to audio file
            output_srt_path: Path for output SRT file
            **kwargs: Additional parameters, including 'task'
            
        Returns:
            Path to saved SRT file
        """
        audio_path = Path(audio_path)
        output_srt_path = Path(output_srt_path)
        
        # Transcribe, passing kwargs along
        result = self.transcribe(audio_path, **kwargs)
        
        # Convert to SRT format
        srt_subs = []
        for idx, segment in enumerate(result["segments"], 1):
            sub = srt.Subtitle(
                index=idx,
                start=datetime.timedelta(seconds=segment["start"]),
                end=datetime.timedelta(seconds=segment["end"]),
                content=segment["text"].strip()
            )
            srt_subs.append(sub)
        
        # Save SRT
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_subs))
            
        logger.info(f"Saved SRT to: {output_srt_path}")
        return output_srt_path
        


    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:
        """
        Transcribe audio file with internal VAD processing.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional transcription parameters, including 'task'
            
        Returns:
            Transcription result dictionary
        """
        audio_path = Path(audio_path)
        logger.info(f"Transcribing with VAD: {audio_path.name}")
        
        # Extract task from kwargs, default to transcribe
        task = kwargs.get('task', 'transcribe')
        
        # Load audio
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Run VAD
        vad_segments = self._run_vad_on_audio(audio_data, sample_rate)
        
        if not vad_segments:
            logger.warning(f"No speech detected in {audio_path.name}")
            return {"segments": [], "text": "", "language": self.language}
        
        # Transcribe VAD segments, passing the task along
        all_segments = []
        for vad_group in vad_segments:
            segments = self._transcribe_vad_group(audio_data, sample_rate, vad_group, task=task)
            all_segments.extend(segments)
            
        return {
            "segments": all_segments,
            "text": " ".join(seg["text"] for seg in all_segments),
            "language": self.language
        }
    

    
    def _run_vad_on_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[List[Dict]]:
        """
        Run VAD on audio data and return grouped segments.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of audio
            
        Returns:
            List of VAD segment groups
        """
        get_speech_timestamps, _, _, _, _ = self.vad_utils
        
        # Resample to 16kHz if needed (VAD requirement)
        VAD_SR = 16000
        if sample_rate != VAD_SR:
            # Simple resampling - for production use librosa or torchaudio
            resample_ratio = VAD_SR / sample_rate
            resampled_length = int(len(audio_data) * resample_ratio)
            indices = np.linspace(0, len(audio_data) - 1, resampled_length).astype(int)
            audio_16k = audio_data[indices]
        else:
            audio_16k = audio_data
            
        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio_16k)
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=VAD_SR,
            threshold=self.vad_threshold
        )
        
        if not speech_timestamps:
            return []
        
        # Expand timestamps (add head/tail room)
        for i in range(len(speech_timestamps)):
            speech_timestamps[i]["start"] = max(0, speech_timestamps[i]["start"] - 3200)  # 0.2s
            speech_timestamps[i]["end"] = min(len(audio_tensor) - 16, 
                                             speech_timestamps[i]["end"] + 20800)  # 1.3s
            # Remove overlap
            if i > 0 and speech_timestamps[i]["start"] < speech_timestamps[i - 1]["end"]:
                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]
        
        # Group timestamps by silence threshold
        groups = [[]]
        for i in range(len(speech_timestamps)):
            if (i > 0 and 
                speech_timestamps[i]["start"] > speech_timestamps[i - 1]["end"] + 
                (self.vad_chunk_threshold * VAD_SR)):
                groups.append([])
            groups[-1].append(speech_timestamps[i])
            
        # Convert sample indices to seconds
        for group in groups:
            for seg in group:
                seg["start_sec"] = seg["start"] / VAD_SR
                seg["end_sec"] = seg["end"] / VAD_SR
                
        return groups
    
    def _transcribe_vad_group(self, audio_data: np.ndarray, sample_rate: int, 
                             vad_group: List[Dict], task: str) -> List[Dict]:
        """
        Transcribe a group of VAD segments.
        
        Args:
            audio_data: Full audio data
            sample_rate: Sample rate
            vad_group: Group of VAD segments
            task: The task for Whisper ('transcribe' or 'translate')
            
        Returns:
            List of transcription segments
        """
        if not vad_group:
            return []
            
        # Get audio slice for this group
        start_sec = vad_group[0]["start_sec"]
        end_sec = vad_group[-1]["end_sec"]
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        
        group_audio = audio_data[start_sample:end_sample]
        
        # Prepare transcription options, now using the passed 'task'
        options = {
            "language": self.language,
            "task": task,
            "temperature": self.temperature,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.65,
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "best_of": 2,
            "beam_size": 2,
            "patience": 2.0,
            "suppress_blank": True,
            "suppress_tokens": "-1",
            "max_initial_timestamp": 1.0,
            "fp16": torch.cuda.is_available()
        }
        
        # Transcribe with retry logic
        result = None
        for attempt in range(self.max_attempts):
            try:
                result = self.whisper_model.transcribe(group_audio, **options)
                
                if (result["segments"] and 
                    result["segments"][-1]["end"] < (end_sec - start_sec) + 10.0):
                    break
                elif attempt + 1 < self.max_attempts:
                    logger.info(f"Retrying transcription (attempt {attempt + 2}/{self.max_attempts})")
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                if attempt + 1 >= self.max_attempts:
                    return []
                    
        if not result or not result["segments"]:
            return []
            
        # Adjust timestamps and apply suppression
        segments = []
        for seg in result["segments"]:
            text = seg["text"]
            avg_logprob = seg.get("avg_logprob", 0)
            
            for suppress_word in self.suppress_low:
                if suppress_word in text:
                    avg_logprob -= 0.15
                    
            for suppress_word in self.suppress_high:
                if suppress_word in text:
                    avg_logprob -= 0.35
                    
            if avg_logprob < -1.0 or seg.get("no_speech_prob", 0) > 0.7:
                continue
                
            adjusted_seg = {
                "start": seg["start"] + start_sec,
                "end": seg["end"] + start_sec,
                "text": text.strip(),
                "avg_logprob": avg_logprob
            }
            segments.append(adjusted_seg)
            
        return segments
