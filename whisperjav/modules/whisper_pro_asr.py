#!/usr/bin/env python3

"""WhisperPro ASR wrapper with parameter validation for standard Whisper."""



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

import logging



from whisperjav.utils.logger import logger





class WhisperProASR:

    """Whisper ASR with internal VAD processing and parameter validation."""

    

    # Define valid parameters for standard Whisper transcribe()

    WHISPER_TRANSCRIBE_PARAMS = {

        'verbose', 'temperature', 'compression_ratio_threshold', 'logprob_threshold',

        'no_speech_threshold', 'condition_on_previous_text', 'initial_prompt',

        'carry_initial_prompt', 'word_timestamps', 'prepend_punctuations',

        'append_punctuations', 'clip_timestamps', 'hallucination_silence_threshold'

    }

    

    # Define valid DecodingOptions parameters

    WHISPER_DECODE_PARAMS = {

        'task', 'language', 'sample_len', 'best_of', 'beam_size', 'patience',

        'length_penalty', 'prompt', 'prefix', 'suppress_tokens', 'suppress_blank',

        'without_timestamps', 'max_initial_timestamp', 'fp16'

    }

    

    def __init__(self,

                 model_load_params: Dict,

                 vad_options: Dict,

                 transcribe_options: Dict,

                 decode_options: Dict):

        """

        Initialize WhisperPro ASR with VAD using structured parameter dictionaries.

        """

        # Unpack model loading parameters

        self.model_name = model_load_params.get("model_name", "large-v2")

        self.device = model_load_params.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        

        # Unpack VAD options

        self.vad_threshold = vad_options.get("threshold", 0.4)

        self.min_speech_duration_ms = vad_options.get("min_speech_duration_ms", 150)

        self.vad_chunk_threshold = vad_options.get("chunk_threshold", 4.0)



        # Store the high-level and low-level options for use in the transcribe call

        self.transcribe_options = transcribe_options

        self.decode_options = decode_options



        # For backward compatibility with existing internal logic if needed

        self.language = self.decode_options.get("language", "ja")

        

        # Suppression lists for Japanese content

        self.suppress_low = ["Thank you", "視聴", "Thanks for"]

        self.suppress_high = ["視聴ありがとうございました", "ご視聴ありがとうございました", 

                              "字幕作成者", "提供", "スポンサー"]

        

        # Process suppress_tokens from decode options

        self._process_suppress_tokens()

        

        # Initialize models

        self._initialize_models()

        

        # Log active sensitivity parameters

        self._log_sensitivity_parameters()

        

    def _process_suppress_tokens(self):

        """Process and merge suppress_tokens with existing suppression lists."""

        suppress_tokens = self.decode_options.get('suppress_tokens', [])

        

        if suppress_tokens and suppress_tokens != "-1":

            # Convert token IDs to a set for efficient lookup

            if isinstance(suppress_tokens, list):

                self.suppress_token_ids = set(suppress_tokens)

            else:

                self.suppress_token_ids = set()

            

            logger.debug(f"Using suppress_tokens: {self.suppress_token_ids}")

        else:

            self.suppress_token_ids = set()

    

    def _log_sensitivity_parameters(self):

        """Log the sensitivity-related parameters for debugging."""

        if logger.isEnabledFor(logging.DEBUG):

            logger.debug("WhisperProASR Sensitivity Parameters:")

            logger.debug(f"  VAD Threshold: {self.vad_threshold}")

            logger.debug(f"  Min Speech Duration: {self.min_speech_duration_ms}ms")

            logger.debug(f"  Temperature: {self.transcribe_options.get('temperature', 'default')}")

            logger.debug(f"  Beam Size: {self.decode_options.get('beam_size', 'default')}")

            logger.debug(f"  Best Of: {self.decode_options.get('best_of', 'default')}")

            logger.debug(f"  Patience: {self.decode_options.get('patience', 'default')}")

            logger.debug(f"  Length Penalty: {self.decode_options.get('length_penalty', 'default')}")

            logger.debug(f"  No Speech Threshold: {self.transcribe_options.get('no_speech_threshold', 'default')}")

            logger.debug(f"  Compression Ratio Threshold: {self.transcribe_options.get('compression_ratio_threshold', 'default')}")

            logger.debug(f"  Logprob Threshold: {self.transcribe_options.get('logprob_threshold', 'default')}")

            logger.debug(f"  Suppress Tokens: {len(self.suppress_token_ids)} tokens")

        

    def _initialize_models(self):

        """Initialize VAD and Whisper models."""

        # Load VAD model

        logger.debug("Loading Silero VAD model...")

        self.vad_model, self.vad_utils = torch.hub.load(

            repo_or_dir="snakers4/silero-vad:v3.1",

            model="silero_vad",

            force_reload=True,

            onnx=False

        )

        

        # Load Whisper model

        logger.debug(f"Loading Whisper model: {self.model_name}")

        self.whisper_model = whisper.load_model(self.model_name, device=self.device)



    def _convert_parameter_types(self, params: Dict) -> Dict:

        """Convert parameter types to match Whisper's expected formats."""

        converted = params.copy()

        

        # Convert temperature from list to tuple if needed

        if 'temperature' in converted:

            temp = converted['temperature']

            if isinstance(temp, list):

                converted['temperature'] = tuple(temp)

            elif isinstance(temp, (int, float)):

                converted['temperature'] = float(temp)

        

        # Ensure fp16 is boolean

        if 'fp16' in converted:

            converted['fp16'] = bool(converted['fp16'])

        

        # Handle suppress_tokens

        if 'suppress_tokens' in converted:

            tokens = converted['suppress_tokens']

            if tokens is None or tokens == "":

                # Empty or None - remove it to use default

                del converted['suppress_tokens']

            elif tokens == "-1":

                # Keep as is - use default suppression

                pass

            elif isinstance(tokens, str) and tokens:

                # Try to parse as comma-separated integers

                try:

                    converted['suppress_tokens'] = [int(t.strip()) for t in tokens.split(',')]

                except:

                    logger.warning(f"Could not parse suppress_tokens: {tokens}")

                    del converted['suppress_tokens']

        

        return converted



    def _separate_parameters(self, all_params: Dict) -> Tuple[Dict, Dict]:

        """Separate parameters into transcribe parameters and decode options."""

        transcribe_params = {}

        decode_params = {}

        unknown_params = []

        

        for key, value in all_params.items():

            if key in self.WHISPER_TRANSCRIBE_PARAMS:

                transcribe_params[key] = value

            elif key in self.WHISPER_DECODE_PARAMS:

                decode_params[key] = value

            else:

                unknown_params.append(key)

        

        if unknown_params and logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"Unknown parameters will be passed as decode_options: {unknown_params}")

            # Add unknown params to decode_params as Whisper might accept them

            for key in unknown_params:

                decode_params[key] = all_params[key]

        

        return transcribe_params, decode_params



    def transcribe_to_srt(self, 

                         audio_path: Union[str, Path], 

                         output_srt_path: Union[str, Path],

                         **kwargs) -> Path:

        """

        Transcribe audio and save as SRT file.

        """

        audio_path = Path(audio_path)

        output_srt_path = Path(output_srt_path)

        

        result = self.transcribe(audio_path, **kwargs)

        

        srt_subs = []

        for idx, segment in enumerate(result["segments"], 1):

            sub = srt.Subtitle(

                index=idx,

                start=datetime.timedelta(seconds=segment["start"]),

                end=datetime.timedelta(seconds=segment["end"]),

                content=segment["text"].strip()

            )

            srt_subs.append(sub)

        

        output_srt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_srt_path, 'w', encoding='utf-8') as f:

            f.write(srt.compose(srt_subs))

            

        logger.debug(f"Saved SRT to: {output_srt_path}")

        return output_srt_path



    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> Dict:

        """

        Transcribe audio file with internal VAD processing.

        """

        audio_path = Path(audio_path)

        logger.debug(f"Transcribing with VAD: {audio_path.name}")

        

        task = kwargs.get('task', 'transcribe')

        

        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

        if audio_data.ndim > 1:

            audio_data = np.mean(audio_data, axis=1)

            

        vad_segments = self._run_vad_on_audio(audio_data, sample_rate)

        

        if not vad_segments:

            logger.debug(f"No speech detected in {audio_path.name}")

            return {"segments": [], "text": "", "language": self.language}

        

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

        """

        get_speech_timestamps, _, _, _, _ = self.vad_utils

        

        VAD_SR = 16000

        if sample_rate != VAD_SR:

            resample_ratio = VAD_SR / sample_rate

            resampled_length = int(len(audio_data) * resample_ratio)

            indices = np.linspace(0, len(audio_data) - 1, resampled_length).astype(int)

            audio_16k = audio_data[indices]

        else:

            audio_16k = audio_data

            

        audio_tensor = torch.FloatTensor(audio_16k)

        

        speech_timestamps = get_speech_timestamps(

            audio_tensor,

            self.vad_model,

            sampling_rate=VAD_SR,

            threshold=self.vad_threshold,

            min_speech_duration_ms=self.min_speech_duration_ms

        )

        

        if not speech_timestamps:

            return []

        

        for i in range(len(speech_timestamps)):

            speech_timestamps[i]["start"] = max(0, speech_timestamps[i]["start"] - 3200)

            speech_timestamps[i]["end"] = min(len(audio_tensor) - 16, speech_timestamps[i]["end"] + 20800)

            if i > 0 and speech_timestamps[i]["start"] < speech_timestamps[i - 1]["end"]:

                speech_timestamps[i]["start"] = speech_timestamps[i - 1]["end"]

        

        groups = [[]]

        for i in range(len(speech_timestamps)):

            if (i > 0 and 

                speech_timestamps[i]["start"] > speech_timestamps[i - 1]["end"] + (self.vad_chunk_threshold * VAD_SR)):

                groups.append([])

            groups[-1].append(speech_timestamps[i])

            

        for group in groups:

            for seg in group:

                seg["start_sec"] = seg["start"] / VAD_SR

                seg["end_sec"] = seg["end"] / VAD_SR

                

        return groups



    

    def _prepare_transcribe_options(self, task: str) -> Tuple[Dict, Dict]:

        """Prepare and validate transcription options for Whisper."""

        

        # Remove incompatible parameters for newer Whisper versions

        INCOMPATIBLE_PARAMS = ['hallucination_silence_threshold', 'carry_initial_prompt']

        

        # Combine all parameters

        all_params = {}

        all_params.update(self.transcribe_options)

        all_params.update(self.decode_options)

        

        # Remove incompatible parameters

        for param in INCOMPATIBLE_PARAMS:

            if param in all_params:

                logger.debug(f"Removing incompatible parameter: {param}")

                all_params.pop(param, None)

        

        # Add runtime options

        all_params['task'] = task

        all_params['language'] = self.language

        all_params['fp16'] = torch.cuda.is_available()

        

        # Convert parameter types

        all_params = self._convert_parameter_types(all_params)

        

        # Separate into transcribe params and decode options

        transcribe_params, decode_params = self._separate_parameters(all_params)

        

        # Handle suppress_tokens merge

        if self.suppress_token_ids:

            existing_tokens = decode_params.get('suppress_tokens', [])

            if existing_tokens == "-1":

                # Use default suppression plus our custom tokens

                decode_params['suppress_tokens'] = list(self.suppress_token_ids)

            elif isinstance(existing_tokens, list):

                # Merge tokens

                decode_params['suppress_tokens'] = list(set(existing_tokens) | self.suppress_token_ids)

        

        # Log final options if debug enabled

        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"Transcribe parameters: {transcribe_params}")

            logger.debug(f"Decode options: {decode_params}")

        

        return transcribe_params, decode_params

    

    def _transcribe_vad_group(self, audio_data: np.ndarray, sample_rate: int, 
                             vad_group: List[Dict], task: str) -> List[Dict]:
        """
        Transcribe a group of VAD segments with validated parameters.
        """
        if not vad_group:
            return []
            
        start_sec = vad_group[0]["start_sec"]
        end_sec = vad_group[-1]["end_sec"]
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        group_audio = audio_data[start_sample:end_sample]
        
        # Get properly separated and validated parameters
        transcribe_params, decode_params = self._prepare_transcribe_options(task)
        
        # Force verbose=None to suppress progress bars
        transcribe_params['verbose'] = None
        
        try:
            # Call transcribe with properly separated parameters
            result = self.whisper_model.transcribe(
                group_audio,
                **transcribe_params,
                **decode_params
            )

        except TypeError as e:
            logger.error(f"Parameter error in transcribe: {e}")
            logger.debug(f"Transcribe params: {transcribe_params}")
            logger.debug(f"Decode params: {decode_params}")
            
            # Fallback to minimal parameters
            minimal_params = {
                'task': task,
                'language': self.language,
                'temperature': transcribe_params.get('temperature', 0.0),
                'beam_size': decode_params.get('beam_size', 5),
                'fp16': torch.cuda.is_available(),
                'verbose': None  # Suppress progress in fallback too
            }
            logger.warning(f"Retrying with minimal parameters: {minimal_params}")
            
            try:
                result = self.whisper_model.transcribe(group_audio, **minimal_params)
            except Exception as e2:
                logger.error(f"Even minimal transcribe failed: {e2}")
                return []

                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            logger.debug(traceback.format_exc())
            return []
                    
        if not result or not result["segments"]:
            return []
            
        # Adjust timestamps and apply text-based suppression
        segments = []
        for seg in result["segments"]:
            text = seg["text"]
            avg_logprob = seg.get("avg_logprob", 0)
            
            # Apply text-based suppression penalties
            for suppress_word in self.suppress_low:
                if suppress_word in text: 
                    avg_logprob -= 0.15
                    
            for suppress_word in self.suppress_high:
                if suppress_word in text: 
                    avg_logprob -= 0.35
                    
            # Check against logprob threshold
            logprob_filter = self.transcribe_options.get("logprob_threshold", -1.0)
            if avg_logprob < logprob_filter:
                logger.debug(f"Filtered segment with logprob {avg_logprob}: {text[:50]}...")
                continue
                
            adjusted_seg = {
                "start": seg["start"] + start_sec,
                "end": seg["end"] + start_sec,
                "text": text.strip(),
                "avg_logprob": avg_logprob
            }
            segments.append(adjusted_seg)
            
        return segments

