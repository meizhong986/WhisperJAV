#!/usr/bin/env python3
"""Stable-ts ASR wrapper for WhisperJAV - Enhanced with parameter validation."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import stable_whisper
from dataclasses import dataclass
import traceback
import logging

from whisperjav.utils.logger import logger

# This dataclass is unchanged.
@dataclass
class LexicalSets:
    """Japanese lexical sets for subtitle processing."""
    sentence_endings: List[str] = None
    polite_forms: List[str] = None
    verb_continuations: List[str] = None
    noun_phrases: List[str] = None
    vocal_extensions: List[str] = None
    initial_fillers: List[str] = None
    non_lexical_sounds: List[str] = None

    def __post_init__(self):
        self.sentence_endings = self.sentence_endings or ['よ', 'ね', 'わ', 'の', 'ぞ', 'ぜ', 'な', 'か', 'かしら', 'かな']
        self.polite_forms = self.polite_forms or ['ます', 'です', 'ました', 'でしょう', 'ましょう', 'ません']
        self.verb_continuations = self.verb_continuations or ['て', 'で', 'た', 'ず', 'ちゃう', 'じまう']
        self.noun_phrases = self.noun_phrases or ['の', 'こと', 'もの', 'とき', 'ところ']
        self.vocal_extensions = self.vocal_extensions or ['~', 'ー', 'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ']
        self.initial_fillers = self.initial_fillers or ['あの', 'えっと', 'えー', 'まあ', 'なんか']
        self.non_lexical_sounds = self.non_lexical_sounds or ['ああ', 'う', 'ん', 'はぁ', 'ふぅ', 'くっ']


class StableTSASR:
    """Stable-ts ASR wrapper supporting both standard and turbo modes with parameter validation."""
    
    # Define which parameters are valid for each backend
    STANDARD_WHISPER_PARAMS = {
        # Transcribe parameters
        'temperature', 'compression_ratio_threshold', 'logprob_threshold',
        'no_speech_threshold', 'condition_on_previous_text', 'initial_prompt',
        'carry_initial_prompt', 'word_timestamps', 'prepend_punctuations',
        'append_punctuations', 'clip_timestamps', 'hallucination_silence_threshold',
        # Decode options
        'task', 'language', 'sample_len', 'best_of', 'beam_size', 'patience',
        'length_penalty', 'prompt', 'prefix', 'suppress_tokens', 'suppress_blank',
        'without_timestamps', 'max_initial_timestamp', 'fp16',
        # Stable-ts specific
        'regroup', 'vad', 'vad_threshold'
    }
    
    FASTER_WHISPER_PARAMS = {
        # Core parameters that faster_whisper accepts
        'task', 'language', 'temperature', 'beam_size', 'best_of', 'patience',
        'length_penalty', 'compression_ratio_threshold', 'no_speech_threshold',
        'condition_on_previous_text', 'initial_prompt', 'prefix',
        'suppress_blank', 'suppress_tokens', 'without_timestamps',
        'max_initial_timestamp', 'word_timestamps',
        # VAD parameters
        'vad', 'vad_threshold',
        # Stable-ts specific
        'regroup',
        # Faster-whisper specific
        'repetition_penalty', 'no_repeat_ngram_size'
    }
    
    def __init__(self,
                 model_load_params: Dict,
                 transcribe_options: Dict,
                 decode_options: Dict,
                 stable_ts_options: Dict,
                 turbo_mode: bool = False):
        """
        Initialize Stable-ts ASR with structured parameter dictionaries.
        """
        # Unpack model loading parameters
        self.model_name = model_load_params.get("model_name", "large-v2")
        self.device = model_load_params.get("device", "cuda")
        self.turbo_mode = turbo_mode
        
        # Store the options for use in the transcribe call
        self.transcribe_options = transcribe_options
        self.decode_options = decode_options
        self.stable_ts_options = stable_ts_options

        self.lexical_sets = LexicalSets()
        
        # Load model
        logger.info(f"Loading Stable-TS model: {self.model_name}")
        if self.turbo_mode:
            # Use faster_whisper backend
            compute_type = model_load_params.get("compute_type", "float16")
            self.model = stable_whisper.load_faster_whisper(self.model_name, device=self.device, compute_type=compute_type)
        else:
            # Use standard whisper backend
            self.model = stable_whisper.load_model(self.model_name, device=self.device)
            
        # Log active sensitivity parameters
        self._log_sensitivity_parameters()
    
    def _log_sensitivity_parameters(self):
        """Log the sensitivity-related parameters for debugging."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("StableTSASR Sensitivity Parameters:")
            logger.debug(f"  Turbo Mode: {self.turbo_mode}")
            logger.debug(f"  Temperature: {self.transcribe_options.get('temperature', 'default')}")
            logger.debug(f"  Beam Size: {self.decode_options.get('beam_size', 'default')}")
            logger.debug(f"  Best Of: {self.decode_options.get('best_of', 'default')}")
            logger.debug(f"  Patience: {self.decode_options.get('patience', 'default')}")
            logger.debug(f"  Length Penalty: {self.decode_options.get('length_penalty', 'default')}")
            logger.debug(f"  No Speech Threshold: {self.transcribe_options.get('no_speech_threshold', 'default')}")
            logger.debug(f"  Compression Ratio Threshold: {self.transcribe_options.get('compression_ratio_threshold', 'default')}")
            if not self.turbo_mode:
                logger.debug(f"  Logprob Threshold: {self.transcribe_options.get('logprob_threshold', 'default')}")
    
    def _convert_parameter_types(self, params: Dict) -> Dict:
        """Convert parameter types to match expected formats."""
        converted = params.copy()
        
        # Convert temperature from list to tuple if needed
        if 'temperature' in converted:
            temp = converted['temperature']
            if isinstance(temp, list):
                converted['temperature'] = tuple(temp)
            elif isinstance(temp, (int, float)):
                converted['temperature'] = (float(temp),)
        
        # Ensure suppress_tokens is properly formatted
        if 'suppress_tokens' in converted:
            tokens = converted['suppress_tokens']
            if tokens == "-1":
                # Keep as is - this means use default suppression
                pass
            elif isinstance(tokens, list):
                # Keep as list of integers
                pass
            elif tokens is None:
                # Remove if None
                del converted['suppress_tokens']
        
        return converted
    
    def _filter_parameters_for_backend(self, params: Dict) -> Dict:
        """Filter parameters based on which backend is being used."""
        if self.turbo_mode:
            valid_params = self.FASTER_WHISPER_PARAMS
            backend_name = "faster_whisper"
        else:
            valid_params = self.STANDARD_WHISPER_PARAMS
            backend_name = "standard_whisper"
        
        # Filter out invalid parameters
        filtered = {}
        removed = []
        
        for key, value in params.items():
            if key in valid_params:
                filtered[key] = value
            else:
                removed.append(key)
        
        if removed and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Removed {len(removed)} parameters not supported by {backend_name}: {removed}")
        
        return filtered
    
    def _prepare_transcribe_parameters(self, **kwargs) -> Dict:
        """Prepare all parameters for transcription with proper validation and conversion."""
        # Start with transcribe options
        params = self.transcribe_options.copy()
        
        # Merge decode options
        params.update(self.decode_options)
        
        # Add stable-ts specific options
        if self.stable_ts_options:
            params.update(self.stable_ts_options)
        
        # Add runtime parameters (like task)
        params.update(kwargs)
        
        # Convert parameter types
        params = self._convert_parameter_types(params)
        
        # Filter based on backend
        params = self._filter_parameters_for_backend(params)
        
        # Special handling for turbo mode
        if self.turbo_mode:
            # Ensure VAD is enabled for turbo mode
            if 'vad' not in params:
                params['vad'] = True
            if 'vad_threshold' not in params:
                params['vad_threshold'] = 0.2
            
            # Cap beam_size if too high
            if 'beam_size' in params and params['beam_size'] > 5:
                logger.debug(f"Turbo mode: Capping beam_size from {params['beam_size']} to 5")
                params['beam_size'] = 5
        
        return params
    
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> stable_whisper.WhisperResult:
        """Transcribe audio file with full parameter validation and backend compatibility."""
        audio_path = Path(audio_path)
        logger.info(f"Transcribing: {audio_path.name}")
        
        # Prepare and validate all parameters
        final_params = self._prepare_transcribe_parameters(**kwargs)
        
        # Log final parameters if debug enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Backend: {'faster_whisper' if self.turbo_mode else 'standard_whisper'}")
            logger.debug(f"Final transcription parameters: {final_params}")
        
        try:
            result = self.model.transcribe(str(audio_path), **final_params)
        except TypeError as e:
            # If we still get a parameter error, log details and retry with minimal params
            logger.error(f"Parameter error: {e}")
            logger.debug(f"Failed parameters: {final_params}")
            
            # Fallback to minimal parameters
            minimal_params = {
                'task': final_params.get('task', 'transcribe'),
                'language': final_params.get('language', 'ja'),
                'temperature': final_params.get('temperature', (0.0,)),
                'beam_size': final_params.get('beam_size', 5)
            }
            
            logger.warning(f"Retrying with minimal parameters: {minimal_params}")
            result = self.model.transcribe(str(audio_path), **minimal_params)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            logger.debug(traceback.format_exc())
            raise
            
        # Apply post-processing
        self._postprocess(result)
        
        return result
            
    def transcribe_to_srt(self, 
                         audio_path: Union[str, Path], 
                         output_srt_path: Union[str, Path],
                         **kwargs) -> Path:
        """Transcribe audio and save as SRT file."""
        result = self.transcribe(audio_path, **kwargs)
        
        final_output_path = Path(output_srt_path)
        self._save_to_srt(result, final_output_path)
        return final_output_path
    
    def _postprocess(self, result: stable_whisper.WhisperResult):
        """Apply Japanese-specific post-processing to transcription result."""
        logger.debug("Applying Japanese post-processing")
        
        # Check if we have any segments to process
        if not result.segments:
            logger.warning("No segments found in transcription result")
            return
            
        try:
            # Phase 1: Structural Splitting
            logger.debug("Phase 1: Structural Splitting")
            result.split_by_punctuation(['。', '？', '！', '…', '．'], lock=True)
            result.split_by_gap(0.75, lock=True)
        except Exception as e:
            logger.error(f"Error in Phase 1 (Structural Splitting): {e}")
            logger.debug(traceback.format_exc())

        try:
            # Phase 2: Linguistic Locking
            logger.debug("Phase 2: Linguistic Locking")
            result.lock(endswith=self.lexical_sets.sentence_endings, left=True, right=False)
            result.lock(endswith=self.lexical_sets.polite_forms, left=False, right=True)
            result.lock(endswith=self.lexical_sets.verb_continuations, left=True, right=False)
            result.lock(endswith=self.lexical_sets.noun_phrases, right=True)
            result.lock(startswith=['が', 'を', 'に', 'で', 'と'], left=True, right=False)
        except Exception as e:
            logger.error(f"Error in Phase 2 (Linguistic Locking): {e}")
            logger.debug(traceback.format_exc())

        try:
            # Phase 3: Content-Specific Handling
            logger.debug("Phase 3: Content-Specific Handling")
            result.lock(endswith=self.lexical_sets.vocal_extensions, right=False)
            result.lock(startswith=self.lexical_sets.initial_fillers, right=True)
        except Exception as e:
            logger.error(f"Error in Phase 3 (Content-Specific Handling): {e}")
            logger.debug(traceback.format_exc())

        # Phase 5: Cleanup
        result.merge_by_gap(min_gap=0.15, max_words=3, max_chars=12)
        result.merge_by_punctuation(punctuation=['、', 'けど', 'でも'],
                                    max_words=5,
                                    max_chars=18)
        
    def _save_to_srt(self, result: stable_whisper.WhisperResult, output_path: Path):
        """Save transcription result to SRT file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving SRT to: {output_path}")
        
        result.to_srt_vtt(str(output_path),
                          word_level=False,
                          segment_level=True,
                          strip=True)