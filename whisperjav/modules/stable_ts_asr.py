#!/usr/bin/env python3
"""Stable-ts ASR wrapper for WhisperJAV."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import stable_whisper
from dataclasses import dataclass
import traceback

from whisperjav.utils.logger import logger

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
    """Stable-ts ASR wrapper supporting both standard and turbo modes."""
    
    def __init__(self,
                 model_name: str = "turbo",
                 language: str = "ja",
                 temperature: float = 0.0,
                 turbo_mode: bool = True,
                 device: Optional[str] = "cuda"):
        self.model_name = "turbo" if turbo_mode else model_name
        self.language = language
        self.temperature = temperature
        self.turbo_mode = turbo_mode
        self.device = device
        self.lexical_sets = LexicalSets()
        
        # Load model
        logger.info(f"Loading Stable-TS model: {self.model_name}")
        if turbo_mode:
            self.model = stable_whisper.load_faster_whisper(self.model_name, device=device)
        else:
            self.model = stable_whisper.load_model(self.model_name, device=device)
            
        # Default transcription parameters
        self.default_params = self._get_default_params()
        
    def _get_default_params(self) -> Dict:
        """Get default transcription parameters based on mode."""
        if self.turbo_mode:
            return {
                'regroup': True,
                'language': self.language,
                'task': 'transcribe',
                'temperature': (0.0, 0.4),
                'beam_size': 5,
                'best_of': 5,
                'patience': 1.0,
                'vad': True,
                'vad_threshold': 0.5,
                'condition_on_previous_text': True,
                'compression_ratio_threshold': 2.4,
                'no_speech_threshold': 0.6,
                'repetition_penalty': 1.0,
                'no_repeat_ngram_size': 0,
                'log_prob_threshold': -1.0,
            }
        else:
            return {
                'regroup': True,
                'language': self.language,
                'task': 'transcribe',
                'temperature': self.temperature,
                'beam_size': 5,
                'vad': True,
                'vad_threshold': 0.5,
                'condition_on_previous_text': True,
                'compression_ratio_threshold': 2.4,
                'no_speech_threshold': 0.6,
            }
    
    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> stable_whisper.WhisperResult:
        """Transcribe audio file and return result object."""
        audio_path = Path(audio_path)
        logger.info(f"Transcribing: {audio_path.name}")
        
        # Merge provided kwargs with defaults
        params = {**self.default_params, **kwargs}
        
        # Transcribe
        if self.turbo_mode:
            result = self.model.transcribe(str(audio_path), **params)
        else:
            result = self.model.transcribe(str(audio_path), **params)
            
        # Apply post-processing
        self._postprocess(result)
        
        return result
            
    def transcribe_to_srt(self, 
                         audio_path: Union[str, Path], 
                         output_srt_path: Union[str, Path],
                         **kwargs) -> Path:
        """Transcribe audio and save as SRT file."""
        # Get the task from kwargs to determine naming
        task = kwargs.get('task', 'transcribe')
        
        # Transcribe
        result = self.transcribe(audio_path, **kwargs)
        
        # Generate the proper output filename
        output_path = Path(output_srt_path)
        media_basename = output_path.stem.replace('_extracted', '')  # Remove _extracted suffix if present
        lang_code = "ja" if task == "transcribe" else "en"
        final_output_path = output_path.parent / f"{media_basename}.{lang_code}.whisperjav.srt"
        
        # Save to SRT
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

        '''
        try:
            # Phase 4: Length/Duration Splitting
            logger.debug("Phase 4: Length/Duration Splitting")
            MAX_CHARS_PER_SUBTITLE = 20
            MAX_DURATION_PER_SUBTITLE = 5.0
            result.split_by_length(MAX_CHARS_PER_SUBTITLE, include_lock=True, even_split=False)
            result.split_by_duration(MAX_DURATION_PER_SUBTITLE, include_lock=True)
        except Exception as e:
            logger.error(f"Error in Phase 4 (Length/Duration Splitting): {e}")
            logger.debug(traceback.format_exc())
        
        
        # Phase 5: Cleanup - FIX: Add try-except only for the problematic operation
        try:
            result.remove_words_by_str(words_to_remove=self.lexical_sets.non_lexical_sounds,
                                       case_sensitive=True,
                                       strip=True)
        except Exception as e:
            logger.warning(f"Could not remove non-lexical sounds: {e}")
            # Continue without this operation
        '''
        result.merge_by_gap(min_gap=0.15, max_words=3, max_chars=12)
        result.merge_by_punctuation(punctuation=['、', 'けど', 'でも'],
                                    max_words=5,
                                    max_chars=18)

        '''
        # Precision Timestamp Refinement
        logger.debug("Refining timestamps")
        result.refine(
            precision=0.05,
            rel_prob_cutoff=0.1,
            rel_word_ts_cutoff=0.25
        )
        '''
        

    def _save_to_srt(self, result: stable_whisper.WhisperResult, output_path: Path):
        """Save transcription result to SRT file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving SRT to: {output_path}")
        
        result.to_srt_vtt(str(output_path),
                          word_level=False,
                          segment_level=True,
                          strip=True)