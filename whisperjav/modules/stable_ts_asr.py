#!/usr/bin/env python3

"""Stable-ts ASR wrapper for WhisperJAV - Enhanced with parameter validation."""



from pathlib import Path

from typing import Dict, List, Optional, Union, Tuple

import stable_whisper

from dataclasses import dataclass

import traceback

import logging



import torch



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

        

        # Pre-cache the appropriate Silero VAD version (VAD is always on)

        if self.turbo_mode:

            # Faster pipeline: use latest version

            logger.debug("Pre-caching latest Silero VAD for faster pipeline...")

            vad_repo = "snakers4/silero-vad"

        else:

            # Fast pipeline: use v4.0

            logger.debug("Pre-caching Silero VAD v4.0 for fast pipeline...")

            vad_repo = "snakers4/silero-vad:v4.0"

        

        try:

            torch.hub.load(

                repo_or_dir=vad_repo,

                model="silero_vad",

                force_reload=True,

                onnx=False

            )

            logger.debug(f"Successfully pre-cached VAD from {vad_repo}")

        except Exception as e:

            logger.warning(f"Failed to pre-cache Silero VAD: {e}")

        

        # Load model

        logger.debug(f"Loading Stable-TS model: {self.model_name}")

        if self.turbo_mode:

            # Use faster_whisper backend

            compute_type = model_load_params.get("compute_type", "float16")

            self.model = stable_whisper.load_faster_whisper(self.model_name, device=self.device, compute_type=compute_type)

        else:

            # Use standard whisper backend

            self.model = stable_whisper.load_model(self.model_name, device=self.device)

            

        # Log active sensitivity parameters

        self._log_sensitivity_parameters()

        

    



    def _precache_silero_vad(self):

        """Pre-cache the appropriate Silero VAD version for stable-ts to use."""

        if hasattr(self, '_vad_precached') and self._vad_precached:

            logger.debug("VAD already pre-cached for this instance")

            return

            

        if self.turbo_mode:

            # Faster pipeline: use latest version

            logger.debug("Pre-caching latest Silero VAD for faster pipeline...")

            vad_repo = "snakers4/silero-vad"

        else:

            # Fast pipeline: use v4.0

            logger.debug("Pre-caching Silero VAD v4.0 for fast pipeline...")

            vad_repo = "snakers4/silero-vad:v4.0"

        

        try:

            # Force reload to ensure we get the specific version

            torch.hub.load(

                repo_or_dir=vad_repo,

                model="silero_vad",

                force_reload=True,

                onnx=False

            )

            self._vad_precached = True

            logger.debug(f"Successfully pre-cached VAD from {vad_repo}")

        except Exception as e:

            logger.warning(f"Failed to pre-cache Silero VAD: {e}")

                

                

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

        

        # Handle suppress_tokens properly for both backends

        if 'suppress_tokens' in converted:

            tokens = converted['suppress_tokens']

            if tokens == "-1":

                # Keep as is - this means use default suppression

                pass

            elif tokens == "" or tokens is None:

                # Empty string or None - remove it entirely

                del converted['suppress_tokens']

            elif isinstance(tokens, list):

                # Keep as list of integers

                pass

            elif isinstance(tokens, str) and tokens:

                # Try to parse as comma-separated integers

                try:

                    converted['suppress_tokens'] = [int(t.strip()) for t in tokens.split(',')]

                except:

                    logger.warning(f"Could not parse suppress_tokens: {tokens}, removing it")

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

        

        # Get the task to determine regroup setting

        task = params.get('task', 'transcribe')

        

        # Enable regroup only for translation (English output)

        # For transcription (Japanese), we use custom _postprocess instead

        if 'regroup' not in params:  # Don't override if explicitly set

            params['regroup'] = (task == 'translate')

            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Auto-setting regroup={params['regroup']} for task='{task}'")

        

        # Ensure VAD is enabled for both modes (since it's always on)

        if 'vad' not in params:

            params['vad'] = True

        if 'vad_threshold' not in params:

            params['vad_threshold'] = 0.35  # Default VAD threshold

        

        # Convert parameter types

        params = self._convert_parameter_types(params)

        

        # Filter based on backend

        params = self._filter_parameters_for_backend(params)

        

        # Special handling for turbo mode

        if self.turbo_mode:

            # Cap beam_size if too high

            if 'beam_size' in params and params['beam_size'] > 5:

                logger.debug(f"Turbo mode: Capping beam_size from {params['beam_size']} to 5")

                params['beam_size'] = 5

                

            return params

        else:

            # For original Whisper, we need to separate parameters

            return self._separate_whisper_parameters(params)

        

        



    def _separate_whisper_parameters(self, all_params: Dict) -> Dict:

        """Separate parameters for original Whisper backend."""

        

        # Ensure language is properly set

        if 'language' not in all_params:

            all_params['language'] = self.decode_options.get('language', 'ja')

        

        # Remove parameters that are incompatible with newer Whisper versions

        INCOMPATIBLE_PARAMS = ['hallucination_silence_threshold', 'carry_initial_prompt']

        for param in INCOMPATIBLE_PARAMS:

            if param in all_params:

                logger.debug(f"Removing incompatible parameter: {param}")

                all_params.pop(param, None)

        

        # Define what goes where

        DIRECT_TRANSCRIBE_PARAMS = {

            'verbose', 'temperature', 'compression_ratio_threshold', 

            'logprob_threshold', 'no_speech_threshold', 

            'condition_on_previous_text', 'initial_prompt',

            'word_timestamps', 'prepend_punctuations', 

            'append_punctuations', 'clip_timestamps',

            # stable-ts specific that go to transcribe

            'regroup', 'vad', 'vad_threshold', 'suppress_silence',

            'suppress_word_ts', 'min_word_dur', 'min_silence_dur',

            'denoiser', 'denoiser_options'

        }

        

        DECODE_OPTIONS_PARAMS = {

            'task', 'language', 'sample_len', 'best_of', 

            'beam_size', 'patience', 'length_penalty', 

            'prompt', 'prefix', 'suppress_tokens', 

            'suppress_blank', 'without_timestamps', 

            'max_initial_timestamp', 'fp16'

        }

        

        transcribe_params = {}

        decode_options = {}

        unknown_params = []

        

        for key, value in all_params.items():

            if key in DIRECT_TRANSCRIBE_PARAMS:

                transcribe_params[key] = value

            elif key in DECODE_OPTIONS_PARAMS:

                decode_options[key] = value

            else:

                unknown_params.append(key)

        

        if unknown_params and logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"Unknown parameters for original Whisper: {unknown_params}")

        

        # Add ignore_compatibility to avoid version warning

        transcribe_params['ignore_compatibility'] = True

        

        # Return as a special structure that transcribe() can understand

        return {

            'transcribe_params': transcribe_params,

            'decode_options': decode_options

        }





    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> stable_whisper.WhisperResult:

        """Transcribe audio file with full parameter validation and backend compatibility."""

        audio_path = Path(audio_path)

        logger.debug(f"Transcribing: {audio_path.name}")

        

        # Add debug logging for incoming kwargs

        logger.debug(f"Incoming kwargs: {kwargs}")

        

        # Prepare and validate all parameters

        final_params = self._prepare_transcribe_parameters(**kwargs)

        

        # Extract task parameter for later use

        task = kwargs.get('task', self.decode_options.get('task', 'transcribe'))



        

        # Enhanced debug logging

        if not self.turbo_mode:

            logger.debug("=== PARAMETER DEBUGGING ===")

            logger.debug(f"All transcribe_options: {self.transcribe_options}")

            logger.debug(f"All decode_options: {self.decode_options}")

            logger.debug(f"All stable_ts_options: {self.stable_ts_options}")

            

            transcribe_params = final_params.get('transcribe_params', {})

            decode_options = final_params.get('decode_options', {})

            

            # Check if hallucination_silence_threshold is in either dict

            if 'hallucination_silence_threshold' in transcribe_params:

                logger.debug(f"hallucination_silence_threshold found in transcribe_params: {transcribe_params['hallucination_silence_threshold']}")

            if 'hallucination_silence_threshold' in decode_options:

                logger.warning(f"WARNING: hallucination_silence_threshold found in decode_options! Value: {decode_options['hallucination_silence_threshold']}")

            

            logger.debug(f"Final transcribe_params keys: {list(transcribe_params.keys())}")

            logger.debug(f"Final decode_options keys: {list(decode_options.keys())}")

            logger.debug("=========================")

        

        # Log final parameters if debug enabled

        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"Backend: {'faster_whisper' if self.turbo_mode else 'standard_whisper'}")

            logger.debug(f"Task: {task}")

            if self.turbo_mode:

                logger.debug(f"Final transcription parameters: {final_params}")

            else:

                logger.debug(f"Transcribe params: {final_params['transcribe_params']}")

                logger.debug(f"Decode options: {final_params['decode_options']}")

        

        try:

            if self.turbo_mode:

                # Faster-whisper: all parameters together

                result = self.model.transcribe(str(audio_path), **final_params)

            else:

                # Original Whisper: separated parameters

                result = self.model.transcribe(

                    str(audio_path), 

                    **final_params['transcribe_params'],

                    **final_params['decode_options']

                )

        except TypeError as e:

            # If we still get a parameter error, log details and retry with minimal params

            logger.error(f"Parameter error: {e}")

            logger.debug(f"Failed parameters: {final_params}")

            

            # Fallback to minimal parameters

            minimal_params = {

                'task': task,  # Use the extracted task

                'language': self.decode_options.get('language', 'ja'),

                'temperature': self.transcribe_options.get('temperature', (0.0,)),

                'beam_size': self.decode_options.get('beam_size', 5)

            }

            

            # Only add ignore_compatibility for original whisper

            if not self.turbo_mode:

                minimal_params['ignore_compatibility'] = True

            

            logger.warning(f"Retrying with minimal parameters: {minimal_params}")

            

            if self.turbo_mode:

                result = self.model.transcribe(str(audio_path), **minimal_params)

            else:

                # For original whisper, split minimal params correctly

                result = self.model.transcribe(

                    str(audio_path),

                    temperature=minimal_params['temperature'],

                    ignore_compatibility=minimal_params['ignore_compatibility'],

                    task=minimal_params['task'],

                    language=minimal_params['language'],

                    beam_size=minimal_params['beam_size']

                )

                

        except Exception as e:

            logger.error(f"Transcription failed: {e}")

            logger.debug(traceback.format_exc())

            raise

        

        # Apply post-processing only for Japanese transcription

        if task == 'transcribe':

            self._postprocess(result)

        else:

            logger.debug(f"Skipping Japanese post-processing for task='{task}' (English output)")

        

        return result



            

    def transcribe_to_srt(self, 

                         audio_path: Union[str, Path], 

                         output_srt_path: Union[str, Path],

                         **kwargs) -> Path:

        """Transcribe audio and save as SRT file."""

        # Ensure task is passed through to transcribe method

        if 'task' not in kwargs:

            kwargs['task'] = self.decode_options.get('task', 'transcribe')

        

        result = self.transcribe(audio_path, **kwargs)

        

        final_output_path = Path(output_srt_path)

        self._save_to_srt(result, final_output_path)

        return final_output_path

        



    def _postprocess_japanese_dialogue(self,

                                        result: stable_whisper.WhisperResult, 

                                        preset: str = "default") -> stable_whisper.WhisperResult:

        """

        Applies a unified, multi-pass regrouping strategy to a stable-ts result

        for optimal Japanese conversational dialogue segmentation.



        Args:

            result: The WhisperResult object from a stable-ts transcription.

            preset: A preset strategy: "default", "high_moan", or "narrative".



        Returns:

            The modified WhisperResult object after post-processing.

        """

        

        

        logger.debug(f"Applying Japanese post-processing with '{preset}' preset.")

        

        if not result.segments:

            logger.debug("No segments found in transcription result. Skipping post-processing.")

            return





        # --- Preset Parameters ---

        gap_threshold = {

            "default": 0.3,

            "high_moan": 0.1,

            "narrative": 0.4,

        }.get(preset, 0.3)



        segment_length = {

            "default": 35,

            "high_moan": 25,

            "narrative": 45,

        }.get(preset, 35)







        # --- Linguistic Ending Clusters ---

        BASE_ENDINGS = ['ね', 'よ', 'わ', 'の', 'ぞ', 'ぜ', 'さ', 'か', 'かな', 'な']

        MODERN_ENDINGS = ['じゃん', 'っしょ', 'んだ', 'わけ', 'かも', 'だろう']

        KANSAI_ENDINGS = ['わ', 'で', 'ねん', 'な', 'や']

        FEMININE_ENDINGS = ['かしら', 'こと', 'わね', 'のよ']

        MASCULINE_ENDINGS = ['ぜ', 'ぞ', 'だい', 'かい']

        

        

        POLITE_FORMS = ['です', 'ます', 'でした', 'ましょう', 'ませんか']

        CASUAL_CONTRACTIONS = ['ちゃ', 'じゃ', 'きゃ', 'にゃ', 'ひゃ', 'みゃ', 'りゃ']

        QUESTION_PARTICLES = ['の', 'か']



        FINAL_ENDINGS_TO_LOCK = list(set(

            BASE_ENDINGS +

            MODERN_ENDINGS +

            KANSAI_ENDINGS +

            FEMININE_ENDINGS +

            MASCULINE_ENDINGS

        ))





        AIZUCHI_FILLERS = [

            'あの', 'あのー', 'ええと', 'えっと', 'まあ', 'なんか', 'こう',

            'うん', 'はい', 'ええ', 'そう', 'えっ', 'あっ',

        ]

        



        EXPRESSIVE_EMOTIONS = ['ああ', 'うう', 'ええ', 'おお', 'はあ', 'ふう', 'あっ', 

            'うっ', 'はっ', 'ふっ', 'んっ'

        ]





        CONVERSATIONAL_VERBAL_ENDINGS = [

            'てる',  # from -te iru

            'でる',  # from -de iru

            'ちゃう', # from -te shimau

            'じゃう', # from -de shimau

            'とく',  # from -te oku

            'どく',  # from -de oku

            'んない'  # from -ranai

        ]







        try:



            # --- Pass 1: Remove Fillers and Aizuchi ---

            logger.debug("Phase 1: Sanitization (Fillers and Aizuchi Removal)")

            result.remove_words_by_str(AIZUCHI_FILLERS, case_sensitive=False, strip=True, verbose=False)



            # --- Pass 2: Structural Anchoring ---

            logger.debug("Phase 2: Structural Anchoring")



            # 2a: Split by strong punctuation marks (full-width and half-width)

            result.regroup("sp= 。 / ？ / ！ / … / ． /. /? /!+1")



            # 2b: Lock known structural boundaries

            result.lock(startswith=['「', '『'], left=True, right=False, strip=False)

            result.lock(endswith=['」', '』'], right=True, left=False, strip=False)

            result.lock(endswith=FINAL_ENDINGS_TO_LOCK, right=True, left=False, strip=True)

            result.lock(endswith=POLITE_FORMS, right=True, strip=True)

            result.lock(endswith=QUESTION_PARTICLES, right=True, strip=True)  # Question particles



            for ending in CONVERSATIONAL_VERBAL_ENDINGS:

                            result.custom_operation('word', 'end', ending, 'lockright', word_level=True)



            '''

            for contraction in CASUAL_CONTRACTIONS:

                result.custom_operation(

                    key='word',

                    operator='end',

                    value=contraction,

                    method='lockright',

                    word_level=True

                )

            '''

            

            

            

            # 2c: Lock expressive/emotive interjections (to avoid splitting moans or sighs)

            result.lock(

                startswith=EXPRESSIVE_EMOTIONS,

                endswith=EXPRESSIVE_EMOTIONS,

                left=True,

                right=True

            )









            # --- Pass 3: Merge & Heuristic Refinement ---

            logger.debug("Phase 3: Heuristic-Based Merging & Refinement")

            result.merge_by_punctuation(

                punctuation=['、', '，', ','],

                max_chars=40,

                max_words=15 #, min_dur=0.8  # Only merge if total merged duration is under 0.8s

            )



            result.merge_by_gap(

                min_gap=gap_threshold,

                max_chars=40,

                max_words=15,

                is_sum_max=True  # Enforce limits on the final merged segment

            )



            # --- Optional: Apply split by long pause (>1.2s) ---

            result.split_by_gap(max_gap=1.2)

            

            



            # --- Pass 4: Final Cleanup & Formatting ---

            logger.debug("Phase 4: Final Cleanup & Formatting")

            result.split_by_length(

                max_chars=segment_length,

                max_words=15,

                even_split=False

            )



            # Proactive safety: split long segments to avoid exceeding subtitle timing norms

            result.split_by_duration(max_dur=8.5, even_split=False, lock=True)





            result.reassign_ids()

            

            logger.debug("Japanese post-processing complete.")

            return result



        except Exception as e:

            logger.error(f"An error occurred during Japanese post-processing: {e}")

            logger.debug(traceback.format_exc())







    def _postprocess(self, result):

        """Apply advanced Japanese-specific post-processing using embedded logic."""

        logger.debug("Applying refined Japanese post-processing of segment results")

        

        # Check if we have any segments to process

        if not result.segments:

            logger.debug("No segments found in transcription result")

            return

            

        try:

            self._postprocess_japanese_dialogue(result, preset="default")

            logger.debug("Post-processing segment results completed successfully")

        except Exception as e:

            logger.error(f"Post-processing segment results failed: {e}")

            logger.debug(traceback.format_exc())









        

    def _save_to_srt(self, result: stable_whisper.WhisperResult, output_path: Path):

        """Save transcription result to SRT file."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving SRT to: {output_path}")

        

        result.to_srt_vtt(str(output_path),

                          word_level=False,

                          segment_level=True,

                          strip=True)

