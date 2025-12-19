"""
Faster-Whisper ASR Component.

High-performance Whisper implementation using CTranslate2.

Parameter values match v1 asr_config.json exactly for backward compatibility.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class FasterWhisperOptions(BaseModel):
    """
    Complete Faster-Whisper options matching v1 asr_config.json structure.

    Combines parameters from:
    - common_decoder_options
    - common_transcriber_options
    - faster_whisper_engine_options
    - exclusive_whisper_plus_faster_whisper
    """

    # === Decoder Options (common_decoder_options) ===
    task: str = Field(
        "transcribe",
        description="Task: 'transcribe' or 'translate'"
    )
    language: str = Field(
        "ja",
        description="Language code for transcription"
    )
    beam_size: int = Field(
        2,
        ge=1, le=20,
        description="Beam size for decoding"
    )
    best_of: int = Field(
        1,
        ge=1, le=10,
        description="Number of candidates when sampling"
    )
    patience: float = Field(
        2.0,
        ge=0.0, le=5.0,
        description="Beam search patience factor"
    )
    length_penalty: Optional[float] = Field(
        None,
        ge=0.0, le=2.0,
        description="Exponential length penalty"
    )
    prefix: Optional[str] = Field(
        None,
        description="Optional text prefix for first window"
    )
    suppress_tokens: Optional[List[int]] = Field(
        None,
        description="Token IDs to suppress. None=default, []=none"
    )
    suppress_blank: bool = Field(
        True,
        description="Suppress blank outputs at start of sampling"
    )
    without_timestamps: bool = Field(
        False,
        description="Only sample text tokens"
    )
    max_initial_timestamp: Optional[float] = Field(
        None,
        ge=0.0,
        description="Max initial timestamp"
    )

    # === Transcriber Options (common_transcriber_options) ===
    temperature: Union[float, List[float]] = Field(
        [0.0, 0.1],
        description="Temperature for sampling. List enables fallback temperatures."
    )
    compression_ratio_threshold: float = Field(
        2.4,
        ge=1.0, le=5.0,
        description="Threshold for gzip compression ratio"
    )
    logprob_threshold: float = Field(
        -1.2,
        ge=-5.0, le=0.0,
        description="Average log probability threshold"
    )
    logprob_margin: float = Field(
        0.2,
        ge=0.0, le=5.0,
        description="Margin for log probability filtering"
    )
    no_speech_threshold: float = Field(
        0.5,
        ge=0.0, le=1.0,
        description="No speech probability threshold"
    )
    drop_nonverbal_vocals: bool = Field(
        False,
        description="Drop non-verbal vocalizations (laughter, coughing, etc.)"
    )
    condition_on_previous_text: bool = Field(
        False,
        description="Condition on previous output"
    )
    initial_prompt: Optional[str] = Field(
        None,
        description="Initial prompt for first window"
    )
    word_timestamps: bool = Field(
        True,
        description="Extract word-level timestamps"
    )
    prepend_punctuations: Optional[str] = Field(
        None,
        description="Punctuations to prepend to next word"
    )
    append_punctuations: Optional[str] = Field(
        None,
        description="Punctuations to append to previous word"
    )
    clip_timestamps: Optional[str] = Field(
        None,
        description="Comma-separated timestamp ranges to clip"
    )

    # === Engine Options (faster_whisper_engine_options) ===
    chunk_length: Optional[int] = Field(
        None,
        ge=1, le=30,
        description="Length of audio chunks in seconds"
    )
    repetition_penalty: float = Field(
        1.5,
        ge=1.0, le=3.0,
        description="Penalty for token repetition"
    )
    no_repeat_ngram_size: int = Field(
        2,
        ge=0, le=10,
        description="N-gram size to prevent repetition"
    )
    prompt_reset_on_temperature: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Temperature to reset prompt"
    )
    hotwords: Optional[str] = Field(
        None,
        description="Hotwords/hint phrases"
    )
    multilingual: bool = Field(
        False,
        description="Enable multilingual model"
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of new tokens"
    )
    language_detection_threshold: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Language detection probability threshold"
    )
    language_detection_segments: Optional[int] = Field(
        None,
        ge=1,
        description="Number of segments for language detection"
    )
    log_progress: bool = Field(
        False,
        description="Log transcription progress"
    )

    # === Exclusive Options (exclusive_whisper_plus_faster_whisper) ===
    hallucination_silence_threshold: Optional[float] = Field(
        2.0,
        ge=0.0, le=10.0,
        description="Skip silent periods longer than this (seconds)"
    )


@register_asr
class FasterWhisperASR(ASRComponent):
    """Faster-Whisper ASR using CTranslate2 backend."""

    # === Metadata ===
    name = "faster_whisper"
    display_name = "Faster Whisper"
    description = "High-performance Whisper using CTranslate2. Best balance of speed and accuracy."
    version = "1.0.0"
    tags = ["asr", "whisper", "ctranslate2", "fast"]

    # === ASR-specific ===
    provider = "faster_whisper"
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["silero", "faster_whisper_vad", "none"]

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = FasterWhisperOptions

    # === Presets - Exact v1 values ===
    presets = {
        "conservative": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=1,
            best_of=1,
            patience=1.2,  # Changed from 1.5
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            logprob_margin=0.1,
            no_speech_threshold=0.74,
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            chunk_length=None,
            repetition_penalty=1.8,
            no_repeat_ngram_size=2,
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=1.5,
        ),
        "balanced": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=1,
            patience=1.4,  # Changed from 2.0
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options
            temperature=[0.0],  # Changed from [0.0, 0.1]
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.2,
            logprob_margin=0.2,
            no_speech_threshold=0.5,
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            chunk_length=None,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2,
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=1.9,  # Changed from 2.0
        ),
        "aggressive": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=3,  # Changed from 2
            best_of=1,
            patience=1.6,  # Changed from 2.0
            length_penalty=None,
            prefix=None,
            suppress_blank=False,  # Different!
            suppress_tokens=[],     # Empty list, not None!
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options
            temperature=[0.0],  # Changed from [0.0, 0.3]
            compression_ratio_threshold=3.0,
            logprob_threshold=-2.5,
            logprob_margin=0.0,
            no_speech_threshold=0.22,
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            chunk_length=30,  # Increased from 14 to avoid ctranslate2 divide-by-zero crash
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=2.1,  # Changed from 2.5
        ),
    }
