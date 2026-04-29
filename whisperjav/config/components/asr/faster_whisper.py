"""
Faster-Whisper ASR Component.

High-performance Whisper implementation using CTranslate2.

These Pydantic presets are the SINGLE SOURCE OF TRUTH for all Faster-Whisper
pipeline parameters (balanced, fast, faster, fidelity modes). As of v1.8.9,
asr_config.json is no longer read for pipeline parameters.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class FasterWhisperOptions(BaseModel):
    """
    Complete Faster-Whisper options — the single source of truth for all
    decoder, transcriber, and engine parameters.

    Parameter categories:
    - Decoder: beam_size, best_of, patience, suppress_blank, etc.
    - Transcriber: temperature, compression_ratio_threshold, logprob_threshold, etc.
    - Engine: repetition_penalty, no_repeat_ngram_size, chunk_length, etc.
    - Exclusive: hallucination_silence_threshold
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
        ge=-2.0, le=2.0,
        description="Exponential length penalty (negative = prefer shorter sequences)"
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
        0.0,
        ge=0.0,
        description="Max initial timestamp (0 = prevent phantom early timestamps)"
    )

    # === Transcriber Options (common_transcriber_options) ===
    temperature: Union[float, List[float]] = Field(
        [0.0],
        description="Temperature for sampling. List enables fallback temperatures."
    )
    compression_ratio_threshold: float = Field(
        2.4,
        ge=1.0, le=5.0,
        description="Threshold for gzip compression ratio"
    )
    logprob_threshold: float = Field(
        -0.75,
        ge=-5.0, le=0.0,
        description="Average log probability threshold"
    )
    logprob_margin: float = Field(
        0.2,
        ge=0.0, le=5.0,
        description="Margin for log probability filtering"
    )
    no_speech_threshold: float = Field(
        0.55,
        ge=0.0, le=1.0,
        description="No speech probability threshold"
    )
    drop_nonverbal_vocals: bool = Field(
        False,
        description="Drop non-verbal vocalizations (laughter, coughing, etc.)"
    )
    post_model_filter_enabled: bool = Field(
        False,
        description="Enable WhisperJAV's post-model gate (logprob filter). "
                    "Default OFF for Faster-Whisper — R6 forensic analysis "
                    "showed the gate drops 60% of raw output including "
                    "correct content. Set True to re-enable the gate."
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
        3,
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
        None,
        ge=0.0, le=10.0,
        description="Skip silent periods longer than this (seconds). None = disabled."
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
    model_id = "large-v3"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["silero", "faster_whisper_vad", "none"]

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = FasterWhisperOptions

    # === Presets - tuned for v1.8.9 ===
    presets = {
        "conservative": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=1,                            # v1.8.10-hf3: 1→2; v1.8.12: 2→1, engine-split retune
            patience=1.0,                         # v1.8.10-hf3: 1.5→1.2; v1.8.12: 1.2→1.0, engine-split retune
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.2,
            logprob_threshold=-0.70,              # v1.8.12: -0.80→-0.70, engine-split retune
            logprob_margin=0.0,
            no_speech_threshold=0.54,             # v1.8.10-hf3: 0.60→0.46; v1.8.12: 0.46→0.54, engine-split retune
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
            no_repeat_ngram_size=3,               # v1.8.10-hf1: 2→3, prevents repetition loops
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 1.5→None, disabled
        ),
        "balanced": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=1,                            # v1.8.12: 2→1, engine-split retune
            patience=1.5,                         # v1.8.10-hf3: 2.0→1.6; v1.8.12: 1.6→1.5, engine-split retune
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.4,
            logprob_threshold=-0.85,              # v1.8.12: -1.00→-0.85, engine-split retune
            logprob_margin=0.0,
            no_speech_threshold=0.71,             # v1.8.10-hf3: 0.70→0.65; v1.8.12: 0.65→0.71, engine-split retune
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
            no_repeat_ngram_size=3,               # v1.8.10-hf1: 2→3, prevents repetition loops
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 2.0→None, disabled
        ),
        "aggressive": FasterWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=3,                          # v1.8.10-hf3: 4→2; v1.8.12: 2→3, engine-split retune
            best_of=2,                            # v1.8.10-hf3: 3→2; v1.8.12: 2→1; v1.8.12.post1: 1→2, fix F5 empty-output regression
            patience=2.0,                         # v1.8.10-hf3: 2.5→2.0, retuned per forensic analysis
            length_penalty=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=None,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0, 0.17],              # v1.8.10-hf3: [0.0]→[0.0, 0.17], light fallback for aggressive
            compression_ratio_threshold=2.6,
            logprob_threshold=-1.00,              # v1.8.10-hf3: -1.30→-1.00, uniform across sensitivities
            logprob_margin=0.0,
            no_speech_threshold=0.84,             # v1.8.10-hf3: 0.90→0.77; v1.8.12: 0.77→0.84, engine-split retune
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,      # v1.8.10-hf1: True→False, prevents hallucination propagation
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            chunk_length=30,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,               # v1.8.10-hf1: 2→3, prevents repetition loops
            prompt_reset_on_temperature=None,
            hotwords=None,
            multilingual=False,
            max_new_tokens=None,
            language_detection_threshold=None,
            language_detection_segments=None,
            log_progress=False,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 4.0→None, disabled
        ),
    }
