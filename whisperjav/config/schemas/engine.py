"""
Engine options schemas for WhisperJAV.

Defines parameters specific to each ASR engine backend.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from .base import BaseConfig


class FasterWhisperEngineOptions(BaseConfig):
    """
    Faster-Whisper specific engine parameters.

    Maps to 'faster_whisper_engine_options' in asr_config.json.
    """

    chunk_length: Optional[int] = Field(
        default=None,
        description="Audio chunk length for processing."
    )
    repetition_penalty: float = Field(
        ge=1.0, le=2.0,
        description="Penalty for token repetition."
    )
    no_repeat_ngram_size: int = Field(
        ge=0,
        description="N-gram size to prevent repetition. Must be int for ctranslate2!"
    )
    prompt_reset_on_temperature: Optional[float] = Field(
        default=None,
        description="Temperature at which to reset prompt."
    )
    hotwords: Optional[str] = Field(
        default=None,
        description="Hotwords to boost recognition."
    )
    multilingual: bool = Field(
        default=False,
        description="Enable multilingual mode."
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum new tokens to generate."
    )
    language_detection_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for language detection."
    )
    language_detection_segments: Optional[int] = Field(
        default=None,
        description="Number of segments for language detection."
    )
    log_progress: bool = Field(
        default=False,
        description="Log transcription progress."
    )


class OpenAIWhisperEngineOptions(BaseConfig):
    """
    OpenAI Whisper specific engine parameters.

    Maps to 'openai_whisper_engine_options' in asr_config.json.
    """

    fp16: bool = Field(
        default=True,
        description="Use FP16 precision."
    )
    hallucination_silence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for hallucination detection during silence."
    )


class StableTSEngineOptions(BaseConfig):
    """
    Stable-TS specific engine parameters (29 parameters).

    Maps to 'stable_ts_engine_options' in asr_config.json.
    These control the stable-ts refinement and alignment features.
    """

    # Streaming and processing
    stream: Optional[bool] = Field(
        default=None,
        description="Enable streaming mode."
    )
    mel_first: Optional[bool] = Field(
        default=None,
        description="Process mel spectrogram first."
    )
    split_callback: Optional[str] = Field(
        default=None,
        description="Callback function for splitting."
    )
    suppress_ts_tokens: bool = Field(
        default=False,
        description="Suppress timestamp tokens."
    )
    gap_padding: str = Field(
        default=" ...",
        description="Padding for gaps in transcript."
    )
    only_ffmpeg: bool = Field(
        default=False,
        description="Use only ffmpeg for audio processing."
    )

    # Word timing
    max_instant_words: float = Field(
        default=0.5,
        description="Maximum duration for instant words."
    )
    avg_prob_threshold: Optional[float] = Field(
        default=None,
        description="Average probability threshold."
    )

    # Non-speech handling
    nonspeech_skip: Optional[float] = Field(
        default=None,
        description="Skip non-speech segments."
    )
    nonspeech_error: float = Field(
        default=0.1,
        description="Error tolerance for non-speech detection."
    )

    # Callbacks and compatibility
    progress_callback: Optional[str] = Field(
        default=None,
        description="Progress callback function."
    )
    ignore_compatibility: bool = Field(
        default=True,
        description="Ignore compatibility checks."
    )

    # Additional models
    extra_models: Optional[str] = Field(
        default=None,
        description="Additional models to use."
    )
    dynamic_heads: Optional[str] = Field(
        default=None,
        description="Dynamic attention heads."
    )

    # Audio processing
    only_voice_freq: bool = Field(
        default=False,
        description="Use only voice frequencies."
    )
    min_word_dur: Optional[float] = Field(
        default=None,
        description="Minimum word duration."
    )
    min_silence_dur: Optional[float] = Field(
        default=None,
        description="Minimum silence duration."
    )

    # Regrouping
    regroup: bool = Field(
        default=True,
        description="Enable text regrouping."
    )
    ts_num: int = Field(
        default=0,
        description="Timestamp number mode."
    )
    ts_noise: Optional[float] = Field(
        default=None,
        description="Timestamp noise tolerance."
    )

    # Silence suppression
    suppress_silence: bool = Field(
        default=True,
        description="Suppress silence in output."
    )
    suppress_word_ts: bool = Field(
        default=True,
        description="Suppress word timestamps."
    )
    suppress_attention: bool = Field(
        default=False,
        description="Suppress attention weights."
    )
    use_word_position: bool = Field(
        default=True,
        description="Use word position information."
    )

    # Quantization
    q_levels: int = Field(
        default=20,
        description="Quantization levels."
    )
    k_size: int = Field(
        default=5,
        description="Kernel size."
    )
    time_scale: Optional[float] = Field(
        default=None,
        description="Time scaling factor."
    )

    # Denoising
    denoiser: Optional[str] = Field(
        default=None,
        description="Denoiser to use."
    )
    denoiser_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Options for denoiser."
    )

    # Demucs separation
    demucs: bool = Field(
        default=False,
        description="Enable Demucs audio separation."
    )
    demucs_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Options for Demucs."
    )
