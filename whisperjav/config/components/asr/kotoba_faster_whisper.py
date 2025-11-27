"""
Kotoba Faster-Whisper ASR Component.

Japanese-optimized Whisper model from kotoba-tech using faster-whisper backend.
Uses internal VAD (faster-whisper's built-in vad_filter) for speech detection.

Reference Implementation:
    - Model: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster
    - Alternative: https://huggingface.co/RoachLin/kotoba-whisper-v2.2-faster
    - Base paper: https://arxiv.org/abs/2412.01307 (Kotoba-Whisper training)
    - Parameter values derived from kotoba-tech model card recommendations

Default VAD parameters (vad_threshold=0.01, min_speech_duration_ms=90, etc.)
are optimized for Japanese speech patterns per kotoba-tech documentation.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class KotobaFasterWhisperOptions(BaseModel):
    """
    Kotoba Faster-Whisper options with internal VAD support.

    Key differences from standard Faster-Whisper:
    - Uses kotoba-tech/kotoba-whisper-v2.0-faster model
    - Includes internal VAD control (vad_filter, vad_parameters)
    - Optimized default values for Japanese speech
    """

    # === Internal VAD Options (faster-whisper built-in) ===
    vad_filter: bool = Field(
        True,
        description="Enable internal VAD filtering (faster-whisper built-in)"
    )
    vad_threshold: float = Field(
        0.01,
        ge=0.0, le=1.0,
        description="VAD threshold (lower=more aggressive, captures more speech)"
    )
    min_speech_duration_ms: int = Field(
        90,
        ge=0, le=5000,
        description="Minimum speech duration in milliseconds"
    )
    max_speech_duration_s: float = Field(
        28.0,
        ge=1.0, le=60.0,
        description="Maximum speech duration in seconds before splitting"
    )
    min_silence_duration_ms: int = Field(
        150,
        ge=0, le=5000,
        description="Minimum silence duration to split on"
    )
    speech_pad_ms: int = Field(
        400,
        ge=0, le=2000,
        description="Padding around detected speech in milliseconds"
    )

    # === Decoder Options ===
    task: str = Field(
        "transcribe",
        description="Task: 'transcribe' or 'translate'"
    )
    language: str = Field(
        "ja",
        description="Language code for transcription (ja=Japanese optimized)"
    )
    beam_size: int = Field(
        3,
        ge=1, le=20,
        description="Beam size for decoding"
    )
    best_of: int = Field(
        3,
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

    # === Transcriber Options ===
    temperature: Union[float, List[float]] = Field(
        [0.0, 0.3],
        description="Temperature for sampling. List enables fallback temperatures."
    )
    compression_ratio_threshold: float = Field(
        2.4,
        ge=1.0, le=5.0,
        description="Threshold for gzip compression ratio"
    )
    logprob_threshold: float = Field(
        -1.5,
        ge=-5.0, le=0.0,
        description="Average log probability threshold"
    )
    no_speech_threshold: float = Field(
        0.34,
        ge=0.0, le=1.0,
        description="No speech probability threshold"
    )
    condition_on_previous_text: bool = Field(
        True,
        description="Condition on previous output (enabled for Kotoba)"
    )
    initial_prompt: Optional[str] = Field(
        None,
        description="Initial prompt for first window"
    )
    word_timestamps: bool = Field(
        False,
        description="Extract word-level timestamps"
    )

    # === Engine Options ===
    repetition_penalty: float = Field(
        1.0,
        ge=1.0, le=3.0,
        description="Penalty for token repetition"
    )
    no_repeat_ngram_size: int = Field(
        0,
        ge=0, le=10,
        description="N-gram size to prevent repetition (0=disabled)"
    )
    log_progress: bool = Field(
        False,
        description="Log transcription progress"
    )


@register_asr
class KotobaFasterWhisperASRComponent(ASRComponent):
    """Kotoba Faster-Whisper ASR using Japanese-optimized model with internal VAD."""

    # === Metadata ===
    name = "kotoba_faster_whisper"
    display_name = "Kotoba Faster-Whisper"
    description = "Japanese-optimized Whisper model from kotoba-tech with internal VAD support."
    version = "1.0.0"
    tags = ["asr", "whisper", "kotoba", "japanese", "ctranslate2", "internal-vad"]

    # === ASR-specific ===
    provider = "faster_whisper"
    model_id = "kotoba-tech/kotoba-whisper-v2.0-faster"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["none"]  # Uses internal VAD, not external

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float32"

    # === Schema ===
    Options = KotobaFasterWhisperOptions

    # === Presets - Based on reference implementation ===
    presets = {
        "conservative": KotobaFasterWhisperOptions(
            # VAD options - less aggressive
            vad_filter=True,
            vad_threshold=0.2,
            min_speech_duration_ms=100,
            max_speech_duration_s=30.0,
            min_silence_duration_ms=200,
            speech_pad_ms=300,
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=3,
            best_of=3,
            patience=2.0,
            length_penalty=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=False,
            # Engine options
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            log_progress=False,
        ),
        "balanced": KotobaFasterWhisperOptions(
            # VAD options - from reference implementation
            vad_filter=True,
            vad_threshold=0.01,
            min_speech_duration_ms=90,
            max_speech_duration_s=28.0,
            min_silence_duration_ms=150,
            speech_pad_ms=400,
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=3,
            best_of=3,
            patience=2.0,
            length_penalty=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            # Transcriber options
            temperature=[0.0, 0.3],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.5,
            no_speech_threshold=0.34,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=False,
            # Engine options
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            log_progress=False,
        ),
        "aggressive": KotobaFasterWhisperOptions(
            # VAD options - very aggressive
            vad_filter=False,
            vad_threshold=0.005,
            min_speech_duration_ms=50,
            max_speech_duration_s=28.0,
            min_silence_duration_ms=100,
            speech_pad_ms=500,
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=5,
            best_of=5,
            patience=2.5,
            length_penalty=None,
            suppress_tokens=[],  # Empty list = suppress nothing
            suppress_blank=False,
            without_timestamps=False,
            # Transcriber options
            temperature=[0.0,0.1,0.5],
            compression_ratio_threshold=3.0,
            logprob_threshold=-2.0,
            no_speech_threshold=0.2,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=False,
            # Engine options
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            log_progress=False,
        ),
    }
