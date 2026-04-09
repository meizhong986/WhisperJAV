"""
OpenAI Whisper ASR Component.

Original OpenAI Whisper implementation for fidelity mode.

Parameter values match v1 asr_config.json exactly for backward compatibility.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class OpenAIWhisperOptions(BaseModel):
    """
    Complete OpenAI Whisper options matching v1 asr_config.json structure.

    Combines parameters from:
    - common_decoder_options
    - common_transcriber_options
    - openai_whisper_engine_options
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
        True,
        description="Enable WhisperJAV's post-model gate (logprob filter). "
                    "Default ON for OpenAI Whisper — R5 forensic analysis "
                    "showed the gate catches hallucinations the sanitizer "
                    "misses while still achieving 83.8% capture. Set False "
                    "to disable the gate."
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

    # === Engine Options (openai_whisper_engine_options) ===
    verbose: Optional[bool] = Field(
        None,
        description="Verbose output"
    )
    carry_initial_prompt: Optional[bool] = Field(
        None,
        description="Carry initial prompt across segments"
    )
    prompt: Optional[str] = Field(
        None,
        description="Prompt for transcription"
    )
    fp16: bool = Field(
        True,
        description="Use FP16 precision"
    )

    # === Exclusive Options (exclusive_whisper_plus_faster_whisper) ===
    hallucination_silence_threshold: Optional[float] = Field(
        None,
        ge=0.0, le=10.0,
        description="Skip silent periods longer than this (seconds). None = disabled."
    )


@register_asr
class OpenAIWhisperASR(ASRComponent):
    """OpenAI Whisper ASR for fidelity mode."""

    # === Metadata ===
    name = "openai_whisper"
    display_name = "OpenAI Whisper"
    description = "Original OpenAI Whisper implementation. Best for maximum fidelity."
    version = "1.0.0"
    tags = ["asr", "whisper", "openai", "fidelity"]

    # === ASR-specific ===
    provider = "openai_whisper"
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["silero", "none"]

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = OpenAIWhisperOptions

    # === Presets - Exact v1 values ===
    presets = {
        "conservative": OpenAIWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=2,                            # v1.8.10-hf3: 1→2, uniform across sensitivities
            patience=1.2,                         # v1.8.10-hf3: 1.5→1.2, retuned per forensic analysis
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.2,
            logprob_threshold=-0.80,
            logprob_margin=0.0,
            no_speech_threshold=0.46,             # v1.8.10-hf3: 0.60→0.46, retuned per forensic analysis
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            verbose=None,
            carry_initial_prompt=None,
            prompt=None,
            fp16=True,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 1.5→None, disabled
        ),
        "balanced": OpenAIWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=2,                            # v1.8.10-hf3: 1→2, uniform across sensitivities
            patience=1.6,                         # v1.8.10-hf3: 2.0→1.6, retuned per forensic analysis
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.00,
            logprob_margin=0.0,
            no_speech_threshold=0.65,             # v1.8.10-hf3: 0.70→0.65, retuned per forensic analysis
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            verbose=None,
            carry_initial_prompt=None,
            prompt=None,
            fp16=True,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 2.0→None, disabled
        ),
        "aggressive": OpenAIWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,                          # v1.8.10-hf3: 4→2, uniform across sensitivities
            best_of=2,                            # v1.8.10-hf3: 3→2, uniform across sensitivities
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
            no_speech_threshold=0.77,             # v1.8.10-hf3: 0.90→0.77, retuned per forensic analysis
            drop_nonverbal_vocals=False,
            condition_on_previous_text=False,      # v1.8.10-hf1: True→False, prevents hallucination propagation
            initial_prompt=None,
            word_timestamps=True,
            prepend_punctuations=None,
            append_punctuations=None,
            clip_timestamps=None,
            # Engine options
            verbose=False,
            carry_initial_prompt=None,
            prompt=None,
            fp16=True,
            # Exclusive options
            hallucination_silence_threshold=None,  # Disabled
        ),
    }
