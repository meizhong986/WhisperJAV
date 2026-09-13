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
            best_of=2,                            # v1.8.10-hf3: 1→2; v1.8.12: 2→1; v1.8.14: 1→2, engine-symmetric quality retune
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
            logprob_threshold=-1.00,              # v1.8.12: -0.80→-0.70; v1.8.14: -0.70→-1.00, gate relaxation
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
            best_of=2,                            # v1.8.10-hf3: 1→2; v1.8.12: 2→1; v1.8.14: 1→2, engine-symmetric quality retune
            patience=1.2,                         # v1.8.10-hf3: 2.0→1.6; v1.8.12: 1.6→1.5; v1.8.14: 1.5→1.2, engine-symmetric quality retune
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.00,              # v1.8.12: -1.00→-0.85; v1.8.14: -0.85→-1.00, gate relaxation
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
            verbose=None,
            carry_initial_prompt=None,
            prompt=None,
            fp16=True,
            # Exclusive options
            hallucination_silence_threshold=None,  # v1.8.10-hf1: 2.0→None, disabled
        ),
        # v1.9.2 aggressive retune (owner O4 + O6, 2026-09-10). His table is scoped to
        # "balanced and fidelity"; fidelity is this component. Six of his seven rows are
        # applied below. The seventh, repetition_penalty=1.50, is NOT APPLIED HERE and
        # cannot be: openai-whisper exposes no such parameter -- neither
        # whisper.transcribe.transcribe() nor whisper.decoding.DecodingOptions accepts it
        # (verified against the installed package). It is a CTranslate2 feature and is
        # applied on the faster_whisper component only.
        "aggressive": OpenAIWhisperOptions(
            # Decoder options
            task="transcribe",
            language="ja",
            beam_size=2,                          # v1.9.2: 3→2, ~95% of the beam-search gain at half the compute
            best_of=2,                            # v1.8.10-hf3: 3→2; v1.8.12: 2→1; v1.8.12.post1: 1→2, mirror faster_whisper aggressive fix
            patience=1.0,                         # v1.9.2: 1.5→1.0, standard beam termination
            length_penalty=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=None,
            without_timestamps=False,
            max_initial_timestamp=0.0,
            # Transcriber options
            temperature=[0.0],                    # v1.9.2 (O4+O6): [0.0, 0.17]→[0.0], no temperature retries, caps worst-case run time
            compression_ratio_threshold=2.2,      # v1.9.2: 2.6→2.2, drops repetitive decoder loops earlier
            logprob_threshold=-1.00,              # v1.9.2: -1.55→-1.00, filters acoustic static while keeping low-confidence speech
            logprob_margin=0.0,
            no_speech_threshold=0.72,             # v1.9.2: 0.84→0.72, admits quiet/low-SNR audio into the decoder
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
