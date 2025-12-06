"""
Stable-TS ASR Component.

OpenAI Whisper with stable-ts enhancements for better timestamp alignment.

Parameter values match v1 asr_config.json exactly for backward compatibility.
"""

from typing import Any, Callable, List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class StableTSOptions(BaseModel):
    """
    Complete Stable-TS options matching v1 asr_config.json structure.

    Combines parameters from:
    - common_decoder_options
    - common_transcriber_options
    - stable_ts_engine_options
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

    # === Stable-TS Engine Options (stable_ts_engine_options) ===
    stream: Optional[bool] = Field(
        None,
        description="Stream mode"
    )
    mel_first: Optional[bool] = Field(
        None,
        description="Compute mel spectrogram first"
    )
    split_callback: Optional[Any] = Field(
        None,
        description="Callback for splitting"
    )
    suppress_ts_tokens: bool = Field(
        False,
        description="Suppress timestamp tokens"
    )
    gap_padding: str = Field(
        " ...",
        description="Padding for gaps"
    )
    only_ffmpeg: bool = Field(
        False,
        description="Use only FFmpeg for audio loading"
    )
    max_instant_words: float = Field(
        0.5,
        ge=0.0, le=1.0,
        description="Maximum instant words ratio"
    )
    avg_prob_threshold: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Average probability threshold"
    )
    nonspeech_skip: Optional[float] = Field(
        None,
        ge=0.0,
        description="Skip non-speech segments longer than this"
    )
    progress_callback: Optional[Any] = Field(
        None,
        description="Progress callback function"
    )
    ignore_compatibility: bool = Field(
        True,
        description="Ignore compatibility checks"
    )
    extra_models: Optional[List[str]] = Field(
        None,
        description="Extra models to use"
    )
    dynamic_heads: Optional[int] = Field(
        None,
        ge=1,
        description="Dynamic attention heads"
    )
    nonspeech_error: float = Field(
        0.1,
        ge=0.0, le=1.0,
        description="Non-speech error tolerance"
    )
    only_voice_freq: bool = Field(
        False,
        description="Filter to voice frequencies only"
    )
    min_word_dur: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Minimum word duration"
    )
    min_silence_dur: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum silence duration"
    )
    regroup: bool = Field(
        True,
        description="Regroup segments for better timing"
    )
    ts_num: int = Field(
        0,
        ge=0,
        description="Timestamp number"
    )
    ts_noise: Optional[float] = Field(
        None,
        ge=0.0,
        description="Timestamp noise"
    )
    suppress_silence: bool = Field(
        True,
        description="Suppress silence in output"
    )
    suppress_word_ts: bool = Field(
        True,
        description="Suppress word timestamps"
    )
    suppress_attention: bool = Field(
        False,
        description="Suppress attention weights"
    )
    use_word_position: bool = Field(
        True,
        description="Use word position for alignment"
    )
    q_levels: int = Field(
        20,
        ge=1, le=100,
        description="Quantization levels"
    )
    k_size: int = Field(
        5,
        ge=1, le=20,
        description="Kernel size for smoothing"
    )
    time_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time scale factor"
    )
    denoiser: Optional[str] = Field(
        None,
        description="Denoiser to use"
    )
    denoiser_options: Optional[dict] = Field(
        None,
        description="Denoiser options"
    )
    demucs: bool = Field(
        False,
        description="Use Demucs for source separation"
    )
    demucs_options: Optional[dict] = Field(
        None,
        description="Demucs options"
    )

    # === VAD Options (stable_ts_vad_options) ===
    vad: bool = Field(
        True,
        description="Use VAD preprocessing"
    )
    vad_threshold: float = Field(
        0.25,
        ge=0.0, le=1.0,
        description="VAD threshold"
    )


@register_asr
class StableTSASR(ASRComponent):
    """Stable-TS ASR with enhanced timestamp alignment."""

    # === Metadata ===
    name = "stable_ts"
    display_name = "Stable-TS"
    description = "OpenAI Whisper with stable-ts for better timestamp alignment and regrouping."
    version = "1.0.0"
    tags = ["asr", "whisper", "stable-ts", "timestamps"]

    # === ASR-specific ===
    provider = "stable_ts"
    model_id = "large-v3"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["none"]  # Stable-TS has built-in VAD

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = StableTSOptions

    # === Presets - Exact v1 values ===
    presets = {
        "conservative": StableTSOptions(
            # Decoder options - same as faster_whisper
            task="transcribe",
            language="ja",
            beam_size=1,
            best_of=1,
            patience=1.5,
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options - same as faster_whisper
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
            # Stable-TS engine options
            stream=None,
            mel_first=None,
            split_callback=None,
            suppress_ts_tokens=False,
            gap_padding=" ...",
            only_ffmpeg=False,
            max_instant_words=0.5,
            avg_prob_threshold=None,
            nonspeech_skip=None,
            progress_callback=None,
            ignore_compatibility=True,
            extra_models=None,
            dynamic_heads=None,
            nonspeech_error=0.1,
            only_voice_freq=False,
            min_word_dur=None,
            min_silence_dur=None,
            regroup=True,
            ts_num=0,
            ts_noise=None,
            suppress_silence=True,
            suppress_word_ts=True,
            suppress_attention=False,
            use_word_position=True,
            q_levels=20,
            k_size=5,
            time_scale=None,
            denoiser=None,
            denoiser_options=None,
            demucs=False,
            demucs_options=None,
            # VAD options
            vad=True,
            vad_threshold=0.35,
        ),
        "balanced": StableTSOptions(
            # Decoder options - same as faster_whisper
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=1,
            patience=2.0,
            length_penalty=None,
            prefix=None,
            suppress_tokens=None,
            suppress_blank=True,
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options - same as faster_whisper
            temperature=[0.0, 0.1],
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
            # Stable-TS engine options
            stream=None,
            mel_first=None,
            split_callback=None,
            suppress_ts_tokens=False,
            gap_padding=" ...",
            only_ffmpeg=False,
            max_instant_words=0.5,
            avg_prob_threshold=None,
            nonspeech_skip=None,
            progress_callback=None,
            ignore_compatibility=True,
            extra_models=None,
            dynamic_heads=None,
            nonspeech_error=0.1,
            only_voice_freq=False,
            min_word_dur=None,
            min_silence_dur=None,
            regroup=True,
            ts_num=0,
            ts_noise=None,
            suppress_silence=True,
            suppress_word_ts=True,
            suppress_attention=False,
            use_word_position=True,
            q_levels=20,
            k_size=5,
            time_scale=None,
            denoiser=None,
            denoiser_options=None,
            demucs=False,
            demucs_options=None,
            # VAD options
            vad=True,
            vad_threshold=0.25,
        ),
        "aggressive": StableTSOptions(
            # Decoder options - same as faster_whisper
            task="transcribe",
            language="ja",
            beam_size=2,
            best_of=1,
            patience=2.0,
            length_penalty=None,
            prefix=None,
            suppress_blank=False,  # Different!
            suppress_tokens=[],     # Empty list, not None!
            without_timestamps=False,
            max_initial_timestamp=None,
            # Transcriber options - same as faster_whisper
            temperature=[0.0, 0.3],
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
            # Stable-TS engine options
            stream=None,
            mel_first=None,
            split_callback=None,
            suppress_ts_tokens=False,
            gap_padding=" ...",
            only_ffmpeg=False,
            max_instant_words=0.5,
            avg_prob_threshold=None,
            nonspeech_skip=None,
            progress_callback=None,
            ignore_compatibility=True,
            extra_models=None,
            dynamic_heads=None,
            nonspeech_error=0.1,
            only_voice_freq=False,
            min_word_dur=None,
            min_silence_dur=None,
            regroup=True,
            ts_num=0,
            ts_noise=None,
            suppress_silence=True,
            suppress_word_ts=True,
            suppress_attention=False,
            use_word_position=True,
            q_levels=20,
            k_size=5,
            time_scale=None,
            denoiser=None,
            denoiser_options=None,
            demucs=False,
            demucs_options=None,
            # VAD options
            vad=True,
            vad_threshold=0.1,
        ),
    }
