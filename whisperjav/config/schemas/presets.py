"""
Sensitivity presets for WhisperJAV Configuration System v2.0.

Contains all preset values from asr_config.json organized by sensitivity level.
"""

from .base import Sensitivity
from .decoder import DecoderOptions
from .engine import FasterWhisperEngineOptions, StableTSEngineOptions
from .transcriber import TranscriberOptions
from .vad import SileroVADOptions, StableTSVADOptions

# Common Transcriber Options Presets
TRANSCRIBER_PRESETS = {
    Sensitivity.BALANCED: TranscriberOptions(
        temperature=[0.0, 0.1],
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.2,
        logprob_margin=0.2,
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=True,
        prepend_punctuations=None,
        append_punctuations=None,
        clip_timestamps=None
    ),
    Sensitivity.CONSERVATIVE: TranscriberOptions(
        temperature=[0.0],
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        logprob_margin=0.1,
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.74,
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=True,
        prepend_punctuations=None,
        append_punctuations=None,
        clip_timestamps=None
    ),
    Sensitivity.AGGRESSIVE: TranscriberOptions(
        temperature=[0.0, 0.3],
        compression_ratio_threshold=3.0,
        logprob_threshold=-2.5,
        logprob_margin=0.0,
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.22,
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=True,
        prepend_punctuations=None,
        append_punctuations=None,
        clip_timestamps=None
    ),
}

# Common Decoder Options Presets
DECODER_PRESETS = {
    Sensitivity.BALANCED: DecoderOptions(
        task="transcribe",
        language="ja",
        best_of=1,
        beam_size=2,
        patience=2.0,
        length_penalty=None,
        prefix=None,
        suppress_tokens=None,
        suppress_blank=True,
        without_timestamps=False,
        max_initial_timestamp=None
    ),
    Sensitivity.CONSERVATIVE: DecoderOptions(
        task="transcribe",
        language="ja",
        best_of=1,
        beam_size=1,
        patience=1.5,
        length_penalty=None,
        prefix=None,
        suppress_tokens=None,
        suppress_blank=True,
        without_timestamps=False,
        max_initial_timestamp=None
    ),
    Sensitivity.AGGRESSIVE: DecoderOptions(
        task="transcribe",
        language="ja",
        best_of=1,
        beam_size=2,
        patience=2.0,
        length_penalty=None,
        prefix=None,
        suppress_tokens=[],  # Empty list, not None
        suppress_blank=False,
        without_timestamps=False,
        max_initial_timestamp=None
    ),
}

# Silero VAD Options Presets
SILERO_VAD_PRESETS = {
    Sensitivity.BALANCED: SileroVADOptions(
        threshold=0.18,
        min_speech_duration_ms=100,
        max_speech_duration_s=11.0,
        min_silence_duration_ms=300,
        neg_threshold=0.15,
        speech_pad_ms=400
    ),
    Sensitivity.CONSERVATIVE: SileroVADOptions(
        threshold=0.35,
        min_speech_duration_ms=150,
        max_speech_duration_s=9.0,
        min_silence_duration_ms=300,
        neg_threshold=0.3,
        speech_pad_ms=400
    ),
    Sensitivity.AGGRESSIVE: SileroVADOptions(
        threshold=0.05,
        min_speech_duration_ms=30,
        max_speech_duration_s=14.0,
        min_silence_duration_ms=300,
        neg_threshold=0.1,
        speech_pad_ms=600
    ),
}

# Stable-TS VAD Options Presets
STABLE_TS_VAD_PRESETS = {
    Sensitivity.BALANCED: StableTSVADOptions(
        vad=True,
        vad_threshold=0.25
    ),
    Sensitivity.CONSERVATIVE: StableTSVADOptions(
        vad=True,
        vad_threshold=0.35
    ),
    Sensitivity.AGGRESSIVE: StableTSVADOptions(
        vad=True,
        vad_threshold=0.1
    ),
}

# Faster-Whisper Engine Options Presets
FASTER_WHISPER_ENGINE_PRESETS = {
    Sensitivity.BALANCED: FasterWhisperEngineOptions(
        chunk_length=None,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,  # Must be int!
        prompt_reset_on_temperature=None,
        hotwords=None,
        multilingual=False,
        max_new_tokens=None,
        language_detection_threshold=None,
        language_detection_segments=None,
        log_progress=False
    ),
    Sensitivity.CONSERVATIVE: FasterWhisperEngineOptions(
        chunk_length=None,
        repetition_penalty=1.8,
        no_repeat_ngram_size=2,
        prompt_reset_on_temperature=None,
        hotwords=None,
        multilingual=False,
        max_new_tokens=None,
        language_detection_threshold=None,
        language_detection_segments=None,
        log_progress=False
    ),
    Sensitivity.AGGRESSIVE: FasterWhisperEngineOptions(
        chunk_length=14,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        prompt_reset_on_temperature=None,
        hotwords=None,
        multilingual=False,
        max_new_tokens=None,
        language_detection_threshold=None,
        language_detection_segments=None,
        log_progress=False
    ),
}

# Stable-TS Engine Options (same values across all sensitivities)
STABLE_TS_ENGINE_OPTIONS = StableTSEngineOptions(
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
    demucs_options=None
)

# Hallucination silence thresholds (exclusive_whisper_plus_faster_whisper)
HALLUCINATION_THRESHOLDS = {
    Sensitivity.BALANCED: 2.0,
    Sensitivity.CONSERVATIVE: 1.5,
    Sensitivity.AGGRESSIVE: 2.5,
}


def get_transcriber_preset(sensitivity: Sensitivity) -> TranscriberOptions:
    """Get transcriber options for the given sensitivity."""
    return TRANSCRIBER_PRESETS[sensitivity]


def get_decoder_preset(sensitivity: Sensitivity) -> DecoderOptions:
    """Get decoder options for the given sensitivity."""
    return DECODER_PRESETS[sensitivity]


def get_silero_vad_preset(sensitivity: Sensitivity) -> SileroVADOptions:
    """Get Silero VAD options for the given sensitivity."""
    return SILERO_VAD_PRESETS[sensitivity]


def get_stable_ts_vad_preset(sensitivity: Sensitivity) -> StableTSVADOptions:
    """Get Stable-TS VAD options for the given sensitivity."""
    return STABLE_TS_VAD_PRESETS[sensitivity]


def get_faster_whisper_engine_preset(sensitivity: Sensitivity) -> FasterWhisperEngineOptions:
    """Get Faster-Whisper engine options for the given sensitivity."""
    return FASTER_WHISPER_ENGINE_PRESETS[sensitivity]
