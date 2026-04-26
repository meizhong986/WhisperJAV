"""
Sensitivity presets for WhisperJAV Configuration System v2.0.

Contains all preset values from asr_config.json organized by sensitivity level.

v1.8.12 status
--------------
This module is LEGACY (v2 config). The runtime source of truth for preset values
is the per-engine component file:

    whisperjav/config/components/asr/faster_whisper.py
    whisperjav/config/components/asr/openai_whisper.py
    whisperjav/config/components/asr/stable_ts.py

Values in this file mirror the **faster-whisper** component for the most common
runtime paths (balanced pipeline). One known engine divergence is NOT reflected
here — openai-whisper uses ``logprob_threshold = -1.30`` for the Aggressive
sensitivity (vs. faster-whisper's ``-1.00``). For the openai-whisper canonical
values, read ``components/asr/openai_whisper.py`` directly.
"""

from .base import Sensitivity
from .decoder import DecoderOptions
from .engine import FasterWhisperEngineOptions, StableTSEngineOptions
from .transcriber import TranscriberOptions
from .vad import SileroVADOptions, StableTSVADOptions

# Common Transcriber Options Presets
TRANSCRIBER_PRESETS = {
    Sensitivity.BALANCED: TranscriberOptions(
        temperature=[0.0],                        # v1.8.10-hf3: [0.0, 0.1]→[0.0]
        compression_ratio_threshold=2.4,
        logprob_threshold=-0.85,                  # v1.8.10-hf3: -1.2→-1.00; v1.8.12: -1.00→-0.85, engine-split retune
        logprob_margin=0.0,                       # v1.8.10-hf3: 0.2→0.0
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.71,                 # v1.8.10-hf3: 0.5→0.65; v1.8.12: 0.65→0.71, engine-split retune
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=True,
        prepend_punctuations=None,
        append_punctuations=None,
        clip_timestamps=None
    ),
    Sensitivity.CONSERVATIVE: TranscriberOptions(
        temperature=[0.0],
        compression_ratio_threshold=2.2,          # v1.8.10-hf3: 2.4→2.2
        logprob_threshold=-0.70,                  # v1.8.10-hf3: -1.0→-0.80; v1.8.12: -0.80→-0.70, engine-split retune
        logprob_margin=0.0,                       # v1.8.10-hf3: 0.1→0.0
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.54,                 # v1.8.10-hf3: 0.74→0.46; v1.8.12: 0.46→0.54, engine-split retune
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=True,
        prepend_punctuations=None,
        append_punctuations=None,
        clip_timestamps=None
    ),
    Sensitivity.AGGRESSIVE: TranscriberOptions(
        temperature=[0.0, 0.17],                  # v1.8.10-hf3: [0.0, 0.3]→[0.0, 0.17]
        compression_ratio_threshold=2.6,          # v1.8.10-hf3: 3.0→2.6
        logprob_threshold=-1.00,                  # v1.8.10-hf3: -2.5→-1.00 (faster-whisper aligned; openai-whisper uses -1.30, see module docstring)
        logprob_margin=0.0,
        drop_nonverbal_vocals=False,
        no_speech_threshold=0.84,                 # v1.8.10-hf3: 0.22→0.77; v1.8.12: 0.77→0.84, engine-split retune
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
        best_of=1,                                # v1.8.10-hf3: 1→2; v1.8.12: 2→1, engine-split retune
        beam_size=2,
        patience=1.5,                             # v1.8.10-hf3: 2.0→1.6; v1.8.12: 1.6→1.5, engine-split retune
        length_penalty=None,
        prefix=None,
        suppress_tokens=None,
        suppress_blank=True,
        without_timestamps=False,
        max_initial_timestamp=0.0                 # v1.8.10-hf3: None→0.0
    ),
    Sensitivity.CONSERVATIVE: DecoderOptions(
        task="transcribe",
        language="ja",
        best_of=1,                                # v1.8.10-hf3: 1→2; v1.8.12: 2→1, engine-split retune
        beam_size=2,                              # v1.8.10-hf3: 1→2
        patience=1.0,                             # v1.8.10-hf3: 1.5→1.2; v1.8.12: 1.2→1.0, engine-split retune
        length_penalty=None,
        prefix=None,
        suppress_tokens=None,
        suppress_blank=True,
        without_timestamps=False,
        max_initial_timestamp=0.0                 # v1.8.10-hf3: None→0.0
    ),
    Sensitivity.AGGRESSIVE: DecoderOptions(
        task="transcribe",
        language="ja",
        best_of=1,                                # v1.8.10-hf3: 1→2; v1.8.12: 2→1, engine-split retune
        beam_size=3,                              # v1.8.10-hf3: kept at 2; v1.8.12: 2→3, engine-split retune
        patience=2.0,
        length_penalty=None,
        prefix=None,
        suppress_tokens=None,                     # v1.8.10-hf3: []→None
        suppress_blank=True,                      # v1.8.10-hf3: False→True
        without_timestamps=False,
        max_initial_timestamp=0.0                 # v1.8.10-hf3: None→0.0
    ),
}

# Silero VAD Options Presets
# DEPRECATED: Use v4 YAML config at config/v4/ecosystems/tools/silero-speech-segmentation.yaml
# This dict is kept for backward compatibility. Values should match the v4 YAML.
# To load in v4: registry.get("silero-speech-segmentation").get_resolved_config("balanced")
SILERO_VAD_PRESETS = {
    Sensitivity.BALANCED: SileroVADOptions(
        threshold=0.28,                       # v1.8.10-hf3: 0.18→0.28
        min_speech_duration_ms=100,
        max_speech_duration_s=7.0,            # v1.8.10-hf3: 11.0→7.0
        min_silence_duration_ms=300,
        speech_pad_ms=400
    ),
    Sensitivity.CONSERVATIVE: SileroVADOptions(
        threshold=0.41,                       # v1.8.10-hf3: 0.35→0.41
        min_speech_duration_ms=150,
        max_speech_duration_s=6.0,            # v1.8.10-hf3: 9.0→6.0
        min_silence_duration_ms=300,
        speech_pad_ms=500
    ),
    Sensitivity.AGGRESSIVE: SileroVADOptions(
        threshold=0.18,                       # v1.8.10-hf3: 0.05→0.18
        min_speech_duration_ms=30,
        max_speech_duration_s=8.0,            # v1.8.10-hf3: 14.0→8.0
        min_silence_duration_ms=300,
        speech_pad_ms=300
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
        no_repeat_ngram_size=3,                   # v1.8.10-hf3: 2→3, match Pydantic presets
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
        no_repeat_ngram_size=3,                   # v1.8.10-hf3: 2→3, match Pydantic presets
        prompt_reset_on_temperature=None,
        hotwords=None,
        multilingual=False,
        max_new_tokens=None,
        language_detection_threshold=None,
        language_detection_segments=None,
        log_progress=False
    ),
    Sensitivity.AGGRESSIVE: FasterWhisperEngineOptions(
        chunk_length=30,
        repetition_penalty=1.3,                   # v1.8.10-hf3: 1.1→1.3, match Pydantic presets
        no_repeat_ngram_size=3,                   # v1.8.10-hf3: 2→3, match Pydantic presets
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
# v1.8.10-hf1: All set to None in Pydantic presets (disabled)
HALLUCINATION_THRESHOLDS = {
    Sensitivity.BALANCED: None,
    Sensitivity.CONSERVATIVE: None,
    Sensitivity.AGGRESSIVE: None,
}


def get_transcriber_preset(sensitivity: Sensitivity) -> TranscriberOptions:
    """Get transcriber options for the given sensitivity."""
    return TRANSCRIBER_PRESETS[sensitivity]


def get_decoder_preset(sensitivity: Sensitivity) -> DecoderOptions:
    """Get decoder options for the given sensitivity."""
    return DECODER_PRESETS[sensitivity]


def get_silero_vad_preset(sensitivity: Sensitivity) -> SileroVADOptions:
    """Get Silero VAD options for the given sensitivity.

    .. deprecated:: 1.7.0
        Use v4 YAML config instead:
        ``registry.get("silero-speech-segmentation").get_resolved_config(sensitivity)``
    """
    return SILERO_VAD_PRESETS[sensitivity]


def get_stable_ts_vad_preset(sensitivity: Sensitivity) -> StableTSVADOptions:
    """Get Stable-TS VAD options for the given sensitivity."""
    return STABLE_TS_VAD_PRESETS[sensitivity]


def get_faster_whisper_engine_preset(sensitivity: Sensitivity) -> FasterWhisperEngineOptions:
    """Get Faster-Whisper engine options for the given sensitivity."""
    return FASTER_WHISPER_ENGINE_PRESETS[sensitivity]
