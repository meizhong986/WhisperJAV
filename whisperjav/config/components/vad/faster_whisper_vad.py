"""
Faster-Whisper Native VAD Component (v1.9.0).

Voice Activity Detection performed by faster-whisper's BUILT-IN Silero VAD
(``transcribe(vad_filter=True, vad_parameters=...)``), as opposed to WhisperJAV's
external per-group Speech Segmenter.

WHY THIS EXISTS
---------------
The external segmenter splits each scene into many small VAD groups and calls
``whisper_model.transcribe()`` once PER GROUP. faster-whisper pads every call to a
30s encoder window, so N small groups = N near-empty encoder passes — the dominant
cost behind the balanced-pipeline 2-3x slowdown. Native VAD lets faster-whisper pack
speech into full 30s windows in a SINGLE call per scene (~1 encoder pass per 30s),
matching the de-facto-fast references (Faster-Whisper-XXL, the v0.7b notebook).

These presets are the SINGLE SOURCE OF TRUTH for the native-VAD VadOptions.

SCALE NOTE (important)
----------------------
``threshold`` here is for faster-whisper's BUNDLED Silero VAD, whose probability
scale differs from the external ``silero-v3.1`` segmenter. The values below were set
deliberately (T2 decision, 2026-06-29) and must NOT be copied from the external
silero presets (whose 0.18-0.41 thresholds would over-trigger here). The non-threshold
millisecond parameters (min_speech / min_silence / speech_pad) ARE model-agnostic and
are inherited from the JAV-tuned silero presets.

Only the keys consumed by ``FasterWhisperProASR._build_vad_parameters`` are defined
(threshold, neg_threshold, min_speech_duration_ms, max_speech_duration_s,
min_silence_duration_ms, speech_pad_ms). Grouping params (chunk_threshold_s,
max_group_duration_s) are intentionally OMITTED — they belong to the external
segmenter, not faster-whisper VadOptions.
"""

from typing import Optional

from pydantic import BaseModel, Field

from whisperjav.config.components.base import VADComponent, register_vad


class FasterWhisperVADOptions(BaseModel):
    """faster-whisper native VadOptions (maps 1:1 to faster_whisper.vad.VadOptions)."""

    threshold: float = Field(
        0.40,
        ge=0.0, le=1.0,
        description="Speech probability threshold for faster-whisper's bundled Silero VAD. "
                    "Lower = more sensitive (captures quieter/breathier speech).",
    )
    neg_threshold: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Lower hysteresis threshold. None = faster-whisper default (threshold - 0.15).",
    )
    min_speech_duration_ms: int = Field(
        100,
        ge=0, le=5000,
        description="Speech chunks shorter than this are discarded.",
    )
    max_speech_duration_s: float = Field(
        15.0,
        ge=0.0, le=300.0,
        description="Maximum duration of a single speech chunk before a forced split. "
                    "Does NOT affect encoder-pass count (faster-whisper still batches "
                    "into 30s windows) — purely a subtitle-granularity knob.",
    )
    min_silence_duration_ms: int = Field(
        300,
        ge=0, le=5000,
        description="Silence shorter than this does not split a speech chunk.",
    )
    speech_pad_ms: int = Field(
        400,
        ge=0, le=2000,
        description="Padding added around each detected speech chunk.",
    )


@register_vad
class FasterWhisperVAD(VADComponent):
    """faster-whisper built-in VAD (vad_filter=True). v1.9.0 balanced default."""

    # === Metadata ===
    name = "faster_whisper_vad"
    display_name = "Faster-Whisper Native VAD"
    description = (
        "faster-whisper's built-in Silero VAD (vad_filter=True). One transcribe() "
        "call per scene — eliminates the per-group encoder-pass overhead."
    )
    version = "1.0.0"
    tags = ["vad", "native", "faster-whisper", "silero"]

    # === VAD-specific ===
    # Native VAD is a faster-whisper feature; only the faster_whisper ASR can use it.
    compatible_asr = ["faster_whisper"]

    # === Schema ===
    Options = FasterWhisperVADOptions

    # === Presets (T2 decision, 2026-06-29) ===
    # threshold / max_speech_duration_s: locked T2 values, calibrated to the
    #   fast references (XXL 0.45, notebook 0.35-0.40) and JAV recall needs.
    # min_speech / min_silence / speech_pad: inherited from the JAV-tuned silero
    #   presets (model-agnostic millisecond params).
    presets = {
        "conservative": FasterWhisperVADOptions(
            threshold=0.45,
            min_speech_duration_ms=150,
            max_speech_duration_s=20.0,
            min_silence_duration_ms=300,
            speech_pad_ms=500,
        ),
        "balanced": FasterWhisperVADOptions(
            threshold=0.40,
            min_speech_duration_ms=100,
            max_speech_duration_s=15.0,
            min_silence_duration_ms=300,
            speech_pad_ms=400,
        ),
        "aggressive": FasterWhisperVADOptions(
            threshold=0.25,
            min_speech_duration_ms=30,
            max_speech_duration_s=9.0,
            min_silence_duration_ms=300,
            speech_pad_ms=300,
        ),
    }
