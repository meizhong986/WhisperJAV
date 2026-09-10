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
``threshold`` here is on the scale of the Silero build the built-in VAD is running,
which differs from the external ``silero-v3.1`` segmenter's scale — the external
presets' 0.18-0.41 thresholds would over-trigger here. The owner set the v1.9.2
values (conservative 0.5 / balanced 0.4 / aggressive 0.3) on 2026-09-09 and they
apply to every selectable build.

VERSION (v1.9.2, requirements S6-S8)
------------------------------------
``version`` selects WHICH Silero build the built-in VAD runs: 3.1 (default), 4.0 or
6.2. WhisperJAV ships all three ONNX models in the wheel; ``FasterWhisperProASR``
reads this field and installs ``whisperjav.modules.silero_vad_adapter``, which
rebinds faster-whisper's model factory. It is NOT a faster-whisper VadOptions field
and ``FasterWhisperProASR._build_vad_parameters`` whitelists it out.

Apart from ``version``, only the keys consumed by ``_build_vad_parameters`` are
defined (threshold, neg_threshold, min_speech_duration_ms, max_speech_duration_s,
min_silence_duration_ms, speech_pad_ms). Grouping params (chunk_threshold_s,
max_group_duration_s) are intentionally OMITTED — they belong to the external
segmenter, not faster-whisper VadOptions.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from whisperjav.config.components.base import VADComponent, register_vad
from whisperjav.modules.silero_vad_adapter import DEFAULT_VAD_VERSION, VAD_VERSIONS


class FasterWhisperVADOptions(BaseModel):
    """faster-whisper native VadOptions (maps 1:1 to faster_whisper.vad.VadOptions)."""

    version: str = Field(
        DEFAULT_VAD_VERSION,
        description="Which Silero VAD build faster-whisper's built-in VAD runs: "
                    f"{' | '.join(VAD_VERSIONS)}. WhisperJAV ships all three ONNX models "
                    "and selects one at runtime (v1.9.2, requirement S6-S8). NOT a "
                    "faster-whisper VadOptions field -- consumed by "
                    "FasterWhisperProASR, which installs the adapter.",
    )
    threshold: float = Field(
        0.40,
        ge=0.0, le=1.0,
        description="Speech probability threshold for the selected Silero VAD build. "
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
        6.0,
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

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        if v not in VAD_VERSIONS:
            raise ValueError(f"version must be one of {VAD_VERSIONS}, got {v!r}")
        return v


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

    # === Presets ===
    # threshold: the owner settled these for v1.9.2 (2026-09-09) -- conservative
    #   0.5 / balanced 0.4 / aggressive 0.3, the SAME range for every Silero
    #   build. Do not re-measure them.  (They replace the T2 2026-06-29 values
    #   0.45 / 0.40 / 0.25.)
    # version: 3.1 on every sensitivity (requirement S8).
    # max_speech_duration_s: the owner set these for v1.9.2 (2026-09-10, O3) after his
    #   feature-length manual test -- conservative 7.0 / balanced 6.0 / aggressive 6.0.
    #   They replace the T2 2026-06-29 values 20.0 / 15.0 / 9.0. This is a subtitle-
    #   granularity knob: it forces a split inside a long unbroken speech chunk and does
    #   NOT change the encoder-pass count.
    # min_speech_duration_ms: aggressive raised 30 -> 80 (owner O3, 2026-09-10);
    #   conservative and balanced keep the JAV-tuned silero values.
    # min_silence / speech_pad: inherited from the JAV-tuned silero presets
    #   (model-agnostic millisecond params).
    presets = {
        "conservative": FasterWhisperVADOptions(
            threshold=0.50,
            min_speech_duration_ms=150,
            max_speech_duration_s=7.0,
            min_silence_duration_ms=300,
            speech_pad_ms=500,
        ),
        "balanced": FasterWhisperVADOptions(
            threshold=0.40,
            min_speech_duration_ms=100,
            max_speech_duration_s=6.0,
            min_silence_duration_ms=300,
            speech_pad_ms=400,
        ),
        "aggressive": FasterWhisperVADOptions(
            threshold=0.30,
            min_speech_duration_ms=80,
            max_speech_duration_s=6.0,
            min_silence_duration_ms=300,
            speech_pad_ms=300,
        ),
    }
