"""
Faster-Whisper ASR Component.

High-performance Whisper implementation using CTranslate2.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class FasterWhisperOptions(BaseModel):
    """Combined decoder and transcriber options for Faster-Whisper."""

    # === Decoder Options ===
    task: str = Field("transcribe", description="Task: 'transcribe' or 'translate'")
    language: str = Field("ja", description="Language code")
    beam_size: int = Field(5, ge=1, le=20, description="Beam size for decoding")
    best_of: int = Field(5, ge=1, le=10, description="Number of candidates")
    patience: float = Field(1.2, ge=0.0, le=3.0, description="Beam search patience")
    length_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Length penalty")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")
    no_repeat_ngram_size: int = Field(0, ge=0, le=10, description="N-gram size to avoid repeating")
    suppress_blank: bool = Field(True, description="Suppress blank outputs")
    suppress_tokens: List[int] = Field(default_factory=lambda: [-1], description="Token IDs to suppress")

    # === Transcriber Options ===
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Sampling temperature")
    compression_ratio_threshold: float = Field(2.4, ge=1.0, le=5.0, description="Compression ratio threshold")
    logprob_threshold: float = Field(-1.0, ge=-5.0, le=0.0, description="Log probability threshold")
    no_speech_threshold: float = Field(0.6, ge=0.0, le=1.0, description="No speech probability threshold")
    condition_on_previous_text: bool = Field(True, description="Condition on previous text")

    # === Engine Options ===
    hallucination_silence_threshold: Optional[float] = Field(
        None, ge=0.0, le=5.0,
        description="Skip silent periods longer than this (seconds)"
    )
    initial_prompt: Optional[str] = Field(None, description="Initial prompt for context")
    prefix: Optional[str] = Field(None, description="Prefix for each segment")
    word_timestamps: bool = Field(False, description="Include word-level timestamps")


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
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["silero", "faster_whisper_vad", "none"]

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = FasterWhisperOptions

    # === Presets ===
    presets = {
        "conservative": FasterWhisperOptions(
            beam_size=3,
            best_of=3,
            patience=1.5,
            temperature=0.0,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.6,
            logprob_threshold=-0.8,
            hallucination_silence_threshold=0.3,
        ),
        "balanced": FasterWhisperOptions(
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            hallucination_silence_threshold=0.2,
        ),
        "aggressive": FasterWhisperOptions(
            beam_size=7,
            best_of=7,
            patience=1.0,
            temperature=0.0,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.2,
            logprob_threshold=-1.2,
            hallucination_silence_threshold=0.1,
        ),
    }
