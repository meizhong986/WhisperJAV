"""
Stable-TS ASR Component.

OpenAI Whisper with stable-ts enhancements for better timestamp alignment.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from whisperjav.config.components.base import ASRComponent, register_asr


class StableTSOptions(BaseModel):
    """Combined decoder, transcriber, and stable-ts engine options."""

    # === Decoder Options ===
    task: str = Field("transcribe", description="Task: 'transcribe' or 'translate'")
    language: str = Field("ja", description="Language code")
    beam_size: int = Field(5, ge=1, le=20, description="Beam size for decoding")
    best_of: int = Field(5, ge=1, le=10, description="Number of candidates")
    patience: float = Field(1.2, ge=0.0, le=3.0, description="Beam search patience")
    length_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Length penalty")
    suppress_blank: bool = Field(True, description="Suppress blank outputs")
    suppress_tokens: List[int] = Field(default_factory=lambda: [-1], description="Token IDs to suppress")

    # === Transcriber Options ===
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Sampling temperature")
    compression_ratio_threshold: float = Field(2.4, ge=1.0, le=5.0, description="Compression ratio threshold")
    logprob_threshold: float = Field(-1.0, ge=-5.0, le=0.0, description="Log probability threshold")
    no_speech_threshold: float = Field(0.6, ge=0.0, le=1.0, description="No speech probability threshold")
    condition_on_previous_text: bool = Field(True, description="Condition on previous text")
    initial_prompt: Optional[str] = Field(None, description="Initial prompt for context")

    # === Stable-TS Engine Options ===
    regroup: bool = Field(True, description="Regroup segments for better timing")
    suppress_silence: bool = Field(True, description="Suppress silence in output")
    suppress_word_ts: bool = Field(True, description="Suppress word timestamps")
    use_word_position: bool = Field(True, description="Use word position for alignment")
    q_levels: int = Field(20, ge=1, le=100, description="Quantization levels")
    k_size: int = Field(5, ge=1, le=20, description="Kernel size for smoothing")
    denoiser: Optional[str] = Field(None, description="Denoiser to use")
    denoiser_options: Optional[dict] = Field(None, description="Denoiser options")
    vad: bool = Field(False, description="Use VAD preprocessing")
    vad_threshold: float = Field(0.35, ge=0.0, le=1.0, description="VAD threshold")
    min_word_dur: float = Field(0.1, ge=0.0, le=1.0, description="Minimum word duration")
    nonspeech_error: float = Field(0.3, ge=0.0, le=1.0, description="Non-speech error tolerance")
    only_voice_freq: bool = Field(False, description="Filter to voice frequencies only")
    prepend_punctuations: str = Field("\"'([{-", description="Punctuations to prepend")
    append_punctuations: str = Field("\"'.,!?:)]}", description="Punctuations to append")


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
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["none"]  # Stable-TS has built-in VAD

    # === Compute ===
    default_device = "cuda"
    default_compute_type = "float16"

    # === Schema ===
    Options = StableTSOptions

    # === Presets ===
    presets = {
        "conservative": StableTSOptions(
            beam_size=3,
            best_of=3,
            patience=1.5,
            temperature=0.0,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.6,
            logprob_threshold=-0.8,
            vad_threshold=0.5,
            nonspeech_error=0.4,
        ),
        "balanced": StableTSOptions(
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            vad_threshold=0.35,
            nonspeech_error=0.3,
        ),
        "aggressive": StableTSOptions(
            beam_size=7,
            best_of=7,
            patience=1.0,
            temperature=0.0,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.2,
            logprob_threshold=-1.2,
            vad_threshold=0.25,
            nonspeech_error=0.2,
        ),
    }
