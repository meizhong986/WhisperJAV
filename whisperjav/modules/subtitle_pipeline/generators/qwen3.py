"""
Qwen3-ASR TextGenerator adapter.

Wraps QwenASR's decoupled text-only mode (load_model_text_only / transcribe_text_only)
behind the TextGenerator protocol.  No new functionality — pure structural extraction.

Lifecycle design (audit C1 resolution):
    The adapter does NOT keep a persistent QwenASR instance.  It creates one in load()
    and destroys it in unload().  This prevents the stale closure problem where a
    reloaded model still references a deleted CUDA tensor from a previous load cycle.

VRAM cleanup (audit M5 resolution):
    unload() uses safe_cuda_cleanup() from whisperjav.utils.gpu_utils instead of
    duplicating inline CUDA cache cleanup logic.
"""

from pathlib import Path
from typing import Any, Optional

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult
from whisperjav.utils.logger import logger


class Qwen3TextGenerator:
    """
    TextGenerator backed by Qwen3-ASR text-only mode.

    Produces raw transcription text (no timestamps) from audio files.
    Manages its own QwenASR lifecycle via load()/unload() for VRAM swapping.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: str = "Japanese",
        repetition_penalty: float = 1.1,
        max_tokens_per_audio_second: float = 20.0,
        attn_implementation: str = "auto",
    ):
        """
        Store configuration for deferred QwenASR construction.

        Args:
            model_id: HuggingFace model ID for Qwen3-ASR.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
            batch_size: Maximum inference batch size.  Can be higher than coupled
                mode since text-only uses ~1.2GB less VRAM (audit M3).
            max_new_tokens: Maximum tokens per utterance.
            language: Language for transcription (e.g., 'Japanese', 'ja').
            repetition_penalty: HF generation_config penalty (1.0 = off).
            max_tokens_per_audio_second: Dynamic per-scene token budget scaling.
            attn_implementation: Attention implementation strategy.
        """
        self._config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "language": language,
            "repetition_penalty": repetition_penalty,
            "max_tokens_per_audio_second": max_tokens_per_audio_second,
            "attn_implementation": attn_implementation,
        }
        self._asr = None  # QwenASR instance, created in load()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    def load(self) -> None:
        """
        Create QwenASR instance and load the text-only model into GPU.

        Fresh instance each time — prevents stale closure (audit C1).
        """
        if self._loaded:
            logger.debug("[Qwen3TextGenerator] Already loaded")
            return

        from whisperjav.modules.qwen_asr import QwenASR

        cfg = self._config
        self._asr = QwenASR(
            model_id=cfg["model_id"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            batch_size=cfg["batch_size"],
            max_new_tokens=cfg["max_new_tokens"],
            language=cfg["language"],
            use_aligner=False,  # Text-only — no aligner
            timestamps="none",
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens_per_audio_second=cfg["max_tokens_per_audio_second"],
            attn_implementation=cfg["attn_implementation"],
        )
        self._asr.load_model_text_only()
        self._loaded = True
        logger.info("[Qwen3TextGenerator] Model loaded (text-only)")

    def unload(self) -> None:
        """
        Unload the model and release VRAM.

        Uses safe_cuda_cleanup() for centralized CUDA cache management (audit M5).
        """
        if not self._loaded:
            return

        if self._asr is not None:
            self._asr.unload_model()
            self._asr = None

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup

        safe_cuda_cleanup()
        self._loaded = False
        logger.info("[Qwen3TextGenerator] Model unloaded")

    def generate(
        self,
        audio_path: Path,
        language: str = "ja",
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'ja', 'en').
            context: Optional context string to improve transcription.

        Returns:
            TranscriptionResult with transcribed text.
        """
        results = self.generate_batch(
            audio_paths=[audio_path],
            language=language,
            contexts=[context] if context else None,
            **kwargs,
        )
        return results[0]

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a batch of audio files to text.

        Delegates to QwenASR.transcribe_text_only() which handles per-scene
        progress reporting and dynamic token budgets.

        Args:
            audio_paths: Paths to audio files (one per scene).
            language: Language code applied to all scenes.
            contexts: Optional per-scene context strings.

        Returns:
            List of TranscriptionResults (one per audio file).
        """
        if not self._loaded:
            raise RuntimeError("Qwen3TextGenerator.generate_batch() called before load(). Call load() first.")

        # Extract audio durations if provided in kwargs (for progress display)
        audio_durations = kwargs.get("audio_durations")

        texts = self._asr.transcribe_text_only(
            audio_paths=list(audio_paths),
            contexts=contexts,
            language=language,
            audio_durations=audio_durations,
        )

        # Wrap raw text strings in TranscriptionResult
        return [
            TranscriptionResult(
                text=t,
                language=language,
                metadata={"generator": "qwen3", "audio_path": str(p)},
            )
            for t, p in zip(texts, audio_paths)
        ]

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
