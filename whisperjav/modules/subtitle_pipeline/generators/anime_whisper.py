"""
anime-whisper TextGenerator adapter.

Wraps the HuggingFace Transformers ASR pipeline for litagin/anime-whisper
behind the TextGenerator protocol.  anime-whisper is a Whisper fine-tune
(kotoba-whisper-v2.0 base) specialized for anime/visual-novel dialogue.

Key model constraints (from model card):
    - NO initial prompt / context — causes hallucinations.  The ``context``
      parameter in generate() is intentionally IGNORED.
    - no_repeat_ngram_size >= 5 required to prevent repetition loops.
    - Japanese only.
    - Output is text-only (no timestamps).  Timestamps come from the
      Qwen3 ForcedAligner downstream in the subtitle pipeline.

Lifecycle design follows Qwen3TextGenerator pattern:
    Fresh HF pipeline per load()/unload() cycle.

VRAM cleanup:
    unload() uses safe_cuda_cleanup() from whisperjav.utils.gpu_utils.
"""

import os
from pathlib import Path
from typing import Any, Optional

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult
from whisperjav.utils.logger import logger


class AnimeWhisperGenerator:
    """
    TextGenerator backed by litagin/anime-whisper (HuggingFace Whisper fine-tune).

    Produces raw transcription text (no timestamps) from audio files.
    Manages its own HF pipeline lifecycle via load()/unload() for VRAM swapping.
    """

    def __init__(
        self,
        model_id: str = "litagin/anime-whisper",
        device: str = "auto",
        dtype: str = "auto",
        no_repeat_ngram_size: int = 5,
        max_new_tokens: int = 448,
    ):
        """
        Store configuration for deferred HF pipeline construction.

        Args:
            model_id: HuggingFace model ID or local path for anime-whisper.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
            no_repeat_ngram_size: N-gram repetition prevention (model card: >=5).
            max_new_tokens: Maximum generated tokens per utterance.
                Model card demo uses 64 (for 15s clips).  For subtitle frames
                (6-48s), 448 is Whisper large's architectural max.
        """
        self._config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_new_tokens": max_new_tokens,
        }
        self._pipe = None  # HF ASR pipeline, created in load()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    # ------------------------------------------------------------------
    # Device / dtype detection (mirrors Qwen3TextGenerator pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device(device: str) -> str:
        """Resolve 'auto' device to concrete device string."""
        if device != "auto":
            return device
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _detect_dtype(device: str, dtype: str):
        """Resolve 'auto' dtype based on device capability."""
        import torch
        if dtype != "auto":
            return getattr(torch, dtype, torch.float32)
        if "cuda" in device:
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Create HF ASR pipeline and load the anime-whisper model into GPU.

        Fresh pipeline each time — prevents stale state across load/unload cycles.
        """
        if self._loaded:
            logger.debug("[AnimeWhisperGenerator] Already loaded")
            return

        # Suppress TF/oneDNN warnings before importing transformers
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        from transformers import pipeline

        cfg = self._config
        device = self._detect_device(cfg["device"])
        dtype = self._detect_dtype(device, cfg["dtype"])

        logger.info("[AnimeWhisperGenerator] Loading model...")
        logger.info("  Model:  %s", cfg["model_id"])
        logger.info("  Device: %s", device)
        logger.info("  Dtype:  %s", dtype)

        import time
        start = time.time()

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg["model_id"],
            device=device,
            dtype=dtype,
        )

        elapsed = time.time() - start
        logger.info("[AnimeWhisperGenerator] Model loaded (%.1fs)", elapsed)
        self._loaded = True

    def unload(self) -> None:
        """
        Unload the model and release VRAM.

        Uses safe_cuda_cleanup() for centralized CUDA cache management.
        """
        if not self._loaded:
            return

        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup
        safe_cuda_cleanup()

        self._loaded = False
        logger.info("[AnimeWhisperGenerator] Model unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
            audio_path: Path to the audio file (WAV, 16kHz mono expected).
            language: Language code (ignored — model is Japanese-only).
            context: IGNORED.  anime-whisper's model card explicitly warns
                that initial prompts cause hallucinations.

        Returns:
            TranscriptionResult with transcribed text.
        """
        if not self._loaded:
            raise RuntimeError(
                "AnimeWhisperGenerator.generate() called before load(). "
                "Call load() first."
            )

        if context:
            logger.debug(
                "[AnimeWhisperGenerator] context parameter ignored "
                "(anime-whisper model constraint: no initial prompt)"
            )

        cfg = self._config

        generate_kwargs = {
            "language": "Japanese",
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
            "repetition_penalty": 1.0,
            "max_new_tokens": cfg["max_new_tokens"],
        }

        try:
            # return_timestamps=True activates Whisper's native long-form
            # decoding (sequential 30s windows) for audio >30s.  We discard
            # the timestamps and only take the text.  This avoids the
            # experimental chunk_length_s path which crashes on Windows
            # (0xC0000409 heap corruption in CUDA kernels).
            result = self._pipe(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )
            text = result.get("text", "").strip()
        except Exception as e:
            logger.error(
                "[AnimeWhisperGenerator] Transcription failed for %s: %s",
                audio_path.name if hasattr(audio_path, "name") else audio_path,
                e,
            )
            text = ""

        return TranscriptionResult(
            text=text,
            language="ja",
            metadata={
                "generator": "anime-whisper",
                "audio_path": str(audio_path),
            },
        )

    def generate_batch(
        self,
        audio_paths: list[Path],
        language: str = "ja",
        contexts: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a batch of audio files to text.

        HF ASR pipeline processes one file at a time, so this iterates
        sequentially.  The VRAM lifecycle (load/unload) is managed by
        the orchestrator, not per-call.

        Args:
            audio_paths: Paths to audio files (one per frame).
            language: Language code (ignored — Japanese-only).
            contexts: IGNORED (anime-whisper constraint).

        Returns:
            List of TranscriptionResults (one per audio file).
        """
        if not self._loaded:
            raise RuntimeError(
                "AnimeWhisperGenerator.generate_batch() called before load(). "
                "Call load() first."
            )

        results = []
        for i, audio_path in enumerate(audio_paths):
            logger.debug(
                "[AnimeWhisperGenerator] Generating %d/%d: %s",
                i + 1, len(audio_paths),
                audio_path.name if hasattr(audio_path, "name") else audio_path,
            )
            result = self.generate(audio_path, language=language)
            results.append(result)

        return results

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
