"""
anime-whisper TextGenerator adapter.

Uses the low-level WhisperProcessor + WhisperForConditionalGeneration API
(NOT the HF ``pipeline()`` wrapper, which crashes with 0xC0000409 on
torch 2.9+ / transformers 4.57+ / Windows / CUDA).

Key model constraints (from developer's demo app + model card):
    - NO initial prompt / context — causes hallucinations.  The ``context``
      parameter in generate() is intentionally IGNORED.
    - no_repeat_ngram_size = 0 recommended by model card (default).
      5-10 is a fallback if repetition hallucinations are noticeable.
    - Greedy decoding: do_sample=False, num_beams=1 (litagin demo).
    - Japanese only.
    - Output is text-only (no timestamps).  Timestamps come from the
      Qwen3 ForcedAligner downstream in the subtitle pipeline.
    - VRAM: ~4GB (Whisper large-v2, ~1.55B params, float16).

Lifecycle design follows Qwen3TextGenerator pattern:
    Fresh model per load()/unload() cycle.

VRAM cleanup:
    unload() uses safe_cuda_cleanup() from whisperjav.utils.gpu_utils.
"""

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult
from whisperjav.utils.logger import logger


class AnimeWhisperGenerator:
    """
    TextGenerator backed by litagin/anime-whisper (HuggingFace Whisper fine-tune).

    Uses WhisperProcessor + WhisperForConditionalGeneration directly,
    loading audio via librosa and running model.generate() under torch.no_grad().
    """

    def __init__(
        self,
        model_id: str = "litagin/anime-whisper",
        device: str = "auto",
        dtype: str = "auto",
        no_repeat_ngram_size: int = 0,
        max_new_tokens: int = 444,
    ):
        """
        Store configuration for deferred model construction.

        Args:
            model_id: HuggingFace model ID or local path for anime-whisper.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
            no_repeat_ngram_size: N-gram repetition prevention (model card default: 0 = disabled).
                Set to 5-10 if repetition hallucinations are noticeable.
            max_new_tokens: Maximum generated tokens per utterance.
                444 = max safe value (max_target_positions 448 minus 4 special
                tokens).  Must be set explicitly because the model's
                generation_config defaults to 4096, which exceeds 448.
                litagin's demo uses 64 for 15s clips; 444 is appropriate
                for our 6-48s framed audio.
        """
        self._config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "max_new_tokens": max_new_tokens,
        }
        self._processor = None
        self._model = None
        self._device = None   # Resolved device string (e.g. "cuda:0")
        self._dtype = None    # Resolved torch dtype (e.g. torch.float16)
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
        Load WhisperProcessor and WhisperForConditionalGeneration.

        Uses the low-level API (not pipeline()) which is proven stable
        on torch 2.9+ / Windows / CUDA.
        """
        if self._loaded:
            logger.debug("[AnimeWhisperGenerator] Already loaded")
            return

        # Suppress TF/oneDNN warnings before importing transformers
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        cfg = self._config
        device = self._detect_device(cfg["device"])
        dtype = self._detect_dtype(device, cfg["dtype"])

        logger.info("[AnimeWhisperGenerator] Loading model...")
        logger.info("  Model:  %s", cfg["model_id"])
        logger.info("  Device: %s", device)
        logger.info("  Dtype:  %s", dtype)

        import time
        start = time.time()

        self._processor = WhisperProcessor.from_pretrained(cfg["model_id"])
        self._model = WhisperForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            torch_dtype=dtype,
        ).to(device)

        self._device = device
        self._dtype = dtype
        self._loaded = True

        elapsed = time.time() - start
        logger.info("[AnimeWhisperGenerator] Model loaded (%.1fs)", elapsed)

    def unload(self) -> None:
        """
        Unload the model and release VRAM.

        Uses safe_cuda_cleanup() for centralized CUDA cache management.
        """
        if not self._loaded:
            return

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup
        safe_cuda_cleanup()

        self._device = None
        self._dtype = None
        self._loaded = False
        logger.info("[AnimeWhisperGenerator] Model unloaded")

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: Path) -> np.ndarray:
        """
        Load audio file as float32 numpy array at 16kHz mono.

        Uses librosa (proven path from working anime-whisper scripts).
        Falls back to soundfile if librosa fails.
        """
        import librosa

        try:
            audio, _sr = librosa.load(str(audio_path), sr=16000)
            return audio
        except Exception as e:
            logger.warning(
                "[AnimeWhisperGenerator] librosa failed for %s: %s — trying soundfile",
                audio_path.name if hasattr(audio_path, "name") else audio_path, e,
            )
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)

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

        import torch

        cfg = self._config

        # Load and preprocess audio (has its own fallback logic)
        audio = self._load_audio(audio_path)

        # Extract features via WhisperProcessor and cast to model dtype
        # (processor outputs float32; model may be float16 on CUDA)
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device=self._device, dtype=self._dtype
        )

        # Generate text tokens — match litagin's demo pattern:
        # greedy decoding (num_beams=1, do_sample=False).
        # max_new_tokens MUST be set explicitly — the model's
        # generation_config defaults to 4096, exceeding max_target_positions=448.
        with torch.no_grad():
            gen_kwargs = {
                "input_features": input_features,
                "language": "ja",
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
                "no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
                "max_new_tokens": cfg["max_new_tokens"],
            }
            generated_ids = self._model.generate(**gen_kwargs)

        # Decode tokens to text
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

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

        Processes one file at a time (Whisper encoder takes fixed-size
        mel spectrogram per utterance).  The VRAM lifecycle (load/unload)
        is managed by the orchestrator, not per-call.

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
