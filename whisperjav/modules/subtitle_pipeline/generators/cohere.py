"""
Cohere Transcribe-03-2026 TextGenerator adapter.

Uses transformers AutoProcessor + AutoModel with trust_remote_code=True
because the class CohereAsrForConditionalGeneration is NOT exposed by
transformers 4.57.6 natively; it ships in the model repo itself
(verified by _cohere_spike on 2026-05-01).

Output is text only.  Cohere Transcribe-03-2026 does NOT provide native
word-level timestamps — see HF discussion #19.  The audio_chunk_index
returned by the processor is a chunking index used by processor.decode()
to reassemble text from overlapping audio chunks; it is NOT a timing
signal.  Word-level timestamps are produced downstream by the Qwen3
ForcedAligner-0.6B (the default per D7).  Users may disable the aligner
via Customize Parameters → aligner=none, falling back to VAD-derived
segment timing only.

HuggingFace gated repo:
    The model is gated.  Loading requires (a) the user has accepted the
    terms at the model page, and (b) HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
    is set in the environment.  load() pre-flights the token and raises a
    helpful diagnostic if either step is missing.

VRAM:
    ~4-8 GB at FP16 (per Cohere model card).  In the orchestrator's
    load/unload swap pattern, peak VRAM is bounded by max(generator,
    aligner), not the sum — so Cohere + Qwen3 ForcedAligner-0.6B never
    coexist in VRAM and Cohere fits comfortably on 8-10 GB cards.

Lifecycle design follows AnimeWhisperGenerator pattern:
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


_MODEL_PAGE_URL = "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026"
_TOKEN_PAGE_URL = "https://huggingface.co/settings/tokens"


class CohereTextGenerator:
    """
    TextGenerator backed by CohereLabs/cohere-transcribe-03-2026 (gated).

    Uses AutoProcessor + AutoModel with trust_remote_code=True, loading
    audio via librosa and running model.generate() under torch.no_grad().
    """

    def __init__(
        self,
        model_id: str = "CohereLabs/cohere-transcribe-03-2026",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "ja",
        punctuation: bool = True,
        max_new_tokens: int = 512,
        trust_remote_code: bool = True,
    ):
        """
        Store configuration for deferred model construction.

        Args:
            model_id: HuggingFace model ID or local path for Cohere Transcribe.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
            language: Language code passed to processor.decode (Cohere supports
                14 languages; default 'ja' for WhisperJAV).
            punctuation: Whether to request punctuation from the processor
                (Cohere processor accepts this kwarg per repka3 PoC).
            max_new_tokens: Maximum generated tokens per utterance.  Default
                512 — Cohere has no Whisper-style max_target_positions=448
                constraint, but 512 is a safe ceiling for JAV-length monologues.
            trust_remote_code: Required True until transformers exposes
                CohereAsrForConditionalGeneration natively.
        """
        self._config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "language": language,
            "punctuation": punctuation,
            "max_new_tokens": max_new_tokens,
            "trust_remote_code": trust_remote_code,
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
    # Device / dtype detection (mirrors AnimeWhisperGenerator pattern)
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
    # HF gating pre-flight
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hf_access() -> None:
        """
        Pre-flight: verify an HF token is present in the environment.

        Cohere Transcribe is a gated repo; without a token, transformers
        returns a less-helpful error.  Catch the missing-token case here
        so we can raise a diagnostic that points users to the FAQ.

        Note: presence of HF_TOKEN does NOT guarantee terms acceptance —
        a 403 GatedRepoError can still surface during from_pretrained.
        That case is handled in _format_load_error.
        """
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "Cohere-Transcribe requires HF_TOKEN to be set in the environment.\n"
                "\n"
                "Setup (one-time):\n"
                f"  1. Visit {_MODEL_PAGE_URL}\n"
                "     and click 'Agree and access repository'.\n"
                f"  2. Create a token at {_TOKEN_PAGE_URL}\n"
                "     ('Read' permission is sufficient).\n"
                "  3. Set HF_TOKEN in your environment:\n"
                "     Windows (persistent): setx HF_TOKEN hf_xxxxxxxxxxxx\n"
                "                           — restart the terminal/GUI to take effect\n"
                "     Windows (current session): $env:HF_TOKEN = \"hf_xxxxxxxxxxxx\"\n"
                "     macOS/Linux: export HF_TOKEN=hf_xxxxxxxxxxxx\n"
                "\n"
                "See FAQ: 'Using Cohere-Transcribe (preview, opt-in)'"
            )

    @staticmethod
    def _format_load_error(exc: Exception) -> str:
        """Produce a helpful diagnostic for known load-failure modes."""
        msg = str(exc).lower()
        if any(t in msg for t in ("gated", "403", "access to model", "not authorized")):
            return (
                "Failed to load Cohere-Transcribe — access to the gated repo was denied.\n"
                "\n"
                "Setup (one-time):\n"
                f"  1. Visit {_MODEL_PAGE_URL}\n"
                "     and click 'Agree and access repository'.\n"
                f"  2. Create a token at {_TOKEN_PAGE_URL}\n"
                "     ('Read' permission is sufficient).\n"
                "  3. Set HF_TOKEN in your environment:\n"
                "     Windows (persistent): setx HF_TOKEN hf_xxxxxxxxxxxx\n"
                "                           — restart the terminal/GUI to take effect\n"
                "     Windows (current session): $env:HF_TOKEN = \"hf_xxxxxxxxxxxx\"\n"
                "     macOS/Linux: export HF_TOKEN=hf_xxxxxxxxxxxx\n"
                "\n"
                "See FAQ: 'Using Cohere-Transcribe (preview, opt-in)'\n"
                f"\nOriginal error: {exc}"
            )
        return f"Failed to load Cohere-Transcribe: {exc}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load AutoProcessor and AutoModel for Cohere Transcribe.

        Pre-flights HF_TOKEN, then loads via trust_remote_code=True.
        Raises RuntimeError with a diagnostic message on gated-repo
        access failure.
        """
        if self._loaded:
            logger.debug("[CohereTextGenerator] Already loaded")
            return

        self._check_hf_access()

        # Suppress TF/oneDNN warnings before importing transformers
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = self._config
        device = self._detect_device(cfg["device"])
        dtype = self._detect_dtype(device, cfg["dtype"])

        logger.info("[CohereTextGenerator] Loading model...")
        logger.info("  Model:  %s", cfg["model_id"])
        logger.info("  Device: %s", device)
        logger.info("  Dtype:  %s", dtype)

        import time
        start = time.time()

        try:
            self._processor = AutoProcessor.from_pretrained(
                cfg["model_id"],
                trust_remote_code=cfg["trust_remote_code"],
            )
            self._model = AutoModel.from_pretrained(
                cfg["model_id"],
                dtype=dtype,
                trust_remote_code=cfg["trust_remote_code"],
            ).to(device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(self._format_load_error(exc)) from exc

        self._device = device
        self._dtype = dtype
        self._loaded = True

        elapsed = time.time() - start
        logger.info("[CohereTextGenerator] Model loaded (%.1fs)", elapsed)

        if "cuda" in device:
            try:
                vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logger.info("[CohereTextGenerator] Peak VRAM after load: %.2f GB", vram_gb)
            except Exception:
                pass

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
        logger.info("[CohereTextGenerator] Model unloaded")

    # ------------------------------------------------------------------
    # Audio loading (shared pattern with AnimeWhisperGenerator)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: Path) -> np.ndarray:
        """
        Load audio file as float32 numpy array at 16 kHz mono.

        Uses librosa as the primary path; falls back to soundfile if librosa
        fails on a particular file.
        """
        import librosa

        try:
            audio, _sr = librosa.load(str(audio_path), sr=16000)
            return audio
        except Exception as e:
            logger.warning(
                "[CohereTextGenerator] librosa failed for %s: %s — trying soundfile",
                audio_path.name if hasattr(audio_path, "name") else audio_path, e,
            )
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
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

        Implementation pattern verified against repka3's reference PoC
        (cohere_transcript.py at the same repo cited in the v1.8.14 plan):
          1. Load 16 kHz mono audio via librosa.
          2. Run processor with language + punctuation kwargs (these are
             PROCESSOR kwargs, not decode kwargs — verified 2026-05-07).
          3. Capture audio_chunk_index from processor output.  This is a
             chunking index used by processor.decode, NOT a timestamp signal.
             Cohere has no native timestamps (HF discussion #19).
          4. Cast the whole BatchEncoding to device + model.dtype.
          5. model.generate under torch.inference_mode().
          6. Move outputs back to CPU before decode.
          7. Build decode_kwargs conditionally — only pass audio_chunk_index
             and language when audio_chunk_index is not None.
          8. processor.decode may return a list (of strings) or a single
             string; handle both forms.

        Args:
            audio_path: Path to the audio file (WAV, 16 kHz mono expected).
            language: Language code (overrides config language for this call).
            context: IGNORED.  Cohere does not accept initial-prompt context
                the way Whisper does; if support becomes relevant, revisit.

        Returns:
            TranscriptionResult with transcribed text.
        """
        if not self._loaded:
            raise RuntimeError(
                "CohereTextGenerator.generate() called before load(). "
                "Call load() first."
            )

        if context:
            logger.debug(
                "[CohereTextGenerator] context parameter ignored "
                "(Cohere does not accept initial prompts the way Whisper does)"
            )

        import torch

        cfg = self._config
        resolved_language = language or cfg["language"]
        resolved_punctuation = cfg["punctuation"]
        resolved_max_new_tokens = cfg["max_new_tokens"]

        # Step 1: Load audio (16 kHz mono float32).
        audio = self._load_audio(audio_path)

        # Step 2: Run processor with language + punctuation kwargs.
        # Verified against repka3 PoC (transformers 5.4.0); same kwargs
        # are accepted by the trust_remote_code path on transformers 4.57.6
        # because the processor class ships in the model repo.
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            language=resolved_language,
            punctuation=resolved_punctuation,
        )

        # Step 3: Capture audio_chunk_index BEFORE the .to() cast — it is
        # a Python int / tensor metadata, not a tensor we want on GPU.
        audio_chunk_index = inputs.get("audio_chunk_index")

        # Step 4: Cast the whole BatchEncoding to device + model.dtype.
        # repka3 uses self.model.dtype here; equivalent to self._dtype which
        # we resolved at load() time.
        inputs = inputs.to(self._device, dtype=self._dtype)

        # Step 5: Generate.  Greedy decoding (do_sample=False, num_beams=1)
        # is set explicitly for determinism even though it is the default
        # behavior — guards against future generation_config drift.
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=resolved_max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        # Step 6: Move outputs to CPU before decode (mirrors repka3 pattern).
        outputs = outputs.cpu()

        # Step 7: Build decode_kwargs conditionally.
        decode_kwargs = {"skip_special_tokens": True}
        if audio_chunk_index is not None:
            decode_kwargs["audio_chunk_index"] = audio_chunk_index
            decode_kwargs["language"] = resolved_language

        transcript = self._processor.decode(outputs, **decode_kwargs)

        # Step 8: Decode may return list or string.
        if isinstance(transcript, list):
            text = transcript[0] if len(transcript) == 1 else "\n".join(transcript)
        else:
            text = transcript

        text = (text or "").strip()

        return TranscriptionResult(
            text=text,
            language=resolved_language,
            metadata={
                "generator": "cohere",
                "audio_path": str(audio_path),
                "audio_chunk_index_present": audio_chunk_index is not None,
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

        Processes one file at a time.  The VRAM lifecycle (load/unload)
        is managed by the orchestrator, not per-call.
        """
        if not self._loaded:
            raise RuntimeError(
                "CohereTextGenerator.generate_batch() called before load(). "
                "Call load() first."
            )

        results = []
        for i, audio_path in enumerate(audio_paths):
            logger.debug(
                "[CohereTextGenerator] Generating %d/%d: %s",
                i + 1, len(audio_paths),
                audio_path.name if hasattr(audio_path, "name") else audio_path,
            )
            result = self.generate(audio_path, language=language)
            results.append(result)

        return results

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
