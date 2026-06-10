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

# Disk-space pre-flight threshold.  The Cohere weights are ~3.85 GB
# (model.safetensors alone is 4.13 GB FP16 raw); we add a 1 GB safety
# margin to cover Xet temp staging and other transformers cache files.
# Verified against the HF API tree listing on 2026-05-07.
_REQUIRED_DOWNLOAD_GB = 5.0
_MODEL_WEIGHT_GB = 3.85

# The Cohere processor takes ISO language CODES ("ja", "en" — per the model
# card examples), but upstream callers may hand us Qwen-style full names
# ("Japanese") because the qwen outer shell defaults qwen_language="Japanese".
# Normalize full names for the model's 14 supported languages; unknown values
# pass through unchanged (the processor raises its own error for those).
_LANGUAGE_NAME_TO_CODE = {
    "english": "en", "french": "fr", "german": "de", "italian": "it",
    "spanish": "es", "portuguese": "pt", "greek": "el", "dutch": "nl",
    "polish": "pl", "chinese": "zh", "mandarin": "zh", "japanese": "ja",
    "korean": "ko", "vietnamese": "vi", "arabic": "ar",
}


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """Map full language names to the ISO codes the Cohere processor expects."""
    if not language:
        return language
    code = _LANGUAGE_NAME_TO_CODE.get(language.strip().lower())
    if code and code != language:
        logger.debug(
            "[CohereTextGenerator] Normalized language %r -> %r "
            "(Cohere processor takes ISO codes)", language, code
        )
        return code
    return language


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
        """Resolve 'auto' dtype based on device capability.

        float16 is NOT usable with this model: the repo's remote modeling
        code masks attention scores via masked_fill(mask, -1e9), and -1e9
        overflows fp16 (max +/-65504) — every forward pass dies with
        "value cannot be converted to type c10::Half without overflow"
        (verified on a real run, 2026-06-10). bfloat16 has fp32-like range
        at the same memory cost, so it is the CUDA default; explicit
        float16 requests are coerced with a warning.
        """
        import torch
        if dtype == "float16":
            logger.warning(
                "[CohereTextGenerator] float16 requested but the Cohere "
                "modeling code overflows fp16 (-1e9 attention mask); "
                "using bfloat16 instead."
            )
            dtype = "bfloat16"
        if dtype != "auto":
            return getattr(torch, dtype, torch.float32)
        if "cuda" in device:
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            # Pre-Ampere GPU without bf16: fp32 is the only safe choice
            # (doubles VRAM to ~8GB, but fp16 cannot produce output at all).
            return torch.float32
        return torch.float32

    # ------------------------------------------------------------------
    # HF gating pre-flight
    # ------------------------------------------------------------------

    @staticmethod
    def _cached_snapshot(model_id: str) -> Optional[str]:
        """
        Return the local HF cache snapshot path if the model is fully
        downloaded, else None.

        Gating only matters for the DOWNLOAD — once the snapshot is on disk,
        the model loads fine with no token at all (local_files_only=True).
        snapshot_download(local_files_only=True) succeeds only when every
        file of the latest cached revision is present, so a partially
        downloaded model correctly falls through to the token preflight.
        """
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_id, local_files_only=True)
        except Exception:
            return None

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
            # Also accept a token stored by `huggingface-cli login` /
            # `huggingface_hub.login()` (~/.cache/huggingface/token) —
            # from_pretrained honors it, so the preflight should too.
            try:
                from huggingface_hub import get_token as _hf_get_token
                token = _hf_get_token()
            except Exception:
                token = None
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
    def _walk_chain(exc: BaseException, max_depth: int = 8) -> list[BaseException]:
        """Walk __cause__ and __context__ to gather the full exception chain.

        transformers wraps low-level errors (disk space, network, Xet) in
        higher-level OSErrors that drop the original message, so we must
        inspect the chain to find the real root cause.
        """
        chain: list[BaseException] = [exc]
        seen = {id(exc)}
        cur: BaseException = exc
        for _ in range(max_depth):
            nxt = cur.__cause__ or cur.__context__
            if nxt is None or id(nxt) in seen:
                break
            chain.append(nxt)
            seen.add(id(nxt))
            cur = nxt
        return chain

    @staticmethod
    def _classify_error(messages: list[str]) -> str:
        """Pick the most specific known failure mode from chain messages.

        Returns one of: 'disk_space', 'gated', 'auth', 'xet', 'network',
        'incomplete_download', 'unknown'.  Order is significant — more
        specific patterns are checked first.
        """
        text = " ".join(messages).lower()
        if any(t in text for t in (
            "os error 112",          # Windows: There is not enough space on the disk
            "no space left",         # POSIX
            "not enough space",
            "errno 28",              # POSIX ENOSPC numeric
            "disk full",
        )):
            return "disk_space"
        if any(t in text for t in ("gated", "403", "access to model", "is restricted", "not in the authorized")):
            return "gated"
        if any(t in text for t in ("401", "unauthorized", "invalid token", "authentication failed")):
            return "auth"
        if "cas service error" in text or "xet_get" in text or "xet-core" in text:
            return "xet"
        if any(t in text for t in (
            "connection",
            "timeout",
            "name or service not known",
            "unreachable",
            "max retries",
            "ssl",
            "proxy",
        )):
            return "network"
        if "can't load the model" in text or (
            "pytorch_model" in text and "directory" in text
        ):
            return "incomplete_download"
        return "unknown"

    @classmethod
    def _format_load_error(cls, exc: Exception) -> str:
        """Produce a helpful diagnostic for known load-failure modes.

        Walks the exception chain so root causes that transformers wraps in
        a generic "Can't load the model" OSError are surfaced clearly.
        Always includes the full chain at the end so no information is lost.
        """
        chain = cls._walk_chain(exc)
        messages = [str(e) for e in chain]
        kind = cls._classify_error(messages)

        # Build the chain summary that always appends to the message.
        chain_summary_lines = ["", "Full error chain (most recent → original cause):"]
        for i, m in enumerate(messages):
            chain_summary_lines.append(f"  [{i}] {m.strip().splitlines()[0][:200]}")
        chain_summary = "\n".join(chain_summary_lines)

        if kind == "disk_space":
            return (
                "Failed to load Cohere-Transcribe — disk ran out of space during download.\n"
                "\n"
                f"The Cohere model weights are about {_MODEL_WEIGHT_GB:.2f} GB. The download "
                "uses HuggingFace's Xet content-addressed delivery, which needs additional\n"
                "temp space during streaming. We recommend at least "
                f"{_REQUIRED_DOWNLOAD_GB:.1f} GB free on the cache volume.\n"
                "\n"
                "Remediation:\n"
                "  1. Free disk space on the volume hosting the HF cache, OR\n"
                "  2. Redirect the HF cache to a drive with more space:\n"
                "     Windows (persistent): setx HUGGINGFACE_HUB_CACHE D:\\hf_cache\n"
                "                           — restart the GUI/terminal after setx\n"
                "     Windows (current session): $env:HUGGINGFACE_HUB_CACHE = \"D:\\hf_cache\"\n"
                "     macOS/Linux: export HUGGINGFACE_HUB_CACHE=/path/with/space\n"
                "  3. Retry — the Cohere download will resume into the new cache.\n"
                "\n"
                "See FAQ: 'Cohere-Transcribe (preview)' → 'Common errors'."
                + chain_summary
            )
        if kind == "gated":
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
                "See FAQ: 'Using Cohere-Transcribe (preview, opt-in)'."
                + chain_summary
            )
        if kind == "auth":
            return (
                "Failed to load Cohere-Transcribe — authentication failed.\n"
                "\n"
                "Your HF_TOKEN may be expired, revoked, or malformed. Recreate a Read\n"
                f"token at {_TOKEN_PAGE_URL} and re-set HF_TOKEN in the environment.\n"
                + chain_summary
            )
        if kind == "xet":
            return (
                "Failed to load Cohere-Transcribe — HuggingFace Xet (CAS) download error.\n"
                "\n"
                "Xet is HuggingFace's content-addressed delivery system. Common causes:\n"
                "  - Disk space exhausted during streaming (most common; check above)\n"
                "  - Network instability (proxy, VPN, or rate limiting)\n"
                "  - Cache directory not writable\n"
                "\n"
                "Remediation:\n"
                "  1. Verify free disk space on the HF cache volume\n"
                f"     ({_REQUIRED_DOWNLOAD_GB:.1f} GB recommended for Cohere)\n"
                "  2. If on a corporate network, check proxy/SSL settings\n"
                "  3. Retry with a stable connection.\n"
                + chain_summary
            )
        if kind == "network":
            return (
                "Failed to load Cohere-Transcribe — network error during download.\n"
                "\n"
                "Common causes: blocked / throttled HuggingFace, proxy / VPN issues,\n"
                "or transient connectivity drops. China users: see the FAQ for the\n"
                "'HF mirror' and 'manual local-path' alternatives.\n"
                + chain_summary
            )
        if kind == "incomplete_download":
            return (
                "Failed to load Cohere-Transcribe — required model files are missing\n"
                "from the cache (likely an interrupted previous download).\n"
                "\n"
                "Remediation:\n"
                "  1. Locate the HF cache:\n"
                "     - Default: %USERPROFILE%\\.cache\\huggingface\\hub\\\n"
                "       (or wherever HUGGINGFACE_HUB_CACHE / HF_HOME points)\n"
                "  2. Delete the partial directory:\n"
                "     models--CohereLabs--cohere-transcribe-03-2026\n"
                "  3. Verify free disk space, then retry.\n"
                + chain_summary
            )
        # Unknown — return the surface message + full chain so the user
        # always has the original cause visible without scrolling logs.
        return f"Failed to load Cohere-Transcribe: {exc}" + chain_summary

    @staticmethod
    def _is_local_path(model_id: str) -> bool:
        """Return True if model_id points to a local filesystem directory."""
        if not model_id:
            return False
        if os.path.isdir(model_id):
            return True
        # Heuristic: HF model IDs are 'org/name' with no path separators or
        # drive letters; a string containing those is almost certainly a path.
        if any(sep in model_id for sep in (os.sep, "/", "\\")) and (
            model_id.startswith(("./", "../", "~", "/", "\\"))
            or (len(model_id) >= 2 and model_id[1] == ":")  # Windows drive letter
        ):
            return True
        return False

    @staticmethod
    def _resolve_hf_cache_dir() -> str:
        """Resolve the HF cache directory using the documented env-var precedence."""
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HUB_CACHE")
            or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None)
            or os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))
        )
        return cache_dir

    @classmethod
    def _check_disk_space(cls, required_gb: float = _REQUIRED_DOWNLOAD_GB) -> None:
        """Pre-flight: verify enough free disk space at the HF cache volume.

        Raises RuntimeError with an actionable diagnostic if the volume
        hosting the HF cache has less than required_gb of free space.
        Walks up the directory tree if the cache dir does not yet exist
        (first run on a fresh machine).
        """
        import shutil

        cache_dir = cls._resolve_hf_cache_dir()

        # Walk up to the nearest existing parent (cache may not exist on first run).
        check_dir = cache_dir
        while not os.path.exists(check_dir):
            parent = os.path.dirname(check_dir)
            if not parent or parent == check_dir:
                break
            check_dir = parent

        try:
            free_bytes = shutil.disk_usage(check_dir).free
        except OSError:
            # Cannot determine free space (rare); skip and let the download
            # raise its own error with the chain-aware diagnostic.
            return

        free_gb = free_bytes / (1024 ** 3)
        if free_gb < required_gb:
            raise RuntimeError(
                "Insufficient disk space for Cohere-Transcribe download.\n"
                f"  HF cache:   {cache_dir}\n"
                f"  Volume:     {check_dir}\n"
                f"  Available:  {free_gb:.2f} GB\n"
                f"  Required:   ~{required_gb:.1f} GB "
                f"(weights ~{_MODEL_WEIGHT_GB:.2f} GB + Xet temp + safety margin)\n"
                "\n"
                "Remediation:\n"
                "  1. Free space on the cache volume, OR\n"
                "  2. Redirect the HF cache to a drive with more space:\n"
                "     Windows (persistent): setx HUGGINGFACE_HUB_CACHE D:\\hf_cache\n"
                "                           — restart the GUI/terminal after setx\n"
                "     Windows (current session): $env:HUGGINGFACE_HUB_CACHE = \"D:\\hf_cache\"\n"
                "     macOS/Linux: export HUGGINGFACE_HUB_CACHE=/path/with/space\n"
                "\n"
                "See FAQ: 'Cohere-Transcribe (preview)' → 'Common errors'."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load AutoProcessor + AutoModelForSpeechSeq2Seq for Cohere Transcribe.

        Class selection: per the model's auto_map metadata,
            AutoModel                  -> CohereAsrModel (encoder only)
            AutoModelForSpeechSeq2Seq  -> CohereAsrForConditionalGeneration
        We need the seq2seq wrapper because generate() requires the decoder.
        AutoModelForSpeechSeq2Seq follows auto_map under trust_remote_code=True,
        so the custom modeling_cohere_asr.py is used as documented.

        Pre-flights performed (skipped when model_id is a local path):
          - HF_TOKEN presence (gating preflight)
          - Disk space at HF cache (~5 GB, weights ~3.85 GB + Xet temp + margin)

        Raises RuntimeError with a chain-aware diagnostic on any load failure.
        """
        if self._loaded:
            logger.debug("[CohereTextGenerator] Already loaded")
            return

        cfg = self._config
        local_path = self._is_local_path(cfg["model_id"])

        # Gating preflights only matter when a DOWNLOAD is needed. If the
        # full snapshot is already in the local HF cache, load offline —
        # no token required, and local_files_only=True below prevents
        # transformers' Hub revision check from raising 401/GatedRepoError
        # for token-less (e.g. stale-environment GUI) processes.
        cached_snapshot = None
        if not local_path:
            cached_snapshot = self._cached_snapshot(cfg["model_id"])
            if cached_snapshot:
                logger.info(
                    "[CohereTextGenerator] Model found in local HF cache — "
                    "loading offline (no HF token needed): %s", cached_snapshot
                )

        if not local_path and not cached_snapshot:
            # Hub-side preflights: only relevant when downloading from HF.
            self._check_hf_access()
            try:
                self._check_disk_space()
            except RuntimeError:
                # Re-raise without wrapping — preflight already produced
                # an actionable diagnostic.
                raise

        # Suppress TF/oneDNN warnings before importing transformers
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        import torch
        from transformers import AutoProcessor

        # AutoModelForSpeechSeq2Seq is the generate-capable wrapper for
        # ASR seq2seq models; lands in transformers >= 4.20-ish.  Fall back
        # to AutoModel only if the env is too old (should not happen in WJ).
        try:
            from transformers import AutoModelForSpeechSeq2Seq as _AutoModelClass
            _model_class_name = "AutoModelForSpeechSeq2Seq"
        except (ImportError, AttributeError):
            from transformers import AutoModel as _AutoModelClass
            _model_class_name = "AutoModel (fallback — old transformers)"
            logger.warning(
                "[CohereTextGenerator] AutoModelForSpeechSeq2Seq unavailable; "
                "falling back to AutoModel. generate() may fail at inference time."
            )

        device = self._detect_device(cfg["device"])
        dtype = self._detect_dtype(device, cfg["dtype"])

        logger.info("[CohereTextGenerator] Loading model...")
        logger.info("  Model:        %s", cfg["model_id"])
        logger.info("  Local path:   %s", local_path)
        logger.info("  Device:       %s", device)
        logger.info("  Dtype:        %s", dtype)
        logger.info("  Loader class: %s", _model_class_name)

        import time
        start = time.time()

        # Offline load when the snapshot is cached: skips Hub revision
        # checks entirely, so no token is needed and no network round-trip.
        _offline = {"local_files_only": True} if cached_snapshot else {}

        try:
            self._processor = AutoProcessor.from_pretrained(
                cfg["model_id"],
                trust_remote_code=cfg["trust_remote_code"],
                **_offline,
            )
            self._model = _AutoModelClass.from_pretrained(
                cfg["model_id"],
                dtype=dtype,
                trust_remote_code=cfg["trust_remote_code"],
                **_offline,
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
        resolved_language = _normalize_language(language or cfg["language"])
        resolved_punctuation = cfg["punctuation"]
        resolved_max_new_tokens = cfg["max_new_tokens"]

        # Step 1: Load audio (16 kHz mono float32).
        audio = self._load_audio(audio_path)

        # Step 2: Build the decoder prompt that FORCES language/punctuation.
        # CRITICAL (root cause of non-Japanese output, found 2026-06-10):
        # the processor IGNORES language=/punctuation= kwargs entirely
        # (processing_cohere_asr.py never reads them — they vanish into
        # **kwargs), and the model performs NO language detection (its own
        # transcribe() docstring). The ONLY language mechanism is the decoder
        # prompt prefix "<|startofcontext|><|startoftranscript|>...<|ja|><|ja|>..."
        # built by model.build_prompt() and fed as decoder_input_ids — mirror
        # the model's own transcribe()/_transcribe_waveforms_batched() flow.
        try:
            self._model._validate_transcribe_language(resolved_language)
        except AttributeError:
            pass  # older/newer revision without the helper — build_prompt below
        except ValueError as exc:
            raise RuntimeError(
                f"Cohere-Transcribe does not support language "
                f"'{resolved_language}': {exc}"
            ) from exc

        if hasattr(self._model, "build_prompt"):
            prompt_text = self._model.build_prompt(
                language=resolved_language, punctuation=resolved_punctuation
            )
        else:
            # Fallback mirrors build_prompt() from the cached repo revision.
            pnc_token = "<|pnc|>" if resolved_punctuation else "<|nopnc|>"
            prompt_text = (
                "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
                f"<|{resolved_language}|><|{resolved_language}|>"
                f"{pnc_token}<|noitn|><|notimestamp|><|nodiarize|>"
            )

        # Step 3: Run processor with audio AND the prompt text (lists, exactly
        # like the model's own batched transcribe path). With text= present
        # the processor tokenizes the prompt into input_ids.
        inputs = self._processor(
            audio=[audio],
            text=[prompt_text],
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Capture audio_chunk_index BEFORE device casts — it is metadata,
        # not a tensor we want on GPU.
        audio_chunk_index = inputs.get("audio_chunk_index")

        # Step 4: Cast per-key: floating tensors get the model dtype,
        # integer tensors (input_ids etc.) must stay long — a blanket
        # .to(device, dtype) would corrupt the prompt ids.
        cast_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                cast_inputs[key] = value.to(self._device, dtype=self._dtype)
            elif hasattr(value, "to"):
                cast_inputs[key] = value.to(self._device)
            else:
                cast_inputs[key] = value
        inputs = cast_inputs

        # Step 5: Rename input_ids -> decoder_input_ids and build the decoder
        # attention mask (same as the model's transcribe path). Providing a
        # real mask also fixes the transformers 4.57 crash ('NoneType' has no
        # attribute 'new_ones') that the old bare-start-token workaround
        # papered over.
        if "input_ids" in inputs and "decoder_input_ids" not in inputs:
            inputs["decoder_input_ids"] = inputs.pop("input_ids")

        extra_gen_kwargs = {}
        prompt_len = 0
        if "decoder_input_ids" in inputs:
            dec_ids = inputs["decoder_input_ids"]
            pad_id = getattr(self._processor.tokenizer, "pad_token_id", None)
            if "decoder_attention_mask" not in inputs:
                if pad_id is None:
                    inputs["decoder_attention_mask"] = torch.ones_like(dec_ids)
                else:
                    inputs["decoder_attention_mask"] = dec_ids.ne(pad_id).long()
            prompt_len = int(inputs["decoder_attention_mask"][0].sum().item())
            extra_gen_kwargs["decoder_start_token_id"] = int(dec_ids[0, 0].item())
        else:
            # Defensive fallback (processor revision without text= support):
            # bare decoder start token keeps generation alive on 4.57, but
            # language is NOT forced — warn loudly.
            logger.warning(
                "[CohereTextGenerator] Processor did not return prompt ids — "
                "language forcing unavailable; output language may drift."
            )
            gen_cfg = getattr(self._model, "generation_config", None)
            start_id = getattr(gen_cfg, "decoder_start_token_id", None) if gen_cfg else None
            if start_id is None:
                start_id = getattr(self._model.config, "decoder_start_token_id", None)
            if start_id is not None:
                extra_gen_kwargs["decoder_input_ids"] = torch.full(
                    (inputs["input_features"].shape[0], 1), start_id,
                    dtype=torch.long, device=self._device,
                )

        # Step 6: Generate.  Greedy decoding (do_sample=False, num_beams=1)
        # is set explicitly for determinism even though it is the default
        # behavior — guards against future generation_config drift.
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **extra_gen_kwargs,
                max_new_tokens=resolved_max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        # Move outputs to CPU before decode (mirrors repka3 pattern).
        outputs = outputs.cpu()

        # Step 6b: Trim the echoed prompt prefix from the generated ids
        # (generate() returns prompt + continuation). The prompt tokens are
        # specials so skip_special_tokens would drop most of them anyway,
        # but trimming deterministically matches the model's own flow.
        if (
            prompt_len
            and outputs.shape[1] >= prompt_len
            and torch.equal(
                outputs[0, :prompt_len],
                inputs["decoder_input_ids"][0, :prompt_len].cpu(),
            )
        ):
            outputs = outputs[:, prompt_len:]

        # Step 7: Build decode_kwargs conditionally.
        decode_kwargs = {"skip_special_tokens": True}
        if audio_chunk_index is not None:
            decode_kwargs["audio_chunk_index"] = audio_chunk_index
            decode_kwargs["language"] = resolved_language

        # The current model-repo revision's processor.decode is a PLAIN proxy
        # to tokenizer.decode (verified in the cached remote source,
        # 2026-06-10): it takes a single 1-D id sequence and has no
        # audio_chunk_index reassembly. model.generate returns a 2-D
        # (batch, seq) tensor, so feeding it whole dies with
        # "int() argument must be ... not 'list'". Try the documented call
        # first (forward-compat with revisions that do implement chunk
        # reassembly), then fall back to batch_decode on the 2-D output.
        try:
            transcript = self._processor.decode(outputs, **decode_kwargs)
        except (TypeError, ValueError):
            logger.debug(
                "[CohereTextGenerator] processor.decode rejected batched "
                "output (plain tokenizer proxy in this repo revision); "
                "falling back to batch_decode."
            )
            transcript = self._processor.batch_decode(
                outputs, skip_special_tokens=True
            )

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
