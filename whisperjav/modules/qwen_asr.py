#!/usr/bin/env python3
"""
Qwen3-ASR Module for WhisperJAV.

Uses the qwen-asr package with transformers backend for automatic speech recognition.
Supports optional ForcedAligner for word-level timestamps, which are then regrouped
into sentence-level segments using stable-ts for proper subtitle generation.

Architecture:
    1. qwen-asr produces word-level timestamps via ForcedAligner
    2. stable_whisper.transcribe_any() regroups words into sentence-level segments
    3. Returns WhisperResult for flexible post-processing

Based on: https://github.com/QwenLM/Qwen3-ASR
"""

import gc
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
import stable_whisper

from whisperjav.utils.logger import logger
from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

# Suppress common transformers warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")


def merge_master_with_timestamps(
    master_text: str,
    timestamps: List[Any]
) -> List[dict]:
    """
    Merge master text (with punctuation) with timestamp metadata (without punctuation).

    The ForcedAligner strips punctuation from words during alignment because punctuation
    has no audio representation. However, the ASR's raw output (master_text) contains
    the complete transcript including punctuation.

    This function reconciles both sources:
    - CONTENT comes from master_text (preserves punctuation)
    - TIMING comes from timestamps (ForcedAligner output)

    Punctuation attachment rules:
    - Leading punctuation (e.g., 「 at start) → attaches to FIRST word
    - Middle punctuation (e.g., 、。between words) → attaches to PRECEDING word
    - Trailing punctuation (e.g., 。at end) → attaches to LAST word

    Args:
        master_text: Complete transcript from ASR (includes punctuation)
        timestamps: List of ForcedAlignItem objects or dicts with:
                   - 'text' or .text: The word (cleaned, no punctuation)
                   - 'start_time' or .start_time: Start time in seconds
                   - 'end_time' or .end_time: End time in seconds

    Returns:
        List of word dicts for stable-ts:
        [{'word': str, 'start': float, 'end': float}, ...]

    Example:
        master_text = "けども、お前が。"
        timestamps = [("けども", 1.0, 1.5), ("お前", 2.0, 2.3), ("が", 2.3, 2.5)]

        Output: [
            {'word': 'けども、', 'start': 1.0, 'end': 1.5},
            {'word': 'お前', 'start': 2.0, 'end': 2.3},
            {'word': 'が。', 'start': 2.3, 'end': 2.5}
        ]
    """
    # Handle edge cases
    if not master_text or not master_text.strip():
        return []

    if not timestamps:
        # No timing data - return master as single "word"
        return [{'word': master_text.strip(), 'start': 0.0, 'end': 0.0}]

    # Helper to get attributes from timestamp objects or dicts
    def get_attr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr)
        return None

    result = []
    master_pos = 0  # Current position in master text

    for ts in timestamps:
        ts_word = get_attr(ts, 'text')
        ts_start = get_attr(ts, 'start_time')
        ts_end = get_attr(ts, 'end_time')

        # Skip empty words
        if not ts_word:
            continue

        # Find this word in master text, starting from current position
        word_start = master_text.find(ts_word, master_pos)

        if word_start == -1:
            # Word not found in master text (shouldn't normally happen)
            # This could occur if ASR text differs from aligner input
            # Fall back to using the timestamp word as-is
            logger.debug(f"Word '{ts_word}' not found in master text at pos {master_pos}")
            result.append({
                'word': ts_word,
                'start': float(ts_start) if ts_start is not None else 0.0,
                'end': float(ts_end) if ts_end is not None else 0.0
            })
            continue

        word_end = word_start + len(ts_word)

        # Check for gap (punctuation/spaces) between previous position and this word
        if word_start > master_pos:
            gap = master_text[master_pos:word_start]

            if result:
                # Attach gap to PREVIOUS word (punctuation follows preceding content)
                result[-1]['word'] += gap
            else:
                # No previous word - this is LEADING content (e.g., opening quote 「)
                # Prepend to current word
                ts_word = gap + ts_word

        # Add this word with its timing
        result.append({
            'word': ts_word,
            'start': float(ts_start) if ts_start is not None else 0.0,
            'end': float(ts_end) if ts_end is not None else 0.0
        })

        # Move position past this word
        master_pos = word_end

    # Handle trailing content after the last timestamp word
    if master_pos < len(master_text):
        trailing = master_text[master_pos:]
        if result:
            # Attach trailing content to last word
            result[-1]['word'] += trailing
        elif trailing.strip():
            # Edge case: no timestamps matched but there's content
            result.append({'word': trailing, 'start': 0.0, 'end': 0.0})

    return result


class QwenASR:
    """
    Qwen3-ASR using transformers backend with stable-ts regrouping.

    This ASR module uses the qwen-asr package for automatic speech recognition
    with optional ForcedAligner for word-level timestamps. Word timestamps are
    then regrouped into sentence-level segments using stable-ts for proper
    subtitle generation.

    Returns WhisperResult for compatibility with stable-ts post-processing.

    Parameter Mapping (ISSUE-005/006):
        WhisperJAV Name              qwen-asr Name
        ─────────────────────────────────────────────────────────
        model_id                  → pretrained_model_name_or_path
        device                    → device_map
        batch_size                → max_inference_batch_size
        aligner_id + use_aligner  → forced_aligner (combined)
        dtype                     → dtype (torch.dtype)
        attn_implementation       → attn_implementation

    Chunking Behavior (ISSUE-007):
        - stable_whisper.transcribe_any() does NOT chunk audio internally
        - It passes the full audio path to qwen_inference() callback
        - qwen-asr handles chunking internally for audio > 180s (with aligner)
        - For audio > 3 minutes with ForcedAligner:
          * qwen-asr splits into 180s chunks
          * Results are merged internally by qwen-asr
          * Timestamps may have minor discontinuities at chunk boundaries
        - Recommendation: Use WhisperJAV scene detection for videos > 3 minutes
          to get explicit control over chunk boundaries

    Aligner Configuration (ISSUE-005):
        - ForcedAligner uses same device/dtype as main model by default
        - To configure aligner separately, use forced_aligner_kwargs in qwen-asr:
          ```python
          # Not exposed in WhisperJAV CLI (use Python API if needed)
          model = Qwen3ASRModel.from_pretrained(
              "Qwen/Qwen3-ASR-1.7B",
              forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
              forced_aligner_kwargs=dict(dtype=torch.float16, device_map="cuda:1"),
          )
          ```
    """

    # Default model IDs
    DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"

    # Default processing parameters
    # NOTE: batch_size=1 produces more accurate transcriptions than larger batches
    DEFAULT_BATCH_SIZE = 1

    # Token limit for generation
    # Calculation: 10 min audio × 400 chars/min × 2 tokens/char = 8000 tokens
    # Default 4096 provides safe coverage for ~5-10 min audio segments
    # For longer audio, scene detection should split into manageable chunks
    DEFAULT_MAX_NEW_TOKENS = 4096

    # qwen-asr internal limits (from qwen_asr.inference.utils)
    # Used for warnings and documentation
    MAX_FORCE_ALIGN_SECONDS = 180  # 3 min limit when using ForcedAligner
    MAX_ASR_SECONDS = 1200  # 20 min limit without ForcedAligner

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        language: Optional[str] = None,  # None = auto-detect
        task: str = "transcribe",
        timestamps: str = "word",  # "word" or "none"
        use_aligner: bool = True,
        aligner_id: str = DEFAULT_ALIGNER_ID,
        context: str = "",  # Context string to help ASR
        attn_implementation: str = "auto",  # Attention: auto, sdpa, flash_attention_2, eager
        # Japanese post-processing (v1.8.4+)
        japanese_postprocess: bool = True,  # Apply Japanese-specific regrouping
        postprocess_preset: str = "high_moan",  # Preset: "high_moan" (default for JAV), "default", "narrative"
        # Generation safety controls (v1.8.9+)
        repetition_penalty: float = 1.0,          # 1.0 = off; >1.0 penalizes repeated tokens via HF generation_config
        max_tokens_per_audio_second: float = 0.0,  # 0 = disabled; >0 = dynamic per-scene token budget scaling
        min_tokens_floor: int = 256,               # minimum token budget when dynamic scaling is active
    ):
        """
        Initialize QwenASR.

        Args:
            model_id: HuggingFace model ID (default: Qwen/Qwen3-ASR-1.7B)
            device: Device to use ('auto', 'cuda', 'cuda:0', 'cpu')
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
            batch_size: Maximum inference batch size
            max_new_tokens: Maximum tokens to generate per utterance
            language: Language code (e.g., 'ja', 'en') or None for auto-detect
            task: Task type ('transcribe' only - Qwen3-ASR doesn't support 'translate')
            timestamps: Timestamp granularity ('word' or 'none')
            use_aligner: Whether to use ForcedAligner for word-level timestamps
            aligner_id: HuggingFace model ID for the aligner
            context: Context string to help qwen-asr improve transcription accuracy
            attn_implementation: Attention implementation ('auto', 'sdpa', 'flash_attention_2', 'eager')
            japanese_postprocess: Whether to apply Japanese-specific regrouping (default: True)
                                  This improves subtitle quality for Japanese dialogue by:
                                  - Removing fillers and backchanneling (aizuchi)
                                  - Anchoring on sentence particles (ね, よ, etc.)
                                  - Merging/splitting for natural reading
            postprocess_preset: Processing preset for Japanese post-processing:
                                - "default": General conversational dialogue
                                - "high_moan": Adult content with frequent vocalizations
                                - "narrative": Longer narrative passages
        """
        self.model_id = model_id
        self.device_request = device
        self.dtype_request = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.task = task
        self.timestamps = timestamps
        self.use_aligner = use_aligner and (timestamps == "word")
        self.aligner_id = aligner_id
        self.context = context
        self.attn_implementation = attn_implementation

        # Japanese post-processing settings (v1.8.4+)
        self.japanese_postprocess = japanese_postprocess
        self.postprocess_preset = postprocess_preset

        # Generation safety controls (v1.8.9+)
        self.repetition_penalty = repetition_penalty
        self.max_tokens_per_audio_second = max_tokens_per_audio_second
        self.min_tokens_floor = min_tokens_floor

        # Initialize post-processor (lightweight, no GPU resources)
        self._postprocessor = JapanesePostProcessor() if japanese_postprocess else None

        # Model is lazily loaded
        self.model = None
        self._standalone_aligner = None  # For decoupled assembly mode
        self._device = None
        self._dtype = None

        # Store detected language from last transcription
        self._detected_language = None

        # Warn if translation is requested (Qwen3-ASR doesn't support it)
        if self.task == "translate":
            logger.warning(
                "Qwen3-ASR does not support translation task. "
                "Output will be in the detected language. "
                "Use WhisperJAV's translation module for translation."
            )
            self.task = "transcribe"

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.device_request == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        elif self.device_request == "cuda":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
        elif self.device_request.startswith("cuda:"):
            if torch.cuda.is_available():
                return self.device_request
            else:
                logger.warning(f"{self.device_request} requested but CUDA not available. Falling back to CPU.")
                return "cpu"
        else:
            return "cpu"

    def _detect_dtype(self, device: str) -> torch.dtype:
        """Detect best dtype for device."""
        if self.dtype_request == "auto":
            if "cuda" in device:
                # Prefer bfloat16 for Qwen3-ASR (recommended in docs)
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
        elif self.dtype_request == "float16":
            return torch.float16
        elif self.dtype_request == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    def _detect_attn_implementation(self) -> str:
        """
        Detect best attention implementation.

        Priority:
        1. If explicit value (not 'auto'), use it
        2. If CUDA and flash-attn available, use flash_attention_2
        3. Otherwise use sdpa (PyTorch native scaled dot product attention)

        Returns:
            Attention implementation string
        """
        if self.attn_implementation != "auto":
            return self.attn_implementation

        # Check if we're on CUDA
        device = self._device or self._detect_device()
        if "cuda" not in device:
            return "sdpa"  # CPU doesn't benefit from flash attention

        # Check if flash-attn is available
        try:
            import flash_attn
            logger.debug("flash-attn detected, using flash_attention_2")
            return "flash_attention_2"
        except ImportError:
            logger.debug("flash-attn not available, using sdpa")
            return "sdpa"

    def _apply_generation_config(self) -> None:
        """
        Apply generation safety controls to the thinker's HF GenerationConfig.

        Sets repetition_penalty on the underlying thinker model's generation_config
        after model load. This is the standard HuggingFace mechanism — the outer
        generate() call reads from generation_config when no explicit kwarg is passed.

        Access chain (verified):
            self.model.model.thinker.generation_config
            - self.model → Qwen3ASRModel
            - .model → Qwen3ASRForConditionalGeneration (HF model)
            - .thinker → Qwen3ASRThinkerForConditionalGeneration (GenerationMixin)
            - .generation_config → HF GenerationConfig

        Wrapped in try/except: if qwen-asr restructures internals, we log a
        warning instead of crashing. The penalty is a quality improvement, not
        a correctness requirement.
        """
        if self.repetition_penalty == 1.0:
            return  # 1.0 = no penalty, nothing to set

        try:
            gen_config = self.model.model.thinker.generation_config
            gen_config.repetition_penalty = self.repetition_penalty
            logger.info(
                "  Generation safety: repetition_penalty=%.2f applied to thinker",
                self.repetition_penalty,
            )
        except AttributeError as e:
            logger.warning(
                "Could not apply repetition_penalty — qwen-asr model structure "
                "may have changed: %s. Generation will proceed without penalty.",
                e,
            )

    def _compute_dynamic_token_limit(self, audio_duration_sec: float) -> int:
        """
        Compute a dynamic max_new_tokens limit scaled to audio duration.

        When max_tokens_per_audio_second > 0, the token budget is proportional
        to the audio length instead of a fixed 4096. This caps the damage from
        degenerate autoregressive loops: a 10-second clip gets ~256 tokens
        instead of burning through 4096.

        The result is clamped to [min_tokens_floor, max_new_tokens] — never
        exceeds the static limit, never goes below the floor.

        Args:
            audio_duration_sec: Duration of the audio clip in seconds.

        Returns:
            Token budget for this clip.
        """
        if self.max_tokens_per_audio_second <= 0 or audio_duration_sec <= 0:
            return self.max_new_tokens
        dynamic = max(
            self.min_tokens_floor,
            int(audio_duration_sec * self.max_tokens_per_audio_second),
        )
        return min(dynamic, self.max_new_tokens)  # never exceed static limit

    def _get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get audio file duration in seconds.

        Uses librosa for fast duration detection without loading entire file.
        Falls back to soundfile if librosa fails.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds, or 0.0 if detection fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return 0.0

        try:
            import librosa
            duration = librosa.get_duration(path=str(audio_path))
            return float(duration)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"librosa duration detection failed: {e}")

        # Fallback to soundfile
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return float(info.duration)
        except ImportError:
            logger.warning("Neither librosa nor soundfile available for duration detection")
            return 0.0
        except Exception as e:
            logger.warning(f"Audio duration detection failed: {e}")
            return 0.0

    def _estimate_tokens(self, audio_duration_seconds: float) -> int:
        """
        Estimate tokens needed for audio transcription.

        Based on Japanese speech characteristics:
        - Average speech rate: ~400 characters per minute (conservative)
        - Tokenization ratio: ~2 tokens per character (for Japanese)

        Formula: tokens = (duration_min) × 400 × 2 = duration_min × 800

        Args:
            audio_duration_seconds: Audio duration in seconds

        Returns:
            Estimated token count
        """
        if audio_duration_seconds <= 0:
            return 0

        duration_minutes = audio_duration_seconds / 60.0
        chars_per_minute = 400  # Conservative estimate for Japanese speech
        tokens_per_char = 2.0   # Conservative tokenization ratio

        estimated_tokens = int(duration_minutes * chars_per_minute * tokens_per_char)
        return estimated_tokens

    def _check_audio_limits(self, audio_path: Union[str, Path]) -> None:
        """
        Check audio against known limits and warn user.

        Checks:
        1. ForcedAligner 3-minute limit (when use_aligner=True)
        2. Token estimation vs max_new_tokens

        Args:
            audio_path: Path to audio file
        """
        duration = self._get_audio_duration(audio_path)
        if duration <= 0:
            return  # Could not detect duration, skip checks

        audio_name = Path(audio_path).name

        # Check ForcedAligner limit
        if self.use_aligner and duration > self.MAX_FORCE_ALIGN_SECONDS:
            logger.warning(
                f"Audio '{audio_name}' duration ({duration:.0f}s) exceeds ForcedAligner limit "
                f"({self.MAX_FORCE_ALIGN_SECONDS}s). qwen-asr will chunk internally, but for best results "
                f"consider enabling scene detection (--qwen-scene auditok)."
            )

        # Check token estimation
        estimated_tokens = self._estimate_tokens(duration)
        if estimated_tokens > self.max_new_tokens:
            logger.warning(
                f"Audio '{audio_name}' ({duration:.0f}s) may require ~{estimated_tokens} tokens, "
                f"but max_new_tokens={self.max_new_tokens}. Output may be truncated. "
                f"Consider using --qwen-max-tokens {estimated_tokens} or enabling scene detection."
            )
        elif estimated_tokens > self.max_new_tokens * 0.8:
            # Warn if approaching limit (80% threshold)
            logger.info(
                f"Audio '{audio_name}' ({duration:.0f}s) estimates ~{estimated_tokens} tokens "
                f"(approaching max_new_tokens={self.max_new_tokens})."
            )

    def load_model(self) -> None:
        """
        Load the Qwen3-ASR model.

        This is called lazily on first transcribe() call.
        """
        if self.model is not None:
            logger.debug("Qwen3-ASR model already loaded")
            return

        # Detect device and dtype
        self._device = self._detect_device()
        self._dtype = self._detect_dtype(self._device)

        # Detect attention implementation
        attn_impl = self._detect_attn_implementation()

        logger.info("Loading Qwen3-ASR model...")
        logger.info(f"  Model:    {self.model_id}")
        logger.info(f"  Device:   {self._device}")
        logger.info(f"  Dtype:    {self._dtype}")
        logger.info(f"  Batch:    {self.batch_size}")
        logger.info(f"  Attention: {attn_impl}")
        logger.info(f"  Aligner:  {self.aligner_id if self.use_aligner else 'disabled'}")
        if self.context:
            logger.info(f"  Context:  '{self.context[:50]}{'...' if len(self.context) > 50 else ''}'")

        # Diagnostic logging
        logger.debug(
            "[QwenASR PID %s] Loading model_id=%s on device=%s dtype=%s attn=%s",
            os.getpid(), self.model_id, self._device, self._dtype, attn_impl
        )

        start_time = time.time()

        try:
            from qwen_asr import Qwen3ASRModel

            # Build model kwargs
            model_kwargs = {
                "dtype": self._dtype,
                "device_map": self._device,
                "max_inference_batch_size": self.batch_size,
                "max_new_tokens": self.max_new_tokens,
            }

            # Add attention implementation if not default
            if attn_impl and attn_impl != "sdpa":
                model_kwargs["attn_implementation"] = attn_impl

            # Add aligner if enabled (with matching device/dtype per official docs)
            if self.use_aligner:
                model_kwargs["forced_aligner"] = self.aligner_id
                model_kwargs["forced_aligner_kwargs"] = {
                    "dtype": self._dtype,
                    "device_map": self._device,
                }

            logger.debug(
                "[QwenASR PID %s] Calling Qwen3ASRModel.from_pretrained(%s)...",
                os.getpid(), self.model_id
            )

            self.model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            load_time = time.time() - start_time
            logger.info(f"  Loaded in {load_time:.1f}s")
            logger.debug(
                "[QwenASR PID %s] Model loaded successfully: %s in %.1fs",
                os.getpid(), self.model_id, load_time
            )

            # Apply generation safety controls (v1.8.9+)
            self._apply_generation_config()

        except ImportError as e:
            logger.error(
                "qwen-asr package not installed. Install with: pip install qwen-asr"
            )
            raise ImportError(
                "qwen-asr package required for QwenASR. "
                "Install with: pip install qwen-asr"
            ) from e
        except Exception as e:
            logger.error(
                "[QwenASR PID %s] FAILED to load model %s: %s: %s",
                os.getpid(), self.model_id, type(e).__name__, e
            )
            raise

    def transcribe(
        self,
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
        context: Optional[str] = None,
    ) -> stable_whisper.WhisperResult:
        """
        Transcribe audio file using Qwen3-ASR with stable-ts regrouping.

        This method:
        1. Runs Qwen3-ASR to get word-level timestamps (via ForcedAligner)
        2. Uses stable_whisper.transcribe_any() to regroup words into sentences
        3. Returns a WhisperResult with properly segmented subtitles

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
                               (progress_percent, status_message)
            artifacts_dir: Optional directory to save debug artifacts. If provided,
                          saves: {basename}_qwen_master.txt (raw ASR text),
                          {basename}_qwen_timestamps.json (ForcedAligner data),
                          {basename}_qwen_merged.json (merged word list)
            context: Optional per-call context string for contextual biasing.
                     When None, uses self.context from __init__ (backward-compatible).
                     When provided, overrides self.context for this specific call.
                     Enables cross-scene context propagation from QwenPipeline.

        Returns:
            stable_whisper.WhisperResult with sentence-level segments
        """
        audio_path = Path(audio_path)

        # Check audio limits and warn user before processing
        self._check_audio_limits(audio_path)

        # Get audio duration for progress reporting (ISSUE-009)
        audio_duration = self._get_audio_duration(audio_path)

        logger.debug(
            "[QwenASR PID %s] transcribe() called for: %s (model=%s)",
            os.getpid(), audio_path.name, self.model_id
        )

        # Lazy load model
        if self.model is None:
            self.load_model()

        # Enhanced progress message with duration info (ISSUE-009)
        if audio_duration > 0:
            duration_str = f"{audio_duration:.0f}s" if audio_duration < 60 else f"{audio_duration/60:.1f}min"
            progress_msg = f"Transcribing {duration_str} audio: {audio_path.name}"
            if audio_duration > 180:
                # Warn about expected processing time for long audio
                logger.info(f"Processing {duration_str} audio - this may take several minutes...")
        else:
            progress_msg = f"Transcribing: {audio_path.name}"

        if progress_callback:
            progress_callback(0.0, progress_msg)

        logger.debug(f"Transcribing: {audio_path.name}")
        logger.debug(f"  Language: {self.language or 'auto-detect'}")
        logger.debug(f"  Timestamps: {self.timestamps}")
        logger.debug(f"  Use Aligner: {self.use_aligner}")

        start_time = time.time()

        # Resolve per-call context override (None = use self.context from __init__)
        effective_context = context if context is not None else self.context

        # Dynamic token budget (v1.8.9+): scale max_new_tokens to audio duration
        scene_token_limit = self._compute_dynamic_token_limit(audio_duration)
        original_max_tokens = None
        if self.model is not None and scene_token_limit != self.max_new_tokens:
            original_max_tokens = self.model.max_new_tokens
            self.model.max_new_tokens = scene_token_limit
            logger.debug(
                "Dynamic token budget: %d tokens for %.1fs audio (%.1f tok/s)",
                scene_token_limit, audio_duration, self.max_tokens_per_audio_second,
            )

        try:
            # If aligner is disabled, fall back to simple transcription
            if not self.use_aligner:
                result = self._transcribe_without_aligner(audio_path, effective_context)
                process_time = time.time() - start_time
                logger.debug(f"Transcription (no aligner) complete in {process_time:.1f}s")
                if progress_callback:
                    progress_callback(1.0, "Transcription complete")
                return result

            # With aligner: use transcribe_any for word-to-sentence regrouping
            result = self._transcribe_with_regrouping(audio_path, progress_callback, artifacts_dir, effective_context)
        finally:
            if original_max_tokens is not None and self.model is not None:
                self.model.max_new_tokens = original_max_tokens

        process_time = time.time() - start_time
        segment_count = len(result.segments) if result.segments else 0
        logger.debug(f"Transcription complete in {process_time:.1f}s, {segment_count} segments")

        # Memory management after transcription (ISSUE-010)
        # For long audio, qwen-asr may accumulate intermediate data
        # Trigger garbage collection to release unused Python objects
        gc.collect()
        if torch.cuda.is_available():
            # Log memory usage for diagnostics
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"GPU memory after transcription: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        if progress_callback:
            progress_callback(1.0, "Transcription complete")

        # Log detected language
        if self._detected_language:
            logger.info(f"Detected language: {self._detected_language}")

        return result

    def _transcribe_with_regrouping(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
        effective_context: str = "",
    ) -> stable_whisper.WhisperResult:
        """
        Transcribe with ForcedAligner and stable-ts regrouping.

        This is the main transcription path that:
        1. Calls qwen-asr to get word-level timestamps
        2. Converts to stable-ts word format
        3. Uses transcribe_any() to regroup into sentence-level segments

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
            WhisperResult with sentence-level segments
        """
        # Store reference to self for use in inference function
        qwen_model = self.model
        # Normalize language to qwen-asr format (e.g., 'ja' -> 'Japanese')
        qwen_language = self._normalize_language_for_qwen(self.language)
        qwen_context = effective_context  # Context string for ASR (may be per-call override)
        detected_language_holder = [None]  # Use list to allow mutation in closure
        save_artifacts_dir = Path(artifacts_dir) if artifacts_dir else None  # Artifact save directory

        def qwen_inference(audio: str, **kwargs) -> List[List[dict]]:
            """
            Inference function for stable_whisper.transcribe_any().

            Calls qwen-asr and returns word-level timestamps in stable-ts format.

            Args:
                audio: Path to audio file (passed by transcribe_any)

            Returns:
                List of word lists, where each word is a dict with:
                    - 'word': The text content
                    - 'start': Start time in seconds
                    - 'end': End time in seconds
            """
            try:
                # DIAGNOSTIC: Log inference call parameters
                logger.info(f"[DIAG] qwen_inference() called")
                logger.info(f"[DIAG]   audio: {audio}")
                logger.info(f"[DIAG]   model.max_new_tokens: {qwen_model.max_new_tokens}")
                logger.info(f"[DIAG]   model.max_inference_batch_size: {qwen_model.max_inference_batch_size}")
                logger.info(f"[DIAG]   context: {repr(qwen_context[:50]) + '...' if qwen_context else '(none)'}")
                logger.info(f"[DIAG]   language: {qwen_language}")

                # Call qwen-asr with timestamps enabled
                results = qwen_model.transcribe(
                    audio=str(audio),
                    context=qwen_context,  # Pass context to improve accuracy
                    language=qwen_language,
                    return_time_stamps=True,
                )

                if not results:
                    logger.warning("qwen-asr returned empty results")
                    return []

                result = results[0]

                # DIAGNOSTIC: Log raw result stats
                raw_text = getattr(result, 'text', '')
                logger.info(f"[DIAG] qwen-asr raw result:")
                logger.info(f"[DIAG]   text length: {len(raw_text)} chars")
                logger.info(f"[DIAG]   first 100 chars: '{raw_text[:100]}...'")

                # Store detected language
                if hasattr(result, 'language'):
                    detected_language_holder[0] = result.language
                    logger.info(f"[DIAG]   detected language: {result.language}")

                # Get timestamps from ForcedAligner
                time_stamps = getattr(result, 'time_stamps', None)
                logger.info(f"[DIAG]   timestamps count: {len(time_stamps) if time_stamps else 0}")

                if not time_stamps:
                    # No word timestamps - return text as single "word"
                    if raw_text.strip():
                        logger.warning("No word timestamps from aligner, returning full text as single segment")
                        return [[{'word': raw_text.strip(), 'start': 0.0, 'end': 0.0}]]
                    return []

                # === CRITICAL: Merge master text with timestamps ===
                # The ForcedAligner strips punctuation from words (clean_token removes non-L/N chars).
                # But the ASR raw_text contains the complete transcript WITH punctuation.
                # We use raw_text as SOURCE OF TRUTH for content, timestamps for timing.
                #
                # This preserves punctuation (、。？！) which represents sentence boundaries.
                words = merge_master_with_timestamps(raw_text, time_stamps)

                if not words:
                    logger.warning("No valid words after merging master text with timestamps")
                    return []

                # DIAGNOSTIC: Log word extraction stats
                total_duration = words[-1]['end'] if words else 0
                total_chars = sum(len(w['word']) for w in words)
                logger.info(f"[DIAG] Words merged: {len(words)} words, {total_chars} chars, duration: {total_duration:.2f}s")
                if words:
                    # Show first/last words (truncate if too long for logging)
                    first_word = words[0]['word'][:20] + ('...' if len(words[0]['word']) > 20 else '')
                    last_word = words[-1]['word'][:20] + ('...' if len(words[-1]['word']) > 20 else '')
                    logger.info(f"[DIAG]   first word: '{first_word}' @ {words[0]['start']:.2f}s")
                    logger.info(f"[DIAG]   last word: '{last_word}' @ {words[-1]['end']:.2f}s")

                # === SAVE DEBUG ARTIFACTS (when artifacts_dir is provided) ===
                if save_artifacts_dir:
                    try:
                        save_artifacts_dir.mkdir(parents=True, exist_ok=True)
                        audio_basename = Path(audio).stem

                        # 1. Save master text (raw ASR output with punctuation)
                        master_file = save_artifacts_dir / f"{audio_basename}_qwen_master.txt"
                        master_file.write_text(raw_text, encoding='utf-8')

                        # 2. Save timestamps (ForcedAligner output)
                        timestamps_data = []
                        for ts in time_stamps:
                            ts_entry = {
                                'text': getattr(ts, 'text', None) or (ts.get('text') if isinstance(ts, dict) else None),
                                'start_time': getattr(ts, 'start_time', None) or (ts.get('start_time') if isinstance(ts, dict) else None),
                                'end_time': getattr(ts, 'end_time', None) or (ts.get('end_time') if isinstance(ts, dict) else None),
                            }
                            timestamps_data.append(ts_entry)

                        timestamps_file = save_artifacts_dir / f"{audio_basename}_qwen_timestamps.json"
                        timestamps_file.write_text(
                            json.dumps(timestamps_data, ensure_ascii=False, indent=2),
                            encoding='utf-8'
                        )

                        # 3. Save merged result (words with punctuation restored)
                        merged_file = save_artifacts_dir / f"{audio_basename}_qwen_merged.json"
                        merged_file.write_text(
                            json.dumps(words, ensure_ascii=False, indent=2),
                            encoding='utf-8'
                        )

                        logger.debug(f"[DIAG] Saved artifacts to: {save_artifacts_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to save debug artifacts: {e}")

                # Return as list of word lists (one list = one segment)
                # transcribe_any expects: List[List[Dict]]
                return [words]

            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory during qwen-asr inference")
                raise
            except Exception as e:
                logger.error(f"Error in qwen_inference: {type(e).__name__}: {e}")
                raise

        # Use stable_whisper.transcribe_any() for word-to-sentence regrouping
        try:
            logger.debug("Calling stable_whisper.transcribe_any() with qwen inference...")

            result = stable_whisper.transcribe_any(
                inference_func=qwen_inference,
                audio=str(audio_path),
                # Audio type: qwen-asr expects file path
                audio_type='str',
                # Regrouping: use default algorithm to group words into sentences
                regroup=True,
                # Disable features we don't need (VAD handled separately in pipeline)
                vad=False,
                demucs=False,
                # Timestamp refinement based on silence
                suppress_silence=True,
                suppress_word_ts=True,
                # Verbose output
                verbose=False,
            )

            # Store detected language
            self._detected_language = detected_language_holder[0]

            # Set language on result if detected
            if self._detected_language and hasattr(result, 'language'):
                # Map qwen language names to codes if needed
                result.language = self._map_language_code(self._detected_language)

            # DIAGNOSTIC: Log regrouped result stats
            segment_count = len(result.segments) if result.segments else 0
            logger.info(f"[DIAG] stable_whisper.transcribe_any() result:")
            logger.info(f"[DIAG]   segment count: {segment_count}")
            if result.segments:
                total_text = ''.join(seg.text for seg in result.segments)
                logger.info(f"[DIAG]   total text length: {len(total_text)} chars")
                logger.info(f"[DIAG]   first segment: '{result.segments[0].text[:50]}...'")
                logger.info(f"[DIAG]   last segment: '{result.segments[-1].text[:50]}...'")

            logger.debug(f"transcribe_any returned {segment_count} segments")

            # Apply Japanese-specific post-processing (v1.8.4+)
            if self._postprocessor and result.segments:
                detected_lang = self._map_language_code(self._detected_language) if self._detected_language else None
                logger.debug(f"Applying Japanese post-processing (preset={self.postprocess_preset}, lang={detected_lang})")
                result = self._postprocessor.process(
                    result,
                    preset=self.postprocess_preset,
                    language=detected_lang,
                    skip_if_not_japanese=True
                )
                logger.debug(f"Post-processing complete, {len(result.segments)} segments after")

            return result

        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory. Trying with reduced batch size...")
            original_batch = self.batch_size
            self.batch_size = max(1, self.batch_size // 2)

            # Reload model with smaller batch
            self.unload_model()
            self.load_model()
            qwen_model = self.model  # C1 fix: update closure reference to new model

            # Retry
            result = stable_whisper.transcribe_any(
                inference_func=qwen_inference,
                audio=str(audio_path),
                audio_type='str',
                regroup=True,
                vad=False,
                demucs=False,
                suppress_silence=True,
                suppress_word_ts=True,
                verbose=False,
            )

            self._detected_language = detected_language_holder[0]
            logger.info(f"Retry successful with batch_size={self.batch_size} (was {original_batch})")

            # Apply Japanese-specific post-processing (v1.8.4+) - also for retry path
            if self._postprocessor and result.segments:
                detected_lang = self._map_language_code(self._detected_language) if self._detected_language else None
                result = self._postprocessor.process(
                    result,
                    preset=self.postprocess_preset,
                    language=detected_lang,
                    skip_if_not_japanese=True
                )

            return result

    def _transcribe_without_aligner(self, audio_path: Path, effective_context: str = "") -> stable_whisper.WhisperResult:
        """
        Transcribe without ForcedAligner (no word timestamps).

        Returns a WhisperResult with a single segment containing the full text.
        This is a fallback when use_aligner=False.

        Args:
            audio_path: Path to audio file
            effective_context: Context string for ASR (may be per-call override)

        Returns:
            WhisperResult with single segment
        """
        try:
            # Normalize language to qwen-asr format (e.g., 'ja' -> 'Japanese')
            qwen_language = self._normalize_language_for_qwen(self.language)

            results = self.model.transcribe(
                audio=str(audio_path),
                context=effective_context,  # Pass context to improve accuracy
                language=qwen_language,
                return_time_stamps=False,
            )

            if not results:
                logger.warning("qwen-asr returned empty results")
                # Return empty result
                return stable_whisper.WhisperResult({'segments': [], 'language': 'ja'})

            result = results[0]
            full_text = getattr(result, 'text', '').strip()

            # Store detected language
            if hasattr(result, 'language'):
                self._detected_language = result.language

            if not full_text:
                logger.warning("qwen-asr returned empty text")
                return stable_whisper.WhisperResult({'segments': [], 'language': 'ja'})

            # Create WhisperResult with single segment
            # Note: Without timestamps, we set start=0 and end=0
            # The pipeline will need to estimate duration from audio length
            result_dict = {
                'language': self._map_language_code(self._detected_language) if self._detected_language else 'ja',
                'segments': [
                    {
                        'start': 0.0,
                        'end': 0.0,  # Unknown without timestamps
                        'text': full_text,
                        'words': [],  # No word-level data
                    }
                ]
            }

            return stable_whisper.WhisperResult(result_dict)

        except Exception as e:
            logger.error(f"Error in _transcribe_without_aligner: {type(e).__name__}: {e}")
            raise

    # =========================================================================
    # Decoupled Assembly Line Methods (v1.8.8+)
    # =========================================================================
    # These methods enable the "assembly" input mode where text generation and
    # forced alignment run as separate VRAM-exclusive phases. This allows:
    #   - Higher batch_size for text-only ASR (no aligner loaded)
    #   - Mid-pipeline sanitization between generation and alignment
    #   - Future vLLM integration (ASR supports vLLM, aligner does not)
    #
    # Usage pattern from QwenPipeline:
    #   asr.load_model_text_only()
    #   texts = asr.transcribe_text_only(paths, contexts, language)
    #   asr.unload_model()           # free ASR VRAM
    #   clean_texts = sanitize(texts) # mid-pipeline cleaning
    #   asr.load_aligner_only()
    #   timestamps = asr.align_standalone(paths, clean_texts, language)
    #   asr.unload_model()           # free aligner VRAM

    def load_model_text_only(self) -> None:
        """
        Load Qwen3-ASR model WITHOUT the ForcedAligner.

        Saves ~1.2GB VRAM compared to coupled loading, enabling higher
        batch_size on consumer GPUs (e.g., batch_size=4 on 12GB cards).
        """
        if self.model is not None:
            logger.debug("ASR model already loaded (text-only mode)")
            return

        # Detect device and dtype
        self._device = self._detect_device()
        self._dtype = self._detect_dtype(self._device)
        attn_impl = self._detect_attn_implementation()

        logger.info("Loading Qwen3-ASR model (text-only, no aligner)...")
        logger.info(f"  Model:    {self.model_id}")
        logger.info(f"  Device:   {self._device}")
        logger.info(f"  Dtype:    {self._dtype}")
        logger.info(f"  Batch:    {self.batch_size}")
        logger.info(f"  Attention: {attn_impl}")

        start_time = time.time()

        try:
            from qwen_asr import Qwen3ASRModel

            model_kwargs = {
                "dtype": self._dtype,
                "device_map": self._device,
                "max_inference_batch_size": self.batch_size,
                "max_new_tokens": self.max_new_tokens,
            }

            if attn_impl and attn_impl != "sdpa":
                model_kwargs["attn_implementation"] = attn_impl

            # NO forced_aligner argument — text-only mode
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_id, **model_kwargs
            )

            load_time = time.time() - start_time
            logger.info(f"  Loaded (text-only) in {load_time:.1f}s")

            # Apply generation safety controls (v1.8.9+)
            self._apply_generation_config()

        except ImportError as e:
            raise ImportError(
                "qwen-asr package required for QwenASR. "
                "Install with: pip install qwen-asr"
            ) from e

    def load_aligner_only(self) -> None:
        """
        Load standalone Qwen3-ForcedAligner WITHOUT the ASR model.

        The aligner is a 0.6B parameter non-autoregressive model that
        maps text tokens to audio positions. Loading it alone uses ~1.2GB VRAM.

        Prerequisite: self.model should be None (unloaded) to free VRAM.
        """
        if self._standalone_aligner is not None:
            logger.debug("Standalone aligner already loaded")
            return

        if self.model is not None:
            logger.warning(
                "ASR model still loaded while loading aligner — "
                "VRAM may be insufficient. Call unload_model() first."
            )

        # Ensure device/dtype are detected
        if self._device is None:
            self._device = self._detect_device()
        if self._dtype is None:
            self._dtype = self._detect_dtype(self._device)

        logger.info("Loading standalone ForcedAligner...")
        logger.info(f"  Aligner:  {self.aligner_id}")
        logger.info(f"  Device:   {self._device}")
        logger.info(f"  Dtype:    {self._dtype}")

        start_time = time.time()

        try:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

            self._standalone_aligner = Qwen3ForcedAligner.from_pretrained(
                self.aligner_id,
                dtype=self._dtype,
                device_map=self._device,
            )

            load_time = time.time() - start_time
            logger.info(f"  Aligner loaded in {load_time:.1f}s")

        except ImportError as e:
            raise ImportError(
                "qwen-asr package required for ForcedAligner. "
                "Install with: pip install qwen-asr"
            ) from e

    def transcribe_text_only(
        self,
        audio_paths: List[Union[str, Path]],
        contexts: Optional[List[str]] = None,
        language: Optional[str] = None,
        audio_durations: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Batch text generation WITHOUT forced alignment.

        Feeds audio to Qwen3-ASR and returns raw transcription text.
        The ASR model must be loaded via load_model_text_only() first
        (or will be loaded lazily).

        Processes scenes one at a time with tqdm progress and per-scene
        logging to eliminate the silent gap during long batch processing.

        Args:
            audio_paths: List of paths to audio files (one per scene)
            contexts: Optional per-scene context strings. If None, empty
                      context is used for all scenes.
            language: Language code (e.g., 'ja') or None for auto-detect.
                      Applied to all scenes uniformly.
            audio_durations: Optional per-scene audio durations in seconds.
                             Used for progress display only.

        Returns:
            List of raw transcription text strings (one per audio file).
            Empty string for scenes that produced no speech.
        """
        from tqdm import tqdm

        n = len(audio_paths)
        if n == 0:
            return []

        # Lazy load
        if self.model is None:
            self.load_model_text_only()

        # Normalize language to qwen-asr format
        qwen_language = self._normalize_language_for_qwen(language)

        # Build context list (per-scene or broadcast)
        if contexts is None:
            ctx_list = [""] * n
        else:
            if len(contexts) != n:
                raise ValueError(
                    f"contexts length ({len(contexts)}) must match "
                    f"audio_paths length ({n})"
                )
            ctx_list = contexts

        # Convert paths to strings for qwen-asr
        audio_strs = [str(p) for p in audio_paths]

        logger.info(
            "Assembly: Text generation — %d scenes, language=%s",
            n, qwen_language or "auto",
        )

        texts = []
        batch_start = time.time()

        with tqdm(
            total=n,
            desc="ASR Text Gen",
            unit="scene",
            bar_format="{desc} {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i in range(n):
                dur = audio_durations[i] if audio_durations else None
                dur_str = f"{dur:.0f}s" if dur else "?"
                pbar.set_description(f"ASR Text Gen | Scene {i+1}/{n} ({dur_str} audio)")

                scene_start = time.time()

                # Dynamic token budget (v1.8.9+): scale max_new_tokens to audio duration
                scene_token_limit = self._compute_dynamic_token_limit(dur or 0)
                original_max_tokens = self.model.max_new_tokens
                self.model.max_new_tokens = scene_token_limit
                try:
                    result = self.model.transcribe(
                        audio=[audio_strs[i]],
                        context=[ctx_list[i]],
                        language=qwen_language,
                        return_time_stamps=False,
                    )
                finally:
                    self.model.max_new_tokens = original_max_tokens

                scene_elapsed = time.time() - scene_start
                text = getattr(result[0], 'text', '').strip() if result else ''

                # Per-scene log line (visible even without tqdm)
                chars = len(text)
                speed = chars / scene_elapsed if scene_elapsed > 0 else 0
                budget_info = f", budget={scene_token_limit}" if scene_token_limit != self.max_new_tokens else ""
                logger.info(
                    "  Scene %d/%d: %d chars in %.1fs (%.1f chars/s) | audio=%s%s",
                    i + 1, n, chars, scene_elapsed, speed, dur_str, budget_info,
                )

                # Store detected language from first non-empty result
                if text and self._detected_language is None and result:
                    lang = getattr(result[0], 'language', None)
                    if lang:
                        self._detected_language = lang

                pbar.update(1)
                texts.append(text)

        elapsed = time.time() - batch_start
        avg = elapsed / n if n > 0 else 0
        logger.info(
            "Assembly: Text generation complete — %d scenes in %.1fs (avg %.1fs/scene)",
            n, elapsed, avg,
        )

        return texts

    def align_standalone(
        self,
        audio_paths: List[Union[str, Path]],
        texts: List[str],
        language: Optional[str] = None,
        audio_durations: Optional[List[float]] = None,
    ) -> List[Optional[Any]]:
        """
        Batch forced alignment using the standalone aligner.

        Takes pre-sanitized text and aligns it against the original audio
        to produce word-level timestamps. The standalone aligner must be
        loaded via load_aligner_only() first (or will be loaded lazily).

        Processes scenes one at a time with tqdm progress and per-scene
        logging to provide continuous feedback during alignment.

        Args:
            audio_paths: List of paths to audio files (one per scene).
            texts: List of sanitized text strings (one per scene).
                   Empty strings are skipped (no alignment needed).
            language: Language code (e.g., 'ja'). Required by the aligner
                      for language-specific tokenization (nagisa for Japanese).
            audio_durations: Optional per-scene audio durations in seconds.
                             Used for progress display only.

        Returns:
            List of ForcedAlignResult objects (one per scene).
            None for scenes with empty text (silence/music).
        """
        from tqdm import tqdm

        n = len(audio_paths)
        if n != len(texts):
            raise ValueError(
                f"audio_paths length ({n}) must match texts length ({len(texts)})"
            )

        # Lazy load
        if self._standalone_aligner is None:
            self.load_aligner_only()

        # Resolve language for aligner (requires canonical name like "Japanese")
        qwen_language = self._normalize_language_for_qwen(language) or "Japanese"

        # Separate scenes with text from empty scenes
        to_align_indices = []
        to_align_audio = []
        to_align_text = []
        for i, (path, text) in enumerate(zip(audio_paths, texts)):
            if text.strip():
                to_align_indices.append(i)
                to_align_audio.append(str(path))
                to_align_text.append(text)

        n_with_text = len(to_align_indices)
        logger.info(
            "Assembly: Alignment — %d/%d scenes with text, language=%s",
            n_with_text, n, qwen_language,
        )

        # Initialize results with None for all scenes
        results: List[Optional[Any]] = [None] * n

        if not to_align_indices:
            logger.info("Assembly: No scenes with text to align")
            return results

        batch_start = time.time()

        with tqdm(
            total=n_with_text,
            desc="Aligning",
            unit="scene",
            bar_format="{desc} {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for k, idx in enumerate(to_align_indices):
                dur = audio_durations[idx] if audio_durations else None
                dur_str = f"{dur:.0f}s" if dur else "?"
                chars = len(to_align_text[k])
                pbar.set_description(
                    f"Aligning | Scene {idx+1}/{n} ({dur_str}, {chars} chars)"
                )

                scene_start = time.time()

                align_result = self._standalone_aligner.align(
                    audio=[to_align_audio[k]],
                    text=[to_align_text[k]],
                    language=qwen_language,
                )

                scene_elapsed = time.time() - scene_start
                results[idx] = align_result[0]

                # Per-scene log line
                logger.info(
                    "  Scene %d/%d: aligned in %.1fs | audio=%s, %d chars",
                    idx + 1, n, scene_elapsed, dur_str, chars,
                )

                pbar.update(1)

        elapsed = time.time() - batch_start
        avg = elapsed / n_with_text if n_with_text > 0 else 0
        logger.info(
            "Assembly: Alignment complete — %d scenes in %.1fs (avg %.1fs/scene)",
            n_with_text, elapsed, avg,
        )

        return results

    # =========================================================================
    # Language Mapping Constants
    # =========================================================================
    # qwen-asr supported languages (from error message in qwen_asr library)
    QWEN_SUPPORTED_LANGUAGES = {
        'Chinese', 'English', 'Cantonese', 'Arabic', 'German', 'French',
        'Spanish', 'Portuguese', 'Indonesian', 'Italian', 'Korean', 'Russian',
        'Thai', 'Vietnamese', 'Japanese', 'Turkish', 'Hindi', 'Malay', 'Dutch',
        'Swedish', 'Danish', 'Finnish', 'Polish', 'Czech', 'Filipino', 'Persian',
        'Greek', 'Romanian', 'Hungarian', 'Macedonian'
    }

    # Comprehensive mapping: various input formats → qwen-asr canonical names
    # Handles: ISO codes, full names, case variations, common aliases
    LANGUAGE_TO_QWEN_MAP = {
        # Japanese - most important for WhisperJAV
        'ja': 'Japanese',
        'jp': 'Japanese',  # Common alias
        'jpn': 'Japanese',  # ISO 639-2
        'japanese': 'Japanese',
        # English
        'en': 'English',
        'eng': 'English',
        'english': 'English',
        # Chinese
        'zh': 'Chinese',
        'zho': 'Chinese',
        'chi': 'Chinese',
        'chinese': 'Chinese',
        'mandarin': 'Chinese',
        # Cantonese
        'yue': 'Cantonese',
        'cantonese': 'Cantonese',
        # Korean
        'ko': 'Korean',
        'kor': 'Korean',
        'korean': 'Korean',
        # Arabic
        'ar': 'Arabic',
        'ara': 'Arabic',
        'arabic': 'Arabic',
        # German
        'de': 'German',
        'deu': 'German',
        'ger': 'German',
        'german': 'German',
        # French
        'fr': 'French',
        'fra': 'French',
        'fre': 'French',
        'french': 'French',
        # Spanish
        'es': 'Spanish',
        'spa': 'Spanish',
        'spanish': 'Spanish',
        # Portuguese
        'pt': 'Portuguese',
        'por': 'Portuguese',
        'portuguese': 'Portuguese',
        # Indonesian
        'id': 'Indonesian',
        'ind': 'Indonesian',
        'indonesian': 'Indonesian',
        # Italian
        'it': 'Italian',
        'ita': 'Italian',
        'italian': 'Italian',
        # Russian
        'ru': 'Russian',
        'rus': 'Russian',
        'russian': 'Russian',
        # Thai
        'th': 'Thai',
        'tha': 'Thai',
        'thai': 'Thai',
        # Vietnamese
        'vi': 'Vietnamese',
        'vie': 'Vietnamese',
        'vietnamese': 'Vietnamese',
        # Turkish
        'tr': 'Turkish',
        'tur': 'Turkish',
        'turkish': 'Turkish',
        # Hindi
        'hi': 'Hindi',
        'hin': 'Hindi',
        'hindi': 'Hindi',
        # Malay
        'ms': 'Malay',
        'msa': 'Malay',
        'may': 'Malay',
        'malay': 'Malay',
        # Dutch
        'nl': 'Dutch',
        'nld': 'Dutch',
        'dut': 'Dutch',
        'dutch': 'Dutch',
        # Swedish
        'sv': 'Swedish',
        'swe': 'Swedish',
        'swedish': 'Swedish',
        # Danish
        'da': 'Danish',
        'dan': 'Danish',
        'danish': 'Danish',
        # Finnish
        'fi': 'Finnish',
        'fin': 'Finnish',
        'finnish': 'Finnish',
        # Polish
        'pl': 'Polish',
        'pol': 'Polish',
        'polish': 'Polish',
        # Czech
        'cs': 'Czech',
        'ces': 'Czech',
        'cze': 'Czech',
        'czech': 'Czech',
        # Filipino
        'fil': 'Filipino',
        'tl': 'Filipino',  # Tagalog maps to Filipino
        'tagalog': 'Filipino',
        'filipino': 'Filipino',
        # Persian
        'fa': 'Persian',
        'fas': 'Persian',
        'per': 'Persian',
        'persian': 'Persian',
        'farsi': 'Persian',
        # Greek
        'el': 'Greek',
        'ell': 'Greek',
        'gre': 'Greek',
        'greek': 'Greek',
        # Romanian
        'ro': 'Romanian',
        'ron': 'Romanian',
        'rum': 'Romanian',
        'romanian': 'Romanian',
        # Hungarian
        'hu': 'Hungarian',
        'hun': 'Hungarian',
        'hungarian': 'Hungarian',
        # Macedonian
        'mk': 'Macedonian',
        'mkd': 'Macedonian',
        'mac': 'Macedonian',
        'macedonian': 'Macedonian',
    }

    # Reverse mapping: qwen-asr names → ISO codes (for stable-ts output)
    QWEN_TO_ISO_MAP = {
        'japanese': 'ja',
        'english': 'en',
        'chinese': 'zh',
        'cantonese': 'yue',
        'korean': 'ko',
        'arabic': 'ar',
        'german': 'de',
        'french': 'fr',
        'spanish': 'es',
        'portuguese': 'pt',
        'indonesian': 'id',
        'italian': 'it',
        'russian': 'ru',
        'thai': 'th',
        'vietnamese': 'vi',
        'turkish': 'tr',
        'hindi': 'hi',
        'malay': 'ms',
        'dutch': 'nl',
        'swedish': 'sv',
        'danish': 'da',
        'finnish': 'fi',
        'polish': 'pl',
        'czech': 'cs',
        'filipino': 'fil',
        'persian': 'fa',
        'greek': 'el',
        'romanian': 'ro',
        'hungarian': 'hu',
        'macedonian': 'mk',
    }

    def _normalize_language_for_qwen(self, language: Optional[str]) -> Optional[str]:
        """
        Normalize language input to qwen-asr expected format.

        qwen-asr expects full language names like 'Japanese', 'English', etc.
        This method accepts various input formats and converts them to the
        canonical qwen-asr format.

        Handles:
            - ISO 639-1 codes: 'ja', 'en', 'zh'
            - ISO 639-2 codes: 'jpn', 'eng', 'zho'
            - Full names: 'Japanese', 'English', 'Chinese'
            - Case variations: 'JA', 'JAPANESE', 'japanese'
            - Common aliases: 'jp' for Japanese

        Args:
            language: Input language in any format, or None for auto-detect

        Returns:
            Canonical qwen-asr language name (e.g., 'Japanese'), or None for auto-detect
        """
        # None = auto-detect (valid for qwen-asr)
        if language is None:
            return None

        # Empty string = auto-detect
        if not language or not language.strip():
            return None

        # Normalize: strip whitespace, lowercase for lookup
        lang_normalized = language.strip().lower()

        # Direct lookup in mapping
        if lang_normalized in self.LANGUAGE_TO_QWEN_MAP:
            result = self.LANGUAGE_TO_QWEN_MAP[lang_normalized]
            logger.debug(f"Language normalized: '{language}' -> '{result}'")
            return result

        # Check if already a valid qwen-asr language name (case-insensitive)
        for supported in self.QWEN_SUPPORTED_LANGUAGES:
            if lang_normalized == supported.lower():
                logger.debug(f"Language already valid: '{language}' -> '{supported}'")
                return supported

        # Unknown language - log warning and pass through
        # (let qwen-asr handle the error with its detailed message)
        logger.warning(
            f"Unknown language '{language}' - passing to qwen-asr as-is. "
            f"Supported: {', '.join(sorted(self.QWEN_SUPPORTED_LANGUAGES))}"
        )
        return language

    def _map_language_code(self, language: str) -> str:
        """
        Map qwen-asr language names to ISO language codes.

        qwen-asr returns language names like "Japanese", "English", etc.
        We need to convert these to codes like "ja", "en" for stable-ts.

        Args:
            language: Language name from qwen-asr

        Returns:
            ISO language code
        """
        if not language:
            return 'ja'  # Default to Japanese

        lang_lower = language.lower()

        # Use the reverse mapping
        if lang_lower in self.QWEN_TO_ISO_MAP:
            return self.QWEN_TO_ISO_MAP[lang_lower]

        # Fallback: use first 2 characters as code
        return lang_lower[:2] if len(lang_lower) >= 2 else 'ja'

    def unload_model(self) -> None:
        """Free GPU memory by unloading the ASR model and/or standalone aligner."""
        unloaded = []

        if self.model is not None:
            try:
                del self.model
            except Exception as e:
                logger.warning(f"Error deleting ASR model: {e}")
            finally:
                self.model = None
            unloaded.append("ASR model")

        if self._standalone_aligner is not None:
            try:
                del self._standalone_aligner
            except Exception as e:
                logger.warning(f"Error deleting standalone aligner: {e}")
            finally:
                self._standalone_aligner = None
            unloaded.append("standalone aligner")

        if unloaded:
            # Force garbage collection
            try:
                gc.collect()
            except Exception as e:
                logger.warning(f"Error during garbage collection: {e}")

            # NOTE: CUDA cache cleanup is handled by caller via safe_cuda_cleanup()
            # This keeps ASR modules free of subprocess-awareness logic.

            logger.debug("Unloaded: %s — GPU memory freed", ", ".join(unloaded))

    def cleanup(self) -> None:
        """Cleanup resources. Alias for unload_model()."""
        self.unload_model()

    @property
    def detected_language(self) -> Optional[str]:
        """Get the language detected in the last transcription."""
        return self._detected_language

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.unload_model()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"QwenASR("
            f"model_id='{self.model_id}', "
            f"device='{self._device or self.device_request}', "
            f"dtype='{self._dtype or self.dtype_request}', "
            f"use_aligner={self.use_aligner}"
            f")"
        )
