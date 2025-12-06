#!/usr/bin/env python3
"""
HuggingFace Transformers ASR Module for WhisperJAV.

Uses the HuggingFace Transformers pipeline with chunked long-form algorithm
for audio transcription. Designed for Japanese audio with kotoba-whisper-v2.2
as default, but supports any HuggingFace whisper model.

Based on: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0#chunked-long-form
"""

# Suppress TensorFlow and oneDNN warnings before importing transformers
# These are loaded as side effects and produce noisy output
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF INFO/WARNING
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Disable oneDNN warnings

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import time

import torch

from whisperjav.utils.logger import logger

# Suppress specific transformers warnings that are informational only
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*is very experimental.*")


class TransformersASR:
    """
    HuggingFace Transformers ASR using chunked long-form algorithm.

    This ASR module uses the HuggingFace transformers pipeline for
    automatic speech recognition with chunk-based processing for
    long audio files.
    """

    # Default values optimized for kotoba-whisper-v2.2
    DEFAULT_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
    DEFAULT_CHUNK_LENGTH = 15  # Optimal for distil-large-v3 architecture
    DEFAULT_STRIDE = None  # None = chunk_length / 6
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_LANGUAGE = "ja"
    DEFAULT_TASK = "transcribe"
    DEFAULT_BEAM_SIZE = 5
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        batch_size: int = DEFAULT_BATCH_SIZE,
        chunk_length_s: int = DEFAULT_CHUNK_LENGTH,
        stride_length_s: Optional[float] = DEFAULT_STRIDE,
        language: str = DEFAULT_LANGUAGE,
        task: str = DEFAULT_TASK,
        timestamps: str = "segment",
        beam_size: int = DEFAULT_BEAM_SIZE,
        temperature: float = DEFAULT_TEMPERATURE,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous: bool = True,
    ):
        """
        Initialize TransformersASR.

        Args:
            model_id: HuggingFace model ID (default: kotoba-tech/kotoba-whisper-v2.2)
            device: Device to use ('auto', 'cuda', 'cpu')
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
            attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager')
            batch_size: Batch size for parallel chunk processing
            chunk_length_s: Chunk length in seconds
            stride_length_s: Overlap between chunks (None = chunk_length/6)
            language: Language code (e.g., 'ja')
            task: Task type ('transcribe' or 'translate')
            timestamps: Timestamp granularity ('segment' or 'word')
            beam_size: Beam size for beam search decoding
            temperature: Sampling temperature (0 = greedy/deterministic)
            compression_ratio_threshold: Filter high compression ratio segments
            logprob_threshold: Filter low confidence segments
            no_speech_threshold: Threshold for non-speech detection
            condition_on_previous: Condition on previous text for coherence
        """
        self.model_id = model_id
        self.device_request = device
        self.dtype_request = dtype
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.language = language
        self.task = task
        self.timestamps = timestamps
        self.beam_size = beam_size
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous = condition_on_previous

        # Pipeline is lazily loaded
        self.pipe = None
        self._device = None
        self._dtype = None

        # Word-level timestamps require batch_size=1
        if self.timestamps == "word" and self.batch_size > 1:
            logger.info(f"Word-level timestamps require batch_size=1. Adjusting from {self.batch_size} to 1.")
            self.batch_size = 1

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
        else:
            return "cpu"

    def _detect_dtype(self, device: str) -> torch.dtype:
        """Detect best dtype for device."""
        if self.dtype_request == "auto":
            if "cuda" in device:
                return torch.bfloat16
            else:
                return torch.float32
        elif self.dtype_request == "float16":
            return torch.float16
        elif self.dtype_request == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    def load_model(self) -> None:
        """
        Load the HuggingFace pipeline.

        This is called lazily on first transcribe() call.
        """
        if self.pipe is not None:
            logger.debug("Pipeline already loaded")
            return

        from transformers import pipeline

        # Detect device and dtype
        self._device = self._detect_device()
        self._dtype = self._detect_dtype(self._device)

        logger.info(f"Loading HF Transformers ASR pipeline...")
        logger.info(f"  Model:    {self.model_id}")
        logger.info(f"  Device:   {self._device}")
        logger.info(f"  Dtype:    {self._dtype}")
        logger.info(f"  Attention: {self.attn_implementation}")
        logger.info(f"  Batch:    {self.batch_size}")

        # Build model_kwargs
        model_kwargs = {}
        if "cuda" in self._device and self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        start_time = time.time()

        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                torch_dtype=self._dtype,
                device=self._device,
                model_kwargs=model_kwargs if model_kwargs else None,
                batch_size=self.batch_size
            )

            load_time = time.time() - start_time
            logger.info(f"  Loaded in {load_time:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio file using chunked long-form algorithm.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
                               (progress_percent, status_message)

        Returns:
            List of segment dictionaries with 'text', 'start', 'end' keys
        """
        # Lazy load model
        if self.pipe is None:
            self.load_model()

        if progress_callback:
            progress_callback(0.0, f"Transcribing: {audio_path.name}")

        # Calculate effective stride
        stride = self.stride_length_s
        if stride is None:
            stride = self.chunk_length_s / 6

        logger.debug(f"Transcribing: {audio_path.name}")
        logger.debug(f"  Chunk: {self.chunk_length_s}s, Stride: {stride:.1f}s")
        logger.debug(f"  Language: {self.language}, Task: {self.task}")
        logger.debug(f"  Timestamps: {self.timestamps}")

        # Log task at INFO level for translation debugging
        if self.task == 'translate':
            logger.info(f"TransformersASR transcribing with task='translate' - output should be in English")

        # Configure generate_kwargs
        generate_kwargs = {
            "language": self.language,
            "task": self.task,
            "num_beams": self.beam_size,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_prev_tokens": self.condition_on_previous,
        }

        # Configure return_timestamps
        if self.timestamps == "word":
            return_timestamps = "word"
        else:
            return_timestamps = True  # Segment-level

        start_time = time.time()

        try:
            result = self.pipe(
                str(audio_path),
                chunk_length_s=self.chunk_length_s,
                stride_length_s=stride,
                return_timestamps=return_timestamps,
                generate_kwargs=generate_kwargs,
                ignore_warning=True,  # Suppress chunk_length_s experimental warning
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA out of memory. Reducing batch size and retrying...")
            # Reduce batch size and retry
            original_batch = self.batch_size
            self.batch_size = max(1, self.batch_size // 2)
            self.unload_model()
            self.load_model()

            result = self.pipe(
                str(audio_path),
                chunk_length_s=self.chunk_length_s,
                stride_length_s=stride,
                return_timestamps=return_timestamps,
                generate_kwargs=generate_kwargs,
                ignore_warning=True,  # Suppress chunk_length_s experimental warning
            )
            logger.info(f"Retry successful with batch_size={self.batch_size} (was {original_batch})")

        process_time = time.time() - start_time

        if progress_callback:
            progress_callback(1.0, "Transcription complete")

        # Convert HF chunks to internal format
        chunks = result.get("chunks", [])
        segments = self._convert_chunks_to_segments(chunks)

        logger.debug(f"Transcription complete in {process_time:.1f}s, {len(segments)} segments")

        # Validate translation output - warn if translation was requested but output appears Japanese
        if self.task == 'translate' and segments:
            # Check if output contains significant Japanese characters
            sample_text = ' '.join(seg['text'] for seg in segments[:5])  # First 5 segments
            japanese_char_count = sum(1 for c in sample_text if '\u3040' <= c <= '\u309f' or  # Hiragana
                                                               '\u30a0' <= c <= '\u30ff' or  # Katakana
                                                               '\u4e00' <= c <= '\u9fff')    # Kanji
            total_chars = len(sample_text.replace(' ', ''))
            if total_chars > 0 and japanese_char_count / total_chars > 0.3:
                logger.warning(f"Translation mode was requested but output appears to be in Japanese "
                               f"({japanese_char_count}/{total_chars} chars are Japanese). "
                               f"This may indicate HuggingFace translation is not working as expected.")
                logger.warning(f"Sample output: {sample_text[:100]}...")
            else:
                logger.info(f"Translation output validation: appears to be English (good)")

        return segments

    def _convert_chunks_to_segments(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """
        Convert HF pipeline chunks to internal segment format.

        Args:
            chunks: List of HF chunks with 'text' and 'timestamp' keys

        Returns:
            List of segments with 'text', 'start', 'end' keys
        """
        segments = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp")

            if not text:
                continue

            # Handle timestamp tuple
            if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start, end = timestamp
                # Handle None values
                if start is None:
                    start = 0.0
                if end is None:
                    # Estimate end time if missing
                    end = start + 2.0
            else:
                # Skip segments without valid timestamps
                continue

            segments.append({
                "text": text,
                "start": float(start),
                "end": float(end)
            })

        return segments

    def unload_model(self) -> None:
        """Free GPU memory by unloading the model."""
        if self.pipe is not None:
            logger.debug("Unloading HF Transformers pipeline...")
            try:
                del self.pipe
            except Exception as e:
                logger.warning(f"Error deleting pipeline: {e}")
            finally:
                self.pipe = None

            # Force garbage collection
            import gc
            try:
                gc.collect()
            except Exception as e:
                logger.warning(f"Error during garbage collection: {e}")

            # NOTE: CUDA cache cleanup is handled by caller via safe_cuda_cleanup()
            # This keeps ASR modules free of subprocess-awareness logic.

            logger.debug("Pipeline unloaded, GPU memory freed")

    def cleanup(self) -> None:
        """Cleanup resources. Alias for unload_model()."""
        self.unload_model()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.unload_model()
        except Exception:
            pass
