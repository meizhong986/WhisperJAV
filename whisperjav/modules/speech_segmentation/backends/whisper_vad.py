"""
Whisper-as-VAD speech segmentation backend.

Uses Whisper model as a neural VAD - transcription segments become speech regions.
This leverages Whisper's deep semantic understanding of speech patterns, providing
more robust speech detection compared to traditional energy/probability-based VADs.

Advantages:
    - Semantic speech understanding (not just acoustic energy)
    - Works well on low SNR audio where traditional VAD fails
    - Gets transcription for free (cached for potential reuse)
    - Proven approach: Used by WhisperX, WhisperSeg (ICASSP 2024)

Variants available:
    - whisper-vad: Default (small model, ~500MB)
    - whisper-vad-tiny: Fastest (~75MB)
    - whisper-vad-base: Balance (~150MB)
    - whisper-vad-medium: Most accurate (~1.5GB)
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging
import json
import hashlib
import os

import numpy as np

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


class WhisperVadSpeechSegmenter:
    """
    Whisper-as-VAD speech segmentation backend.

    Uses faster-whisper to transcribe audio and extract segment timestamps
    as speech regions. Optionally caches transcription results for future use.

    Example:
        segmenter = WhisperVadSpeechSegmenter(variant="whisper-vad")
        result = segmenter.segment(audio_path)
        for seg in result.segments:
            print(f"{seg.start_sec:.2f}s - {seg.end_sec:.2f}s: {seg.metadata.get('text', '')}")
    """

    # Variant configurations
    VARIANTS = {
        "whisper-vad": {
            "model": "small",
            "display_name": "Whisper VAD (small)",
            "description": "Default, good balance (~500MB)",
        },
        "whisper-vad-tiny": {
            "model": "tiny",
            "display_name": "Whisper VAD (tiny)",
            "description": "Fastest, least accurate (~75MB)",
        },
        "whisper-vad-base": {
            "model": "base",
            "display_name": "Whisper VAD (base)",
            "description": "Fast with decent accuracy (~150MB)",
        },
        "whisper-vad-small": {
            "model": "small",
            "display_name": "Whisper VAD (small)",
            "description": "Alias for whisper-vad (~500MB)",
        },
        "whisper-vad-medium": {
            "model": "medium",
            "display_name": "Whisper VAD (medium)",
            "description": "Most accurate, slower (~1.5GB)",
        },
    }

    def __init__(
        self,
        variant: str = "whisper-vad",
        model_size: Optional[str] = None,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        language: str = "ja",
        compute_type: str = "float16",
        device: str = "auto",
        cache_results: bool = True,
        cache_dir: Optional[Path] = None,
        chunk_threshold_s: Optional[float] = None,
        min_speech_duration_ms: int = 100,
        **kwargs
    ):
        """
        Initialize Whisper VAD segmenter.

        Args:
            variant: Segmenter variant (whisper-vad, whisper-vad-tiny, etc.)
            model_size: Override variant's model size (tiny/base/small/medium/large)
            no_speech_threshold: Probability threshold for "no speech" detection.
                                 Higher = stricter speech detection (default 0.6)
            logprob_threshold: Filter segments with avg_logprob below this (default -1.0)
            language: Target language for transcription (default "ja")
            compute_type: GPU compute type (float16/int8/float32)
            device: Device to use (cuda/cpu/auto)
            cache_results: Whether to save transcription JSON for future use
            cache_dir: Directory for cache files (default: .whisperjav_cache next to audio)
            chunk_threshold_s: Gap threshold for segment grouping (default 4.0s)
            min_speech_duration_ms: Minimum speech segment duration (default 100ms)
            **kwargs: Additional parameters for backward compatibility
                - chunk_threshold: Legacy alias for chunk_threshold_s
        """
        # Normalize variant name
        if variant not in self.VARIANTS:
            logger.warning(f"[WhisperVAD] Unknown variant '{variant}', defaulting to whisper-vad")
            variant = "whisper-vad"

        self.variant = variant
        self._model_size = model_size or self.VARIANTS[variant]["model"]
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.language = language
        self.compute_type = compute_type
        self.device = device
        self.cache_results = cache_results
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_speech_duration_ms = min_speech_duration_ms

        # Handle backward compatibility: chunk_threshold (old) -> chunk_threshold_s (new)
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = chunk_threshold_s
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = kwargs["chunk_threshold"]
        else:
            self.chunk_threshold_s = 2.5  # Default (reduced from 4.0 to minimize silence in Whisper input)

        # Lazy-loaded model
        self._model = None

        logger.info(
            f"[WhisperVAD] Initialized: variant={variant}, model={self._model_size}, "
            f"language={language}, no_speech_threshold={no_speech_threshold}"
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return self.variant

    @property
    def display_name(self) -> str:
        """Return user-friendly display name."""
        return self.VARIANTS[self.variant]["display_name"]

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments using Whisper transcription.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Override parameters

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        logger.info(f"[WhisperVAD] Starting segmentation with {self.name} ({self._model_size} model)")

        # Resolve audio path
        audio_path = self._resolve_audio_path(audio, sample_rate)
        if audio_path is None:
            logger.error("[WhisperVAD] Failed to resolve audio path")
            return self._empty_result(start_time)

        try:
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            logger.info(f"[WhisperVAD] Audio duration: {duration:.2f}s")

            # Load model
            self._ensure_model()

            # Transcribe audio
            logger.info("[WhisperVAD] Running Whisper transcription...")
            segments, transcription_info = self._transcribe(audio_path)
            logger.info(f"[WhisperVAD] Transcription complete: {len(segments)} segments")

            # Convert to SpeechSegments
            speech_segments = self._convert_to_speech_segments(segments, sample_rate, duration)

            # Apply minimum duration filter
            speech_segments = self._filter_short_segments(speech_segments)
            logger.info(f"[WhisperVAD] After filtering: {len(speech_segments)} segments")

            # Cache results if enabled
            if self.cache_results:
                self._save_cache(audio_path, segments, transcription_info, duration)

            # Group segments
            groups = self._group_segments(speech_segments)

            elapsed = time.time() - start_time
            coverage = sum(s.duration_sec for s in speech_segments) / duration if duration > 0 else 0

            logger.info(
                f"[WhisperVAD] Segmentation complete: {len(speech_segments)} segments, "
                f"{len(groups)} groups, {coverage:.1%} coverage, {elapsed:.2f}s"
            )

            return SegmentationResult(
                segments=speech_segments,
                groups=groups,
                method=f"{self.name}:{self._model_size}",
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=elapsed,
            )

        except Exception as e:
            logger.error(f"[WhisperVAD] Segmentation failed: {e}", exc_info=True)
            elapsed = time.time() - start_time
            return SegmentationResult(
                segments=[],
                groups=[],
                method=f"{self.name}:error",
                audio_duration_sec=0,
                parameters=self._get_parameters(),
                processing_time_sec=elapsed,
            )

    def _ensure_model(self) -> None:
        """Load Whisper model if not already loaded."""
        if self._model is not None:
            logger.debug("[WhisperVAD] Model already loaded")
            return

        logger.info(f"[WhisperVAD] Loading faster-whisper model: {self._model_size}")
        logger.info("[WhisperVAD] This may download model files on first run...")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for WhisperVAD.\n"
                "Install: pip install faster-whisper"
            )

        # Determine device
        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[WhisperVAD] Auto-detected device: {device}")

        # Adjust compute_type for CPU
        compute_type = self.compute_type
        if device == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"
            logger.info(f"[WhisperVAD] Adjusted compute_type for CPU: {compute_type}")

        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(f"[WhisperVAD] Model loaded successfully on {device}")

    def _transcribe(self, audio_path: Path) -> Tuple[List[Dict], Dict]:
        """
        Transcribe audio and return segments with timestamps.

        Returns:
            Tuple of (segments_list, transcription_info)
        """
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.logprob_threshold,
            vad_filter=False,  # Don't use faster-whisper's Silero VAD
            word_timestamps=False,  # Segment-level is enough for VAD
        )

        # Consume generator to list
        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "compression_ratio": seg.compression_ratio,
            })

        transcription_info = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "duration_after_vad": getattr(info, "duration_after_vad", None),
        }

        return segments, transcription_info

    def _convert_to_speech_segments(
        self,
        segments: List[Dict],
        sample_rate: int,
        duration: float
    ) -> List[SpeechSegment]:
        """Convert Whisper segments to SpeechSegment objects."""
        speech_segments = []

        for seg in segments:
            start_sec = max(0, seg["start"])
            end_sec = min(duration, seg["end"])

            # Skip if segment is outside audio bounds
            if start_sec >= end_sec:
                continue

            # Calculate confidence from avg_logprob (normalize to 0-1 range)
            # avg_logprob is typically negative, closer to 0 is better
            avg_logprob = seg.get("avg_logprob", -0.5)
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

            speech_segments.append(SpeechSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                start_sample=int(start_sec * sample_rate),
                end_sample=int(end_sec * sample_rate),
                confidence=confidence,
                metadata={
                    "text": seg.get("text", ""),
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": seg.get("no_speech_prob", 0),
                    "compression_ratio": seg.get("compression_ratio", 0),
                }
            ))

        return speech_segments

    def _filter_short_segments(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """Filter out segments shorter than minimum duration."""
        min_duration_sec = self.min_speech_duration_ms / 1000.0
        filtered = [s for s in segments if s.duration_sec >= min_duration_sec]

        if len(filtered) < len(segments):
            logger.debug(
                f"[WhisperVAD] Filtered {len(segments) - len(filtered)} short segments "
                f"(< {self.min_speech_duration_ms}ms)"
            )

        return filtered

    def _group_segments(self, segments: List[SpeechSegment]) -> List[List[SpeechSegment]]:
        """Group segments based on time gaps."""
        if not segments:
            return []

        groups: List[List[SpeechSegment]] = [[]]

        for i, segment in enumerate(segments):
            if i > 0:
                prev_end = segments[i - 1].end_sec
                gap = segment.start_sec - prev_end
                if gap > self.chunk_threshold_s:
                    groups.append([])
            groups[-1].append(segment)

        return groups

    def _resolve_audio_path(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Optional[Path]:
        """
        Resolve audio input to a file path.

        If audio is a numpy array, save to temporary file.
        """
        import tempfile

        if isinstance(audio, (str, Path)):
            return Path(audio)

        if isinstance(audio, np.ndarray):
            # Save to temp file
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError("soundfile is required: pip install soundfile")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                self._temp_audio_path = Path(f.name)
                logger.debug(f"[WhisperVAD] Saved temp audio: {f.name}")
                return self._temp_audio_path

        return None

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            # Fallback: load audio to get duration
            try:
                import soundfile as sf
                audio, sr = sf.read(str(audio_path))
                return len(audio) / sr
            except Exception as e:
                logger.warning(f"[WhisperVAD] Could not determine duration: {e}")
                return 0.0

    def _get_cache_path(self, audio_path: Path) -> Path:
        """Get cache file path for the given audio file."""
        if self.cache_dir:
            cache_base = self.cache_dir
        else:
            cache_base = audio_path.parent / ".whisperjav_cache"

        # Create hash of audio file path for unique cache filename
        audio_hash = hashlib.md5(str(audio_path.absolute()).encode()).hexdigest()[:12]
        cache_filename = f"{audio_path.stem}_{audio_hash}_whisper_vad_{self._model_size}.json"

        return cache_base / cache_filename

    def _save_cache(
        self,
        audio_path: Path,
        segments: List[Dict],
        info: Dict,
        duration: float
    ) -> None:
        """Save transcription results to cache file."""
        cache_path = self._get_cache_path(audio_path)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                "audio_file": str(audio_path.absolute()),
                "audio_filename": audio_path.name,
                "model": self._model_size,
                "language": self.language,
                "duration_sec": duration,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "parameters": {
                    "no_speech_threshold": self.no_speech_threshold,
                    "logprob_threshold": self.logprob_threshold,
                    "variant": self.variant,
                },
                "transcription_info": info,
                "segments": segments,
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[WhisperVAD] Cached transcription: {cache_path}")

        except Exception as e:
            logger.warning(f"[WhisperVAD] Failed to save cache: {e}")

    def _empty_result(self, start_time: float) -> SegmentationResult:
        """Return empty result on error."""
        return SegmentationResult(
            segments=[],
            groups=[],
            method=f"{self.name}:error",
            audio_duration_sec=0,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            "variant": self.variant,
            "model_size": self._model_size,
            "language": self.language,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compute_type": self.compute_type,
            "device": self.device,
            "cache_results": self.cache_results,
            "chunk_threshold_s": self.chunk_threshold_s,
            "min_speech_duration_ms": self.min_speech_duration_ms,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        logger.debug("[WhisperVAD] Cleaning up resources...")

        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("[WhisperVAD] Model released")

        # Clean up temp audio file if created
        if hasattr(self, "_temp_audio_path") and self._temp_audio_path:
            try:
                if self._temp_audio_path.exists():
                    self._temp_audio_path.unlink()
                    logger.debug(f"[WhisperVAD] Removed temp file: {self._temp_audio_path}")
            except Exception as e:
                logger.debug(f"[WhisperVAD] Failed to remove temp file: {e}")
            self._temp_audio_path = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("[WhisperVAD] GPU cache cleared")
        except ImportError:
            pass

        logger.info("[WhisperVAD] Resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates."""
        return [16000]  # Whisper models expect 16kHz

    def __repr__(self) -> str:
        return (
            f"WhisperVadSpeechSegmenter("
            f"variant={self.variant!r}, "
            f"model={self._model_size!r}, "
            f"language={self.language!r})"
        )
