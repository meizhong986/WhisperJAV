"""
FireRedVAD speech segmentation backend (v1.9.0, EXPERIMENTAL).

Industrial-grade DFSMN-based VAD from FireRedTeam (~0.6M params, ~2.2MB),
claiming 100+ language coverage. Apache-2.0 licensed.

Requires the fireredvad package:
    pip install fireredvad        (CPU — sufficient for a 0.6M-param model)
    pip install fireredvad[gpu]   (optional GPU)

Model weights are auto-downloaded from HuggingFace (FireRedTeam/FireRedVAD,
VAD subfolder only) on first use, mirroring the WhisperSeg backend's pattern.

See: https://github.com/FireRedTeam/FireRedVAD

Design notes:
- FireRedVAD's offline detector consumes a 16kHz mono 16-bit PCM WAV *file*
  and returns segment timestamps in seconds plus frame probabilities. The
  upstream API has no array input, so array inputs are written to a temporary
  WAV first (cheap relative to inference).
- FireRedVAD's config knobs are frame-based (10ms frames). This backend maps
  WhisperJAV's standard ms/s parameter names onto them, and keeps padding and
  grouping in WhisperJAV (extend_speech_frame stays 0) so start/end pads are
  asymmetric and grouping matches the other backends exactly.
"""

from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import tempfile
import time
import logging
import threading

import numpy as np

from ..base import SpeechSegment, SegmentationResult
from .ten import group_segments

logger = logging.getLogger("whisperjav")

# HuggingFace repo carrying the pretrained models. Only the offline VAD
# subfolder is downloaded (Stream-VAD and AED are not used).
_HF_REPO_ID = "FireRedTeam/FireRedVAD"
_HF_SUBFOLDER = "VAD"

# FireRedVAD frame length. Upstream README defaults (min_speech_frame=20
# ≈ 200ms, max_speech_frame=2000 ≈ 20s, chunk_max_frame=30000 ≈ 300s) are
# consistent with the standard 10ms DFSMN frame.
_FRAME_MS = 10


class FireRedVadSpeechSegmenter:
    """
    FireRedVAD speech segmentation backend (experimental).

    Pipeline:
        1. Normalize input to a 16kHz mono PCM16 WAV file
        2. FireRedVAD offline detection (its own smooth/min/max/merge logic,
           frame-based, mapped from WhisperJAV ms/s parameters)
        3. Apply asymmetric padding (start_pad_ms, end_pad_ms) in WhisperJAV
        4. Group into ASR-compatible chunks (shared group_segments helper)

    Example:
        segmenter = FireRedVadSpeechSegmenter(threshold=0.4)
        result = segmenter.segment(audio_path)
    """

    def __init__(
        self,
        threshold: float = 0.4,
        smooth_window_size: int = 5,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 200,
        max_speech_duration_s: Optional[float] = 6.0,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = None,
        start_pad_ms: int = 50,
        end_pad_ms: int = 150,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Initialize FireRedVAD segmenter.

        Args:
            threshold: Speech probability threshold [0.0, 1.0]; maps to
                FireRedVadConfig.speech_threshold. Upstream default 0.4.
            smooth_window_size: Probability smoothing window (frames).
            min_speech_duration_ms: Minimum speech segment duration; maps to
                min_speech_frame (10ms frames).
            min_silence_duration_ms: Minimum silence gap to keep segments
                separate; maps to both min_silence_frame and
                merge_silence_frame (shorter gaps are merged upstream).
            max_speech_duration_s: Maximum single-segment duration; maps to
                max_speech_frame. None/0 falls back to the upstream 20s cap.
                Default 6s (JAV-capped — upstream's own 20s default allowed
                ~9s segments; upstream splits overlong runs at the
                lowest-probability frame, so only segments exceeding the cap
                are affected).
            chunk_threshold_s: Gap threshold for segment grouping (seconds).
            max_group_duration_s: Maximum duration for a segment group.
            start_pad_ms: Milliseconds to pad before segment start.
            end_pad_ms: Milliseconds to pad after segment end.
            use_gpu: Run the DFSMN model on GPU. Default False — the model is
                ~0.6M params and CPU inference avoids competing with ASR VRAM.
            **kwargs: Ignored (forward compatibility with factory splats).
        """
        self.threshold = float(threshold)
        self.smooth_window_size = int(smooth_window_size)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.max_speech_duration_s = (
            float(max_speech_duration_s) if max_speech_duration_s else 0.0
        )
        self.chunk_threshold_s = float(chunk_threshold_s) if chunk_threshold_s is not None else 1.0
        self.max_group_duration_s = (
            float(max_group_duration_s) if max_group_duration_s is not None else 29.0
        )
        self.start_pad_ms = int(start_pad_ms)
        self.end_pad_ms = int(end_pad_ms)
        self.use_gpu = bool(use_gpu)

        # Lazy-loaded model with thread lock (mirrors TenSpeechSegmenter)
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "firered-vad"

    @property
    def display_name(self) -> str:
        return "FireRedVAD (experimental)"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> str:
        """Download (or reuse cached) FireRedVAD offline-VAD weights from HF."""
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(
            repo_id=_HF_REPO_ID,
            allow_patterns=[f"{_HF_SUBFOLDER}/*"],
        )
        model_dir = os.path.join(snapshot_dir, _HF_SUBFOLDER)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"FireRedVAD snapshot at {snapshot_dir} has no '{_HF_SUBFOLDER}/' "
                "subfolder — upstream repo layout may have changed."
            )
        return model_dir

    def _build_config(self):
        """Map WhisperJAV ms/s parameters onto FireRedVadConfig frame counts."""
        from fireredvad import FireRedVadConfig

        max_speech_frame = (
            int(self.max_speech_duration_s * 1000 / _FRAME_MS)
            if self.max_speech_duration_s > 0
            else 2000  # upstream default (~20s)
        )
        return FireRedVadConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=self.smooth_window_size,
            speech_threshold=self.threshold,
            min_speech_frame=max(1, self.min_speech_duration_ms // _FRAME_MS),
            max_speech_frame=max_speech_frame,
            min_silence_frame=max(1, self.min_silence_duration_ms // _FRAME_MS),
            merge_silence_frame=max(0, self.min_silence_duration_ms // _FRAME_MS),
            extend_speech_frame=0,  # padding is applied in WhisperJAV (asymmetric)
        )

    def _ensure_model(self) -> None:
        """Load FireRedVAD model if not already loaded (thread-safe)."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                from fireredvad import FireRedVad
            except ImportError:
                raise ImportError(
                    "FireRedVAD requires the fireredvad package. Install with:\n"
                    "pip install fireredvad"
                )

            model_dir = self._resolve_model_dir()
            logger.debug(
                "Loading FireRedVAD model from %s (threshold=%s, use_gpu=%s)",
                model_dir, self.threshold, self.use_gpu,
            )
            try:
                self._model = FireRedVad.from_pretrained(model_dir, self._build_config())
                logger.debug("FireRedVAD model loaded")
            except Exception as e:
                logger.error("Failed to load FireRedVAD model: %s", e, exc_info=True)
                raise

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs,
    ) -> SegmentationResult:
        """
        Detect speech segments using FireRedVAD.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio (arrays only; files carry
                their own rate)
            **kwargs: Ignored

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        self._ensure_model()

        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        duration = len(audio_data) / actual_sr if actual_sr else 0.0

        if actual_sr != 16000:
            audio_data = self._resample_audio(audio_data, actual_sr, 16000)
            actual_sr = 16000
            duration = len(audio_data) / actual_sr

        try:
            raw_segments = self._detect(audio_data)
            padded = self._apply_padding(raw_segments, duration)
            segments = self._to_speech_segments(padded)
        except Exception as e:
            logger.error("FireRedVAD segmentation failed: %s", e, exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        groups = group_segments(segments, self.max_group_duration_s, self.chunk_threshold_s)

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _detect(self, audio_16k: np.ndarray) -> List[Dict[str, Any]]:
        """Run FireRedVAD offline detection on 16kHz mono float audio.

        The upstream API consumes a WAV file path, so the (already normalized)
        audio is written to a temporary 16-bit PCM WAV first.

        Returns list of dicts with 'start', 'end' (seconds) and 'confidence'.
        """
        import soundfile as sf

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="fireredvad_")
        os.close(tmp_fd)
        try:
            sf.write(tmp_path, audio_16k.astype(np.float32), 16000, subtype="PCM_16")
            result, probs = self._model.detect(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - best-effort temp cleanup
                pass

        timestamps = (result or {}).get("timestamps") or []
        frame_probs = self._as_prob_array(probs)

        segments: List[Dict[str, Any]] = []
        for ts in timestamps:
            try:
                seg_start, seg_end = float(ts[0]), float(ts[1])
            except (TypeError, ValueError, IndexError):
                logger.warning("FireRedVAD returned unparseable timestamp: %r", ts)
                continue
            if seg_end <= seg_start:
                continue
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "confidence": self._segment_confidence(frame_probs, seg_start, seg_end),
            })
        return segments

    @staticmethod
    def _as_prob_array(probs: Any) -> Optional[np.ndarray]:
        """Best-effort conversion of upstream frame probabilities to 1-D array."""
        if probs is None:
            return None
        try:
            arr = np.asarray(probs, dtype=np.float32).reshape(-1)
            return arr if arr.size else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _segment_confidence(
        frame_probs: Optional[np.ndarray], start_sec: float, end_sec: float
    ) -> float:
        """Mean frame probability over [start, end], or 1.0 if unavailable."""
        if frame_probs is None:
            return 1.0
        start_idx = max(0, int(start_sec * 1000 / _FRAME_MS))
        end_idx = min(len(frame_probs), int(end_sec * 1000 / _FRAME_MS))
        if end_idx <= start_idx:
            return 1.0
        return float(np.mean(frame_probs[start_idx:end_idx]))

    def _apply_padding(
        self, segments: List[Dict[str, Any]], audio_duration: float
    ) -> List[Dict[str, Any]]:
        """Apply start/end padding without creating overlaps (TEN semantics)."""
        start_pad_sec = self.start_pad_ms / 1000.0
        end_pad_sec = self.end_pad_ms / 1000.0

        padded: List[Dict[str, Any]] = []
        for seg in segments:
            padded_start = max(0.0, seg["start"] - start_pad_sec)
            padded_end = min(audio_duration, seg["end"] + end_pad_sec)

            if padded and padded_start < padded[-1]["end"]:
                padded_start = padded[-1]["end"]

            if padded_end > padded_start:
                padded.append({
                    "start": padded_start,
                    "end": padded_end,
                    "confidence": seg.get("confidence", 1.0),
                    "raw_start": seg["start"],
                    "raw_end": seg["end"],
                })
        return padded

    def _to_speech_segments(self, segments: List[Dict[str, Any]]) -> List[SpeechSegment]:
        """Convert internal dicts to SpeechSegment objects (16kHz samples)."""
        return [
            SpeechSegment(
                start_sec=seg["start"],
                end_sec=seg["end"],
                start_sample=int(seg["start"] * 16000),
                end_sample=int(seg["end"] * 16000),
                confidence=seg.get("confidence", 1.0),
                metadata={
                    "raw_start": seg.get("raw_start", seg["start"]),
                    "raw_end": seg.get("raw_end", seg["end"]),
                },
            )
            for seg in segments
        ]

    # ------------------------------------------------------------------
    # Audio utilities (mirrors TenSpeechSegmenter)
    # ------------------------------------------------------------------

    def _load_audio(
        self, audio: Union[np.ndarray, Path, str], sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data, actual_sr

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples).astype(audio.dtype)
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), 1 / ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    # ------------------------------------------------------------------
    # Metadata / lifecycle
    # ------------------------------------------------------------------

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            "threshold": self.threshold,
            "smooth_window_size": self.smooth_window_size,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "start_pad_ms": self.start_pad_ms,
            "end_pad_ms": self.end_pad_ms,
            "use_gpu": self.use_gpu,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.debug("FireRedVAD model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """FireRedVAD operates at 16kHz; other rates are resampled."""
        return [16000]

    def __repr__(self) -> str:
        return f"FireRedVadSpeechSegmenter(threshold={self.threshold})"
