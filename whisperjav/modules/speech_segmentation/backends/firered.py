"""
FireRedVAD speech segmentation backend.

FireRedVAD (Xiaohongshu FireRed Team) — industrial-grade DFSMN-based VAD,
97.57% F1 on FLEURS-VAD-102, 100+ languages. Tiny model (~2MB), fast on CPU.

Requires fireredvad package:
    pip install fireredvad

Model weights (~2.4MB) auto-download from HuggingFace on first use:
    FireRedTeam/FireRedVAD (VAD/cmvn.ark + VAD/model.pth.tar, not gated)

See: https://github.com/FireRedTeam/FireRedVAD

Implementation notes:
- FireRedVAD operates on 16kHz mono int16 audio at 10ms frame shift.
  All frame-based config params are derived from the standard ms/s params
  (1 frame = 10ms).
- The package's native postprocessor already implements the full house
  pipeline (smooth -> threshold -> hysteresis state machine -> merge ->
  extend/pad -> split-long-at-prob-minima), so unlike ten.py we only add
  grouping on top.
- extend_speech_frame pads BOTH sides symmetrically, so speech_pad_ms maps
  to it directly (same semantics as Silero's speech_pad_ms).
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import os
import time
import logging
import threading

import numpy as np

from ..base import SpeechSegment, SegmentationResult
from .ten import group_segments

logger = logging.getLogger("whisperjav")

_HF_REPO = "FireRedTeam/FireRedVAD"
_MODEL_FILES = ("cmvn.ark", "model.pth.tar")
_FRAME_MS = 10  # FireRedVAD frame shift (kaldi fbank, frame_shift=10ms)


def _default_cache_dir() -> Path:
    """House cache location (same convention as bs_roformer)."""
    return Path.home() / ".cache" / "whisperjav" / "fireredvad" / "VAD"


class FireRedSpeechSegmenter:
    """
    FireRedVAD speech segmentation backend.

    Pipeline (mostly native to the fireredvad package):
        1. Kaldi fbank features (80 mel, 10ms shift) + CMVN
        2. DFSMN frame-level speech probabilities
        3. Native postprocess: smooth -> threshold -> min speech/silence
           hysteresis -> pad (extend_speech_frame) -> split long segments
           at probability minima
        4. Group into ASR-compatible chunks (house group_segments)

    Example:
        segmenter = FireRedSpeechSegmenter(threshold=0.4)
        result = segmenter.segment(audio_path)
    """

    def __init__(
        self,
        threshold: float = 0.4,
        min_speech_duration_ms: int = 150,
        min_silence_duration_ms: int = 150,
        max_speech_duration_s: Optional[float] = 5.0,
        speech_pad_ms: int = 100,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = 6.0,
        smooth_window_size: int = 5,
        use_gpu: bool = False,
        model_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize FireRedVAD segmenter.

        Args:
            threshold: Speech probability threshold [0.0, 1.0]. Lower values
                detect more speech (more sensitive). Default 0.4 (upstream).
            min_speech_duration_ms: Minimum speech run to enter SPEECH state.
            min_silence_duration_ms: Minimum silence run to leave SPEECH state
                (shorter gaps stay inside the segment).
            max_speech_duration_s: Maximum single-segment duration. Longer
                segments are split natively at probability minima. None/0
                disables splitting (maps to a very large frame count).
            speech_pad_ms: Symmetric padding applied before AND after each
                segment (FireRedVAD extend_speech_frame).
            chunk_threshold_s: Gap threshold for segment grouping (seconds).
            max_group_duration_s: Maximum duration for a segment group.
            smooth_window_size: Probability smoothing window in frames (10ms).
            use_gpu: Run the DFSMN on CUDA. The model is ~2MB; CPU is fast
                and avoids VRAM contention with ASR models. Default False.
            model_dir: Explicit directory containing cmvn.ark + model.pth.tar.
                Default: auto-download to ~/.cache/whisperjav/fireredvad/VAD.
            **kwargs: Ignored extras for forward compatibility.
        """
        self.threshold = float(threshold)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.max_speech_duration_s = (
            float(max_speech_duration_s)
            if max_speech_duration_s else 0.0
        )
        self.speech_pad_ms = int(speech_pad_ms)
        self.chunk_threshold_s = (
            float(chunk_threshold_s) if chunk_threshold_s is not None else 1.0
        )
        self.max_group_duration_s = (
            float(max_group_duration_s) if max_group_duration_s is not None else 6.0
        )
        self.smooth_window_size = int(smooth_window_size)
        self.use_gpu = bool(use_gpu)
        self.model_dir = str(model_dir) if model_dir else None

        # Lazy-loaded model with thread lock
        self._vad = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "firered"

    @property
    def display_name(self) -> str:
        return "FireRedVAD"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> str:
        """Resolve model assets: explicit dir -> cache -> HF download."""
        if self.model_dir:
            missing = [f for f in _MODEL_FILES
                       if not os.path.isfile(os.path.join(self.model_dir, f))]
            if missing:
                raise FileNotFoundError(
                    f"FireRedVAD model_dir '{self.model_dir}' is missing: "
                    f"{', '.join(missing)}"
                )
            return self.model_dir

        cache_dir = _default_cache_dir()
        if all((cache_dir / f).is_file() for f in _MODEL_FILES):
            logger.debug(f"FireRedVAD using cached model at {cache_dir}")
            return str(cache_dir)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download the FireRedVAD model. "
                "Install with: pip install huggingface_hub"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Downloading FireRedVAD model (~2.4MB) from {_HF_REPO} "
            f"to {cache_dir}..."
        )
        try:
            import shutil
            for fname in _MODEL_FILES:
                # Download into the default HF cache, then copy flat into our
                # cache dir (from_pretrained wants cmvn.ark/model.pth.tar
                # directly in model_dir, without the VAD/ prefix).
                local = hf_hub_download(repo_id=_HF_REPO, filename=f"VAD/{fname}")
                shutil.copy2(local, cache_dir / fname)
            logger.info("FireRedVAD model downloaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download FireRedVAD model from {_HF_REPO}: {e}. "
                f"You can download VAD/cmvn.ark and VAD/model.pth.tar manually "
                f"from https://huggingface.co/{_HF_REPO} into {cache_dir}"
            )
        return str(cache_dir)

    def _ensure_model(self) -> None:
        """Load FireRedVAD model if not already loaded (thread-safe)."""
        if self._vad is not None:
            return

        with self._lock:
            if self._vad is not None:
                return

            try:
                from fireredvad import FireRedVad, FireRedVadConfig
            except ImportError:
                raise ImportError(
                    "FireRedVAD requires the fireredvad package. Install with:\n"
                    "pip install fireredvad"
                )

            # Quiet the package's own loggers. Its long-segment splitter emits
            # "Unexpected short speech segment, check vad_postprocessor.py" for
            # every sub-min_speech_frame remainder it produces — expected with
            # max_speech_duration_s splitting on continuous audio, spams the
            # console, and is not user-actionable (we drop those fragments in
            # _timestamps_to_segments anyway).
            logging.getLogger("fireredvad").setLevel(logging.ERROR)

            model_dir = self._resolve_model_dir()

            # Translate house ms/s params into FireRedVAD 10ms-frame units.
            # max_speech 0/None -> effectively unlimited (1 hour of frames).
            max_speech_frame = (
                int(round(self.max_speech_duration_s * 1000 / _FRAME_MS))
                if self.max_speech_duration_s > 0 else 360000
            )
            config = FireRedVadConfig(
                use_gpu=self.use_gpu,
                smooth_window_size=self.smooth_window_size,
                speech_threshold=self.threshold,
                min_speech_frame=max(1, int(round(self.min_speech_duration_ms / _FRAME_MS))),
                max_speech_frame=max_speech_frame,
                min_silence_frame=max(1, int(round(self.min_silence_duration_ms / _FRAME_MS))),
                merge_silence_frame=0,
                extend_speech_frame=int(round(self.speech_pad_ms / _FRAME_MS)),
            )

            logger.debug(
                f"Loading FireRedVAD model from {model_dir} "
                f"(threshold={self.threshold}, use_gpu={self.use_gpu})"
            )
            try:
                self._vad = FireRedVad.from_pretrained(model_dir, config)
                logger.debug("FireRedVAD model loaded")
            except Exception as e:
                logger.error(f"Failed to load FireRedVAD model: {e}", exc_info=True)
                raise

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments using FireRedVAD.

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Ignored

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        self._ensure_model()

        # Load and prepare audio
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr if actual_sr else 0.0

        # FireRedVAD requires 16kHz - resample if needed
        if actual_sr != 16000:
            audio_data = self._resample_audio(audio_data, actual_sr, 16000)
            actual_sr = 16000
            duration = len(audio_data) / actual_sr

        # FireRedVAD's fbank was trained on int16-scale waveforms
        # (their loader uses sf.read(dtype="int16")). Feeding float [-1,1]
        # audio shifts the feature scale and breaks detection.
        audio_int16 = self._convert_to_int16(audio_data)

        try:
            result, probs = self._vad.detect((audio_int16, 16000))
            timestamps = result.get("timestamps", []) if result else []
            segments = self._timestamps_to_segments(timestamps, probs, duration)
        except Exception as e:
            logger.error(f"FireRedVAD segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        groups = group_segments(
            segments, self.max_group_duration_s, self.chunk_threshold_s
        )

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _timestamps_to_segments(
        self,
        timestamps: List[Tuple[float, float]],
        probs,
        duration: float,
    ) -> List[SpeechSegment]:
        """Convert FireRedVAD (start_s, end_s) tuples to SpeechSegment objects.

        Confidence is the mean frame probability over the segment (10ms frames).
        """
        probs_list: List[float] = []
        try:
            if probs is not None:
                raw = probs.tolist()
                # 1-frame tensors squeeze to a scalar
                probs_list = [float(raw)] if np.isscalar(raw) else [float(p) for p in raw]
        except Exception:
            probs_list = []

        # The package's max_speech splitter can leave remainder fragments
        # shorter than min_speech_duration_ms (the source of its "Unexpected
        # short speech segment" warnings). Drop them — they carry no usable
        # speech and would become degenerate ASR chunks.
        min_dur_s = self.min_speech_duration_ms / 1000.0
        dropped = 0

        segments: List[SpeechSegment] = []
        for start_s, end_s in timestamps:
            start_s = max(0.0, float(start_s))
            end_s = min(float(end_s), duration) if duration > 0 else float(end_s)
            if end_s <= start_s:
                continue
            if (end_s - start_s) < min_dur_s:
                dropped += 1
                continue

            confidence = 1.0
            if probs_list:
                i0 = int(start_s * 1000 / _FRAME_MS)
                i1 = min(int(end_s * 1000 / _FRAME_MS), len(probs_list))
                if i1 > i0:
                    confidence = sum(probs_list[i0:i1]) / (i1 - i0)

            segments.append(SpeechSegment(
                start_sec=start_s,
                end_sec=end_s,
                start_sample=int(start_s * 16000),
                end_sample=int(end_s * 16000),
                confidence=confidence,
            ))
        if dropped:
            logger.debug(
                f"FireRedVAD: dropped {dropped} sub-{self.min_speech_duration_ms}ms "
                f"split-remainder fragment(s)"
            )
        return segments

    # ------------------------------------------------------------------
    # Audio utilities (same conventions as ten.py)
    # ------------------------------------------------------------------

    def _convert_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float audio to int16 scale expected by FireRedVAD fbank."""
        if audio.dtype == np.int16:
            return audio
        if audio.dtype in (np.float32, np.float64):
            audio_clipped = np.clip(audio, -1.0, 1.0)
            return (audio_clipped * 32767).astype(np.int16)
        return audio.astype(np.int16)

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(audio.dtype)
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), 1 / ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "speech_pad_ms": self.speech_pad_ms,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "smooth_window_size": self.smooth_window_size,
            "use_gpu": self.use_gpu,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        with self._lock:
            if self._vad is not None:
                del self._vad
                self._vad = None
                logger.debug("FireRedVAD model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """FireRedVAD operates at 16kHz only; other rates are resampled."""
        return [16000]

    def __repr__(self) -> str:
        return f"FireRedSpeechSegmenter(threshold={self.threshold})"
