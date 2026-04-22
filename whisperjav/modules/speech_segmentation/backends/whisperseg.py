"""
WhisperSeg speech segmentation backend.

Whisper-encoder encoder-decoder VAD exported to ONNX. Frame-level resolution
(20 ms), 30-second input window, trained on ~500 h Japanese ASMR — well-matched
to soft/whispered speech common in JAV content. Zero torch dependency at
runtime; uses onnxruntime + transformers (feature extractor only).

Model hosted on HuggingFace Hub:
    https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx

The state-machine post-processing (hysteresis, duration filters, padding) is
adapted from the vendor's reference inference.py (MIT License). See NOTICES
file at repo root for full attribution.

Installation:
    pip install whisperjav[whisperseg]              # CPU
    pip install whisperjav[whisperseg-gpu]          # CUDA via onnxruntime-gpu

Reference:
    Gu et al., "WhisperSeg: Positive Transfer of the Whisper Speech
    Transformer to Human and Animal Voice Activity Detection" (2023)
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import logging
import multiprocessing
import os
import threading
import time

import numpy as np

from ..base import SpeechSegment, SegmentationResult
from .ten import group_segments  # shared gap/max-duration grouping

logger = logging.getLogger("whisperjav")

# --- Pinned upstream revision (prevents silent upstream changes) ---
_HF_REPO_ID = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
_HF_REVISION = "6ac29e2cbf2f4f8e9b639861766a8639dd666e9c"
_MODEL_FILENAME = "model.onnx"
_METADATA_FILENAME = "model_metadata.json"

# Model architecture constants (match vendor's _metadata.json defaults)
_WHISPER_BASE_MODEL_ID = "openai/whisper-base"
_SAMPLE_RATE = 16000
_DEFAULT_METADATA: Dict[str, Any] = {
    "whisper_model_name": _WHISPER_BASE_MODEL_ID,
    "frame_duration_ms": 20,
    "total_duration_ms": 30000,
}


class WhisperSegSpeechSegmenter:
    """
    WhisperSeg ONNX speech segmentation backend.

    Pipeline:
        1. Chunk audio into 30-second windows (zero-pad final chunk)
        2. Extract 80-ch log-mel features via WhisperFeatureExtractor
        3. ONNX inference → per-frame logits (1500 frames per chunk, 20 ms each)
        4. Sigmoid → per-frame speech probabilities
        5. Silero-compatible state machine (dual-threshold hysteresis,
           min-duration filtering, max-duration force-split)
        6. Post-hoc padding with overlap prevention
        7. Grouping into ≤max_group_duration_s chunks for ASR

    Note: Internally operates at 16 kHz. Non-16 kHz input is resampled.
    Input audio must be mono float32 in [-1, 1]; stereo is averaged to mono.

    Example:
        segmenter = WhisperSegSpeechSegmenter(threshold=0.35)
        result = segmenter.segment(audio_array, sample_rate=16000)
        segmenter.cleanup()
    """

    def __init__(
        self,
        threshold: float = 0.35,
        neg_threshold: Optional[float] = None,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 300,
        max_speech_duration_s: Optional[float] = None,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = None,
        force_cpu: bool = False,
        num_threads: int = 1,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize WhisperSeg segmenter.

        Args:
            threshold: Onset probability threshold [0.0, 1.0]. Default 0.35
                (Silero-v6.2-aligned; vendor default is 0.5).
            neg_threshold: Offset/hysteresis threshold. None = auto-calculate
                as max(threshold - 0.15, 0.01) at segment() time.
            min_speech_duration_ms: Minimum speech segment duration.
            min_silence_duration_ms: Minimum silence duration to end a segment.
            speech_pad_ms: Post-hoc padding applied around each segment
                (overlap-prevented).
            max_speech_duration_s: Force-split segments exceeding this duration.
                None = inherits max_group_duration_s.
            chunk_threshold_s: Gap threshold for post-VAD segment grouping (seconds).
            max_group_duration_s: Maximum duration for a segment group (seconds).
                Default 29.0 (Whisper context limit).
            force_cpu: If True, bypass CUDAExecutionProvider even when available.
            num_threads: CPU threads for onnxruntime. 1 = auto
                (cpu_count // 2 on CPU, passed as-is on GPU).
            model_path: Optional explicit path to a pre-downloaded ONNX file.
                If provided and exists, skips HuggingFace Hub download.
            **kwargs: Absorber for factory-injected params (e.g., version).
        """
        self.threshold = float(threshold)
        self.neg_threshold = float(neg_threshold) if neg_threshold is not None else None
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.force_cpu = bool(force_cpu)
        self.num_threads = int(num_threads)
        self.model_path = model_path

        if chunk_threshold_s is not None:
            self.chunk_threshold_s = float(chunk_threshold_s)
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = float(kwargs["chunk_threshold"])
        else:
            self.chunk_threshold_s = 1.0

        self.max_group_duration_s = (
            float(max_group_duration_s) if max_group_duration_s is not None else 29.0
        )

        if max_speech_duration_s is not None:
            self.max_speech_duration_s = float(max_speech_duration_s)
        else:
            # Inherit from max_group_duration_s (matches Silero-v6.2 pattern)
            self.max_speech_duration_s = self.max_group_duration_s

        # Lazy state — no model load or HF download at __init__
        self._session = None
        self._feature_extractor = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._frame_duration_ms: int = _DEFAULT_METADATA["frame_duration_ms"]
        self._chunk_duration_ms: int = _DEFAULT_METADATA["total_duration_ms"]
        self._chunk_samples: int = int(
            self._chunk_duration_ms * _SAMPLE_RATE / 1000
        )
        self._actual_device: str = "CPU"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Protocol: name, display_name, get_supported_sample_rates
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "whisperseg"

    @property
    def display_name(self) -> str:
        return "WhisperSeg (JA-ASMR)"

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates.

        WhisperSeg operates at 16 kHz only. Other rates are resampled internally.
        """
        return [_SAMPLE_RATE]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _download_model(self) -> Tuple[str, Optional[str]]:
        """Resolve the ONNX model + metadata paths.

        If self.model_path is set and exists, use it directly (and look for a
        metadata sidecar alongside). Otherwise download from HuggingFace Hub
        using the pinned revision.

        Returns:
            Tuple of (model_path, metadata_path or None)
        """
        # User-supplied explicit path
        if self.model_path and os.path.exists(self.model_path):
            sidecar = self.model_path.replace(".onnx", "_metadata.json")
            sidecar_path = sidecar if os.path.exists(sidecar) else None
            logger.info(f"WhisperSeg using local model: {self.model_path}")
            return self.model_path, sidecar_path

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "WhisperSeg requires the huggingface_hub package. "
                "Install with: pip install whisperjav[whisperseg]"
            ) from e

        try:
            model_path = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_MODEL_FILENAME,
                revision=_HF_REVISION,
            )
        except Exception as e:
            raise ImportError(
                f"WhisperSeg failed to download model from HuggingFace Hub "
                f"({_HF_REPO_ID}@{_HF_REVISION[:8]}). Check network connectivity, "
                f"or pre-download and pass model_path. Original error: {e}"
            ) from e

        try:
            metadata_path: Optional[str] = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_METADATA_FILENAME,
                revision=_HF_REVISION,
            )
        except Exception as e:
            # Metadata is optional — fall back to defaults
            logger.debug(
                f"WhisperSeg metadata sidecar not downloaded ({e}); using defaults"
            )
            metadata_path = None

        logger.info(f"WhisperSeg model resolved: {model_path}")
        return model_path, metadata_path

    def _load_metadata(self, metadata_path: Optional[str]) -> Dict[str, Any]:
        """Load metadata sidecar, tolerating missing/corrupt file."""
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    data = json.load(f)
                merged = dict(_DEFAULT_METADATA)
                merged.update(data)
                return merged
            except Exception as e:
                logger.warning(
                    f"WhisperSeg metadata unreadable, using defaults: {e}"
                )
        return dict(_DEFAULT_METADATA)

    def _ensure_model(self) -> None:
        """Thread-safe lazy initialization of ONNX session and feature extractor."""
        if self._session is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._session is not None:
                return

            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError(
                    "WhisperSeg requires onnxruntime. "
                    "Install with: pip install whisperjav[whisperseg] "
                    "(or whisperjav[whisperseg-gpu] for CUDA)"
                ) from e

            try:
                from transformers import WhisperFeatureExtractor
            except ImportError as e:
                raise ImportError(
                    "WhisperSeg requires transformers. "
                    "Install with: pip install whisperjav[whisperseg]"
                ) from e

            model_path, metadata_path = self._download_model()
            self._metadata = self._load_metadata(metadata_path)

            self._frame_duration_ms = int(
                self._metadata.get("frame_duration_ms", 20)
            )
            self._chunk_duration_ms = int(
                self._metadata.get("total_duration_ms", 30000)
            )
            self._chunk_samples = int(
                self._chunk_duration_ms * _SAMPLE_RATE / 1000
            )

            # Build session options
            opts = ort.SessionOptions()

            # Execution providers — CUDA first if available & allowed
            available_providers = ort.get_available_providers()
            use_gpu = (
                not self.force_cpu
                and "CUDAExecutionProvider" in available_providers
            )
            if use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._actual_device = "GPU (CUDA)"
                opts.inter_op_num_threads = self.num_threads
                opts.intra_op_num_threads = self.num_threads
            else:
                providers = ["CPUExecutionProvider"]
                self._actual_device = "CPU"
                if self.num_threads == 1:
                    optimal = max(1, multiprocessing.cpu_count() // 2)
                    opts.inter_op_num_threads = optimal
                    opts.intra_op_num_threads = optimal
                    logger.debug(
                        f"WhisperSeg auto-configured threads: {optimal} "
                        f"(of {multiprocessing.cpu_count()} CPUs)"
                    )
                else:
                    opts.inter_op_num_threads = self.num_threads
                    opts.intra_op_num_threads = self.num_threads

            # Create session
            try:
                self._session = ort.InferenceSession(
                    model_path, providers=providers, sess_options=opts
                )
            except Exception as e:
                logger.error(f"Failed to create WhisperSeg ONNX session: {e}")
                raise

            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]

            # Feature extractor (downloads preprocessor config from HF on first use)
            try:
                self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self._metadata.get(
                        "whisper_model_name", _WHISPER_BASE_MODEL_ID
                    )
                )
            except Exception as e:
                raise ImportError(
                    f"WhisperSeg failed to load WhisperFeatureExtractor for "
                    f"'{self._metadata.get('whisper_model_name')}'. "
                    f"Network required on first run. Original error: {e}"
                ) from e

            logger.info(
                f"WhisperSeg ready: device={self._actual_device}, "
                f"chunk={self._chunk_duration_ms}ms, "
                f"frame={self._frame_duration_ms}ms"
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Run a single 30-second chunk through ONNX. Returns frame probabilities.

        Args:
            audio_chunk: 1-D float32 audio at 16 kHz. Pads with zeros if short,
                truncates if long.

        Returns:
            1-D float32 array of frame probabilities, length frames_per_chunk
            (1500 for the default 30s@20ms configuration).
        """
        # Pad/truncate to exact chunk size
        if len(audio_chunk) < self._chunk_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self._chunk_samples - len(audio_chunk)),
                mode="constant",
            )
        elif len(audio_chunk) > self._chunk_samples:
            audio_chunk = audio_chunk[: self._chunk_samples]

        # WhisperFeatureExtractor returns log-mel spectrogram
        inputs = self._feature_extractor(
            audio_chunk,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
        )

        # ONNX inference
        outputs = self._session.run(
            self._output_names,
            {self._input_name: inputs.input_features},
        )

        # Output is raw logits (NOT probabilities — README claim is wrong;
        # verified against vendor's inference.py line 193)
        frame_logits = outputs[0][0]  # shape: [frames_per_chunk]
        frame_probs = 1.0 / (1.0 + np.exp(-frame_logits))
        return frame_probs.astype(np.float32)

    def _audio_forward(self, audio: np.ndarray) -> np.ndarray:
        """Process full audio as sequential 30 s chunks. No overlap between chunks."""
        if audio.ndim > 1:
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        all_probs: List[np.ndarray] = []
        for i in range(0, len(audio), self._chunk_samples):
            chunk = audio[i : i + self._chunk_samples]
            probs = self._process_chunk(chunk)
            all_probs.append(probs)

        if not all_probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(all_probs)

    # ------------------------------------------------------------------
    # State machine — port of vendor inference.py's get_speech_timestamps
    # ------------------------------------------------------------------

    def _probs_to_segments(
        self,
        speech_probs: np.ndarray,
        audio_duration_sec: float,
    ) -> List[SpeechSegment]:
        """Convert frame probability stream to speech segments.

        Silero-compatible state machine:
        - Onset when prob >= threshold
        - Candidate offset when prob < neg_threshold; confirmed after
          min_silence_duration_ms of sustained silence
        - Force-split at max_speech_duration_s
        - Drop segments shorter than min_speech_duration_ms
        - Post-hoc speech_pad_ms padding with overlap prevention

        Adapted from TransWithAI/Whisper-Vad-EncDec-ASMR-onnx/inference.py (MIT).
        """
        if len(speech_probs) == 0:
            return []

        frame_ms = float(self._frame_duration_ms)

        # Thresholds
        threshold = float(self.threshold)
        neg_threshold = (
            float(self.neg_threshold)
            if self.neg_threshold is not None
            else max(threshold - 0.15, 0.01)
        )

        # Duration conversions (ms → frames)
        min_speech_frames = max(1, int(self.min_speech_duration_ms / frame_ms))
        min_silence_frames = max(1, int(self.min_silence_duration_ms / frame_ms))
        speech_pad_frames = max(0, int(self.speech_pad_ms / frame_ms))
        if self.max_speech_duration_s and self.max_speech_duration_s > 0:
            max_speech_frames = int(self.max_speech_duration_s * 1000.0 / frame_ms)
        else:
            max_speech_frames = len(speech_probs)

        # State-machine pass
        triggered = False
        speeches: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        current_probs: List[float] = []
        temp_end = 0

        for i, p in enumerate(speech_probs):
            prob = float(p)

            if triggered:
                current_probs.append(prob)

            # Onset
            if prob >= threshold and not triggered:
                triggered = True
                current["start"] = i
                current_probs = [prob]
                continue

            # Force-split at max duration
            if triggered and "start" in current:
                duration = i - current["start"]
                if duration > max_speech_frames:
                    current["end"] = current["start"] + max_speech_frames
                    if current_probs:
                        valid = current_probs[: current["end"] - current["start"]]
                        if valid:
                            current["avg_prob"] = float(np.mean(valid))
                            current["min_prob"] = float(np.min(valid))
                            current["max_prob"] = float(np.max(valid))
                    speeches.append(current)
                    current = {}
                    current_probs = []
                    triggered = False
                    temp_end = 0
                    continue

            # Candidate offset with hysteresis
            if prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = i

                if i - temp_end >= min_silence_frames:
                    current["end"] = temp_end
                    if current["end"] - current["start"] >= min_speech_frames:
                        if current_probs:
                            valid = current_probs[: temp_end - current["start"]]
                            if valid:
                                current["avg_prob"] = float(np.mean(valid))
                                current["min_prob"] = float(np.min(valid))
                                current["max_prob"] = float(np.max(valid))
                        speeches.append(current)
                    current = {}
                    current_probs = []
                    triggered = False
                    temp_end = 0
            elif prob >= threshold and temp_end:
                # Speech resumed before silence confirmed — reset temp_end
                temp_end = 0

        # Speech running to end of audio
        if triggered and "start" in current:
            current["end"] = len(speech_probs)
            if current["end"] - current["start"] >= min_speech_frames:
                if current_probs:
                    current["avg_prob"] = float(np.mean(current_probs))
                    current["min_prob"] = float(np.min(current_probs))
                    current["max_prob"] = float(np.max(current_probs))
                speeches.append(current)

        # Post-hoc padding with overlap prevention
        for idx, seg in enumerate(speeches):
            if idx == 0:
                seg["start"] = max(0, seg["start"] - speech_pad_frames)
            else:
                seg["start"] = max(
                    speeches[idx - 1]["end"],
                    seg["start"] - speech_pad_frames,
                )
            if idx < len(speeches) - 1:
                seg["end"] = min(
                    speeches[idx + 1]["start"],
                    seg["end"] + speech_pad_frames,
                )
            else:
                seg["end"] = min(
                    len(speech_probs),
                    seg["end"] + speech_pad_frames,
                )

        # Frame indices → SpeechSegment (both seconds and samples populated)
        results: List[SpeechSegment] = []
        for seg in speeches:
            start_sec = seg["start"] * frame_ms / 1000.0
            end_sec = min(seg["end"] * frame_ms / 1000.0, audio_duration_sec)
            if end_sec <= start_sec:
                continue
            avg_prob = seg.get("avg_prob", 1.0)
            confidence = max(0.0, min(1.0, float(avg_prob)))
            metadata: Dict[str, Any] = {}
            if "min_prob" in seg:
                metadata["min_prob"] = seg["min_prob"]
            if "max_prob" in seg:
                metadata["max_prob"] = seg["max_prob"]
            results.append(
                SpeechSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_sample=int(start_sec * _SAMPLE_RATE),
                    end_sample=int(end_sec * _SAMPLE_RATE),
                    confidence=confidence,
                    metadata=metadata,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public: segment
    # ------------------------------------------------------------------

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = _SAMPLE_RATE,
        **kwargs,
    ) -> SegmentationResult:
        """Detect speech segments.

        Args:
            audio: Audio data as numpy array (float32, mono) or path to file.
            sample_rate: Sample rate of input audio (non-16 kHz is resampled).
            **kwargs: Override parameters (currently unused).

        Returns:
            SegmentationResult. On failure returns an empty result with
            processing time populated.
        """
        start_time = time.time()
        self._ensure_model()

        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration_sec = len(audio_data) / actual_sr if actual_sr > 0 else 0.0

        if actual_sr != _SAMPLE_RATE:
            audio_data = self._resample_audio(audio_data, actual_sr, _SAMPLE_RATE)

        try:
            probs = self._audio_forward(audio_data)
        except Exception as e:
            logger.error(f"WhisperSeg inference failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration_sec,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        segments = self._probs_to_segments(probs, duration_sec)
        groups = group_segments(
            segments,
            max_group_duration_s=self.max_group_duration_s,
            chunk_threshold_s=self.chunk_threshold_s,
        )

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration_sec,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """Return (audio_array, actual_sample_rate). Loads from path if needed."""
        if isinstance(audio, np.ndarray):
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required to load audio files. "
                "Install with: pip install soundfile"
            ) from e

        audio_data, actual_sr = sf.read(str(audio_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data, int(actual_sr)

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample to target_sr. Uses scipy if available, falls back to
        linear interpolation."""
        if orig_sr == target_sr:
            return audio
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

    # ------------------------------------------------------------------
    # Metadata & lifecycle
    # ------------------------------------------------------------------

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current runtime parameters for metadata/observability."""
        return {
            "threshold": self.threshold,
            "neg_threshold": self.neg_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "force_cpu": self.force_cpu,
            "num_threads": self.num_threads,
            "device": self._actual_device,
            "frame_duration_ms": self._frame_duration_ms,
            "chunk_duration_ms": self._chunk_duration_ms,
        }

    def cleanup(self) -> None:
        """Release ONNX session and feature extractor references.

        Note: intentionally does NOT call CUDA release APIs or torch.cuda
        operations. Follows the same precedent as the Silero backend (issue #82):
        mixing explicit CUDA cleanup with subprocess exit on Windows can cause
        STATUS_STACK_BUFFER_OVERRUN. OS reclaims resources on process exit.
        """
        with self._lock:
            if self._session is not None:
                self._session = None
                self._feature_extractor = None
                self._input_name = None
                self._output_names = None
                logger.debug("WhisperSeg resources released")

    def __repr__(self) -> str:
        return (
            f"WhisperSegSpeechSegmenter("
            f"threshold={self.threshold}, "
            f"device={self._actual_device})"
        )
