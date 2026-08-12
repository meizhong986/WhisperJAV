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
        speech_start_threshold: Optional[float] = None,
        force_split_mode: str = "dip",
        segmentation_decoder: str = "hysteresis",
        grow_floor: float = 0.05,
        gap_merge_ms: int = 350,
        split_smooth_ms: int = 120,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 300,
        start_pad_ms: Optional[int] = None,  # Asymmetric: pad before speech onset.
        end_pad_ms: Optional[int] = None,    # Asymmetric: pad after speech offset.
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
            neg_threshold: Offset (hysteresis) threshold. When None, derived as
                max(threshold - 0.15, 0.01) matching vendor inference.py.
                Set explicitly to decouple offset from onset — needed when
                threshold < 0.30 where the formula collapses to 0.01.
            speech_start_threshold: Display-timing refinement (v1.9.0, "3a").
                When set, each segment additionally records the time of the
                first frame whose probability reaches this value in
                metadata["speech_start_sec"]. Consumers (vad_only subtitle
                timing) may use it as the display start while the RAW segment
                boundary still feeds the ASR audio slice — decoupling capture
                from timing. If the segment starts at/above the value, the
                refined start equals the raw start; if it never reaches it,
                no metadata key is written (callers fall back to raw start).
                None (default) disables the computation.
            force_split_mode: What happens when a segment exceeds
                max_speech_duration_s.
                "dip" (default, v1.9.0 smart split): split at the lowest-
                probability frame in the last 40% of the segment; triggered
                stays True (contiguous, no re-onset required).
                "chop" (vendor-faithful, pre-i2): end at exactly
                start + max_speech, reset the state machine (triggered=False)
                so speech must re-cross `threshold` to start a new segment.
                Chop prunes non-speech after the split and anchors every
                segment at a real onset — empirically stronger content
                capture on wide-net presets (threshold 0.15), at the cost of
                near-uniform segment lengths in continuous speech.
            segmentation_decoder: Which decoder converts the probability
                stream into segments.
                "hysteresis" (default): the vendor-lineage streaming state
                machine (threshold/neg_threshold/min_silence + force-split).
                "offline" (v1.9.0): two-level offline decoder — segments are
                SEEDED by runs of prob >= threshold, their edges GROW outward
                while prob >= grow_floor (captures attached quiet speech),
                gaps shorter than gap_merge_ms are merged, longer gaps become
                real cuts, and segments over max_speech_duration_s are split
                at minima of the smoothed probability curve. Decides from the
                whole curve at once instead of frame-by-frame.
            grow_floor: ("offline") Low probability floor for edge growth.
                The capture-vs-cut dial: lower keeps more quiet speech AND
                bridges more gaps (fewer cuts); higher trims tails and cuts
                more gaps. Must be <= threshold.
            gap_merge_ms: ("offline") Gaps shorter than this are merged into
                the surrounding segment; gaps this long or longer become
                segment boundaries (the "cut dialogs at pauses" knob).
            split_smooth_ms: ("offline") Smoothing window for the probability
                curve when choosing split points inside overlong segments.
            min_speech_duration_ms: Minimum speech segment duration.
            min_silence_duration_ms: Minimum silence duration to end a segment.
            speech_pad_ms: Symmetric post-hoc padding around each segment
                (overlap-prevented). Used as the fallback when start_pad_ms /
                end_pad_ms are not supplied.
            start_pad_ms: Asymmetric padding before speech onset. None → inherit
                speech_pad_ms.
            end_pad_ms: Asymmetric padding after speech offset. None → inherit
                speech_pad_ms. (JAV: capturing end-of-speech is most critical, so
                the qwen/anime pipeline defaults end_pad > start_pad.)
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
        self.speech_start_threshold = (
            float(speech_start_threshold) if speech_start_threshold is not None else None
        )
        _mode = str(force_split_mode).strip().lower() if force_split_mode else "dip"
        if _mode not in ("dip", "chop"):
            logger.warning(
                "WhisperSeg: unknown force_split_mode %r, using 'dip'", force_split_mode
            )
            _mode = "dip"
        self.force_split_mode = _mode
        _dec = (
            str(segmentation_decoder).strip().lower()
            if segmentation_decoder else "hysteresis"
        )
        if _dec not in ("hysteresis", "offline"):
            logger.warning(
                "WhisperSeg: unknown segmentation_decoder %r, using 'hysteresis'",
                segmentation_decoder,
            )
            _dec = "hysteresis"
        self.segmentation_decoder = _dec
        self.grow_floor = min(max(0.0, float(grow_floor)), self.threshold)
        self.gap_merge_ms = max(0, int(gap_merge_ms))
        self.split_smooth_ms = max(20, int(split_smooth_ms))
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        # Asymmetric padding (v1.9.0): start/end fall back to symmetric
        # speech_pad_ms when unset, preserving behavior for existing callers.
        self.start_pad_ms = int(start_pad_ms) if start_pad_ms is not None else int(speech_pad_ms)
        self.end_pad_ms = int(end_pad_ms) if end_pad_ms is not None else int(speech_pad_ms)
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

        threshold = float(self.threshold)
        if self.neg_threshold is not None:
            neg_threshold = float(self.neg_threshold)
        else:
            neg_threshold = max(threshold - 0.15, 0.01)

        # "3a" display-timing refinement: absolute frame index of the first
        # probability >= speech_start_threshold within the segment's used span,
        # or None when disabled / never reached. Computed here because the
        # probability stream only exists inside this function.
        sst = self.speech_start_threshold

        def _speech_start_frame(start_frame: int, probs_slice: List[float]) -> Optional[int]:
            if sst is None:
                return None
            for k, pv in enumerate(probs_slice):
                if pv >= sst:
                    return start_frame + k
            return None

        # Duration conversions (ms → frames)
        min_speech_frames = max(1, int(self.min_speech_duration_ms / frame_ms))
        min_silence_frames = max(1, int(self.min_silence_duration_ms / frame_ms))
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

            # Force-split at max duration. Two modes (see __init__ docstring):
            #   "chop" — vendor-faithful pre-i2 behavior: exact ceiling, full
            #            state reset, re-onset >= threshold required.
            #   "dip"  — v1.9.0 smart split seeking the local probability
            #            minimum in the last 40% (ChickenRice-inspired),
            #            contiguous continuation.
            if triggered and "start" in current and self.force_split_mode == "chop":
                duration = i - current["start"]
                if duration > max_speech_frames:
                    current["end"] = current["start"] + max_speech_frames
                    if current_probs:
                        valid = current_probs[:max_speech_frames]
                        if valid:
                            current["avg_prob"] = float(np.mean(valid))
                            current["min_prob"] = float(np.min(valid))
                            current["max_prob"] = float(np.max(valid))
                            current["speech_start"] = _speech_start_frame(
                                current["start"], valid
                            )
                    speeches.append(current)
                    current = {}
                    current_probs = []
                    triggered = False
                    temp_end = 0
                    continue

            if triggered and "start" in current and self.force_split_mode == "dip":
                duration = i - current["start"]
                if duration > max_speech_frames:
                    window_start = int(max_speech_frames * 0.6)
                    window_probs = current_probs[window_start:max_speech_frames]
                    if window_probs:
                        min_prob = float(np.min(window_probs))
                        mean_prob = float(np.mean(window_probs))
                        if min_prob < mean_prob * 0.85:
                            best_offset = window_start + int(np.argmin(window_probs))
                        else:
                            best_offset = max_speech_frames
                    else:
                        best_offset = max_speech_frames
                    split_frame = current["start"] + best_offset
                    current["end"] = split_frame
                    if current_probs:
                        valid = current_probs[:best_offset]
                        if valid:
                            current["avg_prob"] = float(np.mean(valid))
                            current["min_prob"] = float(np.min(valid))
                            current["max_prob"] = float(np.max(valid))
                            current["speech_start"] = _speech_start_frame(
                                current["start"], valid
                            )
                    speeches.append(current)

                    current = {"start": split_frame}
                    current_probs = current_probs[best_offset:]
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
                                current["speech_start"] = _speech_start_frame(
                                    current["start"], valid
                                )
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
                    current["speech_start"] = _speech_start_frame(
                        current["start"], current_probs
                    )
                speeches.append(current)

        return self._pad_and_convert(speeches, len(speech_probs), audio_duration_sec)

    def _pad_and_convert(
        self,
        speeches: List[Dict[str, Any]],
        n_frames: int,
        audio_duration_sec: float,
    ) -> List[SpeechSegment]:
        """Shared tail for both decoders: post-hoc asymmetric padding with
        overlap prevention, then frame indices → SpeechSegment conversion."""
        frame_ms = float(self._frame_duration_ms)
        start_pad_frames = max(0, int(self.start_pad_ms / frame_ms))
        end_pad_frames = max(0, int(self.end_pad_ms / frame_ms))

        for idx, seg in enumerate(speeches):
            if idx == 0:
                seg["start"] = max(0, seg["start"] - start_pad_frames)
            else:
                seg["start"] = max(
                    speeches[idx - 1]["end"],
                    seg["start"] - start_pad_frames,
                )
            if idx < len(speeches) - 1:
                seg["end"] = min(
                    speeches[idx + 1]["start"],
                    seg["end"] + end_pad_frames,
                )
            else:
                seg["end"] = min(n_frames, seg["end"] + end_pad_frames)

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
            if seg.get("speech_start") is not None:
                # Refined display start ("3a"): first frame >= speech_start_threshold.
                # Always >= the (padded) raw start; consumers fall back to
                # start_sec when absent.
                metadata["speech_start_sec"] = min(
                    seg["speech_start"] * frame_ms / 1000.0, end_sec
                )
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
    # Offline decoder (v1.9.0) — two-level TEN-shape pipeline
    # ------------------------------------------------------------------

    def _probs_to_segments_offline(
        self,
        speech_probs: np.ndarray,
        audio_duration_sec: float,
    ) -> List[SpeechSegment]:
        """Offline two-level decoder: seed → grow → merge → drop → split.

        Decides from the whole probability curve at once (we are not
        streaming), unlike the ported hysteresis state machine:

        1. SEED: contiguous runs of prob >= threshold anchor segments.
        2. GROW: keep every contiguous run of prob >= grow_floor that
           contains at least one seed — quiet speech attached to confident
           speech rides along; floor-runs with no seed (noise) are dropped.
        3. MERGE: gaps shorter than gap_merge_ms are absorbed; longer gaps
           become real segment boundaries (dialog cuts).
        4. DROP: segments shorter than min_speech_duration_ms are removed.
        5. SPLIT: segments longer than max_speech_duration_s are split at
           local minima of the smoothed probability curve (spacing-
           constrained), with an even-split fallback — no blind chop.

        Padding + SpeechSegment conversion shared with the hysteresis
        decoder (_pad_and_convert). Pipeline shape adapted from the TEN
        backend (ten.py), with exact frame-index bookkeeping.
        """
        if len(speech_probs) == 0:
            return []

        p = np.asarray(speech_probs, dtype=np.float32)
        frame_ms = float(self._frame_duration_ms)
        n = len(p)

        min_speech_frames = max(1, int(self.min_speech_duration_ms / frame_ms))
        gap_merge_frames = max(1, int(self.gap_merge_ms / frame_ms))
        if self.max_speech_duration_s and self.max_speech_duration_s > 0:
            max_speech_frames = int(self.max_speech_duration_s * 1000.0 / frame_ms)
        else:
            max_speech_frames = n

        seed_mask = p >= self.threshold
        if not seed_mask.any():
            return []
        floor_mask = p >= min(self.grow_floor, self.threshold)

        # Contiguous floor-runs, kept only when they contain a seed (steps 1+2)
        runs: List[List[int]] = []
        in_run = False
        run_start = 0
        for i in range(n):
            if floor_mask[i] and not in_run:
                in_run = True
                run_start = i
            elif not floor_mask[i] and in_run:
                in_run = False
                runs.append([run_start, i])
        if in_run:
            runs.append([run_start, n])
        segs = [r for r in runs if seed_mask[r[0]:r[1]].any()]
        if not segs:
            return []

        # Step 3: merge short gaps
        merged: List[List[int]] = [segs[0][:]]
        for r in segs[1:]:
            if r[0] - merged[-1][1] < gap_merge_frames:
                merged[-1][1] = r[1]
            else:
                merged.append(r[:])

        # Step 4: drop micro-segments
        kept = [r for r in merged if r[1] - r[0] >= min_speech_frames]

        # Step 5: split overlong segments at smoothed minima
        bounded: List[List[int]] = []
        for r in kept:
            bounded.extend(self._split_overlong(p, r[0], r[1], max_speech_frames))

        # Build speech dicts with stats + "3a" refined start per final part
        sst = self.speech_start_threshold
        speeches: List[Dict[str, Any]] = []
        for i0, i1 in bounded:
            window = p[i0:i1]
            seg: Dict[str, Any] = {
                "start": i0,
                "end": i1,
                "avg_prob": float(window.mean()),
                "min_prob": float(window.min()),
                "max_prob": float(window.max()),
            }
            if sst is not None:
                hits = np.flatnonzero(window >= sst)
                if len(hits):
                    seg["speech_start"] = i0 + int(hits[0])
            speeches.append(seg)

        return self._pad_and_convert(speeches, n, audio_duration_sec)

    def _split_overlong(
        self,
        p: np.ndarray,
        i0: int,
        i1: int,
        max_speech_frames: int,
    ) -> List[List[int]]:
        """Split [i0, i1) into parts <= max_speech_frames, preferring minima
        of the smoothed probability curve (0.6s sliver guard between cuts);
        even-split fallback bounds any remainder."""
        if i1 - i0 <= max_speech_frames:
            return [[i0, i1]]

        frame_ms = float(self._frame_duration_ms)
        window = max(1, int(self.split_smooth_ms / frame_ms))
        seg_p = p[i0:i1]
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(seg_p, kernel, mode="same")

        # Local minima of the smoothed curve (interior frames only)
        minima = [
            j for j in range(1, len(smoothed) - 1)
            if smoothed[j] <= smoothed[j - 1] and smoothed[j] <= smoothed[j + 1]
        ]

        # Greedy selection: cut at the deepest minimum inside each
        # (spacing_min, ceiling] lookahead from the previous cut. Among
        # near-tied minima (within 0.02 of the deepest) prefer the LATEST
        # position — on flat stretches this stretches parts toward the
        # ceiling instead of cutting early (more ASR context per part).
        # spacing_min is only a sliver guard (0.6s): a genuine dip shortly
        # after the previous cut SHOULD produce a subtitle-sized part.
        spacing_min = max(1, int(600.0 / frame_ms))
        cuts: List[int] = []
        prev = 0
        while i1 - i0 - prev > max_speech_frames:
            cands = [j for j in minima if prev + spacing_min <= j < prev + max_speech_frames]
            if not cands:
                break
            floor_val = min(smoothed[j] for j in cands)
            best = max(j for j in cands if smoothed[j] <= floor_val + 0.02)
            cuts.append(best)
            prev = best

        parts: List[List[int]] = []
        prev = 0
        for c in cuts:
            parts.append([i0 + prev, i0 + c])
            prev = c
        parts.append([i0 + prev, i1])

        # Fallback: even-split any part still over the ceiling
        final: List[List[int]] = []
        for a, b in parts:
            length = b - a
            if length <= max_speech_frames:
                final.append([a, b])
            else:
                n_parts = int(np.ceil(length / max_speech_frames))
                step = length / n_parts
                for k in range(n_parts):
                    fa = a + int(round(k * step))
                    fb = a + int(round((k + 1) * step)) if k < n_parts - 1 else b
                    if fb > fa:
                        final.append([fa, fb])
        return final

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

        if self.segmentation_decoder == "offline":
            segments = self._probs_to_segments_offline(probs, duration_sec)
        else:
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
            "speech_start_threshold": self.speech_start_threshold,
            "force_split_mode": self.force_split_mode,
            "segmentation_decoder": self.segmentation_decoder,
            "grow_floor": self.grow_floor,
            "gap_merge_ms": self.gap_merge_ms,
            "split_smooth_ms": self.split_smooth_ms,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "start_pad_ms": self.start_pad_ms,
            "end_pad_ms": self.end_pad_ms,
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
