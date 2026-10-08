"""
TEN Framework VAD speech segmentation backend.

Requires ten-vad package to be installed.
Install: pip install -U git+https://github.com/TEN-framework/ten-vad.git

See: https://github.com/TEN-framework/ten-vad

v1.8.10-hf2 changes:
- Added max_speech_duration_s enforcement (splits long segments)
- Implemented min_silence_duration_ms merging (was accepted but unused)
- Added _split_long_segments at probability minima
- Thread-safe model loading
- Fixed incomplete final frame handling
- Refactored pipeline: detect → merge → pad → split → group

v1.9.4 changes (longest-segment split):
- Detection no longer force-ends long regions (the merge step re-joined those cuts)
- Split cuts at the lowest smoothed probability, timed on the frame grid of the
  whole audio (bridged pauses and pads were left out of the old time mapping)
- Every piece is at most max_speech_duration_s long
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging
import threading

import numpy as np

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


def group_segments(
    segments: List[SpeechSegment],
    max_group_duration_s: float = 29.0,
    chunk_threshold_s: float = 1.0,
) -> List[List[SpeechSegment]]:
    """Group speech segments by time gaps and max duration.

    Standalone function for re-grouping without re-running VAD.
    Used by adaptive step-down to create Tier 1 (30s) and Tier 2 (8s)
    groups from the same raw segments.

    Args:
        segments: List of SpeechSegment objects from VAD.
        max_group_duration_s: Maximum duration for a group in seconds.
        chunk_threshold_s: Gap threshold for splitting groups in seconds.

    Returns:
        List of groups, where each group is a list of SpeechSegment objects.
    """
    if not segments:
        return []

    groups: List[List[SpeechSegment]] = [[]]

    for i, segment in enumerate(segments):
        if i > 0:
            prev_end = segments[i - 1].end_sec
            gap = segment.start_sec - prev_end

            # Check if adding this segment would exceed max group duration
            would_exceed_max = False
            if groups[-1]:
                group_start = groups[-1][0].start_sec
                potential_duration = segment.end_sec - group_start
                would_exceed_max = potential_duration > max_group_duration_s

            # Start new group if gap too large OR would exceed max duration
            if gap > chunk_threshold_s or would_exceed_max:
                groups.append([])

        groups[-1].append(segment)

    return groups


class TenSpeechSegmenter:
    """
    TEN Framework VAD speech segmentation backend.

    Uses TEN's lightweight VAD model for fast speech detection.
    Processes audio frame-by-frame at 16kHz with configurable hop size.
    Suitable for real-time and streaming applications.

    Pipeline:
        1. Frame-by-frame flag/probability extraction
        2. Raw segment detection
        3. Merge segments with gaps < min_silence_duration_ms
        4. Apply padding (start_pad_ms, end_pad_ms)
        5. Split segments > max_speech_duration_s at the lowest probability
        6. Group into Whisper-compatible chunks

    Note: TEN VAD requires int16 audio input and operates at 16kHz only.

    Example:
        segmenter = TenSpeechSegmenter(threshold=0.26)
        result = segmenter.segment(audio_path)
    """

    def __init__(
        self,
        threshold: float = 0.26,
        hop_size: int = 256,
        min_speech_duration_ms: int = 81,
        min_silence_duration_ms: int = 100,
        max_speech_duration_s: float = 10.0,
        chunk_threshold_s: Optional[float] = 1.0,
        max_group_duration_s: Optional[float] = None,
        start_pad_ms: int = 50,
        end_pad_ms: int = 150,
        **kwargs
    ):
        """
        Initialize TEN VAD segmenter.

        Args:
            threshold: Speech probability threshold [0.0, 1.0]. Lower values
                detect more speech (more sensitive). Default 0.26.
            hop_size: Frame size in samples (160 or 256 recommended)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence gap to keep segments
                separate. Gaps smaller than this are merged.
            max_speech_duration_s: Maximum duration for a single speech segment
                (pads included). Longer segments are cut, after padding, at the
                lowest speech probability; every piece is at most this long.
                0 disables the split.
            chunk_threshold_s: Gap threshold for segment grouping (seconds)
            max_group_duration_s: Maximum duration for a segment group (seconds).
                Groups are split if adding a segment would exceed this limit.
                Default 29s to stay within Whisper's 30s context window.
            start_pad_ms: Milliseconds to pad before segment start (shifts start earlier).
                Default 50: catches whispered/soft speech onsets that TEN's frame may clip.
            end_pad_ms: Milliseconds to pad after segment end (shifts end later).
                Default 150: catches trailing Japanese particles (ね, よ, わ).
            **kwargs: Additional parameters for backward compatibility
                - chunk_threshold: Legacy alias for chunk_threshold_s
        """
        self.threshold = float(threshold)
        self.hop_size = int(hop_size)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.max_speech_duration_s = float(max_speech_duration_s)
        self.start_pad_ms = int(start_pad_ms)
        self.end_pad_ms = int(end_pad_ms)

        # Handle backward compatibility: chunk_threshold (old) -> chunk_threshold_s (new)
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = float(chunk_threshold_s)
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = float(kwargs["chunk_threshold"])
        else:
            self.chunk_threshold_s = 1.0  # Default for tighter grouping

        # Maximum group duration - prevents groups from exceeding Whisper's context window
        self.max_group_duration_s = float(max_group_duration_s) if max_group_duration_s is not None else 29.0

        # Lazy-loaded model with thread lock
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "ten"

    @property
    def display_name(self) -> str:
        return "TEN VAD"

    def _ensure_model(self) -> None:
        """Load TEN VAD model if not already loaded (thread-safe)."""
        if self._model is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            try:
                from ten_vad import TenVad
            except ImportError:
                raise ImportError(
                    "TEN VAD requires ten-vad package. Install with:\n"
                    "pip install -U git+https://github.com/TEN-framework/ten-vad.git"
                )

            logger.debug(f"Loading TEN VAD model (hop_size={self.hop_size}, threshold={self.threshold})")
            try:
                self._model = TenVad(hop_size=self.hop_size, threshold=self.threshold)
                logger.debug("TEN VAD model loaded")
            except Exception as e:
                logger.error(f"Failed to load TEN VAD model: {e}", exc_info=True)
                raise

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments using TEN VAD.

        Pipeline: detect → merge → pad → split → group

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Override parameters

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        self._ensure_model()

        # Load and prepare audio
        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        duration = len(audio_data) / actual_sr

        # TEN VAD requires 16kHz - resample if needed
        if actual_sr != 16000:
            audio_data = self._resample_audio(audio_data, actual_sr, 16000)
            actual_sr = 16000

        # Convert to int16 (TEN VAD requirement)
        audio_int16 = self._convert_to_int16(audio_data)

        try:
            # 1. Extract frame-level flags and probabilities
            flags = []
            probs = []
            n_samples = len(audio_int16)
            for i in range(0, n_samples, self.hop_size):
                frame = audio_int16[i:i + self.hop_size]
                # Pad incomplete final frame with zeros
                if len(frame) < self.hop_size:
                    frame = np.pad(frame, (0, self.hop_size - len(frame)), mode='constant')
                self._model.process(frame)
                flags.append(self._model.out_flags.value)
                probs.append(self._model.out_probability.value)

            # 2. Detect raw segments
            raw_segments = self._flags_to_segments(flags, probs, actual_sr, duration)

            # 3. Merge segments with small gaps
            merged = self._merge_by_silence(raw_segments)

            # 4. Apply padding
            padded = self._apply_padding(merged, duration)

            # 5. Split segments that exceed max_speech_duration_s
            segments = self._split_long_segments(padded, probs, self.hop_size / actual_sr)

        except Exception as e:
            logger.error(f"TEN VAD segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        # 6. Group segments
        groups = self._group_segments(segments)

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _flags_to_segments(
        self,
        flags: List[int],
        probs: List[float],
        sample_rate: int,
        audio_duration: float
    ) -> List[Dict[str, Any]]:
        """Convert frame-level speech flags to raw speech segments.

        Returns list of dicts with 'start', 'end', 'probs' keys.
        Long regions are not cut here: a cut with no gap would be joined again
        by _merge_by_silence. max_speech_duration_s is enforced once, in
        _split_long_segments. No padding is applied here — that happens in
        _apply_padding.
        """
        raw_segments: List[Dict[str, Any]] = []
        frame_duration = self.hop_size / sample_rate

        in_speech = False
        speech_start = 0.0
        speech_probs: List[float] = []

        for i, flag in enumerate(flags):
            time_sec = i * frame_duration

            if flag == 1 and not in_speech:
                # Speech started
                in_speech = True
                speech_start = time_sec
                speech_probs = [probs[i]]

            elif flag == 1 and in_speech:
                speech_probs.append(probs[i])

            elif flag == 0 and in_speech:
                # Speech ended
                in_speech = False
                speech_end = time_sec

                duration_ms = (speech_end - speech_start) * 1000
                if duration_ms >= self.min_speech_duration_ms:
                    raw_segments.append({
                        'start': speech_start,
                        'end': speech_end,
                        'probs': speech_probs.copy(),
                    })
                speech_probs = []

        # Handle speech extending to end of audio
        if in_speech:
            speech_end = min(len(flags) * frame_duration, audio_duration)
            duration_ms = (speech_end - speech_start) * 1000
            if duration_ms >= self.min_speech_duration_ms:
                raw_segments.append({
                    'start': speech_start,
                    'end': speech_end,
                    'probs': speech_probs,
                })

        return raw_segments

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_by_silence(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge segments where gap is smaller than min_silence_duration_ms.

        This prevents micro-fragmentation in rapid speech with tiny pauses.
        Previously min_silence_duration_ms was accepted but not implemented.
        """
        if not segments or self.min_silence_duration_ms <= 0:
            return segments

        min_gap_sec = self.min_silence_duration_ms / 1000.0
        merged: List[Dict[str, Any]] = []

        for seg in segments:
            if not merged:
                merged.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'probs': seg['probs'].copy(),
                })
            else:
                gap = seg['start'] - merged[-1]['end']
                if gap <= min_gap_sec:
                    # Merge: extend end, concatenate probs
                    merged[-1]['end'] = seg['end']
                    merged[-1]['probs'].extend(seg['probs'])
                else:
                    merged.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'probs': seg['probs'].copy(),
                    })

        return merged

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def _apply_padding(
        self,
        segments: List[Dict[str, Any]],
        audio_duration: float
    ) -> List[Dict[str, Any]]:
        """Apply start_pad_ms and end_pad_ms to shift segment boundaries.

        - Start is shifted earlier by start_pad_ms
        - End is shifted later by end_pad_ms
        - Overlap between consecutive segments is prevented
        """
        start_pad_sec = self.start_pad_ms / 1000.0
        end_pad_sec = self.end_pad_ms / 1000.0

        padded: List[Dict[str, Any]] = []

        for i, seg in enumerate(segments):
            padded_start = max(0.0, seg['start'] - start_pad_sec)
            padded_end = min(audio_duration, seg['end'] + end_pad_sec)

            # Prevent overlap with previous padded segment
            if i > 0 and padded:
                prev_end = padded[-1]['end']
                if padded_start < prev_end:
                    padded_start = prev_end

            # Ensure segment is still valid after adjustments
            if padded_end > padded_start:
                padded.append({
                    'start': padded_start,
                    'end': padded_end,
                    'probs': seg['probs'],
                    'raw_start': seg['start'],
                    'raw_end': seg['end'],
                })

        return padded

    # ------------------------------------------------------------------
    # Split long segments
    # ------------------------------------------------------------------

    # Smoothing span (seconds) of the probability curve used to choose cuts.
    # Short enough that a 100-200 ms pause between words still shows as a dip.
    SPLIT_SMOOTH_S = 0.1

    def _split_long_segments(
        self,
        segments: List[Dict[str, Any]],
        probs: List[float],
        frame_duration: float,
    ) -> List[SpeechSegment]:
        """Cut segments longer than max_speech_duration_s into pieces of at most that length.

        A segment of length L becomes the fewest pieces that fit,
        ceil(L / limit). Each cut goes to the lowest smoothed speech
        probability among the places that still let the rest fit into the
        pieces left, and no piece is shorter than a quarter of the limit.
        Times come from the frame grid of the whole audio (frame i starts at
        i * frame_duration), so pads and bridged pauses inside a segment are
        timed correctly; a bridged pause is usually where the cut lands.
        Pieces touch, so no audio is dropped.
        """
        if not segments or self.max_speech_duration_s <= 0:
            return self._to_speech_segments(segments)

        max_dur = self.max_speech_duration_s
        probs_arr = np.asarray(probs, dtype=np.float64)
        window = max(1, int(round(self.SPLIT_SMOOTH_S / frame_duration)))
        smoothed = np.convolve(probs_arr, np.ones(window) / window, mode='same') if len(probs_arr) else probs_arr

        final: List[Dict[str, Any]] = []
        for seg in segments:
            if seg['end'] - seg['start'] <= max_dur:
                final.append(seg)
                continue
            cuts = self._cut_points(seg['start'], seg['end'], smoothed, frame_duration, max_dur)
            edges = [seg['start']] + cuts + [seg['end']]
            for piece_start, piece_end in zip(edges, edges[1:]):
                final.append(self._make_piece(seg, piece_start, piece_end, probs_arr, frame_duration))

        return self._to_speech_segments(final)

    @staticmethod
    def _cut_points(
        start: float,
        end: float,
        smoothed: np.ndarray,
        frame_duration: float,
        max_dur: float,
    ) -> List[float]:
        """Cut times for one segment: ceil(length / max_dur) pieces, each between
        max_dur / 4 and max_dur long, each cut at the lowest probability allowed."""
        n_pieces = int(np.ceil((end - start) / max_dur - 1e-9))
        cuts: List[float] = []
        last = start
        for k in range(1, n_pieces):
            pieces_after = n_pieces - k
            lo = max(last + max_dur / 4, end - pieces_after * max_dur)
            hi = min(last + max_dur, end - max_dur / 4)
            i0 = int(np.ceil(lo / frame_duration))
            i1 = min(int(np.floor(hi / frame_duration)), len(smoothed) - 1)
            if i1 >= i0:
                window = smoothed[i0:i1 + 1]
                # Lowest point; on a tie the later one (fewer, longer pieces first).
                idx = i0 + len(window) - 1 - int(np.argmin(window[::-1]))
                cut = min(max(idx * frame_duration, lo), hi)
            else:
                cut = hi  # no frame inside the window: cut at its end
            cuts.append(cut)
            last = cut
        return cuts

    @staticmethod
    def _make_piece(
        parent: Dict[str, Any],
        start_time: float,
        end_time: float,
        probs_arr: np.ndarray,
        frame_duration: float,
    ) -> Dict[str, Any]:
        """One piece of a cut segment, with the probabilities of its own frames."""
        i0 = int(start_time / frame_duration)
        i1 = int(np.ceil(end_time / frame_duration))
        return {
            'start': start_time,
            'end': end_time,
            'probs': probs_arr[i0:i1].tolist(),
            'raw_start': max(parent.get('raw_start', start_time), start_time),
            'raw_end': min(parent.get('raw_end', end_time), end_time),
        }

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_speech_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[SpeechSegment]:
        """Convert internal dict list to SpeechSegment objects."""
        result: List[SpeechSegment] = []
        for seg in segments:
            seg_probs = seg.get('probs', [])
            avg_confidence = sum(seg_probs) / len(seg_probs) if seg_probs else 1.0

            result.append(SpeechSegment(
                start_sec=seg['start'],
                end_sec=seg['end'],
                start_sample=int(seg['start'] * 16000),
                end_sample=int(seg['end'] * 16000),
                confidence=avg_confidence,
                metadata={
                    'raw_start': seg.get('raw_start', seg['start']),
                    'raw_end': seg.get('raw_end', seg['end']),
                }
            ))
        return result

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_segments(
        self,
        segments: List[SpeechSegment]
    ) -> List[List[SpeechSegment]]:
        """Group segments based on time gaps.

        Delegates to the standalone group_segments() function.
        """
        return group_segments(segments, self.max_group_duration_s, self.chunk_threshold_s)

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _convert_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 for TEN VAD."""
        if audio.dtype == np.int16:
            return audio

        # Normalize and convert
        if audio.dtype in (np.float32, np.float64):
            # Clamp to [-1, 1] and scale to int16 range
            audio_clipped = np.clip(audio, -1.0, 1.0)
            return (audio_clipped * 32767).astype(np.int16)
        else:
            # Assume it's already in int-like range
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
            # Fallback: simple linear interpolation
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(audio), 1/ratio)
            indices = np.clip(indices, 0, len(audio) - 1).astype(int)
            return audio[indices]

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
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
            "hop_size": self.hop_size,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "start_pad_ms": self.start_pad_ms,
            "end_pad_ms": self.end_pad_ms,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                logger.debug("TEN VAD model resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates.

        Note: TEN VAD only operates at 16kHz internally. Other sample rates
        will be automatically resampled to 16kHz before processing.
        """
        return [16000]

    def __repr__(self) -> str:
        return f"TenSpeechSegmenter(threshold={self.threshold})"
