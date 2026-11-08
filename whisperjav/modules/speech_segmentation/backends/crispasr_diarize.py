"""
CrispASR diarization methods as speech segmenter backends: ``sortformer`` and ``pyannote``.

1.10.2 (owner W594, W596). Each method decides where there is speech; WhisperJAV's own TEN VAD post-processing
(``backends/ten.py``: shortest speech, the silence merge, the pads, the cut of a long segment, the VAD groups) does
the rest. This is the path the viability test measured (``tools/crispasr_viability.py`` CrispDiarSegmenter;
``docs/plans/design/2026-11-08-a2-diarization-viability-results.md``).

- sortformer: CrispASR's Sortformer (NVIDIA Nemotron-3-Diarization). The speech is the union of the speaker turns it
  returns from the audio (``crispasr.diarize_segments_with_turns``; runs of a speaker's probability above 0.5, every
  10 ms). It runs where CrispASR runs it: the GPU of the install's CrispASR build (owner W596 5: no device handling
  for the speech segmenters). CrispASR keeps the model in one cached context for the process; the binding offers no
  call to release it.
- pyannote: CrispASR's pyannote segmentation-3.0, CPU only, a method CrispASR marks experimental. CrispASR 0.8.50's
  binding returns no frame probabilities, so the scene is handed over as a grid of 10 ms ranges: a range it leaves
  unlabelled is silence (all its frames' silence probability above 0.5; ``crispasr_diarize.cpp``
  score_speaker_for_range).

Neither gives a probability curve, so the frames are 0 or 1, and the cut of a speech segment longer than the
longest speech falls in a bridged pause or, on a tie, at the latest allowed point (the report's section 7). Neither
takes a speech threshold through the binding (CrispASR's own 0.5), so no threshold setting reaches them. The model
files are pinned in ``whisperjav/crispasr/manifest.py`` (SEGMENTER_MODELS) and downloaded on first use.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
import logging
import os
import tempfile
import time

import numpy as np

from ..base import SegmentationResult
from .ten import TenSpeechSegmenter

logger = logging.getLogger("whisperjav")

SAMPLE_RATE = 16000
PYANNOTE_GRID_S = 0.01   # the ranges pyannote is asked to label: one CrispASR centisecond each


def crispasr_threads() -> int:
    """CPU threads for CrispASR: min(8, CPU count), as the CrispASR pipelines use (crispasr/engine.py)."""
    return max(1, min(8, os.cpu_count() or 4))


def union_spans(spans: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Overlapping or touching spans (seconds) joined, sorted. Speaker turns of different speakers overlap."""
    out: List[List[float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def spans_to_flags(spans: Sequence[Tuple[float, float]], n_frames: int, frame_s: float) -> List[int]:
    """1 for each frame of the grid (frame i = [i * frame_s, (i + 1) * frame_s)) whose middle lies inside a span."""
    flags = [0] * n_frames
    for s, e in spans:
        i0 = max(0, int(-(-(s / frame_s - 0.5) // 1)))          # first i with (i + 0.5) * frame_s >= s
        i1 = min(n_frames, int(-(-(e / frame_s - 0.5) // 1)))   # first i with (i + 0.5) * frame_s >= e
        for i in range(i0, i1):
            flags[i] = 1
    return flags


class _CrispDiarizationSegmenter(TenSpeechSegmenter):
    """The shared part: the model file, the audio, TEN's post-processing. A subclass names its METHOD and finds the
    speech spans of one scene."""

    METHOD = ""
    DISPLAY_NAME = ""

    def __init__(self, **kwargs):
        # A threshold is not used by these methods (CrispASR's own 0.5); the factory drops it before this point.
        kwargs.pop("threshold", None)
        super().__init__(**kwargs)
        self.n_threads = crispasr_threads()
        self._model_path = None

    @property
    def name(self) -> str:
        return self.METHOD

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    def _ensure_model(self) -> None:
        """The pinned model file on this PC (downloaded on first use), as a path CrispASR can open."""
        if self._model_path is not None:
            return
        with self._lock:
            if self._model_path is not None:
                return
            try:
                import crispasr
            except ImportError as e:
                raise ImportError(f"The speech segmenter {self.METHOD} is run by CrispASR, which could not be "
                                  f"loaded on this computer: {e}") from e
            from whisperjav.crispasr import install, manifest, native_log
            native_log.quiet_debug_lines(crispasr)
            model = manifest.SEGMENTER_MODELS[self.METHOD]
            path = install.ensure_model(model)
            self._model_path = str(install.stage_model(path, Path(tempfile.gettempdir()), model.sha256,
                                                       keep_name=True))
            logger.debug(f"{self.DISPLAY_NAME}: model {self._model_path}")

    def _speech_spans(self, pcm: np.ndarray, duration: float) -> List[Tuple[float, float]]:
        raise NotImplementedError

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """Find speech in one scene: the method's spans, then TEN's merge, pads, cut and groups."""
        start_time = time.time()
        self._ensure_model()

        audio_data, actual_sr = self._load_audio(audio, sample_rate)
        audio_data = np.asarray(audio_data)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if actual_sr != SAMPLE_RATE:
            audio_data = self._resample_audio(audio_data, actual_sr, SAMPLE_RATE)
        pcm = np.ascontiguousarray(audio_data, dtype=np.float32)
        duration = len(pcm) / SAMPLE_RATE

        try:
            spans = self._speech_spans(pcm, duration) if len(pcm) else []
            frame_s = self.hop_size / SAMPLE_RATE
            n_frames = (len(pcm) + self.hop_size - 1) // self.hop_size   # TEN's own frame count
            flags = spans_to_flags(spans, n_frames, frame_s)
            probs = [float(f) for f in flags]
            raw = self._flags_to_segments(flags, probs, SAMPLE_RATE, duration)
            merged = self._merge_by_silence(raw)
            padded = self._apply_padding(merged, duration)
            segments = self._split_long_segments(padded, probs, frame_s)
        except Exception as e:
            logger.error(f"{self.DISPLAY_NAME} segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=self.name,
                audio_duration_sec=duration,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
                error=str(e) or type(e).__name__,
            )

        groups = self._group_segments(segments)
        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=self.name,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=time.time() - start_time,
        )

    def _get_parameters(self) -> Dict[str, Any]:
        params = super()._get_parameters()
        params.pop("threshold", None)
        params["method"] = self.METHOD
        return params

    def cleanup(self) -> None:
        """Nothing to release: CrispASR keeps the model in its own cached context (no call to free it)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(hop_size={self.hop_size})"


class SortformerSpeechSegmenter(_CrispDiarizationSegmenter):
    """CrispASR Sortformer (NVIDIA Nemotron-3-Diarization): the union of its speaker turns is the speech."""

    METHOD = "sortformer"
    DISPLAY_NAME = "Sortformer (NVIDIA Nemotron)"

    def _speech_spans(self, pcm: np.ndarray, duration: float) -> List[Tuple[float, float]]:
        from crispasr import _binding as b
        ok, turns = b.diarize_segments_with_turns([b.DiarizeSegment(0.0, duration)], pcm,
                                                  method=b.DiarizeMethod.SORTFORMER,
                                                  pyannote_model_path=self._model_path,
                                                  n_threads=self.n_threads)
        if not ok:
            raise RuntimeError("CrispASR's Sortformer reported a failure")
        return union_spans([(t.t0, t.t1) for t in turns])


class PyannoteSpeechSegmenter(_CrispDiarizationSegmenter):
    """CrispASR pyannote segmentation-3.0 (experimental in CrispASR; CPU): the 10 ms ranges it labels are speech."""

    METHOD = "pyannote"
    DISPLAY_NAME = "pyannote 3.0"

    def _speech_spans(self, pcm: np.ndarray, duration: float) -> List[Tuple[float, float]]:
        from crispasr import _binding as b
        n = int(round(duration / PYANNOTE_GRID_S))
        grid = [b.DiarizeSegment(round(i * PYANNOTE_GRID_S, 2), round((i + 1) * PYANNOTE_GRID_S, 2))
                for i in range(n)]
        ok = b.diarize_segments(grid, pcm, method=b.DiarizeMethod.PYANNOTE, pyannote_model_path=self._model_path,
                                n_threads=self.n_threads)
        if not ok:
            raise RuntimeError("CrispASR's pyannote reported a failure")
        return union_spans([(g.t0, g.t1) for g in grid if g.speaker >= 0])
