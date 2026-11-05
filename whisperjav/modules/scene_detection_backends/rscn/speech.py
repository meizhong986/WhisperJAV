"""R-SCN's speech measure, behind a contract so its backend can be swapped (owner W451 C3, W450 2f).

Audio in, speech stretches with probabilities out. The scene detector uses them only to keep cuts out of speech, to
confirm black-picture breaks and for the speech marks; never to remove audio. The adapter reuses the code of
WhisperJAV's speech segmenter backends (SpeechSegmenterFactory), run here by the scene detector on the original audio;
it never uses the pipeline's speech segmenter or its results, which come later, scene by scene (owner W442).
Design: docs/plans/design/2026-11-02-rscn-design.md section 8.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from whisperjav.utils.logger import logger

RATE = 16000  # the speech measure reads 16 kHz mono float32


@dataclass(frozen=True)
class SpeechStretch:
    start_s: float
    end_s: float
    probability: float = 1.0


@dataclass
class SpeechMeasurement:
    stretches: List[SpeechStretch]
    backend: str
    settings: Dict[str, Any] = field(default_factory=dict)
    failed_pieces: List[Tuple[float, float, str]] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "settings": self.settings, "stretches": len(self.stretches),
                "speech_s": round(sum(x.end_s - x.start_s for x in self.stretches), 3),
                "failed_pieces": [{"start_s": round(a, 3), "end_s": round(b, 3), "error": e}
                                  for a, b, e in self.failed_pieces],
                "seconds": round(self.seconds, 3)}


@runtime_checkable
class SpeechMeasure(Protocol):
    """The contract. ``measure`` takes the whole file's audio (16 kHz mono float32) and returns stretches in
    seconds from its start, sorted and not overlapping."""

    @property
    def name(self) -> str: ...

    def measure(self, audio_16k: np.ndarray) -> SpeechMeasurement: ...

    def close(self) -> None: ...


class SpeechMeasureError(RuntimeError):
    """The speech measure cannot be built (its backend is missing or fails to load)."""


class SegmenterSpeechMeasure:
    """Any WhisperJAV speech segmenter backend as a speech measure, run in pieces of ``piece_s`` seconds
    (a few minutes, so a long film is not given to a VAD at once; W429 1c). Uses the backend's speech segments
    and their confidence, never its groups."""

    def __init__(self, backend: str, piece_s: float = 180.0, params: Optional[Dict[str, Any]] = None,
                 factory: Any = None, overlap_s: float = 0.0):
        self.backend = backend
        self.piece_s = float(piece_s)
        # Each piece reads overlap_s more audio past its end; the stretches are merged, so speech across a piece's
        # edge is not cut there by the VAD's own rules at the end of its input.
        self.overlap_s = max(0.0, float(overlap_s))
        self.params = dict(params or {})
        self._factory = factory
        self._segmenter = None

    @property
    def name(self) -> str:
        return self.backend

    def _open(self):
        if self._segmenter is None:
            factory = self._factory
            if factory is None:
                from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory as factory
            try:
                self._segmenter = factory.create(self.backend, **self.params)
            except Exception as e:  # ImportError, a missing model, a bad setting
                raise SpeechMeasureError(f"the speech measure '{self.backend}' cannot be used: {e}") from e
        return self._segmenter

    def measure(self, audio_16k: np.ndarray) -> SpeechMeasurement:
        started = time.time()
        segmenter = self._open()
        audio = np.asarray(audio_16k, dtype=np.float32)
        piece = max(1, int(round(self.piece_s * RATE)))
        extra = int(round(self.overlap_s * RATE))
        stretches: List[SpeechStretch] = []
        failed: List[Tuple[float, float, str]] = []
        for first in range(0, len(audio), piece):
            chunk = audio[first:first + piece + extra]
            offset = first / RATE
            end = offset + len(chunk) / RATE
            if len(chunk) < RATE // 10:
                continue
            try:
                result = segmenter.segment(chunk, sample_rate=RATE)
            except Exception as e:
                failed.append((offset, end, str(e) or type(e).__name__))
                continue
            error = getattr(result, "error", None)
            if error:
                failed.append((offset, end, str(error)))
                continue
            for seg in getattr(result, "segments", []) or []:
                a = offset + max(0.0, float(seg.start_sec))
                b = offset + min(len(chunk) / RATE, float(seg.end_sec))
                if b > a:
                    stretches.append(SpeechStretch(a, b, float(getattr(seg, "confidence", 1.0) or 0.0)))
        if failed:
            logger.warning(f"R-SCN: the speech measure ({self.backend}) failed on {len(failed)} piece(s) of the audio; "
                           f"cuts there are placed by the sound level only.")
        settings = {"backend": self.backend, "piece_s": self.piece_s, "overlap_s": self.overlap_s, **self.params}
        return SpeechMeasurement(merge(stretches), self.backend, settings, failed, time.time() - started)

    def close(self) -> None:
        if self._segmenter is not None:
            try:
                self._segmenter.cleanup()
            except Exception:  # best effort
                pass
            self._segmenter = None


def merge(stretches: List[SpeechStretch], gap_s: float = 0.0) -> List[SpeechStretch]:
    """Sorted, without overlaps: stretches that touch or overlap become one (probability: duration-weighted mean)."""
    out: List[SpeechStretch] = []
    for x in sorted(stretches, key=lambda v: (v.start_s, v.end_s)):
        if out and x.start_s <= out[-1].end_s + gap_s:
            last = out[-1]
            da, db = last.end_s - last.start_s, x.end_s - x.start_s
            p = (last.probability * da + x.probability * db) / max(da + db, 1e-9)
            out[-1] = SpeechStretch(last.start_s, max(last.end_s, x.end_s), p)
        else:
            out.append(x)
    return out


def create_speech_measure(name: str, piece_s: float = 180.0, params: Optional[Dict[str, Any]] = None,
                          overlap_s: float = 0.0) -> Optional[SpeechMeasure]:
    """The speech measure named in the settings: ``none`` gives None (no measure); any other name is a WhisperJAV
    speech segmenter backend (first default firered-vad)."""
    key = str(name or "").strip().lower()
    if key in ("", "none"):
        return None
    return SegmenterSpeechMeasure(key, piece_s, params, overlap_s=overlap_s)


def known_backend(name: str) -> bool:
    """True when ``name`` is ``none`` or a speech segmenter backend name the factory knows (no import of a model)."""
    key = str(name or "").strip().lower()
    if key == "none":
        return True
    try:
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
        names = set(SpeechSegmenterFactory.list_backends())
    except Exception:
        return True  # cannot check here; the build of the measure reports it
    base = key.split("-")[0]
    return key in names or (key.startswith(("silero", "nemo", "whisper-vad")) and base in {n.split("-")[0] for n in names})
