"""The ASR call of the CrispASR pipelines, behind one contract (route B, owner W176).

The shared chain hands an engine the audio of one VAD group and gets back WhisperJAV's own result
types: segments, words and tokens, with times measured from the start of the audio passed in. It
never sees CrispASR's objects, so a later change of route or of CrispASR's result changes this
module only (route B design 5).

Route B: CrispASR's Python binding, ``crispasr.Session`` (0.8.44). The model is opened once per pass
(owner W202 5a) and closed by ``cleanup()``, which ``BasePipeline.cleanup()`` calls at the end of
the pass. ``crispasr`` is imported only when an engine opens, so importing this module stays light.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError

SAMPLE_RATE = 16000
LANGUAGE = "ja"


@dataclass(frozen=True)
class CrispToken:
    text: str
    start: Optional[float]       # seconds from the start of the audio passed in; None when the model gives none
    end: Optional[float]
    confidence: Optional[float]  # the binding's probability; None when it reports none (its -1.0)


@dataclass(frozen=True)
class CrispWord:
    text: str
    start: Optional[float]  # as CrispToken
    end: Optional[float]
    confidence: float       # the binding's probability; the binding gives 1.0 when it has none, so "none"
                            # and a real 1.0 look the same


@dataclass(frozen=True)
class CrispSegment:
    text: str
    start: float  # seconds from the start of the audio passed in
    end: float
    words: Tuple[CrispWord, ...] = ()
    tokens: Tuple[CrispToken, ...] = ()


class CrispEngine(Protocol):
    """The contract between the shared chain and the ASR call."""

    def open(self) -> None: ...                                         # raises CrispASRSetupError

    def transcribe(self, audio: np.ndarray) -> List[CrispSegment]: ...  # 16 kHz mono float32, one VAD group;
                                                                        # raises CrispASRRunError

    def close(self) -> None: ...                                        # safe to call twice

    def cleanup(self) -> None: ...                                      # close()


def default_threads() -> int:
    """CPU threads for CrispASR: min(8, CPU count). On the processor 8 threads were about 11 % faster
    than 4 (binding checks of 2026-10-24, item 4); the binding's own default is 4."""
    return max(1, min(8, os.cpu_count() or 4))


def segments_from_binding(session_segments: Sequence) -> List[CrispSegment]:
    """The binding's ``SessionSegment`` list as WhisperJAV's types. Nothing is dropped or cleaned here:
    text as given, a missing time stays None, confidence as given (a token's -1.0 becomes None)."""
    out = []
    for seg in session_segments:
        words = tuple(CrispWord(str(w.text), w.start, w.end, float(w.confidence)) for w in (seg.words or ()))
        tokens = tuple(CrispToken(str(t.text), t.start, t.end,
                                  None if t.confidence is None or t.confidence < 0 else float(t.confidence))
                       for t in (getattr(seg, "tokens", None) or ()))
        out.append(CrispSegment(str(seg.text or ""), float(seg.start), float(seg.end), words, tokens))
    return out


class BindingEngine:
    """Route B: one ``crispasr.Session`` for one model."""

    def __init__(self, model_path: Path, backend: str, use_gpu: bool, n_threads: Optional[int] = None,
                 label: str = ""):
        self.model_path = Path(model_path)
        self.backend = backend      # what the opened session must report (manifest ModelFile.backend)
        self.use_gpu = bool(use_gpu)
        self.n_threads = n_threads or default_threads()
        self.label = label or self.model_path.name
        self._session = None

    @property
    def is_open(self) -> bool:
        return self._session is not None

    def open(self) -> None:
        if self._session is not None:
            return
        try:
            import crispasr
        except ImportError as e:
            raise CrispASRSetupError(f"CrispASR could not be loaded on this computer: {e}") from e
        try:
            # The model family is detected from the file; it is checked after opening, not passed.
            session = crispasr.Session(str(self.model_path), n_threads=self.n_threads, use_gpu=self.use_gpu)
        except OSError as e:
            raise CrispASRSetupError(f"CrispASR could not be loaded on this computer: {e}") from e
        except RuntimeError as e:
            raise CrispASRSetupError(f"The model {self.label} could not be opened ({self.model_path}): {e}") from e
        found = getattr(session, "backend", "")
        if found != self.backend:
            try:
                session.close()
            except Exception:
                pass
            raise CrispASRSetupError(f"The model {self.label} could not be opened ({self.model_path}): "
                                     f"CrispASR read it as '{found}', not '{self.backend}'")
        self._session = session

    def transcribe(self, audio: np.ndarray) -> List[CrispSegment]:
        if self._session is None:
            # The binding would pass a closed session's null handle straight to the library.
            raise CrispASRRunError("the CrispASR model is not open")
        try:
            result = self._session.transcribe(audio, sample_rate=SAMPLE_RATE, language=LANGUAGE)
        except (RuntimeError, ValueError) as e:
            raise CrispASRRunError(str(e)) from e
        return segments_from_binding(result)

    def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()

    def cleanup(self) -> None:
        self.close()
