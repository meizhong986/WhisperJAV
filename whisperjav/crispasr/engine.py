"""The ASR call of the CrispASR pipelines, behind one contract (route B, owner W176).

The shared chain hands an engine the audio of one VAD group and gets back WhisperJAV's own result
types: segments, words and tokens, with times measured from the start of the audio passed in. It
never sees CrispASR's objects, so a later change of route or of CrispASR's result changes this
module only (route B design 5).

Route B: CrispASR's Python binding, ``crispasr.Session`` (0.8.50 since 1.10.1; 0.8.44 in 1.10.0). The model is opened once per pass
(owner W202 5a) and closed by ``cleanup()``, which ``BasePipeline.cleanup()`` calls at the end of
the pass. ``crispasr`` is imported only when an engine opens, so importing this module stays light.
"""

import codecs
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from whisperjav.crispasr import native_log
from whisperjav.crispasr.errors import CrispASRRunError, CrispASRSetupError
from whisperjav.modules.model_audio import too_short

SAMPLE_RATE = 16000
LANGUAGE = "ja"
# The fewest tokens a call may write when a pipeline sets a token limit per second of audio (owner W406 i1).
MIN_NEW_TOKENS = 32


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


def _confidence(t) -> Optional[float]:
    return None if t.confidence is None or t.confidence < 0 else float(t.confidence)


def tokens_from_binding(binding_tokens: Sequence) -> Tuple[CrispToken, ...]:
    """The binding's ``SessionToken``s as ``CrispToken``s, with whole characters.

    A byte-level BPE token (Whisper) can split one UTF-8 character over two tokens; the binding's ``text`` then
    shows "\\ufffd" for each half, and ``raw_text`` holds the bytes (its SessionToken docstring). The bytes are
    decoded in order: a token carries the characters that are complete at its end. A character begun in one
    token and finished in the next goes with the next, which starts where the first ended. A token with no
    complete character is dropped. Without ``raw_text`` the token's ``text`` is kept as given."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    out: List[CrispToken] = []
    held_from: Optional[float] = None  # where a character carried over from an earlier token began
    for t in binding_tokens:
        raw = getattr(t, "raw_text", b"") or b""
        if not raw:
            tail = decoder.decode(b"", final=True)
            if tail:
                out.append(CrispToken(tail, held_from, held_from, None))
            decoder.reset()
            held_from = None
            out.append(CrispToken(str(t.text), t.start, t.end, _confidence(t)))
            continue
        text = decoder.decode(bytes(raw))
        pending = bool(decoder.getstate()[0])
        if text:
            start = held_from if held_from is not None else t.start
            out.append(CrispToken(text, start, t.end, _confidence(t)))
            held_from = None
        if pending and held_from is None:
            held_from = t.end if t.end is not None else t.start
    tail = decoder.decode(b"", final=True)
    if tail:
        out.append(CrispToken(tail, held_from, held_from, None))
    return tuple(out)


def segments_from_binding(session_segments: Sequence) -> List[CrispSegment]:
    """The binding's ``SessionSegment`` list as WhisperJAV's types. Nothing is dropped or cleaned here:
    text as given, a missing time stays None, confidence as given (a token's -1.0 becomes None); a token's
    text is rebuilt from its bytes where one character spans two tokens (tokens_from_binding)."""
    out = []
    for seg in session_segments:
        words = tuple(CrispWord(str(w.text), w.start, w.end, float(w.confidence)) for w in (seg.words or ()))
        tokens = tokens_from_binding(getattr(seg, "tokens", None) or ())
        out.append(CrispSegment(str(seg.text or ""), float(seg.start), float(seg.end), words, tokens))
    return out


class BindingEngine:
    """Route B: one ``crispasr.Session`` for one model."""

    def __init__(self, model_path: Path, backend: str, use_gpu: bool, n_threads: Optional[int] = None,
                 label: str = "", tokens_per_second: Optional[float] = None, max_new_tokens: Optional[int] = None,
                 ask: Optional[str] = None, decoding: Optional[Mapping[str, Any]] = None):
        self.model_path = Path(model_path)
        self.backend = backend      # what the opened session must report (manifest ModelFile.backend)
        self.use_gpu = bool(use_gpu)
        self.n_threads = n_threads or default_threads()
        self.label = label or self.model_path.name
        # None: the model's own limit. Otherwise each call may write at most
        # max(MIN_NEW_TOKENS, ceil(tokens_per_second x the audio's seconds)) tokens (EngineRow.tokens_per_second).
        self.tokens_per_second = tokens_per_second
        # A fixed limit for every call (MOSS-CrispASR's max_new_tokens, moss_prompt.py); wins over tokens_per_second.
        self.max_new_tokens = max_new_tokens
        # The whole instruction for the model (Session.set_ask; MOSS-CrispASR, moss_prompt.py), or None for the
        # model's own. Given before every call, so no state of an earlier call can replace it.
        self.ask = ask
        # How the model picks its next token (MOSS-CrispASR: moss_prompt.MossSettings.decoding(); owner W492 M1,
        # W501), set on the session once it is open; None = CrispASR's own (greedy).
        self.decoding = dict(decoding) if decoding else None
        self._session = None

    def token_limit(self, n_samples: int) -> Optional[int]:
        """The token limit for one call on ``n_samples`` of 16 kHz audio, or None for the model's own."""
        if self.max_new_tokens:
            return int(self.max_new_tokens)
        if not self.tokens_per_second:
            return None
        return max(MIN_NEW_TOKENS, math.ceil(self.tokens_per_second * n_samples / SAMPLE_RATE))

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
        native_log.quiet_debug_lines(crispasr)   # its GPU library's DEBUG lines only with --debug (W350)
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
        if self.decoding:
            try:
                self._set_decoding(session, self.decoding)
            except CrispASRSetupError:
                try:
                    session.close()
                except Exception:
                    pass
                raise
        self._session = session

    @staticmethod
    def _set_decoding(session, decoding: Mapping[str, Any]) -> None:
        """Sampling and beam size on the session (CrispASR 0.8.50, #564: honoured by MOSS's decoder; older builds
        accept the calls and ignore them). Beam size above 1 runs beam search, which does not sample."""
        try:
            session.set_beam_size(max(1, int(decoding.get("beam_size") or 1)))
            if decoding.get("do_sample"):
                session.set_temperature(float(decoding["temperature"]), int(decoding.get("seed") or 0))
                session.set_top_k(int(decoding.get("top_k") or 0))
                session.set_do_sample(True)
            else:
                session.set_do_sample(False)
        except (RuntimeError, ValueError, AttributeError) as e:
            raise CrispASRSetupError(f"CrispASR could not take the decoding settings {dict(decoding)}: {e}") from e

    def transcribe(self, audio: np.ndarray) -> List[CrispSegment]:
        if self._session is None:
            # The binding would pass a closed session's null handle straight to the library.
            raise CrispASRRunError("the CrispASR model is not open")
        if too_short(len(audio), SAMPLE_RATE):
            # The shortest audio a model is given (owner W461; modules/model_audio.py): no text under 0.2 s.
            return []
        try:
            limit = self.token_limit(len(audio))
            if limit is not None:
                self._session.set_max_new_tokens(limit)   # per call: the limit follows the VAD group's length
            if self.ask:
                self._session.set_ask(self.ask)
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
