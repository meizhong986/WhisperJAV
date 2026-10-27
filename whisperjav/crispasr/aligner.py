"""The CrispASR aligner: when each character of a known text is spoken (item 12, C1; 1.10.0).

One component for every user of it (design note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md,
section 4.2): ChronosJAV's aligner slot (whisperjav/modules/subtitle_pipeline/aligners/crispasr.py) and,
from step 4, Cohere-CrispASR. It hands CrispASR's ``align_words`` the audio of one VAD group and the text an
ASR model wrote for it, and returns WhisperJAV's own ``TimedUnit`` list; callers never see CrispASR's objects.

What the binding gives (probe P1, 2026-10-27, design note section 14.2): for Japanese one unit per character,
with a mark that follows it attached ("た。"); wav2vec2's times are 20 ms spikes at each character's onset,
Qwen3-ForcedAligner's are spans. The library picks the aligner family from the file's name, so the file is
opened under its own name (install.stage_model keep_name=True). It runs where the CrispASR build runs; the
call has no device argument, so ``--device cpu`` cannot hold it on the CPU.

``crispasr`` is imported only in ``load()``, so importing this module stays light.
"""

import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from whisperjav.crispasr import install
from whisperjav.crispasr.engine import SAMPLE_RATE, default_threads
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import ALIGNER_WARNINGS, ModelFile, aligner_model_for
from whisperjav.utils.logger import logger

# Text approved by the owner as drafted (W268); design note section 9, Q9.
GPU_DESPITE_CPU_TEXT = ("{label}: the {model} aligner still asks for the GPU with --device cpu; CrispASR cannot hold "
                        "it on the CPU.")
NO_ALIGN_TEXT = ("This CrispASR ({version}) has no forced alignment (align_words); run the WhisperJAV "
                 "installer again.")


@dataclass(frozen=True)
class TimedUnit:
    text: str     # one character as the aligner timed it, with the marks and spaces that go with it
    start: float  # seconds from the start of the audio passed in
    end: float


def _compact(text: str) -> str:
    return "".join(text.split())


def _is_opening(ch: str) -> bool:
    return unicodedata.category(ch) in ("Ps", "Pi")


def _skippable(ch: str) -> bool:
    """A character the aligner may leave out: a space, a punctuation mark or a symbol (〜, ♪)."""
    return ch.isspace() or unicodedata.category(ch)[0] in "PS"


def units_from_binding(text: str, aligned: Sequence, audio_s: Optional[float]) -> List[TimedUnit]:
    """The binding's ``AlignedWord`` list as ``TimedUnit``s covering every character of ``text``.

    - The units are matched to the text character by character, in order; the text's own characters are
      kept. A mark or space of the text that no unit carries goes with the unit before it.
    - An opening bracket or quote at the end of a unit goes with the unit after it, where it belongs.
    - If the units do not spell the text (marks and spaces aside), nothing is guessed: ``[]`` (no alignment).
    - The last unit ends at the end of the audio passed in: CTC aligners give each character's onset, not
      where speech ends (rule T2: the end stays the VAD group's end). Inner units keep their own times, so
      the pauses between them stay visible.
    """
    if not text or not text.strip() or not aligned:
        return []
    pieces: List[List] = []  # [text, start, end]
    pos = 0
    for w in aligned:
        body = _compact(str(w.text))
        if not body:
            continue
        begin, first = pos, None
        for ch in body:
            while pos < len(text) and text[pos] != ch:
                if not _skippable(text[pos]):
                    return []
                pos += 1
            if pos >= len(text):
                return []
            first = pos if first is None else first
            pos += 1
        lead, piece = text[begin:first], text[first:pos]
        if lead and pieces:
            pieces[-1][0] += lead
        else:
            piece = lead + piece
        start = float(w.start)
        pieces.append([piece, start, max(start, float(w.end))])
    rest = text[pos:]
    if not pieces or any(not _skippable(ch) for ch in rest):
        return []
    pieces[-1][0] += rest
    for i in range(len(pieces) - 1):
        piece = pieces[i][0]
        cut = len(piece)
        while cut > 0 and _is_opening(piece[cut - 1]):
            cut -= 1
        if 0 < cut < len(piece):
            pieces[i][0], pieces[i + 1][0] = piece[:cut], piece[cut:] + pieces[i + 1][0]
    if audio_s:
        pieces[-1][2] = max(pieces[-1][1], float(audio_s))
    return [TimedUnit(t, s, e) for t, s, e in pieces]


class CrispAligner:
    """One aligner model, loaded once and reused until ``unload()``."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 temp_dir: Optional[Path] = None, n_threads: Optional[int] = None, label: str = "WhisperJAV"):
        self.model: ModelFile = aligner_model_for(model_name)
        self.device = device
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "whisperjav"
        self.n_threads = n_threads or default_threads()
        self.label = label
        self._crispasr = None
        self._path: Optional[Path] = None

    @property
    def is_loaded(self) -> bool:
        return self._crispasr is not None

    def load(self) -> None:
        """CrispASR, the device and the model file. Raises CrispASRSetupError with the reason."""
        if self._crispasr is not None:
            return
        runtime = install.runtime_status(self.device)
        try:
            import crispasr
        except (ImportError, OSError) as e:
            raise CrispASRSetupError(install.NOT_LOADED_TEXT.format(reason=e)) from e
        if not hasattr(crispasr, "align_words"):
            raise CrispASRSetupError(NO_ALIGN_TEXT.format(version=runtime.version))
        if self._path is None or not self._path.is_file():
            path = install.ensure_model(self.model)
            self._path = install.stage_model(path, self.temp_dir, self.model.sha256, keep_name=True)
        warning = ALIGNER_WARNINGS.get(self.model.name)
        if warning:
            logger.warning(f"{self.label}: {warning}")
        if (self.device or "").strip().lower() == "cpu" and self._gpu_on_its_own():
            logger.warning(GPU_DESPITE_CPU_TEXT.format(label=self.label, model=self.model.label))
        else:
            logger.info(runtime.device_line(f"{self.label} (aligner {self.model.label})"))
        self._crispasr = crispasr

    @staticmethod
    def _gpu_on_its_own() -> bool:
        """Whether this CrispASR build would use a GPU with --device auto: a CUDA build with a usable NVIDIA
        GPU, the Vulkan build, the Apple GPU (item 12 review: not every CUDA build has a usable GPU)."""
        return install.runtime_status(None).use_gpu

    def align(self, audio: np.ndarray, text: str) -> List[TimedUnit]:
        """The timed units of ``text`` in ``audio`` (16 kHz mono float32); ``[]`` when nothing was aligned."""
        if self._crispasr is None:
            raise RuntimeError("the CrispASR aligner is not loaded")
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        if not text or not text.strip() or pcm.size == 0:
            return []
        try:
            aligned = self._crispasr.align_words(str(self._path), text, pcm, n_threads=self.n_threads)
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug(f"{self.label}: the {self.model.label} aligner failed on one VAD group: {e}")
            return []
        return units_from_binding(text, aligned, pcm.size / SAMPLE_RATE)

    def unload(self) -> None:
        """Release the library's cached aligner; safe to call twice."""
        crispasr, self._crispasr = self._crispasr, None
        if crispasr is not None:
            try:
                crispasr.free_aligner_cache()
            except RuntimeError as e:
                logger.debug(f"{self.label}: the aligner cache could not be released: {e}")
