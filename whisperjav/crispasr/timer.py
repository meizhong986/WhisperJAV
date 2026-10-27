"""The CrispASR timing model: where speech starts in a VAD group, and where a new sentence starts (item 12, C3).

Whisper small, run by the binding's Whisper-only ``crispasr.CrispASR`` class with DTW token times (owner W255;
design note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md, sections 5.1 and 14.3). Its text is thrown
away; what is kept per Whisper segment is the DTW time of its first text token (where that segment's speech
starts), ``no_speech_prob``, and whether it had text. The special tokens (segment start, timestamps, end of text)
carry no DTW time, so nothing here says where speech ends (probe P2c).

The class has no setting to hold it on the CPU: it runs where the CrispASR build runs, also under
``--device cpu``. ``crispasr`` is imported only in ``load()``, so importing this module stays light.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from whisperjav.crispasr import install
from whisperjav.crispasr.engine import LANGUAGE
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import ModelFile, timing_model_for
from whisperjav.utils.logger import logger

# whisper.cpp's alignment-head presets for DTW (8 = small).
DTW_PRESET = {"whisper-small": 8}
# Greedy, no carried context, a cap that only stops a runaway decode: the text is thrown away, and a 3 s
# VAD group's Japanese text is far below 96 tokens.
DECODE_OPTIONS = {"token_timestamps": 1, "no_context": 1, "temperature": 0.0, "max_tokens": 96}

# Text approved by the owner as drafted (W268); design note section 9, Q9.
GPU_DESPITE_CPU_TEXT = ("{label}: the {model} timing model still asks for the GPU with --device cpu; CrispASR cannot "
                        "hold it on the CPU.")
NO_WHISPER_TEXT = ("This CrispASR ({version}) has no Whisper engine with DTW timing; run the WhisperJAV "
                   "installer again.")


@dataclass(frozen=True)
class WhisperSegment:
    onset: Optional[float]  # DTW time of the first text token, seconds from the start of the audio passed in
    no_speech_prob: float
    has_text: bool


def _is_text_token(text: str) -> bool:
    return bool(text.strip()) and not text.startswith("[_")


def segments_from_binding(segments) -> List[WhisperSegment]:
    """The binding's ``Segment`` list as WhisperJAV's ``WhisperSegment``s."""
    out = []
    for seg in segments:
        onset = next((t.dtw_time for t in (getattr(seg, "tokens", None) or ())
                      if _is_text_token(str(t.text)) and t.dtw_time is not None), None)
        out.append(WhisperSegment(None if onset is None else float(onset), float(seg.no_speech_prob),
                                  bool(str(seg.text or "").strip())))
    return out


class CrispTimer:
    """One Whisper timing model, loaded once and reused until ``unload()``."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 temp_dir: Optional[Path] = None, label: str = "WhisperJAV"):
        self.model: ModelFile = timing_model_for(model_name)
        self.device = device
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "whisperjav"
        self.label = label
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """CrispASR, the device and the model file. Raises CrispASRSetupError with the reason."""
        if self._model is not None:
            return
        runtime = install.runtime_status(self.device)
        try:
            import crispasr
        except (ImportError, OSError) as e:
            raise CrispASRSetupError(install.NOT_LOADED_TEXT.format(reason=e)) from e
        if not hasattr(crispasr, "CrispASR"):
            raise CrispASRSetupError(NO_WHISPER_TEXT.format(version=runtime.version))
        path = install.stage_model(install.ensure_model(self.model), self.temp_dir, self.model.sha256,
                                   keep_name=True)
        try:
            model = crispasr.CrispASR(str(path), dtw=True, dtw_aheads_preset=DTW_PRESET[self.model.name])
        except (RuntimeError, OSError, ValueError) as e:
            raise CrispASRSetupError(f"The timing model {self.model.label} could not be opened ({path}): {e}") from e
        # Said only where the build would use a GPU on its own (--device auto): a CUDA build with a usable
        # NVIDIA GPU, the Vulkan build, the Apple GPU (item 12 review).
        if (self.device or "").strip().lower() == "cpu" and install.runtime_status(None).use_gpu:
            logger.warning(GPU_DESPITE_CPU_TEXT.format(label=self.label, model=self.model.label))
        else:
            logger.info(runtime.device_line(f"{self.label} (timing model {self.model.label})"))
        self._model = model

    def time(self, audio: np.ndarray) -> List[WhisperSegment]:
        """Whisper's segments for ``audio`` (16 kHz mono float32); ``[]`` when Whisper gave nothing or failed."""
        if self._model is None:
            raise RuntimeError("the CrispASR timing model is not loaded")
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        if pcm.size == 0:
            return []
        try:
            segments = self._model.transcribe_pcm(pcm, language=LANGUAGE, decode_options=dict(DECODE_OPTIONS))
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug(f"{self.label}: the {self.model.label} timing model failed on one VAD group: {e}")
            return []
        return segments_from_binding(segments)

    def unload(self) -> None:
        """Close the model; safe to call twice."""
        model, self._model = self._model, None
        if model is not None:
            try:
                model.close()
            except (RuntimeError, OSError) as e:
                logger.debug(f"{self.label}: the timing model could not be closed: {e}")
