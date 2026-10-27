"""
CrispASR Whisper timing model SpeechTimer adapter (1.10.0, item 12 C3).

Wraps the CrispASR timing model (whisperjav/crispasr/timer.py: Whisper small with DTW token times) behind the
SpeechTimer protocol. For each frame (one VAD group) it returns timing_rules.TimedSegment entries: where each
Whisper segment's speech starts, its no_speech_prob, and whether it had text. Whisper's text is never used;
the rules in timing_rules.py move the subtitle's start and split the frame (design note
docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md, section 5).

A failed load() raises, so the orchestrator keeps the subtitles as they are and says why (timing_unavailable).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from whisperjav.modules.subtitle_pipeline.aligners.crispasr import frame_audio_16k
from whisperjav.modules.subtitle_pipeline.timing_rules import TimedSegment
from whisperjav.utils.logger import logger


class CrispASRTimer:
    """SpeechTimer backed by CrispASR's Whisper engine with DTW."""

    # Takes in-memory frame audio, so the orchestrator can run pathless (no per-VAD-group temp WAVs).
    accepts_array = True

    def __init__(
        self,
        model: str = "whisper-small",
        device: str = "auto",
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            model: Timing model short name (whisperjav/crispasr/manifest.py TIMING_MODELS).
            device: 'auto', 'cuda' or 'cpu' (CrispASR's device rule; the model itself follows the build).
            temp_dir: Where a model file is staged when its folder name is not plain ASCII.
        """
        from whisperjav.crispasr.manifest import timing_model_for

        timing_model_for(model)  # an unknown name fails here, with the choices in its text
        self._config = {"model": model, "device": device, "temp_dir": temp_dir}
        self._timer = None

    @property
    def is_loaded(self) -> bool:
        return self._timer is not None

    @property
    def display_name(self) -> str:
        """The timing model as the run summary names it: its label (Whisper small)."""
        from whisperjav.crispasr.manifest import timing_model_for

        return timing_model_for(self._config["model"]).label

    def load(self) -> None:
        """Make CrispASR and the model file ready; raises CrispASRSetupError with the reason."""
        if self._timer is not None:
            return

        from whisperjav.crispasr.timer import CrispTimer

        cfg = self._config
        timer = CrispTimer(model_name=cfg["model"], device=cfg["device"], temp_dir=cfg["temp_dir"],
                           label="ChronosJAV")
        timer.load()
        self._timer = timer
        logger.info("[CrispASRTimer] Timing model %s loaded", cfg["model"])

    def unload(self) -> None:
        if self._timer is None:
            return
        self._timer.unload()
        self._timer = None
        logger.info("[CrispASRTimer] Timing model unloaded")

    def time(self, audio: Union[Path, str, np.ndarray]) -> list[TimedSegment]:
        """The frame's timed segments, onsets in seconds from the frame's start."""
        if self._timer is None:
            raise RuntimeError("CrispASRTimer.time() called before load(). Call load() first.")
        return [TimedSegment(s.onset, s.no_speech_prob, s.has_text)
                for s in self._timer.time(frame_audio_16k(audio))]

    def cleanup(self) -> None:
        self.unload()
