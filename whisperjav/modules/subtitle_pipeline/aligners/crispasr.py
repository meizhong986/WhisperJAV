"""
CrispASR GGUF aligner TextAligner adapter (1.10.0, item 12 C1).

Wraps the shared CrispASR aligner (whisperjav/crispasr/aligner.py) behind the TextAligner protocol, so
ChronosJAV can time its text with a GGUF aligner run by CrispASR: wav2vec2-ja (the default), Parakeet CTC
ja or Qwen3-ForcedAligner GGUF (design note docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md,
section 4). It runs where the install's CrispASR build runs, so on Vulkan and Apple installs it uses the
GPU that ChronosJAV's PyTorch models cannot.

Each timed character becomes one WordTimestamp, with the marks that follow it ("た。"), as the Qwen3
adapter's merge_master_with_timestamps() attaches punctuation to the word before it. Times are relative to
the audio passed in; the last character ends at the end of that audio (see units_from_binding).

A failed load() raises, so the orchestrator keeps the text and takes its times from the speech segments
(#436 path in DecoupledSubtitlePipeline._step5_7_align).
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from whisperjav.modules.subtitle_pipeline.types import AlignmentResult, WordTimestamp
from whisperjav.utils.logger import logger

SAMPLE_RATE = 16000


def frame_audio_16k(audio: Union[Path, str, np.ndarray]) -> np.ndarray:
    """One frame's audio as mono float32 at 16 kHz: an array as the orchestrator hands it (16 kHz, as for the
    Qwen3 adapter), or a WAV file, resampled when it is not at 16 kHz. Shared with the CrispASR timer adapter."""
    if isinstance(audio, np.ndarray):
        data = audio.astype(np.float32, copy=False)
        return data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data
    import soundfile as sf

    data, rate = sf.read(str(audio), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if rate != SAMPLE_RATE:
        from whisperjav.modules.speech_enhancement.base import resample_audio

        data = resample_audio(data, rate, SAMPLE_RATE).astype(np.float32)
    return data


class CrispASRAligner:
    """TextAligner backed by a CrispASR GGUF aligner (crispasr.align_words)."""

    # Takes in-memory frame audio, so the orchestrator can run pathless (no per-VAD-group temp WAVs).
    accepts_array = True

    def __init__(
        self,
        model: str = "wav2vec2-ja",
        device: str = "auto",
        language: str = "ja",
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Store configuration for deferred loading.

        Args:
            model: Aligner short name (whisperjav/crispasr/manifest.py ALIGNER_MODELS).
            device: 'auto', 'cuda' or 'cpu' (CrispASR's device rule; the aligner itself follows the build).
            language: Kept for the protocol; the aligners here are Japanese models.
            temp_dir: Where a model file is staged when its folder name is not plain ASCII.
        """
        from whisperjav.crispasr.manifest import aligner_model_for

        aligner_model_for(model)  # an unknown name fails here, with the choices in its text
        self._config = {"model": model, "device": device, "language": language, "temp_dir": temp_dir}
        self._aligner = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the aligner is currently loaded."""
        return self._loaded

    @property
    def display_name(self) -> str:
        """The aligner as the run summary names it: its short name."""
        return self._config["model"]

    def load(self) -> None:
        """Make CrispASR and the model file ready; raises CrispASRSetupError with the reason."""
        if self._loaded:
            return

        from whisperjav.crispasr.aligner import CrispAligner

        cfg = self._config
        aligner = CrispAligner(model_name=cfg["model"], device=cfg["device"], temp_dir=cfg["temp_dir"],
                               label="ChronosJAV")
        aligner.load()
        self._aligner = aligner
        self._loaded = True
        # CrispASR opens the model file at the first align call; a failure there gives no times for
        # that VAD group, which the orchestrator counts and reports.
        logger.info("[CrispASRAligner] Aligner %s ready", cfg["model"])

    def unload(self) -> None:
        """Release the aligner."""
        if not self._loaded:
            return
        if self._aligner is not None:
            self._aligner.unload()
            self._aligner = None
        self._loaded = False
        logger.info("[CrispASRAligner] Aligner unloaded")

    def align(
        self,
        audio_path: Union[Path, np.ndarray],
        text: str,
        language: str = "ja",
        **kwargs: Any,
    ) -> AlignmentResult:
        """Align text to a single audio file or array."""
        return self.align_batch(audio_paths=[audio_path], texts=[text], language=language, **kwargs)[0]

    def align_batch(
        self,
        audio_paths: list[Union[Path, np.ndarray]],
        texts: list[str],
        language: str = "ja",
        **kwargs: Any,
    ) -> list[AlignmentResult]:
        """Align each text to its audio; one AlignmentResult per entry."""
        if not self._loaded:
            raise RuntimeError("CrispASRAligner.align_batch() called before load(). Call load() first.")

        results = []
        for i, (audio, text) in enumerate(zip(audio_paths, texts)):
            units = self._aligner.align(frame_audio_16k(audio), text) if text and text.strip() else []
            if not units:
                results.append(AlignmentResult(words=[], metadata={"scene_index": i, "skipped": True}))
                continue
            words = [WordTimestamp(word=u.text, start=u.start, end=u.end) for u in units]
            results.append(AlignmentResult(
                words=words,
                metadata={"scene_index": i, "aligner": f"crispasr:{self._config['model']}",
                          "merged_word_count": len(words)},
            ))
        return results

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
