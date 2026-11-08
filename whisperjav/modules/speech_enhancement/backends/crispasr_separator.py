#!/usr/bin/env python3
"""
Vocal isolation run by CrispASR: Mel-Band-RoFormer and Mini-BS-RoFormer.

1.10.2 (owner W594, W596): two speech enhancer backends, ``mel-band-roformer`` and ``mini-bs-roformer``. They
separate the voice from music and effects and keep the voice, as htdemucs does.

How a scene goes through, as the viability test ran it (tools/crispasr_viability.py CrispEnhancer;
docs/plans/design/2026-11-07-phase-a-results.md): the mono scene at 44.1 kHz is duplicated into stereo (as
WhisperJAV's htdemucs does), CrispASR's ``Session.separate`` splits it, the "vocals" stem is averaged back to mono
at 44.1 kHz; WhisperJAV's own enhance_scenes() resamples it to 16 kHz.

The model files are pinned in ``whisperjav/crispasr/manifest.py`` (ENHANCER_MODELS) and downloaded from Hugging
Face on first use (``--offline`` applies). The device follows the run's ``--device`` as for the other speech
enhancers: cpu = CrispASR's use_gpu off, anything else on (on a CPU-only CrispASR build CrispASR runs it on the CPU).
Measured on an RTX 3060 (Phase A): Mel-Band-RoFormer 4.4 times real time, 2.4-3.4 GB of GPU memory, about 0.02 times
real time on the CPU; Mini-BS-RoFormer 9.5 times on the GPU, 0.7 times on the CPU.

Failures follow the rule of every other clean-up (owner, 2026-09-17): something that means this clean-up cannot be
had at all (CrispASR not loadable, the model not downloadable or not openable) is raised as
SpeechEnhancerUnavailable and stops the run; a failure on one scene comes back as a failed result and that scene
goes through uncleaned.
"""

from pathlib import Path
from typing import List, Optional, Union
import logging
import os
import tempfile
import time

import numpy as np

from ..base import (
    EnhancementResult,
    SpeechEnhancerUnavailable,
    create_failed_result,
    load_audio_to_array,
    resample_audio,
)

logger = logging.getLogger("whisperjav")

SEPARATOR_SAMPLE_RATE = 44100   # CrispASR's separators take 44.1 kHz stereo (_binding.py: Session.separate)
STEM = "vocals"


def _crispasr_threads() -> int:
    """CPU threads for CrispASR: min(8, CPU count), as the CrispASR pipelines use (crispasr/engine.py)."""
    return max(1, min(8, os.cpu_count() or 4))


class _CrispASRSeparator:
    """The shared part of the two backends. A subclass names its BACKEND and DISPLAY_NAME."""

    BACKEND = ""
    DISPLAY_NAME = ""

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None, **kwargs):
        """
        Args:
            model: accepted for the BACKEND[:DETAIL] form; only "vocals" (the default) is offered.
            device: the run's device: "cpu" runs CrispASR with use_gpu off; anything else with it on.
            **kwargs: accepted and ignored, for protocol compatibility.
        """
        detail = (model or STEM).strip().lower()
        if detail not in (STEM, self.BACKEND):
            raise SpeechEnhancerUnavailable(
                f"{self.BACKEND}: unknown model {model!r}. Only '{STEM}' is offered.")
        self._device = device
        self._use_gpu = str(device or "auto").lower() != "cpu"
        self._session = None

    # ------------------------------------------------------------------ setup

    def _ensure_initialized(self) -> bool:
        """Download (first use) and open the model. Raises SpeechEnhancerUnavailable when it cannot be had."""
        if self._session is not None:
            return True
        try:
            import crispasr
        except ImportError as e:
            raise SpeechEnhancerUnavailable(
                f"{self.DISPLAY_NAME} is run by CrispASR, which could not be loaded on this computer ({e}). "
                f"CrispASR is installed with WhisperJAV; reinstalling WhisperJAV restores it.")
        from whisperjav.crispasr import install, manifest, native_log
        from whisperjav.crispasr.errors import CrispASRSetupError
        native_log.quiet_debug_lines(crispasr)
        model = manifest.ENHANCER_MODELS[self.BACKEND]
        try:
            path = install.ensure_model(model)
            path = install.stage_model(path, Path(tempfile.gettempdir()), model.sha256, keep_name=True)
        except CrispASRSetupError as e:
            raise SpeechEnhancerUnavailable(f"{self.DISPLAY_NAME}: {e}")
        logger.info("Loading %s for vocal isolation (%s, %s)", self.DISPLAY_NAME, model.filename,
                    "GPU when the CrispASR build has one" if self._use_gpu else "CPU")
        try:
            self._session = crispasr.Session(str(path), n_threads=_crispasr_threads(), backend=model.backend,
                                             use_gpu=self._use_gpu, flash_attn=True)
        except Exception as e:
            raise SpeechEnhancerUnavailable(f"{self.DISPLAY_NAME}: the model could not be opened ({path}): {e}")
        return True

    # -------------------------------------------------------------- protocol

    @property
    def name(self) -> str:
        return self.BACKEND

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    def get_preferred_sample_rate(self) -> int:
        return SEPARATOR_SAMPLE_RATE

    def get_output_sample_rate(self) -> int:
        return SEPARATOR_SAMPLE_RATE

    def get_supported_models(self) -> List[str]:
        return [STEM]

    def is_lightweight(self) -> bool:
        return False

    def cleanup(self) -> None:
        """Close the CrispASR session (frees its model and GPU memory)."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception as e:
                logger.debug(f"{self.BACKEND}: closing the CrispASR session failed ({e})")
            finally:
                self._session = None

    # --------------------------------------------------------------- the work

    def enhance(self, audio: Union[np.ndarray, Path, str], sample_rate: int, **kwargs) -> EnhancementResult:
        """Keep the voice of one scene. A failure on this scene comes back as a failed result."""
        start_time = time.time()
        self._ensure_initialized()   # before the audio: a clean-up that cannot be had stops the run first

        try:
            audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
        except Exception as e:
            raise SpeechEnhancerUnavailable(f"{self.BACKEND}: the audio to clean up could not be read ({e}).")

        try:
            mono = np.asarray(audio_data, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1).astype(np.float32)
            if actual_sr != SEPARATOR_SAMPLE_RATE:
                mono = np.asarray(resample_audio(mono, actual_sr, SEPARATOR_SAMPLE_RATE), dtype=np.float32)
            stereo = np.ascontiguousarray(np.repeat(mono[:, None], 2, axis=1))
            stems = self._session.separate(stereo)
            if STEM not in stems:
                raise SpeechEnhancerUnavailable(
                    f"{self.DISPLAY_NAME}: the model gave no '{STEM}' stem (stems: {sorted(stems)}).")
            vocals = np.asarray(stems[STEM], dtype=np.float32).reshape(-1, 2).mean(axis=1).astype(np.float32)
            return EnhancementResult(
                audio=vocals,
                sample_rate=SEPARATOR_SAMPLE_RATE,
                method=self.BACKEND,
                parameters={"model": self.BACKEND, "stem": STEM, "use_gpu": self._use_gpu,
                            "input_sr": actual_sr, "output_sr": SEPARATOR_SAMPLE_RATE},
                processing_time_sec=time.time() - start_time,
                metadata={"input_samples": len(audio_data), "output_samples": len(vocals)},
                success=True,
                error_message=None,
            )
        except SpeechEnhancerUnavailable:
            raise
        except Exception as e:
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=self.BACKEND,
                error_message=f"{self.BACKEND}: isolating the voice failed ({e}).",
                processing_time_sec=time.time() - start_time,
            )


class MelBandRoformerSpeechEnhancer(_CrispASRSeparator):
    """CrispASR Mel-Band-RoFormer (vocals), 457 MB."""

    BACKEND = "mel-band-roformer"
    DISPLAY_NAME = "Mel-Band-RoFormer (CrispASR)"


class MiniBsRoformerSpeechEnhancer(_CrispASRSeparator):
    """CrispASR Mini-BS-RoFormer (vocals), 92 MB (1.10.2, owner W594)."""

    BACKEND = "mini-bs-roformer"
    DISPLAY_NAME = "Mini-BS-RoFormer (CrispASR)"
