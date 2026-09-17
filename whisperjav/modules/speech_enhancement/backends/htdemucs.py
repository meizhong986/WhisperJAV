#!/usr/bin/env python3
"""
htdemucs vocal isolation (Demucs v4).

Separates the voice from music and effects, and hands the recogniser the voice
alone. Useful where a scene has background music or heavy effects over the
dialogue -- the case BS-RoFormer also serves, with a different model.

The demucs package is an ordinary WhisperJAV dependency (owner's decision,
2026-09-17, replacing an earlier one): it is installed with everything else, not
by hand. It adds three packages -- demucs, lameenc and sphn -- does not touch
torch, and all three publish wheels for every Python WhisperJAV declares
(3.10-3.13) on Windows, Linux and macOS, so nothing is built from source.

One thing still comes from the network the first time:

  the model weights    about 84 MB, fetched once from Meta's CDN
                       (dl.fbaipublicfiles.com) and cached by torch.hub.
                       NOTE: not Hugging Face, so --offline, --hf-endpoint and
                       any Hugging Face mirror do not apply to it.

A note on failures, honestly stated. This backend currently raises
SpeechEnhancerUnavailable for everything that goes wrong, including a failure on
one piece of audio, while the other four enhancers report a per-scene failure and
let the run continue. That inconsistency predates the error-handling rules being
settled with the owner and is NOT a decided design: the agreed table governs, and
this backend follows it once it exists.

Model:
    htdemucs -- Hybrid Transformer Demucs, the v4 default. Four stems; we keep
    "vocals".

Example:
    whisperjav video.mp4 --ensemble --pass1-speech-enhancer htdemucs
"""

from typing import Union, List, Optional
from pathlib import Path
import time
import logging

import numpy as np

from ..base import (
    EnhancementResult,
    SpeechEnhancerUnavailable,
    load_audio_to_array,
    resample_audio,
    resolve_torch_device,
)

logger = logging.getLogger("whisperjav")


# htdemucs is trained at 44.1 kHz and expects stereo.
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MODEL = "htdemucs"

# The models this backend offers. Everything here is a Demucs v4 bag of models;
# htdemucs is the plain one and the only one enabled, the rest are listed so a
# name typed by hand is refused with a useful message rather than a stack trace.
SUPPORTED_MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]

# What to say if the package is missing. It ships with WhisperJAV, so this means
# a damaged or partial installation rather than a choice the user has yet to
# make. Kept here so the pre-flight check and the backend use the same words.
INSTALL_HINT = "pip install demucs"
WEIGHTS_MB = 84
WEIGHTS_HOST = "dl.fbaipublicfiles.com"


def is_package_installed() -> bool:
    """True when the demucs package can be imported."""
    import importlib.util
    return importlib.util.find_spec("demucs") is not None


class HtDemucsSpeechEnhancer:
    """Vocal isolation with Demucs v4 (htdemucs)."""

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Which Demucs v4 model to use (default: htdemucs).
            device: "cuda", "cpu", or None to choose the best available.
            **kwargs: Accepted and ignored, for protocol compatibility.
        """
        self._model_name = model or DEFAULT_MODEL
        if self._model_name not in SUPPORTED_MODELS:
            raise SpeechEnhancerUnavailable(
                "htdemucs: unknown model {!r}. Choose from: {}.".format(
                    self._model_name, ", ".join(SUPPORTED_MODELS)))
        self._device = device
        self._separator = None
        self._resolved_device = None
        self._initialized = False

    # ------------------------------------------------------------------ setup

    def _ensure_initialized(self) -> bool:
        """
        Load the model, fetching the weights on first use.

        Raises SpeechEnhancerUnavailable, with what the user can do about it,
        rather than returning False: this backend does not degrade silently.
        """
        if self._initialized:
            return True

        if not is_package_installed():
            raise SpeechEnhancerUnavailable(
                "htdemucs needs the demucs package, which is not installed. "
                "Install it with: {}".format(INSTALL_HINT))

        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model  # noqa: F401  (checked early)
        except ImportError as e:
            raise SpeechEnhancerUnavailable(
                "htdemucs: the demucs package is installed but could not be "
                "loaded ({}). Reinstalling it usually fixes this: {}".format(
                    e, INSTALL_HINT))

        device = resolve_torch_device(self._device)
        logger.info("Loading %s for vocal isolation on %s (first run downloads "
                    "about %s MB from %s)",
                    self._model_name, device, WEIGHTS_MB, WEIGHTS_HOST)

        try:
            model = get_model(self._model_name)
        except Exception as e:
            raise SpeechEnhancerUnavailable(
                "htdemucs: the model weights could not be fetched from {} "
                "({}). They are about {} MB and are downloaded once.".format(
                    WEIGHTS_HOST, e, WEIGHTS_MB))

        try:
            model.to(device)
            model.eval()
        except Exception as e:
            # Almost always "out of memory" on a small card.
            raise SpeechEnhancerUnavailable(
                "htdemucs: the model could not be placed on {} ({}). On a card "
                "with little memory, run this enhancer on the processor instead "
                "of the graphics card, or choose a lighter clean-up.".format(
                    device, e))

        self._separator = model
        self._resolved_device = device
        self._initialized = True
        logger.info("%s ready on %s", self._model_name, device)
        return True

    # -------------------------------------------------------------- protocol

    @property
    def name(self) -> str:
        return "htdemucs"

    @property
    def display_name(self) -> str:
        return "htdemucs (Demucs v4 vocal isolation)"

    def get_preferred_sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    def get_output_sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    def get_supported_models(self) -> List[str]:
        return list(SUPPORTED_MODELS)

    def is_lightweight(self) -> bool:
        return False

    def cleanup(self) -> None:
        """Release the model and any memory it holds on the graphics card."""
        self._separator = None
        self._initialized = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- the work

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Keep the voice and drop the rest.

        Every failure here is raised as SpeechEnhancerUnavailable and stops the
        run: the user chose this clean-up, and carrying on would hand back
        subtitles made from audio they asked to have cleaned up, from a run that
        exited 0. Nothing in this backend returns a failed result.
        """
        start_time = time.time()

        # Before the audio, not after: if the model cannot be loaded at all, the
        # user should be told that rather than made to wait for a read first.
        self._ensure_initialized()

        try:
            audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
        except Exception as e:
            raise SpeechEnhancerUnavailable(
                "htdemucs: the audio to clean up could not be read ({}).".format(e))

        try:
            import torch
            from demucs.apply import apply_model

            model_sr = DEFAULT_SAMPLE_RATE
            if actual_sr != model_sr:
                audio_for_model = resample_audio(audio_data, actual_sr, model_sr)
                logger.debug("Resampled %sHz -> %sHz for htdemucs", actual_sr, model_sr)
            else:
                audio_for_model = audio_data

            vocals = self._separate_vocals(audio_for_model, torch, apply_model)

            return EnhancementResult(
                audio=vocals,
                sample_rate=model_sr,
                method="htdemucs-{}".format(self._model_name),
                parameters={
                    "model": self._model_name,
                    "stem": "vocals",
                    "device": self._resolved_device,
                    "input_sr": actual_sr,
                    "output_sr": model_sr,
                },
                processing_time_sec=time.time() - start_time,
                metadata={
                    "input_samples": len(audio_data),
                    "output_samples": len(vocals),
                },
                success=True,
                error_message=None,
            )

        except SpeechEnhancerUnavailable:
            raise
        except Exception as e:
            message = str(e).lower()
            # Match what running out of memory actually says. "cuda" on its own
            # appears in plenty of unrelated faults, and telling someone to free
            # graphics memory they have not run out of sends them the wrong way.
            if ("out of memory" in message
                    or "outofmemory" in message
                    or "can't allocate memory" in message):
                raise SpeechEnhancerUnavailable(
                    "htdemucs: the graphics card ran out of memory while "
                    "isolating the voice ({}). Run this enhancer on the "
                    "processor instead, or choose a lighter clean-up.".format(e))
            raise SpeechEnhancerUnavailable(
                "htdemucs: isolating the voice failed ({}).".format(e))

    def _separate_vocals(self, audio: np.ndarray, torch, apply_model) -> np.ndarray:
        """Run the model and return the vocal stem as mono float32."""
        # Demucs wants (channels, samples) stereo. Mono is duplicated, which is
        # what its own command-line front end does with a mono file.
        if audio.ndim == 1:
            stereo = np.stack([audio, audio], axis=0)
        else:
            stereo = audio if audio.shape[0] <= audio.shape[1] else audio.T
            if stereo.shape[0] == 1:
                stereo = np.repeat(stereo, 2, axis=0)

        tensor = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))

        # Demucs normalises by the mixture's own loudness; doing it here and
        # undoing it after keeps the output at the level the caller gave us.
        reference = tensor.mean(0)
        mean, std = reference.mean(), reference.std()
        if float(std) == 0.0:
            std = torch.tensor(1.0)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            stems = apply_model(
                self._separator,
                tensor[None],
                device=self._resolved_device,
                progress=False,
            )[0]
        stems = stems * std + mean

        sources = list(self._separator.sources)
        if "vocals" not in sources:
            raise SpeechEnhancerUnavailable(
                "htdemucs: model {!r} has no vocal stem (it separates: {}).".format(
                    self._model_name, ", ".join(sources)))
        vocals = stems[sources.index("vocals")]

        # Back to the mono float32 the rest of WhisperJAV works in.
        return vocals.mean(0).cpu().numpy().astype(np.float32)
