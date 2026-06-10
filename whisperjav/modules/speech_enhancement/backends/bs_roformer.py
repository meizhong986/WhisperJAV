"""
BS-RoFormer speech enhancement backend.

Uses BS-RoFormer for vocal isolation and music source separation.
Particularly useful for extracting vocals from audio with background music.

Available models:
- vocals: Extract vocals (isolate speech/singing)
- other: Extract non-vocal audio

Installation: pip install bs-roformer-infer

Note: BS-RoFormer is primarily designed for music source separation
at 44.1kHz. For speech-only content, ClearVoice may be more appropriate.
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import sys
import time
import tempfile
import numpy as np
import logging

from ..base import (
    EnhancementResult,
    load_audio_to_array,
    create_failed_result,
    resample_audio,
    resolve_torch_device,
)

logger = logging.getLogger("whisperjav")

# Model configurations
_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "vocals": {
        "sample_rate": 44100,
        "description": "Vocal isolation (extract speech/singing)",
        "stem": "vocals",
    },
    "other": {
        "sample_rate": 44100,
        "description": "Non-vocal audio extraction",
        "stem": "other",
    },
}

DEFAULT_MODEL = "vocals"
DEFAULT_SAMPLE_RATE = 44100


class _FilteredStdout:
    """stdout proxy used during demix_track().

    Keeps useful progress (the one-shot "Estimated total processing time..."
    line) but drops the per-chunk carriage-return redraw spam
    ("Estimated time remaining...") that otherwise stacks up in piped logs.
    """

    _DROP = "Estimated time remaining"

    def __init__(self):
        self._real = sys.__stdout__

    def write(self, s):
        try:
            if s and self._DROP not in s:
                self._real.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


class BSRoformerSpeechEnhancer:
    """
    BS-RoFormer vocal isolation backend.

    Uses BS-RoFormer to separate vocals from background music/noise.
    Best suited for content with music or complex background audio.

    Example:
        enhancer = BSRoformerSpeechEnhancer(model="vocals")
        result = enhancer.enhance(audio_array, sample_rate=44100)
        vocals = result.audio

        # Always cleanup when done
        enhancer.cleanup()
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        ckpt_path: Optional[str] = None,
        config_path: Optional[str] = None,
        variant: str = "revive2",
        num_overlap: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize BS-RoFormer enhancer.

        Args:
            model: Stem to extract ("vocals" or "other").
            device: Device to use ("cuda", "cpu", "auto").
            ckpt_path: Optional explicit path to a .ckpt (overrides download).
            config_path: Optional explicit path to the model config .yaml.
            variant: Which pcunwa BS-Roformer-Revive model to auto-download when
                ckpt_path is not given — "revive2" (best Bleedless, recommended for
                ASR), "revive3e" (max Fullness), or "revive" (v1). Bleedless wins
                for transcription: cleanest vocal stem = best ASR input.
            **kwargs: Additional parameters (ignored).
        """
        self._model_name = model if model in _MODEL_INFO else DEFAULT_MODEL
        self._device = device
        self._ckpt_path = ckpt_path
        self._config_path = config_path
        self._variant = (variant or "revive2").lower()
        # num_overlap: chunk overlap during separation. Higher = better quality
        # but slower (config default is 2). Lower (e.g. 1) ~halves separation time
        # at a small quality cost. None = use the model config's value.
        try:
            self._num_overlap = int(num_overlap) if num_overlap is not None else None
        except (TypeError, ValueError):
            self._num_overlap = None
        self._separator = None
        self._initialized = False
        self._config = None
        self._device_torch = None
        self._target_instrument = None

        if model not in _MODEL_INFO:
            logger.warning(
                f"Unknown BS-RoFormer model '{model}', using {DEFAULT_MODEL}. "
                f"Available: {list(_MODEL_INFO.keys())}"
            )

        logger.debug(f"BSRoformerSpeechEnhancer configured: model={self._model_name}")

    # Public BS-Roformer-Revive mirror (the package's own registry URLs are dead).
    # Revive 2 = best Bleedless (cleanest stem → best ASR input, our default).
    _HF_BASE = "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/"
    _VARIANT_CKPT = {
        "revive": "bs_roformer_revive.ckpt",
        "revive2": "bs_roformer_revive2.ckpt",
        "revive3e": "bs_roformer_revive3e.ckpt",
    }
    _CONFIG_FILE = "config.yaml"

    def _resolve_assets(self):
        """Return (ckpt_path, config_path) as Paths, or (None, None) on failure.

        Resolution order: explicit user paths -> cache -> download from HF.
        """
        from pathlib import Path
        # 1) explicit paths
        if self._ckpt_path and self._config_path:
            cp, gp = Path(self._ckpt_path), Path(self._config_path)
            if cp.exists() and gp.exists():
                logger.info(f"BS-RoFormer: using explicit checkpoint {cp.name}")
                return cp, gp
            logger.warning("BS-RoFormer explicit ckpt/config path missing; falling back to download")

        variant = self._variant if self._variant in self._VARIANT_CKPT else "revive2"
        ckpt_name = self._VARIANT_CKPT[variant]
        cache_dir = Path.home() / ".cache" / "whisperjav" / "bs_roformer" / variant
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / ckpt_name
        cfg_path = cache_dir / self._CONFIG_FILE

        # 2) cache
        if ckpt_path.exists() and cfg_path.exists():
            return ckpt_path, cfg_path

        # 3) download
        import urllib.request
        try:
            if not cfg_path.exists():
                logger.info(f"Downloading BS-RoFormer config -> {cfg_path}")
                urllib.request.urlretrieve(self._HF_BASE + self._CONFIG_FILE, str(cfg_path))
            if not ckpt_path.exists():
                logger.info(f"Downloading BS-RoFormer weights ({variant}, ~639MB) -> {ckpt_path} (first run only)")
                urllib.request.urlretrieve(self._HF_BASE + ckpt_name, str(ckpt_path))
        except Exception as e:
            logger.error(f"BS-RoFormer download failed: {e}")
            # Clean partial files so a retry re-downloads cleanly.
            for p in (ckpt_path, cfg_path):
                try:
                    if p.exists() and p.stat().st_size == 0:
                        p.unlink()
                except Exception:
                    pass
            return None, None

        if ckpt_path.exists() and cfg_path.exists():
            return ckpt_path, cfg_path
        return None, None

    def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of BS-RoFormer model.

        Uses the real bs-roformer-infer (>=0.1.0) API: download the model
        checkpoint+config from the registry, build the net via
        get_model_from_config, load weights, and move to device. Separation
        itself is done later via demix_track() (see _process_audio).

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        try:
            import torch
            import yaml
            from ml_collections import ConfigDict
            from bs_roformer.utils import get_model_from_config
            from bs_roformer.inference import SafeLoaderWithTuple
        except ImportError as e:
            logger.error(f"bs-roformer not installed or incompatible: {e}")
            logger.error("Install with: pip install bs-roformer-infer")
            return False

        try:
            # Resolve ckpt + config. The bs-roformer-infer registry URLs are all
            # dead upstream (gated/401 or 404), so we self-host the download from
            # the public pcunwa/BS-Roformer-Revive HF repo. Order:
            #   1) explicit ckpt_path/config_path (user-supplied)
            #   2) cached file from a previous run
            #   3) download the selected Revive variant from HF
            ckpt_path, cfg_path = self._resolve_assets()
            if ckpt_path is None or cfg_path is None:
                logger.error("BS-RoFormer model assets unavailable (download failed)")
                return False
            logger.info(f"Loading BS-RoFormer model (variant={self._variant}, stem={self._model_name})")

            # Load config + build model.
            with open(cfg_path) as f:
                self._config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))

            # Override chunk overlap if requested (speed/quality knob).
            if self._num_overlap is not None:
                ov = max(1, self._num_overlap)
                try:
                    if hasattr(self._config, "inference"):
                        self._config.inference.num_overlap = ov
                    logger.info(f"BS-RoFormer: num_overlap overridden to {ov} (config default was 2)")
                except Exception as e:
                    logger.warning(f"Could not set num_overlap={ov}: {e}")

            model = get_model_from_config("bs_roformer", self._config)
            if model is None:
                logger.error("get_model_from_config returned None for bs_roformer")
                return False
            state = torch.load(str(ckpt_path), map_location=torch.device("cpu"))
            model.load_state_dict(state)

            # Resolve device (cuda > cpu; bs-roformer is heavy, MPS unsupported here).
            device = resolve_torch_device(self._device)
            if device == "mps":
                logger.info("BS-RoFormer: MPS not supported, using CPU")
                device = "cpu"
            self._device_torch = torch.device(device)
            model = model.to(self._device_torch)
            model.eval()
            self._separator = model

            # Which instrument to keep (vocals for speech isolation).
            self._target_instrument = (
                getattr(self._config.training, "target_instrument", None)
                or self._model_name
            )

            self._initialized = True
            logger.info(f"BS-RoFormer loaded successfully on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BS-RoFormer: {e}")
            return False

    @property
    def name(self) -> str:
        return "bs-roformer"

    @property
    def display_name(self) -> str:
        stem = _MODEL_INFO.get(self._model_name, {}).get("stem", "vocals")
        return f"BS-RoFormer ({stem})"

    def enhance(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int,
        **kwargs
    ) -> EnhancementResult:
        """
        Extract vocals using BS-RoFormer.

        Args:
            audio: Audio data as numpy array (float32, mono), or path to file
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters (ignored)

        Returns:
            EnhancementResult with extracted vocals, or original on failure

        Note:
            On failure, returns original audio with success=False
            for graceful degradation.
        """
        start_time = time.time()

        # Load audio if path provided
        try:
            audio_data, actual_sr = load_audio_to_array(audio, sample_rate)
        except Exception as e:
            return create_failed_result(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=sample_rate,
                method=f"bs-roformer-{self._model_name}",
                error_message=f"Failed to load audio: {e}",
                processing_time_sec=time.time() - start_time,
            )

        # Initialize model if needed
        if not self._ensure_initialized():
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"bs-roformer-{self._model_name}",
                error_message="Failed to initialize BS-RoFormer model",
                processing_time_sec=time.time() - start_time,
            )

        try:
            model_sr = DEFAULT_SAMPLE_RATE  # 44100

            # Resample to model's expected rate if needed
            if actual_sr != model_sr:
                audio_for_model = resample_audio(audio_data, actual_sr, model_sr)
                logger.debug(f"Resampled {actual_sr}Hz -> {model_sr}Hz for BS-RoFormer")
            else:
                audio_for_model = audio_data

            # Process with BS-RoFormer
            separated = self._process_audio(audio_for_model, model_sr)

            processing_time = time.time() - start_time

            return EnhancementResult(
                audio=separated,
                sample_rate=model_sr,
                method=f"bs-roformer-{self._model_name}",
                parameters={
                    "model": self._model_name,
                    "stem": _MODEL_INFO[self._model_name]["stem"],
                    "input_sr": actual_sr,
                    "output_sr": model_sr,
                },
                processing_time_sec=processing_time,
                metadata={
                    "input_samples": len(audio_data),
                    "output_samples": len(separated),
                },
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.warning(f"BS-RoFormer separation failed: {e}")
            return create_failed_result(
                audio=audio_data,
                sample_rate=actual_sr,
                method=f"bs-roformer-{self._model_name}",
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
            )

    def _process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through BS-RoFormer model.

        Args:
            audio: Audio data (float32, mono, at model's sample rate)
            sample_rate: Sample rate of audio

        Returns:
            Separated audio array (vocals or other stem)
        """
        import torch
        from bs_roformer.utils import demix_track

        stem = _MODEL_INFO[self._model_name]["stem"]

        # BS-RoFormer expects stereo (channels, samples). Build it.
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio], axis=0)  # (2, samples)
        else:
            audio_stereo = audio
            # Normalize to (channels, samples)
            if audio_stereo.shape[0] > audio_stereo.shape[1]:
                audio_stereo = audio_stereo.T
            if audio_stereo.shape[0] == 1:
                audio_stereo = np.repeat(audio_stereo, 2, axis=0)

        mixture = torch.tensor(audio_stereo, dtype=torch.float32)

        # demix_track returns ({instrument: ndarray(channels, samples)}, _).
        # The package writes progress to stdout: a useful one-shot ETA line
        # ("Estimated total processing time...") plus a per-chunk redraw spam
        # ("Estimated time remaining...", carriage-return based) that stacks up
        # in a piped log. Filter: keep the useful ETA, drop the \r redraw spam.
        import contextlib
        with contextlib.redirect_stdout(_FilteredStdout()):
            res, _ = demix_track(self._config, self._separator, mixture, self._device_torch)

        if stem in res:
            separated = res[stem]
        elif self._target_instrument in res:
            separated = res[self._target_instrument]
        else:
            separated = list(res.values())[0]

        # demix_track output is (channels, samples) -> collapse to mono.
        if isinstance(separated, np.ndarray):
            if separated.ndim > 1:
                if separated.shape[0] <= 2:      # (channels, samples)
                    separated = np.mean(separated, axis=0)
                else:                             # (samples, channels)
                    separated = np.mean(separated, axis=1)
            return separated.astype(np.float32)

        raise RuntimeError(f"Unexpected result type from BS-RoFormer: {type(separated)}")

    def get_preferred_sample_rate(self) -> int:
        """Return 44100Hz (standard for music/BS-RoFormer)."""
        return DEFAULT_SAMPLE_RATE

    def get_output_sample_rate(self) -> int:
        """Return 44100Hz (same as input for BS-RoFormer)."""
        return DEFAULT_SAMPLE_RATE

    def cleanup(self) -> None:
        """Release model resources."""
        if self._separator is not None:
            try:
                del self._separator
                self._separator = None
                self._initialized = False

                # Force garbage collection for GPU memory
                import gc
                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                logger.debug("BS-RoFormer resources released")
            except Exception as e:
                logger.warning(f"Error during BS-RoFormer cleanup: {e}")

    def get_supported_models(self) -> List[str]:
        """Return list of supported model variants."""
        return list(_MODEL_INFO.keys())

    def __repr__(self) -> str:
        return f"BSRoformerSpeechEnhancer(model={self._model_name})"

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
