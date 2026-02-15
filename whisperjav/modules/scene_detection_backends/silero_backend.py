"""
Silero VAD scene detector with two-pass strategy.

Pass 1 (Coarse): Inherited from AuditokSceneDetector — auditok finds chapters
Pass 2 (Fine): Silero VAD chunks each chapter based on speech boundaries

NO assistive processing — Silero VAD is trained on natural audio, and
bandpass/DRC would change signal distribution outside its training data.
(Architecture decision Q1 in sprint plan.)

Implements the SceneDetector Protocol via inheritance from AuditokSceneDetector.
Extracted from DynamicSceneDetector (Phase 2 of Sprint 3 refactoring).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .auditok_backend import AuditokSceneConfig, AuditokSceneDetector
from .base import SceneDetectionResult

logger = logging.getLogger("whisperjav")


@dataclass
class SileroSceneConfig(AuditokSceneConfig):
    """
    Configuration for the Silero scene detector.

    Extends AuditokSceneConfig with Silero VAD parameters for Pass 2.
    Assistive processing is forced off (Silero doesn't benefit from it).
    """
    # Silero VAD parameters (conservative defaults for scene detection)
    silero_threshold: float = 0.08
    silero_neg_threshold: float = 0.15
    silero_min_silence_ms: int = 1500
    silero_min_speech_ms: int = 100
    silero_max_speech_s: float = 600.0
    silero_min_silence_at_max: int = 500
    silero_speech_pad_ms: int = 200

    def __post_init__(self):
        """Derive defaults and force assist_processing off."""
        super().__post_init__()
        # Silero does NOT benefit from bandpass/DRC (Q1)
        self.assist_processing = False


class SileroSceneDetector(AuditokSceneDetector):
    """
    Two-pass scene detector: auditok Pass 1 + Silero VAD Pass 2.

    Inherits the full two-pass flow from AuditokSceneDetector.
    Overrides _detect_pass2() to use Silero VAD instead of auditok
    for fine-grained speech boundary detection.

    The Silero model is loaded once at construction time and reused
    for all regions. Per-region resampling to 16kHz is done as needed
    (trades CPU for memory — correct for large file handling).
    """

    def __init__(
        self,
        config: Optional[SileroSceneConfig] = None,
        **kwargs,
    ):
        """
        Initialize the Silero scene detector.

        Args:
            config: Typed configuration. If None, built from kwargs.
            **kwargs: Legacy DynamicSceneDetector-style parameters.
        """
        if config is not None:
            silero_config = config
        else:
            silero_config = self._build_silero_config(kwargs)

        # Initialize parent (AuditokSceneDetector) with the config
        super().__init__(config=silero_config)

        # Store Silero-specific config reference
        self._silero_config = silero_config

        # Load Silero VAD model
        self._load_silero_model()

        # VAD segments collected during detection (for metadata)
        self._vad_segments: List[Dict[str, Any]] = []

    @staticmethod
    def _build_silero_config(kwargs: dict) -> SileroSceneConfig:
        """Build SileroSceneConfig from legacy kwargs."""
        # Build base auditok config first
        base = AuditokSceneDetector._build_config_from_kwargs(kwargs)

        return SileroSceneConfig(
            # Copy all auditok fields
            max_duration=base.max_duration,
            min_duration=base.min_duration,
            pass1_min_duration=base.pass1_min_duration,
            pass1_max_duration=base.pass1_max_duration,
            pass1_max_silence=base.pass1_max_silence,
            pass1_energy_threshold=base.pass1_energy_threshold,
            pass2_min_duration=base.pass2_min_duration,
            pass2_max_duration=base.pass2_max_duration,
            pass2_max_silence=base.pass2_max_silence,
            pass2_energy_threshold=base.pass2_energy_threshold,
            brute_force_fallback=base.brute_force_fallback,
            brute_force_chunk_s=base.brute_force_chunk_s,
            pad_edges_s=base.pad_edges_s,
            verbose_summary=base.verbose_summary,
            force_mono=base.force_mono,
            # Silero-specific parameters
            silero_threshold=float(kwargs.get("silero_threshold", 0.08)),
            silero_neg_threshold=float(kwargs.get("silero_neg_threshold", 0.15)),
            silero_min_silence_ms=int(kwargs.get("silero_min_silence_ms", 1500)),
            silero_min_speech_ms=int(kwargs.get("silero_min_speech_ms", 100)),
            silero_max_speech_s=float(kwargs.get("silero_max_speech_s", 600.0)),
            silero_min_silence_at_max=int(kwargs.get("silero_min_silence_at_max", 500)),
            silero_speech_pad_ms=int(kwargs.get("silero_speech_pad_ms", 200)),
        )

    def _load_silero_model(self) -> None:
        """Load Silero VAD model (pip-installed silero-vad package v6.x)."""
        logger.info("Loading Silero VAD (pip package) for scene detection...")
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps

            logger.info(
                "Silero VAD (pip package v6.x) loaded successfully "
                "for scene detection"
            )
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise RuntimeError(
                "Silero VAD method selected but model failed to load. "
                "Fall back to --scene-detection-method auditok"
            ) from e

    @property
    def name(self) -> str:
        return "silero"

    @property
    def display_name(self) -> str:
        return "Silero VAD (Two-Pass)"

    def detect_scenes(self, audio_path, output_dir, media_basename, **kwargs):
        """Override to clear VAD segments before each run."""
        self._vad_segments = []
        result = super().detect_scenes(
            audio_path, output_dir, media_basename, **kwargs
        )
        # Attach VAD segment data to the result
        result.vad_segments = self._vad_segments if self._vad_segments else None
        return result

    def _detect_pass2(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        region_start: float,
        region_end: float,
        region_duration: float,
    ) -> list:
        """
        Pass 2: Use Silero VAD for fine scene splitting.

        Per-region resampling to 16kHz is the correct approach:
        - Most regions don't reach Pass 2 (only oversized ones)
        - Region audio is bounded (30-60s typically)
        - Memory efficiency > CPU for large files
        """
        import torch
        import librosa

        det_start = int(region_start * sample_rate)
        det_end = int(region_end * sample_rate)
        region_audio = audio_data[det_start:det_end]

        # Resample to 16kHz for Silero VAD (if needed)
        if sample_rate != 16000:
            logger.debug(
                f"Resampling {region_duration:.1f}s region from "
                f"{sample_rate}Hz to 16kHz for Silero"
            )
            audio_16k = librosa.resample(
                region_audio, orig_sr=sample_rate, target_sr=16000
            )
        else:
            audio_16k = region_audio

        audio_tensor = torch.FloatTensor(audio_16k)

        # Run Silero VAD with error handling for graceful degradation
        cfg = self._silero_config
        try:
            speech_timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=16000,
                threshold=cfg.silero_threshold,
                neg_threshold=cfg.silero_neg_threshold,
                min_silence_duration_ms=cfg.silero_min_silence_ms,
                min_speech_duration_ms=cfg.silero_min_speech_ms,
                max_speech_duration_s=cfg.silero_max_speech_s,
                min_silence_at_max_speech=cfg.silero_min_silence_at_max,
                speech_pad_ms=cfg.silero_speech_pad_ms,
                return_seconds=True,
            )
        except Exception as e:
            logger.error(
                f"Silero VAD failed at region starting "
                f"{region_start:.2f}s: {e}"
            )
            return []  # Graceful degradation → triggers brute-force fallback

        # Convert Silero format to auditok-compatible format
        # Silero returns: [{'start': 0.5, 'end': 3.2}, ...]
        # Need objects with .start/.end (seconds, relative to region)
        class SileroRegion:
            """Wrapper for Silero timestamps in auditok-compatible format."""
            def __init__(self, start_sec, end_sec):
                self.start = start_sec
                self.end = end_sec

        sub_regions = [
            SileroRegion(seg["start"], seg["end"])
            for seg in speech_timestamps
        ]

        # Collect VAD segments with absolute timestamps for metadata
        for seg in speech_timestamps:
            self._vad_segments.append({
                "start_sec": round(region_start + seg["start"], 3),
                "end_sec": round(region_start + seg["end"], 3),
            })

        logger.debug(
            f"Silero VAD detected {len(sub_regions)} segments in Pass 2"
        )
        return sub_regions

    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Return detection parameters including Silero-specific ones."""
        params = super()._get_parameters_dict()
        cfg = self._silero_config
        params.update({
            "silero_threshold": cfg.silero_threshold,
            "silero_neg_threshold": cfg.silero_neg_threshold,
            "silero_min_silence_ms": cfg.silero_min_silence_ms,
            "silero_min_speech_ms": cfg.silero_min_speech_ms,
            "silero_max_speech_s": cfg.silero_max_speech_s,
            "silero_speech_pad_ms": cfg.silero_speech_pad_ms,
        })
        return params

    def cleanup(self) -> None:
        """Release Silero model and clear state."""
        super().cleanup()
        self._vad_model = None
        self._vad_segments = []

    def __repr__(self) -> str:
        cfg = self._silero_config
        return (
            f"SileroSceneDetector(max_dur={cfg.max_duration}s, "
            f"silero_threshold={cfg.silero_threshold})"
        )
