"""
Auditok-based scene detector with two-pass strategy.

Pass 1 (Coarse): Find natural chapter boundaries via long silences
Pass 2 (Fine): Chunk each chapter to consumer's max_duration using auditok

Optional assistive processing (bandpass + DRC) for Pass 2 on challenging audio.

Implements the SceneDetector Protocol.
Extracted from DynamicSceneDetector (Phase 2 of Sprint 3 refactoring).
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .base import SceneDetectionError, SceneDetectionResult, SceneInfo
from .utils import brute_force_split, save_scene_wav

logger = logging.getLogger("whisperjav")

# Lazy import auditok at module level with availability check
try:
    import auditok
    AUDITOK_AVAILABLE = True
except ImportError:
    AUDITOK_AVAILABLE = False


@dataclass
class AuditokSceneConfig:
    """
    Typed configuration for the Auditok scene detector.

    Organized by purpose:
    - Core segment bounds (what the consumer needs)
    - Pass 1 parameters (chapter discovery)
    - Pass 2 parameters (consumer-driven chunking)
    - Assist processing (bandpass + DRC for Pass 2)
    - Fallback behavior
    - Edge handling
    """
    # Core segment bounds
    max_duration: float = 29.0
    min_duration: float = 0.3

    # Pass 1: Coarse — find natural chapter boundaries
    pass1_min_duration: float = 0.3
    pass1_max_duration: float = 2700.0
    pass1_max_silence: float = 1.8
    pass1_energy_threshold: int = 38

    # Pass 2: Fine — chunk chapters to consumer's max_duration
    pass2_min_duration: float = 0.3
    pass2_max_duration: Optional[float] = None  # Derived from max_duration - 1.0
    pass2_max_silence: float = 0.94
    pass2_energy_threshold: int = 50

    # Assist processing (bandpass + DRC for Pass 2 detection)
    assist_processing: bool = False
    bandpass_low_hz: int = 200
    bandpass_high_hz: int = 4000
    drc_threshold_db: float = -24.0
    drc_ratio: float = 4.0
    drc_attack_ms: float = 5.0
    drc_release_ms: float = 100.0
    skip_assist_on_loud_dbfs: float = -5.0

    # Fallback
    brute_force_fallback: bool = True
    brute_force_chunk_s: Optional[float] = None  # Defaults to max_duration

    # Edge handling
    pad_edges_s: float = 0.0

    # Output
    verbose_summary: bool = True
    force_mono: bool = True

    def __post_init__(self):
        """Derive defaults that depend on other fields."""
        if self.pass2_max_duration is None:
            self.pass2_max_duration = max(self.max_duration - 1.0, self.min_duration)
        if self.brute_force_chunk_s is None:
            self.brute_force_chunk_s = self.max_duration


class AuditokSceneDetector:
    """
    Two-pass scene detector using auditok for both passes.

    Pass 1 finds natural chapter boundaries via long silences.
    Pass 2 chunks each chapter to the consumer's max_duration.

    If Pass 2 finds no sub-regions, falls back to brute-force splitting.

    Fixes from DynamicSceneDetector:
    - Counting bug: counts only scenes that survive min_duration filter
    - Silent failure: raises SceneDetectionError on audio load failure
    - Mutable state: cleared at start of each detect_scenes() call
    """

    def __init__(
        self,
        config: Optional[AuditokSceneConfig] = None,
        **kwargs,
    ):
        """
        Initialize the auditok scene detector.

        Args:
            config: Typed configuration. If None, built from kwargs.
            **kwargs: Legacy DynamicSceneDetector-style parameters.
                      Supports _s suffix aliases (e.g., max_duration_s).
        """
        if not AUDITOK_AVAILABLE:
            raise ImportError("The 'auditok' library is required for AuditokSceneDetector.")

        if config is not None:
            self._config = config
        else:
            self._config = self._build_config_from_kwargs(kwargs)

        # Detection result (populated by detect_scenes, cleared each call)
        self._last_result: Optional[SceneDetectionResult] = None

        logger.debug(
            f"AuditokSceneDetector cfg: "
            f"max_dur={self._config.max_duration}s, min_dur={self._config.min_duration}s, "
            f"pass1(max_dur={self._config.pass1_max_duration}, "
            f"max_sil={self._config.pass1_max_silence}, "
            f"thr={self._config.pass1_energy_threshold}), "
            f"pass2(max_dur={self._config.pass2_max_duration}, "
            f"max_sil={self._config.pass2_max_silence}, "
            f"thr={self._config.pass2_energy_threshold}), "
            f"assist={self._config.assist_processing}"
        )

    @staticmethod
    def _build_config_from_kwargs(kwargs: dict) -> AuditokSceneConfig:
        """
        Build typed config from legacy DynamicSceneDetector-style kwargs.

        Handles _s suffix aliases (e.g., max_duration_s → max_duration).
        Unknown keys are silently ignored for backward compatibility.
        """
        def _get(key: str, default):
            """Get value, checking _s suffixed alias first."""
            s_key = f"{key}_s" if not key.endswith("_s") else key
            base_key = key.rstrip("_s") if key.endswith("_s") else key
            # _s suffix takes precedence (more specific)
            if s_key in kwargs:
                return type(default)(kwargs[s_key])
            if base_key in kwargs:
                return type(default)(kwargs[base_key])
            return default

        max_duration = float(kwargs.get("max_duration_s", kwargs.get("max_duration", 29.0)))
        min_duration = float(kwargs.get("min_duration_s", kwargs.get("min_duration", 0.3)))

        # Pass 1 — fall back to legacy top-level names
        legacy_max_silence = float(kwargs.get("max_silence", 1.8))
        legacy_energy_threshold = int(kwargs.get("energy_threshold", 38))

        pass1_min_duration = float(kwargs.get(
            "pass1_min_duration_s",
            kwargs.get("pass1_min_duration", 0.3)
        ))
        pass1_max_duration = float(kwargs.get(
            "pass1_max_duration_s",
            kwargs.get("pass1_max_duration", 2700.0)
        ))
        pass1_max_silence = float(kwargs.get(
            "pass1_max_silence_s",
            kwargs.get("pass1_max_silence", legacy_max_silence)
        ))
        pass1_energy_threshold = int(kwargs.get(
            "pass1_energy_threshold", legacy_energy_threshold
        ))

        # Pass 2
        pass2_min_duration = float(kwargs.get(
            "pass2_min_duration_s",
            kwargs.get("pass2_min_duration", 0.3)
        ))
        pass2_max_duration_raw = kwargs.get(
            "pass2_max_duration_s",
            kwargs.get("pass2_max_duration", None)
        )
        pass2_max_duration = float(pass2_max_duration_raw) if pass2_max_duration_raw is not None else None
        pass2_max_silence = float(kwargs.get(
            "pass2_max_silence_s",
            kwargs.get("pass2_max_silence", 0.94)
        ))
        pass2_energy_threshold = int(kwargs.get("pass2_energy_threshold", 50))

        # Brute force
        brute_force_chunk_raw = kwargs.get("brute_force_chunk_s", None)
        brute_force_chunk_s = float(brute_force_chunk_raw) if brute_force_chunk_raw is not None else None

        return AuditokSceneConfig(
            max_duration=max_duration,
            min_duration=min_duration,
            pass1_min_duration=pass1_min_duration,
            pass1_max_duration=pass1_max_duration,
            pass1_max_silence=pass1_max_silence,
            pass1_energy_threshold=pass1_energy_threshold,
            pass2_min_duration=pass2_min_duration,
            pass2_max_duration=pass2_max_duration,
            pass2_max_silence=pass2_max_silence,
            pass2_energy_threshold=pass2_energy_threshold,
            assist_processing=bool(kwargs.get("assist_processing", False)),
            bandpass_low_hz=int(kwargs.get("bandpass_low_hz", 200)),
            bandpass_high_hz=int(kwargs.get("bandpass_high_hz", 4000)),
            drc_threshold_db=float(kwargs.get("drc_threshold_db", -24.0)),
            drc_ratio=float(kwargs.get("drc_ratio", 4.0)),
            drc_attack_ms=float(kwargs.get("drc_attack_ms", 5.0)),
            drc_release_ms=float(kwargs.get("drc_release_ms", 100.0)),
            skip_assist_on_loud_dbfs=float(kwargs.get("skip_assist_on_loud_dbfs", -5.0)),
            brute_force_fallback=bool(kwargs.get("brute_force_fallback", True)),
            brute_force_chunk_s=brute_force_chunk_s,
            pad_edges_s=float(kwargs.get("pad_edges_s", 0.0)),
            verbose_summary=bool(kwargs.get("verbose_summary", True)),
            force_mono=bool(kwargs.get("force_mono", True)),
        )

    @property
    def name(self) -> str:
        return "auditok"

    @property
    def display_name(self) -> str:
        return "Auditok (Silence-Based)"

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        **kwargs,
    ) -> SceneDetectionResult:
        """
        Detect scenes using two-pass auditok strategy.

        Pass 1: Find coarse chapter boundaries via long silences
        Pass 2: Chunk oversized chapters to consumer's max_duration

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save scene WAV files
            media_basename: Base name for output files

        Returns:
            SceneDetectionResult with detected scenes and metadata

        Raises:
            SceneDetectionError: If audio file cannot be loaded
        """
        start_time = time.time()
        cfg = self._config

        logger.info(f"Starting auditok scene detection for: {audio_path}")

        # --- Load audio ---
        audio_data, sample_rate = self._load_audio(audio_path)
        total_duration = len(audio_data) / sample_rate

        # --- Pass 1: Coarse chapter detection ---
        story_lines = self._detect_pass1(audio_data, sample_rate, total_duration)

        # Capture coarse boundaries for metadata
        coarse_boundaries = [
            {
                "scene_index": idx,
                "start_time_seconds": round(region.start, 3),
                "end_time_seconds": round(region.end, 3),
                "duration_seconds": round(region.end - region.start, 3),
            }
            for idx, region in enumerate(story_lines)
        ]

        if not story_lines:
            # Valid result: no speech found (not an error)
            self._log_empty_pass1(audio_data, sample_rate, total_duration)
            self._last_result = SceneDetectionResult(
                scenes=[],
                method=self.name,
                audio_duration_sec=total_duration,
                parameters=self._get_parameters_dict(),
                processing_time_sec=time.time() - start_time,
                coarse_boundaries=coarse_boundaries,
            )
            return self._last_result

        if len(story_lines) > 50:
            logger.info(
                f"Pass 2: Processing {len(story_lines)} story lines "
                f"(this may take a moment for long files)..."
            )

        # --- Pass 2: Fine chunking ---
        output_dir.mkdir(parents=True, exist_ok=True)
        scenes, counters = self._process_story_lines(
            story_lines, audio_data, sample_rate, total_duration,
            output_dir, media_basename,
        )

        # --- Summary ---
        if cfg.verbose_summary:
            self._log_summary(
                story_lines, scenes, counters, total_duration
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Detected and saved {len(scenes)} final scenes "
            f"in {processing_time:.1f}s."
        )

        self._last_result = SceneDetectionResult(
            scenes=scenes,
            method=self.name,
            audio_duration_sec=total_duration,
            parameters=self._get_parameters_dict(),
            processing_time_sec=processing_time,
            coarse_boundaries=coarse_boundaries,
        )
        return self._last_result

    def cleanup(self) -> None:
        """Release resources."""
        self._last_result = None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio from disk. Raises SceneDetectionError on failure.

        Returns (audio_data, sample_rate) at native sample rate.
        No resampling — auditok handles any sample rate natively.
        """
        try:
            from whisperjav.modules.scene_detection import load_audio_unified
            audio_data, sample_rate = load_audio_unified(
                audio_path, target_sr=None, force_mono=self._config.force_mono
            )
        except Exception as e:
            raise SceneDetectionError(
                f"Failed to load audio file {audio_path}: {e}"
            ) from e

        duration = len(audio_data) / sample_rate
        logger.info(
            f"Audio loaded: {duration:.1f}s @ {sample_rate}Hz "
            f"(native SR, no resample needed)"
        )

        # Warn about large files
        duration_hours = duration / 3600
        if duration_hours > 1.5:
            estimated_gb = audio_data.nbytes / (1024 ** 3)
            logger.warning(
                f"Large file detected: {duration_hours:.1f} hours "
                f"(~{estimated_gb:.1f} GB memory). "
                f"Processing may take several minutes. This is normal."
            )

        return audio_data, sample_rate

    def _detect_pass1(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> list:
        """
        Pass 1: Find coarse chapter boundaries via long silences.

        Returns list of auditok region objects.
        """
        cfg = self._config
        logger.info(
            f"Pass 1: Starting auditok scene detection on "
            f"{total_duration:.1f}s audio @ {sample_rate}Hz..."
        )

        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        pass1_params = {
            "sampling_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
            "min_dur": cfg.pass1_min_duration,
            "max_dur": cfg.pass1_max_duration,
            "max_silence": min(total_duration * 0.95, cfg.pass1_max_silence),
            "energy_threshold": cfg.pass1_energy_threshold,
            "drop_trailing_silence": True,
        }

        story_lines = list(auditok.split(audio_bytes, **pass1_params))
        logger.info(f"Pass 1: Found {len(story_lines)} coarse story line(s).")
        return story_lines

    def _process_story_lines(
        self,
        story_lines: list,
        audio_data: np.ndarray,
        sample_rate: int,
        total_duration: float,
        output_dir: Path,
        media_basename: str,
    ) -> Tuple[List[SceneInfo], Dict[str, int]]:
        """
        Process each story line: direct save if small enough, else Pass 2 split.

        Returns (scenes, counters) where counters tracks direct/granular/brute counts.
        """
        cfg = self._config
        scenes: List[SceneInfo] = []
        scene_idx = 0
        counters = {"direct": 0, "granular": 0, "brute_force": 0}

        total_storylines = len(story_lines)
        last_logged_pct = -1

        for region_idx, region in enumerate(story_lines):
            # Progress logging
            current_pct = int((region_idx / max(total_storylines, 1)) * 100)
            if total_storylines > 20 and current_pct >= last_logged_pct + 25:
                logger.info(
                    f"Pass 2: Processing story line "
                    f"{region_idx + 1}/{total_storylines} ({current_pct}%)"
                )
                last_logged_pct = current_pct

            region_start = region.start
            region_end = region.end
            region_duration = region_end - region_start

            # Direct save if fits within max_duration
            if cfg.min_duration <= region_duration <= cfg.max_duration:
                s, e = self._clamp_with_pad(region_start, region_end, total_duration)
                start_sample = int(s * sample_rate)
                end_sample = int(e * sample_rate)
                scene_path = save_scene_wav(
                    audio_data[start_sample:end_sample],
                    sample_rate, scene_idx, output_dir, media_basename,
                )
                scenes.append(SceneInfo(
                    start_sec=s, end_sec=e, scene_path=scene_path,
                    detection_pass=1,
                ))
                scene_idx += 1
                counters["direct"] += 1
                continue

            # Pass 2: Split oversized region
            sub_regions = self._detect_pass2(
                audio_data, sample_rate, region_start, region_end, region_duration
            )

            if sub_regions:
                logger.debug(
                    f"Pass 2 split region into {len(sub_regions)} sub-scenes."
                )
                for sub in sub_regions:
                    sub_start = region_start + sub.start
                    sub_end = region_start + sub.end
                    sub_dur = sub_end - sub_start
                    if sub_dur < cfg.min_duration:
                        continue
                    s, e = self._clamp_with_pad(sub_start, sub_end, total_duration)
                    start_sample = int(s * sample_rate)
                    end_sample = int(e * sample_rate)
                    scene_path = save_scene_wav(
                        audio_data[start_sample:end_sample],
                        sample_rate, scene_idx, output_dir, media_basename,
                    )
                    scenes.append(SceneInfo(
                        start_sec=s, end_sec=e, scene_path=scene_path,
                        detection_pass=2,
                    ))
                    scene_idx += 1
                    counters["granular"] += 1  # Count AFTER min_duration filter
            else:
                # Brute-force fallback
                if not cfg.brute_force_fallback:
                    logger.warning(
                        f"Pass 2 found no sub-regions in region {region_idx}; "
                        f"skipping fallback."
                    )
                    continue

                logger.warning(
                    f"Pass 2 found no sub-regions in region {region_idx}, "
                    f"using brute-force splitting."
                )
                bf_scenes = brute_force_split(
                    region_start, region_end,
                    cfg.brute_force_chunk_s, cfg.min_duration,
                )
                for bf in bf_scenes:
                    s, e = self._clamp_with_pad(
                        bf.start_sec, bf.end_sec, total_duration
                    )
                    start_sample = int(s * sample_rate)
                    end_sample = int(e * sample_rate)
                    scene_path = save_scene_wav(
                        audio_data[start_sample:end_sample],
                        sample_rate, scene_idx, output_dir, media_basename,
                    )
                    scenes.append(SceneInfo(
                        start_sec=s, end_sec=e, scene_path=scene_path,
                        detection_pass=2,
                        metadata={"split_method": "brute_force"},
                    ))
                    scene_idx += 1
                    counters["brute_force"] += 1

        return scenes, counters

    def _detect_pass2(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        region_start: float,
        region_end: float,
        region_duration: float,
    ) -> list:
        """
        Pass 2: Find fine sub-regions within a coarse chapter using auditok.

        Subclasses (SileroSceneDetector) override this method for
        different Pass 2 strategies.

        Args:
            audio_data: Full audio array (at native sample rate)
            sample_rate: Audio sample rate
            region_start: Start of the coarse region (seconds)
            region_end: End of the coarse region (seconds)
            region_duration: Duration of the region (seconds)

        Returns:
            List of objects with .start and .end attributes (seconds,
            relative to region start)
        """
        cfg = self._config

        det_start = int(region_start * sample_rate)
        det_end = int(region_end * sample_rate)
        region_audio = audio_data[det_start:det_end]

        # Optional assistive processing (bandpass + DRC)
        if cfg.assist_processing:
            region_audio = self._apply_assistive_processing(
                region_audio, sample_rate
            )

        region_bytes = (region_audio * 32767).astype(np.int16).tobytes()
        pass2_params = {
            "sampling_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
            "min_dur": cfg.pass2_min_duration,
            "max_dur": cfg.pass2_max_duration,
            "max_silence": min(region_duration * 0.95, cfg.pass2_max_silence),
            "energy_threshold": cfg.pass2_energy_threshold,
            "drop_trailing_silence": True,
        }

        return list(auditok.split(region_bytes, **pass2_params))

    def _apply_assistive_processing(
        self, audio_chunk: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Apply bandpass filter and DRC to enhance speech detection for Pass 2.

        Only used when assist_processing=True. Silero backends should NOT
        use this (Silero is trained on natural audio — Q1 in sprint plan).
        """
        from pydub import AudioSegment
        from pydub.effects import compress_dynamic_range
        from scipy.signal import butter, filtfilt

        cfg = self._config

        # Skip on loud chunks
        peak_dbfs = 20 * np.log10(np.max(np.abs(audio_chunk)) + 1e-9)
        if peak_dbfs > cfg.skip_assist_on_loud_dbfs:
            logger.debug(
                f"Peak {peak_dbfs:.2f} dBFS >= "
                f"{cfg.skip_assist_on_loud_dbfs:.2f} dBFS; skipping assist."
            )
            return audio_chunk

        # Bandpass filter
        nyquist = 0.5 * sample_rate
        low = max(10.0, float(cfg.bandpass_low_hz)) / nyquist
        high = min(nyquist - 1.0, float(cfg.bandpass_high_hz)) / nyquist
        high = min(max(high, low + 1e-4), 0.999)
        b, a = butter(5, [low, high], btype="band")
        filtered_audio = filtfilt(b, a, audio_chunk.copy())

        # Dynamic range compression via pydub
        audio_segment = AudioSegment(
            (filtered_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
        compressed = compress_dynamic_range(
            audio_segment,
            threshold=cfg.drc_threshold_db,
            ratio=cfg.drc_ratio,
            attack=cfg.drc_attack_ms,
            release=cfg.drc_release_ms,
        )
        processed = (
            np.array(compressed.get_array_of_samples()).astype(np.float32) / 32768.0
        )

        # Safety clip
        if np.max(np.abs(processed)) > 1.0:
            processed = np.clip(processed, -1.0, 1.0)

        return processed

    def _clamp_with_pad(
        self, start: float, end: float, total_duration: float
    ) -> Tuple[float, float]:
        """Apply edge padding and clamp to audio boundaries."""
        pad = self._config.pad_edges_s
        s = max(0.0, start - pad)
        e = min(total_duration, end + pad)
        if e < s:
            e = s
        return s, e

    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Return detection parameters for metadata."""
        cfg = self._config
        return {
            "max_duration": cfg.max_duration,
            "min_duration": cfg.min_duration,
            "pass1_max_silence": cfg.pass1_max_silence,
            "pass1_energy_threshold": cfg.pass1_energy_threshold,
            "pass2_max_duration": cfg.pass2_max_duration,
            "pass2_max_silence": cfg.pass2_max_silence,
            "pass2_energy_threshold": cfg.pass2_energy_threshold,
            "assist_processing": cfg.assist_processing,
            "brute_force_fallback": cfg.brute_force_fallback,
        }

    def _log_empty_pass1(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> None:
        """Diagnostic logging when Pass 1 finds no story lines."""
        cfg = self._config
        peak = float(np.max(np.abs(audio_data)))
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        logger.warning(
            f"Pass 1 found NO story lines!\n"
            f"  Audio stats: duration={total_duration:.1f}s, "
            f"peak={peak:.4f}, rms={rms:.6f}\n"
            f"  Params: energy_threshold={cfg.pass1_energy_threshold}, "
            f"min_dur={cfg.pass1_min_duration:.1f}s, "
            f"max_silence={cfg.pass1_max_silence:.1f}s\n"
            f"  Suggestions:\n"
            f"    - If peak/rms are very low, audio may be silent\n"
            f"    - Try lowering energy_threshold "
            f"(current: {cfg.pass1_energy_threshold})\n"
            f"    - Check if audio extraction succeeded"
        )

    def _log_summary(
        self,
        story_lines: list,
        scenes: List[SceneInfo],
        counters: Dict[str, int],
        total_duration: float,
    ) -> None:
        """Log detection summary with statistics."""
        durations = [s.duration_sec for s in scenes]

        if durations:
            sorted_d = sorted(durations)
            n = len(sorted_d)
            median = (
                (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
                if n % 2 == 0
                else sorted_d[n // 2]
            )
            total_scene_dur = sum(durations)

            summary_lines = [
                "", "=" * 50,
                f"{self.display_name} Scene Detection Summary",
                "=" * 50,
                f"Total Story Lines Found: {len(story_lines)}",
                f" - Segments saved directly: {counters['direct']}",
                f" - Segments from granular split (Pass 2): {counters['granular']}",
                f" - Segments from brute-force split: {counters['brute_force']}",
                "-" * 50,
                f"Total Final Scenes Saved: {len(scenes)}",
                f"Scene Duration Statistics:",
                f" - Shortest: {min(durations):.2f}s",
                f" - Longest: {max(durations):.2f}s",
                f" - Mean: {total_scene_dur / n:.2f}s",
                f" - Median: {median:.2f}s",
                f" - Total: {total_scene_dur:.1f}s ({total_scene_dur / 60:.1f}m)",
                "=" * 50, "",
            ]
        else:
            summary_lines = [
                "", "=" * 50,
                f"{self.display_name} Scene Detection Summary",
                "=" * 50,
                f"Total Story Lines Found: {len(story_lines)}",
                f" - Segments saved directly: {counters['direct']}",
                f" - Segments from granular split (Pass 2): {counters['granular']}",
                f" - Segments from brute-force split: {counters['brute_force']}",
                "-" * 50,
                f"Total Final Scenes Saved: {len(scenes)}",
                "=" * 50, "",
            ]

        logger.info("\n".join(summary_lines))

    def __repr__(self) -> str:
        cfg = self._config
        return (
            f"AuditokSceneDetector(max_dur={cfg.max_duration}s, "
            f"min_dur={cfg.min_duration}s, "
            f"assist={cfg.assist_processing})"
        )
