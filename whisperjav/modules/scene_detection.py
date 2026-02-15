#!/usr/bin/env python3
"""
Audio scene detection module using a multi-feature, adaptive approach.
This version incorporates a two-pass strategy, Auditok coarse, STFT, DRC, onset-based
recovery, and metadata output.

==============================================================================
Audio scene detection using a two-pass strategy:
  Pass 1 (Coarse): Auditok finds natural chapter boundaries via long silences
  Pass 2 (Fine):   Auditok or Silero chunks each chapter to consumer's max_duration

Active class: DynamicSceneDetector (supports methods: auditok, silero, semantic)
Utility: load_audio_unified() — audio loading for all backends

UPSTREAM USAGE:
- balanced_pipeline.py, fast_pipeline.py, fidelity_pipeline.py
- kotoba_faster_whisper_pipeline.py, transformers_pipeline.py
- qwen_pipeline.py, ensemble/pass_worker.py

Last updated: 2026-02-14 (Phase 0 cleanup: removed legacy SceneDetector,
AdaptiveSceneDetector, analyze_scene_metadata)
==============================================================================
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party libraries
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.signal import butter, filtfilt

# Use the global WhisperJAV logger
from whisperjav.utils.logger import logger


def load_audio_unified(audio_path: Path, target_sr: Optional[int] = None, force_mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Unified audio loading function used by all scene detectors.
    Provides consistent audio format and sample rate handling.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (None = preserve original)
        force_mono: Convert to mono if stereo
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Use soundfile for consistent loading with optional resampling
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
        
        # Convert stereo to mono efficiently if needed
        if force_mono and audio_data.ndim > 1:
            logger.debug("Converting stereo to mono")
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if target sample rate specified
        if target_sr is not None and sample_rate != target_sr:
            logger.debug(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
            
        logger.debug(f"Audio loaded: {len(audio_data)/sample_rate:.1f}s @ {sample_rate}Hz, {'mono' if audio_data.ndim == 1 else 'stereo'}")
        return audio_data, sample_rate
        
    # H2: Specific exception handling with actionable guidance
    except sf.SoundFileError as e:
        file_size_mb = audio_path.stat().st_size / 1024 / 1024 if audio_path.exists() else 0
        logger.error(
            f"SoundFile cannot read {audio_path}: {e}\n"
            f"  File size: {file_size_mb:.1f} MB\n"
            f"  Suggestion: Verify file integrity, try converting with ffmpeg:\n"
            f"    ffmpeg -i \"{audio_path}\" -acodec pcm_s16le output.wav"
        )
        raise
    except MemoryError as e:
        file_size_mb = audio_path.stat().st_size / 1024 / 1024 if audio_path.exists() else 0
        logger.error(
            f"Out of memory loading {audio_path} ({file_size_mb:.1f} MB): {e}\n"
            f"  Suggestion: Close other applications or use smaller files"
        )
        raise
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {audio_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {type(e).__name__}: {e}")
        raise


# Try to import auditok for the coarse-splitting pass
AUDITOK_AVAILABLE = False
try:
    import auditok
    AUDITOK_AVAILABLE = True
except ImportError:
    logger.warning("Auditok not available. Scene detection functionality will be limited.")




class DynamicSceneDetector:
    """
    Handles audio scene detection using a two-pass strategy with optional
    assistive processing for improved accuracy on challenging audio.

    This class is designed as a drop-in replacement for the original SceneDetector,
    maintaining the same public interface while offering more robust internal logic.
    """

    def __init__(self,
                 # Method selection (NEW)
                 method: str = "auditok",
                 # Core final segment bounds
                 max_duration: float = 29.0,
                 min_duration: float = 0.3,
                 # Legacy pass-1 defaults (kept for BC if *_s not provided)
                 max_silence: float = 1.8,
                 energy_threshold: int = 38,
                 # Assistive detection flags
                 assist_processing: bool = False,
                 verbose_summary: bool = True,
                 # DEPRECATED (v1.8.0+, Issue #129): These parameters have no effect.
                 # Auditok handles any sample rate natively. Kept for backward compatibility.
                 target_sr: int = 16000,  # DEPRECATED - no longer used
                 force_mono: bool = True,
                 preserve_original_sr: bool = True,  # DEPRECATED - no longer used
                 # New: pass-1 controls
                 pass1_min_duration_s: Optional[float] = None,
                 pass1_max_duration_s: Optional[float] = None,
                 pass1_max_silence_s: Optional[float] = None,
                 pass1_energy_threshold: Optional[int] = None,
                 # New: pass-2 controls
                 pass2_min_duration_s: float = 0.3,
                 pass2_max_duration_s: Optional[float] = None,
                 pass2_max_silence_s: float = 0.94,
                 pass2_energy_threshold: int = 50,
                 # New: assist processing shaping
                 bandpass_low_hz: int = 200,
                 bandpass_high_hz: int = 4000,
                 drc_threshold_db: float = -24.0,
                 drc_ratio: float = 4.0,
                 drc_attack_ms: float = 5.0,
                 drc_release_ms: float = 100.0,
                 skip_assist_on_loud_dbfs: float = -5.0,
                 # New: fallback and shaping
                 brute_force_fallback: bool = True,
                 brute_force_chunk_s: Optional[float] = None,
                 pad_edges_s: float = 0.0,
                 fade_ms: int = 0,
                 # Accept *_s aliases via kwargs for config flexibility
                 **kwargs):
        """
        Initialize the dynamic audio scene detector.
        """
        if not AUDITOK_AVAILABLE:
            raise ImportError("The 'auditok' library is required for DynamicSceneDetector.")

        # Store method selection
        self.method = str(kwargs.get("method", method))

        # Allow *_s aliases from config (e.g., max_duration_s)
        max_duration = float(kwargs.get("max_duration_s", max_duration))
        min_duration = float(kwargs.get("min_duration_s", min_duration))

        # Core parameters
        self.max_duration = max_duration
        self.min_duration = min_duration

        # Detection and saving audio handling
        self.target_sr = int(kwargs.get("target_sr", target_sr))
        self.force_mono = bool(kwargs.get("force_mono", force_mono))
        self.preserve_original_sr = bool(kwargs.get("preserve_original_sr", preserve_original_sr))

        # DEPRECATED (v1.8.0+, Issue #129): target_sr and preserve_original_sr are no longer used.
        # Auditok handles any sample rate natively; Silero Pass 2 resamples internally.
        # These parameters are kept for backward compatibility but have no effect.
        if "target_sr" in kwargs or target_sr != 16000:
            logger.debug("Note: target_sr parameter is deprecated (v1.8.0+). Auditok uses native sample rate.")
        if "preserve_original_sr" in kwargs:
            logger.debug("Note: preserve_original_sr parameter is deprecated (v1.8.0+). Native SR always preserved.")

        # --- Pass 1 Parameters (Coarse Splitting) ---
        # Defaults map to legacy args if explicit *_s not provided
        self.pass1_min_duration = float(kwargs.get("pass1_min_duration_s", pass1_min_duration_s if pass1_min_duration_s is not None else 0.3))
        self.pass1_max_duration = float(kwargs.get("pass1_max_duration_s", pass1_max_duration_s if pass1_max_duration_s is not None else 2700.0))
        self.pass1_max_silence = float(kwargs.get("pass1_max_silence_s", pass1_max_silence_s if pass1_max_silence_s is not None else max_silence))
        self.pass1_energy_threshold = int(kwargs.get("pass1_energy_threshold", pass1_energy_threshold if pass1_energy_threshold is not None else energy_threshold))

        # --- Pass 2 Parameters (Fine Splitting) ---
        # If pass2_max_duration_s not provided, derive from max_duration - 1.0
        derived_pass2_max = max(self.max_duration - 1.0, self.min_duration)
        self.pass2_min_duration = float(kwargs.get("pass2_min_duration_s", pass2_min_duration_s))
        self.pass2_max_duration = float(kwargs.get("pass2_max_duration_s", pass2_max_duration_s if pass2_max_duration_s is not None else derived_pass2_max))
        self.pass2_max_silence = float(kwargs.get("pass2_max_silence_s", pass2_max_silence_s))
        self.pass2_energy_threshold = int(kwargs.get("pass2_energy_threshold", pass2_energy_threshold))

        # --- Enhancement Flags and Shaping ---
        self.assist_processing = bool(kwargs.get("assist_processing", assist_processing))
        self.verbose_summary = bool(kwargs.get("verbose_summary", verbose_summary))

        self.bandpass_low_hz = int(kwargs.get("bandpass_low_hz", bandpass_low_hz))
        self.bandpass_high_hz = int(kwargs.get("bandpass_high_hz", bandpass_high_hz))
        self.drc_threshold_db = float(kwargs.get("drc_threshold_db", drc_threshold_db))
        self.drc_ratio = float(kwargs.get("drc_ratio", drc_ratio))
        self.drc_attack_ms = float(kwargs.get("drc_attack_ms", drc_attack_ms))
        self.drc_release_ms = float(kwargs.get("drc_release_ms", drc_release_ms))
        self.skip_assist_on_loud_dbfs = float(kwargs.get("skip_assist_on_loud_dbfs", skip_assist_on_loud_dbfs))

        self.brute_force_fallback = bool(kwargs.get("brute_force_fallback", brute_force_fallback))
        # default fallback chunk equals max_duration
        self.brute_force_chunk_s = float(kwargs.get("brute_force_chunk_s", brute_force_chunk_s if brute_force_chunk_s is not None else self.max_duration))
        self.pad_edges_s = float(kwargs.get("pad_edges_s", pad_edges_s))
        self.fade_ms = int(kwargs.get("fade_ms", fade_ms))

        # --- Data Contract: Scene metadata and VAD segments for visualization ---
        # Populated by detect_scenes() for external consumers (e.g., visualization tools)
        self.scenes_detected: List[Dict] = []  # Rich metadata per scene
        self.vad_segments: List[Dict] = []     # VAD speech regions (when using Silero)
        self.coarse_boundaries: List[Dict] = []  # Pass 1 coarse scene boundaries (before splitting)
        self.vad_method: Optional[str] = None  # 'silero' or None
        self.vad_params: Dict = {}             # VAD parameters used

        # Initialize Silero VAD if needed
        if self.method == "silero":
            self._init_silero_vad(kwargs)

        # Initialize Semantic Clustering adapter if needed (lazy loaded)
        self._semantic_adapter = None
        if self.method == "semantic":
            self._init_semantic_adapter(kwargs)

        logger.debug(
            "DynamicSceneDetector cfg: "
            f"method={self.method}, force_mono={self.force_mono}, "
            f"max_dur={self.max_duration}s, min_dur={self.min_duration}s, "
            f"pass1(max_dur={self.pass1_max_duration}, max_sil={self.pass1_max_silence}, thr={self.pass1_energy_threshold}), "
            f"pass2(max_dur={self.pass2_max_duration}, max_sil={self.pass2_max_silence}, thr={self.pass2_energy_threshold}), "
            f"assist={self.assist_processing}"
        )

    def _init_silero_vad(self, config: dict):
        """
        Initialize Silero VAD model and parameters for Pass 2.

        Uses pip-installed silero-vad package (v6.x) for scene detection.
        This is separate from the torch.hub v3.1 used by ASR in whisper_pro_asr.py.
        """
        # Read Silero-specific parameters from flat config structure
        # Default values are conservative for scene detection (not ultra-sensitive)
        self.silero_threshold = float(config.get('silero_threshold', 0.08))
        self.silero_neg_threshold = float(config.get('silero_neg_threshold', 0.15))
        self.silero_min_silence_ms = int(config.get('silero_min_silence_ms', 1500))
        self.silero_min_speech_ms = int(config.get('silero_min_speech_ms', 100))
        self.silero_max_speech_s = float(config.get('silero_max_speech_s', 600))
        self.silero_min_silence_at_max = int(config.get('silero_min_silence_at_max', 500))
        self.silero_speech_pad_ms = int(config.get('silero_speech_pad_ms', 200))

        # Load Silero VAD model (pip-installed version, latest v6.x)
        logger.info("Loading Silero VAD (pip package) for scene detection...")

        try:
            # Import pip-installed silero-vad package
            from silero_vad import load_silero_vad, get_speech_timestamps

            # Load the model (automatically uses latest version from pip)
            self.vad_model = load_silero_vad()

            # Store the get_speech_timestamps function
            self.get_speech_timestamps = get_speech_timestamps

            logger.info("Silero VAD (pip package v6.x) loaded successfully for scene detection")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise RuntimeError(
                "Silero VAD method selected but model failed to load. "
                "Fall back to --scene-detection-method auditok"
            ) from e

    def _init_semantic_adapter(self, config: dict):
        """
        Initialize Semantic Audio Clustering adapter for scene detection.

        Uses lazy import to avoid loading dependencies unless needed.
        """
        logger.info("Initializing Semantic Audio Clustering adapter...")

        try:
            # Lazy import to avoid loading dependencies unless method="semantic"
            from whisperjav.modules.scene_detection_backends.semantic_adapter import (
                SemanticClusteringAdapter,
                SemanticClusteringConfig,
            )

            # Build config from parameters
            # Use SemanticClusteringConfig defaults as fallbacks — NOT self.min_duration/max_duration
            # which hold auditok values (0.3/29.0). Semantic needs its own defaults (20.0/420.0).
            _sem_defaults = SemanticClusteringConfig()
            adapter_config = SemanticClusteringConfig(
                min_duration=float(config.get("min_duration", _sem_defaults.min_duration)),
                max_duration=float(config.get("max_duration", _sem_defaults.max_duration)),
                snap_window=float(config.get("snap_window", _sem_defaults.snap_window)),
                clustering_threshold=float(config.get("clustering_threshold", _sem_defaults.clustering_threshold)),
                sample_rate=int(config.get("sample_rate", _sem_defaults.sample_rate)),
                preserve_original_sr=bool(config.get("preserve_original_sr", _sem_defaults.preserve_original_sr)),
                visualize=bool(config.get("visualize", _sem_defaults.visualize)),
            )

            # Create adapter instance
            self._semantic_adapter = SemanticClusteringAdapter(
                config=adapter_config,
                preset=str(config.get("preset", "default")),
            )

            logger.info("Semantic Audio Clustering adapter initialized")

        except ImportError as e:
            logger.error(f"Failed to load Semantic Audio Clustering: {e}")
            raise RuntimeError(
                "Semantic method selected but dependencies not available. "
                "Ensure 'semantic_audio_clustering' module is installed. "
                "Fall back to --scene-detection-method auditok"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Semantic adapter: {e}")
            raise

    def _detect_pass2_silero(self, region_audio: np.ndarray, sr: int, region_start_sec: float) -> List:
        """
        Use Silero VAD for Pass 2 fine scene splitting.

        Per-region resampling is the correct approach for memory efficiency:
        - Most regions don't hit Silero (only regions > max_duration go through Pass 2)
        - Region audio is small (30-60 seconds bounded by pass1_max_duration)
        - Memory is the bigger constraint for long files on consumer hardware

        Returns: List of timestamp tuples compatible with Auditok format
        """
        import torch
        import librosa

        # Resample to 16kHz for Silero VAD (if needed)
        # Per-region resampling trades CPU for memory - correct for large file handling
        if sr != 16000:
            region_duration = len(region_audio) / sr
            logger.debug(f"Resampling {region_duration:.1f}s region from {sr}Hz to 16kHz for Silero")
            audio_16k = librosa.resample(region_audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = region_audio

        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio_16k)

        # M3: Wrap Silero VAD call in error handling for graceful degradation
        try:
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=16000,
                threshold=self.silero_threshold,
                neg_threshold=self.silero_neg_threshold,
                min_silence_duration_ms=self.silero_min_silence_ms,
                min_speech_duration_ms=self.silero_min_speech_ms,
                max_speech_duration_s=self.silero_max_speech_s,
                min_silence_at_max_speech=self.silero_min_silence_at_max,
                speech_pad_ms=self.silero_speech_pad_ms,
                return_seconds=True
            )
        except Exception as e:
            logger.error(f"Silero VAD failed at region starting {region_start_sec:.2f}s: {e}")
            return []  # Graceful degradation - empty region list triggers brute-force fallback

        # Convert Silero format to Auditok-compatible format
        # Silero returns: [{'start': 0.5, 'end': 3.2}, {'start': 5.0, 'end': 8.5}, ...]
        # Need to return objects with .start and .end attributes (in seconds relative to region)

        class SileroRegion:
            """Wrapper to make Silero timestamps compatible with Auditok interface"""
            def __init__(self, start_sec, end_sec):
                self.start = start_sec  # In seconds
                self.end = end_sec      # In seconds

        sub_regions = [
            SileroRegion(seg['start'], seg['end'])
            for seg in speech_timestamps
        ]

        # Data contract: Store VAD segments with absolute timestamps
        for seg in speech_timestamps:
            self.vad_segments.append({
                "start_sec": round(region_start_sec + seg['start'], 3),
                "end_sec": round(region_start_sec + seg['end'], 3),
            })

        logger.debug(f"Silero VAD detected {len(sub_regions)} segments in Pass 2")

        return sub_regions

    def _apply_assistive_processing(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies a bandpass filter and DRC to enhance speech detection for Pass 2.
        """
        # Skip on loud chunks
        peak_dbfs = 20 * np.log10(np.max(np.abs(audio_chunk)) + 1e-9)
        if peak_dbfs > self.skip_assist_on_loud_dbfs:
            logger.debug(f"Peak {peak_dbfs:.2f} dBFS >= {self.skip_assist_on_loud_dbfs:.2f} dBFS; skipping assist.")
            return audio_chunk

        # Bandpass
        nyquist = 0.5 * sample_rate
        low = max(10.0, float(self.bandpass_low_hz)) / nyquist
        high = min(nyquist - 1.0, float(self.bandpass_high_hz)) / nyquist
        # Guard band limits
        high = min(max(high, low + 1e-4), 0.999)
        b, a = butter(5, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio_chunk.copy())

        # DRC via pydub
        audio_segment = AudioSegment(
            (filtered_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        compressed_segment = compress_dynamic_range(
            audio_segment,
            threshold=self.drc_threshold_db,
            ratio=self.drc_ratio,
            attack=self.drc_attack_ms,
            release=self.drc_release_ms
        )
        processed_samples = np.array(compressed_segment.get_array_of_samples()).astype(np.float32) / 32768.0

        # Safety clip
        post_peak = np.max(np.abs(processed_samples))
        if post_peak > 1.0:
            processed_samples = np.clip(processed_samples, -1.0, 1.0)

        return processed_samples

    def _save_scene(self, audio_data: np.ndarray, sample_rate: int,
                    scene_idx: int, output_dir: Path,
                    media_basename: str) -> Path:
        """
        Save scene as PCM16 WAV.

        H1: Added validation to catch empty/corrupt scenes before writing.
        """
        # H1: Validate audio data before saving
        if audio_data is None or len(audio_data) == 0:
            logger.error(f"Scene {scene_idx}: Cannot save empty or None audio data")
            raise ValueError(f"Empty audio data for scene {scene_idx}")

        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename

        try:
            # M2: mkdir moved to detect_scenes() before the loop (not per-scene)
            sf.write(str(scene_path), audio_data, sample_rate, subtype='PCM_16')
        except PermissionError as e:
            logger.error(f"Scene {scene_idx}: Permission denied writing to {scene_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"Scene {scene_idx}: OS error writing {scene_path} (disk full?): {e}")
            raise

        return scene_path

    def _record_scene_metadata(
        self,
        scene_idx: int,
        start_sec: float,
        end_sec: float,
        detection_pass: int,
        scene_path: Optional[Path] = None,
    ) -> None:
        """
        Record rich scene metadata for the data contract.

        Args:
            scene_idx: Scene index (0-based)
            start_sec: Start time in seconds
            end_sec: End time in seconds
            detection_pass: 1 for Pass 1 (coarse/direct), 2 for Pass 2 (fine/split)
            scene_path: Optional path to the saved scene file
        """
        self.scenes_detected.append({
            "scene_index": scene_idx,
            "start_time_seconds": round(start_sec, 3),
            "end_time_seconds": round(end_sec, 3),
            "duration_seconds": round(end_sec - start_sec, 3),
            "detection_pass": detection_pass,
            "filename": scene_path.name if scene_path else None,
        })

    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str) -> List[Tuple[Path, float, float, float]]:
        """
        Splits audio into scenes using a robust two-pass approach.

        Side Effects:
            Populates self.scenes_detected with rich metadata per scene including:
            - scene_index, start_time_seconds, end_time_seconds, duration_seconds
            - detection_pass: 1 (coarse/direct) or 2 (fine/split)

            When using Silero VAD, also populates:
            - self.vad_segments: List of {'start_sec', 'end_sec'} speech regions
            - self.vad_method: 'silero'
            - self.vad_params: VAD parameters used
        """
        logger.info(f"Starting dynamic scene detection for: {audio_path}")

        # Reset data contract fields for fresh detection run
        self.scenes_detected = []
        self.vad_segments = []
        self.coarse_boundaries = []
        self.vad_method = self.method if self.method in ("silero", "semantic") else None
        if self.method == "silero":
            self.vad_params = {
                "threshold": self.silero_threshold,
                "neg_threshold": self.silero_neg_threshold,
                "min_silence_duration_ms": self.silero_min_silence_ms,
                "min_speech_duration_ms": self.silero_min_speech_ms,
                "max_speech_duration_s": self.silero_max_speech_s,
                "speech_pad_ms": self.silero_speech_pad_ms,
            }

        # --- Semantic Clustering: Delegate entirely to adapter ---
        if self.method == "semantic":
            if self._semantic_adapter is None:
                raise RuntimeError(
                    "Semantic adapter not initialized. "
                    "This should not happen if method='semantic' was set correctly."
                )

            # Delegate to semantic adapter
            scene_tuples = self._semantic_adapter.detect_scenes(
                audio_path=audio_path,
                output_dir=output_dir,
                media_basename=media_basename,
            )

            # Copy data contract fields from adapter to this instance
            self.scenes_detected = self._semantic_adapter.scenes_detected
            self.vad_segments = self._semantic_adapter.vad_segments
            self.coarse_boundaries = self._semantic_adapter.coarse_boundaries
            self.vad_method = self._semantic_adapter.vad_method
            self.vad_params = self._semantic_adapter.vad_params

            logger.info(f"Semantic scene detection complete: {len(scene_tuples)} scenes")
            return scene_tuples

        # --- Auditok/Silero: Original two-pass approach ---

        # Load audio once from disk - NO resampling needed for scene detection
        # Auditok handles any sample rate natively. Silero Pass 2 resamples internally.
        # This eliminates the 10-30 minute resample bottleneck for 2+ hour files (Issue #129).
        try:
            original_audio, orig_sr = load_audio_unified(audio_path, target_sr=None, force_mono=self.force_mono)
            # Use native sample rate - auditok is SR-agnostic, silero handles its own resampling
            detection_audio = original_audio
            det_sr = orig_sr
            det_duration = len(detection_audio) / det_sr
            logger.info(f"Audio loaded: {det_duration:.1f}s @ {det_sr}Hz (native SR, no resample needed)")

            # Warn user about long files (Issue #129 hardening)
            duration_hours = det_duration / 3600
            if duration_hours > 1.5:
                # M1: Use .nbytes for accurate memory estimation (accounts for actual dtype)
                estimated_gb = detection_audio.nbytes / (1024 ** 3)
                logger.warning(
                    f"Large file detected: {duration_hours:.1f} hours (~{estimated_gb:.1f} GB memory). "
                    f"Processing may take several minutes. This is normal."
                )

        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return []

        # Single audio stream for both detection and saving (no dual storage)
        # After Issue #129 fix, detection_audio IS original_audio at native SR
        save_audio = detection_audio
        save_sr = det_sr
        total_duration_s = len(save_audio) / save_sr  # seconds baseline

        # Pass 1 on detection audio
        logger.info(f"Pass 1: Starting auditok scene detection on {det_duration:.1f}s audio @ {det_sr}Hz...")
        det_bytes = (detection_audio * 32767).astype(np.int16).tobytes()
        pass1_params = {
            "sampling_rate": det_sr,
            "channels": 1,
            "sample_width": 2,
            "min_dur": self.pass1_min_duration,
            "max_dur": self.pass1_max_duration,
            "max_silence": min(det_duration * 0.95, self.pass1_max_silence),
            "energy_threshold": self.pass1_energy_threshold,
            "drop_trailing_silence": True
        }
        story_lines = list(auditok.split(det_bytes, **pass1_params))
        logger.info(f"Pass 1: Found {len(story_lines)} coarse story line(s).")

        # C3: Diagnostic logging for empty results (helps users understand why detection failed)
        if len(story_lines) == 0:
            peak = float(np.max(np.abs(detection_audio)))
            rms = float(np.sqrt(np.mean(detection_audio ** 2)))
            logger.warning(
                f"Pass 1 found NO story lines!\n"
                f"  Audio stats: duration={det_duration:.1f}s, peak={peak:.4f}, rms={rms:.6f}\n"
                f"  Params: energy_threshold={pass1_params['energy_threshold']}, "
                f"min_dur={pass1_params['min_dur']:.1f}s, max_silence={pass1_params['max_silence']:.1f}s\n"
                f"  Suggestions:\n"
                f"    - If peak/rms are very low, audio may be silent or nearly silent\n"
                f"    - Try lowering energy_threshold (current: {pass1_params['energy_threshold']})\n"
                f"    - Check if audio extraction succeeded (verify source file)"
            )
            return []

        if len(story_lines) > 50:
            logger.info(f"Pass 2: Processing {len(story_lines)} story lines (this may take a moment for long files)...")

        # Data contract: Capture coarse boundaries BEFORE any splitting (for visualization)
        for idx, region in enumerate(story_lines):
            self.coarse_boundaries.append({
                "scene_index": idx,
                "start_time_seconds": round(region.start, 3),
                "end_time_seconds": round(region.end, 3),
                "duration_seconds": round(region.end - region.start, 3),
            })

        final_scene_tuples: List[Tuple[Path, float, float, float]] = []
        scene_idx = 0

        # Counters
        storyline_direct_saves = 0
        granular_segments_count = 0
        brute_force_segments_count = 0

        # Helper to clamp with optional edge padding
        def clamp_with_pad(s: float, e: float) -> Tuple[float, float]:
            s_pad = max(0.0, s - self.pad_edges_s)
            e_pad = min(total_duration_s, e + self.pad_edges_s)
            if e_pad < s_pad:
                e_pad = s_pad
            return s_pad, e_pad

        # Progress tracking for long file processing
        total_storylines = len(story_lines)
        last_logged_pct = -1

        # M2: Create output directory once before the loop (not per-scene)
        output_dir.mkdir(parents=True, exist_ok=True)

        for region_idx, region in enumerate(story_lines):
            # Log progress at 0%, 25%, 50%, 75%, 100% (or every 15 for small batches)
            current_pct = int((region_idx / max(total_storylines, 1)) * 100)
            if total_storylines > 20 and current_pct >= last_logged_pct + 25:
                logger.info(f"Pass 2: Processing story line {region_idx + 1}/{total_storylines} ({current_pct}%)")
                last_logged_pct = current_pct
            elif total_storylines <= 20 and (region_idx % 5 == 0 or region_idx == total_storylines - 1):
                logger.debug(f"Processing story line {region_idx + 1}/{total_storylines}")

            region_start_sec = region.start
            region_end_sec = region.end
            region_duration = region_end_sec - region_start_sec

            # Direct save if short enough
            if region_duration <= self.max_duration and region_duration >= self.min_duration:
                s_sec, e_sec = clamp_with_pad(region_start_sec, region_end_sec)
                start_sample = int(s_sec * save_sr)
                end_sample = int(e_sec * save_sr)
                scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                # Data contract: Record Pass 1 scene (coarse/direct)
                self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=1, scene_path=scene_path)
                scene_idx += 1
                storyline_direct_saves += 1
                continue

            # Pass 2 on detection audio chunk
            det_start = int(region_start_sec * det_sr)
            det_end = int(region_end_sec * det_sr)
            region_audio_det = detection_audio[det_start:det_end]

            # Pass 2: Method-dependent detection
            if self.method == "silero":
                # Use Silero VAD for Pass 2
                sub_regions = self._detect_pass2_silero(
                    region_audio_det,
                    det_sr,
                    region_start_sec
                )
            else:
                # Existing Auditok Pass 2 (keep as-is)
                audio_for_detection = region_audio_det
                if self.assist_processing:
                    audio_for_detection = self._apply_assistive_processing(region_audio_det, det_sr)

                region_bytes_for_detection = (audio_for_detection * 32767).astype(np.int16).tobytes()
                pass2_params = {
                    "sampling_rate": det_sr,
                    "channels": 1,
                    "sample_width": 2,
                    "min_dur": self.pass2_min_duration,
                    "max_dur": self.pass2_max_duration,
                    "max_silence": min(region_duration * 0.95, self.pass2_max_silence),
                    "energy_threshold": self.pass2_energy_threshold,
                    "drop_trailing_silence": True
                }
                sub_regions = list(auditok.split(region_bytes_for_detection, **pass2_params))

            if sub_regions:
                granular_segments_count += len(sub_regions)
                logger.debug(f"Pass 2 split region into {len(sub_regions)} sub-scenes.")
                for sub in sub_regions:
                    sub_start = region_start_sec + sub.start
                    sub_end = region_start_sec + sub.end
                    sub_dur = sub_end - sub_start
                    if sub_dur < self.min_duration:
                        continue
                    s_sec, e_sec = clamp_with_pad(sub_start, sub_end)
                    start_sample = int(s_sec * save_sr)
                    end_sample = int(e_sec * save_sr)
                    scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                    final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                    # Data contract: Record Pass 2 scene (fine/split)
                    self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=2, scene_path=scene_path)
                    scene_idx += 1
            else:
                if not self.brute_force_fallback:
                    logger.warning(f"Pass 2 found no sub-regions in region {region_idx}; skipping fallback.")
                    continue
                logger.warning(f"Pass 2 found no sub-regions in region {region_idx}, using brute-force splitting.")
                chunk_len = self.brute_force_chunk_s
                num_chunks = int(np.ceil(region_duration / max(chunk_len, self.min_duration)))
                brute_force_segments_count += num_chunks
                for i in range(num_chunks):
                    sub_start = region_start_sec + i * chunk_len
                    sub_end = min(region_start_sec + (i + 1) * chunk_len, region_end_sec)
                    sub_dur = sub_end - sub_start
                    if sub_dur < self.min_duration:
                        continue
                    s_sec, e_sec = clamp_with_pad(sub_start, sub_end)
                    start_sample = int(s_sec * save_sr)
                    end_sample = int(e_sec * save_sr)
                    scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                    final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                    # Data contract: Record brute-force scene (treated as Pass 2 fallback)
                    self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=2, scene_path=scene_path)
                    scene_idx += 1

        if self.verbose_summary:
            # Calculate duration statistics
            durations = [duration for _, _, _, duration in final_scene_tuples]
            if durations:
                min_duration = min(durations)
                max_duration = max(durations)
                mean_duration = sum(durations) / len(durations)
                
                # Calculate median duration
                sorted_durations = sorted(durations)
                n = len(sorted_durations)
                if n % 2 == 0:
                    median_duration = (sorted_durations[n//2 - 1] + sorted_durations[n//2]) / 2
                else:
                    median_duration = sorted_durations[n//2]
                
                total_scene_duration = sum(durations)
                
                summary_lines = [
                    "", "="*50,
                    "Dynamic Scene Detection Summary",
                    "="*50,
                    f"Total Story Lines Found: {len(story_lines)}",
                    f" - Segments saved directly: {storyline_direct_saves}",
                    f" - Segments from granular split (Pass 2): {granular_segments_count}",
                    f" - Segments from brute-force split: {brute_force_segments_count}",
                    "-" * 50,
                    f"Total Final Scenes Saved: {len(final_scene_tuples)}",
                    f"Scene Duration Statistics:",
                    f" - Shortest: {min_duration:.2f}s",
                    f" - Longest: {max_duration:.2f}s", 
                    f" - Mean: {mean_duration:.2f}s",
                    f" - Median: {median_duration:.2f}s",
                    f" - Total: {total_scene_duration:.1f}s ({total_scene_duration/60:.1f}m)",
                    "="*50, ""
                ]
            else:
                summary_lines = [
                    "", "="*50,
                    "Dynamic Scene Detection Summary",
                    "="*50,
                    f"Total Story Lines Found: {len(story_lines)}",
                    f" - Segments saved directly: {storyline_direct_saves}",
                    f" - Segments from granular split (Pass 2): {granular_segments_count}",
                    f" - Segments from brute-force split: {brute_force_segments_count}",
                    "-" * 50,
                    f"Total Final Scenes Saved: {len(final_scene_tuples)}",
                    "="*50, ""
                ]
            logger.info('\n'.join(summary_lines))

        logger.info(f"Detected and saved {len(final_scene_tuples)} final scenes.")
        return final_scene_tuples

    def get_detection_metadata(self) -> Dict:
        """
        Export scene detection metadata for the data contract.

        Returns a dictionary suitable for saving to the master metadata JSON file.
        Should be called after detect_scenes() to get populated data.

        Returns:
            Dict with 'scenes_detected', 'coarse_boundaries', 'vad_segments', 'vad_method', 'vad_params'
        """
        return {
            "scenes_detected": self.scenes_detected,
            "coarse_boundaries": self.coarse_boundaries if self.coarse_boundaries else None,
            "vad_segments": self.vad_segments if self.vad_segments else None,
            "vad_method": self.vad_method,
            "vad_params": self.vad_params if self.vad_params else None,
        }
