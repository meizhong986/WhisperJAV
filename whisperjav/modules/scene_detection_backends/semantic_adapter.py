"""
Semantic Audio Clustering Adapter for WhisperJAV Scene Detection.

This module provides a loose-coupling integration between WhisperJAV and
the SemanticAudioClustering engine. It acts as an adapter that:

1. Calls the external SemanticAudioClustering engine to analyze audio
2. Parses the resulting JSON metadata
3. Extracts audio segments using WhisperJAV's patterns
4. Returns scene tuples compatible with WhisperJAV's pipeline

The SemanticAudioClustering engine performs texture-based segmentation using
MFCC features and agglomerative clustering, grouping similar acoustic patterns
together rather than splitting at silence points.

Integration Contract:
    - Input: audio_path, output_dir, media_basename
    - Output: List[Tuple[Path, float, float, float]] (scene_path, start, end, duration)
    - Side Effects: Saves WAV files, populates data contract fields
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from whisperjav.utils.logger import logger


@dataclass
class SemanticClusteringConfig:
    """
    Configuration for Semantic Audio Clustering.

    These parameters control the segmentation behavior of the
    SemanticAudioClustering engine.

    Attributes:
        min_duration: Minimum segment duration in seconds (segments shorter
                      than this will be merged with neighbors)
        max_duration: Maximum segment duration in seconds (segments longer
                      than this will be split at lowest energy points)
        snap_window: Window size in seconds for snapping boundaries to silence
        clustering_threshold: Distance threshold for agglomerative clustering
                              (lower = more segments, higher = fewer segments)
        sample_rate: Target sample rate for audio processing
    """
    min_duration: float = 20.0
    max_duration: float = 420.0
    snap_window: float = 5.0
    clustering_threshold: float = 18.0
    sample_rate: int = 16000

    # Additional options for WhisperJAV integration
    preserve_original_sr: bool = True  # Save WAVs at original sample rate
    visualize: bool = False  # Generate visualization PNG

    # Presets support
    preset: str = "default"


# Preset configurations matching the YAML config
SEMANTIC_PRESETS = {
    "default": SemanticClusteringConfig(),
    "dialogue_heavy": SemanticClusteringConfig(
        min_duration=15.0,
        max_duration=300.0,
        snap_window=3.0,
        clustering_threshold=15.0,
    ),
    "music_heavy": SemanticClusteringConfig(
        min_duration=10.0,
        max_duration=180.0,
        snap_window=2.0,
        clustering_threshold=12.0,
    ),
    "action_content": SemanticClusteringConfig(
        min_duration=25.0,
        max_duration=420.0,
        snap_window=8.0,
        clustering_threshold=22.0,
    ),
    "conservative": SemanticClusteringConfig(
        min_duration=30.0,
        max_duration=420.0,
        snap_window=6.0,
        clustering_threshold=22.0,
    ),
    "aggressive": SemanticClusteringConfig(
        min_duration=10.0,
        max_duration=180.0,
        snap_window=2.0,
        clustering_threshold=10.0,
    ),
}


class SemanticClusteringAdapter:
    """
    Adapter that bridges SemanticAudioClustering to WhisperJAV's scene detection contract.

    This class provides loose coupling between WhisperJAV and the external
    SemanticAudioClustering engine. The engine is lazy-loaded only when needed,
    ensuring no impact on WhisperJAV when other scene detection methods are used.

    Responsibilities:
        1. Lazy-load SemanticAudioClustering module on first use
        2. Call the engine to generate JSON metadata
        3. Parse JSON and transform to WhisperJAV format
        4. Extract audio segments and save as WAV files
        5. Populate data contract fields for downstream consumers

    Data Contract (matches DynamicSceneDetector):
        - scenes_detected: List of rich metadata per scene
        - vad_segments: Not used (set to empty list)
        - coarse_boundaries: Not used (semantic uses different paradigm)
        - vad_method: Set to "semantic_clustering"
        - vad_params: Engine configuration used

    Example:
        >>> adapter = SemanticClusteringAdapter(config)
        >>> scenes = adapter.detect_scenes(audio_path, output_dir, "my_video")
        >>> # scenes = [(Path, start, end, duration), ...]
    """

    def __init__(
        self,
        config: Optional[SemanticClusteringConfig] = None,
        preset: str = "default",
        logger_instance: Optional[logging.Logger] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ):
        """
        Initialize the Semantic Clustering Adapter.

        Args:
            config: Configuration object. If None, uses preset or default.
            preset: Preset name to use if config not provided.
            logger_instance: Optional logger for the adapter and engine.
            progress_callback: Optional callback for progress reporting.
            **kwargs: Additional parameters passed to config or engine.
        """
        # Handle preset selection
        if config is None:
            preset_name = kwargs.get("preset", preset)
            if preset_name in SEMANTIC_PRESETS:
                config = SEMANTIC_PRESETS[preset_name]
                logger.debug(f"Using semantic clustering preset: {preset_name}")
            else:
                config = SemanticClusteringConfig()
                logger.warning(f"Unknown preset '{preset_name}', using default")

        # Apply any kwargs overrides to config
        for key in ["min_duration", "max_duration", "snap_window",
                    "clustering_threshold", "sample_rate"]:
            if key in kwargs:
                setattr(config, key, kwargs[key])

        self.config = config
        self.logger = logger_instance
        self.progress_callback = progress_callback

        # Lazy-loaded engine reference
        self._engine = None
        self._engine_config_class = None

        # Data contract fields (matching DynamicSceneDetector interface)
        self.scenes_detected: List[Dict] = []
        self.vad_segments: List[Dict] = []  # Not used by semantic clustering
        self.coarse_boundaries: List[Dict] = []  # Not applicable
        self.vad_method: Optional[str] = "semantic_clustering"
        self.vad_params: Dict = {}

        # Store the full semantic metadata for future use
        self.semantic_metadata: Optional[Dict] = None

        logger.debug(
            f"SemanticClusteringAdapter initialized: "
            f"min_dur={config.min_duration}s, max_dur={config.max_duration}s, "
            f"threshold={config.clustering_threshold}"
        )

    def _ensure_engine(self) -> Dict:
        """
        Lazy import of SemanticAudioClustering to avoid dependency loading.

        Returns:
            Dict with 'process' function and 'config_class' for the engine.

        Raises:
            RuntimeError: If SemanticAudioClustering module is not available.
        """
        if self._engine is not None:
            return {"process": self._engine, "config_class": self._engine_config_class}

        try:
            # Import from vendored location (whisperjav/vendor/)
            from whisperjav.vendor.semantic_audio_clustering import (
                process_movie_v7,
                SegmentationConfig,
                __version__ as engine_version
            )

            self._engine = process_movie_v7
            self._engine_config_class = SegmentationConfig

            logger.info(
                f"SemanticAudioClustering engine loaded (v{engine_version})"
            )

            return {"process": self._engine, "config_class": self._engine_config_class}

        except ImportError as e:
            error_msg = (
                "SemanticAudioClustering module is required for method='semantic'. "
                "The vendored module at whisperjav/vendor/semantic_audio_clustering.py "
                "could not be loaded. Ensure all dependencies are installed:\n"
                "  pip install librosa scikit-learn soundfile"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def detect_scenes(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
        temp_dir: Optional[Path] = None
    ) -> List[Tuple[Path, float, float, float]]:
        """
        Detect scenes using Semantic Audio Clustering.

        This method:
        1. Calls the SemanticAudioClustering engine to analyze the audio
        2. Parses the resulting JSON metadata
        3. Extracts audio segments based on asr_processing timestamps
        4. Saves segments as WAV files
        5. Returns scene tuples compatible with WhisperJAV's pipeline

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save scene WAV files
            media_basename: Base name for output files (e.g., "my_video")
            temp_dir: Optional temp directory for intermediate files

        Returns:
            List of tuples: (scene_path, start_sec, end_sec, duration)
            Using asr_processing timestamps (with overlap) for correct ASR alignment.

        Side Effects:
            - Saves WAV files to output_dir
            - Saves JSON metadata to temp_dir (or output_dir)
            - Populates self.scenes_detected, self.semantic_metadata, etc.
        """
        logger.info(f"Starting Semantic Audio Clustering for: {audio_path}")

        # Reset data contract fields
        self.scenes_detected = []
        self.vad_segments = []
        self.coarse_boundaries = []
        self.semantic_metadata = None
        self.vad_params = {
            "min_duration": self.config.min_duration,
            "max_duration": self.config.max_duration,
            "snap_window": self.config.snap_window,
            "clustering_threshold": self.config.clustering_threshold,
        }

        # Determine JSON output path
        work_dir = temp_dir or output_dir
        json_output_path = work_dir / f"{media_basename}_semantic.json"

        # Report progress
        if self.progress_callback:
            self.progress_callback(0.05, "Loading Semantic Audio Clustering engine...")

        # 1. Get the engine (lazy load)
        engine = self._ensure_engine()

        # 2. Build engine config
        engine_config = engine["config_class"](
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
            snap_window=self.config.snap_window,
            clustering_threshold=self.config.clustering_threshold,
            sample_rate=self.config.sample_rate,
        )

        # 3. Run the engine
        if self.progress_callback:
            self.progress_callback(0.1, "Analyzing audio texture...")

        try:
            engine["process"](
                file_path=str(audio_path),
                output_path=str(json_output_path),
                config=engine_config,
                visualize=self.config.visualize,
                logger=self.logger,
                progress_callback=self._wrap_progress_callback(0.1, 0.7),
            )
        except Exception as e:
            logger.error(f"SemanticAudioClustering engine failed: {e}")
            raise

        # 4. Parse JSON metadata
        if self.progress_callback:
            self.progress_callback(0.75, "Parsing segmentation results...")

        if not json_output_path.exists():
            raise RuntimeError(f"Engine did not produce output file: {json_output_path}")

        with open(json_output_path, "r", encoding="utf-8") as f:
            self.semantic_metadata = json.load(f)

        # 5. Transform and extract audio segments
        if self.progress_callback:
            self.progress_callback(0.8, "Extracting audio segments...")

        scene_tuples = self._transform_and_split(
            audio_path=audio_path,
            output_dir=output_dir,
            media_basename=media_basename,
        )

        if self.progress_callback:
            self.progress_callback(1.0, "Semantic scene detection complete")

        logger.info(
            f"Semantic clustering complete: {len(scene_tuples)} scenes extracted"
        )

        return scene_tuples

    def _wrap_progress_callback(
        self,
        start: float,
        end: float
    ) -> Optional[Callable[[float, str], None]]:
        """
        Wrap progress callback to map engine progress [0,1] to adapter range [start,end].
        """
        if self.progress_callback is None:
            return None

        def wrapped(progress: float, message: str):
            # Map [0,1] to [start, end]
            mapped = start + (progress * (end - start))
            self.progress_callback(mapped, message)

        return wrapped

    def _transform_and_split(
        self,
        audio_path: Path,
        output_dir: Path,
        media_basename: str,
    ) -> List[Tuple[Path, float, float, float]]:
        """
        Transform JSON segments to WhisperJAV format and extract audio.

        Uses asr_processing timestamps (with overlap) for extraction,
        ensuring correct timestamp alignment for downstream ASR.
        """
        # Load source audio
        try:
            audio_data, source_sr = sf.read(str(audio_path), dtype="float32")
        except Exception as e:
            logger.error(f"Failed to load audio for extraction: {e}")
            raise

        # Convert to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Determine output sample rate
        if self.config.preserve_original_sr:
            output_sr = source_sr
        else:
            output_sr = self.config.sample_rate
            if source_sr != output_sr:
                import librosa
                audio_data = librosa.resample(
                    audio_data, orig_sr=source_sr, target_sr=output_sr
                )

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        segments = self.semantic_metadata.get("segments", [])
        final_scene_tuples: List[Tuple[Path, float, float, float]] = []

        total_segments = len(segments)

        for idx, segment in enumerate(segments):
            # Use asr_processing timestamps (with overlap) for extraction
            # This ensures correct timestamp alignment for downstream ASR
            asr_ts = segment.get("asr_processing", segment.get("timestamps", {}))

            start_sec = asr_ts.get("start", 0.0)
            end_sec = asr_ts.get("end", 0.0)
            duration = end_sec - start_sec

            if duration <= 0:
                logger.warning(f"Skipping segment {idx} with invalid duration: {duration}")
                continue

            # Calculate sample boundaries
            start_sample = int(start_sec * output_sr)
            end_sample = int(end_sec * output_sr)

            # Clamp to audio bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if end_sample <= start_sample:
                logger.warning(f"Skipping segment {idx}: sample range invalid")
                continue

            # Extract audio segment
            scene_audio = audio_data[start_sample:end_sample]

            # Save scene WAV
            scene_filename = f"{media_basename}_scene_{idx:04d}.wav"
            scene_path = output_dir / scene_filename

            sf.write(str(scene_path), scene_audio, output_sr, subtype="PCM_16")

            # Add to result tuples
            # IMPORTANT: Pass asr_processing timestamps (what we extracted)
            final_scene_tuples.append((scene_path, start_sec, end_sec, duration))

            # Populate data contract
            self.scenes_detected.append({
                "scene_index": idx,
                "start_time_seconds": round(start_sec, 3),
                "end_time_seconds": round(end_sec, 3),
                "duration_seconds": round(duration, 3),
                "detection_pass": 0,  # Semantic is single-pass (0 = no pass distinction)
                "filename": scene_filename,
                "context": segment.get("context", {}),
                "asr_prompt": segment.get("asr_prompt", ""),
            })

            # Progress update (sparse to avoid spam)
            if self.progress_callback and idx % 10 == 0:
                progress = 0.8 + (0.15 * (idx / total_segments))
                self.progress_callback(progress, f"Extracting scene {idx+1}/{total_segments}")

        return final_scene_tuples

    def get_detection_metadata(self) -> Dict:
        """
        Export scene detection metadata for the data contract.

        Returns a dictionary suitable for saving to the master metadata JSON file.
        Should be called after detect_scenes() to get populated data.

        Returns:
            Dict matching DynamicSceneDetector.get_detection_metadata() format,
            plus additional semantic-specific metadata.
        """
        return {
            "scenes_detected": self.scenes_detected,
            "coarse_boundaries": None,  # Not applicable to semantic clustering
            "vad_segments": None,  # Not used
            "vad_method": self.vad_method,
            "vad_params": self.vad_params,
            # Additional semantic-specific data
            "semantic_metadata": self.semantic_metadata,
        }

    def get_scene_context(self, scene_index: int) -> Optional[Dict]:
        """
        Get context information for a specific scene.

        Useful for retrieving ASR prompts and scene type labels
        for context-aware transcription.

        Args:
            scene_index: Index of the scene (0-based)

        Returns:
            Dict with 'label', 'confidence', 'loudness_db', 'asr_prompt'
            or None if scene not found.
        """
        if self.semantic_metadata is None:
            return None

        segments = self.semantic_metadata.get("segments", [])
        for segment in segments:
            if segment.get("segment_index") == scene_index:
                return {
                    "context": segment.get("context", {}),
                    "asr_prompt": segment.get("asr_prompt", ""),
                }

        return None
