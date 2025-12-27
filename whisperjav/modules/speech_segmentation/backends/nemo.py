"""
NVIDIA NeMo VAD speech segmentation backend.

Two variants available:
    - nemo-lite: Fast frame-level VAD using EncDecFrameClassificationModel (~0.5GB)
    - nemo-diarization: Full NeuralDiarizer stack with speaker awareness (~4GB)

Model: nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 (official NVIDIA pretrained)

Requires nemo_toolkit[asr] to be installed.
Install: pip install nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main
"""

from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging
import os
import json
import tempfile
import shutil

import numpy as np

from ..base import SpeechSegment, SegmentationResult

logger = logging.getLogger("whisperjav")


class NemoModelLoadError(Exception):
    """Raised when NeMo model loading fails after recovery attempts."""
    pass


class NemoSpeechSegmenter:
    """
    NVIDIA NeMo VAD speech segmentation backend.

    Two variants available:
        - nemo-lite: Fast frame-level VAD using EncDecFrameClassificationModel (~0.5GB)
        - nemo-diarization: Full NeuralDiarizer stack with speaker awareness (~4GB)

    Example:
        segmenter = NemoSpeechSegmenter(variant="nemo-lite")
        result = segmenter.segment(audio_path)
    """

    # Variant configurations
    VARIANTS = {
        "nemo-lite": {
            "use_diarizer": False,
            "display_name": "NeMo Lite",
            "description": "Fast frame-level VAD (~0.5GB)",
        },
        "nemo-diarization": {
            "use_diarizer": True,
            "display_name": "NeMo Diarization",
            "description": "Full speaker-aware diarization (~4GB)",
        },
    }

    # Domain-specific NeMo config files (for diarization variant)
    DOMAIN_CONFIGS = {
        "general": "diar_infer_general.yaml",
        "meeting": "diar_infer_meeting.yaml",
        "telephonic": "diar_infer_telephonic.yaml",
    }

    # Frame VAD model (official NVIDIA pretrained)
    VAD_MODEL = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"

    def __init__(
        self,
        variant: str = "nemo-lite",
        onset: float = 0.4,
        offset: float = 0.3,
        pad_onset: float = 0.2,
        pad_offset: float = 0.10,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 200,
        filter_speech_first: bool = True,
        chunk_threshold_s: Optional[float] = None,
        max_group_duration_s: Optional[float] = None,
        diarization_domain: str = "general",
        temp_dir: Optional[Path] = None,
        use_overlap_smoothing: bool = False,
        smoothing_method: str = "median",
        **kwargs
    ):
        """
        Initialize NeMo VAD segmenter.

        Args:
            variant: Segmenter variant ("nemo-lite" or "nemo-diarization")
            onset: Speech start threshold (hysteresis high) [0.0, 1.0]
                   Default 0.4 is sensitive - catches soft speech/whispers
            offset: Speech end threshold (hysteresis low) [0.0, 1.0]
                   Default 0.3 maintains speech during brief probability dips
            pad_onset: Padding before speech segments (seconds)
                   Default 0.2 captures lead-in sounds
            pad_offset: Padding after speech segments (seconds)
                   Default 0.10 captures trailing sounds (sentence-final particles)
            min_speech_duration_ms: Minimum speech segment duration
                   Default 100ms catches short exclamations
            min_silence_duration_ms: Minimum silence between segments
                   Default 200ms merges nearby segments
            filter_speech_first: If True, apply min_duration_on filter before
                   min_duration_off (removes short speech first). Default True.
            chunk_threshold_s: Gap threshold for segment grouping
            max_group_duration_s: Maximum duration for a segment group (seconds).
                   Groups are split if adding a segment would exceed this limit.
                   Default 29s to stay within Whisper's 30s context window.
            diarization_domain: Domain for diarization ("general", "meeting", "telephonic")
            temp_dir: Directory for temporary files (uses system temp if None)
            use_overlap_smoothing: If True, apply overlapping window smoothing before
                   segmentation (recommended for higher accuracy, slower). Default False.
            smoothing_method: Smoothing method for overlap processing ("median" or "mean").
                   Default "median" for robust noise handling.
            **kwargs: Additional parameters for backward compatibility
                - chunk_threshold: Legacy alias for chunk_threshold_s
        """
        # Normalize variant name
        if variant in ("nemo", "nemo-lite"):
            variant = "nemo-lite"
        elif variant in ("nemo-diarization",):
            variant = "nemo-diarization"
        else:
            logger.warning(f"[NeMo] Unknown variant '{variant}', defaulting to nemo-lite")
            variant = "nemo-lite"

        self.variant = variant
        self._use_diarizer = self.VARIANTS[variant]["use_diarizer"]
        self.diarization_domain = diarization_domain
        self.onset = onset
        self.offset = offset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.filter_speech_first = filter_speech_first
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.use_overlap_smoothing = use_overlap_smoothing
        self.smoothing_method = smoothing_method

        # Handle backward compatibility: chunk_threshold (old) -> chunk_threshold_s (new)
        if chunk_threshold_s is not None:
            self.chunk_threshold_s = chunk_threshold_s
        elif "chunk_threshold" in kwargs:
            self.chunk_threshold_s = kwargs["chunk_threshold"]
        else:
            self.chunk_threshold_s = 2.5  # Default (reduced from 4.0 to minimize silence in Whisper input)

        # Maximum group duration - prevents groups from exceeding Whisper's context window
        self.max_group_duration_s = max_group_duration_s if max_group_duration_s is not None else 29.0

        # Lazy-loaded models
        self._diarizer = None
        self._vad_model = None

        # Track work directory for cleanup
        self._work_dir = None

        logger.info(
            f"[NeMo] Initialized: variant={variant}, "
            f"onset={onset}, offset={offset}, pad_onset={pad_onset}, pad_offset={pad_offset}"
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return self.variant

    @property
    def display_name(self) -> str:
        """Return user-friendly display name."""
        return self.VARIANTS[self.variant]["display_name"]

    def segment(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int = 16000,
        **kwargs
    ) -> SegmentationResult:
        """
        Detect speech segments using NeMo VAD.

        Behavior determined by variant:
            - nemo-lite: EncDecFrameClassificationModel (lightweight, fast, ~0.5GB)
            - nemo-diarization: NeuralDiarizer (full stack, slow, ~4GB)

        Args:
            audio: Audio data as numpy array, or path to audio file
            sample_rate: Sample rate of input audio
            **kwargs: Override parameters

        Returns:
            SegmentationResult with detected speech segments
        """
        start_time = time.time()
        logger.info(f"[NeMo] Starting segmentation with {self.name}")

        # Load audio first (can succeed even if model fails)
        try:
            logger.debug("[NeMo] Loading audio data...")
            audio_data, actual_sr = self._load_audio(audio, sample_rate)
            duration = len(audio_data) / actual_sr
            logger.info(f"[NeMo] Audio loaded: {duration:.2f}s at {actual_sr}Hz")
        except Exception as e:
            logger.error(f"[NeMo] Failed to load audio: {e}", exc_info=True)
            return SegmentationResult(
                segments=[],
                groups=[],
                method=f"{self.name}:audio_error",
                audio_duration_sec=0,
                parameters=self._get_parameters(),
                processing_time_sec=time.time() - start_time,
            )

        # Branch based on variant
        # nemo-diarization: Full NeuralDiarizer stack (~4GB)
        # nemo-lite: Lightweight frame VAD (~0.5GB)
        if self._use_diarizer:
            try:
                logger.info("[NeMo] Using NeuralDiarizer (full stack - this is SLOW)")
                logger.info("[NeMo] First run will download ~4GB of models...")
                segments = self._segment_with_diarizer(audio_data, actual_sr)
                method = f"{self.name}:diarizer"
                logger.info(f"[NeMo] NeuralDiarizer succeeded: {len(segments)} segments found")
            except NemoModelLoadError as e:
                # Model loading failed completely - return empty result with clear error
                logger.error(f"[NeMo] Model unavailable: {e}")
                elapsed = time.time() - start_time
                return SegmentationResult(
                    segments=[],
                    groups=[],
                    method=f"{self.name}:model_unavailable",
                    audio_duration_sec=duration,
                    parameters=self._get_parameters(),
                    processing_time_sec=elapsed,
                )
            except Exception as e:
                logger.warning(f"[NeMo] NeuralDiarizer failed: {e}")
                logger.info("[NeMo] Falling back to lightweight frame VAD...")
                try:
                    segments = self._segment_with_frame_vad(audio_data, actual_sr)
                    method = f"{self.name}:frame_vad_fallback"
                    logger.info(f"[NeMo] Fallback VAD succeeded: {len(segments)} segments found")
                except NemoModelLoadError as e2:
                    logger.error(f"[NeMo] Model unavailable: {e2}")
                    elapsed = time.time() - start_time
                    return SegmentationResult(
                        segments=[],
                        groups=[],
                        method=f"{self.name}:model_unavailable",
                        audio_duration_sec=duration,
                        parameters=self._get_parameters(),
                        processing_time_sec=elapsed,
                    )
                except Exception as e2:
                    logger.error(f"[NeMo] Fallback VAD also failed: {e2}", exc_info=True)
                    elapsed = time.time() - start_time
                    logger.error(f"[NeMo] Segmentation failed after {elapsed:.2f}s")
                    return SegmentationResult(
                        segments=[],
                        groups=[],
                        method=f"{self.name}:error",
                        audio_duration_sec=duration,
                        parameters=self._get_parameters(),
                        processing_time_sec=elapsed,
                    )
        else:
            # nemo-lite: lightweight frame VAD
            try:
                logger.info("[NeMo] Using lightweight frame VAD (EncDecFrameClassificationModel)")
                segments = self._segment_with_frame_vad(audio_data, actual_sr)
                method = f"{self.name}:frame_vad"
                logger.info(f"[NeMo] Frame VAD succeeded: {len(segments)} segments found")
            except NemoModelLoadError as e:
                logger.error(f"[NeMo] Model unavailable: {e}")
                elapsed = time.time() - start_time
                return SegmentationResult(
                    segments=[],
                    groups=[],
                    method=f"{self.name}:model_unavailable",
                    audio_duration_sec=duration,
                    parameters=self._get_parameters(),
                    processing_time_sec=elapsed,
                )
            except Exception as e:
                logger.error(f"[NeMo] Frame VAD failed: {e}", exc_info=True)
                elapsed = time.time() - start_time
                logger.error(f"[NeMo] Segmentation failed after {elapsed:.2f}s")
                return SegmentationResult(
                    segments=[],
                    groups=[],
                    method=f"{self.name}:error",
                    audio_duration_sec=duration,
                    parameters=self._get_parameters(),
                    processing_time_sec=elapsed,
                )

        # Group segments
        logger.debug("[NeMo] Grouping segments...")
        groups = self._group_segments(segments)

        elapsed = time.time() - start_time
        coverage = sum(s.duration_sec for s in segments) / duration if duration > 0 else 0
        logger.info(
            f"[NeMo] Segmentation complete: {len(segments)} segments, "
            f"{len(groups)} groups, {coverage:.1%} coverage, {elapsed:.2f}s"
        )

        return SegmentationResult(
            segments=segments,
            groups=groups,
            method=method,
            audio_duration_sec=duration,
            parameters=self._get_parameters(),
            processing_time_sec=elapsed,
        )

    def _segment_with_diarizer(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Primary method: Full NeuralDiarizer stack.

        Uses VAD + speaker embeddings + MSDD for accurate speech detection
        with speaker labels preserved in metadata.
        """
        logger.debug("[NeMo/Diarizer] Importing NeMo modules...")
        try:
            from omegaconf import OmegaConf
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
            import torch
            import torchaudio
            logger.debug("[NeMo/Diarizer] NeMo imports successful")
        except ImportError as e:
            raise ImportError(
                f"NeuralDiarizer requires nemo_toolkit with MSDD support: {e}\n"
                "Install: pip install nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
            )

        # Prepare work directory
        logger.debug("[NeMo/Diarizer] Preparing work directory...")
        work_dir = self._prepare_work_dir()
        self._work_dir = work_dir
        logger.debug(f"[NeMo/Diarizer] Work directory: {work_dir}")

        try:
            # Save audio as mono 16kHz WAV (NeMo requirement)
            logger.debug("[NeMo/Diarizer] Saving audio as mono 16kHz WAV...")
            wav_path = self._save_mono_wav(audio_data, sample_rate, work_dir)
            logger.debug(f"[NeMo/Diarizer] Audio saved to: {wav_path}")

            # Create manifest file
            logger.debug("[NeMo/Diarizer] Creating manifest file...")
            manifest_path = self._create_manifest(wav_path, work_dir)
            logger.debug(f"[NeMo/Diarizer] Manifest created: {manifest_path}")

            # Create NeMo config with hysteresis parameters
            logger.info("[NeMo/Diarizer] Creating diarizer configuration...")
            config = self._create_diarizer_config(work_dir)
            logger.debug(f"[NeMo/Diarizer] Config created with domain={self.diarization_domain}")

            # Run diarization
            logger.info(
                f"[NeMo/Diarizer] Running NeuralDiarizer (domain={self.diarization_domain}, "
                f"onset={self.onset}, offset={self.offset})..."
            )
            logger.info("[NeMo/Diarizer] This may take a while for model downloads on first run...")

            diarizer = NeuralDiarizer(cfg=config)
            logger.debug("[NeMo/Diarizer] Diarizer initialized, starting diarization...")
            diarizer.diarize()
            logger.info("[NeMo/Diarizer] Diarization complete")

            # Parse RTTM output
            rttm_path = work_dir / "pred_rttms" / "mono_file.rttm"
            logger.debug(f"[NeMo/Diarizer] Looking for RTTM at: {rttm_path}")

            if not rttm_path.exists():
                # Check if pred_rttms directory exists
                pred_rttms_dir = work_dir / "pred_rttms"
                if pred_rttms_dir.exists():
                    files = list(pred_rttms_dir.iterdir())
                    logger.error(f"[NeMo/Diarizer] pred_rttms contents: {files}")
                else:
                    logger.error("[NeMo/Diarizer] pred_rttms directory does not exist")
                raise FileNotFoundError(f"RTTM output not found: {rttm_path}")

            logger.debug("[NeMo/Diarizer] Parsing RTTM output...")
            segments = self._parse_rttm(rttm_path, sample_rate)
            logger.info(f"[NeMo/Diarizer] Parsed {len(segments)} segments from RTTM")

            # Cleanup diarizer to free GPU memory
            logger.debug("[NeMo/Diarizer] Cleaning up diarizer resources...")
            del diarizer
            # WARNING: torch.cuda.empty_cache() can cause crashes in subprocess workers
            # on Windows due to conflicts between PyTorch and ctranslate2 CUDA contexts.
            # Consider removing if NeMo is used in ensemble subprocess mode.
            # See: Silero VAD cleanup() for safer pattern (issue #82 root cause).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("[NeMo/Diarizer] GPU cache cleared")

            return segments

        finally:
            # Clean up work directory
            logger.debug("[NeMo/Diarizer] Cleaning up work directory...")
            self._cleanup_work_dir(work_dir)

    def _segment_with_frame_vad(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """
        Lightweight EncDecFrameClassificationModel (nemo-lite).

        Uses NeMo's proper VAD utilities:
        1. generate_vad_frame_pred() - generates frame-level predictions
        2. generate_vad_segment_table() - postprocesses to segments with hysteresis

        Requires ~0.5GB model download on first run.
        """
        logger.info("[NeMo/FrameVAD] Using EncDecFrameClassificationModel with proper VAD utilities")
        self._ensure_vad_model()

        # Import NeMo VAD utilities
        try:
            from nemo.collections.asr.parts.utils.vad_utils import (
                generate_vad_frame_pred,
                generate_vad_segment_table,
                generate_overlap_vad_seq,
            )
        except ImportError as e:
            raise ImportError(
                f"NeMo VAD utilities not available: {e}\n"
                "Install: pip install nemo_toolkit[asr]"
            )

        import soundfile as sf

        # Create work directory for VAD outputs
        work_dir = self._prepare_work_dir()
        self._work_dir = work_dir

        try:
            # Save audio to temp file
            logger.debug("[NeMo/FrameVAD] Saving audio to work directory...")
            audio_path = work_dir / "audio.wav"
            sf.write(str(audio_path), audio_data, sample_rate)
            logger.debug(f"[NeMo/FrameVAD] Audio saved: {audio_path}")

            # Create NeMo manifest file
            # Note: For VAD inference, label should be "0" (placeholder, not used for inference)
            logger.debug("[NeMo/FrameVAD] Creating manifest file...")
            manifest_path = work_dir / "input_manifest.json"
            duration_sec = len(audio_data) / sample_rate
            manifest_entry = {
                "audio_filepath": str(audio_path),
                "offset": 0,
                "duration": duration_sec,
                "label": "0",  # Placeholder label for inference
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest_entry, f)
                f.write("\n")
            logger.debug(f"[NeMo/FrameVAD] Manifest created: {manifest_path}")

            # Create output directories
            frame_pred_dir = work_dir / "frame_predictions"
            rttm_dir = work_dir / "rttm_output"
            frame_pred_dir.mkdir(exist_ok=True)
            rttm_dir.mkdir(exist_ok=True)

            # Step 1: Generate frame-level predictions
            logger.info("[NeMo/FrameVAD] Generating frame-level predictions...")
            # Frame VAD uses 20ms shift (0.02s) by default
            frame_length_in_sec = 0.02

            # Configure test data for the model (required before generate_vad_frame_pred)
            test_data_config = {
                "manifest_filepath": str(manifest_path),
                "sample_rate": 16000,
                "labels": ["0", "1"],
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 0,
            }
            from omegaconf import DictConfig
            self._vad_model.setup_test_data(DictConfig(test_data_config))

            pred_dir = generate_vad_frame_pred(
                vad_model=self._vad_model,
                window_length_in_sec=0.0,  # 0 = use model default
                shift_length_in_sec=frame_length_in_sec,
                manifest_vad_input=str(manifest_path),
                out_dir=str(frame_pred_dir),
            )
            logger.debug(f"[NeMo/FrameVAD] Frame predictions directory: {pred_dir}")

            # Robust frame file location handling
            # NeMo may save .frame files in different locations depending on version
            frame_file_name = audio_path.stem + ".frame"
            frame_file_found = None

            # Search multiple possible locations
            search_locations = [
                frame_pred_dir / frame_file_name,              # Expected location (out_dir)
                audio_path.with_suffix(".frame"),              # Next to audio file (NeMo bug)
                Path(pred_dir) / frame_file_name if pred_dir else None,  # Returned pred_dir
                work_dir / frame_file_name,                    # Work directory root
            ]

            for loc in search_locations:
                if loc and loc.exists():
                    frame_file_found = loc
                    logger.debug(f"[NeMo/FrameVAD] Found frame file at: {loc}")
                    break

            if frame_file_found is None:
                # Last resort: glob search in work_dir and frame_pred_dir
                logger.warning("[NeMo/FrameVAD] Frame file not in expected locations, searching...")
                for search_dir in [frame_pred_dir, work_dir, audio_path.parent]:
                    frame_files = list(search_dir.glob("*.frame"))
                    if frame_files:
                        frame_file_found = frame_files[0]
                        logger.debug(f"[NeMo/FrameVAD] Found frame file via glob: {frame_file_found}")
                        break

            if frame_file_found is None:
                raise FileNotFoundError(
                    f"[NeMo/FrameVAD] Frame prediction file not found. "
                    f"Searched: {[str(loc) for loc in search_locations if loc]}"
                )

            # Ensure frame file is in frame_pred_dir for subsequent steps
            frame_file_dst = frame_pred_dir / frame_file_name
            if frame_file_found != frame_file_dst:
                shutil.copy(frame_file_found, frame_file_dst)
                logger.debug(f"[NeMo/FrameVAD] Copied frame file to: {frame_file_dst}")

            # Update pred_dir to point to frame_pred_dir
            pred_dir = str(frame_pred_dir)

            # Step 2 (Optional): Apply overlapping window smoothing
            # This resolves conflicts when frames are covered by multiple windows
            if self.use_overlap_smoothing:
                logger.info(
                    f"[NeMo/FrameVAD] Applying overlap smoothing: "
                    f"method={self.smoothing_method}"
                )
                smoothed_dir = work_dir / "smoothed_predictions"
                smoothed_dir.mkdir(exist_ok=True)

                try:
                    smoothed_pred_dir = generate_overlap_vad_seq(
                        frame_pred_dir=pred_dir,
                        smoothing_method=self.smoothing_method,
                        overlap=0.5,  # Standard 50% overlap
                        window_length_in_sec=0.63,  # NeMo default
                        shift_length_in_sec=frame_length_in_sec,
                        num_workers=0,  # Avoid multiprocessing issues
                        out_dir=str(smoothed_dir),
                    )
                    pred_dir = smoothed_pred_dir
                    logger.debug(f"[NeMo/FrameVAD] Smoothed predictions: {smoothed_pred_dir}")
                except Exception as e:
                    logger.warning(
                        f"[NeMo/FrameVAD] Overlap smoothing failed (continuing without): {e}"
                    )

            # Step 3: Postprocess frame predictions to segments
            logger.info(
                f"[NeMo/FrameVAD] Postprocessing with hysteresis: "
                f"onset={self.onset}, offset={self.offset}, "
                f"pad_onset={self.pad_onset}, pad_offset={self.pad_offset}"
            )

            # Build postprocessing params (matches reference implementation)
            postprocessing_params = {
                "onset": self.onset,
                "offset": self.offset,
                "pad_onset": self.pad_onset,
                "pad_offset": self.pad_offset,
                "min_duration_on": self.min_speech_duration_ms / 1000.0,  # convert ms to sec
                "min_duration_off": self.min_silence_duration_ms / 1000.0,  # convert ms to sec
                "filter_speech_first": self.filter_speech_first,
            }

            rttm_out_dir = generate_vad_segment_table(
                vad_pred_dir=pred_dir,
                postprocessing_params=postprocessing_params,
                frame_length_in_sec=frame_length_in_sec,
                num_workers=0,  # Avoid multiprocessing issues
                use_rttm=True,
                out_dir=str(rttm_dir),
            )
            logger.debug(f"[NeMo/FrameVAD] RTTM output saved to: {rttm_out_dir}")

            # Step 4: Parse RTTM output
            # Look for RTTM file (named after audio file)
            rttm_files = list(Path(rttm_out_dir).glob("*.rttm"))
            if not rttm_files:
                logger.warning("[NeMo/FrameVAD] No RTTM files found, returning empty segments")
                return []

            rttm_path = rttm_files[0]
            logger.debug(f"[NeMo/FrameVAD] Parsing RTTM file: {rttm_path}")
            segments = self._parse_rttm(rttm_path, sample_rate)
            logger.info(f"[NeMo/FrameVAD] Extracted {len(segments)} segments")

            return segments

        finally:
            # Clean up work directory
            logger.debug("[NeMo/FrameVAD] Cleaning up work directory...")
            self._cleanup_work_dir(work_dir)
            self._work_dir = None

    def _ensure_vad_model(self) -> None:
        """Load frame VAD model with defensive cache handling."""
        if self._vad_model is not None:
            logger.debug("[NeMo/FrameVAD] Model already loaded")
            return

        logger.debug("[NeMo/FrameVAD] Importing EncDecFrameClassificationModel...")
        try:
            from nemo.collections.asr.models import EncDecFrameClassificationModel
        except ImportError:
            raise ImportError(
                "NeMo VAD requires nemo_toolkit. Install with:\n"
                "pip install nemo_toolkit[asr]"
            )

        logger.info(f"[NeMo/FrameVAD] Loading model: {self.VAD_MODEL}")
        logger.info("[NeMo/FrameVAD] This may download model files on first run...")

        # Attempt 1: Try loading model
        try:
            self._vad_model = EncDecFrameClassificationModel.from_pretrained(
                self.VAD_MODEL,
                strict=False
            )
            self._vad_model.eval()
            logger.info("[NeMo/FrameVAD] Model loaded successfully")
            return

        except FileNotFoundError as e:
            # Cache corruption detected - missing model_config.yaml or similar
            error_str = str(e).lower()
            if "model_config" in error_str or "config" in error_str or ".yaml" in error_str:
                logger.error(f"[NeMo/FrameVAD] Cache corruption detected: {e}")

                # NeMo bug workaround: Extract .nemo then load via restore_from
                logger.info("[NeMo/FrameVAD] Attempting workaround: manual extraction + restore_from...")
                nemo_path = self._download_and_extract_nemo()
                if nemo_path:
                    try:
                        self._vad_model = EncDecFrameClassificationModel.restore_from(
                            str(nemo_path),
                            strict=False
                        )
                        self._vad_model.eval()
                        logger.info("[NeMo/FrameVAD] Model loaded via restore_from workaround")
                        return
                    except Exception as restore_error:
                        logger.error(f"[NeMo/FrameVAD] restore_from failed: {restore_error}")
                        raise NemoModelLoadError(
                            f"Failed to load NeMo model via restore_from workaround.\n"
                            f"Original error: {e}\n"
                            f"restore_from error: {restore_error}\n"
                            f"This appears to be a NeMo bug with HuggingFace model extraction.\n"
                            f"Try: pip install --upgrade nemo_toolkit"
                        ) from restore_error
                else:
                    raise NemoModelLoadError(
                        f"Failed to download/extract NeMo model.\n"
                        f"Original error: {e}\n"
                        f"Try manually clearing: ~/.cache/huggingface/hub/ and ~/.cache/torch/NeMo/"
                    ) from e
            else:
                # FileNotFoundError but not cache-related
                logger.error(f"[NeMo/FrameVAD] File not found: {e}", exc_info=True)
                raise NemoModelLoadError(f"Failed to load NeMo model: {e}") from e

        except Exception as e:
            logger.error(f"[NeMo/FrameVAD] Unexpected error loading model: {e}", exc_info=True)
            raise NemoModelLoadError(f"Failed to load NeMo model: {e}") from e

    def _download_and_extract_nemo(self) -> Optional[Path]:
        """
        Download .nemo file from HuggingFace and return path to it.

        NeMo's HuggingFace integration has a bug where it downloads the .nemo
        file but doesn't extract it. This method downloads using huggingface_hub
        directly, bypassing NeMo's buggy logic.

        Returns:
            Path to the downloaded .nemo file, or None if download failed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.warning("[NeMo] huggingface_hub not available for direct download")
            return None

        try:
            # Download .nemo file directly
            logger.info("[NeMo] Downloading .nemo file via huggingface_hub...")
            nemo_path = hf_hub_download(
                repo_id="nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
                filename="frame_vad_multilingual_marblenet_v2.0.nemo"
            )
            logger.info(f"[NeMo] Downloaded to: {nemo_path}")
            return Path(nemo_path)

        except Exception as e:
            logger.error(f"[NeMo] Failed to download .nemo file: {e}")
            return None

    def _clear_corrupted_cache(self) -> None:
        """Clear corrupted NeMo/HuggingFace cache for this model."""
        cache_locations = [
            # HuggingFace Hub cache
            Path.home() / ".cache" / "huggingface" / "hub",
            # NeMo cache (torch)
            Path.home() / ".cache" / "torch" / "NeMo",
            # Alternative NeMo location
            Path.home() / ".nemo",
        ]

        model_patterns = ["Frame_VAD", "MarbleNet", "nvidia"]

        for cache_dir in cache_locations:
            if not cache_dir.exists():
                continue
            try:
                for item in cache_dir.iterdir():
                    if any(pattern in item.name for pattern in model_patterns):
                        logger.info(f"[NeMo] Removing corrupted cache: {item.name}")
                        try:
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            else:
                                item.unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"[NeMo] Failed to remove {item}: {e}")
            except PermissionError as e:
                logger.warning(f"[NeMo] Permission denied accessing {cache_dir}: {e}")
            except Exception as e:
                logger.warning(f"[NeMo] Error scanning cache dir {cache_dir}: {e}")

    def _extract_nemo_archive(self) -> bool:
        """
        Extract .nemo archive if model_config.yaml is missing.

        NeMo's HuggingFace integration has a bug where it downloads the .nemo
        file but doesn't extract it. This method manually extracts the archive.

        Returns:
            True if extraction succeeded or wasn't needed, False otherwise.
        """
        import tarfile

        # Find NeMo cache directory
        nemo_cache = Path.home() / ".cache" / "torch" / "NeMo"
        if not nemo_cache.exists():
            return False

        # Search for Frame_VAD model directories
        for version_dir in nemo_cache.iterdir():
            if not version_dir.is_dir():
                continue
            hf_cache = version_dir / "hf_hub_cache" / "nvidia" / "Frame_VAD_Multilingual_MarbleNet_v2.0"
            if not hf_cache.exists():
                continue

            # Find the commit hash directory
            for commit_dir in hf_cache.iterdir():
                if not commit_dir.is_dir() or commit_dir.name.startswith('.'):
                    continue

                config_path = commit_dir / "model_config.yaml"
                if config_path.exists():
                    logger.debug(f"[NeMo] model_config.yaml already exists at {config_path}")
                    return True

                # Look for .nemo file to extract
                for nemo_file in commit_dir.glob("*.nemo"):
                    logger.info(f"[NeMo] Found unextracted .nemo file: {nemo_file.name}")
                    try:
                        with tarfile.open(nemo_file, 'r') as tar:
                            tar.extractall(path=commit_dir)
                        logger.info(f"[NeMo] Extracted .nemo archive to {commit_dir}")

                        if config_path.exists():
                            logger.info("[NeMo] model_config.yaml now available")
                            return True
                    except Exception as e:
                        logger.warning(f"[NeMo] Failed to extract {nemo_file}: {e}")

        return False

    def _prepare_work_dir(self) -> Path:
        """Create temporary work directory for NeuralDiarizer."""
        if self.temp_dir:
            work_dir = self.temp_dir / f"nemo_diar_{int(time.time())}"
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="nemo_diar_"))

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "data").mkdir(exist_ok=True)

        return work_dir

    def _cleanup_work_dir(self, work_dir: Path) -> None:
        """Clean up temporary work directory."""
        try:
            if work_dir and work_dir.exists():
                shutil.rmtree(work_dir)
                logger.debug(f"[NeMo] Cleaned up work dir: {work_dir}")
        except Exception as e:
            logger.warning(f"[NeMo] Failed to cleanup work dir: {e}")

    def _save_mono_wav(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        work_dir: Path
    ) -> Path:
        """Save audio as mono 16kHz WAV for NeMo."""
        import torch
        import torchaudio

        # Convert to torch tensor
        if audio_data.ndim == 1:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
        else:
            # Convert stereo to mono
            audio_tensor = torch.from_numpy(audio_data.mean(axis=1)).unsqueeze(0).float()

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.debug(f"[NeMo] Resampling from {sample_rate}Hz to 16000Hz...")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)

        # Save
        wav_path = work_dir / "mono_file.wav"
        torchaudio.save(str(wav_path), audio_tensor, 16000)

        return wav_path

    def _create_manifest(self, wav_path: Path, work_dir: Path) -> Path:
        """Create NeMo manifest file."""
        manifest_path = work_dir / "data" / "input_manifest.json"

        meta = {
            "audio_filepath": str(wav_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }

        with open(manifest_path, "w") as f:
            json.dump(meta, f)
            f.write("\n")

        return manifest_path

    def _create_diarizer_config(self, work_dir: Path):
        """Create NeMo diarizer config with hysteresis parameters."""
        from omegaconf import OmegaConf

        # Try to import wget, fall back to urllib if not available
        try:
            import wget
            use_wget = True
        except ImportError:
            import urllib.request
            use_wget = False
            logger.debug("[NeMo/Config] wget not available, using urllib")

        # Download domain-specific config from NeMo repo
        config_name = self.DOMAIN_CONFIGS.get(self.diarization_domain, "diar_infer_general.yaml")
        config_url = (
            f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/"
            f"examples/speaker_tasks/diarization/conf/inference/{config_name}"
        )

        config_path = work_dir / config_name
        if not config_path.exists():
            try:
                logger.info(f"[NeMo/Config] Downloading config: {config_name}")
                if use_wget:
                    wget.download(config_url, str(config_path), bar=None)
                else:
                    urllib.request.urlretrieve(config_url, str(config_path))
                logger.debug(f"[NeMo/Config] Config downloaded to: {config_path}")
            except Exception as e:
                logger.warning(f"[NeMo/Config] Failed to download config: {e}")
                logger.info("[NeMo/Config] Using minimal built-in config...")
                config = self._create_minimal_config(work_dir)
                return config

        logger.debug("[NeMo/Config] Loading config file...")
        config = OmegaConf.load(str(config_path))

        # Configure paths
        config.diarizer.manifest_filepath = str(work_dir / "data" / "input_manifest.json")
        config.diarizer.out_dir = str(work_dir)

        # Configure VAD with hysteresis (key improvement from reference!)
        logger.debug(
            f"[NeMo/Config] Setting VAD params: onset={self.onset}, "
            f"offset={self.offset}, pad_offset={self.pad_offset}"
        )
        config.diarizer.vad.model_path = "vad_multilingual_marblenet"
        config.diarizer.vad.parameters.onset = self.onset
        config.diarizer.vad.parameters.offset = self.offset
        config.diarizer.vad.parameters.pad_offset = self.pad_offset

        # Configure speaker embeddings
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.oracle_vad = False

        # Configure MSDD
        config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

        # Configure clustering
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        # Workaround for multiprocessing issues
        config.num_workers = 0

        logger.debug("[NeMo/Config] Configuration ready")
        return config

    def _create_minimal_config(self, work_dir: Path):
        """Create minimal diarizer config if download fails."""
        from omegaconf import OmegaConf

        logger.debug("[NeMo/Config] Creating minimal config...")
        config = OmegaConf.create({
            "num_workers": 0,
            "diarizer": {
                "manifest_filepath": str(work_dir / "data" / "input_manifest.json"),
                "out_dir": str(work_dir),
                "oracle_vad": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": self.onset,
                        "offset": self.offset,
                        "pad_offset": self.pad_offset,
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                    }
                }
            }
        })

        return config

    def _parse_rttm(self, rttm_path: Path, sample_rate: int) -> List[SpeechSegment]:
        """
        Parse RTTM file, preserving speaker labels in metadata.

        RTTM format: SPEAKER file 1 <start> <duration> <NA> <NA> speaker_X <NA>
        """
        segments = []
        lines_processed = 0
        lines_skipped = 0

        logger.debug(f"[NeMo/RTTM] Reading RTTM file: {rttm_path}")
        with open(rttm_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("SPEAKER"):
                    continue

                lines_processed += 1
                parts = line.split()
                if len(parts) < 8:
                    lines_skipped += 1
                    continue

                try:
                    start_sec = float(parts[3])
                    duration_sec = float(parts[4])
                    speaker_id = parts[7]  # e.g., "speaker_0"

                    end_sec = start_sec + duration_sec

                    # Apply minimum duration filter
                    if duration_sec * 1000 >= self.min_speech_duration_ms:
                        segments.append(SpeechSegment(
                            start_sec=start_sec,
                            end_sec=end_sec,
                            start_sample=int(start_sec * sample_rate),
                            end_sample=int(end_sec * sample_rate),
                            confidence=1.0,
                            metadata={"speaker": speaker_id}
                        ))
                    else:
                        lines_skipped += 1
                        logger.debug(
                            f"[NeMo/RTTM] Skipped short segment: "
                            f"{start_sec:.2f}-{end_sec:.2f}s ({duration_sec*1000:.0f}ms)"
                        )
                except (ValueError, IndexError) as e:
                    lines_skipped += 1
                    logger.warning(f"[NeMo/RTTM] Failed to parse line: {line}, error: {e}")
                    continue

        # Sort by start time
        segments.sort(key=lambda s: s.start_sec)

        logger.debug(
            f"[NeMo/RTTM] Parsed {len(segments)} segments from "
            f"{lines_processed} RTTM lines ({lines_skipped} skipped)"
        )
        return segments

    def _group_segments(
        self,
        segments: List[SpeechSegment]
    ) -> List[List[SpeechSegment]]:
        """Group segments based on time gaps.

        Groups are split if the gap exceeds chunk_threshold_s OR if adding
        a segment would cause the group duration to exceed max_group_duration_s.
        """
        if not segments:
            return []

        groups: List[List[SpeechSegment]] = [[]]

        for i, segment in enumerate(segments):
            if i > 0:
                prev_end = segments[i - 1].end_sec
                gap = segment.start_sec - prev_end

                # Check if adding this segment would exceed max group duration
                would_exceed_max = False
                if groups[-1]:
                    group_start = groups[-1][0].start_sec
                    potential_duration = segment.end_sec - group_start
                    would_exceed_max = potential_duration > self.max_group_duration_s

                # Start new group if gap too large OR would exceed max duration
                if gap > self.chunk_threshold_s or would_exceed_max:
                    groups.append([])

            groups[-1].append(segment)

        return groups

    def _load_audio(
        self,
        audio: Union[np.ndarray, Path, str],
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """Load audio from path or return array directly."""
        if isinstance(audio, np.ndarray):
            logger.debug(f"[NeMo] Using provided audio array: {len(audio)} samples")
            return audio, sample_rate

        audio_path = Path(audio) if isinstance(audio, str) else audio
        logger.debug(f"[NeMo] Loading audio from: {audio_path}")

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile is required for loading audio files")

        audio_data, actual_sr = sf.read(str(audio_path), dtype='float32')

        if audio_data.ndim > 1:
            logger.debug(f"[NeMo] Converting stereo to mono")
            audio_data = np.mean(audio_data, axis=1)

        return audio_data, actual_sr

    def _get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        params = {
            "variant": self.variant,
            "onset": self.onset,
            "offset": self.offset,
            "pad_onset": self.pad_onset,
            "pad_offset": self.pad_offset,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "filter_speech_first": self.filter_speech_first,
            "chunk_threshold_s": self.chunk_threshold_s,
            "max_group_duration_s": self.max_group_duration_s,
            "use_overlap_smoothing": self.use_overlap_smoothing,
            "smoothing_method": self.smoothing_method,
        }
        # Include diarization_domain only for diarization variant
        if self._use_diarizer:
            params["diarization_domain"] = self.diarization_domain
        return params

    def cleanup(self) -> None:
        """Release model resources."""
        logger.debug("[NeMo] Cleaning up resources...")

        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None
            logger.debug("[NeMo] VAD model released")

        if self._work_dir and self._work_dir.exists():
            self._cleanup_work_dir(self._work_dir)
            self._work_dir = None

        import gc
        gc.collect()

        # WARNING: torch.cuda.empty_cache() can cause crashes in subprocess workers
        # on Windows due to conflicts between PyTorch and ctranslate2 CUDA contexts.
        # Consider removing if NeMo is used in ensemble subprocess mode.
        # See: Silero VAD cleanup() for safer pattern (issue #82 root cause).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("[NeMo] GPU cache cleared")
        except ImportError:
            pass

        logger.info("[NeMo] Resources released")

    def get_supported_sample_rates(self) -> List[int]:
        """Return supported sample rates."""
        return [16000]  # NeMo models require 16kHz

    def __repr__(self) -> str:
        return (
            f"NemoSpeechSegmenter("
            f"variant={self.variant!r}, "
            f"onset={self.onset}, "
            f"offset={self.offset})"
        )
