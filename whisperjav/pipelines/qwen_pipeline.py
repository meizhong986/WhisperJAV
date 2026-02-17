#!/usr/bin/env python3
"""
Dedicated Qwen3-ASR Pipeline for WhisperJAV.

A single-purpose pipeline for Qwen3-ASR transcription following the
"redundancy over reuse" principle. No backend switching, no conditionals
for other ASR types.

8-Phase Flow:
    1. Audio Extraction (48kHz)
    2. Scene Detection (optional, default: none)
    3. Speech Enhancement (optional, VRAM Block 1)
    4. Speech Segmentation / VAD (optional)
    5. ASR Transcription (VRAM Block 2, includes sentencer)
    6. Scene SRT Generation (micro-subs)
    7. SRT Stitching
    8. Sanitisation

See: docs/architecture/ADR-004-dedicated-qwen-pipeline.md
"""

import gc
import json
import os
import shutil
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import stable_whisper

from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.speech_enhancement import (
    SCENE_EXTRACTION_SR,
    create_enhancer_direct,
    enhance_scenes,
)
from whisperjav.modules.srt_postprocessing import SRTPostProcessor, normalize_language_code
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger

# Lazy imports to avoid loading heavy modules until needed:
#   - SceneDetectorFactory (from whisperjav.modules.scene_detection_backends)
#   - QwenASR (from whisperjav.modules.qwen_asr)
#   - SpeechSegmenterFactory (from whisperjav.modules.speech_segmentation)
#   - torch (for CUDA cleanup)


class InputMode(Enum):
    """
    Controls how audio is chunked before being passed to the ASR.

    The Qwen3-ASR model is a Large Audio-Language Model (LALM) designed for
    long-form transcription. Tiny fragments strip context, causing hallucinations
    which then cause the ForcedAligner to fail.

    Modes:
        ASSEMBLY (Decoupled Assembly Line):
            Separates text generation from forced alignment into distinct
            VRAM-exclusive phases. Flow:
                Batch(ASR text-only) → Sanitize → VRAM Swap → Batch(Align)
            Benefits: higher batch size (ASR-only needs less VRAM), mid-pipeline
            hallucination removal prevents aligner NULL failures, and creates
            the architectural socket for future vLLM integration.
            Scene max duration enforced at 120s (aligner hard limit is 180s).

        CONTEXT_AWARE:
            Feeds full scenes (~30-90s chunks) directly to coupled ASR+Aligner.
            Preserves LALM context. Good quality but limited to batch_size=1
            because both models are loaded simultaneously.

        VAD_SLICING:
            Chops audio into VAD segments (up to ~29s groups via TEN).
            Destroys LALM context but useful for regression testing or
            when VAD-based filtering is specifically needed.
    """
    ASSEMBLY = "assembly"
    CONTEXT_AWARE = "context_aware"
    VAD_SLICING = "vad_slicing"


class TimestampMode(Enum):
    """
    Controls how subtitle timestamps are resolved when VAD is active.

    Bridges Phase 4 (Speech Segmentation) and Phase 5 (ASR) — each VAD group
    has known time boundaries from the speech segmenter, and the ASR may also
    produce word-level timestamps via the ForcedAligner. This mode controls
    which source is used and how they interact.

    Modes:
        ALIGNER_WITH_INTERPOLATION (New Default):
            Primary: aligner word timestamps → granular sentencing.
            For NULL timestamps: mathematically interpolate based on character
            length between valid anchor timestamps. Creates smooth, readable
            subtitles instead of snapping to VAD boundaries.

        ALIGNER_WITH_VAD_FALLBACK (Legacy):
            Primary: aligner word timestamps → granular sentencing.
            Fallback: VAD group boundaries for segments where aligner returned null.
            Subtitles snap to coarse VAD boundaries when aligner fails.

        ALIGNER_ONLY:
            Aligner timestamps only. Segments with null timestamps keep zeros.
            Use if you want to diagnose aligner failures without masking.

        VAD_ONLY:
            VAD group boundaries only. Aligner timestamps are discarded.
            Use if the aligner is unreliable for a particular audio type.
    """
    ALIGNER_WITH_INTERPOLATION = "aligner_interpolation"
    ALIGNER_WITH_VAD_FALLBACK = "aligner_vad_fallback"
    ALIGNER_ONLY = "aligner_only"
    VAD_ONLY = "vad_only"


class QwenPipeline(BasePipeline):
    """
    Dedicated pipeline for Qwen3-ASR transcription.

    Orchestrates the 8-phase flow using existing modules.
    Follows the "redundancy over reuse" principle — no backend switching.
    """

    def __init__(
        self,
        # Standard pipeline params (passed to BasePipeline)
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        keep_temp_files: bool = False,
        progress_display=None,

        # === Input Mode Configuration (v1.8.7+) ===
        # Controls audio input strategy for Qwen3-ASR (LALM)
        qwen_input_mode: str = "vad_slicing",  # "vad_slicing" (default) or "context_aware"
        qwen_safe_chunking: bool = True,  # Enforce scene boundaries for context-aware mode

        # Scene detection (Phase 2)
        # Default to "semantic" for context-aware mode - it has true MERGE logic
        # to guarantee min_duration is met (unlike auditok/silero which only filter)
        scene_detector: str = "semantic",

        # Speech enhancement (Phase 3)
        speech_enhancer: str = "none",
        speech_enhancer_model: Optional[str] = None,

        # Speech segmentation / VAD (Phase 4)
        speech_segmenter: str = "ten",  # Default to TEN backend for VAD
        segmenter_max_group_duration: float = 29.0,  # Max group size in seconds (CLI: --qwen-max-group-duration)
        segmenter_config: Optional[Dict[str, Any]] = None,  # GUI/CLI custom segmenter params

        # Qwen ASR (Phase 5)
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: str = "Japanese",
        timestamps: str = "word",
        aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        context: str = "",
        context_file: Optional[str] = None,
        attn_implementation: str = "auto",

        # Timestamp resolution (bridges Phase 4 VAD and Phase 5 ASR)
        timestamp_mode: str = "aligner_interpolation",  # New default for context-aware

        # Japanese post-processing — DISABLED for Qwen3 (C2 fix, ADR-006).
        # Qwen3 uses AssemblyTextCleaner, not the Whisper-era JapanesePostProcessor.
        # Parameter kept for backward compatibility; value is always overridden to False.
        japanese_postprocess: bool = False,
        postprocess_preset: str = "high_moan",

        # Assembly text cleaner (Step 4 of assembly mode)
        assembly_cleaner: bool = True,  # Enable/disable pre-alignment text cleaning

        # Adaptive Step-Down (v1.8.10+)
        stepdown_enabled: bool = True,
        stepdown_initial_group: float = 30.0,
        stepdown_fallback_group: float = 6.0,

        # Generation safety controls (v1.8.9+)
        repetition_penalty: float = 1.1,              # Pipeline default: conservative penalty for JAV
        max_tokens_per_audio_second: float = 20.0,    # Pipeline default: dynamic scaling enabled

        # Output
        subs_language: str = "native",

        **kwargs,
    ):
        """Initialize QwenPipeline with Qwen-specific configuration."""
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            **kwargs,
        )

        # Progress display
        self.progress_display = progress_display

        # === Context-Aware Chunking Configuration ===
        # Input mode: controls whether to feed full scenes (context_aware) or
        # tiny VAD fragments (vad_slicing) to the ASR
        try:
            self.input_mode = InputMode(qwen_input_mode)
        except ValueError:
            logger.warning(
                "Unknown qwen_input_mode '%s', defaulting to 'context_aware'",
                qwen_input_mode,
            )
            self.input_mode = InputMode.CONTEXT_AWARE

        # Safe chunking: when True, enforces scene boundaries to stay
        # within ForcedAligner's 180s architectural limit
        self.safe_chunking = qwen_safe_chunking

        # Scene detection config
        self.scene_method = scene_detector

        # Speech enhancement config
        self.enhancer_backend = speech_enhancer
        self.enhancer_model = speech_enhancer_model

        # Speech segmentation config
        self.segmenter_backend = speech_segmenter
        self.segmenter_max_group_duration = segmenter_max_group_duration
        self.segmenter_config = segmenter_config or {}

        # Adaptive Step-Down config (v1.8.10+)
        self.stepdown_enabled = stepdown_enabled
        self.stepdown_initial_group = stepdown_initial_group
        self.stepdown_fallback_group = stepdown_fallback_group

        # Timestamp resolution mode
        try:
            self.timestamp_mode = TimestampMode(timestamp_mode)
        except ValueError:
            logger.warning(
                "Unknown timestamp_mode '%s', defaulting to 'aligner_interpolation'",
                timestamp_mode,
            )
            self.timestamp_mode = TimestampMode.ALIGNER_WITH_INTERPOLATION

        # Deprecation warning: japanese_postprocess is always False for Qwen3
        if japanese_postprocess:
            logger.warning(
                "japanese_postprocess=True has no effect for QwenPipeline. "
                "Qwen3-ASR uses AssemblyTextCleaner, not JapanesePostProcessor. "
                "This parameter is deprecated and will be removed in a future version."
            )

        # Assembly text cleaner toggle (for --qwen-assembly-cleaner on|off)
        self.assembly_cleaner_enabled = assembly_cleaner

        # Qwen ASR config (stored as dict for deferred construction)
        self._asr_config = {
            "model_id": model_id,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "language": language,
            "timestamps": timestamps,
            "use_aligner": timestamps == "word",
            "aligner_id": aligner_id,
            "context": self._resolve_context(context, context_file),
            "attn_implementation": attn_implementation,
            "japanese_postprocess": False,  # C2 fix: Qwen3 uses AssemblyTextCleaner, not JapanesePostProcessor
            "postprocess_preset": postprocess_preset,
            "repetition_penalty": repetition_penalty,
            "max_tokens_per_audio_second": max_tokens_per_audio_second,
        }
        self.model_id = model_id

        # Output config
        self.subs_language = subs_language
        self.lang_code = normalize_language_code(language or "ja")

        # Shared modules (lightweight, safe to create in __init__)
        self.audio_extractor = AudioExtractor(sample_rate=SCENE_EXTRACTION_SR)
        self.stitcher = SRTStitcher()
        self.postprocessor = SRTPostProcessor(language=self.lang_code)

        # Decoupled Subtitle Pipeline (assembly mode only, ADR-006)
        # Components are lightweight — no model loading until process() time
        self._subtitle_pipeline = None
        if self.input_mode == InputMode.ASSEMBLY:
            self._subtitle_pipeline = self._build_subtitle_pipeline()

        logger.debug(
            "[QwenPipeline PID %s] Initialized (model=%s, input_mode=%s, safe_chunking=%s, "
            "scene=%s, enhancer=%s, segmenter=%s, timestamps=%s)",
            os.getpid(), model_id, self.input_mode.value, self.safe_chunking,
            scene_detector, speech_enhancer, speech_segmenter, self.timestamp_mode.value,
        )

    def cleanup(self):
        """Release pipeline resources including the subtitle pipeline orchestrator."""
        # Clean up DecoupledSubtitlePipeline components
        if self._subtitle_pipeline is not None:
            try:
                self._subtitle_pipeline.cleanup()
            except Exception as e:
                logger.warning("Subtitle pipeline cleanup failed (non-fatal): %s", e)
            self._subtitle_pipeline = None

        # Delegate to base class (handles scene_detector, enhancer, asr, CUDA)
        super().cleanup()

    def get_mode_name(self) -> str:
        return "qwen"

    # ------------------------------------------------------------------
    # Context resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_context(context: str, context_file: Optional[str]) -> str:
        """Resolve context from inline string and/or file path."""
        parts = []
        if context:
            parts.append(context)
        if context_file:
            try:
                file_content = Path(context_file).read_text(encoding="utf-8").strip()
                if file_content:
                    parts.append(file_content)
                    logger.info(f"Loaded context from file: {context_file} ({len(file_content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to load context file '{context_file}': {e}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Decoupled Subtitle Pipeline construction (ADR-006)
    # ------------------------------------------------------------------

    def _build_subtitle_pipeline(self):
        """
        Construct the DecoupledSubtitlePipeline for assembly mode.

        Components are lightweight wrappers — no models are loaded until
        the orchestrator calls load() during process_scenes().  This is
        safe to call in __init__().
        """
        from whisperjav.modules.assembly_text_cleaner import AssemblyCleanerConfig
        from whisperjav.modules.subtitle_pipeline.aligners.factory import TextAlignerFactory
        from whisperjav.modules.subtitle_pipeline.cleaners.factory import TextCleanerFactory
        from whisperjav.modules.subtitle_pipeline.framers.factory import TemporalFramerFactory
        from whisperjav.modules.subtitle_pipeline.generators.factory import TextGeneratorFactory
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
        from whisperjav.modules.subtitle_pipeline.types import (
            HardeningConfig,
        )
        from whisperjav.modules.subtitle_pipeline.types import (
            TimestampMode as NewTimestampMode,
        )

        cfg = self._asr_config

        # TemporalFramer: full-scene (assembly processes entire scenes)
        framer = TemporalFramerFactory.create("full-scene")

        # TextGenerator: Qwen3 text-only mode
        generator = TextGeneratorFactory.create(
            "qwen3",
            model_id=cfg["model_id"],
            device=cfg["device"],
            dtype=cfg["dtype"],
            batch_size=cfg["batch_size"],
            max_new_tokens=cfg["max_new_tokens"],
            language=cfg["language"],
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens_per_audio_second=cfg["max_tokens_per_audio_second"],
            attn_implementation=cfg["attn_implementation"],
        )

        # TextCleaner: Qwen3 assembly cleaner (or passthrough if disabled)
        if self.assembly_cleaner_enabled:
            cleaner_config = AssemblyCleanerConfig(enabled=True)
            cleaner = TextCleanerFactory.create(
                "qwen3",
                config=cleaner_config,
                language=cfg.get("language", "ja"),
            )
        else:
            cleaner = TextCleanerFactory.create("passthrough")

        # TextAligner: Qwen3 ForcedAligner (or None if timestamps disabled)
        aligner = None
        if cfg.get("use_aligner", True):
            aligner = TextAlignerFactory.create(
                "qwen3",
                aligner_id=cfg["aligner_id"],
                device=cfg["device"],
                dtype=cfg["dtype"],
                language=cfg["language"],
            )

        # Map old TimestampMode enum to new
        new_ts_mode = NewTimestampMode(self.timestamp_mode.value)

        return DecoupledSubtitlePipeline(
            framer=framer,
            generator=generator,
            cleaner=cleaner,
            aligner=aligner,
            hardening_config=HardeningConfig(timestamp_mode=new_ts_mode),
            language=cfg.get("language", "ja"),
        )

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through the 8-phase Qwen pipeline.

        Args:
            media_info: Dict with 'path', 'basename', 'type', 'duration', etc.

        Returns:
            Master metadata dictionary with paths, stats, quality metrics.
        """
        input_file = Path(media_info["path"])
        media_basename = media_info["basename"]
        pipeline_start = time.time()

        logger.info(
            "[QwenPipeline PID %s] Processing: %s (model=%s)",
            os.getpid(), input_file.name, self.model_id,
        )

        master_metadata = {
            "metadata_master": {
                "structure_version": "1.0.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "input_file": str(input_file),
            "basename": media_basename,
            "pipeline": "qwen",
            "model_id": self.model_id,
            "stages": {},
            "output_files": {},
            "summary": {},
        }

        # ==============================================================
        # PHASE 1: AUDIO EXTRACTION
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 1: Extracting audio from %s", os.getpid(), input_file.name)
        phase1_start = time.time()

        audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
        extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)

        master_metadata["duration_seconds"] = duration
        master_metadata["stages"]["extraction"] = {
            "audio_path": str(extracted_audio),
            "duration": duration,
            "sample_rate": SCENE_EXTRACTION_SR,
            "time_sec": time.time() - phase1_start,
        }
        logger.info("[QwenPipeline PID %s] Phase 1: Complete (%.1fs audio)", os.getpid(), duration)

        # ==============================================================
        # PHASE 2: SCENE DETECTION (optional, default: none)
        # ==============================================================
        # When safe_chunking is enabled, enforce scene boundaries
        # to stay within the ForcedAligner's 180s architectural limit.
        logger.info(
            "[QwenPipeline PID %s] Phase 2: Scene detection (method=%s, safe_chunking=%s)",
            os.getpid(), self.scene_method, self.safe_chunking,
        )
        phase2_start = time.time()

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)

        from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

        # Configure scene detector parameters
        scene_detector_kwargs = {"method": self.scene_method}

        # === Safe Chunking Override (v1.8.7+) ===
        # When safe_chunking is True, enforce scene boundaries.
        # Assembly mode uses wider windows (up to 120s) to preserve LALM
        # context while staying under the ForcedAligner 180s hard limit.
        # Other modes use tighter windows (30-90s).
        if self.safe_chunking:
            if self.input_mode == InputMode.ASSEMBLY:
                # Assembly mode: wider scenes for LALM context, but strictly
                # under the aligner's 180s limit (120s gives safe headroom)
                scene_detector_kwargs["min_duration"] = 30   # seconds
                scene_detector_kwargs["max_duration"] = 120  # seconds
                logger.info(
                    "[QwenPipeline PID %s] Phase 2: Assembly safe chunking "
                    "(min=30s, max=120s, aligner limit=180s)",
                    os.getpid(),
                )
            else:
                # Context-aware / VAD modes: allow shorter scenes so the
                # detector can split at natural silence boundaries closer
                # to 15s.  Combined with step-down (Tier 1 30s → Tier 2 6s)
                # this gives the aligner the best chance per group.
                scene_detector_kwargs["min_duration"] = 12  # seconds
                scene_detector_kwargs["max_duration"] = 90  # seconds
                logger.info(
                    "[QwenPipeline PID %s] Phase 2: Safe chunking enabled "
                    "(min=12s, max=90s)",
                    os.getpid(),
                )

        scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_detector_kwargs)
        result = scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
        scene_paths = result.to_legacy_tuples()
        scene_detector.cleanup()
        logger.info(
            "[QwenPipeline PID %s] Phase 2: Detected %d scenes (method=%s)",
            os.getpid(), len(scene_paths), self.scene_method,
        )

        # Store full scene detection metadata (matching balanced/fidelity pattern)
        detection_meta = result.to_metadata_dict()
        master_metadata["stages"]["scene_detection"] = {
            "method": self.scene_method,
            "scenes_detected": len(scene_paths),
            "time_sec": time.time() - phase2_start,
        }
        # Include structured per-scene data for diagnostics/benchmarking
        if detection_meta.get("scenes_detected"):
            master_metadata["scenes_detected"] = detection_meta["scenes_detected"]
        if detection_meta.get("coarse_boundaries"):
            master_metadata["coarse_boundaries"] = detection_meta["coarse_boundaries"]

        # ==============================================================
        # PHASE 3: SPEECH ENHANCEMENT (optional, VRAM Block 1)
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 3: Speech enhancement (backend=%s)", os.getpid(), self.enhancer_backend)
        phase3_start = time.time()

        enhancer = create_enhancer_direct(
            backend=self.enhancer_backend,
            model=self.enhancer_model,
        )

        def _enhancement_progress(scene_num, total, scene_name):
            logger.debug(f"Enhancing scene {scene_num}/{total}: {scene_name}")

        scene_paths = enhance_scenes(
            scene_paths, enhancer, self.temp_dir,
            progress_callback=_enhancement_progress,
        )

        # VRAM Block 1 cleanup — release enhancer before loading ASR
        enhancer.cleanup()
        del enhancer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        master_metadata["stages"]["enhancement"] = {
            "backend": self.enhancer_backend,
            "time_sec": time.time() - phase3_start,
        }
        logger.info("[QwenPipeline PID %s] Phase 3: Complete (%.1fs)", os.getpid(), time.time() - phase3_start)

        # ==============================================================
        # PHASE 4: SPEECH SEGMENTATION / VAD (optional)
        # ==============================================================
        speech_regions_per_scene = {}  # scene_idx -> SegmentationResult

        if self.segmenter_backend != "none":
            logger.info("[QwenPipeline PID %s] Phase 4: Speech segmentation (backend=%s)", os.getpid(), self.segmenter_backend)
            phase4_start = time.time()

            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
            segmenter_kwargs = {"max_group_duration_s": self.segmenter_max_group_duration}
            segmenter_kwargs.update(self.segmenter_config)
            segmenter = SpeechSegmenterFactory.create(
                self.segmenter_backend,
                **segmenter_kwargs,
            )

            for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                try:
                    seg_result = segmenter.segment(scene_path, sample_rate=16000)
                    speech_regions_per_scene[idx] = seg_result
                    logger.debug(
                        "Phase 4: Scene %d/%d — %d speech segments (coverage=%.1f%%)",
                        idx + 1, len(scene_paths),
                        len(seg_result.segments),
                        seg_result.speech_coverage_ratio * 100,
                    )
                except Exception as e:
                    logger.warning(f"Phase 4: Scene {idx + 1} segmentation failed: {e}, will transcribe full scene")

            segmenter.cleanup()
            del segmenter

            master_metadata["stages"]["segmentation"] = {
                "backend": self.segmenter_backend,
                "scenes_with_vad": len(speech_regions_per_scene),
                "time_sec": time.time() - phase4_start,
            }
            logger.info("[QwenPipeline PID %s] Phase 4: Complete (%.1fs)", os.getpid(), time.time() - phase4_start)
        else:
            logger.info("[QwenPipeline PID %s] Phase 4: Skipped (segmenter=none)", os.getpid())

        # Collect VAD speech regions per scene for diagnostics (Phase 2)
        vad_regions_data: Dict[int, list] = {}
        for _v_idx, _v_seg in speech_regions_per_scene.items():
            vad_regions_data[_v_idx] = [
                {"start": round(s.start_sec, 3), "end": round(s.end_sec, 3)}
                for s in _v_seg.segments
            ]

        # ==============================================================
        # PHASE 5: ASR TRANSCRIPTION (VRAM Block 2)
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 5: ASR transcription (model=%s, mode=%s)",
                    os.getpid(), self.model_id, self.input_mode.value)
        phase5_start = time.time()

        from whisperjav.modules.qwen_asr import QwenASR

        # Debug artifacts directory (master text, timestamps, merged words)
        raw_subs_dir = self.temp_dir / "raw_subs"
        raw_subs_dir.mkdir(exist_ok=True)

        scene_results: List[Tuple[Optional[stable_whisper.WhisperResult], int]] = []

        if self.input_mode == InputMode.ASSEMBLY:
            # ==============================================================
            # ASSEMBLY MODE via DecoupledSubtitlePipeline (ADR-006)
            # ==============================================================
            # Orchestrator handles the full decoupled flow:
            #   Frame → Generate → Clean → VRAM Swap → Align → Sentinel
            #   → Reconstruct → Harden
            # ==============================================================

            # Set artifacts directory for debug output
            self._subtitle_pipeline.artifacts_dir = raw_subs_dir

            # Prepare orchestrator inputs from scene_paths tuples
            orch_audio_paths = [Path(sp[0]) for sp in scene_paths]
            orch_durations = [sp[3] for sp in scene_paths]

            # Convert Phase 4 speech regions: {idx: SegmentationResult} → List[List[Tuple]]
            orch_speech_regions = None
            if speech_regions_per_scene:
                orch_speech_regions = []
                for idx in range(len(scene_paths)):
                    if idx in speech_regions_per_scene:
                        seg_result = speech_regions_per_scene[idx]
                        orch_speech_regions.append(
                            [(s.start_sec, s.end_sec) for s in seg_result.segments]
                        )
                    else:
                        orch_speech_regions.append([])

            # Run orchestrator
            orch_results = self._subtitle_pipeline.process_scenes(
                scene_audio_paths=orch_audio_paths,
                scene_durations=orch_durations,
                scene_speech_regions=orch_speech_regions,
            )

            # Convert to Phase 6 format: (WhisperResult, scene_idx)
            for idx, (result, _diag) in enumerate(orch_results):
                scene_results.append((result, idx))

            # Map orchestrator sentinel stats → existing format
            orch_stats = self._subtitle_pipeline.sentinel_stats
            self._sentinel_stats = {
                "alignment_collapses": orch_stats.get("collapsed_scenes", 0),
                "alignment_recoveries": orch_stats.get("recovered_scenes", 0),
            }

            # Save per-scene diagnostics JSON
            for idx, (_result, diag) in enumerate(orch_results):
                try:
                    diag_path = raw_subs_dir / f"scene_{idx:04d}_diagnostics.json"
                    diag_path.write_text(
                        json.dumps(diag, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        else:
            # ==============================================================
            # COUPLED MODES: Context-Aware / VAD Slicing
            # ==============================================================
            from whisperjav.modules.alignment_sentinel import (
                assess_alignment_quality,
                extract_words_from_result,
                redistribute_collapsed_words,
            )

            asr = QwenASR(**self._asr_config)

            # Context biasing: user context applies to first scene only (MVP).
            user_context = self._asr_config.get("context", "")

            # Sentinel tracking — instance-level so _transcribe_speech_regions()
            # can also accumulate stats (same contract as assembly mode)
            self._sentinel_stats = {
                "alignment_collapses": 0,
                "alignment_recoveries": 0,
            }

            # Per-scene diagnostics (Phase 2: pipeline metadata emission)
            self._scene_diagnostics: Dict[int, dict] = {}

            for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                scene_num = idx + 1
                logger.info(
                    "[QwenPipeline PID %s] Phase 5: Transcribing scene %d/%d (%.1fs)",
                    os.getpid(), scene_num, len(scene_paths), dur_sec,
                )

                try:
                    # Context biasing: user context for first scene only (MVP)
                    scene_context = user_context if idx == 0 else ""

                    # === Input Mode Branching ===
                    if self.input_mode == InputMode.CONTEXT_AWARE:
                        logger.debug(
                            "Phase 5: Scene %d — CONTEXT_AWARE mode, transcribing full scene (%.1fs)",
                            scene_num, dur_sec,
                        )
                        result = asr.transcribe(
                            scene_path,
                            context=scene_context if scene_context else None,
                            artifacts_dir=raw_subs_dir,
                        )

                        # Diagnostics tracking
                        _diag_sentinel = {"status": "N/A", "assessment": None, "recovery": None}
                        _diag_interp_count = 0

                        if result and result.segments:
                            # ── Alignment Sentinel: detect collapsed timestamps ──
                            words = extract_words_from_result(result)
                            assessment = assess_alignment_quality(words, dur_sec)
                            _diag_sentinel = {
                                "status": assessment["status"],
                                "assessment": assessment,
                                "recovery": None,
                            }

                            if assessment["status"] == "COLLAPSED":
                                self._sentinel_stats["alignment_collapses"] += 1
                                logger.warning(
                                    "[SENTINEL] Scene %d/%d: Alignment Collapse — "
                                    "coverage=%.1f%%, CPS=%.1f, %d chars in %.3fs span",
                                    scene_num, len(scene_paths),
                                    assessment["coverage_ratio"] * 100,
                                    assessment["aggregate_cps"],
                                    assessment["char_count"],
                                    assessment["word_span_sec"],
                                )

                                # VAD regions for this scene (if available from Phase 4)
                                vad_regions = None
                                if idx in speech_regions_per_scene:
                                    vad_regions = [
                                        (s.start_sec, s.end_sec)
                                        for s in speech_regions_per_scene[idx].segments
                                    ]

                                corrected_words = redistribute_collapsed_words(
                                    words, dur_sec, vad_regions,
                                )
                                result = self._reconstruct_from_words(
                                    corrected_words, scene_path,
                                )
                                self._sentinel_stats["alignment_recoveries"] += 1

                                recovery_method = "VAD_RESCUE" if vad_regions else "PROPORTIONAL"
                                _diag_sentinel["recovery"] = {
                                    "strategy": "vad_guided" if vad_regions else "proportional",
                                    "words_redistributed": len(corrected_words),
                                }
                                logger.info(
                                    "[SENTINEL] Scene %d/%d: Recovered via %s",
                                    scene_num, len(scene_paths), recovery_method,
                                )

                            # Interpolation handles remaining null gaps in the
                            # (possibly recovered) result
                            if self.timestamp_mode == TimestampMode.ALIGNER_WITH_INTERPOLATION:
                                _diag_interp_count = self._apply_timestamp_interpolation(result)
                                if _diag_interp_count:
                                    logger.debug(
                                        "Phase 5: Scene %d — interpolated timestamps for %d segments",
                                        scene_num, _diag_interp_count,
                                    )

                        # Build per-scene diagnostics
                        _total_segs = len(result.segments) if result and result.segments else 0
                        self._scene_diagnostics[idx] = {
                            "schema_version": "1.1.0",
                            "scene_index": idx,
                            "scene_start_sec": start_sec,
                            "scene_end_sec": end_sec,
                            "scene_duration_sec": dur_sec,
                            "input_mode": self.input_mode.value,
                            "sentinel": _diag_sentinel,
                            "timing_sources": {
                                "aligner_native": max(0, _total_segs - _diag_interp_count),
                                "interpolated": _diag_interp_count,
                                "total_segments": _total_segs,
                            },
                            "vad_regions": vad_regions_data.get(idx, []),
                        }

                    elif self.input_mode == InputMode.VAD_SLICING and idx in speech_regions_per_scene:
                        logger.debug(
                            "Phase 5: Scene %d — VAD_SLICING mode, transcribing %d speech regions",
                            scene_num, len(speech_regions_per_scene[idx].groups),
                        )
                        result = self._transcribe_speech_regions(
                            scene_path, speech_regions_per_scene[idx],
                            asr=asr,
                            context=scene_context if scene_context else None,
                            artifacts_dir=raw_subs_dir,
                            scene_idx=idx,
                            scene_start_sec=start_sec,
                            scene_end_sec=end_sec,
                            scene_duration_sec=dur_sec,
                        )
                    else:
                        logger.debug(
                            "Phase 5: Scene %d — no VAD data, transcribing full scene (%.1fs)",
                            scene_num, dur_sec,
                        )
                        result = asr.transcribe(
                            scene_path,
                            context=scene_context if scene_context else None,
                            artifacts_dir=raw_subs_dir,
                        )

                        # Diagnostics tracking (fallback path)
                        _diag_sentinel = {"status": "N/A", "assessment": None, "recovery": None}

                        # ── Alignment Sentinel: detect collapsed timestamps ──
                        if result and result.segments:
                            words = extract_words_from_result(result)
                            assessment = assess_alignment_quality(words, dur_sec)
                            _diag_sentinel = {
                                "status": assessment["status"],
                                "assessment": assessment,
                                "recovery": None,
                            }

                            if assessment["status"] == "COLLAPSED":
                                self._sentinel_stats["alignment_collapses"] += 1
                                logger.warning(
                                    "[SENTINEL] Scene %d/%d: Alignment Collapse — "
                                    "coverage=%.1f%%, CPS=%.1f, %d chars in %.3fs span",
                                    scene_num, len(scene_paths),
                                    assessment["coverage_ratio"] * 100,
                                    assessment["aggregate_cps"],
                                    assessment["char_count"],
                                    assessment["word_span_sec"],
                                )

                                corrected_words = redistribute_collapsed_words(
                                    words, dur_sec,
                                )
                                result = self._reconstruct_from_words(
                                    corrected_words, scene_path,
                                )
                                self._sentinel_stats["alignment_recoveries"] += 1
                                _diag_sentinel["recovery"] = {
                                    "strategy": "proportional",
                                    "words_redistributed": len(corrected_words),
                                }
                                logger.info(
                                    "[SENTINEL] Scene %d/%d: Recovered via PROPORTIONAL",
                                    scene_num, len(scene_paths),
                                )

                        # Build per-scene diagnostics (fallback path)
                        _total_segs = len(result.segments) if result and result.segments else 0
                        self._scene_diagnostics[idx] = {
                            "schema_version": "1.1.0",
                            "scene_index": idx,
                            "scene_start_sec": start_sec,
                            "scene_end_sec": end_sec,
                            "scene_duration_sec": dur_sec,
                            "input_mode": "fallback",
                            "sentinel": _diag_sentinel,
                            "timing_sources": {
                                "aligner_native": _total_segs,
                                "total_segments": _total_segs,
                            },
                            "vad_regions": vad_regions_data.get(idx, []),
                        }

                    segment_count = len(result.segments) if result and result.segments else 0
                    scene_results.append((result, idx))

                    if segment_count == 0:
                        logger.info(f"Phase 5: Scene {scene_num} produced 0 segments (may be non-speech audio)")
                    else:
                        logger.debug(f"Phase 5: Scene {scene_num} produced {segment_count} segments")

                except Exception as e:
                    logger.warning(f"Phase 5: Scene {scene_num} failed: {e}, skipping")
                    scene_results.append((None, idx))

            # VRAM Block 2 cleanup (coupled modes)
            asr.cleanup()
            del asr
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # self._sentinel_stats already accumulated during the loop
            # (including from _transcribe_speech_regions() calls)

            # Save per-scene diagnostics JSON (Phase 2)
            for _d_idx, _d_data in getattr(self, "_scene_diagnostics", {}).items():
                try:
                    diag_path = raw_subs_dir / f"scene_{_d_idx:04d}_diagnostics.json"
                    diag_path.write_text(
                        json.dumps(_d_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

        # Sentinel stats (populated by both assembly and coupled modes)
        sentinel_stats = getattr(self, "_sentinel_stats", {})

        master_metadata["stages"]["asr"] = {
            "model_id": self.model_id,
            "input_mode": self.input_mode.value,
            "timestamp_mode": self.timestamp_mode.value,
            "safe_chunking": self.safe_chunking,
            "scenes_transcribed": sum(1 for r, _ in scene_results if r is not None and r.segments),
            "scenes_empty": sum(1 for r, _ in scene_results if r is not None and not r.segments),
            "scenes_failed": sum(1 for r, _ in scene_results if r is None),
            "alignment_collapses": sentinel_stats.get("alignment_collapses", 0),
            "alignment_recoveries": sentinel_stats.get("alignment_recoveries", 0),
            "time_sec": time.time() - phase5_start,
        }
        logger.info("[QwenPipeline PID %s] Phase 5: Complete (%.1fs)", os.getpid(), time.time() - phase5_start)

        # ==============================================================
        # PHASE 6: SCENE SRT GENERATION (micro-subs)
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 6: Generating scene SRT files", os.getpid())

        scene_srts_dir = self.temp_dir / "scene_srts"
        scene_srts_dir.mkdir(exist_ok=True)
        scene_srt_info: List[Tuple[Path, float]] = []

        for result, idx in scene_results:
            if result is None or not result.segments:
                continue

            scene_path, start_sec, end_sec, dur_sec = scene_paths[idx]
            scene_srt_path = scene_srts_dir / f"{media_basename}_scene_{idx:04d}.srt"

            try:
                result.to_srt_vtt(
                    str(scene_srt_path),
                    word_level=False,
                    segment_level=True,
                    strip=True,
                )

                if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                    scene_srt_info.append((scene_srt_path, start_sec))
                    logger.debug(f"Phase 6: Generated {scene_srt_path.name}")
                else:
                    logger.warning(f"Phase 6: Scene {idx + 1} SRT is empty after generation")
            except Exception as e:
                logger.warning(f"Phase 6: Failed to generate SRT for scene {idx + 1}: {e}")

        # ==============================================================
        # PHASE 7: STITCHING
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 7: Stitching %d scene SRTs", os.getpid(), len(scene_srt_info))

        stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"

        if scene_srt_info:
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            logger.info("[QwenPipeline PID %s] Phase 7: Stitched %d subtitles", os.getpid(), num_subtitles)
        else:
            # No valid scene SRTs — create empty file
            stitched_srt_path.write_text("", encoding="utf-8")
            num_subtitles = 0
            logger.warning("[QwenPipeline PID %s] Phase 7: No scene SRTs to stitch (0 subtitles)", os.getpid())

        master_metadata["stages"]["stitching"] = {
            "total_subtitles": num_subtitles,
            "scenes_contributed": len(scene_srt_info),
        }

        # ==============================================================
        # PHASE 8: SANITISATION
        # ==============================================================
        # NOTE: The legacy WhisperJAV sanitizer (SubtitleSanitizer + TimingAdjuster)
        # was designed for Whisper's artefacts and hallucination patterns.
        # For Qwen pipeline, Phase 8 is bypassed until a Qwen-specific
        # sanitizer is implemented.  The stitched SRT from Phase 7 is used
        # as the final output directly.
        logger.info("[QwenPipeline PID %s] Phase 8: Skipped (legacy sanitizer disabled for Qwen)", os.getpid())
        phase8_start = time.time()

        final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"

        if num_subtitles > 0:
            shutil.copy2(stitched_srt_path, final_srt_path)
            stats = {"total_subtitles": num_subtitles, "sanitizer_skipped": True}
            processed_path = final_srt_path
            logger.info(
                "[QwenPipeline PID %s] Phase 8: %d subtitles passed through (no sanitization)",
                os.getpid(), num_subtitles,
            )
        else:
            # Copy empty stitched file as final output
            final_srt_path.write_text("", encoding="utf-8")
            stats = {"total_subtitles": 0}
            processed_path = final_srt_path

        master_metadata["stages"]["sanitisation"] = {
            "stats": stats,
            "time_sec": time.time() - phase8_start,
        }
        master_metadata["srt_path"] = str(processed_path)

        # Metadata contract: keys expected by main.py's aggregate_subtitle_metrics()
        master_metadata["output_files"]["final_srt"] = str(final_srt_path)
        master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
        master_metadata["summary"]["final_subtitles_refined"] = (
            stats.get("total_subtitles", 0) - stats.get("empty_removed", 0)
        )
        master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
        master_metadata["summary"]["quality_metrics"] = {
            "hallucinations_removed": stats.get("removed_hallucinations", 0),
            "repetitions_removed": stats.get("removed_repetitions", 0),
            "duration_adjustments": stats.get("duration_adjustments", 0),
            "empty_removed": stats.get("empty_removed", 0),
            "cps_filtered": stats.get("cps_filtered", 0),
            "logprob_filtered": 0,       # N/A for Qwen pipeline
            "nonverbal_filtered": 0,     # N/A for Qwen pipeline
        }

        # ==============================================================
        # PHASE 9: ANALYTICS
        # ==============================================================
        try:
            from whisperjav.modules.pipeline_analytics import (
                compute_analytics,
                print_summary,
                save_analytics,
            )
            analytics = compute_analytics(
                raw_subs_dir, final_srt_path, title=media_basename,
            )
            print_summary(analytics, title=media_basename)

            # Save analytics JSON alongside the final SRT
            analytics_path = final_srt_path.with_suffix(".analytics.json")
            save_analytics(analytics, analytics_path)
            master_metadata["output_files"]["analytics"] = str(analytics_path)
        except Exception as e:
            logger.debug("Phase 9: Analytics failed (non-fatal): %s", e)

        # ==============================================================
        # COMPLETE
        # ==============================================================
        total_time = time.time() - pipeline_start
        master_metadata["total_time_sec"] = total_time
        master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)

        logger.info(
            "[QwenPipeline PID %s] Complete: %s (%d subtitles in %s)",
            os.getpid(),
            final_srt_path.name,
            master_metadata["summary"]["final_subtitles_refined"],
            str(timedelta(seconds=int(total_time))),
        )

        # Save metadata JSON if debug mode
        if self.save_metadata_json:
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)

        # Cleanup temp files
        self.cleanup_temp_files(media_basename)

        return master_metadata

    # ------------------------------------------------------------------
    # Pre-ASR speech region transcription (WhisperProASR pattern)
    # ------------------------------------------------------------------

    def _transcribe_speech_regions(
        self,
        scene_path: Path,
        seg_result,
        asr,
        context: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        scene_idx: Optional[int] = None,
        scene_start_sec: float = 0.0,
        scene_end_sec: float = 0.0,
        scene_duration_sec: float = 0.0,
    ) -> Optional[stable_whisper.WhisperResult]:
        """
        Transcribe individual speech regions from a scene.

        Follows the WhisperProASR pattern: for each VAD group, slice the scene
        audio to extract only the speech portion, transcribe that clip, and
        offset timestamps back to scene-relative time.

        When stepdown_enabled=True, uses a two-tier strategy:
            Tier 1: Try context-rich groups (30s) for better ASR quality.
            Tier 2: Re-group collapsed Tier 1 segments at tight (8s) groups.

        Args:
            scene_path: Path to scene WAV (16kHz mono, from Phase 3)
            seg_result: SegmentationResult from Phase 4
            asr: QwenASR instance
            context: Optional context string for ASR
            artifacts_dir: Optional dir for debug artifacts

        Returns:
            Combined WhisperResult with all speech regions, or None if empty.
        """
        import numpy as np
        import soundfile as sf

        from whisperjav.modules.alignment_sentinel import (
            assess_alignment_quality,
            extract_words_from_result,
            redistribute_collapsed_words,
        )

        # Read scene audio once
        audio_data, sr = sf.read(str(scene_path))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        speech_regions_dir = self.temp_dir / "speech_regions"
        speech_regions_dir.mkdir(exist_ok=True)

        # Per-scene diagnostics accumulators
        _diag = {
            "collapses": 0, "recoveries": 0,
            "interp_count": 0, "vad_fallback_count": 0,
            "total_fallback_applied": 0,
            # Step-down specific
            "tier1_groups": 0, "tier1_accepted": 0, "tier1_collapsed": 0,
            "tier2_groups": 0, "tier2_accepted": 0, "tier2_collapsed": 0,
            # Per-group detail records (v1.1.0 analytics)
            "group_details": [],
        }

        def _transcribe_group(group, group_idx, total_groups, tag="",
                              skip_recovery_on_collapse=False, tier=0):
            """Transcribe a single VAD group.

            Args:
                skip_recovery_on_collapse: If True, return early on COLLAPSED
                    without applying recovery (for step-down Tier 1 deferral).
                tier: Step-down tier (0=standard, 1=Tier 1, 2=Tier 2).
                    Used only for per-group analytics records.

            Returns:
                (result, assessment_status) where result is a WhisperResult
                with timestamps offset to scene-relative time, or None.
                assessment_status is "OK", "COLLAPSED", or None if no result.
            """
            if not group:
                return None, None

            group_start_sec = group[0].start_sec
            group_end_sec = group[-1].end_sec
            group_duration = group_end_sec - group_start_sec

            # Per-group detail record for analytics (v1.1.0).
            # Built progressively, appended to _diag at every exit point.
            _detail = {
                "tier": tier,
                "group_index": group_idx,
                "time_range": [round(group_start_sec, 3),
                               round(group_end_sec, 3)],
                "duration_sec": round(group_duration, 3),
                "segments_in_group": len(group),
                "outcome": "skipped",
                "sentinel_status": None,
                "sentinel_triggers": [],
                "assessment_snapshot": None,
                "subs_produced": 0,
            }

            if group_duration < 0.1:
                logger.debug(
                    "Phase 5%s: Skipping speech region %d/%d (%.3fs < 0.1s)",
                    tag, group_idx + 1, total_groups, group_duration,
                )
                _diag["group_details"].append(_detail)
                return None, None

            # Slice audio for this speech group
            start_sample = int(group_start_sec * sr)
            end_sample = int(group_end_sec * sr)
            group_audio = audio_data[start_sample:end_sample]

            # Write to temp WAV (QwenASR requires file paths)
            # Include tier tag in filename to avoid T1/T2 collisions
            tag_suffix = tag.strip().replace("[", "").replace("]", "").lower()
            if tag_suffix:
                region_stem = f"{scene_path.stem}_region_{tag_suffix}_{group_idx:04d}"
            else:
                region_stem = f"{scene_path.stem}_region_{group_idx:04d}"
            region_path = speech_regions_dir / f"{region_stem}.wav"
            sf.write(str(region_path), group_audio, sr)

            logger.debug(
                "Phase 5%s: Transcribing speech region %d/%d [%.2f-%.2fs] (%.1fs)",
                tag, group_idx + 1, total_groups,
                group_start_sec, group_end_sec, group_duration,
            )

            # Transcribe the speech region
            try:
                result = asr.transcribe(
                    region_path,
                    context=context,
                    artifacts_dir=artifacts_dir,
                )
            except Exception as e:
                logger.warning(
                    "Phase 5%s: Speech region %d/%d transcription failed: %s",
                    tag, group_idx + 1, total_groups, e,
                )
                _diag["group_details"].append(_detail)
                return None, None

            if result is None or not result.segments:
                _diag["group_details"].append(_detail)
                return None, None

            # ── Alignment Sentinel: detect collapsed timestamps ──
            words = extract_words_from_result(result)
            assessment = assess_alignment_quality(words, group_duration)
            status = assessment["status"]

            # Update detail record with assessment (v1.1.0)
            _detail["sentinel_status"] = status
            _detail["sentinel_triggers"] = assessment.get("triggers", [])
            _detail["assessment_snapshot"] = {
                "word_count": assessment.get("word_count", 0),
                "char_count": assessment.get("char_count", 0),
                "coverage_ratio": round(assessment.get("coverage_ratio", 0.0), 4),
                "aggregate_cps": round(assessment.get("aggregate_cps", 0.0), 1),
                "zero_position_ratio": round(
                    assessment.get("zero_position_ratio", 0.0), 4),
                "degenerate_ratio": round(
                    assessment.get("degenerate_ratio", 0.0), 4),
            }

            if status == "COLLAPSED":
                _diag["collapses"] += 1
                self._sentinel_stats["alignment_collapses"] += 1

                if skip_recovery_on_collapse:
                    # Caller will handle re-grouping (step-down Tier 2)
                    _detail["outcome"] = "collapsed_deferred"
                    _diag["group_details"].append(_detail)
                    return None, "COLLAPSED"

            # ── Apply recovery (if collapsed) + timestamp resolution + offset ──
            if status == "COLLAPSED":
                # Extract group-relative VAD regions for Strategy C recovery.
                # group[] contains SpeechSegments with scene-relative times;
                # offset to group-relative (0-based) for the redistributor.
                group_speech_regions = [
                    (seg.start_sec - group_start_sec,
                     seg.end_sec - group_start_sec)
                    for seg in group
                ]
                corrected_words = redistribute_collapsed_words(
                    words, group_duration, group_speech_regions,
                )
                strategy = ("VAD-GUIDED" if group_speech_regions
                            else "PROPORTIONAL")
                result = self._reconstruct_from_words(
                    corrected_words, region_path,
                )
                self._sentinel_stats["alignment_recoveries"] += 1
                _diag["recoveries"] += 1
                logger.warning(
                    "[SENTINEL]%s Speech region %d/%d: Alignment collapse "
                    "recovered (%s, %d VAD regions)",
                    tag, group_idx + 1, total_groups,
                    strategy, len(group_speech_regions),
                )

                # Update detail record with recovery info (v1.1.0)
                _strategy_key = "vad_guided" if group_speech_regions else "proportional"
                _detail["outcome"] = f"recovered_{_strategy_key}"
                _detail["recovery"] = {
                    "strategy": _strategy_key,
                    "speech_regions_count": len(group_speech_regions),
                    "words_redistributed": len(corrected_words),
                }
            else:
                _detail["outcome"] = "accepted"

            # ── Timestamp resolution ──
            if self.timestamp_mode == TimestampMode.VAD_ONLY:
                self._apply_vad_only_timestamps(result, group_duration)
            elif self.timestamp_mode == TimestampMode.ALIGNER_WITH_INTERPOLATION:
                interp = self._apply_timestamp_interpolation(result, group_duration)
                if interp:
                    _diag["total_fallback_applied"] += interp
                    _diag["interp_count"] += interp
            elif self.timestamp_mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK:
                fb = self._apply_vad_timestamp_fallback(result, group_duration)
                if fb:
                    _diag["total_fallback_applied"] += fb
                    _diag["vad_fallback_count"] += fb

            # ── Clamp to group bounds (region-relative: 0 to group_duration) ──
            # Prevents interpolation overflow or aligner drift from producing
            # timestamps beyond the group's audio slice.
            # Operate on words only (not segment properties) because stable-ts
            # Segment.start/.end setters propagate to underlying words.
            for seg in result.segments:
                if hasattr(seg, 'words') and seg.words:
                    for word in seg.words:
                        word.start = max(0.0, min(word.start, group_duration))
                        word.end = max(word.start, min(word.end, group_duration))
                else:
                    seg.start = max(0.0, min(seg.start, group_duration))
                    seg.end = max(seg.start, min(seg.end, group_duration))

            # Offset timestamps from group-relative to scene-relative
            self._offset_result_timestamps(result, group_start_sec)

            # ── Clamp to scene bounds (scene-relative: 0 to scene_duration) ──
            # Safety net: after offset, ensure no timestamp exceeds scene boundary.
            if scene_duration_sec > 0:
                for seg in result.segments:
                    if hasattr(seg, 'words') and seg.words:
                        for word in seg.words:
                            word.start = max(0.0, min(word.start, scene_duration_sec))
                            word.end = max(word.start, min(word.end, scene_duration_sec))
                    else:
                        seg.start = max(0.0, min(seg.start, scene_duration_sec))
                        seg.end = max(seg.start, min(seg.end, scene_duration_sec))

            # Finalize detail record (v1.1.0)
            _detail["subs_produced"] = (
                len(result.segments) if result and result.segments else 0
            )
            _diag["group_details"].append(_detail)

            return result, status

        # ==================================================================
        # Main dispatch: step-down vs. standard
        # ==================================================================
        combined_result = None

        if self.stepdown_enabled:
            from whisperjav.modules.speech_segmentation import group_segments

            # Tier 1: Context-rich grouping from raw segments
            tier1_groups = group_segments(
                seg_result.segments, self.stepdown_initial_group,
            )
            _diag["tier1_groups"] = len(tier1_groups)
            tier1_results = []
            collapsed_segments = []

            logger.info(
                "Phase 5 [step-down]: Tier 1 — %d groups at %.0fs from %d raw segments",
                len(tier1_groups), self.stepdown_initial_group,
                len(seg_result.segments),
            )

            for group_idx, group in enumerate(tier1_groups):
                result, status = _transcribe_group(
                    group, group_idx, len(tier1_groups), " [T1]",
                    skip_recovery_on_collapse=True, tier=1,
                )

                if status == "COLLAPSED":
                    # Queue raw segments for Tier 2 instead of recovering here
                    collapsed_segments.extend(group)
                    _diag["tier1_collapsed"] += 1
                    logger.info(
                        "Phase 5 [step-down T1]: Group %d/%d COLLAPSED — "
                        "queued %d segments for Tier 2",
                        group_idx + 1, len(tier1_groups), len(group),
                    )
                    continue

                if result is None:
                    continue

                tier1_results.append(result)
                _diag["tier1_accepted"] += 1

            # Tier 2: Tight grouping for collapsed segments only
            tier2_results = []
            if collapsed_segments:
                tier2_groups = group_segments(
                    collapsed_segments, self.stepdown_fallback_group,
                )
                _diag["tier2_groups"] = len(tier2_groups)

                logger.info(
                    "Phase 5 [step-down]: Tier 2 — %d groups at %.0fs from %d collapsed segments",
                    len(tier2_groups), self.stepdown_fallback_group,
                    len(collapsed_segments),
                )

                for group_idx, group in enumerate(tier2_groups):
                    result, status = _transcribe_group(
                        group, group_idx, len(tier2_groups), " [T2]",
                        tier=2,
                    )
                    if result is None:
                        continue

                    if status == "COLLAPSED":
                        # Final safety net: recovery already applied by _transcribe_group
                        # since _finalize_group_result handles COLLAPSED status
                        _diag["tier2_collapsed"] += 1
                    else:
                        _diag["tier2_accepted"] += 1

                    tier2_results.append(result)

            # Merge all results in timeline order
            all_results = tier1_results + tier2_results
            if all_results:
                combined_result = all_results[0]
                for r in all_results[1:]:
                    if r and r.segments:
                        combined_result.segments.extend(r.segments)
                # Sort by timeline (Tier 1 + Tier 2 may interleave)
                combined_result.segments.sort(key=lambda s: s.start)

            total_groups = _diag["tier1_groups"] + _diag["tier2_groups"]
            logger.info(
                "Phase 5 [step-down]: T1 %d/%d accepted, T2 %d/%d accepted",
                _diag["tier1_accepted"], _diag["tier1_groups"],
                _diag["tier2_accepted"], _diag["tier2_groups"],
            )

        else:
            # Standard path (no step-down)
            total_groups = len(seg_result.groups)

            for group_idx, group in enumerate(seg_result.groups):
                result, status = _transcribe_group(
                    group, group_idx, total_groups, "",
                )
                if result is None:
                    continue

                # In standard mode, COLLAPSED groups are already recovered
                # inside _transcribe_group (proportional redistribution)

                if combined_result is None:
                    combined_result = result
                else:
                    combined_result.segments.extend(result.segments)

        # Ensure all segments are in chronological order (defensive sort).
        # The step-down path sorts explicitly; this covers the standard path
        # and acts as a safety net for any edge cases.
        if combined_result and combined_result.segments:
            combined_result.segments.sort(key=lambda s: s.start)

        if combined_result and combined_result.segments:
            logger.debug(
                "Phase 5: Combined %d segments from %d speech groups",
                len(combined_result.segments), total_groups,
            )
            if _diag["total_fallback_applied"]:
                logger.info(
                    "Phase 5: VAD timestamp fallback applied to %d/%d total segments (mode=%s)",
                    _diag["total_fallback_applied"],
                    len(combined_result.segments),
                    self.timestamp_mode.value,
                )

        # Build per-scene diagnostics (VAD_SLICING)
        if scene_idx is not None and hasattr(self, "_scene_diagnostics"):
            _total_segs = len(combined_result.segments) if combined_result and combined_result.segments else 0
            _aligner_native = max(0, _total_segs - _diag["interp_count"] - _diag["vad_fallback_count"])
            self._scene_diagnostics[scene_idx] = {
                "schema_version": "1.1.0",
                "scene_index": scene_idx,
                "scene_start_sec": scene_start_sec,
                "scene_end_sec": scene_end_sec,
                "scene_duration_sec": scene_duration_sec,
                "input_mode": "vad_slicing",
                "sentinel": {
                    "status": "COLLAPSED" if _diag["collapses"] > 0 else "OK",
                    "assessment": None,
                    "recovery": {
                        "strategy": "vad_guided",
                        "groups_recovered": _diag["recoveries"],
                    } if _diag["recoveries"] > 0 else None,
                },
                "timing_sources": {
                    "aligner_native": _aligner_native,
                    "vad_fallback": _diag["vad_fallback_count"],
                    "interpolated": _diag["interp_count"],
                    "total_segments": _total_segs,
                },
                "stepdown": {
                    "enabled": self.stepdown_enabled,
                    "tier1_groups": _diag["tier1_groups"],
                    "tier1_accepted": _diag["tier1_accepted"],
                    "tier1_collapsed": _diag["tier1_collapsed"],
                    "tier2_groups": _diag["tier2_groups"],
                    "tier2_accepted": _diag["tier2_accepted"],
                    "tier2_collapsed": _diag["tier2_collapsed"],
                } if self.stepdown_enabled else None,
                "vad_regions": [
                    {"start": round(s.start_sec, 3), "end": round(s.end_sec, 3)}
                    for s in seg_result.segments
                ],
                # Per-group detail records (v1.1.0 analytics)
                "group_details": _diag["group_details"],
            }

        return combined_result

    @staticmethod
    def _offset_result_timestamps(
        result: stable_whisper.WhisperResult, offset_sec: float,
    ):
        """
        Offset all timestamps in a WhisperResult by offset_sec.

        Uses word-level operations only when the segment has words,
        because stable-ts Segment.start/.end are properties whose setters
        propagate to words[0].start / words[-1].end.  Setting both the
        segment property AND the word attribute would double-offset the
        first/last word.
        """
        for seg in result.segments:
            if hasattr(seg, 'words') and seg.words:
                for word in seg.words:
                    word.start += offset_sec
                    word.end += offset_sec
            else:
                seg.start += offset_sec
                seg.end += offset_sec

    # ------------------------------------------------------------------
    # Sentinel reconstruction helper (shared by assembly + coupled modes)
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_from_words(
        words: List[Dict],
        audio_path,
    ) -> stable_whisper.WhisperResult:
        """
        Reconstruct a WhisperResult from word dicts via transcribe_any.

        Uses stable-ts's transcribe_any() with a pre-computed inference function
        (Option A) to get proper sentence-level regrouping. This is the same
        pattern used inline in assembly Step 8, extracted for reuse by both
        assembly and coupled-mode sentinel recovery.

        Args:
            words: List of {'word': str, 'start': float, 'end': float} dicts.
            audio_path: Path to the audio file (needed by transcribe_any for
                        duration metadata, not re-transcribed).

        Returns:
            Reconstructed WhisperResult with sentence-level regrouping.
        """
        precomputed = words

        def precomputed_inference(audio, **kwargs):
            return [precomputed]

        return stable_whisper.transcribe_any(
            inference_func=precomputed_inference,
            audio=str(audio_path),
            audio_type='str',
            regroup=True,
            vad=False,
            demucs=False,
            suppress_silence=True,
            suppress_word_ts=True,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Timestamp resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_vad_timestamp_fallback(
        result: stable_whisper.WhisperResult,
        group_duration: float,
    ) -> int:
        """
        Apply VAD timestamp fallback for segments where the aligner returned null.

        When the ForcedAligner fails to assign timestamps, words get start=0.0
        and end=0.0 (from merge_master_with_timestamps). After stable-ts
        regrouping, these become segments with start=0.0 and end=0.0.

        This method detects such segments and distributes them proportionally
        by character count across the group duration.  The text is real (ASR
        produced it), and the timing is approximate but evenly spread — much
        better than stacking all segments at the same timestamp.

        Timestamps at this point are region-relative (0-based).  The caller
        applies _offset_result_timestamps() afterward to shift to scene time.

        Detection heuristic: seg.end <= 0.0 indicates aligner failure.
        A legitimate segment at the start of a clip will have end > 0.0
        (non-zero duration).

        Args:
            result: WhisperResult with region-relative timestamps
            group_duration: Duration of the VAD group in seconds

        Returns:
            Number of segments that received fallback timestamps
        """
        if not result or not result.segments:
            return 0

        # Collect segments needing fallback
        fallback_segs = [seg for seg in result.segments if seg.end <= 0.0]
        if not fallback_segs:
            return 0

        # Proportional distribution by character count
        total_chars = sum(len(seg.text.strip()) or 1 for seg in fallback_segs)
        cumulative = 0
        for seg in fallback_segs:
            seg_chars = len(seg.text.strip()) or 1
            frac_start = cumulative / total_chars
            frac_end = (cumulative + seg_chars) / total_chars
            if hasattr(seg, 'words') and seg.words:
                # Word-only assignment (avoid double-offset via property setters)
                word_total = sum(len(w.word) or 1 for w in seg.words)
                w_cum = 0
                seg_span = (frac_end - frac_start) * group_duration
                seg_base = frac_start * group_duration
                for w in seg.words:
                    w_chars = len(w.word) or 1
                    w.start = seg_base + (w_cum / word_total) * seg_span
                    w.end = seg_base + ((w_cum + w_chars) / word_total) * seg_span
                    w_cum += w_chars
            else:
                seg.start = frac_start * group_duration
                seg.end = frac_end * group_duration
            cumulative += seg_chars

        return len(fallback_segs)

    @staticmethod
    def _apply_vad_only_timestamps(
        result: stable_whisper.WhisperResult,
        group_duration: float,
    ) -> None:
        """
        Force VAD group timestamps on all segments, discarding aligner timestamps.

        Used in VAD_ONLY mode where aligner timestamps are intentionally ignored
        in favour of the speech segmenter's time boundaries.

        All segments receive the full group time range (0 → group_duration).
        After _offset_result_timestamps, this becomes group_start → group_end.

        Args:
            result: WhisperResult with region-relative timestamps
            group_duration: Duration of the VAD group in seconds
        """
        if not result or not result.segments:
            return

        for seg in result.segments:
            seg.start = 0.0
            seg.end = group_duration

    @staticmethod
    def _apply_timestamp_interpolation(
        result: stable_whisper.WhisperResult,
        group_duration: float = 0.0,
    ) -> int:
        """
        Interpolate timestamps for segments where the aligner returned NULL.

        This is the new default for CONTEXT_AWARE mode (v1.8.7+). Instead of
        snapping to coarse VAD boundaries, we mathematically fill gaps based
        on character length between valid anchor timestamps.

        Algorithm:
            1. Identify "anchor" segments with valid timestamps (end > 0)
            2. Identify "gap" segments between anchors with NULL timestamps (end <= 0)
            3. For each gap:
               - GapDuration = Start(NextAnchor) - End(PrevAnchor)
               - TotalChars = sum of character lengths in gap segments
               - Assign duration proportional to character length

        Edge cases:
            - Leading NULLs (no previous anchor): Use 0.0 as start
            - Trailing NULLs (no next anchor): Use last anchor's end as start,
              estimate duration from character count, capped to group_duration
            - All NULLs (total aligner failure): Distribute across full result

        Args:
            result: WhisperResult with segments to process
            group_duration: Duration of the VAD group in seconds (0 = no cap).
                Used to prevent trailing gap interpolation from exceeding the
                group's time boundary.

        Returns:
            Number of segments that received interpolated timestamps
        """
        if not result or not result.segments:
            return 0

        segments = result.segments
        n = len(segments)
        interpolated_count = 0

        # Find all anchor indices (segments with valid timestamps)
        anchors = []
        for i, seg in enumerate(segments):
            if seg.end > 0:
                anchors.append(i)

        # If no anchors at all, we can't interpolate - mark all as needing attention
        if not anchors:
            logger.debug("No anchor timestamps found, cannot interpolate")
            return 0

        # Process gaps between anchors
        # Also handle leading gap (before first anchor) and trailing gap (after last anchor)

        def interpolate_gap(gap_indices: List[int], start_time: float, end_time: float):
            """Distribute time proportionally by character count."""
            nonlocal interpolated_count

            if not gap_indices:
                return

            # Calculate total characters in gap
            total_chars = sum(len(segments[i].text.strip()) for i in gap_indices)
            if total_chars == 0:
                total_chars = len(gap_indices)  # Fallback: equal distribution

            gap_duration = end_time - start_time
            if gap_duration <= 0:
                gap_duration = 0.5 * len(gap_indices)  # Fallback: 0.5s per segment

            current_time = start_time
            for idx in gap_indices:
                seg = segments[idx]
                seg_chars = len(seg.text.strip()) or 1
                seg_duration = gap_duration * (seg_chars / total_chars)

                seg.start = current_time
                seg.end = current_time + seg_duration
                current_time = seg.end
                interpolated_count += 1

        # Handle leading gap (segments before first anchor)
        if anchors[0] > 0:
            leading_indices = list(range(0, anchors[0]))
            leading_end = segments[anchors[0]].start
            interpolate_gap(leading_indices, 0.0, leading_end)

        # Handle gaps between anchors
        for i in range(len(anchors) - 1):
            prev_anchor_idx = anchors[i]
            next_anchor_idx = anchors[i + 1]

            # Find gap indices between these anchors
            gap_indices = []
            for j in range(prev_anchor_idx + 1, next_anchor_idx):
                if segments[j].end <= 0:
                    gap_indices.append(j)

            if gap_indices:
                gap_start = segments[prev_anchor_idx].end
                gap_end = segments[next_anchor_idx].start
                interpolate_gap(gap_indices, gap_start, gap_end)

        # Handle trailing gap (segments after last anchor)
        if anchors[-1] < n - 1:
            trailing_indices = []
            for j in range(anchors[-1] + 1, n):
                if segments[j].end <= 0:
                    trailing_indices.append(j)

            if trailing_indices:
                trailing_start = segments[anchors[-1]].end
                # Estimate trailing duration: ~50ms per character (conservative)
                total_trailing_chars = sum(len(segments[i].text.strip()) for i in trailing_indices)
                estimated_duration = max(0.5, total_trailing_chars * 0.05)

                # Cap to group_duration to prevent overflow beyond the group boundary
                if group_duration > 0:
                    max_trailing = max(0.0, group_duration - trailing_start)
                    if estimated_duration > max_trailing:
                        logger.debug(
                            "Interpolation: trailing gap capped from %.2fs to %.2fs "
                            "(group_duration=%.2fs, trailing_start=%.2fs)",
                            estimated_duration, max_trailing,
                            group_duration, trailing_start,
                        )
                        estimated_duration = max(0.1, max_trailing)

                interpolate_gap(trailing_indices, trailing_start, trailing_start + estimated_duration)

        return interpolated_count

