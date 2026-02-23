#!/usr/bin/env python3
"""
Dedicated Qwen3-ASR Pipeline for WhisperJAV.

A single-purpose pipeline for Qwen3-ASR transcription following the
"redundancy over reuse" principle. No backend switching, no conditionals
for other ASR types.

9-Phase Flow:
    1. Audio Extraction (48kHz)
    2. Scene Detection (default: semantic, safe chunking 12-48s)
    3. Speech Enhancement (optional, VRAM Block 1)
    4. Speech Segmentation / VAD (default: TEN)
    5. ASR Transcription (VRAM Block 2, DecoupledSubtitlePipeline)
    6. Scene SRT Generation (micro-subs)
    7. SRT Stitching
    8. Sanitisation (skipped — no Qwen-specific sanitizer)
    9. Analytics

Since Phase 4 (Strangulation), only the assembly mode code path exists.
Legacy mode names (context_aware, vad_slicing) map to assembly with
appropriate framer overrides.

See: docs/architecture/ADR-004-dedicated-qwen-pipeline.md
See: docs/architecture/QWEN-PIPELINE-REFERENCE.md
"""

import json
import os
import shutil
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import stable_whisper

from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.speech_enhancement import (
    SCENE_EXTRACTION_SR,
    create_enhancer_direct,
    enhance_scenes,
)
from whisperjav.modules.srt_postprocessing import SRTPostProcessor, normalize_language_code
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.modules.subtitle_pipeline.types import TimestampMode
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger

# Lazy imports to avoid loading heavy modules until needed:
#   - SceneDetectorFactory (from whisperjav.modules.scene_detection_backends)
#   - SpeechSegmenterFactory (from whisperjav.modules.speech_segmentation)
#   - DecoupledSubtitlePipeline (from whisperjav.modules.subtitle_pipeline.orchestrator)


class InputMode(Enum):
    """
    Audio input mode for Qwen3-ASR pipeline.

    Since Phase 4 (strangulation), only ASSEMBLY is the active code path.
    CONTEXT_AWARE and VAD_SLICING are retained as enum values for backward
    compatibility — they are mapped to ASSEMBLY with appropriate framer
    overrides in QwenPipeline.__init__.

    Modes:
        ASSEMBLY: The sole active mode. Decoupled text generation + forced
            alignment via DecoupledSubtitlePipeline orchestrator.
        CONTEXT_AWARE: Deprecated. Maps to ASSEMBLY + full-scene framer.
        VAD_SLICING: Deprecated. Maps to ASSEMBLY + vad-grouped framer.
    """
    ASSEMBLY = "assembly"
    CONTEXT_AWARE = "context_aware"
    VAD_SLICING = "vad_slicing"


class QwenPipeline(BasePipeline):
    """
    Dedicated pipeline for Qwen3-ASR transcription.

    Orchestrates the 9-phase flow using existing modules.
    Follows the "redundancy over reuse" principle — no backend switching.
    """

    def __init__(
        self,
        # Standard pipeline params (passed to BasePipeline)
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        keep_temp_files: bool = False,
        progress_display=None,

        # === Input Mode Configuration ===
        # Controls audio input strategy for Qwen3-ASR (LALM)
        qwen_input_mode: str = "assembly",  # "assembly" (default), "context_aware", or "vad_slicing"
        qwen_safe_chunking: bool = True,  # Enforce 12-48s scene boundaries for ForcedAligner
        scene_min_duration: Optional[float] = None,  # Override min scene duration (default: 12s)
        scene_max_duration: Optional[float] = None,  # Override max scene duration (default: 48s)

        # Temporal framing for assembly mode (GAP-5)
        qwen_framer: str = "full-scene",  # "full-scene", "vad-grouped", "srt-source"
        framer_srt_path: Optional[str] = None,  # SRT file path (for srt-source framer)

        # Scene detection (Phase 2)
        # Default to "semantic" - it has true MERGE logic to guarantee
        # min_duration is met (unlike auditok/silero which only filter)
        scene_detector: str = "semantic",

        # Speech enhancement (Phase 3)
        speech_enhancer: str = "none",
        speech_enhancer_model: Optional[str] = None,

        # Speech segmentation / VAD (Phase 4)
        speech_segmenter: str = "ten",  # Default to TEN backend for VAD
        segmenter_max_group_duration: float = 6.0,  # Max group size in seconds (CLI: --qwen-max-group-duration)
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
        timestamp_mode: str = "aligner_interpolation",  # Pipeline default; CLI overrides to aligner_vad_fallback

        # Japanese post-processing — DISABLED for Qwen3 (C2 fix, ADR-006).
        # Qwen3 uses AssemblyTextCleaner, not the Whisper-era JapanesePostProcessor.
        # Parameter kept for backward compatibility; value is always overridden to False.
        japanese_postprocess: bool = False,
        postprocess_preset: str = "high_moan",

        # Assembly text cleaner (Step 4 of assembly mode)
        assembly_cleaner: bool = True,  # Enable/disable pre-alignment text cleaning

        # Adaptive Step-Down
        stepdown_enabled: bool = True,
        stepdown_initial_group: float = 6.0,
        stepdown_fallback_group: float = 6.0,

        # Generation safety controls
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

        # === Input Mode Configuration ===
        # Parse input mode — legacy modes are mapped to assembly configs
        try:
            self.input_mode = InputMode(qwen_input_mode)
        except ValueError:
            logger.warning(
                "Unknown qwen_input_mode '%s', defaulting to 'assembly'",
                qwen_input_mode,
            )
            self.input_mode = InputMode.ASSEMBLY

        # Legacy mode mapping: both coupled modes → assembly with appropriate framer
        if self.input_mode == InputMode.CONTEXT_AWARE:
            logger.warning(
                "input_mode='context_aware' is deprecated and now maps to "
                "assembly mode with full-scene framer. "
                "Use --qwen-framer full-scene instead."
            )
            self.input_mode = InputMode.ASSEMBLY
            qwen_framer = "full-scene"  # Override framer to match legacy behavior
        elif self.input_mode == InputMode.VAD_SLICING:
            logger.warning(
                "input_mode='vad_slicing' is deprecated and now maps to "
                "assembly mode with vad-grouped framer. "
                "Use --qwen-framer vad-grouped instead."
            )
            self.input_mode = InputMode.ASSEMBLY
            qwen_framer = "vad-grouped"  # Override framer to match legacy behavior

        # Safe chunking: when True, enforces scene boundaries to stay
        # within ForcedAligner's 180s architectural limit
        self.safe_chunking = qwen_safe_chunking
        self.scene_min_override = scene_min_duration  # None = use default (12s)
        self.scene_max_override = scene_max_duration  # None = use default (48s)

        # Temporal framing for assembly mode (GAP-5)
        self.framer_backend = qwen_framer
        self.framer_srt_path = framer_srt_path

        # Scene detection config
        self.scene_method = scene_detector

        # Speech enhancement config
        self.enhancer_backend = speech_enhancer
        self.enhancer_model = speech_enhancer_model

        # Speech segmentation config
        self.segmenter_backend = speech_segmenter
        self.segmenter_max_group_duration = segmenter_max_group_duration
        self.segmenter_config = segmenter_config or {}

        # Adaptive Step-Down config
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

        # Decoupled Subtitle Pipeline (ADR-006) — always constructed
        # Components are lightweight — no model loading until process() time
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
        from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, StepDownConfig

        cfg = self._asr_config

        # TemporalFramer: selected by --qwen-framer (default: full-scene)
        framer_kwargs = {}
        if self.framer_backend == "vad-grouped":
            framer_kwargs = {
                "segmenter_backend": self.segmenter_backend,
                "max_group_duration_s": self.segmenter_max_group_duration,
                "segmenter_config": self.segmenter_config,
            }
        elif self.framer_backend == "srt-source":
            if not self.framer_srt_path:
                raise ValueError(
                    "--qwen-framer srt-source requires --qwen-framer-srt-path"
                )
            framer_kwargs = {"srt_path": self.framer_srt_path}
        elif self.framer_backend == "manual":
            raise ValueError(
                "--qwen-framer manual is only available via the Python API "
                "(requires timestamps argument)"
            )
        framer = TemporalFramerFactory.create(self.framer_backend, **framer_kwargs)

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
        # G1 fix: vad_only mode has no use for the 0.6B aligner — skip loading
        # entirely so we save VRAM and take Branch B (aligner-free) in the
        # orchestrator's _step9_reconstruct_and_harden().
        aligner = None
        if self.timestamp_mode == TimestampMode.VAD_ONLY:
            # No aligner → Branch B; no collapse possible → step-down irrelevant
            pass
        elif cfg.get("use_aligner", True):
            aligner = TextAlignerFactory.create(
                "qwen3",
                aligner_id=cfg["aligner_id"],
                device=cfg["device"],
                dtype=cfg["dtype"],
                language=cfg["language"],
            )

        # Step-down retry config (uses existing Qwen pipeline params)
        # Disabled when aligner is None (no alignment = no collapse = nothing to step-down)
        stepdown_cfg = None
        if self.stepdown_enabled and aligner is not None:
            stepdown_cfg = StepDownConfig(
                enabled=True,
                fallback_max_group_s=self.stepdown_fallback_group,
            )

        return DecoupledSubtitlePipeline(
            framer=framer,
            generator=generator,
            cleaner=cleaner,
            aligner=aligner,
            hardening_config=HardeningConfig(timestamp_mode=self.timestamp_mode),
            language=cfg.get("language", "ja"),
            context=cfg.get("context", ""),
            stepdown_config=stepdown_cfg,
        )

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through the 9-phase Qwen pipeline.

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

        # === Safe Chunking Override ===
        # When safe_chunking is True, enforce scene boundaries to keep
        # scenes within the Qwen ASR + ForcedAligner sweet spot.
        # Tight scenes (12-48s) reduce timestamp drift and give the
        # aligner shorter, more manageable audio to align.
        if self.safe_chunking:
            min_dur = self.scene_min_override if self.scene_min_override is not None else 12
            max_dur = self.scene_max_override if self.scene_max_override is not None else 48
            scene_detector_kwargs["min_duration"] = min_dur
            scene_detector_kwargs["max_duration"] = max_dur
            logger.info(
                "[QwenPipeline PID %s] Phase 2: Safe chunking "
                "(min=%ss, max=%ss, aligner limit=180s)",
                os.getpid(), min_dur, max_dur,
            )

        scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_detector_kwargs)
        result = scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
        scene_paths = result.to_legacy_tuples()
        scene_detector.cleanup()
        logger.info(
            "[QwenPipeline PID %s] Phase 2: Detected %d scenes (method=%s)",
            os.getpid(), len(scene_paths), self.scene_method,
        )

        # Scene duration statistics
        if scene_paths:
            durations = [sp[3] for sp in scene_paths]
            logger.info(
                "[QwenPipeline PID %s] Phase 2: Scene durations — "
                "total %.0fs, range %.0f–%.0fs, mean %.0fs",
                os.getpid(),
                sum(durations),
                min(durations),
                max(durations),
                sum(durations) / len(durations),
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
        from whisperjav.utils.gpu_utils import safe_cuda_cleanup
        safe_cuda_cleanup()

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

        # ==============================================================
        # PHASE 5: ASR TRANSCRIPTION (VRAM Block 2)
        # ==============================================================
        logger.info("[QwenPipeline PID %s] Phase 5: ASR transcription (model=%s, mode=%s)",
                    os.getpid(), self.model_id, self.input_mode.value)
        phase5_start = time.time()

        # Debug artifacts directory (master text, timestamps, merged words)
        raw_subs_dir = self.temp_dir / "raw_subs"
        raw_subs_dir.mkdir(exist_ok=True)

        scene_results: List[Tuple[Optional[stable_whisper.WhisperResult], int]] = []

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

        # Phase 5 summary
        n_success = sum(1 for r, _ in orch_results if r is not None and r.segments)
        n_empty = sum(1 for r, _ in orch_results if r is not None and not r.segments)
        n_failed = sum(1 for r, _ in orch_results if r is None)
        _total_segs = sum(len(r.segments) for r, _ in orch_results if r is not None and r.segments)
        logger.info("[QwenPipeline] Phase 5 assembly summary:")
        logger.info(
            "  Scenes:    %d success, %d empty, %d failed (of %d)",
            n_success, n_empty, n_failed, len(orch_results),
        )
        logger.info(
            "  Segments:  %d total (%.1f avg/scene)",
            _total_segs, _total_segs / max(n_success, 1),
        )
        logger.info(
            "  Sentinel:  %d collapses, %d recoveries",
            orch_stats.get("collapsed_scenes", 0),
            orch_stats.get("recovered_scenes", 0),
        )
        # Sentinel stats (populated by assembly mode orchestrator)
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



