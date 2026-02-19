#!/usr/bin/env python3
"""
Configuration-driven pipeline using the DecoupledSubtitlePipeline orchestrator.

Model-agnostic: pipeline behavior is determined by the combination of component
backends selected (generator, framer, cleaner, aligner).  New models are deployed
by registering a TextGenerator backend — no pipeline code changes needed.

9-Phase Flow (same infrastructure as QwenPipeline):
    1. Audio Extraction (48kHz)
    2. Scene Detection (optional)
    3. Speech Enhancement (optional, VRAM Block 1)
    4. Speech Segmentation / VAD (optional)
    5. ASR via DecoupledSubtitlePipeline orchestrator
    6. Scene SRT Generation (micro-subs)
    7. SRT Stitching
    8. Sanitisation (placeholder — bypassed pending model-specific sanitizer)
    9. Analytics

See: docs/architecture/IMPL-001-subtitle-pipeline-convergence.md (Phase 2)
"""

import json
import os
import shutil
import time
from datetime import timedelta
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
from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, TimestampMode
from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.utils.logger import logger

# Lazy imports to avoid loading heavy modules until needed:
#   - SceneDetectorFactory (from whisperjav.modules.scene_detection_backends)
#   - SpeechSegmenterFactory (from whisperjav.modules.speech_segmentation)
#   - subtitle_pipeline factories (generators, framers, cleaners, aligners)
#   - DecoupledSubtitlePipeline orchestrator


class DecoupledPipeline(BasePipeline):
    """
    Configuration-driven pipeline using the DecoupledSubtitlePipeline orchestrator.

    Model-agnostic entry point: component backends are selected by name via
    factory registries.  Adding a new ASR model requires only:
      1. Write a TextGenerator class (~100-200 lines)
      2. Register it in TextGeneratorFactory (one line)
      3. User selects it via --generator <name>

    No orchestrator changes.  No hardening changes.  No framer changes.
    """

    def __init__(
        self,
        # Standard pipeline params (passed to BasePipeline)
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        keep_temp_files: bool = False,
        progress_display=None,

        # Component backend selection
        generator_backend: str = "qwen3",
        framer_backend: str = "full-scene",
        cleaner_backend: str = "qwen3",
        aligner_backend: str = "qwen3",  # "none" for aligner-free workflows

        # Component-specific configs (passed to factory.create(**config))
        generator_config: Optional[Dict[str, Any]] = None,
        framer_config: Optional[Dict[str, Any]] = None,
        cleaner_config: Optional[Dict[str, Any]] = None,
        aligner_config: Optional[Dict[str, Any]] = None,

        # Pipeline-level config
        timestamp_mode: str = "aligner_interpolation",
        context: str = "",
        context_file: Optional[str] = None,
        language: str = "Japanese",

        # Scene detection (Phase 2)
        scene_detector: str = "semantic",
        safe_chunking: bool = True,
        scene_min_duration: float = 12.0,
        scene_max_duration: float = 48.0,

        # Speech enhancement (Phase 3)
        speech_enhancer: str = "none",
        speech_enhancer_model: Optional[str] = None,

        # Speech segmentation / VAD (Phase 4)
        speech_segmenter: str = "ten",
        segmenter_max_group_duration: float = 6.0,
        segmenter_config: Optional[Dict[str, Any]] = None,

        # Step-down (Phase 3 of IMPL-001, currently disabled)
        stepdown_enabled: bool = False,
        stepdown_fallback_group_s: float = 6.0,

        # Output
        subs_language: str = "native",

        **kwargs,
    ):
        """Initialize DecoupledPipeline with component-driven configuration."""
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            **kwargs,
        )

        # Progress display
        self.progress_display = progress_display

        # Component backend selection
        self.generator_backend = generator_backend
        self.framer_backend = framer_backend
        self.cleaner_backend = cleaner_backend
        self.aligner_backend = aligner_backend

        # Component configs (default to empty dict if None)
        self.generator_config = generator_config or {}
        self.framer_config = framer_config or {}
        self.cleaner_config = cleaner_config or {}
        self.aligner_config = aligner_config or {}

        # Timestamp resolution mode
        try:
            self.timestamp_mode = TimestampMode(timestamp_mode)
        except ValueError:
            logger.warning(
                "Unknown timestamp_mode '%s', defaulting to 'aligner_interpolation'",
                timestamp_mode,
            )
            self.timestamp_mode = TimestampMode.ALIGNER_WITH_INTERPOLATION

        # Context resolution
        self.context = self._resolve_context(context, context_file)
        self.language = language

        # Scene detection config
        self.scene_method = scene_detector
        self.safe_chunking = safe_chunking
        self.scene_min_duration = scene_min_duration
        self.scene_max_duration = scene_max_duration

        # Speech enhancement config
        self.enhancer_backend = speech_enhancer
        self.enhancer_model = speech_enhancer_model

        # Speech segmentation config
        self.segmenter_backend = speech_segmenter
        self.segmenter_max_group_duration = segmenter_max_group_duration
        self.segmenter_config = segmenter_config or {}

        # Step-down config (Phase 3 of IMPL-001)
        self.stepdown_enabled = stepdown_enabled
        self.stepdown_fallback_group_s = stepdown_fallback_group_s

        # Output config
        self.subs_language = subs_language
        self.lang_code = normalize_language_code(language or "ja")

        # Shared modules (lightweight, safe to create in __init__)
        self.audio_extractor = AudioExtractor(sample_rate=SCENE_EXTRACTION_SR)
        self.stitcher = SRTStitcher()
        self.postprocessor = SRTPostProcessor(language=self.lang_code)

        # Build the orchestrator (components are lightweight — no model loading)
        self._subtitle_pipeline = self._build_subtitle_pipeline()

        logger.debug(
            "[DecoupledPipeline PID %s] Initialized (generator=%s, framer=%s, "
            "cleaner=%s, aligner=%s, scene=%s, timestamps=%s)",
            os.getpid(), generator_backend, framer_backend,
            cleaner_backend, aligner_backend, scene_detector,
            self.timestamp_mode.value,
        )

    def cleanup(self):
        """Release pipeline resources including the subtitle pipeline orchestrator."""
        if self._subtitle_pipeline is not None:
            try:
                self._subtitle_pipeline.cleanup()
            except Exception as e:
                logger.warning("Subtitle pipeline cleanup failed (non-fatal): %s", e)
            self._subtitle_pipeline = None

        super().cleanup()

    def get_mode_name(self) -> str:
        return "decoupled"

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
    # Orchestrator construction
    # ------------------------------------------------------------------

    def _build_subtitle_pipeline(self):
        """
        Construct the DecoupledSubtitlePipeline from component backend names.

        Each factory.create() produces a lightweight wrapper — no models are
        loaded until the orchestrator calls load() during process_scenes().
        """
        from whisperjav.modules.subtitle_pipeline.aligners.factory import TextAlignerFactory
        from whisperjav.modules.subtitle_pipeline.cleaners.factory import TextCleanerFactory
        from whisperjav.modules.subtitle_pipeline.framers.factory import TemporalFramerFactory
        from whisperjav.modules.subtitle_pipeline.generators.factory import TextGeneratorFactory
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline

        # TemporalFramer
        framer = TemporalFramerFactory.create(self.framer_backend, **self.framer_config)

        # TextGenerator
        generator = TextGeneratorFactory.create(self.generator_backend, **self.generator_config)

        # TextCleaner
        cleaner = TextCleanerFactory.create(self.cleaner_backend, **self.cleaner_config)

        # TextAligner (may be "none" for aligner-free workflows)
        aligner = TextAlignerFactory.create(self.aligner_backend, **self.aligner_config)

        return DecoupledSubtitlePipeline(
            framer=framer,
            generator=generator,
            cleaner=cleaner,
            aligner=aligner,
            hardening_config=HardeningConfig(timestamp_mode=self.timestamp_mode),
            language=self.language,
            context=self.context,
        )

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through the 9-phase decoupled pipeline.

        Args:
            media_info: Dict with 'path', 'basename', 'type', 'duration', etc.

        Returns:
            Master metadata dictionary with paths, stats, quality metrics.
        """
        input_file = Path(media_info["path"])
        media_basename = media_info["basename"]
        pipeline_start = time.time()

        logger.info(
            "[DecoupledPipeline PID %s] Processing: %s (generator=%s, framer=%s)",
            os.getpid(), input_file.name, self.generator_backend, self.framer_backend,
        )

        master_metadata = {
            "metadata_master": {
                "structure_version": "1.0.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "input_file": str(input_file),
            "basename": media_basename,
            "pipeline": "decoupled",
            "generator_backend": self.generator_backend,
            "framer_backend": self.framer_backend,
            "cleaner_backend": self.cleaner_backend,
            "aligner_backend": self.aligner_backend,
            "stages": {},
            "output_files": {},
            "summary": {},
        }

        # ==============================================================
        # PHASE 1: AUDIO EXTRACTION
        # ==============================================================
        logger.info("[DecoupledPipeline PID %s] Phase 1: Extracting audio from %s", os.getpid(), input_file.name)
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
        logger.info("[DecoupledPipeline PID %s] Phase 1: Complete (%.1fs audio)", os.getpid(), duration)

        # ==============================================================
        # PHASE 2: SCENE DETECTION
        # ==============================================================
        logger.info(
            "[DecoupledPipeline PID %s] Phase 2: Scene detection (method=%s, safe_chunking=%s)",
            os.getpid(), self.scene_method, self.safe_chunking,
        )
        phase2_start = time.time()

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)

        from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

        scene_detector_kwargs = {"method": self.scene_method}

        if self.safe_chunking:
            scene_detector_kwargs["min_duration"] = self.scene_min_duration
            scene_detector_kwargs["max_duration"] = self.scene_max_duration
            logger.info(
                "[DecoupledPipeline PID %s] Phase 2: Safe chunking (min=%.0fs, max=%.0fs)",
                os.getpid(), self.scene_min_duration, self.scene_max_duration,
            )

        scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_detector_kwargs)
        result = scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
        scene_paths = result.to_legacy_tuples()
        scene_detector.cleanup()

        logger.info(
            "[DecoupledPipeline PID %s] Phase 2: Detected %d scenes (method=%s)",
            os.getpid(), len(scene_paths), self.scene_method,
        )

        if scene_paths:
            durations = [sp[3] for sp in scene_paths]
            logger.info(
                "[DecoupledPipeline PID %s] Phase 2: Scene durations — "
                "total %.0fs, range %.0f–%.0fs, mean %.0fs",
                os.getpid(),
                sum(durations), min(durations), max(durations),
                sum(durations) / len(durations),
            )

        detection_meta = result.to_metadata_dict()
        master_metadata["stages"]["scene_detection"] = {
            "method": self.scene_method,
            "scenes_detected": len(scene_paths),
            "time_sec": time.time() - phase2_start,
        }
        if detection_meta.get("scenes_detected"):
            master_metadata["scenes_detected"] = detection_meta["scenes_detected"]
        if detection_meta.get("coarse_boundaries"):
            master_metadata["coarse_boundaries"] = detection_meta["coarse_boundaries"]

        # ==============================================================
        # PHASE 3: SPEECH ENHANCEMENT (optional, VRAM Block 1)
        # ==============================================================
        logger.info("[DecoupledPipeline PID %s] Phase 3: Speech enhancement (backend=%s)", os.getpid(), self.enhancer_backend)
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
        logger.info("[DecoupledPipeline PID %s] Phase 3: Complete (%.1fs)", os.getpid(), time.time() - phase3_start)

        # ==============================================================
        # PHASE 4: SPEECH SEGMENTATION / VAD (optional)
        # ==============================================================
        speech_regions_per_scene: Dict[int, Any] = {}

        if self.segmenter_backend != "none":
            logger.info("[DecoupledPipeline PID %s] Phase 4: Speech segmentation (backend=%s)", os.getpid(), self.segmenter_backend)
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
            logger.info("[DecoupledPipeline PID %s] Phase 4: Complete (%.1fs)", os.getpid(), time.time() - phase4_start)
        else:
            logger.info("[DecoupledPipeline PID %s] Phase 4: Skipped (segmenter=none)", os.getpid())

        # ==============================================================
        # PHASE 5: ASR VIA ORCHESTRATOR
        # ==============================================================
        logger.info(
            "[DecoupledPipeline PID %s] Phase 5: ASR (generator=%s, framer=%s, aligner=%s)",
            os.getpid(), self.generator_backend, self.framer_backend, self.aligner_backend,
        )
        phase5_start = time.time()

        # Debug artifacts directory
        raw_subs_dir = self.temp_dir / "raw_subs"
        raw_subs_dir.mkdir(exist_ok=True)
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
        scene_results: List[Tuple[Optional[stable_whisper.WhisperResult], int]] = []
        for idx, (asr_result, _diag) in enumerate(orch_results):
            scene_results.append((asr_result, idx))

        # Sentinel stats from orchestrator
        orch_stats = self._subtitle_pipeline.sentinel_stats
        sentinel_stats = {
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
        total_segs = sum(len(r.segments) for r, _ in orch_results if r is not None and r.segments)
        logger.info("[DecoupledPipeline] Phase 5 summary:")
        logger.info(
            "  Scenes:    %d success, %d empty, %d failed (of %d)",
            n_success, n_empty, n_failed, len(orch_results),
        )
        logger.info(
            "  Segments:  %d total (%.1f avg/scene)",
            total_segs, total_segs / max(n_success, 1),
        )
        logger.info(
            "  Sentinel:  %d collapses, %d recoveries",
            sentinel_stats.get("alignment_collapses", 0),
            sentinel_stats.get("alignment_recoveries", 0),
        )

        master_metadata["stages"]["asr"] = {
            "generator_backend": self.generator_backend,
            "framer_backend": self.framer_backend,
            "cleaner_backend": self.cleaner_backend,
            "aligner_backend": self.aligner_backend,
            "timestamp_mode": self.timestamp_mode.value,
            "safe_chunking": self.safe_chunking,
            "scenes_transcribed": n_success,
            "scenes_empty": n_empty,
            "scenes_failed": n_failed,
            "alignment_collapses": sentinel_stats.get("alignment_collapses", 0),
            "alignment_recoveries": sentinel_stats.get("alignment_recoveries", 0),
            "time_sec": time.time() - phase5_start,
        }
        logger.info("[DecoupledPipeline PID %s] Phase 5: Complete (%.1fs)", os.getpid(), time.time() - phase5_start)

        # ==============================================================
        # PHASE 6: SCENE SRT GENERATION (micro-subs)
        # ==============================================================
        logger.info("[DecoupledPipeline PID %s] Phase 6: Generating scene SRT files", os.getpid())

        scene_srts_dir = self.temp_dir / "scene_srts"
        scene_srts_dir.mkdir(exist_ok=True)
        scene_srt_info: List[Tuple[Path, float]] = []

        for asr_result, idx in scene_results:
            if asr_result is None or not asr_result.segments:
                continue

            scene_path, start_sec, end_sec, dur_sec = scene_paths[idx]
            scene_srt_path = scene_srts_dir / f"{media_basename}_scene_{idx:04d}.srt"

            try:
                asr_result.to_srt_vtt(
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
        logger.info("[DecoupledPipeline PID %s] Phase 7: Stitching %d scene SRTs", os.getpid(), len(scene_srt_info))

        stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"

        if scene_srt_info:
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            logger.info("[DecoupledPipeline PID %s] Phase 7: Stitched %d subtitles", os.getpid(), num_subtitles)
        else:
            stitched_srt_path.write_text("", encoding="utf-8")
            num_subtitles = 0
            logger.warning("[DecoupledPipeline PID %s] Phase 7: No scene SRTs to stitch (0 subtitles)", os.getpid())

        master_metadata["stages"]["stitching"] = {
            "total_subtitles": num_subtitles,
            "scenes_contributed": len(scene_srt_info),
        }

        # ==============================================================
        # PHASE 8: SANITISATION (placeholder)
        # ==============================================================
        logger.info("[DecoupledPipeline PID %s] Phase 8: Skipped (sanitizer not yet implemented for generic pipeline)", os.getpid())
        phase8_start = time.time()

        final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"

        if num_subtitles > 0:
            shutil.copy2(stitched_srt_path, final_srt_path)
            stats = {"total_subtitles": num_subtitles, "sanitizer_skipped": True}
            logger.info(
                "[DecoupledPipeline PID %s] Phase 8: %d subtitles passed through (no sanitization)",
                os.getpid(), num_subtitles,
            )
        else:
            final_srt_path.write_text("", encoding="utf-8")
            stats = {"total_subtitles": 0}

        master_metadata["stages"]["sanitisation"] = {
            "stats": stats,
            "time_sec": time.time() - phase8_start,
        }
        master_metadata["srt_path"] = str(final_srt_path)

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
            "logprob_filtered": 0,
            "nonverbal_filtered": 0,
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
            "[DecoupledPipeline PID %s] Complete: %s (%d subtitles in %s)",
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
