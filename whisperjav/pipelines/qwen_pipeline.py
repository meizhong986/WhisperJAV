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
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import stable_whisper

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.srt_postprocessing import SRTPostProcessor, normalize_language_code
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger

from whisperjav.modules.speech_enhancement import (
    create_enhancer_direct,
    enhance_scenes,
    SCENE_EXTRACTION_SR,
)

# Lazy imports to avoid loading heavy modules until needed:
#   - DynamicSceneDetector (from whisperjav.modules.scene_detection)
#   - QwenASR (from whisperjav.modules.qwen_asr)
#   - SpeechSegmenterFactory (from whisperjav.modules.speech_segmentation)
#   - torch (for CUDA cleanup)


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

        # Scene detection (Phase 2)
        scene_detector: str = "none",

        # Speech enhancement (Phase 3)
        speech_enhancer: str = "none",
        speech_enhancer_model: Optional[str] = None,

        # Speech segmentation / VAD (Phase 4)
        speech_segmenter: str = "none",

        # Qwen ASR (Phase 5)
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        language: Optional[str] = None,
        timestamps: str = "word",
        aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        context: str = "",
        context_file: Optional[str] = None,
        attn_implementation: str = "auto",

        # Japanese post-processing (inside Phase 5)
        japanese_postprocess: bool = True,
        postprocess_preset: str = "high_moan",

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

        # Scene detection config
        self.scene_method = scene_detector

        # Speech enhancement config
        self.enhancer_backend = speech_enhancer
        self.enhancer_model = speech_enhancer_model

        # Speech segmentation config
        self.segmenter_backend = speech_segmenter

        # Cross-scene context propagation (disabled for MVP; structure retained
        # for future enablement once quality gates are in place)
        self.cross_scene_context = False

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
            "japanese_postprocess": japanese_postprocess,
            "postprocess_preset": postprocess_preset,
        }
        self.model_id = model_id

        # Output config
        self.subs_language = subs_language
        self.lang_code = normalize_language_code(language or "ja")

        # Shared modules (lightweight, safe to create in __init__)
        self.audio_extractor = AudioExtractor(sample_rate=SCENE_EXTRACTION_SR)
        self.stitcher = SRTStitcher()
        self.postprocessor = SRTPostProcessor(language=self.lang_code)

        logger.debug(
            "[QwenPipeline PID %s] Initialized (model=%s, scene=%s, enhancer=%s, segmenter=%s)",
            os.getpid(), model_id, scene_detector, speech_enhancer, speech_segmenter,
        )

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
        logger.info("[QwenPipeline PID %s] Phase 2: Scene detection (method=%s)", os.getpid(), self.scene_method)
        phase2_start = time.time()

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)

        if self.scene_method == "none":
            # No scene detection — treat full audio as a single scene
            scene_paths = [(extracted_audio, 0.0, duration, duration)]
            logger.info("[QwenPipeline PID %s] Phase 2: Scene detection disabled, using full audio as single scene", os.getpid())
        else:
            from whisperjav.modules.scene_detection import DynamicSceneDetector
            scene_detector = DynamicSceneDetector(method=self.scene_method)
            scene_paths = scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            logger.info(
                "[QwenPipeline PID %s] Phase 2: Detected %d scenes (method=%s)",
                os.getpid(), len(scene_paths), self.scene_method,
            )

        master_metadata["stages"]["scene_detection"] = {
            "method": self.scene_method,
            "scenes_detected": len(scene_paths),
            "time_sec": time.time() - phase2_start,
        }

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
            segmenter = SpeechSegmenterFactory.create(self.segmenter_backend)

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
                    logger.warning(f"Phase 4: Scene {idx + 1} segmentation failed: {e}, skipping VAD filter")

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
        logger.info("[QwenPipeline PID %s] Phase 5: ASR transcription (model=%s)", os.getpid(), self.model_id)
        phase5_start = time.time()

        from whisperjav.modules.qwen_asr import QwenASR
        asr = QwenASR(**self._asr_config)

        # Context biasing: user context applies to first scene only (MVP).
        # Cross-scene propagation is gated by self.cross_scene_context flag.
        user_context = self._asr_config.get("context", "")
        previous_tail = ""

        scene_results: List[Tuple[Optional[stable_whisper.WhisperResult], int]] = []

        for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
            scene_num = idx + 1
            logger.info(
                "[QwenPipeline PID %s] Phase 5: Transcribing scene %d/%d (%.1fs)",
                os.getpid(), scene_num, len(scene_paths), dur_sec,
            )

            try:
                # Assemble per-scene context
                if self.cross_scene_context:
                    # Cross-scene propagation: user glossary + previous scene tail
                    if previous_tail and user_context:
                        scene_context = f"{user_context}\n{previous_tail}"
                    elif previous_tail:
                        scene_context = previous_tail
                    else:
                        scene_context = user_context
                else:
                    # MVP: user context for first scene only, no context for rest
                    scene_context = user_context if idx == 0 else ""

                # Transcribe with per-scene context
                result = asr.transcribe(
                    scene_path,
                    context=scene_context if scene_context else None,
                )

                # Post-ASR VAD filter: remove segments outside speech regions
                if idx in speech_regions_per_scene and result.segments:
                    result = self._apply_vad_filter(result, speech_regions_per_scene[idx])

                segment_count = len(result.segments) if result.segments else 0
                scene_results.append((result, idx))

                # Extract tail for cross-scene context propagation (when enabled)
                if self.cross_scene_context and result.segments:
                    tail_segments = result.segments[-2:] if len(result.segments) >= 2 else result.segments
                    previous_tail = " ".join(seg.text.strip() for seg in tail_segments if seg.text.strip())
                # If 0 segments (silence/music), previous_tail carries forward

                if segment_count == 0:
                    logger.info(f"Phase 5: Scene {scene_num} produced 0 segments (may be non-speech audio)")
                else:
                    logger.debug(f"Phase 5: Scene {scene_num} produced {segment_count} segments")

            except Exception as e:
                logger.warning(f"Phase 5: Scene {scene_num} failed: {e}, skipping")
                scene_results.append((None, idx))

        # VRAM Block 2 cleanup
        asr.cleanup()
        del asr
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        master_metadata["stages"]["asr"] = {
            "model_id": self.model_id,
            "scenes_transcribed": sum(1 for r, _ in scene_results if r is not None and r.segments),
            "scenes_empty": sum(1 for r, _ in scene_results if r is not None and not r.segments),
            "scenes_failed": sum(1 for r, _ in scene_results if r is None),
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
        logger.info("[QwenPipeline PID %s] Phase 8: Sanitising output", os.getpid())
        phase8_start = time.time()

        final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"

        if num_subtitles > 0:
            processed_path, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)

            # Copy sanitized output to the final named path (sanitizer writes
            # next to input as *.sanitized.srt; we need the canonical name)
            if processed_path != final_srt_path:
                shutil.copy2(processed_path, final_srt_path)
                logger.debug("Copied final SRT from %s to %s", processed_path, final_srt_path)

            logger.info(
                "[QwenPipeline PID %s] Phase 8: %d subtitles, %d hallucinations removed",
                os.getpid(),
                stats.get("total_subtitles", 0),
                stats.get("removed_hallucinations", 0),
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

        # Metadata contract: keys expected by main.py
        master_metadata["output_files"]["final_srt"] = str(final_srt_path)
        master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
        master_metadata["summary"]["final_subtitles_refined"] = (
            stats.get("total_subtitles", 0) - stats.get("empty_removed", 0)
        )

        # ==============================================================
        # COMPLETE
        # ==============================================================
        total_time = time.time() - pipeline_start
        master_metadata["total_time_sec"] = total_time

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
    # Post-ASR VAD filter
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_vad_filter(
        result: stable_whisper.WhisperResult,
        seg_result,
        min_overlap_ratio: float = 0.3,
    ) -> stable_whisper.WhisperResult:
        """
        Filter ASR segments by overlap with VAD speech regions.

        Keeps only ASR segments that overlap sufficiently with detected speech.
        Same pattern as TransformersPipeline.

        Args:
            result: WhisperResult from QwenASR
            seg_result: SegmentationResult from SpeechSegmenter
            min_overlap_ratio: Minimum overlap ratio to keep a segment

        Returns:
            Filtered WhisperResult (modified in-place)
        """
        if not seg_result.segments or not result.segments:
            return result

        speech_regions = [(s.start_sec, s.end_sec) for s in seg_result.segments]

        segments_to_remove = []
        for seg in result.segments:
            seg_start = seg.start
            seg_end = seg.end
            seg_duration = seg_end - seg_start
            if seg_duration <= 0:
                continue

            # Calculate overlap with any speech region
            max_overlap = 0.0
            for sp_start, sp_end in speech_regions:
                overlap_start = max(seg_start, sp_start)
                overlap_end = min(seg_end, sp_end)
                overlap = max(0.0, overlap_end - overlap_start)
                max_overlap = max(max_overlap, overlap)

            overlap_ratio = max_overlap / seg_duration
            if overlap_ratio < min_overlap_ratio:
                segments_to_remove.append(seg)

        if segments_to_remove:
            original_count = len(result.segments)
            for seg in segments_to_remove:
                try:
                    result.remove_segment(seg)
                except (ValueError, AttributeError):
                    pass  # Segment may already be removed or method differs
            filtered_count = len(result.segments)
            logger.debug(
                "VAD filter: %d -> %d segments (removed %d non-speech)",
                original_count, filtered_count, original_count - filtered_count,
            )

        return result
