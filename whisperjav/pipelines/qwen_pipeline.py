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
from enum import Enum
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


class InputMode(Enum):
    """
    Controls how audio is chunked before being passed to the ASR.

    The Qwen3-ASR model is a Large Audio-Language Model (LALM) designed for
    long-form transcription. Tiny fragments strip context, causing hallucinations
    which then cause the ForcedAligner to fail.

    Modes:
        CONTEXT_AWARE (New Default):
            Feeds full scenes (~180s chunks) directly to ASR.
            Respects the model's need for context while staying within the
            ForcedAligner's 300s architectural limit.

        VAD_SLICING (Legacy):
            Chops audio into tiny VAD segments (often <5s).
            Destroys LALM context but useful for regression testing or
            when VAD-based filtering is specifically needed.
    """
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

        # Timestamp resolution (bridges Phase 4 VAD and Phase 5 ASR)
        timestamp_mode: str = "aligner_interpolation",  # New default for context-aware

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

        # Safe chunking: when True, enforces 150-210s scene boundaries to stay
        # within ForcedAligner's 300s architectural limit
        self.safe_chunking = qwen_safe_chunking

        # Scene detection config
        self.scene_method = scene_detector

        # Speech enhancement config
        self.enhancer_backend = speech_enhancer
        self.enhancer_model = speech_enhancer_model

        # Speech segmentation config
        self.segmenter_backend = speech_segmenter
        self.segmenter_max_group_duration = segmenter_max_group_duration

        # Timestamp resolution mode
        try:
            self.timestamp_mode = TimestampMode(timestamp_mode)
        except ValueError:
            logger.warning(
                "Unknown timestamp_mode '%s', defaulting to 'aligner_interpolation'",
                timestamp_mode,
            )
            self.timestamp_mode = TimestampMode.ALIGNER_WITH_INTERPOLATION

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
            "[QwenPipeline PID %s] Initialized (model=%s, input_mode=%s, safe_chunking=%s, "
            "scene=%s, enhancer=%s, segmenter=%s, timestamps=%s)",
            os.getpid(), model_id, self.input_mode.value, self.safe_chunking,
            scene_detector, speech_enhancer, speech_segmenter, self.timestamp_mode.value,
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
        # When safe_chunking is enabled, enforce 150-210s scene boundaries
        # to stay within the ForcedAligner's 300s architectural limit while
        # providing enough context for the LALM (3 minutes = sweet spot).
        logger.info(
            "[QwenPipeline PID %s] Phase 2: Scene detection (method=%s, safe_chunking=%s)",
            os.getpid(), self.scene_method, self.safe_chunking,
        )
        phase2_start = time.time()

        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)

        if self.scene_method == "none":
            # No scene detection — treat full audio as a single scene
            scene_paths = [(extracted_audio, 0.0, duration, duration)]
            logger.info("[QwenPipeline PID %s] Phase 2: Scene detection disabled, using full audio as single scene", os.getpid())
        else:
            from whisperjav.modules.scene_detection import DynamicSceneDetector

            # Configure scene detector parameters
            scene_detector_kwargs = {"method": self.scene_method}

            # === Safe Chunking Override (v1.8.7+) ===
            # When safe_chunking is True, enforce scene boundaries (30-90s).
            # This creates manageable blocks for semantic scene detection.
            if self.safe_chunking:
                # Parameter names must match DynamicSceneDetector contract
                scene_detector_kwargs["min_duration"] = 30  # seconds
                scene_detector_kwargs["max_duration"] = 90  # seconds
                logger.info(
                    "[QwenPipeline PID %s] Phase 2: Safe chunking enabled "
                    "(min=30s, max=90s)",
                    os.getpid(),
                )

            scene_detector = DynamicSceneDetector(**scene_detector_kwargs)
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
            segmenter = SpeechSegmenterFactory.create(
                self.segmenter_backend,
                max_group_duration_s=self.segmenter_max_group_duration,
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
        logger.info("[QwenPipeline PID %s] Phase 5: ASR transcription (model=%s)", os.getpid(), self.model_id)
        phase5_start = time.time()

        from whisperjav.modules.qwen_asr import QwenASR
        asr = QwenASR(**self._asr_config)

        # Debug artifacts directory (master text, timestamps, merged words)
        raw_subs_dir = self.temp_dir / "raw_subs"
        raw_subs_dir.mkdir(exist_ok=True)

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

                # === Input Mode Branching (v1.8.7+) ===
                # CONTEXT_AWARE: Feed full scene (~180s) to ASR for LALM context
                # VAD_SLICING:   Chop into tiny VAD fragments (legacy behavior)
                if self.input_mode == InputMode.CONTEXT_AWARE:
                    # --- NEW DEFAULT: Context-Aware Transcription ---
                    # Feed the FULL scene file (approx 180s from Phase 2) directly
                    # to the ASR. This gives Qwen3 the 3-minute context it needs
                    # as a Large Audio-Language Model.
                    logger.debug(
                        "Phase 5: Scene %d — CONTEXT_AWARE mode, transcribing full scene (%.1fs)",
                        scene_num, dur_sec,
                    )
                    result = asr.transcribe(
                        scene_path,
                        context=scene_context if scene_context else None,
                        artifacts_dir=raw_subs_dir,
                    )

                    # Apply timestamp interpolation if configured
                    if result and result.segments:
                        if self.timestamp_mode == TimestampMode.ALIGNER_WITH_INTERPOLATION:
                            interp_count = self._apply_timestamp_interpolation(result)
                            if interp_count:
                                logger.debug(
                                    "Phase 5: Scene %d — interpolated timestamps for %d segments",
                                    scene_num, interp_count,
                                )

                elif self.input_mode == InputMode.VAD_SLICING and idx in speech_regions_per_scene:
                    # --- LEGACY: VAD Slicing Transcription ---
                    # Chops audio into tiny VAD segments. Destroys LALM context
                    # but useful for regression testing or specific VAD use cases.
                    logger.debug(
                        "Phase 5: Scene %d — VAD_SLICING mode, transcribing %d speech regions",
                        scene_num, len(speech_regions_per_scene[idx].groups),
                    )
                    result = self._transcribe_speech_regions(
                        scene_path, speech_regions_per_scene[idx],
                        asr=asr,
                        context=scene_context if scene_context else None,
                        artifacts_dir=raw_subs_dir,
                    )
                else:
                    # No VAD data or VAD_SLICING without VAD: transcribe full scene
                    logger.debug(
                        "Phase 5: Scene %d — no VAD data, transcribing full scene (%.1fs)",
                        scene_num, dur_sec,
                    )
                    result = asr.transcribe(
                        scene_path,
                        context=scene_context if scene_context else None,
                        artifacts_dir=raw_subs_dir,
                    )

                segment_count = len(result.segments) if result and result.segments else 0
                scene_results.append((result, idx))

                # Extract tail for cross-scene context propagation (when enabled)
                if self.cross_scene_context and result and result.segments:
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
            "input_mode": self.input_mode.value,
            "timestamp_mode": self.timestamp_mode.value,
            "safe_chunking": self.safe_chunking,
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
    # Pre-ASR speech region transcription (WhisperProASR pattern)
    # ------------------------------------------------------------------

    def _transcribe_speech_regions(
        self,
        scene_path: Path,
        seg_result,
        asr,
        context: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> Optional[stable_whisper.WhisperResult]:
        """
        Transcribe individual speech regions from a scene.

        Follows the WhisperProASR pattern: for each VAD group, slice the scene
        audio to extract only the speech portion, transcribe that clip, and
        offset timestamps back to scene-relative time.

        This is the correct pipeline flow:
            scene audio → VAD groups → per-group transcription → offset → combine

        Args:
            scene_path: Path to scene WAV (16kHz mono, from Phase 3)
            seg_result: SegmentationResult from Phase 4
            asr: QwenASR instance
            context: Optional context string for ASR
            artifacts_dir: Optional dir for debug artifacts

        Returns:
            Combined WhisperResult with all speech regions, or None if empty.
        """
        import soundfile as sf
        import numpy as np

        # Read scene audio once
        audio_data, sr = sf.read(str(scene_path))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        speech_regions_dir = self.temp_dir / "speech_regions"
        speech_regions_dir.mkdir(exist_ok=True)

        combined_result = None
        total_groups = len(seg_result.groups)
        total_fallback_applied = 0

        for group_idx, group in enumerate(seg_result.groups):
            if not group:
                continue

            # Group time range (same pattern as WhisperProASR._transcribe_vad_group)
            group_start_sec = group[0].start_sec
            group_end_sec = group[-1].end_sec

            # Skip very short groups
            group_duration = group_end_sec - group_start_sec
            if group_duration < 0.1:
                logger.debug(
                    "Phase 5: Skipping speech region %d/%d (%.3fs < 0.1s)",
                    group_idx + 1, total_groups, group_duration,
                )
                continue

            # Slice audio for this speech group
            start_sample = int(group_start_sec * sr)
            end_sample = int(group_end_sec * sr)
            group_audio = audio_data[start_sample:end_sample]

            # Write to temp WAV (QwenASR requires file paths)
            region_stem = f"{scene_path.stem}_region_{group_idx:04d}"
            region_path = speech_regions_dir / f"{region_stem}.wav"
            sf.write(str(region_path), group_audio, sr)

            logger.debug(
                "Phase 5: Transcribing speech region %d/%d [%.2f-%.2fs] (%.1fs)",
                group_idx + 1, total_groups,
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
                    "Phase 5: Speech region %d/%d transcription failed: %s",
                    group_idx + 1, total_groups, e,
                )
                continue

            if result is None or not result.segments:
                continue

            # ── Timestamp resolution ──────────────────────────────────
            # Apply configured timestamp mode BEFORE offsetting to
            # scene-relative time.  At this point, timestamps are
            # region-relative (0-based, within the sliced audio clip).
            if self.timestamp_mode == TimestampMode.VAD_ONLY:
                # Discard aligner timestamps; use VAD group boundaries
                self._apply_vad_only_timestamps(result, group_duration)
            elif self.timestamp_mode == TimestampMode.ALIGNER_WITH_INTERPOLATION:
                # Interpolate timestamps proportionally by character length
                interp = self._apply_timestamp_interpolation(result)
                if interp:
                    total_fallback_applied += interp
                    logger.debug(
                        "Phase 5: Interpolated timestamps for %d/%d segments "
                        "in speech region %d/%d [%.2f-%.2fs]",
                        interp, len(result.segments),
                        group_idx + 1, total_groups,
                        group_start_sec, group_end_sec,
                    )
            elif self.timestamp_mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK:
                # Fall back to VAD boundaries for zero-timestamp segments
                fb = self._apply_vad_timestamp_fallback(result, group_duration)
                if fb:
                    total_fallback_applied += fb
                    logger.info(
                        "Phase 5: VAD timestamp fallback applied to %d/%d segments "
                        "in speech region %d/%d [%.2f-%.2fs]",
                        fb, len(result.segments),
                        group_idx + 1, total_groups,
                        group_start_sec, group_end_sec,
                    )
            # ALIGNER_ONLY: use aligner timestamps as-is (no fallback)

            # Offset timestamps by group start (WhisperProASR pattern:
            # start_value = seg.start + start_sec)
            self._offset_result_timestamps(result, group_start_sec)

            # Combine: first result becomes the container
            if combined_result is None:
                combined_result = result
            else:
                combined_result.segments.extend(result.segments)

        if combined_result and combined_result.segments:
            logger.debug(
                "Phase 5: Combined %d segments from %d speech groups",
                len(combined_result.segments), total_groups,
            )
            if total_fallback_applied:
                logger.info(
                    "Phase 5: VAD timestamp fallback applied to %d/%d total segments (mode=%s)",
                    total_fallback_applied, len(combined_result.segments),
                    self.timestamp_mode.value,
                )

        return combined_result

    @staticmethod
    def _offset_result_timestamps(
        result: stable_whisper.WhisperResult, offset_sec: float,
    ):
        """
        Offset all timestamps in a WhisperResult by offset_sec.

        Same principle as WhisperProASR._process_segments() adding start_sec
        to each segment's start/end values.
        """
        for seg in result.segments:
            seg.start += offset_sec
            seg.end += offset_sec
            if hasattr(seg, 'words') and seg.words:
                for word in seg.words:
                    word.start += offset_sec
                    word.end += offset_sec

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

        This method detects such segments and assigns the VAD group's duration
        as a coarse but correct timestamp.  The text is real (ASR produced it),
        and the timing is approximate (speech occurred within the VAD window).

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

        fallback_count = 0
        for seg in result.segments:
            if seg.end <= 0.0:
                seg.start = 0.0
                seg.end = group_duration
                fallback_count += 1

        return fallback_count

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
              estimate duration from character count
            - All NULLs (total aligner failure): Distribute across full result

        Args:
            result: WhisperResult with segments to process

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
                # Estimate trailing duration: 0.5s per character (conservative)
                total_trailing_chars = sum(len(segments[i].text.strip()) for i in trailing_indices)
                estimated_duration = max(0.5, total_trailing_chars * 0.05)  # ~50ms per char
                interpolate_gap(trailing_indices, trailing_start, trailing_start + estimated_duration)

        return interpolated_count

    # ------------------------------------------------------------------
    # Post-ASR VAD filter (DEPRECATED — replaced by pre-ASR segmentation
    # above; retained for reference only, no longer called)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_vad_filter(
        result: stable_whisper.WhisperResult,
        seg_result,
        min_overlap_ratio: float = 0.3,
    ) -> stable_whisper.WhisperResult:
        """
        Filter ASR segments by overlap with VAD speech regions.

        DEPRECATED: Replaced by _transcribe_speech_regions() which does
        pre-ASR segmentation (the correct pipeline flow).

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
