#!/usr/bin/env python3
"""V3 Architecture. Balanced pipeline implementation - scene detection with FasterWhisperPro ASR."""

import shutil
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor

from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger

from whisperjav.modules.segment_classification import SegmentClassifier
from whisperjav.modules.audio_preprocessing import AudioPreprocessor
from whisperjav.modules.srt_postproduction import SRTPostProduction

from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.progress_aggregator import AsyncProgressReporter
from whisperjav.utils.parameter_tracer import NullTracer

from whisperjav.modules.speech_enhancement import (
    create_enhancer_from_config,
    enhance_scenes,
    SCENE_EXTRACTION_SR,
)

# =============================================================================
# IMMORTAL OBJECT PATTERN - Prevents ctranslate2 Destructor Crash
# =============================================================================
# The ctranslate2 C++ destructor crashes with Access Violation (0xC0000005) or
# Stack Buffer Overrun (0xC0000409) on Windows when garbage collected during
# Python shutdown. This is a known upstream issue in ctranslate2/faster-whisper.
#
# Solution: Store ASR reference here to prevent garbage collection during normal
# execution. The nuclear exit (os._exit(0)) in main() terminates the process
# without running Python's shutdown sequence — the OS reclaims all memory.
#
# References:
# - https://github.com/SYSTRAN/faster-whisper/issues/1293
# - https://github.com/SYSTRAN/faster-whisper/issues/71
# - https://github.com/OpenNMT/CTranslate2/issues/1782
# =============================================================================
_IMMORTAL_ASR_REFERENCE = None


def safe_cleanup_immortal_asr() -> bool:
    """
    Clear the immortal ASR reference. Does NOT trigger destructors.

    IMPORTANT: This function intentionally does NOT call cleanup(), del, or gc.collect()
    on the ASR model. The ctranslate2 C++ destructor crashes with 0xC0000409
    (STATUS_STACK_BUFFER_OVERRUN) on Windows — a native structured exception that
    Python's try/except CANNOT catch. Even "controlled" cleanup triggers this crash.

    The correct approach is to skip all destructor-triggering operations and rely on
    os._exit(0) (nuclear exit) to terminate the process. The OS kernel reclaims all
    GPU memory when the process dies. No resource leak, no crash.

    This function exists for backwards compatibility. Callers should use os._exit(0)
    instead of attempting ASR cleanup.

    See: https://github.com/meizhong986/WhisperJAV/issues/125

    Returns:
        True if reference was held, False if already None.
    """
    global _IMMORTAL_ASR_REFERENCE

    if _IMMORTAL_ASR_REFERENCE is None:
        return False

    # Do NOT call cleanup(), del, or gc.collect() — these trigger the native crash.
    # Just clear the Python reference. The nuclear exit (os._exit) handles the rest.
    logger.debug(
        "Immortal ASR reference cleared (destructor skipped — nuclear exit will handle)"
    )
    _IMMORTAL_ASR_REFERENCE = None
    return True


class BalancedPipeline(BasePipeline):
    """Balanced pipeline using scene detection with FasterWhisperPro ASR (faster-whisper via stable-ts, VAD-enhanced)."""

    def __init__(self,
                 output_dir: str,
                 temp_dir: str,
                 keep_temp_files: bool,
                 subs_language: str,
                 resolved_config: Dict,
                 progress_display=None,
                 **kwargs):
        """
        Initializes the BalancedPipeline using V3 structured configuration.

        Args:
            output_dir: Output directory for subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            resolved_config: V3 structured configuration from TranscriptionTunerV3
            progress_display: Progress display object
            **kwargs: Additional parameters for base class
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Extract progress reporter and parameter tracer from kwargs
        self.progress_reporter = kwargs.get('progress_reporter', None)
        self.tracer = kwargs.get('parameter_tracer', NullTracer())

        # --- V3 STRUCTURED CONFIG UNPACKING ---
        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]

        # Set the ASR task based on the chosen output language
        self.asr_task = task  # Use the task from resolved config directly

        # Extract feature configurations
        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})

        # Store params for metadata logging
        self.scene_detection_params = scene_opts
        self.vad_params = params.get("vad", {})

        # Implement the smart model-switching logic (preserved from V2)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == 'direct-to-english' and model_cfg.get("model_name") == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_cfg["model_name"] = 'large-v2'

        # Store full pipeline options for diagnostic metadata (after model switching)
        self.pipeline_options = {
            "model": effective_model_cfg,
            "decoder": params.get("decoder", {}),
            "provider": params.get("provider", {}),
            "vad": self.vad_params,
            "task": task
        }
        # --- END V3 CONFIG UNPACKING ---

        # =================================================================
        # SCOPE-BASED RESOURCE MANAGEMENT (v1.7.3+)
        # GPU models are NOT created in __init__. We store CONFIGS only.
        # Models are created as LOCAL VARIABLES inside process() and
        # explicitly destroyed after use to prevent VRAM overlap.
        # =================================================================

        # Speech enhancement CONFIG (model created in process())
        self._enhancer_config = resolved_config  # Store full config for enhancer creation

        # v1.7.4+ Clean Contract: ALWAYS extract at 48kHz for scene files
        # Enhancement ALWAYS runs (even "none" backend does 48kHz→16kHz resampling)
        # Instantiate modules with V3 structured config
        self.audio_extractor = AudioExtractor(sample_rate=SCENE_EXTRACTION_SR)
        self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_opts)

        # ASR CONFIG (model created lazily on first process() call)
        self._asr_config = {
            'model_config': effective_model_cfg,
            'params': params,
            'task': task,
            'tracer': self.tracer
        }
        # ASR instance - created once, reused for all files in batch
        # Named with underscore to prevent base_pipeline.cleanup() from touching it
        # (base cleanup looks for self.asr, not self._asr)
        self._asr = None

        self.stitcher = SRTStitcher()

        # Language code for post-processor and output filenames
        # Use 'en' for direct-to-english translation, otherwise use the selected source language
        if self.subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            # Get language from decoder params (set by CLI --language)
            self.lang_code = params["decoder"].get("language", "ja")
        self.standard_postprocessor = StandardPostProcessor(language=self.lang_code, **post_proc_opts)

        # Optional modules (if enhancement features are enabled)
        if kwargs.get('smart_postprocessing', False):
            self.smart_postprocessor = SRTPostProduction()
        if kwargs.get('adaptive_classification', False):
            self.classifier = SegmentClassifier()
        if kwargs.get('adaptive_audio_enhancement', False):
            self.preprocessor = AudioPreprocessor()

    def _ensure_asr(self):
        """
        Lazy ASR initialization - create once, reuse for all files.

        This implements the Model Reuse pattern to prevent ctranslate2 destructor
        crashes during multi-file batch processing. The ASR model is created on
        first call and stored in self._asr. Subsequent calls return the same instance.

        The _IMMORTAL_ASR_REFERENCE global is set once to prevent garbage collection.
        Nuclear Exit (os._exit(0)) in pass_worker.py handles final cleanup.

        Returns:
            FasterWhisperProASR: The shared ASR instance
        """
        global _IMMORTAL_ASR_REFERENCE

        if self._asr is None:
            logger.info("Initializing ASR model (exclusive VRAM block)")
            self._asr = FasterWhisperProASR(**self._asr_config)

            # Store in immortal reference ONCE - this reference never changes
            _IMMORTAL_ASR_REFERENCE = self._asr
            logger.debug("ASR stored in _IMMORTAL_ASR_REFERENCE (one-time, destructor prevention)")
        else:
            logger.debug("Reusing existing ASR model instance")

        return self._asr

    def process(self, media_info: Dict) -> Dict:
        """Process media file through balanced pipeline with scene detection and VAD-enhanced ASR."""
        start_time = time.time()

        input_file = media_info['path']
        media_basename = media_info['basename']

        # Report file start if async reporter available
        if self.progress_reporter:
            self.progress_reporter.report_file_start(
                filename=media_basename,
                file_number=media_info.get('file_number', 1),
                total_files=media_info.get('total_files', 1)
            )

        # Trace file start
        self.tracer.emit_file_start(
            filename=media_basename,
            file_number=media_info.get('file_number', 1),
            total_files=media_info.get('total_files', 1),
            media_info=media_info
        )

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        # NOTE: reset_statistics moved to after ASR initialization (deferred loading)

        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        master_metadata["config"]["pipeline_options"] = self.pipeline_options

        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 6)
            self.progress.set_current_step("Transforming audio", 1, 6)

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration)

            # Trace audio extraction
            self.tracer.emit_audio_extraction(str(audio_path), duration)

            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting audio scenes", 2, 6)
            self.progress.set_current_step("Detecting audio scenes", 2, 6)

            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            detection_result = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            scene_paths = detection_result.to_legacy_tuples()

            # Extract structured metadata from detection result
            detection_meta = detection_result.to_metadata_dict()
            master_metadata["scenes_detected"] = detection_meta["scenes_detected"]
            # Include VAD segments if available (Silero method)
            if detection_meta.get("vad_segments"):
                master_metadata["vad_segments"] = detection_meta["vad_segments"]
                master_metadata["vad_method"] = detection_meta.get("vad_method")
                master_metadata["vad_params"] = detection_meta.get("vad_params")
            # Include coarse boundaries (Pass 1 scene boundaries before splitting)
            if detection_meta.get("coarse_boundaries"):
                master_metadata["coarse_boundaries"] = detection_meta["coarse_boundaries"]
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed",
                scene_count=len(scene_paths), scenes_dir=str(scenes_dir))

            # Trace scene detection
            self.tracer.emit_scene_detection(
                method=self.scene_detector.name,
                params=self.scene_detection_params,
                scenes_found=len(scene_paths),
                scene_stats={
                    "total_duration": sum(d for _, _, _, d in scene_paths),
                    "shortest": min((d for _, _, _, d in scene_paths), default=0),
                    "longest": max((d for _, _, _, d in scene_paths), default=0),
                }
            )

            # =================================================================
            # PHASE 1: SPEECH ENHANCEMENT (Exclusive VRAM Block)
            # Enhancer is a LOCAL variable - created, used, and DESTROYED
            # before ASR is loaded. This prevents the "VRAM Sandwich".
            # =================================================================
            import gc
            try:
                import torch
                _torch_available = torch.cuda.is_available()
            except ImportError:
                _torch_available = False

            # v1.7.4+ Clean Contract: Enhancement ALWAYS runs
            # Even "none" backend does 48kHz→16kHz resampling for VAD/ASR
            self.progress.set_current_step("Preparing audio for ASR", 3, 6)

            # A. Load Enhancer (always succeeds - "none" backend is fallback)
            enhancer = create_enhancer_from_config(self._enhancer_config)
            enhancer_name = enhancer.name
            enhancer_display = enhancer.display_name
            logger.info(f"Processing {len(scene_paths)} scenes with {enhancer_display}")

            def enhancement_progress(scene_num, total, name):
                if scene_num == 1 or scene_num % 5 == 0 or scene_num == total:
                    pct = (scene_num / total) * 100
                    print(f"\rProcessing: [{scene_num}/{total}] {pct:.0f}%", end='', flush=True)

            # B. Process scenes (enhancement or passthrough resampling)
            scene_paths = enhance_scenes(
                scene_paths,
                enhancer,
                self.temp_dir,
                progress_callback=enhancement_progress,
            )
            print()  # Newline after progress

            master_metadata["config"]["speech_enhancement"] = {
                "enabled": True,
                "backend": enhancer_name,
            }

            # C. DESTROY Enhancer - This is the "JIT Unload"
            # We must confirm VRAM is near-zero before loading ASR
            logger.debug("Destroying enhancer to free VRAM before ASR load")
            enhancer.cleanup()
            del enhancer
            gc.collect()
            if _torch_available:
                try:
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cleared after enhancement - VRAM should be near-zero")
                except Exception as e:
                    # CUDA context may be corrupted from prior OOM during enhancement
                    # Log and continue - ASR phase will either work (fresh allocation) or fail explicitly
                    logger.warning(f"CUDA cache clear failed after enhancement: {e}")

            # =================================================================
            # PHASE 2: ASR TRANSCRIPTION (Model Reuse Pattern)
            # ASR is created ONCE and reused for all files in batch.
            # This prevents ctranslate2 destructor crashes that occurred when
            # multiple ASR instances were created and the old ones were GC'd.
            # The _ensure_asr() method handles lazy initialization and stores
            # the reference in _IMMORTAL_ASR_REFERENCE (one-time).
            # Nuclear Exit (os._exit(0)) in pass_worker.py handles final cleanup.
            # =================================================================

            # Get or create the shared ASR instance
            asr = self._ensure_asr()

            # Reset per-file statistics (safe - just Python dict assignment)
            if hasattr(asr, "reset_statistics"):
                asr.reset_statistics()

            # Trace ASR config before transcription
            self.tracer.emit_asr_config(
                model=asr.model_name,
                backend="faster-whisper",
                params=self.pipeline_options.get("decoder", {})
            )

            # Step 4: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing scenes with VAD", 4, 6)
            self.progress.set_current_step("Transcribing scenes with VAD", 4, 6)

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            # Start scene transcription with unified progress management
            self.progress.start_subtask("Transcribing scenes", len(scene_paths))

            # Check if we have access to unified progress manager for external library suppression
            unified_manager = getattr(self.progress, 'unified_manager', None)

            # Scene-level progress tracking variables
            last_update_time = time.time()
            transcription_start_time = time.time()
            update_interval = 30  # seconds
            batch_update_size = 5  # scenes
            total_scenes = len(scene_paths)

            # Print initial scene transcription header (always visible)
            print(f"\nTranscribing {total_scenes} scenes with VAD-enhanced processing:")

            # Accumulate VAD segments across all scenes for visualization data contract
            all_vad_segments = []

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                scene_num = idx + 1

                # Show scene-level progress for user feedback (bypass adapter filtering)
                should_show_update = (
                    scene_num == 1 or  # Always show first scene
                    scene_num % batch_update_size == 0 or  # Every 5 scenes
                    time.time() - last_update_time > update_interval or  # Every 30 seconds
                    scene_num == len(scene_paths)  # Always show last scene
                )

                if should_show_update:
                    # Create tqdm-style progress bar (ASCII-compatible for Windows)
                    progress_pct = (scene_num / total_scenes) * 100
                    bar_width = 30
                    filled_width = int(bar_width * scene_num / total_scenes)
                    progress_bar = '=' * filled_width + '-' * (bar_width - filled_width)

                    # Calculate ETA (only after processing a few scenes)
                    eta_text = ""
                    if scene_num > 3:
                        elapsed = time.time() - transcription_start_time
                        avg_time_per_scene = elapsed / scene_num
                        remaining_scenes = total_scenes - scene_num
                        eta_seconds = remaining_scenes * avg_time_per_scene

                        if eta_seconds > 60:
                            eta_text = f" | ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {eta_seconds:.0f}s"

                    # Format scene filename (truncate if needed)
                    scene_filename = scene_path.name
                    if len(scene_filename) > 25:
                        scene_filename = scene_filename[:22] + "..."

                    # Direct console output (bypasses adapter filtering)
                    progress_line = f"\rTranscribing: [{progress_bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_filename}{eta_text}"
                    print(progress_line, end='', flush=True)

                    last_update_time = time.time()

                try:
                    # Use unified progress manager's external suppression if available
                    if unified_manager:
                        with unified_manager.suppress_external_progress():
                            asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
                    else:
                        asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)

                    # Process results - simplified to reduce message spam
                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["no_speech_detected"] = True

                    # Collect VAD segments from ASR (adjusted by scene start offset)
                    if hasattr(asr, 'get_last_vad_segments'):
                        scene_vad = asr.get_last_vad_segments()
                        for seg in scene_vad:
                            all_vad_segments.append({
                                "start_sec": round(start_time_sec + seg["start_sec"], 3),
                                "end_sec": round(start_time_sec + seg["end_sec"], 3),
                            })

                    self.progress.update_subtask(1)

                except Exception as e:
                    # Show errors with scene context
                    self.progress.show_message(f"Scene {scene_num}/{len(scene_paths)} failed: {str(e)}", "error", 2.0)
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    self.progress.update_subtask(1)

            self.progress.finish_subtask()

            # Save accumulated ASR-level VAD segments to metadata for visualization
            if all_vad_segments:
                master_metadata["vad_segments"] = all_vad_segments
                master_metadata["vad_method"] = "silero"
                master_metadata["vad_params"] = self.vad_params

            # Print completion message for scene transcription (always visible)
            print(f"\n[DONE] Completed transcription of {total_scenes} scenes")

            # Step 5: Stitch scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Combining scene transcriptions", 5, 6)
            self.progress.set_current_step("Combining scene transcriptions", 5, 6)

            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed",
                subtitle_count=num_subtitles, output_path=str(stitched_srt_path))

            # Step 6: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 6, 6)
            self.progress.set_current_step("Post-processing subtitles", 6, 6)

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.standard_postprocessor.process(stitched_srt_path, final_srt_path)

            # Ensure the final SRT is in the output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)
                logger.debug(f"Copied final SRT from {processed_srt_path} to {final_srt_path}")

            # Move raw_subs folder to output directory
            temp_raw_subs_path = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                # Create raw_subs directory if it doesn't exist
                final_raw_subs_path.mkdir(exist_ok=True)

                # Copy only files related to current media_basename to avoid ghost files
                for file in temp_raw_subs_path.glob(f"{media_basename}*"):
                    dest_file = final_raw_subs_path / file.name
                    shutil.copy2(file, dest_file)
                    logger.debug(f"Copied {file.name} to raw_subs")

                logger.debug(f"Copied relevant raw_subs files to: {final_raw_subs_path}")

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed", statistics=stats, output_path=str(final_srt_path))

            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0)
            }

            logprob_filtered = 0
            nonverbal_filtered = 0
            if hasattr(asr, "get_filter_statistics"):
                filter_stats = asr.get_filter_statistics() or {}
                logprob_filtered = filter_stats.get('logprob_filtered', 0)
                nonverbal_filtered = filter_stats.get('nonverbal_filtered', 0)

            # C. ASR CLEANUP INTENTIONALLY SKIPPED (Model Reuse Pattern)
            # ================================================================
            # DO NOT call asr.cleanup(), del asr, or gc.collect() here!
            #
            # The ASR model (self._asr) is REUSED across all files in the batch.
            # This prevents ctranslate2 C++ destructor crashes that occurred when
            # multiple ASR instances were created and old ones were garbage collected.
            #
            # The model is stored in both self._asr and _IMMORTAL_ASR_REFERENCE.
            # Nuclear Exit (os._exit(0)) in pass_worker.py terminates the process
            # without calling Python destructors - the OS reclaims all memory.
            #
            # VRAM stays allocated (~3GB) until process exit, but this is safe
            # because the same model is reused for all files.
            # ================================================================
            logger.debug("ASR cleanup skipped (Model Reuse Pattern - Nuclear Exit will handle)")

            master_metadata["summary"]["final_subtitles_raw"] += logprob_filtered + nonverbal_filtered
            master_metadata["summary"]["quality_metrics"].update({
                "logprob_filtered": logprob_filtered,
                "nonverbal_filtered": nonverbal_filtered,
                "cps_filtered": stats.get('cps_filtered', 0)
            })

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"

            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            self.cleanup_temp_files(media_basename)

            # Trace postprocessing
            self.tracer.emit_postprocessing(stats)

            # Trace completion
            self.tracer.emit_completion(
                success=True,
                final_subtitles=master_metadata["summary"]["final_subtitles_refined"],
                total_duration=total_time,
                output_path=str(final_srt_path)
            )

            # Report completion
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=True,
                    stats={
                        'subtitles': master_metadata["summary"]["final_subtitles_refined"],
                        'duration': total_time,
                        'scenes': len(scene_paths)
                    }
                )

            return master_metadata

        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e))
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)

            # Trace failure
            self.tracer.emit_completion(
                success=False,
                final_subtitles=0,
                total_duration=time.time() - start_time,
                output_path="",
                error=str(e)
            )

            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )

            raise

    def get_mode_name(self) -> str:
        return "balanced"

