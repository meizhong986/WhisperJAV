#!/usr/bin/env python3
"""Fast pipeline implementation - scene detection with standard Whisper."""

import shutil
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.stable_ts_asr import StableTSASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor

from whisperjav.modules.scene_detection import SceneDetector
from whisperjav.modules.scene_detection import AdaptiveSceneDetector
from whisperjav.modules.scene_detection import DynamicSceneDetector

from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger

from whisperjav.modules.segment_classification import SegmentClassifier
from whisperjav.modules.audio_preprocessing import AudioPreprocessor
from whisperjav.modules.srt_postproduction import SRTPostProduction
from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.progress_aggregator import AsyncProgressReporter

class FastPipeline(BasePipeline):
    """Fast pipeline using standard Whisper with scene detection (V3 Architecture)."""
    
    def __init__(self, 
                 output_dir: str, 
                 temp_dir: str, 
                 keep_temp_files: bool, 
                 subs_language: str, 
                 resolved_config: Dict, 
                 progress_display=None, 
                 **kwargs):
        """
        Initializes the FastPipeline using V3 structured configuration.
        
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
        
        # ADD THIS LINE - Extract progress reporter from kwargs
        self.progress_reporter = kwargs.get('progress_reporter', None)

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
        '''
        self.scene_detection_params = {
            "detector_type": "AdaptiveSceneDetector",
            "using_defaults": True
        }        
        '''
        # Implement the smart model-switching logic (preserved from V2)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == 'direct-to-english' and model_cfg.get("model_name") == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_cfg["model_name"] = 'large-v2'
        # --- END V3 CONFIG UNPACKING ---

        # Instantiate modules with V3 structured config
        self.audio_extractor = AudioExtractor()
        #self.scene_detector = SceneDetector(**scene_opts)
        #self.scene_detector = AdaptiveSceneDetector()  # Use all defaults
        self.scene_detector = DynamicSceneDetector(**scene_opts)


        # Pass structured config to StableTSASR
        # NOTE: 'fast' pipeline uses standard whisper backend (turbo_mode=False)
        self.asr = StableTSASR(
            model_config=effective_model_cfg,
            params=params,
            task=task,
            turbo_mode=True
        )

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
            logger.debug("Smart Post-Processing enabled.")

        
    def get_mode_name(self) -> str:
        return "fast"



    def process(self, media_info: Dict) -> Dict:
        """Process media file through fast pipeline with mandatory scene detection."""
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
        
        logger.info(f"Starting FAST pipeline for: {input_file}")
        logger.info(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
        
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )
        
        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        
        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 5)
            logger.info("Step 1: Transforming audio")
            
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path),
                duration_seconds=duration
            )
            
            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting audio scenes", 2, 5)
            logger.info("Step 2: Detecting scenes")
            
            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            logger.debug(f"Detected {len(scene_paths)} scenes")
            
            master_metadata["scenes_detected"] = []
            for idx, (scene_path, start_time_sec, end_time_sec, duration_sec) in enumerate(scene_paths):
                scene_info = {
                    "scene_index": idx,
                    "filename": scene_path.name,
                    "start_time_seconds": round(start_time_sec, 3),
                    "end_time_seconds": round(end_time_sec, 3),
                    "duration_seconds": round(duration_sec, 3),
                    "path": str(scene_path)
                }
                master_metadata["scenes_detected"].append(scene_info)
            
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed",
                scene_count=len(scene_paths),
                scenes_dir=str(scenes_dir)
            )
            
            if hasattr(self, 'smart_postprocessor'):
                logger.debug("Smart Post-Processing enabled.")

            # Step 3: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing scenes with Faster-Whisper", 3, 5)
            logger.debug("Step 3: Transcribing scenes with Faster-Whisper")
            self.progress.set_current_step("Transcribing scenes", 3, 5)
            
            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            
            if len(scene_paths) > 1:
                self.progress.start_subtask("Transcribing scenes", len(scene_paths))
            
            # Add scene-level progress tracking
            total_scenes = len(scene_paths)
            print(f"\nTranscribing {total_scenes} scenes:")
            last_update_time = time.time()
            transcription_start_time = time.time()
            
            scene_srt_info = []
            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_num = idx + 1
                logger.debug(f"Transcribing scene {scene_num}/{total_scenes}: {scene_path.name}")
                
                # Show progress for first scene, every 5 scenes, every 30 seconds, or last scene
                current_time = time.time()
                should_show_progress = (
                    scene_num == 1 or 
                    scene_num == total_scenes or
                    scene_num % 5 == 0 or
                    (current_time - last_update_time) > 30
                )
                
                if should_show_progress:
                    # Calculate progress percentage
                    progress_pct = (scene_num / total_scenes) * 100
                    
                    # Create ASCII progress bar (30 characters wide)
                    filled = int(progress_pct / 3.33)  # 30 chars total
                    progress_bar = '=' * filled + '-' * (30 - filled)
                    
                    # Calculate ETA if we have processed at least one scene
                    eta_text = ""
                    if scene_num > 1:
                        elapsed = current_time - transcription_start_time
                        avg_time_per_scene = elapsed / (scene_num - 1)
                        remaining_scenes = total_scenes - scene_num
                        eta_seconds = remaining_scenes * avg_time_per_scene
                        
                        if eta_seconds > 60:
                            eta_text = f" | ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {eta_seconds:.0f}s"
                    
                    # Truncate scene filename if too long
                    scene_filename = scene_path.name
                    if len(scene_filename) > 25:
                        scene_filename = scene_filename[:22] + "..."
                    
                    # Print progress line
                    progress_line = f"\rTranscribing: [{progress_bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_filename}{eta_text}"
                    print(progress_line, end='', flush=True)
                    last_update_time = current_time
                
                # Report scene start to async system
                if self.progress_reporter:
                    self.progress_reporter.report_scene_progress(
                        scene_index=idx,
                        total_scenes=len(scene_paths),
                        status='processing',
                        details={'scene_file': scene_path.name}
                    )
                
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                try:
                    self.asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                        
                        # Report success
                        if self.progress_reporter:
                            self.progress_reporter.report_scene_progress(
                                scene_index=idx,
                                total_scenes=len(scene_paths),
                                status='complete',
                                details={'subtitles_found': True}
                            )
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                        
                        # Report no subtitles
                        if self.progress_reporter:
                            self.progress_reporter.report_scene_progress(
                                scene_index=idx,
                                total_scenes=len(scene_paths),
                                status='complete',
                                details={'subtitles_found': False}
                            )
                    
                    # Update progress after each scene
                    if len(scene_paths) > 1:
                        self.progress.update_subtask(1)
                        
                except Exception as e:
                    logger.error(f"Failed to transcribe scene {idx}: {e}")
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    
                    # Report failure
                    if self.progress_reporter:
                        self.progress_reporter.report_scene_progress(
                            scene_index=idx,
                            total_scenes=len(scene_paths),
                            status='failed',
                            details={'error': str(e)}
                        )
                    
                    # Still update progress even on error
                    if len(scene_paths) > 1:
                        self.progress.update_subtask(1)
                    continue
            
            # Show completion message for scene transcription
            if total_scenes > 0:
                print(f"\n[DONE] Completed transcription of {total_scenes} scenes")
            
            # Finish the subtask progress
            if len(scene_paths) > 1:
                self.progress.finish_subtask()
            
            master_metadata["summary"]["scenes_processed_successfully"] = len(scene_srt_info)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                model=self.asr.model_name,
                scenes_transcribed=len(scene_srt_info),
                total_scenes=len(scene_paths)
            )
            
            # Step 4: Stitch scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Combining scene transcriptions", 4, 5)
            logger.debug("Step 4: Combining scene transcriptions")
            
            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed",
                output_path=str(stitched_srt_path),
                subtitle_count=num_subtitles
            )
            
            # Step 5: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing final SRT", 5, 5)
            logger.info("Step 5: Post-processing final SRT")

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"

            # Capture the returned path
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
                master_metadata, "postprocessing", "completed", statistics=stats, output_path=str(final_srt_path)
            )
            
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
            
            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            self.cleanup_temp_files(media_basename)
            
            logger.debug(f"Completed in {total_time:.1f} seconds")
            logger.info(f"Output saved to: {final_srt_path}")
            
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
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed",
                error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )
            
            raise