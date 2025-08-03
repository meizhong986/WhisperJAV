#!/usr/bin/env python3
"""V3 Architectur. Balanced pipeline implementation - scene detection with WhisperPro ASR."""

import shutil 
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.whisper_pro_asr import WhisperProASR
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

class BalancedPipeline(BasePipeline):
    """Balanced pipeline using scene detection with WhisperPro ASR (VAD-enhanced)."""
    
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
            subs_language: Language for subtitles ('japanese' or 'english-direct')
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
        self.vad_params = params.get("vad", {})
        
        # Implement the smart model-switching logic (preserved from V2)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == 'english-direct' and model_cfg.get("model_name") == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_cfg["model_name"] = 'large-v2'
        # --- END V3 CONFIG UNPACKING ---

        # Instantiate modules with V3 structured config
        self.audio_extractor = AudioExtractor()
        #self.scene_detector = SceneDetector(**scene_opts)
        #self.scene_detector = AdaptiveSceneDetector()  # Use all defaults
        self.scene_detector = DynamicSceneDetector(**scene_opts)
        
        
        # Pass structured config to WhisperProASR
        self.asr = WhisperProASR(
            model_config=effective_model_cfg,
            params=params,
            task=task
        )

        self.stitcher = SRTStitcher()
        
        # Language code for post-processor
        lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
        self.standard_postprocessor = StandardPostProcessor(language=lang_code, **post_proc_opts)

        # Optional modules (if enhancement features are enabled)
        if kwargs.get('smart_postprocessing', False):
            self.smart_postprocessor = SRTPostProduction()
        if kwargs.get('adaptive_classification', False):
            self.classifier = SegmentClassifier()
        if kwargs.get('adaptive_audio_enhancement', False):
            self.preprocessor = AudioPreprocessor()



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
        
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )
        
        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        
        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 5)
            self.progress.set_current_step("Transforming audio", 1, 5)
            
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed", 
                output_path=str(audio_path), duration_seconds=duration)

            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting audio scenes", 2, 5)
            self.progress.set_current_step("Detecting audio scenes", 2, 5)
            
            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            
            master_metadata["scenes_detected"] = []
            for idx, (scene_path, start_time_sec, end_time_sec, duration_sec) in enumerate(scene_paths):
                scene_info = {
                    "scene_index": idx, "filename": scene_path.name,
                    "start_time_seconds": round(start_time_sec, 3), 
                    "end_time_seconds": round(end_time_sec, 3),
                    "duration_seconds": round(duration_sec, 3), "path": str(scene_path)
                }
                master_metadata["scenes_detected"].append(scene_info)
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed", 
                scene_count=len(scene_paths), scenes_dir=str(scenes_dir))
            
            # Step 3: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing scenes with VAD", 3, 5)
            self.progress.set_current_step("Transcribing scenes with VAD", 3, 5)
            
            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []
            
            # Show sub-progress for scene transcription
            self.progress.start_subtask("Transcribing scenes", len(scene_paths))

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                
                # Report scene start to async system
                if self.progress_reporter:
                    self.progress_reporter.report_scene_progress(
                        scene_index=idx,
                        total_scenes=len(scene_paths),
                        status='processing',
                        details={'scene_file': scene_path.name}
                    )
                
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
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["no_speech_detected"] = True
                        
                        # Report no speech
                        if self.progress_reporter:
                            self.progress_reporter.report_scene_progress(
                                scene_index=idx,
                                total_scenes=len(scene_paths),
                                status='complete',
                                details={'subtitles_found': False, 'no_speech': True}
                            )
                        
                    self.progress.update_subtask(1)
                    
                except Exception as e:
                    # Report failure
                    if self.progress_reporter:
                        self.progress_reporter.report_scene_progress(
                            scene_index=idx,
                            total_scenes=len(scene_paths),
                            status='failed',
                            details={'error': str(e)}
                        )
                    
                    self.progress.show_message(f"Scene {idx+1} transcription failed: {str(e)}", "warning", 1.0)
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    self.progress.update_subtask(1)
            
            self.progress.finish_subtask()
            
            # Step 4: Stitch scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Combining scene transcriptions", 4, 5)
            self.progress.set_current_step("Combining scene transcriptions", 4, 5)
            
            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed", 
                subtitle_count=num_subtitles, output_path=str(stitched_srt_path))

            # Step 5: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 5, 5)
            self.progress.set_current_step("Post-processing subtitles", 5, 5)
            
            lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
            final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
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

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            self.cleanup_temp_files(media_basename)
            
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
            
            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )
            
            raise

    def get_mode_name(self) -> str:
        return "balanced"

