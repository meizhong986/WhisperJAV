#!/usr/bin/env python3
"""V3 architect. Faster pipeline implementation - direct transcription without chunking."""

import shutil
from pathlib import Path
from typing import Dict
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.stable_ts_asr import StableTSASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.progress_aggregator import AsyncProgressReporter
from whisperjav.utils.parameter_tracer import NullTracer

class FasterPipeline(BasePipeline):
    """Faster pipeline using Whisper turbo mode without chunking."""

    def __init__(self, 
                 output_dir: str, 
                 temp_dir: str, 
                 keep_temp_files: bool, 
                 subs_language: str,
                 resolved_config: Dict, 
                 progress_display=None, 
                 **kwargs):
        """
        Initializes the FasterPipeline using V3 structured configuration.
        
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
        
        # Set the ASR task based on the user's choice
        self.asr_task = task  # Use the task from resolved config directly
        
        # Extract feature configurations (only post-processing for faster pipeline)
        post_proc_opts = features.get("post_processing", {})
        
        # Store params for metadata logging
        self.vad_params = params.get("vad", {})
        self.scene_detection_params = {}  # Faster pipeline doesn't use scene detection

        # Implement the smart model-switching logic
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

        # Instantiate modules with V3 structured config
        self.audio_extractor = AudioExtractor()

        # Pass structured config to StableTSASR
        # NOTE: 'faster' pipeline uses turbo mode (faster-whisper backend)
        self.asr = StableTSASR(
            model_config=effective_model_cfg,
            params=params,
            task=task,
            turbo_mode=True
        )

        # Language code for post-processor and output filenames
        # Use 'en' for direct-to-english translation, otherwise use the selected source language
        if self.subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            # Get language from decoder params (set by CLI --language)
            self.lang_code = params["decoder"].get("language", "ja")
        self.postprocessor = SRTPostProcessor(language=self.lang_code, **post_proc_opts)

        # Log if smart post-processing is enabled
        if kwargs.get('smart_postprocessing', False):
            logger.debug("Smart Post-Processing enabled.")

    def get_mode_name(self) -> str:
        return "faster"
        

    def process(self, media_info: Dict) -> Dict:
        """Process media file through faster pipeline."""
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

        logger.info(f"Starting FASTER pipeline for: {input_file}")
        logger.debug(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
         
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        if hasattr(self.asr, "reset_statistics"):
            self.asr.reset_statistics()
        
        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        master_metadata["config"]["pipeline_options"] = self.pipeline_options
        
        if hasattr(self, 'smart_postprocessor'):
            logger.debug("Smart Post-Processing enabled.")
            
        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 3)
            self.progress.set_current_step("Transforming audio", 1, 3)
            
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path),
                duration_seconds=duration
            )

            # Trace audio extraction
            self.tracer.emit_audio_extraction(str(audio_path), duration)

            # Trace ASR config before transcription (faster mode uses stable-ts with faster-whisper)
            self.tracer.emit_asr_config(
                model=self.asr.model_name,
                backend="stable-ts/faster-whisper",
                params=self.pipeline_options.get("decoder", {})
            )

            # Step 2: Transcribe entire audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing (progress bar for the faster mode is shown in your terminal...)", 2, 3)
                # Report transcription start with method info
                self.progress_reporter.report('transcription_start', 
                                            filename=media_basename,
                                            method='direct',
                                            model=self.asr.model_name,
                                            duration=duration)
            self.progress.set_current_step("Transcribing (progress bar for the faster mode is shown in your terminal...)", 2, 3)
            
            logger.info("Starting transcription of entire audio ...")
            raw_srt_path = self.temp_dir / f"{media_basename}_raw.srt"

            self.asr.transcribe_to_srt(audio_path, raw_srt_path, task=self.asr_task)
            
            # Report transcription complete
            if self.progress_reporter:
                self.progress_reporter.report('transcription_complete',
                                            filename=media_basename,
                                            duration=time.time() - start_time)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                model=self.asr.model_name,
                output_path=str(raw_srt_path)
            )
            
            # Step 3: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 3, 3)
            self.progress.set_current_step("Post-processing subtitles", 3, 3)

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
           
            # The postprocessor returns the path to the new sanitized file in the temp directory
            processed_srt, stats = self.postprocessor.process(raw_srt_path, final_srt_path)

            # Trace postprocessing
            self.tracer.emit_postprocessing(stats)

            # Explicitly move the sanitized file from the temp location to the final destination
            shutil.move(processed_srt, final_srt_path)

            # Move raw_subs BEFORE cleanup
            temp_raw_subs_path = raw_srt_path.parent / "raw_subs"
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
            
            # Now cleanup temp files
            self.cleanup_temp_files(media_basename)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                output_path=str(final_srt_path),
                statistics=stats
            )
            
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["raw_srt"] = str(raw_srt_path)
            
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = stats.get('total_subtitles', 0)
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0)
            }

            logprob_filtered = 0
            nonverbal_filtered = 0
            if hasattr(self.asr, "get_filter_statistics"):
                filter_stats = self.asr.get_filter_statistics() or {}
                logprob_filtered = filter_stats.get('logprob_filtered', 0)
                nonverbal_filtered = filter_stats.get('nonverbal_filtered', 0)

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
            
            logger.debug(f"FASTER pipeline completed in {total_time:.1f} seconds")
            logger.info(f"Output saved to: {final_srt_path}")
            
            # Report completion
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=True,
                    stats={
                        'subtitles': master_metadata["summary"]["final_subtitles_refined"],
                        'duration': total_time,
                        'scenes': 1  # Direct transcription has no scenes
                    }
                )

            # Trace completion
            self.tracer.emit_completion(
                success=True,
                final_subtitles=master_metadata["summary"]["final_subtitles_refined"],
                total_duration=total_time,
                output_path=str(final_srt_path)
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

            # Trace failure
            self.tracer.emit_completion(
                success=False,
                final_subtitles=0,
                total_duration=time.time() - start_time,
                output_path="",
                error=str(e)
            )

            raise
