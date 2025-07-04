#!/usr/bin/env python3
"""Faster pipeline implementation - direct transcription without chunking."""

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

class FasterPipeline(BasePipeline):
    """Faster pipeline using Whisper turbo mode without chunking."""

    def __init__(self, 
                 output_dir: str, 
                 temp_dir: str, 
                 keep_temp_files: bool, 
                 subs_language: str,
                 resolved_params: Dict, 
                 progress_display=None, 
                 **kwargs):
        """
        Initializes the FasterPipeline using resolved configuration parameters.
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)
        
        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Unpack the resolved parameter dictionaries from the TranscriptionTuner
        load_params = resolved_params.get('model_load_params', {})
        self.transcribe_options = resolved_params.get('transcribe_options', {})
        self.decode_options = resolved_params.get('decode_options', {})
        stable_ts_opts = resolved_params.get('stable_ts_options', {})
        post_proc_opts = resolved_params.get('post_processing_options', {})
        
        # Set the ASR task based on the user's choice
        self.transcribe_options['task'] = 'translate' if self.subs_language == 'english-direct' else 'transcribe'

        # Implement the smart model-switching logic
        effective_model_name = load_params.get("model_name", "turbo")
        if self.subs_language == 'english-direct' and effective_model_name == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_name = 'large-v2'
        
        # Instantiate modules with the final, correct parameters
        self.audio_extractor = AudioExtractor()
        
        self.asr = StableTSASR(
            model_load_params={**load_params, 'model_name': effective_model_name},
            transcribe_options=self.transcribe_options,
            decode_options=self.decode_options,
            stable_ts_options=stable_ts_opts,
            turbo_mode=True # This is specific to the faster pipeline
        )
        

        lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
        self.postprocessor = SRTPostProcessor(language=lang_code, **post_proc_opts)


    def get_mode_name(self) -> str:
        return "faster"
        
    # --- THIS FUNCTION HAS BEEN RESTORED TO ITS COMPLETE BASELINE VERSION ---
    def process(self, media_info: Dict) -> Dict:
        """Process media file through faster pipeline."""
        start_time = time.time()
        
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        logger.info(f"Starting FASTER pipeline for: {input_file}")
        logger.debug(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
         
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )
        
        master_metadata["config"]["pipeline_options"] = {
            "model": f"whisper-({self.asr.model_name})",
            "device": self.asr.device,
            "language": "ja"
        }
        
        if self.smart_postprocessing:
            logger.debug("Smart Post-Processing enabled.")
            
        try:
            self.progress.set_current_step("Transforming audio", 1, 3)
            
            
            #logger.info("Step 1: Transforming audio")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path),
                duration_seconds=duration
            )
            
            #logger.info("Step 2: Transcribing")
            self.progress.set_current_step("Transcribing (this may take a while...)", 2, 3)
            logger.info("Starting transcription of entire audio ...")
            raw_srt_path = self.temp_dir / f"{media_basename}_raw.srt"

            self.asr.transcribe_to_srt(audio_path, raw_srt_path, task=self.transcribe_options['task'])
            
           
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                model=self.asr.model_name,
                output_path=str(raw_srt_path)
            )
            
            
            #logger.info("Step 3: Post-processing SRT")
            self.progress.set_current_step("Post-processing subtitles", 3, 3)

            lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
            final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
           
            # The postprocessor returns the path to the new sanitized file in the temp directory
            processed_srt, stats = self.postprocessor.process(raw_srt_path, final_srt_path)
            
            # Explicitly move the sanitized file from the temp location to the final destination
            shutil.move(processed_srt, final_srt_path)

            # FIX: Move raw_subs BEFORE cleanup
            temp_raw_subs_path = raw_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                # Create raw_subs directory if it doesn't exist
                final_raw_subs_path.mkdir(exist_ok=True)
                
                # CHANGED: Copy only files related to current media_basename to avoid ghost files
                for file in temp_raw_subs_path.glob(f"{media_basename}*"):
                    dest_file = final_raw_subs_path / file.name
                    shutil.copy2(file, dest_file)
                    logger.debug(f"Copied {file.name} to raw_subs")
                    
                logger.debug(f"Copied relevant raw_subs files to: {final_raw_subs_path}")
            
            # FIX: Now cleanup temp files
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
            
            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            self.cleanup_temp_files(media_basename)
            
            logger.debug(f"FASTER pipeline completed in {total_time:.1f} seconds")
            logger.info(f"Output saved to: {final_srt_path}")
            
            return master_metadata
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed",
                error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise
