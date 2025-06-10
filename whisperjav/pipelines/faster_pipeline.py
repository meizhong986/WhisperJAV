#!/usr/bin/env python3
"""Faster pipeline implementation - direct transcription without chunking."""

from pathlib import Path
from typing import Dict
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.stable_ts_asr import StableTSASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor
from whisperjav.utils.logger import logger


class FasterPipeline(BasePipeline):
    """Faster pipeline using Whisper turbo mode without chunking."""
    
    def __init__(self, 
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False):
        super().__init__(output_dir, temp_dir, keep_temp_files)
        
        # Initialize modules (NO MediaDiscovery!)
        self.audio_extractor = AudioExtractor()
        self.asr = StableTSASR(turbo_mode=True)
        self.postprocessor = SRTPostProcessor()
        
    def get_mode_name(self) -> str:
        return "faster"
        
    def process(self, media_info: Dict) -> Dict:
        """Process media file through faster pipeline."""
        start_time = time.time()
        
        # Extract info from the media_info dictionary passed from main
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        logger.info(f"Starting FASTER pipeline for: {input_file}")
        logger.info(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
         
        # Create master metadata with the already discovered media info
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info  # Pass the complete media info
        )
        
        # Update config
        master_metadata["config"]["pipeline_options"] = {
            "model": "whisper-turbo-faster-whisper",
            "device": self.asr.device,
            "language": "ja"
        }
        
        try:
            # Step 1: Audio extraction (No more media discovery here!)
            logger.info("Step 1: Extracting audio")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            
            # Update metadata
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path),
                duration_seconds=duration
            )
            
            # Step 2: Direct transcription with turbo model
            logger.info("Step 2: Transcribing with Whisper Turbo")
            raw_srt_path = self.temp_dir / f"{media_basename}_raw.srt"
            self.asr.transcribe_to_srt(audio_path, raw_srt_path)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                model="whisper-turbo",
                output_path=str(raw_srt_path)
            )
            
            # Step 3: Post-processing
            logger.info("Step 3: Post-processing SRT")
            final_srt_path = self.output_dir / f"{media_basename}.ja.whisperjav.srt"
            processed_srt, stats = self.postprocessor.process(raw_srt_path, final_srt_path)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                output_path=str(final_srt_path),
                statistics=stats
            )
            
            # Update final output info
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["raw_srt"] = str(raw_srt_path)
            
            # Update summary
            master_metadata["summary"]["final_subtitles_refined"] = stats['total_subtitles'] - stats['empty_removed']
            master_metadata["summary"]["final_subtitles_raw"] = stats['total_subtitles']
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats['removed_hallucinations'],
                "repetitions_removed": stats['removed_repetitions'],
                "duration_adjustments": stats['duration_adjustments'],
                "empty_removed": stats['empty_removed']
            }
            
            # Calculate total processing time
            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            
            # Update the updated_at timestamp
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            # Save metadata
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            # Cleanup if requested
            self.cleanup_temp_files(media_basename)
            
            logger.info(f"FASTER pipeline completed in {total_time:.1f} seconds")
            logger.info(f"Output saved to: {final_srt_path}")
            
            return master_metadata
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed",
                error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise