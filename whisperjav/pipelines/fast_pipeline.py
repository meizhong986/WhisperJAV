#!/usr/bin/env python3
"""Fast pipeline implementation - using standard Whisper with scene detection."""

from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.stable_ts_asr import StableTSASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor
from whisperjav.modules.scene_detection import SceneDetector
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger

class FastPipeline(BasePipeline):
    """Fast pipeline using standard Whisper with mandatory scene detection."""
    
    def __init__(self, 
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False,
                 model_name: str = "turbo",
                 # Scene detection parameters
                 scene_max_duration: float = 30.0,
                 scene_min_duration: float = 0.2,
                 scene_max_silence: float = 2.0,
                 scene_energy_threshold: int = 50):
        super().__init__(output_dir, temp_dir, keep_temp_files)
        
        # Store scene detection parameters for metadata
        self.scene_detection_params = {
            "min_dur": scene_min_duration,
            "max_dur": scene_max_duration,
            "max_silence": scene_max_silence,
            "energy": scene_energy_threshold
        }
        
        # Initialize modules (No MediaDiscovery!)
        self.audio_extractor = AudioExtractor()
        self.scene_detector = SceneDetector(
            max_duration=scene_max_duration,
            min_duration=scene_min_duration,
            max_silence=scene_max_silence,
            energy_threshold=scene_energy_threshold
        )
        self.asr = StableTSASR(
            model_name=model_name,
            turbo_mode=False,  # Use standard whisper, not faster-whisper
            device="cuda"
        )
        self.stitcher = SRTStitcher()
        self.postprocessor = SRTPostProcessor()
        
    def get_mode_name(self) -> str:
        return "fast"
        
    def process(self, media_info: Dict) -> Dict:
        """Process media file through fast pipeline with mandatory scene detection."""
        start_time = time.time()
        
        # Extract info from the media_info dictionary passed from main
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        logger.info(f"Starting FAST pipeline for: {input_file}")
        logger.info(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
        
        # Create master metadata with the already discovered media info
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )
        
        # Update config with actual scene detection parameters
        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["pipeline_options"] = {
            "model": "whisper-turbo-standard",
            "device": self.asr.device,
            "language": "ja"
        }
        
        try:
            # Step 1: Audio extraction
            logger.info("Step 1: Extracting audio")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path),
                duration_seconds=duration
            )
            
            # Step 2: Detect audio scenes
            logger.info("Step 2: Detecting audio scenes")
            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            
            scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            logger.info(f"Detected {len(scene_paths)} scenes")
            
            # Populate scenes_detected array in metadata
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
            
            # Step 3: Transcribe each scene
            logger.info("Step 3: Transcribing scenes with standard Whisper (turbo model)")
            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            
            scene_srt_info = []
            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                logger.info(f"Transcribing scene {idx+1}/{len(scene_paths)}: {scene_path.name}")
                
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                try:
                    self.asr.transcribe_to_srt(scene_path, scene_srt_path)
                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                except Exception as e:
                    logger.error(f"Failed to transcribe scene {idx}: {e}")
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    continue
            
            master_metadata["summary"]["scenes_processed_successfully"] = len(scene_srt_info)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                model="whisper-turbo-standard",
                scenes_transcribed=len(scene_srt_info),
                total_scenes=len(scene_paths)
            )
            
            # Step 4: Combine scene SRTs
            logger.info("Step 4: Combining scene transcriptions")
            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed",
                output_path=str(stitched_srt_path),
                subtitle_count=num_subtitles
            )
            
            # Step 5: Post-processing
            logger.info("Step 5: Post-processing final SRT")
            final_srt_path = self.output_dir / f"{media_basename}.ja.whisperjav.srt"
            _, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)
            
            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                output_path=str(final_srt_path),
                statistics=stats
            )
            
            # Update final output info and summary
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            
            master_metadata["summary"]["final_subtitles_refined"] = stats['total_subtitles'] - stats['empty_removed']
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats['removed_hallucinations'],
                "repetitions_removed": stats['removed_repetitions'],
                "duration_adjustments": stats['duration_adjustments'],
                "empty_removed": stats['empty_removed']
            }
            
            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"
            
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            
            self.cleanup_temp_files(media_basename)
            
            logger.info(f"FAST pipeline completed in {total_time:.1f} seconds")
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