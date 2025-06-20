#!/usr/bin/env python3
"""Balanced pipeline implementation - scene detection with WhisperPro ASR."""

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
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger

from whisperjav.modules.segment_classification import SegmentClassifier
from whisperjav.modules.audio_preprocessing import AudioPreprocessor
from whisperjav.modules.srt_postproduction import SRTPostProduction


class BalancedPipeline(BasePipeline):
    """Balanced pipeline using scene detection with WhisperPro ASR (VAD-enhanced)."""
    
    # --- FUNCTION MODIFIED ---
    def __init__(self, 
                 output_dir: str, 
                 temp_dir: str, 
                 keep_temp_files: bool, 
                 subs_language: str,
                 resolved_params: Dict, 
                 **kwargs):
        """
        Initializes the BalancedPipeline using resolved configuration parameters.
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)
        
        self.subs_language = subs_language

        # Unpack the resolved parameter dictionaries
        load_params = resolved_params.get('model_load_params', {})
        self.transcribe_options = resolved_params.get('transcribe_options', {})
        self.decode_options = resolved_params.get('decode_options', {})
        vad_opts = resolved_params.get('vad_options', {})
        post_proc_opts = resolved_params.get('post_processing_options', {})
        scene_opts = resolved_params.get('scene_options', {})

        # Set the ASR task based on the chosen output language
        self.asr_task = 'translate' if self.subs_language == 'english-direct' else 'transcribe'
        
        # Store params for metadata logging
        self.scene_detection_params = scene_opts
        self.vad_params = vad_opts
        
        # Implement the smart model-switching logic
        effective_model_name = load_params.get("model_name", "large-v2")
        if self.subs_language == 'english-direct' and effective_model_name == 'turbo':
            logger.info("Direct translation requested. Switching model from 'turbo' to 'large-v2' to perform translation.")
            effective_model_name = 'large-v2'

        # Instantiate modules
        self.audio_extractor = AudioExtractor()
        self.scene_detector = SceneDetector(**scene_opts)
        
        # Pass the effective model name to the load_params dictionary
        final_load_params = {**load_params, 'model_name': effective_model_name}
        
        self.asr = WhisperProASR(
            model_load_params=final_load_params,
            vad_options=vad_opts,
            transcribe_options=self.transcribe_options,
            decode_options=self.decode_options
        )
    
        self.stitcher = SRTStitcher()
        
        
        lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
        self.standard_postprocessor = StandardPostProcessor(language=lang_code, **post_proc_opts)

        
    
        self.smart_postprocessor = SRTPostProduction()
        self.classifier = SegmentClassifier()
        self.preprocessor = AudioPreprocessor()

    def get_mode_name(self) -> str:
        return "balanced"

    # --- THIS FUNCTION HAS BEEN RESTORED TO ITS COMPLETE BASELINE VERSION ---
    def process(self, media_info: Dict) -> Dict:
        """Process media file through balanced pipeline with scene detection and VAD-enhanced ASR."""
        start_time = time.time()
        
        input_file = media_info['path']
        media_basename = media_info['basename']
        
        logger.info(f"Starting BALANCED pipeline for: {input_file}")
        logger.info(f"Media type: {media_info['type']}, Duration: {media_info.get('duration', 'unknown')}s")
        
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )
        
        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        
        try:
            logger.info("Step 1: Extracting audio")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed", output_path=str(audio_path), duration_seconds=duration)

            logger.info("Step 2: Detecting audio scenes")
            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            logger.info(f"Detected {len(scene_paths)} scenes")
            
            master_metadata["scenes_detected"] = []
            for idx, (scene_path, start_time_sec, end_time_sec, duration_sec) in enumerate(scene_paths):
                scene_info = {
                    "scene_index": idx, "filename": scene_path.name,
                    "start_time_seconds": round(start_time_sec, 3), "end_time_seconds": round(end_time_sec, 3),
                    "duration_seconds": round(duration_sec, 3), "path": str(scene_path)
                }
                master_metadata["scenes_detected"].append(scene_info)
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed", scene_count=len(scene_paths), scenes_dir=str(scenes_dir))
            
            logger.info("Step 3: Transcribing scenes with WhisperPro ASR (VAD-enhanced)")
            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                logger.info(f"Transcribing scene {idx+1}/{len(scene_paths)}: {scene_path.name}")
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                try:
                    self.asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["no_speech_detected"] = True
                except Exception as e:
                    logger.error(f"Failed to transcribe scene {idx}: {e}")
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
            
            logger.info("Step 4: Combining scene transcriptions")
            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed", subtitle_count=num_subtitles, output_path=str(stitched_srt_path))

            logger.info("Step 5: Post-processing final SRT")
            lang_code = 'en' if self.subs_language == 'english-direct' else 'ja'
            final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
            _, stats = self.standard_postprocessor.process(stitched_srt_path, final_srt_path)
            
            # FIX: Move raw_subs folder to output directory
            temp_raw_subs_path = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                shutil.copytree(temp_raw_subs_path, final_raw_subs_path, dirs_exist_ok=True)
                logger.info(f"Copied raw_subs to: {final_raw_subs_path}")



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
            
            logger.info(f"BALANCED pipeline completed in {total_time:.1f} seconds")
            logger.info(f"Output saved to: {final_srt_path}")
            
            return master_metadata
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(master_metadata, "error", "failed", error_message=str(e))
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise
