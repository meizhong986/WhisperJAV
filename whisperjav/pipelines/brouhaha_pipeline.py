#!/usr/bin/env python3
"""Brouhaha pipeline: scene detection + quality-aware ASR."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time
import shutil

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.quality_aware_asr import QualityAwareASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor
from whisperjav.modules.scene_detection import DynamicSceneDetector
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger


class BrouhahaPipeline(BasePipeline):
    """Scene detection with QualityAware ASR (brouhaha-based routing)."""

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool,
        subs_language: str,
        resolved_config: Dict,
        progress_display=None,
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)

        self.progress = progress_display
        self.subs_language = subs_language

        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]

        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})

        self.audio_extractor = AudioExtractor()
        self.scene_detector = DynamicSceneDetector(**scene_opts)
        self.asr = QualityAwareASR(model_config=model_cfg, params=params, task=task)
        self.stitcher = SRTStitcher()

        lang_code = "en" if subs_language == "english-direct" else "ja"
        self.standard_postprocessor = StandardPostProcessor(language=lang_code, **post_proc_opts)

    def process(self, media_info: Dict) -> Dict:
        start_time = time.time()
        input_file = media_info["path"]
        media_basename = media_info["basename"]

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file, mode=self.get_mode_name(), media_info=media_info
        )

        # Step 1: Extract audio
        self.progress.set_current_step("Transforming audio", 1, 5) if self.progress else None
        audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
        extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
        master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
        master_metadata["input_info"]["audio_duration_seconds"] = duration

        # Step 2: Detect scenes
        self.progress.set_current_step("Detecting audio scenes", 2, 5) if self.progress else None
        scenes_dir = self.temp_dir / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)

        # Step 3: Transcribe scenes using QualityAware ASR
        self.progress.set_current_step("Transcribing scenes (quality-aware)", 3, 5) if self.progress else None
        scene_srts_dir = self.temp_dir / "scene_srts"
        scene_srts_dir.mkdir(exist_ok=True)
        scene_srt_info: List[Tuple[Path, float]] = []
        for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
            scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
            try:
                self.asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr.whisper_params.get("task"))
                if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                    scene_srt_info.append((scene_srt_path, start_time_sec))
            except Exception as e:
                logger.error(f"Scene {idx} transcription failed: {e}")

        # Step 4: Stitch
        self.progress.set_current_step("Combining scene transcriptions", 4, 5) if self.progress else None
        stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
        num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)

        # Step 5: Post-process
        self.progress.set_current_step("Post-processing subtitles", 5, 5) if self.progress else None
        lang_code = "en" if self.subs_language == "english-direct" else "ja"
        final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
        processed_srt_path, stats = self.standard_postprocessor.process(stitched_srt_path, final_srt_path)
        if processed_srt_path != final_srt_path:
            shutil.copy2(processed_srt_path, final_srt_path)

        total_time = time.time() - start_time
        master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
        master_metadata["output_files"]["final_srt"] = str(final_srt_path)
        master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)

        return master_metadata

    def get_mode_name(self) -> str:
        return "brouhaha"
