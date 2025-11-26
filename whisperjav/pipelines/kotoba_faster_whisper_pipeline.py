#!/usr/bin/env python3
"""
Kotoba Faster-Whisper Pipeline.

Japanese-optimized pipeline using kotoba-tech/kotoba-whisper-v2.0-faster model
with internal VAD support and mandatory scene detection.

Key Features:
- Scene detection is ALWAYS enabled (method: auditok or silero)
- Uses faster-whisper's internal VAD (controllable via --no-vad)
- Optimized for Japanese speech recognition
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.kotoba_faster_whisper_asr import KotobaFasterWhisperASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor
from whisperjav.modules.scene_detection import DynamicSceneDetector
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress


class KotobaFasterWhisperPipeline(BasePipeline):
    """
    Japanese-optimized pipeline using Kotoba Faster-Whisper model.

    Scene detection is ALWAYS enabled. User selects method (auditok or silero).
    Internal VAD is enabled by default but can be disabled via config.
    """

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool,
        subs_language: str,
        resolved_config: Dict,
        scene_method: str = "auditok",
        progress_display=None,
        **kwargs
    ):
        """
        Initialize Kotoba Faster-Whisper Pipeline.

        Args:
            output_dir: Output directory for final subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            resolved_config: V3 structured configuration from resolver
            scene_method: Scene detection method ('auditok' or 'silero')
            progress_display: Progress display object
            **kwargs: Additional parameters for base class
        """
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            **kwargs
        )

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language
        self.scene_method = scene_method
        self.progress_reporter = kwargs.get('progress_reporter', None)

        # --- V3 STRUCTURED CONFIG UNPACKING ---
        model_cfg = resolved_config.get("model", {})
        params = resolved_config.get("params", {})
        features = resolved_config.get("features", {})
        task = resolved_config.get("task", "transcribe")

        self.asr_task = task

        # Extract feature configurations
        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})

        # Store params for metadata logging
        self.scene_detection_params = {
            "method": self.scene_method,
            **scene_opts
        }

        # Get ASR params (may contain vad_filter, etc.)
        asr_params = params.get("asr", params)  # Support both param structures
        self.vad_params = {
            "vad_filter": asr_params.get("vad_filter", True),
            "vad_threshold": asr_params.get("vad_threshold", 0.01),
        }

        # --- END V3 CONFIG UNPACKING ---

        # Initialize modules
        self.audio_extractor = AudioExtractor()

        # Scene detector (ALWAYS enabled)
        self.scene_detector = DynamicSceneDetector(method=self.scene_method, **scene_opts)

        # Kotoba ASR with internal VAD
        self.asr = KotobaFasterWhisperASR(
            model_config=model_cfg,
            params=asr_params,
            task=task
        )

        self.stitcher = SRTStitcher()

        # Language code for post-processor and output filenames
        if self.subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            self.lang_code = asr_params.get("language", "ja")

        self.postprocessor = SRTPostProcessor(language=self.lang_code, **post_proc_opts)

    def get_mode_name(self) -> str:
        """Return pipeline mode name."""
        return "kotoba-faster-whisper"

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through Kotoba Faster-Whisper pipeline.

        Steps:
        1. Extract audio
        2. Detect scenes (ALWAYS - using configured method)
        3. Transcribe each scene with KotobaFasterWhisperASR
        4. Stitch scene SRTs together
        5. Post-process final SRT

        Args:
            media_info: Media information dict

        Returns:
            Master metadata dict
        """
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
                self.progress_reporter.report_step("Extracting audio", 1, 5)
            self.progress.set_current_step("Extracting audio", 1, 5)

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration
            )

            # Step 2: Detect scenes (ALWAYS enabled)
            if self.progress_reporter:
                self.progress_reporter.report_step(f"Detecting scenes ({self.scene_method})", 2, 5)
            self.progress.set_current_step(f"Detecting scenes ({self.scene_method})", 2, 5)

            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)

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
                scene_count=len(scene_paths), scenes_dir=str(scenes_dir),
                method=self.scene_method
            )

            # Step 3: Transcribe scenes with Kotoba ASR
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing with Kotoba ASR", 3, 5)
            self.progress.set_current_step("Transcribing with Kotoba ASR", 3, 5)

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            self.progress.start_subtask("Transcribing scenes", len(scene_paths))

            total_scenes = len(scene_paths)
            transcription_start_time = time.time()

            print(f"\nTranscribing {total_scenes} scenes with Kotoba Faster-Whisper:")

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                scene_num = idx + 1

                # Progress display
                progress_pct = (scene_num / total_scenes) * 100
                bar_width = 30
                filled_width = int(bar_width * scene_num / total_scenes)
                progress_bar = '=' * filled_width + '-' * (bar_width - filled_width)

                # Calculate ETA
                eta_text = ""
                if scene_num > 2:
                    elapsed = time.time() - transcription_start_time
                    avg_time_per_scene = elapsed / scene_num
                    remaining_scenes = total_scenes - scene_num
                    eta_seconds = remaining_scenes * avg_time_per_scene
                    if eta_seconds > 60:
                        eta_text = f" | ETA: {eta_seconds/60:.1f}m"
                    else:
                        eta_text = f" | ETA: {eta_seconds:.0f}s"

                scene_filename = scene_path.name[:25] + "..." if len(scene_path.name) > 25 else scene_path.name
                progress_line = f"\rTranscribing: [{progress_bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_filename}{eta_text}"
                print(progress_line, end='', flush=True)

                try:
                    self.asr.transcribe_to_srt(scene_path, scene_srt_path)

                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["no_speech_detected"] = True

                    self.progress.update_subtask(1)

                except Exception as e:
                    logger.error(f"Scene {scene_num} transcription failed: {e}")
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    self.progress.update_subtask(1)

            self.progress.finish_subtask()
            print(f"\n[DONE] Completed transcription of {total_scenes} scenes")

            master_metadata["summary"]["scenes_processed_successfully"] = len(scene_srt_info)
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                scenes_transcribed=len(scene_srt_info)
            )

            # Step 4: Stitch scene SRTs
            if self.progress_reporter:
                self.progress_reporter.report_step("Stitching scene transcriptions", 4, 5)
            self.progress.set_current_step("Stitching scene transcriptions", 4, 5)

            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed",
                subtitle_count=num_subtitles, output_path=str(stitched_srt_path)
            )

            # Step 5: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 5, 5)
            self.progress.set_current_step("Post-processing subtitles", 5, 5)

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)

            # Ensure final SRT is in output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)
                logger.debug(f"Copied final SRT to {final_srt_path}")

            # Move raw_subs folder to output directory
            temp_raw_subs_path = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                final_raw_subs_path.mkdir(exist_ok=True)
                for file in temp_raw_subs_path.glob(f"{media_basename}*"):
                    dest_file = final_raw_subs_path / file.name
                    shutil.copy2(file, dest_file)
                logger.debug(f"Copied raw_subs files to: {final_raw_subs_path}")

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                statistics=stats, output_path=str(final_srt_path)
            )

            # Update metadata
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0),
                "cps_filtered": stats.get('cps_filtered', 0)
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

            logger.info(f"Output saved to: {final_srt_path}")
            return master_metadata

        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)

            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )

            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'asr') and self.asr:
            self.asr.cleanup()
        super().cleanup()
