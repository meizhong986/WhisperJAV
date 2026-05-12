#!/usr/bin/env python3
"""
WhisperCpp pipeline implementation - scene detection with PyWhisperCpp ASR.

This pipeline provides Metal GPU acceleration on Apple Silicon via whisper.cpp,
solving the faster-whisper MPS limitation. Uses auditok scene detection by default.
"""

import shutil
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.pywhispercpp_asr import PyWhisperCppASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor
from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.parameter_tracer import NullTracer

from whisperjav.modules.speech_enhancement import (
    create_enhancer_from_config,
    enhance_scenes,
    get_extraction_sample_rate,
    is_passthrough_backend,
)


class WhisperCppPipeline(BasePipeline):
    """
    Whisper.cpp pipeline with Metal GPU acceleration on Apple Silicon.

    Uses auditok scene detection by default. Designed for Mac users who want
    faster transcription without the CPU limitation of faster-whisper.
    """

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
        """
        Initialize WhisperCpp pipeline.

        Args:
            output_dir: Output directory for subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            resolved_config: V3 structured configuration
            progress_display: Progress display object
            **kwargs: Additional parameters
        """
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
        )

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Extract kwargs
        self.progress_reporter = kwargs.get("progress_reporter", None)
        self.tracer = kwargs.get("parameter_tracer", NullTracer())

        # --- V3 CONFIG UNPACKING ---
        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]

        self.asr_task = task

        # Extract feature configurations
        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})

        self.scene_detection_params = scene_opts
        self.vad_params = params.get("vad", {})

        # Model switching logic (same as balanced)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == "direct-to-english" and model_cfg.get("model_name") == "turbo":
            logger.info("Direct translation requested. Switching to 'large-v2'.")
            effective_model_cfg["model_name"] = "large-v2"

        self.pipeline_options = {
            "model": effective_model_cfg,
            "decoder": params.get("decoder", {}),
            "provider": params.get("provider", {}),
            "vad": self.vad_params,
            "task": task,
        }

        # Speech enhancement config
        self._enhancer_config = resolved_config
        enhancer_params = params.get("speech_enhancer", {})
        self._enhancer_backend_name = enhancer_params.get("backend", "none") or "none"
        self._enhancer_is_passthrough = is_passthrough_backend(self._enhancer_backend_name)

        # Audio extraction at appropriate sample rate
        extraction_sr = get_extraction_sample_rate(self._enhancer_backend_name)
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)
        self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_opts)

        # ASR config (lazy initialization)
        self._asr_config = {
            "model_config": effective_model_cfg,
            "params": params,
            "task": task,
            "tracer": self.tracer,
        }
        self._asr = None

        self.stitcher = SRTStitcher()

        # Language code
        if self.subs_language == "direct-to-english":
            self.lang_code = "en"
        else:
            # Handle both 'decoder' (legacy) and 'asr' (pywhispercpp) structures
            decoder_params = params.get("decoder", params.get("asr", {}))
            self.lang_code = decoder_params.get("language", "ja")
        self.standard_postprocessor = StandardPostProcessor(
            language=self.lang_code, **post_proc_opts
        )

    def get_mode_name(self) -> str:
        """Return pipeline mode name."""
        return "whispercpp"

    def _ensure_asr(self) -> PyWhisperCppASR:
        """
        Lazy ASR initialization - create once, reuse for all files.

        Note: whisper.cpp doesn't have the ctranslate2 destructor crash issue,
        so we don't need the IMMORTAL_OBJECT pattern. Just standard lazy loading.

        Returns:
            PyWhisperCppASR instance
        """
        if self._asr is None:
            logger.info("Initializing whisper.cpp ASR model (Metal GPU on Mac)")
            self._asr = PyWhisperCppASR(**self._asr_config)
        else:
            logger.debug("Reusing whisper.cpp ASR instance")

        return self._asr

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through whispercpp pipeline.

        Args:
            media_info: Media file information dict

        Returns:
            Processing result dict
        """
        start_time = time.time()
        input_file = media_info["path"]
        media_basename = media_info["basename"]

        # Progress reporting
        if self.progress_reporter:
            self.progress_reporter.report_file_start(
                filename=media_basename,
                file_number=media_info.get("file_number", 1),
                total_files=media_info.get("total_files", 1),
            )

        self.tracer.emit_file_start(
            filename=media_basename,
            file_number=media_info.get("file_number", 1),
            total_files=media_info.get("total_files", 1),
            media_info=media_info,
        )

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info,
        )

        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        master_metadata["config"]["pipeline_options"] = self.pipeline_options

        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Extracting audio", 1, 5)
            self.progress.set_current_step("Extracting audio", 1, 5)

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration

            self.tracer.emit_audio_extraction(str(audio_path), duration)

            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting scenes", 2, 5)
            self.progress.set_current_step("Detecting scenes", 2, 5)

            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            detection_result = self.scene_detector.detect_scenes(
                extracted_audio, scenes_dir, media_basename
            )
            scene_paths = detection_result.to_legacy_tuples()

            detection_meta = detection_result.to_metadata_dict()
            master_metadata["scenes_detected"] = detection_meta["scenes_detected"]
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)

            self.tracer.emit_scene_detection(
                method=self.scene_detector.name,
                params=self.scene_detection_params,
                scenes_found=len(scene_paths),
                scene_stats={
                    "total_duration": sum(d for _, _, _, d in scene_paths),
                    "shortest": min((d for _, _, _, d in scene_paths), default=0),
                    "longest": max((d for _, _, _, d in scene_paths), default=0),
                },
            )

            # Step 3: Speech enhancement (optional)
            self.progress.set_current_step("Preparing audio", 3, 5)

            if self._enhancer_is_passthrough:
                logger.info(
                    "Speech enhancer is passthrough — %d scenes at 16kHz",
                    len(scene_paths),
                )
                enhancer_name = "none"
            else:
                import gc

                enhancer = create_enhancer_from_config(self._enhancer_config)
                enhancer_name = enhancer.name
                logger.info(f"Enhancing {len(scene_paths)} scenes with {enhancer.display_name}")

                def enhancement_progress(scene_num, total, name):
                    if scene_num == 1 or scene_num % 5 == 0 or scene_num == total:
                        pct = (scene_num / total) * 100
                        print(f"\rEnhancing: [{scene_num}/{total}] {pct:.0f}%", end="", flush=True)

                scene_paths = enhance_scenes(
                    scene_paths,
                    enhancer,
                    self.temp_dir,
                    progress_callback=enhancement_progress,
                )
                print()

                # Cleanup enhancer before ASR
                enhancer.cleanup()
                del enhancer
                gc.collect()

            master_metadata["config"]["speech_enhancement"] = {
                "enabled": not self._enhancer_is_passthrough,
                "backend": enhancer_name,
            }

            # Step 4: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing with whisper.cpp", 4, 5)
            self.progress.set_current_step("Transcribing with whisper.cpp", 4, 5)

            asr = self._ensure_asr()

            self.tracer.emit_asr_config(
                model=asr.model_name,
                backend="whisper.cpp",
                params=self.pipeline_options.get("decoder", {}),
            )

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            self.progress.start_subtask("Transcribing scenes", len(scene_paths))
            total_scenes = len(scene_paths)

            print(f"\nTranscribing {total_scenes} scenes with whisper.cpp (Metal GPU):")

            last_update_time = time.time()
            update_interval = 30

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                scene_num = idx + 1

                # Progress update
                should_show = (
                    scene_num == 1
                    or scene_num % 5 == 0
                    or time.time() - last_update_time > update_interval
                    or scene_num == total_scenes
                )

                if should_show:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / scene_num if scene_num > 0 else 0
                    remaining = avg_time * (total_scenes - scene_num)
                    print(
                        f"\r  [{scene_num}/{total_scenes}] "
                        f"{scene_num/total_scenes*100:.0f}% "
                        f"(~{remaining/60:.1f} min remaining)",
                        end="",
                        flush=True,
                    )
                    last_update_time = time.time()

                # Transcribe scene
                try:
                    asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)

                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                except Exception as e:
                    logger.warning(f"Scene {scene_num} transcription failed: {e}")
                    continue

            print()  # Newline after progress

            logger.info(f"[DONE] Transcribed {len(scene_srt_info)} scenes")

            # Step 5: Stitch SRTs
            if self.progress_reporter:
                self.progress_reporter.report_step("Stitching subtitles", 5, 5)
            self.progress.set_current_step("Stitching subtitles", 5, 5)

            final_srt_path = self.output_dir / f"{media_basename}.srt"

            if scene_srt_info:
                self.stitcher.stitch(scene_srt_info, final_srt_path)
                master_metadata["output_files"]["subtitle_file"] = str(final_srt_path)
                master_metadata["output_files"]["final_srt"] = str(final_srt_path)

                # Count subtitles
                if final_srt_path.exists():
                    with open(final_srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    import srt
                    subs = list(srt.parse(content))
                    subtitle_count = len(subs)
                    master_metadata["summary"]["final_subtitles_refined"] = subtitle_count
                    master_metadata["summary"]["final_subtitles_raw"] = subtitle_count
                else:
                    master_metadata["summary"]["final_subtitles_refined"] = 0
                    master_metadata["summary"]["final_subtitles_raw"] = 0

                # Post-processing
                postproc_path, postproc_stats = self.standard_postprocessor.process(final_srt_path)
                master_metadata["processing_stages"]["postprocessing"] = {
                    "applied": postproc_stats.get("applied", []),
                    "changes": postproc_stats.get("changes", {}),
                }
            else:
                logger.warning("No scenes were successfully transcribed")
                master_metadata["output_files"]["subtitle_file"] = None
                master_metadata["output_files"]["final_srt"] = None
                master_metadata["summary"]["final_subtitles_refined"] = 0
                master_metadata["summary"]["final_subtitles_raw"] = 0

            # Metadata
            processing_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = processing_time
            master_metadata["summary"]["scenes_per_minute"] = (
                len(scene_srt_info) / (processing_time / 60) if processing_time > 0 else 0
            )

            self.tracer.emit_completion(
                success=True,
                output_path=str(final_srt_path),
                processing_time=processing_time,
            )

            return master_metadata

        except Exception as e:
            logger.error(f"WhisperCpp pipeline failed: {e}")
            master_metadata["errors"].append({
                "type": "pipeline_error",
                "message": str(e),
            })
            master_metadata["summary"]["pipeline_error"] = str(e)

            return master_metadata

    def cleanup(self) -> None:
        """
        Cleanup pipeline resources.
        """
        if self._asr is not None:
            self._asr.cleanup()
            self._asr = None
            logger.debug("Whisper.cpp ASR cleaned up")

        super().cleanup()