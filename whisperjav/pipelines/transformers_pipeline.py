#!/usr/bin/env python3
"""
HuggingFace Transformers Pipeline for WhisperJAV.

Uses the HuggingFace Transformers pipeline with chunked long-form algorithm
for Japanese audio transcription. Designed as a drop-in mode with dedicated
CLI arguments (--hf-*) that don't affect other pipelines.

Key Features:
- Uses HuggingFace Transformers ASR (default: kotoba-whisper-v2.2)
- Optional scene detection (none, auditok, silero)
- No external VAD needed (HF pipeline handles chunking internally)
- Supports any HuggingFace whisper model via --hf-model-id
"""

import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import srt

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.transformers_asr import TransformersASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress

from whisperjav.modules.speech_enhancement import (
    create_enhancer_direct,
    get_extraction_sample_rate,
    enhance_scenes,
    enhance_single_audio,
)


class TransformersPipeline(BasePipeline):
    """
    HuggingFace Transformers-based ASR pipeline.

    This pipeline uses the HuggingFace Transformers library with chunked
    long-form transcription. It's optimized for Japanese with kotoba-whisper
    as the default model, but supports any HuggingFace whisper model.

    Scene detection is optional (default: none). When enabled, audio is
    split into scenes before transcription, with timestamps adjusted
    during stitching.
    """

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        progress_display=None,
        # HF Transformers specific config
        hf_model_id: str = "kotoba-tech/kotoba-whisper-v2.2",
        hf_chunk_length: int = 15,
        hf_stride: Optional[float] = None,
        hf_batch_size: int = 16,
        hf_scene: str = "none",
        hf_beam_size: int = 5,
        hf_temperature: float = 0.0,
        hf_attn: str = "sdpa",
        hf_timestamps: str = "segment",
        hf_language: str = "ja",
        hf_task: str = "transcribe",
        hf_device: str = "auto",
        hf_dtype: str = "auto",
        # Speech enhancement (default: zipenhancer for lightweight SOTA quality)
        hf_speech_enhancer: str = "zipenhancer",
        hf_speech_enhancer_model: Optional[str] = None,
        # Standard options
        subs_language: str = "native",
        **kwargs
    ):
        """
        Initialize HuggingFace Transformers Pipeline.

        Args:
            output_dir: Output directory for final subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            progress_display: Progress display object
            hf_model_id: HuggingFace model ID
            hf_chunk_length: Chunk length in seconds
            hf_stride: Overlap between chunks (None = chunk_length/6)
            hf_batch_size: Batch size for parallel processing
            hf_scene: Scene detection method ('none', 'auditok', 'silero')
            hf_beam_size: Beam size for beam search
            hf_temperature: Sampling temperature
            hf_attn: Attention implementation
            hf_timestamps: Timestamp granularity ('segment' or 'word')
            hf_language: Language code
            hf_task: Task type ('transcribe' or 'translate')
            hf_device: Device to use
            hf_dtype: Data type
            hf_speech_enhancer: Speech enhancement backend ('none', 'zipenhancer', 'clearvoice', 'bs-roformer')
            hf_speech_enhancer_model: Optional model variant for enhancer
            subs_language: Subtitle language ('native' or 'direct-to-english')
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

        # Store HF config
        self.hf_config = {
            "model_id": hf_model_id,
            "chunk_length_s": hf_chunk_length,
            "stride_length_s": hf_stride,
            "batch_size": hf_batch_size,
            "beam_size": hf_beam_size,
            "temperature": hf_temperature,
            "attn_implementation": hf_attn,
            "timestamps": hf_timestamps,
            "language": hf_language,
            "task": hf_task,
            "device": hf_device,
            "dtype": hf_dtype,
        }

        # Scene detection config
        self.scene_method = hf_scene
        if self.scene_method not in ("none", "auditok", "silero"):
            raise ValueError(f"Invalid scene method: {self.scene_method}. Must be 'none', 'auditok', or 'silero'")

        # Determine output language code
        if subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            self.lang_code = hf_language

        # Initialize speech enhancer (if enabled)
        self.speech_enhancer = create_enhancer_direct(
            backend=hf_speech_enhancer,
            model=hf_speech_enhancer_model
        )

        # Determine extraction sample rate based on enhancer
        extraction_sr = get_extraction_sample_rate(self.speech_enhancer)
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)

        # Scene detector (only if enabled)
        self.scene_detector = None
        if self.scene_method != "none":
            from whisperjav.modules.scene_detection import DynamicSceneDetector
            self.scene_detector = DynamicSceneDetector(method=self.scene_method)

        # ASR module (lazy loaded)
        self.asr = TransformersASR(
            model_id=hf_model_id,
            device=hf_device,
            dtype=hf_dtype,
            attn_implementation=hf_attn,
            batch_size=hf_batch_size,
            chunk_length_s=hf_chunk_length,
            stride_length_s=hf_stride,
            language=hf_language,
            task=hf_task,
            timestamps=hf_timestamps,
            beam_size=hf_beam_size,
            temperature=hf_temperature,
        )

        # SRT stitcher (for multi-scene)
        self.stitcher = SRTStitcher()

        # Post-processor
        self.postprocessor = SRTPostProcessor(language=self.lang_code)

        logger.info(f"TransformersPipeline initialized")
        logger.info(f"  Model: {hf_model_id}")
        logger.info(f"  Scene detection: {self.scene_method}")
        if self.speech_enhancer:
            logger.info(f"  Speech enhancer: {self.speech_enhancer.display_name}")
            logger.info(f"  Extraction SR: {extraction_sr}Hz (enhancer preferred rate)")
        else:
            logger.info(f"  Speech enhancer: none")

    def get_mode_name(self) -> str:
        """Return pipeline mode name."""
        return "transformers"

    def _segments_to_srt(self, segments: List[Dict[str, Any]], offset: float = 0.0) -> str:
        """
        Convert ASR segments to SRT format string.

        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
            offset: Time offset to add to all timestamps (for scene stitching)

        Returns:
            SRT formatted string
        """
        subtitles = []

        for idx, seg in enumerate(segments, 1):
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_sec = seg.get("start", 0.0) + offset
            end_sec = seg.get("end", start_sec + 2.0) + offset

            # Create timedelta objects for srt library
            start_td = timedelta(seconds=start_sec)
            end_td = timedelta(seconds=end_sec)

            # Ensure end is after start
            if end_td <= start_td:
                end_td = start_td + timedelta(milliseconds=100)

            subtitle = srt.Subtitle(
                index=idx,
                start=start_td,
                end=end_td,
                content=text
            )
            subtitles.append(subtitle)

        return srt.compose(subtitles)

    def _write_srt(self, srt_content: str, output_path: Path) -> int:
        """
        Write SRT content to file.

        Args:
            srt_content: SRT formatted string
            output_path: Path to write

        Returns:
            Number of subtitles written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(srt_content, encoding="utf-8")

        # Count subtitles
        count = srt_content.count("\n\n")
        return count

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through HuggingFace Transformers pipeline.

        Steps:
        1. Extract audio (if video)
        2. Optionally detect scenes
        3. Transcribe (per scene or full file)
        4. Generate SRT from segments
        5. Stitch (if scenes)
        6. Post-process
        7. Return metadata

        Args:
            media_info: Media information dict

        Returns:
            Master metadata dict
        """
        start_time = time.time()

        input_file = media_info['path']
        media_basename = media_info['basename']

        # Create master metadata
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        master_metadata["config"]["hf_config"] = self.hf_config
        master_metadata["config"]["scene_detection"] = self.scene_method

        try:
            # Step 1: Extract audio
            self.progress.set_current_step("Extracting audio", 1, 5)
            logger.info("Step 1/5: Extracting audio...")

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration
            )

            # Step 2: Optional scene detection
            scene_paths = None
            if self.scene_method != "none":
                self.progress.set_current_step(f"Detecting scenes ({self.scene_method})", 2, 5)
                logger.info(f"Step 2/5: Detecting scenes ({self.scene_method})...")

                scenes_dir = self.temp_dir / "scenes"
                scenes_dir.mkdir(exist_ok=True)
                scene_paths = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)

                if not scene_paths:
                    logger.warning("Scene detection produced zero scenes. Processing full audio.")
                    scene_paths = None
                else:
                    master_metadata["scenes_detected"] = []
                    for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                        master_metadata["scenes_detected"].append({
                            "scene_index": idx,
                            "filename": scene_path.name,
                            "start_time_seconds": round(start_sec, 3),
                            "end_time_seconds": round(end_sec, 3),
                            "duration_seconds": round(dur_sec, 3),
                            "path": str(scene_path)
                        })
                    master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "completed",
                    scene_count=len(scene_paths) if scene_paths else 0,
                    method=self.scene_method
                )
            else:
                self.progress.set_current_step("Skipping scene detection", 2, 5)
                logger.info("Step 2/5: Skipping scene detection (disabled)")
                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "skipped"
                )

            # Step 2.5: Speech enhancement (if enabled)
            if self.speech_enhancer:
                enhancer_name = self.speech_enhancer.name
                logger.info(f"Step 2.5: Enhancing audio with {self.speech_enhancer.display_name}...")

                if scene_paths:
                    # Enhance each scene
                    scene_paths = enhance_scenes(
                        scene_paths,
                        self.speech_enhancer,
                        self.temp_dir,
                        progress_callback=lambda n, t, name: logger.debug(
                            f"Enhancing scene {n}/{t}: {name}"
                        )
                    )
                    master_metadata["config"]["speech_enhancement"] = {
                        "backend": enhancer_name,
                        "enhanced_scenes": len(scene_paths)
                    }
                else:
                    # Enhance full audio file
                    enhanced_path = self.temp_dir / f"{media_basename}_enhanced.wav"
                    extracted_audio = enhance_single_audio(
                        extracted_audio,
                        self.speech_enhancer,
                        output_path=enhanced_path
                    )
                    master_metadata["config"]["speech_enhancement"] = {
                        "backend": enhancer_name,
                        "enhanced_full_audio": True
                    }

                # Free GPU memory before ASR
                self.speech_enhancer.cleanup()
                self.speech_enhancer = None
                logger.info(f"Speech enhancement complete, GPU memory released")

                self.metadata_manager.update_processing_stage(
                    master_metadata, "speech_enhancement", "completed",
                    backend=enhancer_name
                )
            else:
                self.metadata_manager.update_processing_stage(
                    master_metadata, "speech_enhancement", "skipped"
                )

            # Step 3: Transcription
            self.progress.set_current_step("Transcribing with HF Transformers", 3, 5)
            logger.info("Step 3/5: Transcribing with HuggingFace Transformers...")

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)

            if scene_paths:
                # Multi-scene transcription
                scene_srt_info = []
                total_scenes = len(scene_paths)

                print(f"\nTranscribing {total_scenes} scenes with HF Transformers:")
                transcription_start = time.time()

                for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                    scene_num = idx + 1

                    # Progress display
                    progress_pct = (scene_num / total_scenes) * 100
                    bar_width = 30
                    filled = int(bar_width * scene_num / total_scenes)
                    bar = '=' * filled + '-' * (bar_width - filled)

                    # ETA calculation
                    eta_text = ""
                    if scene_num > 2:
                        elapsed = time.time() - transcription_start
                        avg_per_scene = elapsed / scene_num
                        remaining = (total_scenes - scene_num) * avg_per_scene
                        if remaining > 60:
                            eta_text = f" | ETA: {remaining/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {remaining:.0f}s"

                    scene_name = scene_path.name[:25] + "..." if len(scene_path.name) > 25 else scene_path.name
                    print(f"\rTranscribing: [{bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_name}{eta_text}", end='', flush=True)

                    try:
                        # Transcribe scene
                        segments = self.asr.transcribe(scene_path)

                        if segments:
                            # Convert to SRT (timestamps relative to scene start)
                            srt_content = self._segments_to_srt(segments)
                            scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                            self._write_srt(srt_content, scene_srt_path)

                            scene_srt_info.append((scene_srt_path, start_sec))
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                            master_metadata["scenes_detected"][idx]["segment_count"] = len(segments)
                        else:
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["no_speech_detected"] = True

                    except Exception as e:
                        logger.error(f"Scene {scene_num} transcription failed: {e}")
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                        master_metadata["scenes_detected"][idx]["error"] = str(e)

                print(f"\n[DONE] Completed transcription of {total_scenes} scenes")

                # Step 4: Stitch scene SRTs
                self.progress.set_current_step("Stitching scene transcriptions", 4, 5)
                logger.info("Step 4/5: Stitching scene transcriptions...")

                stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "transcription", "completed",
                    scenes_transcribed=len(scene_srt_info)
                )
                self.metadata_manager.update_processing_stage(
                    master_metadata, "stitching", "completed",
                    subtitle_count=num_subtitles
                )

            else:
                # Full-file transcription (no scenes)
                logger.info("Transcribing full audio file...")

                segments = self.asr.transcribe(extracted_audio)

                # Convert to SRT
                srt_content = self._segments_to_srt(segments)
                stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                num_subtitles = self._write_srt(srt_content, stitched_srt_path)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "transcription", "completed",
                    segment_count=len(segments)
                )
                self.metadata_manager.update_processing_stage(
                    master_metadata, "stitching", "skipped"
                )

                print(f"[DONE] Transcription complete: {len(segments)} segments")

            # Step 5: Post-process
            self.progress.set_current_step("Post-processing subtitles", 5, 5)
            logger.info("Step 5/5: Post-processing subtitles...")

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)

            # Ensure final SRT is in output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)

            # Move raw_subs folder to output directory
            temp_raw_subs = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs.exists():
                final_raw_subs = self.output_dir / "raw_subs"
                final_raw_subs.mkdir(exist_ok=True)
                for file in temp_raw_subs.glob(f"{media_basename}*"):
                    shutil.copy2(file, final_raw_subs / file.name)

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

            logger.info(f"Output saved to: {final_srt_path}")
            logger.info(f"Total processing time: {total_time:.1f}s")

            return master_metadata

        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise

    def cleanup(self) -> None:
        """Clean up resources including speech enhancer and ASR model.

        NOTE: In subprocess workers (ensemble mode), all cleanup is delegated
        to super().cleanup() which will skip operations - OS handles resource
        reclamation on process exit.
        """
        import os

        # In subprocess, skip directly to parent which will also skip.
        # This avoids CUDA crashes during process termination.
        if os.environ.get('WHISPERJAV_SUBPROCESS_WORKER') == '1':
            super().cleanup()
            return

        if hasattr(self, 'speech_enhancer') and self.speech_enhancer:
            self.speech_enhancer.cleanup()
            self.speech_enhancer = None
        if hasattr(self, 'asr') and self.asr:
            self.asr.cleanup()
        super().cleanup()
