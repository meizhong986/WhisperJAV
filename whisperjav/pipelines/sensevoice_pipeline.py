#!/usr/bin/env python3
"""
SenseVoice Pipeline for WhisperJAV (issue #350).

Drop-in ASR mode built on the FunASR SenseVoiceSmall model. SenseVoice is a
fast non-autoregressive recognizer with strong Japanese support and built-in
audio-event tagging, at ~1GB VRAM.

Flow (mirrors TransformersPipeline, trimmed):
    1. Extract audio (16kHz mono)
    2. Optional scene detection (none / auditok / silero / semantic)
    3. Transcribe each scene with SenseVoiceASR
    4. Build per-scene SRT, stitch with scene offsets
    5. SRT post-processing
    6. Return master metadata

Scene detection supplies subtitle granularity because SenseVoice returns
merged-utterance text without reliable intra-clip timestamps. With scene
detection disabled the whole file becomes a small number of subtitles.

funasr is an optional dependency; it is only imported when the ASR actually
loads (see modules/sensevoice_asr.py).
"""

import shutil
import time
from datetime import timedelta
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import srt

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.sensevoice_asr import SenseVoiceASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor, normalize_language_code
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.modules.speech_enhancement import (
    create_enhancer_direct,
    enhance_scenes,
    enhance_single_audio,
    get_extraction_sample_rate,
    is_passthrough_backend,
)
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress


class SenseVoicePipeline(BasePipeline):
    """SenseVoice (FunASR) ASR pipeline."""

    VALID_SCENE_METHODS = ("none", "auditok", "silero", "semantic")

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        save_metadata_json: bool = False,
        progress_display=None,
        # SenseVoice ASR config
        sv_model_id: str = "iic/SenseVoiceSmall",
        sv_device: str = "auto",
        sv_language: str = "ja",
        sv_use_itn: bool = True,
        sv_vad_model: Optional[str] = "fsmn-vad",
        sv_scene: str = "auditok",
        sv_task: str = "transcribe",
        # Speech enhancement (issue #350) — e.g. bs-roformer vocal isolation.
        # Cleans audio BEFORE SenseVoice sees it (helps noisy/music scenes).
        sv_speech_enhancer: str = "none",
        sv_speech_enhancer_model: Optional[str] = None,
        sv_bsrf_overlap: Optional[int] = None,
        # Customize-modal params (issue #350)
        sv_ban_emo_unk: bool = False,
        sv_merge_vad: bool = True,
        sv_merge_length_s: int = 15,
        sv_vad_max_segment_ms: int = 60000,
        sv_batch_size_s: int = 60,
        # Post-processing controls (issue #350). The shared SubtitleSanitizer is
        # tuned for Whisper's failure modes and over-removes SenseVoice output:
        # SenseVoice emits one line per scene spanning the whole scene duration,
        # so genuine short lines look "abnormally slow" and its short utterances
        # collide with the Whisper hallucination blacklist. Defaults are relaxed
        # for SenseVoice; power users can re-enable each filter.
        sv_remove_hallucinations: bool = False,
        sv_remove_repetitions: bool = True,
        sv_remove_cps_outliers: bool = False,
        # Speaker diarization (funasr 1.3.9+): per-sentence [spkN] labels.
        sv_diarize: bool = False,
        sv_spk_num: Optional[int] = None,  # force N speakers (0/None = auto)
        subs_language: str = "native",
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            save_metadata_json=save_metadata_json,
            **kwargs,
        )

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Output language code
        if subs_language == "direct-to-english":
            # SenseVoice can't translate; honour the requested code but warn.
            logger.warning(
                "SenseVoice does not translate; 'direct-to-english' will still "
                "produce source-language text. Use --mode qwen/transformers for "
                "English output."
            )
            self.lang_code = "en"
        else:
            self.lang_code = normalize_language_code(sv_language or "ja")

        # Diarization runs on the WHOLE file (funasr's own VAD segments it) so
        # cam++ produces globally-consistent speaker ids. Per-scene diarization
        # is meaningless (ids reset every clip) AND crashes funasr on tiny clips,
        # so when diarize is on we force scene detection OFF.
        self.diarize = bool(sv_diarize)

        # Scene detection
        self.scene_method = sv_scene
        if self.scene_method not in self.VALID_SCENE_METHODS:
            raise ValueError(
                f"Invalid scene method: {self.scene_method}. "
                f"Must be one of {self.VALID_SCENE_METHODS}"
            )
        if self.diarize and self.scene_method != "none":
            logger.info(
                "Diarization ON — disabling scene detection; SenseVoice runs "
                "on the full audio so speaker ids stay consistent."
            )
            self.scene_method = "none"

        # Speech enhancement CONFIG (enhancer model created in process()).
        # Extract at 48kHz for a real enhancer, 16kHz for passthrough/none.
        self._enhancer_config = {
            "backend": sv_speech_enhancer or "none",
            "model": sv_speech_enhancer_model,
        }
        # BS-RoFormer overlap knob — only forwarded to the enhancer factory.
        self._bsrf_overlap = sv_bsrf_overlap
        self._enhancer_is_passthrough = is_passthrough_backend(self._enhancer_config["backend"])
        extraction_sr = get_extraction_sample_rate(self._enhancer_config["backend"])
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)

        self.scene_detector = None
        if self.scene_method != "none":
            from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
            self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(
                method=self.scene_method
            )

        # ASR CONFIG (model created in process() as a local var, VRAM-scoped)
        self._asr_config = {
            "model_id": sv_model_id,
            "device": sv_device,
            "language": sv_language,
            "use_itn": sv_use_itn,
            "vad_model": sv_vad_model,
            "task": sv_task,
            "ban_emo_unk": sv_ban_emo_unk,
            "merge_vad": sv_merge_vad,
            "merge_length_s": sv_merge_length_s,
            "vad_max_segment_ms": sv_vad_max_segment_ms,
            "batch_size_s": sv_batch_size_s,
            "diarize": sv_diarize,
            "spk_num": sv_spk_num,
        }
        self.sv_config = dict(self._asr_config)

        self.stitcher = SRTStitcher()
        self.postprocessor = SRTPostProcessor(
            language=self.lang_code,
            remove_hallucinations=sv_remove_hallucinations,
            remove_repetitions=sv_remove_repetitions,
            remove_cps_outliers=sv_remove_cps_outliers,
        )

        logger.info("SenseVoicePipeline initialized")
        logger.info(f"  Model: {sv_model_id}")
        logger.info(f"  Language: {sv_language}")
        logger.info(f"  Scene detection: {self.scene_method}")
        logger.info(f"  Speech enhancer: {self._enhancer_config['backend']}")
        logger.info(
            f"  Post-proc: hallucinations={sv_remove_hallucinations} "
            f"repetitions={sv_remove_repetitions} cps_filter={sv_remove_cps_outliers}"
        )
        logger.info(f"  Diarization: {sv_diarize}"
                    + (f" (forced {sv_spk_num} speakers)" if sv_diarize and sv_spk_num else ""))

    def get_mode_name(self) -> str:
        return "sensevoice"

    # ------------------------------------------------------------------ #
    # SRT helpers (same contract as TransformersPipeline)
    # ------------------------------------------------------------------ #
    def _segments_to_srt(self, segments: List[Dict[str, Any]], offset: float = 0.0) -> str:
        subtitles = []
        for idx, seg in enumerate(segments, 1):
            text = seg.get("text", "").strip()
            if not text:
                continue
            start_sec = seg.get("start", 0.0) + offset
            end_sec = seg.get("end", start_sec + 2.0) + offset
            start_td = timedelta(seconds=start_sec)
            end_td = timedelta(seconds=end_sec)
            if end_td <= start_td:
                end_td = start_td + timedelta(milliseconds=100)
            subtitles.append(srt.Subtitle(index=idx, start=start_td, end=end_td, content=text))
        return srt.compose(subtitles)

    def _write_srt(self, srt_content: str, output_path: Path) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(srt_content, encoding="utf-8")
        return srt_content.count("\n\n")

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #
    def process(self, media_info: Dict) -> Dict:
        import os
        start_time = time.time()

        input_file = media_info["path"]
        media_basename = media_info["basename"]

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info,
        )
        master_metadata["config"]["asr_backend"] = "sensevoice"
        master_metadata["config"]["sensevoice_config"] = self.sv_config
        master_metadata["config"]["scene_detection"] = self.scene_method

        try:
            # Step 1: Extract audio (16kHz mono)
            self.progress.set_current_step("Extracting audio", 1, 5)
            logger.info("Step 1/5: Extracting audio...")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration,
            )

            # Step 2: Optional scene detection
            scene_paths = None
            if self.scene_method != "none":
                self.progress.set_current_step(f"Detecting scenes ({self.scene_method})", 2, 5)
                logger.info(f"Step 2/5: Detecting scenes ({self.scene_method})...")
                scenes_dir = self.temp_dir / "scenes"
                scenes_dir.mkdir(exist_ok=True)
                detection_result = self.scene_detector.detect_scenes(
                    extracted_audio, scenes_dir, media_basename
                )
                scene_paths = detection_result.to_legacy_tuples()
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
                            "path": str(scene_path),
                        })
                    master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "completed",
                    scene_count=len(scene_paths) if scene_paths else 0,
                    method=self.scene_method,
                )
            else:
                self.progress.set_current_step("Skipping scene detection", 2, 5)
                logger.info("Step 2/5: Skipping scene detection (disabled)")
                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "skipped"
                )

            import gc
            try:
                import torch
                _torch_available = torch.cuda.is_available()
            except ImportError:
                _torch_available = False

            # =============================================================
            # Step 2.5: SPEECH ENHANCEMENT (exclusive VRAM block).
            # A real enhancer (e.g. bs-roformer) is loaded, used to clean the
            # scene audio, then DESTROYED before the ASR loads (VRAM sandwich).
            # "none"/passthrough just means scenes are already at 16kHz.
            # =============================================================
            if self._enhancer_is_passthrough:
                logger.info("Speech enhancer is passthrough — skipping enhancement")
                master_metadata["config"]["speech_enhancement"] = {"backend": "none", "skipped": True}
            else:
                _enh_kwargs = {}
                if self._bsrf_overlap is not None:
                    _enh_kwargs["num_overlap"] = self._bsrf_overlap
                enhancer = create_enhancer_direct(
                    backend=self._enhancer_config["backend"],
                    model=self._enhancer_config["model"],
                    **_enh_kwargs,
                )
                enhancer_name = enhancer.name
                logger.info(f"Step 2.5: Preparing audio with {enhancer.display_name}...")
                try:
                    if scene_paths:
                        scene_paths = enhance_scenes(
                            scene_paths, enhancer, self.temp_dir,
                            progress_callback=lambda n, t, name: logger.debug(
                                f"Enhancing scene {n}/{t}: {name}"
                            ),
                        )
                        master_metadata["config"]["speech_enhancement"] = {
                            "backend": enhancer_name, "enhanced_scenes": len(scene_paths),
                        }
                    else:
                        enhanced_path = self.temp_dir / f"{media_basename}_enhanced.wav"
                        extracted_audio = enhance_single_audio(
                            extracted_audio, enhancer, output_path=enhanced_path
                        )
                        master_metadata["config"]["speech_enhancement"] = {
                            "backend": enhancer_name, "enhanced_full_audio": True,
                        }
                finally:
                    # Destroy enhancer to free VRAM before SenseVoice loads.
                    logger.debug("Destroying enhancer to free VRAM before ASR load")
                    try:
                        enhancer.cleanup()
                    except Exception as e:
                        logger.warning(f"Enhancer cleanup failed (non-fatal): {e}")
                    del enhancer
                    gc.collect()
                    if _torch_available:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                logger.info("Audio preparation complete, GPU memory released")
                self.metadata_manager.update_processing_stage(
                    master_metadata, "speech_enhancement", "completed", backend=enhancer_name,
                )

            # Step 3: ASR (VRAM-scoped local variable)

            logger.info("Initializing SenseVoice ASR model (exclusive VRAM block)")
            asr = SenseVoiceASR(**self._asr_config)

            self.progress.set_current_step("Transcribing with SenseVoice", 3, 5)
            logger.info("Step 3/5: Transcribing with SenseVoice...")

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)

            try:
                if scene_paths:
                    scene_srt_info = []
                    total_scenes = len(scene_paths)
                    print(f"\nTranscribing {total_scenes} scenes with SenseVoice:")
                    transcription_start = time.time()

                    for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                        scene_num = idx + 1
                        progress_pct = (scene_num / total_scenes) * 100
                        bar_width = 30
                        filled = int(bar_width * scene_num / total_scenes)
                        bar = "=" * filled + "-" * (bar_width - filled)
                        eta_text = ""
                        if scene_num > 2:
                            elapsed = time.time() - transcription_start
                            avg = elapsed / scene_num
                            remaining = (total_scenes - scene_num) * avg
                            eta_text = (f" | ETA: {remaining/60:.1f}m" if remaining > 60
                                        else f" | ETA: {remaining:.0f}s")
                        scene_name = (scene_path.name[:25] + "..."
                                      if len(scene_path.name) > 25 else scene_path.name)
                        print(f"\rTranscribing: [{bar}] {scene_num}/{total_scenes} "
                              f"[{progress_pct:.1f}%] | {scene_name}{eta_text}", end="", flush=True)

                        try:
                            segments = asr.transcribe(scene_path)
                            if segments:
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

                    self.progress.set_current_step("Stitching scene transcriptions", 4, 5)
                    logger.info("Step 4/5: Stitching scene transcriptions...")
                    stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                    num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
                    self.metadata_manager.update_processing_stage(
                        master_metadata, "transcription", "completed",
                        scenes_transcribed=len(scene_srt_info),
                    )
                    self.metadata_manager.update_processing_stage(
                        master_metadata, "stitching", "completed", subtitle_count=num_subtitles,
                    )
                else:
                    # Full-file transcription (also the diarization path: funasr's
                    # internal VAD chunks the whole file, so there are no scenes
                    # to loop over and report progress from). Give the user some
                    # sign of life since this single call can run for minutes —
                    # funasr's own per-chunk progress bar is re-enabled for
                    # diarization (see SenseVoiceASR), and we time the call.
                    if self.diarize:
                        logger.info(
                            "Transcribing full audio with diarization (one pass; "
                            "funasr VAD segments internally — this can take several "
                            "minutes on a long file)..."
                        )
                        print("\nDiarizing full audio (single pass) — progress below:", flush=True)
                    else:
                        logger.info("Transcribing full audio file...")
                    _t0 = time.time()
                    segments = asr.transcribe(extracted_audio)
                    _elapsed = time.time() - _t0
                    srt_content = self._segments_to_srt(segments)
                    stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                    num_subtitles = self._write_srt(srt_content, stitched_srt_path)
                    self.metadata_manager.update_processing_stage(
                        master_metadata, "transcription", "completed", segment_count=len(segments),
                    )
                    self.metadata_manager.update_processing_stage(
                        master_metadata, "stitching", "skipped",
                    )
                    logger.info(f"Full-audio transcription took {_elapsed:.1f}s")
                    print(f"[DONE] Transcription complete: {len(segments)} segments "
                          f"in {_elapsed:.1f}s")
            finally:
                # Destroy ASR while interpreter is stable (VRAM + safe destructor)
                try:
                    asr.cleanup()
                except Exception as e:
                    logger.warning(f"SenseVoice ASR cleanup failed (non-fatal): {e}")
                del asr
                gc.collect()
                if _torch_available:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            # Step 5: Post-process
            self.progress.set_current_step("Post-processing subtitles", 5, 5)
            logger.info("Step 5/5: Post-processing subtitles...")
            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)

            # Move raw_subs folder to output directory (if produced)
            temp_raw_subs = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs.exists():
                final_raw_subs = self.output_dir / "raw_subs"
                final_raw_subs.mkdir(exist_ok=True)
                for file in temp_raw_subs.glob(f"{media_basename}*"):
                    shutil.copy2(file, final_raw_subs / file.name)

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                statistics=stats, output_path=str(final_srt_path),
            )
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = (
                stats.get("total_subtitles", 0) - stats.get("empty_removed", 0)
            )
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles

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
                master_metadata, "error", "failed", error_message=str(e),
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise

    def cleanup(self) -> None:
        super().cleanup()
