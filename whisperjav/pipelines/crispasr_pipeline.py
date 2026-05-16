#!/usr/bin/env python3
"""
CrispASR Pipeline for WhisperJAV.

A simple, standalone, first-class pipeline that subprocess-calls the complete
``crispasr`` external provider — exactly the way SubtitleEdit cleanly shells
out to it.  It is a peer of ``balanced``/``qwen``/``transformers`` with **zero
relationship to XXL/BYOP**.  It refactors no existing code and is usable both
standalone (``--mode crispasr``) and in either ensemble pass
(``--pass1/2-pipeline crispasr``).

CrispASR is a *complete* external provider: it owns its own VAD / enhancement /
alignment internally.  WhisperJAV only extracts a 16 kHz mono WAV, feeds it in,
and consumes the SRT it writes.  No WhisperJAV scene detection, speech
segmentation, speech enhancement, or SRT post-processing is applied — coupling
those to CrispASR internals is the over-engineering trap that was explicitly
killed (doc 08 §3a).

Design:  docs/plans/crispasr_v190/08_crispasr_simple_pipeline_design.md
Durable record:  memory/feedback_crispasr_simple_standalone_pipeline.md
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import srt

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.pipelines.crispasr_runner import run_crispasr, DEFAULT_BACKEND
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.srt_postprocessing import normalize_language_code
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress


class CrispASRPipeline(BasePipeline):
    """Standalone pipeline that delegates transcription to the crispasr binary."""

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        save_metadata_json: bool = False,
        progress_display=None,
        crispasr_exe: Optional[str] = None,
        crispasr_backend: str = DEFAULT_BACKEND,
        crispasr_args: str = "",
        crispasr_language: str = "ja",
        subs_language: str = "native",
        **kwargs,
    ):
        """
        Args:
            output_dir: Output directory for the final SRT.
            temp_dir: Temporary directory for audio extraction / engine output.
            keep_temp_files: Whether to keep temporary files.
            save_metadata_json: Preserve metadata JSON (enabled by --debug).
            progress_display: Progress display object.
            crispasr_exe: Path to the crispasr external executable.
            crispasr_backend: Curated CrispASR backend (see crispasr_runner).
            crispasr_args: Extra args passed verbatim to crispasr (shlex-split).
            crispasr_language: Source language code (e.g. 'ja').
            subs_language: 'native' or 'direct-to-english'.
            **kwargs: Forwarded to BasePipeline (ignores unknown extras).
        """
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            save_metadata_json=save_metadata_json,
            **kwargs,
        )

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language
        self.crispasr_exe = crispasr_exe
        self.crispasr_backend = crispasr_backend or DEFAULT_BACKEND
        self.crispasr_args = crispasr_args or ""
        self.crispasr_language = crispasr_language or "ja"

        # Output language code for the filename (mirrors the other pipelines).
        if subs_language == "direct-to-english":
            self.lang_code = "en"
        else:
            self.lang_code = normalize_language_code(self.crispasr_language)

        # CrispASR is a self-contained external provider: WhisperJAV extracts a
        # 16 kHz mono WAV and feeds it in; crispasr owns its own VAD internally.
        self.audio_extractor = AudioExtractor(sample_rate=16000)

        logger.info("CrispASRPipeline initialized")
        logger.info(f"  Backend: {self.crispasr_backend}")
        logger.info(f"  Executable: {self.crispasr_exe}")
        logger.info(f"  Language: {self.lang_code}")

    def get_mode_name(self) -> str:
        return "crispasr"

    def process(self, media_info: Dict) -> Dict:
        """Extract audio, transcribe via crispasr, place the SRT at the standard path."""
        start_time = time.time()

        input_file = media_info["path"]
        media_basename = media_info["basename"]

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info,
        )
        master_metadata["config"]["crispasr"] = {
            "backend": self.crispasr_backend,
            "executable": str(self.crispasr_exe),
            "extra_args": self.crispasr_args,
            "language": self.lang_code,
        }

        crispasr_out_dir = self.temp_dir / "crispasr_out"

        try:
            if not self.crispasr_exe:
                raise ValueError(
                    "CrispASR executable path is not set. Pass --crispasr-exe "
                    "(or set the CrispASR executable in the GUI)."
                )

            # Step 1: Extract audio to a 16 kHz mono WAV.
            self.progress.set_current_step("Extracting audio", 1, 3)
            logger.info("Step 1/3: Extracting audio...")
            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(
                input_file, audio_path
            )
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration,
            )

            # Step 2: Transcribe via the crispasr external provider.
            self.progress.set_current_step(
                f"Transcribing with CrispASR ({self.crispasr_backend})", 2, 3
            )
            logger.info(
                f"Step 2/3: Transcribing with CrispASR ({self.crispasr_backend})..."
            )
            crispasr_srt = run_crispasr(
                input_file=str(extracted_audio),
                exe_path=str(self.crispasr_exe),
                backend=self.crispasr_backend,
                language=self.lang_code,
                output_dir=str(crispasr_out_dir),
                extra_args=self.crispasr_args,
            )
            self.metadata_manager.update_processing_stage(
                master_metadata, "transcription", "completed",
                backend=self.crispasr_backend, srt_path=str(crispasr_srt),
            )

            # Step 3: Place the SRT at WhisperJAV's standard output path.
            # CrispASR is a complete provider — its SRT is consumed as-is
            # (no WhisperJAV sanitizer / regroup), the SubtitleEdit model.
            self.progress.set_current_step("Finalizing subtitles", 3, 3)
            logger.info("Step 3/3: Finalizing subtitles...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            final_srt_path = (
                self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            )
            shutil.copy2(crispasr_srt, final_srt_path)

            # Count subtitles for the result-summary contract that the
            # ensemble worker / batch reporter expect.
            try:
                subs = list(
                    srt.parse(
                        final_srt_path.read_text(encoding="utf-8", errors="replace")
                    )
                )
                num_subtitles = len(subs)
            except Exception as parse_err:
                logger.warning(
                    f"Could not parse produced SRT to count subtitles: {parse_err}"
                )
                num_subtitles = 0

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                output_path=str(final_srt_path), subtitle_count=num_subtitles,
            )

            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = num_subtitles
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(
                total_time, 2
            )
            master_metadata["metadata_master"]["updated_at"] = (
                datetime.now().isoformat() + "Z"
            )

            self.metadata_manager.save_master_metadata(master_metadata, media_basename)

            # Tidy the engine output dir (base cleanup_temp_files does not
            # know about it); always remove unless --keep-temp.
            if not self.keep_temp_files and crispasr_out_dir.exists():
                shutil.rmtree(crispasr_out_dir, ignore_errors=True)
            self.cleanup_temp_files(media_basename)

            logger.info(f"Output saved to: {final_srt_path}")
            logger.info(
                f"Total processing time: {total_time:.1f}s ({num_subtitles} subtitles)"
            )
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
        """No GPU models are held; delegate to BasePipeline (temp / CUDA noop)."""
        super().cleanup()
