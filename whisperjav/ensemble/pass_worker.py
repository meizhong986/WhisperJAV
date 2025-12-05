"""Subprocess helpers for ensemble pass execution."""
from __future__ import annotations

import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperjav.config.legacy import resolve_legacy_pipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.pipelines.kotoba_faster_whisper_pipeline import (
    KotobaFasterWhisperPipeline,
)
from whisperjav.pipelines.transformers_pipeline import TransformersPipeline
from whisperjav.utils.logger import logger

from .utils import resolve_language_code

PIPELINE_CLASSES = {
    "balanced": BalancedPipeline,
    "fast": FastPipeline,
    "faster": FasterPipeline,
    "fidelity": FidelityPipeline,
    "kotoba-faster-whisper": KotobaFasterWhisperPipeline,
    "transformers": TransformersPipeline,
}

DEFAULT_HF_PARAMS = {
    "hf_model_id": "kotoba-tech/kotoba-whisper-v2.2",
    "hf_chunk_length": 15,
    "hf_stride": None,
    "hf_batch_size": 16,
    "hf_scene": "none",
    "hf_beam_size": 5,
    "hf_temperature": 0.0,
    "hf_attn": "sdpa",
    "hf_timestamps": "segment",
    "hf_language": "ja",
    "hf_task": "transcribe",
    "hf_device": "auto",
    "hf_dtype": "auto",
}


@dataclass
class WorkerPayload:
    pass_number: int
    media_files: List[Dict[str, Any]]
    pass_config: Dict[str, Any]
    output_dir: str
    temp_dir: str
    keep_temp_files: bool
    subs_language: str
    extra_kwargs: Dict[str, Any]
    language_code: str


@dataclass
class FileResult:
    basename: str
    status: str
    srt_path: Optional[str] = None
    subtitles: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


def prepare_transformers_params(pass_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return hf_* parameters for the Transformers pipeline with overrides applied."""

    params = DEFAULT_HF_PARAMS.copy()

    hf_params = pass_config.get("hf_params") or {}
    mapping = {
        "model_id": "hf_model_id",
        "chunk_length_s": "hf_chunk_length",
        "stride_length_s": "hf_stride",
        "batch_size": "hf_batch_size",
        "scene": "hf_scene",
        "beam_size": "hf_beam_size",
        "temperature": "hf_temperature",
        "attn_implementation": "hf_attn",
        "timestamps": "hf_timestamps",
        "language": "hf_language",
        "task": "hf_task",
        "device": "hf_device",
        "dtype": "hf_dtype",
    }

    for key, value in hf_params.items():
        if key in mapping:
            params[mapping[key]] = value
        elif key.startswith("hf_"):
            params[key] = value

    overrides = pass_config.get("overrides") or {}
    override_mapping = {
        "language": "hf_language",
        "device": "hf_device",
        "dtype": "hf_dtype",
        "task": "hf_task",
    }
    for key, hf_key in override_mapping.items():
        if key in overrides:
            params[hf_key] = overrides[key]

    return params


def run_pass_worker(payload: WorkerPayload) -> Dict[str, Any]:
    """Entry point executed in a separate process for a single pass."""
    results: List[FileResult] = []
    pass_config = payload.pass_config
    pass_number = payload.pass_number
    media_files = payload.media_files
    language_code = payload.language_code or resolve_language_code(pass_config, payload.subs_language)

    logger.info(
        "[Worker %s] Starting pass %s for %s file(s)",
        os.getpid(),
        pass_number,
        len(media_files),
        extra={'color': 'blue'},
    )

    if not media_files:
        return {"results": [], "worker_error": None}

    pass_temp_dir = Path(payload.temp_dir) / f"pass{pass_number}_worker"

    try:
        pipeline = _build_pipeline(
            pass_config=pass_config,
            pass_number=pass_number,
            output_dir=payload.output_dir,
            keep_temp_files=payload.keep_temp_files,
            subs_language=payload.subs_language,
            extra_kwargs=payload.extra_kwargs,
            pass_temp_dir=pass_temp_dir,
        )
    except Exception:  # pragma: no cover - fatal config issues propagated
        logger.exception("[Worker %s] Failed to initialize pipeline", os.getpid())
        if not payload.keep_temp_files and pass_temp_dir.exists():
            shutil.rmtree(pass_temp_dir, ignore_errors=True)
        return {
            "results": [
                FileResult(
                    basename=info["basename"],
                    status="failed",
                    error="Pipeline initialization failure",
                ).__dict__
                for info in media_files
            ],
            "worker_error": traceback.format_exc(),
        }

    try:
        for media_info in media_files:
            basename = media_info["basename"]
            logger.info(
                "[Worker %s] Pass %s processing %s",
                os.getpid(),
                pass_number,
                basename,
                extra={'color': 'blue'},
            )
            try:
                result = pipeline.process({
                    **media_info,
                    "path": Path(media_info["path"]),
                })
                final_srt = Path(result["output_files"]["final_srt"])
                pass_output = Path(payload.output_dir) / f"{basename}.{language_code}.pass{pass_number}.srt"
                if final_srt.exists():
                    shutil.copy2(final_srt, pass_output)
                    if not payload.keep_temp_files:
                        try:
                            same_file = final_srt.samefile(pass_output)
                        except FileNotFoundError:
                            same_file = False
                        except OSError:
                            same_file = False

                        if not same_file:
                            try:
                                final_srt.unlink(missing_ok=True)
                            except OSError:
                                logger.debug(
                                    "Pass %s: Unable to remove legacy output %s",
                                    pass_number,
                                    final_srt,
                                )
                results.append(
                    FileResult(
                        basename=basename,
                        status="completed",
                        srt_path=str(pass_output),
                        subtitles=result["summary"].get("final_subtitles_refined", 0),
                        processing_time=result["summary"].get("total_processing_time_seconds", 0.0),
                    )
                )
            except Exception:
                logger.exception(
                    "[Worker %s] Pass %s failed for %s", os.getpid(), pass_number, basename
                )
                results.append(
                    FileResult(
                        basename=basename,
                        status="failed",
                        error=traceback.format_exc(),
                    )
                )
    finally:
        try:
            pipeline.cleanup()
        except Exception:
            logger.warning("[Worker %s] Pipeline cleanup reported errors", os.getpid(), exc_info=True)
        if not payload.keep_temp_files and pass_temp_dir.exists():
            shutil.rmtree(pass_temp_dir, ignore_errors=True)

    return {"results": [r.__dict__ for r in results], "worker_error": None}


def _build_pipeline(
    pass_config: Dict[str, Any],
    pass_number: int,
    output_dir: str,
    keep_temp_files: bool,
    subs_language: str,
    extra_kwargs: Dict[str, Any],
    pass_temp_dir: Path,
):
    pipeline_name = pass_config["pipeline"]
    pipeline_class = PIPELINE_CLASSES.get(pipeline_name)
    if not pipeline_class:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    pass_temp_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_name == "transformers":
        hf_defaults = prepare_transformers_params(pass_config)
        return TransformersPipeline(
            output_dir=output_dir,
            temp_dir=str(pass_temp_dir),
            keep_temp_files=keep_temp_files,
            progress_display=None,
            subs_language=subs_language,
            **hf_defaults,
        )

    resolved_config = resolve_legacy_pipeline(
        pipeline_name=pipeline_name,
        sensitivity=pass_config.get("sensitivity", "balanced"),
        task="transcribe",
        overrides=pass_config.get("overrides"),
    )

    if pass_config.get("params"):
        unknown_params = apply_custom_params(resolved_config, pass_config["params"], pass_number)
        if unknown_params:
            logger.warning(
                "Pass %s: Unrecognized custom parameters forwarded to provider: %s",
                pass_number,
                ", ".join(sorted(unknown_params)),
            )

    pipeline = pipeline_class(
        output_dir=output_dir,
        temp_dir=str(pass_temp_dir),
        keep_temp_files=keep_temp_files,
        subs_language=subs_language,
        resolved_config=resolved_config,
        progress_display=None,
        **extra_kwargs,
    )
    return pipeline


def apply_custom_params(
    resolved_config: Dict[str, Any],
    custom_params: Dict[str, Any],
    pass_number: int,
) -> List[str]:
    model_config = resolved_config["model"]
    params = resolved_config["params"]

    is_v3_config = "asr" in params
    vad_names = {
        "threshold",
        "neg_threshold",
        "min_speech_duration_ms",
        "max_speech_duration_s",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }

    unknown_params: List[str] = []

    if is_v3_config:
        asr_params = params["asr"]
        for key, value in custom_params.items():
            if key == "model_name":
                model_config["model_name"] = value
                logger.debug("Pass %s: Overriding model to %s", pass_number, value)
            elif key == "device":
                model_config["device"] = value
                logger.debug("Pass %s: Overriding device to %s", pass_number, value)
            else:
                asr_params[key] = value
                logger.debug("Pass %s: Set ASR param %s", pass_number, key)
        return unknown_params

    decoder_params = params.get("decoder", {})
    provider_params = params.get("provider", {})
    vad_params = params.get("vad", {})

    for key, value in custom_params.items():
        if key == "model_name":
            model_config["model_name"] = value
            logger.debug("Pass %s: Overriding model to %s", pass_number, value)
        elif key == "device":
            model_config["device"] = value
            logger.debug("Pass %s: Overriding device to %s", pass_number, value)
        elif key in vad_names:
            vad_params[key] = value
        elif key in decoder_params:
            decoder_params[key] = value
        elif key in provider_params:
            provider_params[key] = value
        else:
            provider_params[key] = value
            logger.debug("Pass %s: Added provider param %s", pass_number, key)
            unknown_params.append(key)

    if unknown_params:
        logger.debug("Pass %s: Unknown params forwarded: %s", pass_number, ", ".join(unknown_params))

    return unknown_params
