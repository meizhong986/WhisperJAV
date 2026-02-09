"""Subprocess helpers for ensemble pass execution."""
from __future__ import annotations

import os
import warnings

# Suppress transformers warnings BEFORE any imports that might trigger them
# These must be set early in subprocess workers (spawn context = fresh process)
warnings.filterwarnings("ignore", message=".*chunk_length_s.*is very experimental.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

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
from whisperjav.pipelines.qwen_pipeline import QwenPipeline
from whisperjav.utils.logger import logger, setup_logger
from whisperjav.utils.parameter_tracer import create_tracer, NullTracer

from .utils import resolve_language_code

PIPELINE_CLASSES = {
    "balanced": BalancedPipeline,
    "fast": FastPipeline,
    "faster": FasterPipeline,
    "fidelity": FidelityPipeline,
    "kotoba-faster-whisper": KotobaFasterWhisperPipeline,
    "transformers": TransformersPipeline,
    "qwen": QwenPipeline,  # Dedicated Qwen3-ASR pipeline (ADR-004)
}

DEFAULT_HF_PARAMS = {
    "hf_model_id": "kotoba-tech/kotoba-whisper-bilingual-v1.0",
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

# =============================================================================
# Parameter Category Constants
# These define explicit routing rules for custom parameters
# =============================================================================

# Model-level params (handled separately, always valid)
MODEL_PARAMS = {
    "model_name",
    "device",
}

# Decoder params - common across all Whisper backends
DECODER_PARAMS = {
    "task",
    "language",
    "beam_size",
    "best_of",
    "patience",
    "length_penalty",
    "prefix",
    "suppress_tokens",
    "suppress_blank",
    "without_timestamps",
    "max_initial_timestamp",
}

# VAD params - used by Silero VAD, not passed to Whisper
VAD_PARAMS = {
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "max_speech_duration_s",
    "min_silence_duration_ms",
    "speech_pad_ms",
}

# Provider params - common transcriber options shared by all backends
PROVIDER_PARAMS_COMMON = {
    "temperature",
    "compression_ratio_threshold",
    "logprob_threshold",
    "logprob_margin",
    "no_speech_threshold",
    "drop_nonverbal_vocals",
    "condition_on_previous_text",
    "initial_prompt",
    "word_timestamps",
    "prepend_punctuations",
    "append_punctuations",
    "clip_timestamps",
}

# Provider params specific to faster-whisper backend
PROVIDER_PARAMS_FASTER_WHISPER = {
    "chunk_length",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "prompt_reset_on_temperature",
    "hotwords",
    "multilingual",
    "max_new_tokens",
    "language_detection_threshold",
    "language_detection_segments",
    "log_progress",
    "hallucination_silence_threshold",
}

# Provider params specific to openai-whisper backend
PROVIDER_PARAMS_OPENAI_WHISPER = {
    "verbose",
    "carry_initial_prompt",
    "prompt",
    "fp16",
    "hallucination_silence_threshold",
}

# Provider params specific to stable-ts backend
PROVIDER_PARAMS_STABLE_TS = {
    "stream",
    "mel_first",
    "split_callback",
    "suppress_ts_tokens",
    "gap_padding",
    "only_ffmpeg",
    "max_instant_words",
    "avg_prob_threshold",
    "nonspeech_skip",
    "progress_callback",
    "ignore_compatibility",
    "extra_models",
    "dynamic_heads",
    "nonspeech_error",
    "only_voice_freq",
    "min_word_dur",
    "min_silence_dur",
    "regroup",
    "ts_num",
    "ts_noise",
    "suppress_silence",
    "suppress_word_ts",
    "suppress_attention",
    "use_word_position",
    "q_levels",
    "k_size",
    "time_scale",
    "denoiser",
    "denoiser_options",
    "demucs",
    "demucs_options",
    "vad",
    "vad_threshold",
}

# Feature params - handled at pipeline level, NOT passed to ASR modules
# These should be discarded from custom params with a warning
FEATURE_PARAMS = {
    "scene_detection_method",
    "scene_detection",
    "post_processing",
}

# Scene detection params - routed to features.scene_detection for DynamicSceneDetector
# These match the kwargs expected by DynamicSceneDetector.__init__()
SCENE_DETECTION_PARAMS = {
    # Core options
    "max_duration_s",
    "min_duration_s",
    "target_sr",
    "force_mono",
    "preserve_original_sr",
    "verbose_summary",
    # Pass 1: Coarse segmentation (auditok)
    "pass1_min_duration_s",
    "pass1_max_duration_s",
    "pass1_max_silence_s",
    "pass1_energy_threshold",
    # Pass 2: Fine segmentation (auditok)
    "pass2_min_duration_s",
    "pass2_max_duration_s",
    "pass2_max_silence_s",
    "pass2_energy_threshold",
    # Audio preprocessing
    "bandpass_low_hz",
    "bandpass_high_hz",
    "drc_threshold_db",
    "drc_ratio",
    "drc_attack_ms",
    "drc_release_ms",
    # Fallback options
    "brute_force_fallback",
    "brute_force_chunk_s",
    "pad_edges_s",
    "fade_ms",
    # Silero VAD options (for silero scene detection)
    "silero_threshold",
    "silero_neg_threshold",
    "silero_min_silence_ms",
    "silero_min_speech_ms",
    "silero_max_speech_s",
    "silero_min_silence_at_max",
    "silero_speech_pad_ms",
}

# Map pipeline names to their ASR backends for param validation
PIPELINE_BACKENDS = {
    "balanced": "faster_whisper",
    "fast": "stable_ts",
    "faster": "stable_ts",
    "fidelity": "openai_whisper",
    "kotoba-faster-whisper": "kotoba_faster_whisper",
}


def get_valid_provider_params(pipeline_name: str) -> set:
    """
    Return the set of valid provider params for a given pipeline.

    Args:
        pipeline_name: Name of the pipeline (e.g., 'balanced', 'fidelity')

    Returns:
        Set of valid provider parameter names for the pipeline's backend
    """
    backend = PIPELINE_BACKENDS.get(pipeline_name, "faster_whisper")

    valid = PROVIDER_PARAMS_COMMON.copy()

    if backend in ("faster_whisper", "kotoba_faster_whisper"):
        valid.update(PROVIDER_PARAMS_FASTER_WHISPER)
    elif backend == "openai_whisper":
        valid.update(PROVIDER_PARAMS_OPENAI_WHISPER)
    elif backend == "stable_ts":
        valid.update(PROVIDER_PARAMS_STABLE_TS)

    return valid


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
    log_level: str = "INFO"  # Propagate log level to subprocess workers
    trace_file_path: Optional[str] = None  # Path for parameter tracer output


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

    # Track which hf_* keys were explicitly set by user (via primary mapping)
    user_set_keys = set()

    for key, value in hf_params.items():
        if key in mapping:
            params[mapping[key]] = value
            user_set_keys.add(mapping[key])
        elif key.startswith("hf_"):
            params[key] = value
            user_set_keys.add(key)

    # Handle legacy param name mappings (for backward compatibility)
    # Users might pass legacy names like "scene_detection_method" instead of "scene"
    # Only apply if the standard key wasn't already set by user
    legacy_to_hf = {
        "scene_detection_method": "hf_scene",
    }
    for legacy_key, hf_key in legacy_to_hf.items():
        if legacy_key in hf_params and hf_key not in user_set_keys:
            params[hf_key] = hf_params[legacy_key]
            logger.debug("Mapped legacy param '%s' to '%s'", legacy_key, hf_key)

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


# Default Qwen3-ASR parameters
DEFAULT_QWEN_PARAMS = {
    "qwen_model_id": "Qwen/Qwen3-ASR-1.7B",
    "qwen_device": "auto",
    "qwen_dtype": "auto",
    "qwen_batch_size": 1,
    "qwen_max_tokens": 4096,
    "qwen_language": "Japanese",  # WhisperJAV is purpose-built for Japanese
    "qwen_timestamps": "word",
    "qwen_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "qwen_scene": "semantic",
    "qwen_context": "",
    "qwen_context_file": None,
    "qwen_attn": "auto",
    "qwen_enhancer": "none",
    "qwen_enhancer_model": None,
    "qwen_segmenter": "none",
    "qwen_japanese_postprocess": True,
    "qwen_postprocess_preset": "high_moan",
    "qwen_input_mode": "vad_slicing",
    "qwen_safe_chunking": True,
    "qwen_timestamp_mode": "aligner_vad_fallback",
    "qwen_assembly_cleaner": True,
    "qwen_repetition_penalty": 1.1,
    "qwen_max_tokens_per_second": 20.0,
    "qwen_max_group_duration": 29.0,
}


def prepare_qwen_params(pass_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return qwen_* parameters for the Qwen3-ASR pipeline with overrides applied."""

    params = DEFAULT_QWEN_PARAMS.copy()

    qwen_params = pass_config.get("qwen_params") or {}
    mapping = {
        "model_id": "qwen_model_id",
        "device": "qwen_device",
        "dtype": "qwen_dtype",
        "batch_size": "qwen_batch_size",
        "max_new_tokens": "qwen_max_tokens",
        "language": "qwen_language",
        "timestamps": "qwen_timestamps",
        "aligner_id": "qwen_aligner",
        "use_aligner": "qwen_use_aligner",  # Special handling below
        "scene": "qwen_scene",
        "context": "qwen_context",
        "context_file": "qwen_context_file",
        "attn_implementation": "qwen_attn",
        "japanese_postprocess": "qwen_japanese_postprocess",
        "postprocess_preset": "qwen_postprocess_preset",
        "input_mode": "qwen_input_mode",
        "safe_chunking": "qwen_safe_chunking",
        "timestamp_mode": "qwen_timestamp_mode",
        "assembly_cleaner": "qwen_assembly_cleaner",
        "repetition_penalty": "qwen_repetition_penalty",
        "max_tokens_per_audio_second": "qwen_max_tokens_per_second",
        "max_group_duration": "qwen_max_group_duration",
    }

    # Track which qwen_* keys were explicitly set by user
    user_set_keys = set()

    for key, value in qwen_params.items():
        if key in mapping:
            params[mapping[key]] = value
            user_set_keys.add(mapping[key])
        elif key.startswith("qwen_"):
            params[key] = value
            user_set_keys.add(key)

    # Handle use_aligner special case: if False, set aligner to None
    if qwen_params.get("use_aligner") is False:
        params["qwen_aligner"] = None

    return params


def _write_dropbox_and_exit(result_file: str, result: Dict[str, Any], tracer, exit_code: int) -> None:
    """
    Write result to Drop-Box file and perform Nuclear Exit.

    This helper function:
    1. Flushes the tracer (if active)
    2. Flushes all logging handlers
    3. Writes the result to the Drop-Box file
    4. Calls os._exit() to skip Python's shutdown sequence

    The Nuclear Exit prevents ctranslate2 C++ destructors from running
    during Python interpreter shutdown, which crashes on Windows.

    Args:
        result_file: Path to write the pickled result
        result: Dictionary to pickle (results + worker_error)
        tracer: Parameter tracer to flush
        exit_code: Exit code (0=success, 1=error)
    """
    import pickle
    import logging

    try:
        # 1. Flush tracer (JSONL file handler)
        if tracer is not None:
            try:
                tracer.close()
            except Exception:
                pass  # Non-critical

        # 2. Flush all logging handlers
        # This ensures we see all logs before the process dies
        try:
            logging.shutdown()
        except Exception:
            pass  # Non-critical

        # 3. Write result to Drop-Box
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)

        logger.debug("[Worker %s] Drop-Box written: %s", os.getpid(), result_file)

    except Exception as e:
        # If we can't write the Drop-Box, log and exit with error
        try:
            logger.error("[Worker %s] Failed to write Drop-Box: %s", os.getpid(), e)
            logging.shutdown()
        except Exception:
            pass
        exit_code = 1

    # 4. NUCLEAR EXIT - Skip Python shutdown sequence
    # This prevents ctranslate2 C++ destructor from crashing
    os._exit(exit_code)


def run_pass_worker(payload: WorkerPayload, result_file: str) -> None:
    """
    Entry point executed in a separate process for a single pass.

    IMPORTANT: This function uses the "Drop-Box + Nuclear Exit" pattern.
    - Results are written to `result_file` (pickle format) instead of returned
    - Process exits via os._exit(0) to skip Python's shutdown sequence
    - This prevents ctranslate2 C++ destructor crashes on Windows

    Args:
        payload: WorkerPayload with all processing configuration
        result_file: Path where results will be pickled (the "Drop-Box")
    """
    import pickle
    import logging

    # Mark this as a subprocess worker - cleanup routines can check this
    # to skip risky CUDA operations that can crash on Windows
    os.environ['WHISPERJAV_SUBPROCESS_WORKER'] = '1'

    # Reconfigure logger with the log level from main process
    # (subprocess starts fresh with default INFO level due to 'spawn' context)
    setup_logger("whisperjav", payload.log_level)

    # Create parameter tracer if trace file path is provided
    # Use append=True since the main process already created the file
    # JSONL format allows concurrent writers with line-buffered output
    tracer = create_tracer(payload.trace_file_path, append=True) if payload.trace_file_path else NullTracer()

    # Ensure clean GPU state in spawned worker process.
    # This is defensive - 'spawn' gives us a fresh Python interpreter,
    # but explicit cleanup prevents any edge cases with residual GPU state.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # Non-critical, continue with processing

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
        # Write empty result to Drop-Box and exit
        _write_dropbox_and_exit(result_file, {"results": [], "worker_error": None}, tracer, 0)

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
            tracer=tracer,
        )
    except Exception:  # pragma: no cover - fatal config issues propagated
        logger.exception("[Worker %s] Failed to initialize pipeline", os.getpid())
        if not payload.keep_temp_files and pass_temp_dir.exists():
            shutil.rmtree(pass_temp_dir, ignore_errors=True)
        # Write error to Drop-Box and exit
        error_result = {
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
        _write_dropbox_and_exit(result_file, error_result, tracer, 1)

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
            logger.debug(
                "[Worker %s] Pass %s: File details - path=%s, basename_len=%d",
                os.getpid(), pass_number, media_info.get("path"), len(basename)
            )
            try:
                result = pipeline.process({
                    **media_info,
                    "path": Path(media_info["path"]),
                })

                # Debug log: pipeline result summary
                logger.debug(
                    "[Worker %s] Pass %s: Pipeline returned - output_files=%s, summary=%s",
                    os.getpid(), pass_number,
                    list(result.get("output_files", {}).keys()),
                    {k: v for k, v in result.get("summary", {}).items() if k in [
                        "final_subtitles_refined", "total_scenes_detected", "total_processing_time_seconds"
                    ]}
                )

                # Defensive check: ensure final_srt path exists in result
                final_srt_path = result.get("output_files", {}).get("final_srt")
                if not final_srt_path:
                    raise ValueError("Pipeline did not return final_srt path in output_files")

                final_srt = Path(final_srt_path)
                pass_output = Path(payload.output_dir) / f"{basename}.{language_code}.pass{pass_number}.srt"

                if not final_srt.exists():
                    # BUG FIX: Previously this case silently fell through and marked as "completed"
                    # even though no file was created. Now we properly fail.
                    raise FileNotFoundError(
                        f"Pipeline completed but output file not found: {final_srt}"
                    )

                # Use atomic move instead of copy+delete to prevent orphan files
                # This ensures exactly one output file exists after the operation,
                # regardless of --keep-temp-files flag or filesystem errors.
                # shutil.move() handles cross-filesystem moves internally (copy+delete).
                try:
                    # Remove destination if it exists (from a previous run)
                    if pass_output.exists():
                        pass_output.unlink()
                    shutil.move(str(final_srt), str(pass_output))
                    logger.debug(
                        "Pass %s: Moved %s -> %s",
                        pass_number,
                        final_srt.name,
                        pass_output.name,
                    )
                except OSError as move_err:
                    # Fallback: if move fails (e.g., cross-device, permissions),
                    # copy and attempt delete. This handles edge cases like
                    # network drives or permission-restricted directories.
                    logger.debug(
                        "Pass %s: Move failed (%s), falling back to copy",
                        pass_number,
                        move_err,
                    )
                    shutil.copy2(final_srt, pass_output)
                    try:
                        final_srt.unlink()
                    except OSError:
                        # Log at warning level since this leaves an orphan file
                        logger.warning(
                            "Pass %s: Could not remove %s (orphan file may remain)",
                            pass_number,
                            final_srt.name,
                        )

                # Verify the output file was actually created before marking as completed
                if not pass_output.exists():
                    raise FileNotFoundError(
                        f"File operation succeeded but output file not found: {pass_output}"
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
                logger.debug(
                    "[Worker %s] Pass %s: SUCCESS - %s -> %d subtitles in %.1fs",
                    os.getpid(), pass_number, basename,
                    result["summary"].get("final_subtitles_refined", 0),
                    result["summary"].get("total_processing_time_seconds", 0.0)
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
        # Defensive cleanup: Use BaseException to catch more error types.
        # On Windows, CUDA driver crashes during cleanup can terminate the process
        # before Python exception handling kicks in. Since this is a subprocess that
        # will be terminated after returning, GPU memory is automatically freed by
        # the OS - explicit CUDA cleanup is optional and we shouldn't crash trying.
        try:
            pipeline.cleanup()
        except SystemExit:
            # Re-raise SystemExit to allow clean process termination
            raise
        except BaseException as e:
            # Catch ALL other exceptions including KeyboardInterrupt, MemoryError, etc.
            # Log but don't crash - the process will exit anyway and free resources
            logger.warning(
                "[Worker %s] Pipeline cleanup failed (non-fatal, resources will be freed on exit): %s",
                os.getpid(), e
            )

        # Temp directory cleanup - also make defensive
        if not payload.keep_temp_files and pass_temp_dir.exists():
            try:
                shutil.rmtree(pass_temp_dir, ignore_errors=True)
            except BaseException:
                pass  # Non-critical, OS will clean up if needed

    # Log pass completion summary for user feedback
    completed_count = sum(1 for r in results if r.status == 'completed')
    failed_count = sum(1 for r in results if r.status == 'failed')
    total_subs = sum(r.subtitles for r in results if r.status == 'completed')
    total_time = sum(r.processing_time for r in results if r.status == 'completed')

    if completed_count > 0:
        logger.info(
            "[Worker %s] Pass %s completed: %d file(s), %d subtitles, %.1fs",
            os.getpid(), pass_number, completed_count, total_subs, total_time
        )
    if failed_count > 0:
        logger.warning(
            "[Worker %s] Pass %s had %d failed file(s)",
            os.getpid(), pass_number, failed_count
        )

    logger.info("[Worker %s] Worker exiting via Nuclear Exit", os.getpid())

    # Write result to Drop-Box and Nuclear Exit
    final_result = {"results": [r.__dict__ for r in results], "worker_error": None}
    _write_dropbox_and_exit(result_file, final_result, tracer, 0)


def _build_pipeline(
    pass_config: Dict[str, Any],
    pass_number: int,
    output_dir: str,
    keep_temp_files: bool,
    subs_language: str,
    extra_kwargs: Dict[str, Any],
    pass_temp_dir: Path,
    tracer=None,
):
    pipeline_name = pass_config["pipeline"]
    pipeline_class = PIPELINE_CLASSES.get(pipeline_name)
    if not pipeline_class:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    pass_temp_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic: Log full pass_config for debugging Pass 2 issues
    logger.debug(
        "[Worker %s] Pass %s: Building pipeline '%s' with config keys: %s",
        os.getpid(), pass_number, pipeline_name, list(pass_config.keys())
    )
    logger.debug(
        "[Worker %s] Pass %s: pass_config details - model=%s, sensitivity=%s, scene_detector=%s, speech_segmenter=%s",
        os.getpid(), pass_number,
        pass_config.get("model"),
        pass_config.get("sensitivity"),
        pass_config.get("scene_detector"),
        pass_config.get("speech_segmenter"),
    )

    # Derive ASR task from subs_language
    asr_task = "translate" if subs_language == "direct-to-english" else "transcribe"
    logger.debug(
        "[Worker %s] Pass %s: subs_language=%s -> asr_task=%s",
        os.getpid(), pass_number, subs_language, asr_task
    )

    if pipeline_name == "transformers":
        hf_defaults = prepare_transformers_params(pass_config)
        # Override task based on subs_language
        hf_defaults["hf_task"] = asr_task
        logger.debug(
            "[Worker %s] Pass %s: Transformers defaults BEFORE overrides - hf_model_id=%s, hf_task=%s",
            os.getpid(), pass_number, hf_defaults.get("hf_model_id"), hf_defaults.get("hf_task")
        )
        # Apply GUI-specified overrides for transformers pipeline
        if pass_config.get("model"):
            hf_defaults["hf_model_id"] = pass_config["model"]
            logger.debug("Pass %s: Override hf_model_id = %s", pass_number, pass_config["model"])
        if pass_config.get("scene_detector") and pass_config["scene_detector"] != "none":
            hf_defaults["hf_scene"] = pass_config["scene_detector"]
            logger.debug("Pass %s: Override hf_scene = %s", pass_number, pass_config["scene_detector"])
        elif pass_config.get("scene_detector") == "none":
            hf_defaults["hf_scene"] = "none"
        # Apply source language override for transformers pipeline (fixes Issue #104)
        # The language code (e.g., 'en', 'ja') is set in main.py from --language argument
        if pass_config.get("language"):
            hf_defaults["hf_language"] = pass_config["language"]
            logger.debug("Pass %s: Override hf_language = %s", pass_number, pass_config["language"])
        # Apply speech enhancer override for transformers pipeline
        if pass_config.get("speech_enhancer"):
            enhancer_backend, enhancer_model = _parse_speech_enhancer(pass_config["speech_enhancer"])
            hf_defaults["hf_speech_enhancer"] = enhancer_backend
            if enhancer_model:
                hf_defaults["hf_speech_enhancer_model"] = enhancer_model
            logger.debug("Pass %s: Override hf_speech_enhancer = %s, model = %s", pass_number, enhancer_backend, enhancer_model)
        logger.debug(
            "[Worker %s] Pass %s: Creating TransformersPipeline with hf_model_id=%s, hf_task=%s, hf_language=%s",
            os.getpid(), pass_number,
            hf_defaults.get("hf_model_id"),
            hf_defaults.get("hf_task"),
            hf_defaults.get("hf_language"),
        )
        try:
            pipeline = TransformersPipeline(
                output_dir=output_dir,
                temp_dir=str(pass_temp_dir),
                keep_temp_files=keep_temp_files,
                progress_display=None,
                subs_language=subs_language,
                parameter_tracer=tracer,
                **hf_defaults,
            )
            logger.debug(
                "[Worker %s] Pass %s: TransformersPipeline created successfully",
                os.getpid(), pass_number
            )
            return pipeline
        except Exception as e:
            logger.error(
                "[Worker %s] Pass %s: FAILED to create TransformersPipeline - %s: %s",
                os.getpid(), pass_number, type(e).__name__, e
            )
            raise

    # Qwen3-ASR pipeline (uses TransformersPipeline with asr_backend="qwen")
    if pipeline_name == "qwen":
        qwen_defaults = prepare_qwen_params(pass_config)
        logger.debug(
            "[Worker %s] Pass %s: Qwen defaults BEFORE overrides - qwen_model_id=%s, qwen_scene=%s",
            os.getpid(), pass_number, qwen_defaults.get("qwen_model_id"), qwen_defaults.get("qwen_scene")
        )
        # Apply GUI-specified overrides for qwen pipeline
        if pass_config.get("model"):
            qwen_defaults["qwen_model_id"] = pass_config["model"]
            logger.debug("Pass %s: Override qwen_model_id = %s", pass_number, pass_config["model"])
        if pass_config.get("scene_detector") and pass_config["scene_detector"] != "none":
            qwen_defaults["qwen_scene"] = pass_config["scene_detector"]
            logger.debug("Pass %s: Override qwen_scene = %s", pass_number, pass_config["scene_detector"])
        elif pass_config.get("scene_detector") == "none":
            qwen_defaults["qwen_scene"] = "none"
        # Apply source language override for qwen pipeline
        if pass_config.get("language"):
            qwen_defaults["qwen_language"] = pass_config["language"]
            logger.debug("Pass %s: Override qwen_language = %s", pass_number, pass_config["language"])
        # Apply speech enhancer override for qwen pipeline
        if pass_config.get("speech_enhancer"):
            enhancer_backend, enhancer_model = _parse_speech_enhancer(pass_config["speech_enhancer"])
            qwen_defaults["qwen_enhancer"] = enhancer_backend
            if enhancer_model:
                qwen_defaults["qwen_enhancer_model"] = enhancer_model
            logger.debug("Pass %s: Override qwen_enhancer = %s, model = %s", pass_number, enhancer_backend, enhancer_model)
        # Apply speech segmenter override for qwen pipeline (post-ASR VAD filter)
        if pass_config.get("speech_segmenter"):
            qwen_defaults["qwen_segmenter"] = pass_config["speech_segmenter"]
            logger.debug("Pass %s: Override qwen_segmenter = %s", pass_number, pass_config["speech_segmenter"])
        # Map qwen_* prefixed params to QwenPipeline's parameter names
        qwen_pipeline_params = {
            "model_id": qwen_defaults.get("qwen_model_id", "Qwen/Qwen3-ASR-1.7B"),
            "device": qwen_defaults.get("qwen_device", "auto"),
            "dtype": qwen_defaults.get("qwen_dtype", "auto"),
            "batch_size": qwen_defaults.get("qwen_batch_size", 1),
            "max_new_tokens": qwen_defaults.get("qwen_max_tokens", 4096),
            "language": (lambda _l: None if _l in (None, "null", "") else _l)(qwen_defaults.get("qwen_language", "Japanese")),
            "timestamps": qwen_defaults.get("qwen_timestamps", "word"),
            "aligner_id": qwen_defaults.get("qwen_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
            "scene_detector": qwen_defaults.get("qwen_scene", "none"),
            "context": qwen_defaults.get("qwen_context", ""),
            "context_file": qwen_defaults.get("qwen_context_file", None),
            "attn_implementation": qwen_defaults.get("qwen_attn", "auto"),
            "speech_enhancer": qwen_defaults.get("qwen_enhancer", "none"),
            "speech_enhancer_model": qwen_defaults.get("qwen_enhancer_model", None),
            "speech_segmenter": qwen_defaults.get("qwen_segmenter", "none"),
            "japanese_postprocess": qwen_defaults.get("qwen_japanese_postprocess", True),
            "postprocess_preset": qwen_defaults.get("qwen_postprocess_preset", "high_moan"),
            "qwen_input_mode": qwen_defaults.get("qwen_input_mode", "vad_slicing"),
            "qwen_safe_chunking": qwen_defaults.get("qwen_safe_chunking", True),
            "timestamp_mode": qwen_defaults.get("qwen_timestamp_mode", "aligner_vad_fallback"),
            "assembly_cleaner": qwen_defaults.get("qwen_assembly_cleaner", True),
            "repetition_penalty": qwen_defaults.get("qwen_repetition_penalty", 1.1),
            "max_tokens_per_audio_second": qwen_defaults.get("qwen_max_tokens_per_second", 20.0),
            "segmenter_max_group_duration": qwen_defaults.get("qwen_max_group_duration", 29.0),
        }
        logger.debug(
            "[Worker %s] Pass %s: Creating QwenPipeline with model_id=%s, scene=%s, segmenter=%s",
            os.getpid(), pass_number,
            qwen_pipeline_params["model_id"],
            qwen_pipeline_params["scene_detector"],
            qwen_pipeline_params["speech_segmenter"],
        )
        try:
            pipeline = QwenPipeline(
                output_dir=output_dir,
                temp_dir=str(pass_temp_dir),
                keep_temp_files=keep_temp_files,
                progress_display=None,
                subs_language=subs_language,
                **qwen_pipeline_params,
            )
            logger.debug(
                "[Worker %s] Pass %s: QwenPipeline created successfully",
                os.getpid(), pass_number
            )
            return pipeline
        except Exception as e:
            logger.error(
                "[Worker %s] Pass %s: FAILED to create QwenPipeline - %s: %s",
                os.getpid(), pass_number, type(e).__name__, e
            )
            raise

    resolved_config = resolve_legacy_pipeline(
        pipeline_name=pipeline_name,
        sensitivity=pass_config.get("sensitivity", "balanced"),
        task=asr_task,  # Derived from subs_language above
        overrides=pass_config.get("overrides"),
        device=pass_config.get("device"),  # None = auto-detect
        compute_type=pass_config.get("compute_type"),  # None = auto
    )

    # Apply source language from pass_config (fixes Issue #104)
    # The language code (e.g., 'en', 'ja') is set in main.py from --language argument
    if pass_config.get("language"):
        resolved_config["language"] = pass_config["language"]
        # Also update decoder params so ASR uses correct language
        if "params" in resolved_config and "decoder" in resolved_config["params"]:
            resolved_config["params"]["decoder"]["language"] = pass_config["language"]
        logger.debug("Pass %s: Applied source language = %s", pass_number, pass_config["language"])

    # Apply GUI-specified overrides for legacy pipelines
    _apply_gui_overrides(resolved_config, pass_config, pass_number)

    if pass_config.get("params"):
        unknown_params = apply_custom_params(
            resolved_config=resolved_config,
            custom_params=pass_config["params"],
            pass_number=pass_number,
            pipeline_name=pipeline_name,
        )
        if unknown_params:
            logger.warning(
                "Pass %s: %d unrecognized parameter(s) were ignored: %s",
                pass_number,
                len(unknown_params),
                ", ".join(sorted(unknown_params)),
            )

    pipeline = pipeline_class(
        output_dir=output_dir,
        temp_dir=str(pass_temp_dir),
        keep_temp_files=keep_temp_files,
        subs_language=subs_language,
        resolved_config=resolved_config,
        progress_display=None,
        parameter_tracer=tracer,
        **extra_kwargs,
    )
    return pipeline


def apply_custom_params(
    resolved_config: Dict[str, Any],
    custom_params: Dict[str, Any],
    pass_number: int,
    pipeline_name: str,
) -> List[str]:
    """
    Apply custom parameters to resolved config with explicit category routing.

    This function routes custom parameters to the correct config section based on
    explicit category membership, not key existence. Feature params are discarded
    (not passed to ASR), and unknown params are rejected with warnings.

    Args:
        resolved_config: The resolved configuration from resolve_legacy_pipeline()
        custom_params: User-provided custom parameters
        pass_number: Pass number (1 or 2) for logging
        pipeline_name: Pipeline name for backend-specific param validation

    Returns:
        List of unknown parameter names that were not applied
    """
    model_config = resolved_config["model"]
    params = resolved_config["params"]

    # Determine config structure (V3 vs legacy)
    is_v3_config = "asr" in params

    # Track results
    unknown_params: List[str] = []
    discarded_params: List[str] = []

    # Get valid provider params for this pipeline's backend
    valid_provider_params = get_valid_provider_params(pipeline_name)

    # Ensure features.scene_detection exists for scene param routing
    if "features" not in resolved_config:
        resolved_config["features"] = {}
    if "scene_detection" not in resolved_config["features"]:
        resolved_config["features"]["scene_detection"] = {}
    if resolved_config["features"]["scene_detection"] is None:
        resolved_config["features"]["scene_detection"] = {}

    for key, value in custom_params.items():
        # 1. Model-level params (always handled first)
        if key in MODEL_PARAMS:
            model_config[key] = value
            logger.debug("Pass %s: Set model.%s = %s", pass_number, key, value)
            continue

        # 2. Feature params - discard with info log (not applicable to ASR)
        if key in FEATURE_PARAMS:
            discarded_params.append(key)
            logger.debug(
                "Pass %s: Discarding feature param '%s' (not applicable to ASR)",
                pass_number, key
            )
            continue

        # 3. Scene detection params - route to features.scene_detection
        if key in SCENE_DETECTION_PARAMS:
            resolved_config["features"]["scene_detection"][key] = value
            logger.debug("Pass %s: Set scene_detection.%s = %s", pass_number, key, value)
            continue

        # 4. Route based on config structure
        if is_v3_config:
            # V3 structure: params["asr"] contains all ASR params
            asr_params = params["asr"]
            if key in VAD_PARAMS:
                # VAD params need special handling in V3
                if "vad" not in params:
                    params["vad"] = {}
                params["vad"][key] = value
                logger.debug("Pass %s: Set vad.%s", pass_number, key)
            elif key in DECODER_PARAMS or key in valid_provider_params:
                asr_params[key] = value
                logger.debug("Pass %s: Set asr.%s", pass_number, key)
            else:
                # Unknown param in V3 - track but don't add
                unknown_params.append(key)
                logger.warning(
                    "Pass %s: Unknown param '%s' not applied (not valid for %s)",
                    pass_number, key, pipeline_name
                )
        else:
            # Legacy structure: params has decoder/provider/vad
            decoder_params = params.get("decoder", {})
            provider_params = params.get("provider", {})
            vad_params = params.get("vad", {})

            if key in DECODER_PARAMS:
                decoder_params[key] = value
                logger.debug("Pass %s: Set decoder.%s", pass_number, key)
            elif key in VAD_PARAMS:
                vad_params[key] = value
                logger.debug("Pass %s: Set vad.%s", pass_number, key)
            elif key in valid_provider_params:
                provider_params[key] = value
                logger.debug("Pass %s: Set provider.%s", pass_number, key)
            else:
                # Unknown param - do NOT add to provider, just track it
                unknown_params.append(key)
                logger.warning(
                    "Pass %s: Unknown param '%s' not applied (not valid for %s)",
                    pass_number, key, pipeline_name
                )

    # Log summary of discarded feature params
    if discarded_params:
        logger.info(
            "Pass %s: Discarded %d feature param(s): %s",
            pass_number, len(discarded_params), ", ".join(sorted(discarded_params))
        )

    return unknown_params


# Map GUI speech enhancer values to factory-compatible backend names
# Supports two formats:
#   1. Legacy: "clearvoice" -> uses default model
#   2. New:    "clearvoice:MossFormer2_SE_48K" -> backend + specific model
#   3. FFmpeg DSP: "ffmpeg-dsp:loudnorm,denoise" -> backend + comma-separated effects
SPEECH_ENHANCER_MAP = {
    "": "none",                 # Empty string -> disabled
    "ffmpeg-dsp": "ffmpeg-dsp", # FFmpeg DSP audio filters
    "none": "none",             # Explicit disable
    "zipenhancer": "zipenhancer",       # ZipEnhancer 16kHz (recommended, lightweight)
    "clearvoice": "clearvoice", # Default ClearVoice (uses MossFormer2_SE_48K)
    "bs-roformer": "bs-roformer",       # BS-RoFormer vocal isolation
}


def _parse_speech_enhancer(enhancer_value: str) -> tuple:
    """
    Parse speech enhancer value into (backend, model).

    Supports formats:
        - "none" -> ("none", None)
        - "clearvoice" -> ("clearvoice", None)  # Uses factory default
        - "clearvoice:MossFormer2_SE_48K" -> ("clearvoice", "MossFormer2_SE_48K")
        - "bs-roformer:vocals" -> ("bs-roformer", "vocals")

    Returns:
        Tuple of (backend_name, model_name or None)
    """
    if not enhancer_value or enhancer_value == "none":
        return ("none", None)

    if ":" in enhancer_value:
        parts = enhancer_value.split(":", 1)
        backend = parts[0]
        model = parts[1] if len(parts) > 1 else None
        return (backend, model)

    # Legacy format without model
    backend = SPEECH_ENHANCER_MAP.get(enhancer_value, enhancer_value)
    return (backend, None)

# Map GUI speech segmenter values to factory-compatible backend names
# IMPORTANT: The factory extracts version/variant from the name itself (e.g., "silero-v3.1" -> version="v3.1")
# DO NOT strip version/variant suffixes - pass them through so the factory can process them correctly
# This map is primarily for:
#   1. Handling empty string default
#   2. Normalizing legacy/alias names
#   3. Validation via .get() fallback
SPEECH_SEGMENTER_MAP = {
    "": "silero",           # Empty string -> default silero (factory defaults to v4.0)
    "silero": "silero",     # Base silero -> factory defaults to v4.0
    "silero-v4.0": "silero-v4.0",   # Preserve version for factory extraction
    "silero-v3.1": "silero-v3.1",   # Preserve version for factory extraction
    "nemo": "nemo-lite",    # Base nemo -> default to nemo-lite variant
    "nemo-lite": "nemo-lite",       # Preserve variant for factory
    "nemo-diarization": "nemo-diarization",  # Preserve variant for factory
    "whisper-vad": "whisper-vad",   # Preserve for factory variant extraction
    "whisper-vad-tiny": "whisper-vad-tiny",
    "whisper-vad-base": "whisper-vad-base",
    "whisper-vad-small": "whisper-vad-small",
    "whisper-vad-medium": "whisper-vad-medium",
    "ten": "ten",
    "none": "none",
}


def _apply_gui_overrides(
    resolved_config: Dict[str, Any],
    pass_config: Dict[str, Any],
    pass_number: int,
) -> None:
    """
    Apply GUI-specified overrides to resolved config for legacy pipelines.

    This function handles the new GUI fields (model, scene_detector, speech_segmenter)
    that are passed at the top level of pass_config, not in the 'params' dict.

    Args:
        resolved_config: The resolved configuration from resolve_legacy_pipeline()
        pass_config: Pass configuration from GUI/CLI with new override fields
        pass_number: Pass number (1 or 2) for logging
    """
    # Override model name if specified
    if pass_config.get("model"):
        model_name = pass_config["model"]
        if "model" in resolved_config:
            resolved_config["model"]["model_name"] = model_name
            logger.debug("Pass %s: Override model_name = %s", pass_number, model_name)

    # Override scene detection method if specified
    scene_detector = pass_config.get("scene_detector")
    if scene_detector:
        if "features" not in resolved_config:
            resolved_config["features"] = {}

        if scene_detector == "none":
            # Disable scene detection
            # Use empty dict with enabled=False instead of None to avoid TypeError
            # when balanced_pipeline.py does DynamicSceneDetector(**scene_opts)
            resolved_config["features"]["scene_detection"] = {"enabled": False}
            if "workflow" in resolved_config and "features" in resolved_config["workflow"]:
                resolved_config["workflow"]["features"]["scene_detection"] = False
            logger.debug("Pass %s: Disabled scene detection", pass_number)
        else:
            # Set scene detection method (auditok or silero)
            if "scene_detection" not in resolved_config["features"]:
                resolved_config["features"]["scene_detection"] = {}
            if resolved_config["features"]["scene_detection"] is None:
                resolved_config["features"]["scene_detection"] = {}
            resolved_config["features"]["scene_detection"]["method"] = scene_detector
            logger.debug("Pass %s: Override scene_detector = %s", pass_number, scene_detector)

    # Override speech segmenter if specified
    # Note: ASR modules read from params["speech_segmenter"]["backend"], not params["vad"]["backend"]
    speech_segmenter = pass_config.get("speech_segmenter")
    if speech_segmenter is not None:  # Allow empty string for default
        segmenter_backend = SPEECH_SEGMENTER_MAP.get(speech_segmenter, speech_segmenter)

        # Ensure params structure exists
        if "params" not in resolved_config:
            resolved_config["params"] = {}
        if "speech_segmenter" not in resolved_config["params"]:
            resolved_config["params"]["speech_segmenter"] = {}

        if segmenter_backend == "none":
            # Disable speech segmentation
            resolved_config["params"]["speech_segmenter"]["backend"] = "none"
            if "workflow" in resolved_config:
                resolved_config["workflow"]["vad"] = "none"
            logger.debug("Pass %s: Disabled speech segmentation (backend = none)", pass_number)
        else:
            # Set speech segmenter backend
            resolved_config["params"]["speech_segmenter"]["backend"] = segmenter_backend
            if "workflow" in resolved_config:
                resolved_config["workflow"]["vad"] = segmenter_backend
            logger.debug("Pass %s: Override speech_segmenter = %s", pass_number, segmenter_backend)

    # Override speech enhancer if specified
    speech_enhancer = pass_config.get("speech_enhancer")
    if speech_enhancer is not None:  # Allow empty string for default
        enhancer_backend, enhancer_model = _parse_speech_enhancer(speech_enhancer)

        # Ensure params structure exists
        if "params" not in resolved_config:
            resolved_config["params"] = {}
        if "speech_enhancer" not in resolved_config["params"]:
            resolved_config["params"]["speech_enhancer"] = {}

        if enhancer_backend == "none":
            # Disable speech enhancement
            resolved_config["params"]["speech_enhancer"]["backend"] = "none"
            logger.debug("Pass %s: Disabled speech enhancement (backend = none)", pass_number)
        else:
            # Set speech enhancer backend and model
            resolved_config["params"]["speech_enhancer"]["backend"] = enhancer_backend
            if enhancer_model:
                resolved_config["params"]["speech_enhancer"]["model"] = enhancer_model
            logger.debug("Pass %s: Override speech_enhancer = %s, model = %s", pass_number, enhancer_backend, enhancer_model)
