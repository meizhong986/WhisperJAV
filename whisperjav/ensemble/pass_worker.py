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


def run_pass_worker(payload: WorkerPayload) -> Dict[str, Any]:
    """Entry point executed in a separate process for a single pass."""
    # Mark this as a subprocess worker - cleanup routines can check this
    # to skip risky CUDA operations that can crash on Windows
    os.environ['WHISPERJAV_SUBPROCESS_WORKER'] = '1'

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
                    "[Worker %s] Pass %s: SUCCESS - %s → %d subtitles in %.1fs",
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

    logger.info("[Worker %s] Worker exiting", os.getpid())

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
        # Apply GUI-specified overrides for transformers pipeline
        if pass_config.get("model"):
            hf_defaults["hf_model_id"] = pass_config["model"]
            logger.debug("Pass %s: Override hf_model_id = %s", pass_number, pass_config["model"])
        if pass_config.get("scene_detector") and pass_config["scene_detector"] != "none":
            hf_defaults["hf_scene"] = pass_config["scene_detector"]
            logger.debug("Pass %s: Override hf_scene = %s", pass_number, pass_config["scene_detector"])
        elif pass_config.get("scene_detector") == "none":
            hf_defaults["hf_scene"] = "none"
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

        # 3. Route based on config structure
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


# Map GUI speech segmenter values to factory-compatible backend names
# IMPORTANT: The factory extracts version/variant from the name itself (e.g., "silero-v3.1" → version="v3.1")
# DO NOT strip version/variant suffixes - pass them through so the factory can process them correctly
# This map is primarily for:
#   1. Handling empty string default
#   2. Normalizing legacy/alias names
#   3. Validation via .get() fallback
SPEECH_SEGMENTER_MAP = {
    "": "silero",           # Empty string → default silero (factory defaults to v4.0)
    "silero": "silero",     # Base silero → factory defaults to v4.0
    "silero-v4.0": "silero-v4.0",   # Preserve version for factory extraction
    "silero-v3.1": "silero-v3.1",   # Preserve version for factory extraction
    "nemo": "nemo-lite",    # Base nemo → default to nemo-lite variant
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
            resolved_config["features"]["scene_detection"] = None
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
