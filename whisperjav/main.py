#!/usr/bin/env python3
"""WhisperJAV Main Entry Point - V3 Enhanced with all improvements."""

# ===========================================================================
# EARLY WARNING SUPPRESSION - Must be before any library imports
# ===========================================================================
# Suppress noisy library warnings that don't affect functionality
import os
import warnings

# TensorFlow/oneDNN warnings - suppress before TF is loaded as side effect
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Suppress specific Python warnings from dependencies
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*is very experimental.*")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*deprecated.*")
# ===========================================================================

import argparse
import sys
from pathlib import Path
import json
import tempfile
from typing import Dict, List, Any
import io
import shutil
import subprocess

# Fix stdout before any imports that might use logging
def fix_stdout():
    """Ensure stdout is available, create a wrapper if needed."""
    if sys.stdout is None or (hasattr(sys.stdout, 'closed') and sys.stdout.closed):
        sys.stdout = io.TextIOWrapper(
            io.BufferedWriter(io.FileIO(1, 'w')), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
    if sys.stderr is None or (hasattr(sys.stderr, 'closed') and sys.stderr.closed):
        sys.stderr = io.TextIOWrapper(
            io.BufferedWriter(io.FileIO(2, 'w')), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )

# Fix stdout before any other imports
fix_stdout()

from whisperjav.utils.logger import setup_logger, logger
from whisperjav.utils.device_detector import get_best_device
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.pipelines.kotoba_faster_whisper_pipeline import KotobaFasterWhisperPipeline
from whisperjav.config.legacy import resolve_legacy_pipeline, resolve_ensemble_config
from whisperjav.__version__ import __version__


from whisperjav.utils.preflight_check import enforce_gpu_requirement, run_preflight_checks
from whisperjav.utils.progress_aggregator import VerbosityLevel, create_progress_handler
from whisperjav.utils.async_processor import AsyncPipelineManager, ProcessingStatus
from whisperjav.utils.parameter_tracer import create_tracer
from whisperjav.config.manager import ConfigManager, quick_update_ui_preference

# Translation service - direct function call instead of subprocess
from whisperjav.translate import translate_with_config, TranslationError, ConfigurationError


# Language code mapping for Whisper
LANGUAGE_CODE_MAP = {
    'japanese': 'ja',
    'korean': 'ko',
    'chinese': 'zh',
    'english': 'en'
}


def build_translation_context(args) -> str:
    """Build extra_context string for translation from CLI arguments."""
    context_parts = []

    if getattr(args, 'translate_title', None):
        context_parts.append(f"Movie title: {args.translate_title}")

    if getattr(args, 'translate_actress', None):
        context_parts.append(f"Actress: {args.translate_actress}")

    if getattr(args, 'translate_plot', None):
        context_parts.append(f"Plot: {args.translate_plot}")

    return '\n'.join(context_parts) if context_parts else None


# --- UNCONDITIONAL CUDA CHECK ---
# This code runs the moment the module is loaded,
# ensuring the check is never bypassed.
# Bypass for help/version/check and accept-cpu-mode
args = sys.argv[1:]
bypass_flags = ['--check', '--help', '-h', '--version', '-v']
accept_cpu = '--accept-cpu-mode' in args
if not any(flag in args for flag in bypass_flags):
    enforce_gpu_requirement(accept_cpu_mode=accept_cpu)
# --- END OF CHECK ---


def safe_print(message: str):
    """Safely print a message, handling closed stdout."""
    try:
        print(message)
    except (ValueError, AttributeError, OSError):
        try:
            sys.stderr.write(message + '\n')
        except:
            pass


def print_banner():
    """Print application banner."""
    banner = f"""
╔═══════════════════════════════════════════════════╗
║          WhisperJAV v{__version__}                        ║
║   Japanese Adult Video Subtitle Generator         ║
║                                                   ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
    safe_print(banner)


def parse_arguments():
    """Parse command line arguments with all enhancements."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV - Generate accurate subtitles for Japanese adult videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core arguments
    parser.add_argument("input", nargs="*", help="Input media file(s), directory, or wildcard pattern.")
    # Note: kotoba-faster-whisper temporarily hidden from user selection (implementation preserved)
    parser.add_argument("--mode", choices=["fidelity", "balanced", "fast", "faster", "transformers"], default="balanced",
                       help="Processing mode (default: balanced)")
    parser.add_argument("--model", default=None,
                       help="Override the default Whisper model (e.g., large-v2, turbo, large). Overrides config default.")
    parser.add_argument("--language", choices=["japanese", "korean", "chinese", "english"],
                       default="japanese",
                       help="Source audio language for transcription (default: japanese)")
    parser.add_argument("--config", default=None, help="Path to a JSON configuration file")
    parser.add_argument("--subs-language", choices=["native", "direct-to-english"],
                       default="native", help="Subtitle output format: 'native' (transcribe in source language) or 'direct-to-english' (translate to English via Whisper)")

    # Ensemble mode arguments (for direct component specification)
    ensemble_group = parser.add_argument_group("Ensemble Mode Options")
    ensemble_group.add_argument("--asr", default=None,
                               help="ASR component name (e.g., faster_whisper, stable_ts, openai_whisper)")
    ensemble_group.add_argument("--vad", default=None,
                               help="VAD component name (e.g., silero, none)")
    ensemble_group.add_argument("--features", default=None,
                               help="Comma-separated feature names (e.g., auditok_scene_detection)")
    ensemble_group.add_argument("--task", choices=["transcribe", "translate"], default=None,
                               help="Task type (overrides --subs-language if provided)")
    ensemble_group.add_argument("--overrides", default=None,
                               help="JSON string of parameter overrides")

    # Two-pass ensemble mode arguments
    twopass_group = parser.add_argument_group("Two-Pass Ensemble Options")
    twopass_group.add_argument("--ensemble", action="store_true",
                               help="Enable two-pass ensemble mode")
    # Note: kotoba-faster-whisper temporarily hidden from user selection (implementation preserved)
    twopass_group.add_argument("--pass1-pipeline", default="balanced",
                               choices=["balanced", "fast", "faster", "fidelity", "transformers"],
                               help="Pipeline for pass 1 (default: balanced)")
    twopass_group.add_argument("--pass1-sensitivity", default="balanced",
                               choices=["conservative", "balanced", "aggressive"],
                               help="Sensitivity for pass 1 (default: balanced)")
    twopass_group.add_argument("--pass1-overrides", default=None,
                               help="JSON string of parameter overrides for pass 1 (deprecated)")
    twopass_group.add_argument("--pass1-params", default=None,
                               help="JSON string of full parameters for pass 1 (custom mode)")
    twopass_group.add_argument("--pass1-hf-params", default=None,
                               help="JSON string of HuggingFace Transformers parameters for pass 1 (when pipeline=transformers)")
    twopass_group.add_argument("--pass1-scene-detector", default=None,
                               choices=["auditok", "silero", "semantic", "none"],
                               help="Scene detection method for pass 1 (default: auditok)")
    twopass_group.add_argument("--pass1-speech-segmenter", default=None,
                               help="Speech segmenter backend for pass 1 (e.g., silero, ten, nemo, whisper-vad, none)")
    twopass_group.add_argument("--pass1-speech-enhancer", default=None,
                               help="Speech enhancer for pass 1 (e.g., none)")
    twopass_group.add_argument("--pass1-model", default=None,
                               help="Model name for pass 1 (e.g., large-v2, kotoba-whisper-v2.0)")
    # Note: kotoba-faster-whisper temporarily hidden from user selection (implementation preserved)
    twopass_group.add_argument("--pass2-pipeline", default=None,
                               choices=["balanced", "fast", "faster", "fidelity", "transformers"],
                               help="Pipeline for pass 2 (enables pass 2)")
    twopass_group.add_argument("--pass2-sensitivity", default="balanced",
                               choices=["conservative", "balanced", "aggressive"],
                               help="Sensitivity for pass 2 (default: balanced)")
    twopass_group.add_argument("--pass2-overrides", default=None,
                               help="JSON string of parameter overrides for pass 2 (deprecated)")
    twopass_group.add_argument("--pass2-params", default=None,
                               help="JSON string of full parameters for pass 2 (custom mode)")
    twopass_group.add_argument("--pass2-hf-params", default=None,
                               help="JSON string of HuggingFace Transformers parameters for pass 2 (when pipeline=transformers)")
    twopass_group.add_argument("--pass2-scene-detector", default=None,
                               choices=["auditok", "silero", "semantic", "none"],
                               help="Scene detection method for pass 2 (default: none)")
    twopass_group.add_argument("--pass2-speech-segmenter", default=None,
                               help="Speech segmenter backend for pass 2 (e.g., silero, ten, nemo, whisper-vad, none)")
    twopass_group.add_argument("--pass2-speech-enhancer", default=None,
                               help="Speech enhancer for pass 2 (e.g., none)")
    twopass_group.add_argument("--pass2-model", default=None,
                               help="Model name for pass 2 (e.g., large-v2, kotoba-whisper-v2.0)")
    twopass_group.add_argument("--merge-strategy", default="pass1_primary",
                               choices=["pass1_primary", "pass2_primary", "smart_merge", "full_merge", "pass1_overlap", "pass2_overlap"],
                               help="Merge strategy for two-pass results (default: pass1_primary)")

    # Environment check
    parser.add_argument("--check", action="store_true", help="Run environment checks and exit")
    parser.add_argument("--check-verbose", action="store_true", help="Run verbose environment checks")
    parser.add_argument("--accept-cpu-mode", action="store_true",
                       help="Accept CPU-only mode without GPU warning (skip GPU performance check)")

    # Hardware configuration (device and compute type override)
    hardware_group = parser.add_argument_group("Hardware Configuration")
    hardware_group.add_argument("--device", type=str, default=None,
                               choices=["auto", "cuda", "cpu"],
                               help="Device to use for ASR processing (default: auto-detect). "
                                    "Note: MPS (Apple Silicon) is auto-detected but CTranslate2 backends "
                                    "will fall back to CPU with Accelerate optimization.")
    hardware_group.add_argument("--compute-type", type=str, default=None,
                               choices=["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"],
                               help="Compute type for ASR processing (default: auto). "
                                    "CTranslate2 backends (faster_whisper, kotoba) use 'auto' to optimize "
                                    "based on GPU capability. PyTorch backends use float16 (GPU) or float32 (CPU).")

    # Path and logging
    path_group = parser.add_argument_group("Path and Logging Options")
    path_group.add_argument("--output-dir", default="./output", help="Output directory")
    path_group.add_argument("--temp-dir", default=None, help="Temporary directory")
    path_group.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    path_group.add_argument("--skip-existing", action="store_true",
                           help="Skip files that already have subtitle output (useful for resuming batch processing)")
    path_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                           default="INFO", help="Logging level")
    path_group.add_argument("--log-file", help="Log file path")
    path_group.add_argument("--stats-file", help="Save processing statistics to JSON")
    path_group.add_argument("--dump-params", metavar="FILE",
                           help="Dump resolved parameters to JSON file and exit (no processing)")
    path_group.add_argument("--trace-params", metavar="FILE",
                           help="Stream parameter snapshots to JSON Lines file during execution (real-time observability)")

    # Progress control
    progress_group = parser.add_argument_group("Progress Display Options")
    progress_group.add_argument("--no-progress", action="store_true",
                               help="Disable progress bars")
    progress_group.add_argument("--verbosity",
                               choices=["quiet", "summary", "normal", "verbose"],
                               default=None,
                               help="Console output verbosity (overrides config)")
    progress_group.add_argument("--debug", action="store_true",
                               help="Enable debug logging and preserve metadata JSON files in temp directory")
    progress_group.add_argument("--crash-trace", action="store_true",
                               help="Enable crash tracing (writes to crash_traces/ for debugging ctranslate2 crashes)")

    # Enhancement features
    enhancement_group = parser.add_argument_group("Optional Enhancement Features")
    enhancement_group.add_argument("--adaptive-classification", action="store_true")
    enhancement_group.add_argument("--adaptive-audio-enhancement", action="store_true")
    enhancement_group.add_argument("--smart-postprocessing", action="store_true")
    
    # Transcription tuning
    tuning_group = parser.add_argument_group("Transcription Tuning")
    tuning_group.add_argument("--sensitivity",
                             choices=["conservative", "balanced", "aggressive"],
                             default="aggressive", help="Transcription sensitivity")
    tuning_group.add_argument("--scene-detection-method",
                             type=str,
                             choices=["auditok", "silero", "semantic"],
                             default=None,  # None = use config default
                             metavar="METHOD",
                             help=(
                                 "Scene detection method: "
                                 "auditok (energy-based, default), "
                                 "silero (VAD-based), or "
                                 "semantic (texture-based clustering)"
                             ))
    tuning_group.add_argument("--no-vad", action="store_true",
                             help="Disable VAD speech segmentation (balanced/fidelity: skip Silero VAD; kotoba: disable faster-whisper VAD)")
    tuning_group.add_argument("--speech-segmenter",
                             type=str,
                             choices=[
                                 "silero", "silero-v4.0", "silero-v3.1",
                                 "nemo", "nemo-lite",
                                 "whisper-vad", "whisper-vad-tiny", "whisper-vad-base", "whisper-vad-medium",
                                 "ten", "none"
                             ],
                             default=None,  # None = use silero (default)
                             metavar="BACKEND",
                             help=(
                                 "Speech segmentation backend: "
                                 "silero/silero-v4.0 (default), silero-v3.1, "
                                 "nemo/nemo-lite (fast frame VAD ~0.5GB), "
                                 "whisper-vad (neural VAD using Whisper small model ~500MB), "
                                 "whisper-vad-tiny/base/medium (other model sizes), "
                                 "ten (TEN Framework), none (disable segmentation)"
                             ))

    # Async processing
    async_group = parser.add_argument_group("Processing Options")
    async_group.add_argument("--async-processing", action="store_true",
                            help="Use async processing (better for GUIs)")
    async_group.add_argument("--max-workers", type=int, default=1,
                            help="Max concurrent workers (default: 1)")
    
    # Subtitle signature options
    signature_group = parser.add_argument_group("Subtitle Attribution")
    signature_group.add_argument("--credit", type=str,
                                help="Producer credit text to add at beginning of subtitles")
    signature_group.add_argument("--no-signature", action="store_true",
                                help="Disable WhisperJAV technical signature at end")

    # Translation options (optional group)
    translation_group = parser.add_argument_group("Translation Options")
    translation_group.add_argument(
        "--translate",
        action="store_true",
        help="Translate subtitles after generation"
    )
    translation_group.add_argument(
        "--translate-provider",
        choices=["deepseek", "openrouter", "gemini", "claude", "gpt", "glm", "groq", "local"],
        default="deepseek",
        help="Translation AI provider (default: deepseek). Use 'local' for offline LLM translation."
    )
    translation_group.add_argument(
        "--translate-target",
        choices=["english", "indonesian", "spanish", "chinese"],
        default="english",
        help="Target language for translation (default: english)"
    )
    translation_group.add_argument(
        "--translate-tone",
        choices=["standard", "pornify"],
        default="standard",
        help="Translation style (default: standard)"
    )
    translation_group.add_argument(
        "--translate-api-key",
        help="Translation API key (overrides environment variable)"
    )
    translation_group.add_argument(
        "--translate-model",
        help="Translation model override. For local provider: llama-8b (default, 6GB VRAM), gemma-9b (best, 8GB VRAM), llama-3b (basic, 3GB VRAM), or 'auto'"
    )
    translation_group.add_argument(
        "--translate-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers for local LLM (-1=all to GPU, 0=CPU only). Only used with --translate-provider local"
    )
    translation_group.add_argument(
        "--translate-quiet",
        action="store_true",
        help="Hide translation progress messages (default: show progress)"
    )
    translation_group.add_argument(
        "--translate-title",
        help="Movie title for translation context"
    )
    translation_group.add_argument(
        "--translate-actress",
        help="Actress name for translation context"
    )
    translation_group.add_argument(
        "--translate-plot",
        help="Plot summary for translation context"
    )

    # HuggingFace Transformers mode arguments
    hf_group = parser.add_argument_group("HuggingFace Transformers Mode Options (--mode transformers)")
    hf_group.add_argument("--hf-model-id", type=str,
                         default="kotoba-tech/kotoba-whisper-bilingual-v1.0",
                         help="HuggingFace model ID (default: kotoba-tech/kotoba-whisper-bilingual-v1.0)")
    hf_group.add_argument("--hf-chunk-length", type=int, default=15,
                         help="Chunk length in seconds (default: 15)")
    hf_group.add_argument("--hf-stride", type=float, default=None,
                         help="Stride/overlap between chunks in seconds (default: chunk_length/6)")
    hf_group.add_argument("--hf-batch-size", type=int, default=8,
                         help="Batch size for parallel chunk processing (default: 8)")
    hf_group.add_argument("--hf-scene", type=str, default="none",
                         choices=["none", "auditok", "silero"],
                         help="Scene detection method (default: none)")
    hf_group.add_argument("--hf-beam-size", type=int, default=5,
                         help="Beam size for beam search (default: 5)")
    hf_group.add_argument("--hf-temperature", type=float, default=0.0,
                         help="Sampling temperature (0=greedy, default: 0.0)")
    hf_group.add_argument("--hf-attn", type=str, default="sdpa",
                         choices=["sdpa", "flash_attention_2", "eager"],
                         help="Attention implementation (default: sdpa)")
    hf_group.add_argument("--hf-timestamps", type=str, default="segment",
                         choices=["segment", "word"],
                         help="Timestamp granularity (default: segment)")
    hf_group.add_argument("--hf-language", type=str, default="ja",
                         help="Language code for transcription (default: ja)")
    hf_group.add_argument("--hf-device", type=str, default="auto",
                         choices=["auto", "cuda", "cpu"],
                         help="Device to use (default: auto)")
    hf_group.add_argument("--hf-dtype", type=str, default="auto",
                         choices=["auto", "float16", "bfloat16", "float32"],
                         help="Data type (default: auto)")

    parser.add_argument("--version", action="version", version=f"WhisperJAV {__version__}")

    return parser.parse_args()


def add_signatures_to_srt(srt_path: str, producer_credit: str = None,
                          add_technical_sig: bool = True,
                          mode: str = "fidelity",
                          sensitivity: str = "balanced",
                          version: str = __version__):
    """Add producer credit and/or technical signature to SRT file.

    Args:
        srt_path: Path to the SRT file to modify
        producer_credit: Optional producer credit text to add at beginning
        add_technical_sig: Whether to add technical signature at end
        mode: Processing mode used (faster/fast/fidelity)
        sensitivity: Sensitivity level used
        version: WhisperJAV version
    """
    import pysrt
    
    try:
        srt_path = Path(srt_path)
        if not srt_path.exists():
            logger.warning(f"SRT file not found for signature addition: {srt_path}")
            return
            
        subs = pysrt.open(str(srt_path), encoding='utf-8')
        if not subs or len(subs) == 0:
            logger.debug(f"Empty or invalid SRT file, skipping signatures: {srt_path}")
            return
        
        # Add producer credit at beginning if provided
        if producer_credit and producer_credit.strip():
            credit_sub = pysrt.SubRipItem(
                index=0,
                start=pysrt.SubRipTime(milliseconds=0),
                end=pysrt.SubRipTime(milliseconds=100),
                text=producer_credit.strip()
            )
            
            # Reindex existing subtitles
            for sub in subs:
                sub.index += 1
            
            # Insert credit at beginning
            subs.insert(0, credit_sub)
            logger.debug(f"Added producer credit: {producer_credit}")
        
        # Add technical signature at end if enabled
        if add_technical_sig:
            last_sub = subs[-1]
            
            # Format signature text (compact format)
            sig_text = f"WhisperJAV {version} | {mode.capitalize()}/{sensitivity.capitalize()}"
            
            # Calculate timing (500ms after last subtitle ends)
            start_ms = last_sub.end.ordinal + 500
            end_ms = start_ms + 2000
            
            tech_sig = pysrt.SubRipItem(
                index=last_sub.index + 1,
                start=pysrt.SubRipTime(milliseconds=start_ms),
                end=pysrt.SubRipTime(milliseconds=end_ms),
                text=sig_text
            )
            
            subs.append(tech_sig)
            logger.debug(f"Added technical signature: {sig_text}")
        
        # Save back to file
        subs.save(str(srt_path), encoding='utf-8')
        logger.debug(f"Signatures added successfully to {srt_path}")
        
    except Exception as e:
        logger.warning(f"Could not add signatures to {srt_path}: {e}")
        # Don't fail the whole process if signatures can't be added


def cleanup_temp_directory(temp_dir: str):
    """Clean up the temporary directory after processing."""
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        return
    
    if temp_path.name == "whisperjav" and temp_path.parent == Path(tempfile.gettempdir()):
        logger.debug(f"Cleaning up WhisperJAV temp directory: {temp_path}")
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
            logger.debug("Temp directory cleaned up successfully")
        except Exception as e:
            logger.error(f"Error removing temp directory: {e}")
    else:
        logger.debug(f"Cleaning up temp directory contents: {temp_path}")
        try:
            subdirs_to_clean = ["scenes", "scene_srts", "raw_subs"]
            for subdir in subdirs_to_clean:
                subdir_path = temp_path / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path, ignore_errors=True)
                    logger.debug(f"Removed temp subdirectory: {subdir_path}")
            
            for file in temp_path.glob("*"):
                if file.is_file():
                    file.unlink()
                    logger.debug(f"Removed temp file: {file}")
                    
            logger.info("Temp directory contents cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")


def aggregate_subtitle_metrics(metadata_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate subtitle/filter metrics across successful files."""
    totals = {
        "files_processed": 0,
        "raw_subtitles": 0,
        "final_subtitles": 0,
        "hallucinations_removed": 0,
        "repetitions_removed": 0,
        "logprob_filtered": 0,
        "cps_filtered": 0,
        "nonverbal_filtered": 0,
        "empty_removed": 0,
        "other_removed": 0,
        "processing_time_seconds": 0.0
    }

    for metadata in metadata_entries:
        if not metadata:
            continue

        summary = metadata.get("summary", {})
        quality_metrics = summary.get("quality_metrics", {})

        totals["files_processed"] += 1
        totals["raw_subtitles"] += int(summary.get("final_subtitles_raw", 0) or 0)
        totals["final_subtitles"] += int(summary.get("final_subtitles_refined", 0) or 0)
        totals["hallucinations_removed"] += int(quality_metrics.get("hallucinations_removed", 0) or 0)
        totals["repetitions_removed"] += int(quality_metrics.get("repetitions_removed", 0) or 0)
        totals["logprob_filtered"] += int(quality_metrics.get("logprob_filtered", 0) or 0)
        totals["cps_filtered"] += int(quality_metrics.get("cps_filtered", 0) or 0)
        totals["nonverbal_filtered"] += int(quality_metrics.get("nonverbal_filtered", 0) or 0)

        totals["processing_time_seconds"] += float(
            summary.get("total_processing_time_seconds", 0.0) or 0.0
        )

        empty_removed = int(quality_metrics.get("empty_removed", 0) or 0)
        totals["empty_removed"] += empty_removed

        other_removed = empty_removed - (
            int(quality_metrics.get("hallucinations_removed", 0) or 0)
            + int(quality_metrics.get("repetitions_removed", 0) or 0)
        )
        if other_removed < 0:
            other_removed = 0
        totals["other_removed"] += other_removed

    return totals


def print_subtitle_metrics(totals: Dict[str, int]):
    """Pretty-print aggregated subtitle/filter metrics."""
    if not totals or totals.get("files_processed", 0) == 0:
        return

    print("\nSubtitle Filter Totals")
    print("-" * 50)
    print(f"Files with metrics : {totals['files_processed']}")
    print(f"Raw subtitles     : {totals['raw_subtitles']}")
    print(f"Final subtitles   : {totals['final_subtitles']}")
    print("Filter removals:")
    print(f"  Logprob threshold : {totals['logprob_filtered']}")
    print(f"  Non-verbal filter : {totals['nonverbal_filtered']}")
    print(f"  CPS limiter       : {totals['cps_filtered']}")
    print(f"  Hallucination cut : {totals['hallucinations_removed']}")
    print(f"  Repetition cull   : {totals['repetitions_removed']}")
    print(f"  Other removals    : {totals['other_removed']}")
    discrepancy = (totals['raw_subtitles'] - totals['final_subtitles']) - (
        totals['logprob_filtered'] + totals['nonverbal_filtered'] + totals['cps_filtered'] +
        totals['hallucinations_removed'] + totals['repetitions_removed'] + totals['other_removed']
    )
    if discrepancy:
        print(f"  Untracked delta    : {discrepancy}")


def process_files_sync(media_files: List[Dict], args: argparse.Namespace, resolved_config: Dict):
    """Process files synchronously with enhanced progress reporting."""
    
    # Import unified progress components at function start
    from whisperjav.utils.unified_progress import UnifiedProgressManager, VerbosityLevel as UnifiedVerbosityLevel
    from whisperjav.utils.progress_adapter import ProgressDisplayAdapter
    
    # Determine verbosity level from args or config
    verbosity_mapping = {
        'quiet': UnifiedVerbosityLevel.QUIET,
        'summary': UnifiedVerbosityLevel.STANDARD,  # Map summary to standard
        'standard': UnifiedVerbosityLevel.STANDARD,
        'normal': UnifiedVerbosityLevel.STANDARD,   # Map normal to standard
        'detailed': UnifiedVerbosityLevel.DETAILED,
        'verbose': UnifiedVerbosityLevel.DEBUG,     # Map verbose to debug
        'debug': UnifiedVerbosityLevel.DEBUG
    }
    
    if args.verbosity:
        verbosity = verbosity_mapping.get(args.verbosity, UnifiedVerbosityLevel.STANDARD)
    else:
        # Get from config manager
        config_manager = ConfigManager(args.config)
        ui_prefs = config_manager.get_ui_preferences()
        verbosity_str = ui_prefs.get('console_verbosity', 'summary')
        verbosity = verbosity_mapping.get(verbosity_str, UnifiedVerbosityLevel.STANDARD)
    
    enhancement_kwargs = {
        "adaptive_classification": args.adaptive_classification,
        "adaptive_audio_enhancement": args.adaptive_audio_enhancement,
        "smart_postprocessing": args.smart_postprocessing
    }

    # Create parameter tracer for real-time observability (if requested)
    tracer = create_tracer(args.trace_params)
    if args.trace_params:
        # Emit initial configuration snapshot
        tracer.emit_config(
            mode=args.mode,
            sensitivity=args.sensitivity,
            resolved_config=resolved_config,
            cli_args={
                "model": args.model,
                "language": getattr(args, 'language', 'japanese'),
                "subs_language": args.subs_language,
                "ensemble": getattr(args, 'ensemble', False),
                "scene_detection_method": getattr(args, 'scene_detection_method', None),
            }
        )
        logger.info(f"Parameter tracing enabled: {args.trace_params}")

    # Initialize unified progress system for better UX and reduced spam
    if args.no_progress:
        from whisperjav.utils.progress_adapter import DummyProgressAdapter
        progress = DummyProgressAdapter()
    else:
        # Create unified manager and adapter
        unified_manager = UnifiedProgressManager(verbosity=verbosity)
        unified_manager.total_files = len(media_files)  # Store for reference
        progress = ProgressDisplayAdapter(unified_manager)
    
    pipeline_args = {
        "output_dir": args.output_dir,
        "temp_dir": args.temp_dir,
        "keep_temp_files": args.keep_temp,
        "save_metadata_json": getattr(args, 'debug', False),  # --debug enables metadata JSON preservation
        "subs_language": args.subs_language,
        "resolved_config": resolved_config,
        "progress_display": progress,
        "parameter_tracer": tracer,
        **enhancement_kwargs
    }

    # Select pipeline
    if args.mode == "faster":
        pipeline = FasterPipeline(**pipeline_args)
    elif args.mode == "fast":
        pipeline = FastPipeline(**pipeline_args)
    elif args.mode == "balanced":
        pipeline = BalancedPipeline(**pipeline_args)
    elif args.mode == "kotoba-faster-whisper":
        # Kotoba Faster-Whisper pipeline with scene detection (always on)
        scene_method = getattr(args, 'scene_detection_method', None) or 'auditok'
        pipeline = KotobaFasterWhisperPipeline(
            scene_method=scene_method,
            **pipeline_args
        )
    elif args.mode == "transformers":
        # HuggingFace Transformers pipeline with dedicated --hf-* arguments
        from whisperjav.pipelines.transformers_pipeline import TransformersPipeline
        pipeline = TransformersPipeline(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            keep_temp_files=args.keep_temp,
            save_metadata_json=getattr(args, 'debug', False),  # --debug enables metadata JSON preservation
            progress_display=progress,
            hf_model_id=getattr(args, 'hf_model_id', 'kotoba-tech/kotoba-whisper-bilingual-v1.0'),
            hf_chunk_length=getattr(args, 'hf_chunk_length', 15),
            hf_stride=getattr(args, 'hf_stride', None),
            hf_batch_size=getattr(args, 'hf_batch_size', 8),
            hf_scene=getattr(args, 'hf_scene', 'none'),
            hf_beam_size=getattr(args, 'hf_beam_size', 5),
            hf_temperature=getattr(args, 'hf_temperature', 0.0),
            hf_attn=getattr(args, 'hf_attn', 'sdpa'),
            hf_timestamps=getattr(args, 'hf_timestamps', 'segment'),
            hf_language=getattr(args, 'hf_language', 'ja'),
            hf_task='translate' if args.subs_language == 'direct-to-english' else 'transcribe',
            hf_device=getattr(args, 'hf_device', 'auto'),
            hf_dtype=getattr(args, 'hf_dtype', 'auto'),
            subs_language=args.subs_language,
        )
    else:  # fidelity
        pipeline = FidelityPipeline(**pipeline_args)
    
    all_stats, failed_files = [], []
    
    # Calculate expected output lang_code for skip-existing check
    if args.subs_language == 'direct-to-english':
        output_lang_code = 'en'
    else:
        output_lang_code = LANGUAGE_CODE_MAP.get(getattr(args, 'language', 'japanese'), 'ja')

    skipped_count = 0

    try:
        for i, media_info in enumerate(media_files, 1):
            file_path_str = media_info.get('path', 'Unknown File')
            file_name = Path(file_path_str).name
            media_basename = media_info.get('basename', Path(file_path_str).stem)

            # Check if output already exists (--skip-existing)
            if getattr(args, 'skip_existing', False):
                expected_output = Path(args.output_dir) / f"{media_basename}.{output_lang_code}.whisperjav.srt"
                if expected_output.exists():
                    logger.info(f"Skipping (output exists): {file_name}")
                    skipped_count += 1
                    all_stats.append({"file": file_path_str, "status": "skipped", "reason": "output_exists"})
                    continue

            progress.set_current_file(file_path_str, i)

            try:
                metadata = pipeline.process(media_info)
                all_stats.append({"file": file_path_str, "status": "success", "metadata": metadata})
                
                subtitle_count = metadata.get("summary", {}).get("final_subtitles_refined", 0)
                output_path = metadata.get("output_files", {}).get("final_srt", "")
                
                # Add signatures to the generated subtitle file
                if output_path and Path(output_path).exists():
                    add_signatures_to_srt(
                        srt_path=output_path,
                        producer_credit=args.credit if hasattr(args, 'credit') else None,
                        add_technical_sig=not args.no_signature if hasattr(args, 'no_signature') else True,
                        mode=args.mode,
                        sensitivity=args.sensitivity,
                        version=__version__
                    )

                # Translation step (if requested)
                if args.translate and output_path:
                    try:
                        logger.info("Starting translation...")

                        # Determine stream mode based on progress settings
                        stream_mode = not getattr(args, 'no_progress', False) and not getattr(args, 'translate_quiet', False)

                        # Call translation service directly (no subprocess)
                        translated_path = translate_with_config(
                            input_path=output_path,
                            provider=args.translate_provider,
                            target_lang=args.translate_target,
                            tone=args.translate_tone,
                            api_key=getattr(args, 'translate_api_key', None),
                            model=getattr(args, 'translate_model', None),
                            stream=stream_mode,
                            debug=getattr(args, 'debug', False),
                            extra_context=build_translation_context(args),
                            n_gpu_layers=getattr(args, 'translate_gpu_layers', -1)
                        )

                        if translated_path:
                            metadata.setdefault("output_files", {})["translated_srt"] = str(translated_path)
                            logger.info(f"Translation complete: {translated_path.name}")
                        else:
                            logger.error("Translation failed: no output generated")

                    except ConfigurationError as e:
                        logger.error(f"Translation configuration error: {e}")
                        # Don't re-raise - continue with next file
                    except TranslationError as e:
                        logger.error(f"Translation failed: {e}")
                        # Don't re-raise - continue with next file
                    except FileNotFoundError as e:
                        logger.error(f"Translation failed: {e}")
                        # Don't re-raise - continue with next file
                    except Exception as e:
                        logger.error(f"Translation failed: {e}")
                        # Don't re-raise - continue with next file

                progress.show_file_complete(file_name, subtitle_count, output_path)
                
                progress.update_overall(1)
                
            except Exception as e:
                progress.show_message(f"Failed: {file_name} - {str(e)}", "error", 3.0)
                logger.error(f"Failed to process {file_path_str}: {e}", exc_info=True)
                failed_files.append(file_path_str)
                all_stats.append({"file": file_path_str, "status": "failed", "error": str(e)})
                progress.update_overall(1)
                
    finally:
        # Ensure pipeline cleanup is called even if an exception occurs
        # This releases GPU memory and prevents VRAM leaks
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during pipeline cleanup: {cleanup_error}")
        progress.close()
    
    successful_metadata = [
        entry.get("metadata")
        for entry in all_stats
        if entry.get("status") == "success" and isinstance(entry.get("metadata"), dict)
    ]
    subtitle_totals = aggregate_subtitle_metrics(successful_metadata)

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {len(media_files)}")
    print(f"Successful: {len(media_files) - len(failed_files) - skipped_count}")
    if skipped_count > 0:
        print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed: {len(failed_files)}")
    if subtitle_totals.get("processing_time_seconds"):
        total_time = subtitle_totals["processing_time_seconds"]
        print(f"Processing time (s): {total_time:.2f}")
        if subtitle_totals.get("files_processed"):
            avg_time = total_time / subtitle_totals["files_processed"]
            print(f"Average per file (s): {avg_time:.2f}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")

    print_subtitle_metrics(subtitle_totals)
    
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.stats_file}")

    # Close parameter tracer (flushes remaining data)
    tracer.close()

    if not args.keep_temp:
        cleanup_temp_directory(args.temp_dir)


def process_files_async(media_files: List[Dict], args: argparse.Namespace, resolved_config: Dict):
    """Process files asynchronously using the new async processor."""
    
    # Determine verbosity
    if args.verbosity:
        verbosity = VerbosityLevel(args.verbosity)
    else:
        config_manager = ConfigManager(args.config)
        ui_prefs = config_manager.get_ui_preferences()
        verbosity = VerbosityLevel(ui_prefs.get('console_verbosity', 'summary'))
    
    # Map language name to Whisper language code
    language_code = LANGUAGE_CODE_MAP.get(args.language, 'ja')

    # Update resolved config with runtime options
    resolved_config['output_dir'] = args.output_dir
    resolved_config['temp_dir'] = args.temp_dir
    resolved_config['keep_temp_files'] = args.keep_temp
    resolved_config['subs_language'] = args.subs_language
    resolved_config['language'] = language_code  # Whisper language code
    
    # Enhancement options
    for opt in ['adaptive_classification', 'adaptive_audio_enhancement', 'smart_postprocessing']:
        resolved_config[opt] = getattr(args, opt)
    
    # Add scene detection method for kotoba pipeline
    resolved_config['scene_method'] = getattr(args, 'scene_detection_method', None) or 'auditok'
    
    # Create async manager
    def progress_callback(message: Dict):
        """Handle progress messages."""
        msg_type = message.get('type')
        if msg_type == 'file_start':
            print(f"\nProcessing: {message['filename']}")
        elif msg_type == 'task_complete':
            if message['success']:
                print(f"✓ Completed: {message['task_id']}")
            else:
                print(f"✗ Failed: {message['task_id']} - {message.get('error', 'Unknown error')}")
    
    manager = AsyncPipelineManager(
        ui_update_callback=progress_callback,
        verbosity=verbosity
    )

    # Calculate expected output lang_code for skip-existing check
    if args.subs_language == 'direct-to-english':
        output_lang_code = 'en'
    else:
        output_lang_code = language_code

    # Filter out files with existing outputs if --skip-existing is set
    skipped_files = []
    files_to_process = media_files
    if getattr(args, 'skip_existing', False):
        files_to_process = []
        for media_info in media_files:
            file_path_str = media_info.get('path', 'Unknown File')
            media_basename = media_info.get('basename', Path(file_path_str).stem)
            expected_output = Path(args.output_dir) / f"{media_basename}.{output_lang_code}.whisperjav.srt"
            if expected_output.exists():
                logger.info(f"Skipping (output exists): {Path(file_path_str).name}")
                skipped_files.append(file_path_str)
            else:
                files_to_process.append(media_info)

        if skipped_files:
            print(f"\nSkipping {len(skipped_files)} files with existing outputs")

    try:
        # Process files
        print(f"\nProcessing {len(files_to_process)} files asynchronously...")
        task_ids = manager.process_files(files_to_process, args.mode, resolved_config)
        
        # Get actual task objects from the processor
        tasks = []
        for task_id in task_ids:
            task = manager.processor.get_task_status(task_id)
            if task:
                tasks.append(task)
        
        # Add signatures to successfully processed files
        for task in tasks:
            if task.status == ProcessingStatus.COMPLETED and hasattr(task, 'result'):
                # Try to get the output path from the task result
                if isinstance(task.result, dict):
                    output_path = task.result.get("output_files", {}).get("final_srt", "")
                    if output_path and Path(output_path).exists():
                        add_signatures_to_srt(
                            srt_path=output_path,
                            producer_credit=args.credit if hasattr(args, 'credit') else None,
                            add_technical_sig=not args.no_signature if hasattr(args, 'no_signature') else True,
                            mode=args.mode,
                            sensitivity=args.sensitivity,
                            version=__version__
                        )

                        # Translation step (if requested)
                        if args.translate and output_path:
                            try:
                                logger.info("Starting translation...")

                                # Determine stream mode based on progress settings
                                stream_mode = not getattr(args, 'no_progress', False) and not getattr(args, 'translate_quiet', False)

                                # Call translation service directly (no subprocess)
                                translated_path = translate_with_config(
                                    input_path=output_path,
                                    provider=args.translate_provider,
                                    target_lang=args.translate_target,
                                    tone=args.translate_tone,
                                    api_key=getattr(args, 'translate_api_key', None),
                                    model=getattr(args, 'translate_model', None),
                                    stream=stream_mode,
                                    debug=getattr(args, 'debug', False),
                                    extra_context=build_translation_context(args),
                                    n_gpu_layers=getattr(args, 'translate_gpu_layers', -1)
                                )

                                if translated_path:
                                    task.result.setdefault("output_files", {})["translated_srt"] = str(translated_path)
                                    logger.info(f"Translation complete: {translated_path.name}")
                                else:
                                    logger.error("Translation failed: no output generated")

                            except ConfigurationError as e:
                                logger.error(f"Translation configuration error: {e}")
                                # Don't re-raise - continue with next file
                            except TranslationError as e:
                                logger.error(f"Translation failed: {e}")
                                # Don't re-raise - continue with next file
                            except FileNotFoundError as e:
                                logger.error(f"Translation failed: {e}")
                                # Don't re-raise - continue with next file
                            except Exception as e:
                                logger.error(f"Translation failed: {e}")
                                # Don't re-raise - continue with next file
        
        # Summarize results
        successful = sum(1 for t in tasks if t.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == ProcessingStatus.FAILED)
        cancelled = sum(1 for t in tasks if t.status == ProcessingStatus.CANCELLED)
        subtitle_totals = aggregate_subtitle_metrics([
            t.result for t in tasks
            if t.status == ProcessingStatus.COMPLETED and isinstance(t.result, dict)
        ])
        
        print("\n" + "="*50)
        print("ASYNC PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files: {len(media_files)}")
        print(f"Successful: {successful}")
        if len(skipped_files) > 0:
            print(f"Skipped (already processed): {len(skipped_files)}")
        print(f"Failed: {failed}")
        if cancelled > 0:
            print(f"Cancelled: {cancelled}")

        print_subtitle_metrics(subtitle_totals)
        
        # Save stats if requested
        if args.stats_file:
            stats = []
            for task in tasks:
                stats.append({
                    'file': task.media_info['path'],
                    'status': task.status.value,
                    'duration': task.end_time - task.start_time if task.end_time else None,
                    'error': str(task.error) if task.error else None
                })
            
            with open(args.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"\nStatistics saved to: {args.stats_file}")
    
    finally:
        manager.shutdown()
        
        if not args.keep_temp:
            cleanup_temp_directory(args.temp_dir)


def main():
    """Enhanced main entry point with all V3 improvements."""
    args = parse_arguments()
    
    # Run environment checks if requested
    if args.check or args.check_verbose:
        run_preflight_checks(verbose=args.check_verbose)
        sys.exit(0)

    # Enforce GPU requirement (CUDA/MPS) with optional bypass
    enforce_gpu_requirement(accept_cpu_mode=args.accept_cpu_mode)

    # Setup logging
    global logger
    log_level = "DEBUG" if args.debug else args.log_level
    logger = setup_logger("whisperjav", log_level, args.log_file)

    if args.debug:
        logger.info("=" * 70)
        logger.info("DEBUG MODE ENABLED - Comprehensive diagnostic logging active")
        logger.info("=" * 70)

    # Enable crash tracing if requested
    if args.crash_trace:
        from whisperjav.utils.crash_tracer import get_tracer
        tracer = get_tracer()
        tracer.enable()
        logger.info("=" * 70)
        logger.info("CRASH TRACING ENABLED - Writing to crash_traces/")
        logger.info("=" * 70)

    print_banner()

    # Skip input validation if --dump-params is used (diagnostic mode)
    if not args.input and not args.dump_params:
        logger.error("No input files specified. Use -h for help.")
        sys.exit(1)
    
    # Configuration management
    config_path = Path(args.config) if args.config else None
    
    # Update UI preferences if verbosity override provided
    if args.verbosity:
        quick_update_ui_preference('console_verbosity', args.verbosity, config_path)
    
    # V3.0 Configuration resolution (Component-based)
    # Use --task if provided, otherwise determine from subs-language
    if hasattr(args, 'task') and args.task:
        task = args.task
        logger.debug(f"Task set from --task argument: '{task}'")
    else:
        task = 'translate' if args.subs_language == 'direct-to-english' else 'transcribe'
        logger.debug(f"Task derived from --subs-language='{args.subs_language}' -> task='{task}'")

    # Log task determination for debugging translation issues
    logger.info(f"ASR task: {task}" + (" (translating to English)" if task == 'translate' else " (transcribing in source language)"))
    if task == 'translate':
        logger.info("Translation mode: Output subtitles will be in English")

    # Explicit warning if direct-to-english was requested but task is not translate
    if args.subs_language == 'direct-to-english' and task != 'translate':
        logger.warning(f"WARNING: --subs-language=direct-to-english was set but task='{task}' (expected 'translate'). "
                      "This may indicate a configuration issue.")

    # Map language name to Whisper language code
    language_code = LANGUAGE_CODE_MAP.get(args.language, 'ja')
    logger.info(f"Transcription language: {args.language} ({language_code})")

    try:
        # Check if two-pass ensemble mode
        if args.ensemble:
            logger.info("Two-pass ensemble mode enabled")
            # Config will be handled by EnsembleOrchestrator
            resolved_config = None  # Not used in ensemble mode
        # Check if component-based ensemble mode (--asr provided)
        elif args.asr:
            # Ensemble mode: use ensemble resolver
            logger.info(f"Ensemble mode: ASR={args.asr}, VAD={args.vad or 'none'}")

            # Parse features
            features = []
            if args.features:
                features = [f.strip() for f in args.features.split(',') if f.strip()]

            # Parse overrides
            overrides = None
            if args.overrides:
                import json
                try:
                    overrides = json.loads(args.overrides)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --overrides: {e}")
                    sys.exit(1)

            # Resolve using ensemble resolver (returns legacy-compatible structure)
            resolved_config = resolve_ensemble_config(
                asr=args.asr,
                vad=args.vad or 'none',
                task=task,
                features=features,
                overrides=overrides,
                device=args.device,
                compute_type=args.compute_type,
            )

        elif args.mode == "transformers":
            # Transformers mode: uses dedicated --hf-* arguments, not legacy config
            # No config resolution needed - pipeline receives args directly
            resolved_config = None
            logger.debug("Transformers mode: skipping legacy config resolution (uses --hf-* args)")

        else:
            # Legacy mode: use pipeline resolver
            resolved_config = resolve_legacy_pipeline(
                pipeline_name=args.mode,
                sensitivity=args.sensitivity,
                task=task,
                device=args.device,
                compute_type=args.compute_type,
            )

        if args.scene_detection_method:
            logger.info(f"Using scene detection method: {args.scene_detection_method}")

    except Exception as e:
        logger.error(f"Failed to resolve configuration: {e}")
        sys.exit(1)

    logger.debug(f"Resolved configuration for pipeline='{args.mode}', sensitivity='{args.sensitivity}', task='{task}'")
    
    # Apply model override if specified via CLI (not for ensemble mode)
    if args.model and resolved_config is not None:
        logger.info(f"Overriding model with CLI argument: {args.model}")
        # Determine device and compute_type for model override
        # Priority: CLI args > auto-detection
        override_device = args.device if args.device and args.device != "auto" else get_best_device()
        override_compute_type = args.compute_type if args.compute_type and args.compute_type != "auto" else "int8"
        # Create a model configuration for the CLI-specified model
        override_model_config = {
            "provider": "openai_whisper",  # Default provider
            "model_name": args.model,
            "device": override_device,
            "compute_type": override_compute_type,
            "supported_tasks": ["transcribe", "translate"]
        }
        resolved_config["model"] = override_model_config
        logger.debug(f"Model override applied: {args.model} (device={override_device}, compute_type={override_compute_type})")

    # Apply language override to decoder params (not for ensemble mode)
    if resolved_config is not None and "params" in resolved_config and "decoder" in resolved_config["params"]:
        resolved_config["params"]["decoder"]["language"] = language_code
        logger.debug(f"Language override applied to decoder params: {language_code}")

    # Apply --no-vad override for supported modes
    if getattr(args, 'no_vad', False) and resolved_config is not None:
        if "params" not in resolved_config:
            resolved_config["params"] = {}

        if args.mode == "kotoba-faster-whisper":
            # Kotoba uses faster-whisper's internal vad_filter
            if "asr" not in resolved_config["params"]:
                resolved_config["params"]["asr"] = {}
            resolved_config["params"]["asr"]["vad_filter"] = False
            logger.info("Internal VAD disabled via --no-vad flag (kotoba mode)")
        elif args.mode in ["balanced", "fidelity"]:
            # Balanced/fidelity use Speech Segmenter - set backend to "none" to disable
            if "params" not in resolved_config:
                resolved_config["params"] = {}
            if "speech_segmenter" not in resolved_config["params"]:
                resolved_config["params"]["speech_segmenter"] = {}
            resolved_config["params"]["speech_segmenter"]["backend"] = "none"
            logger.info("Speech segmentation disabled via --no-vad flag (backend set to 'none')")

    # Apply --speech-segmenter override
    speech_segmenter = getattr(args, 'speech_segmenter', None)
    if speech_segmenter is not None and resolved_config is not None:
        if "params" not in resolved_config:
            resolved_config["params"] = {}
        if "speech_segmenter" not in resolved_config["params"]:
            resolved_config["params"]["speech_segmenter"] = {}
        resolved_config["params"]["speech_segmenter"]["backend"] = speech_segmenter
        logger.info(f"Speech segmenter set to: {speech_segmenter}")
        # Note: Speech Segmenter factory handles "none" backend internally

    # Handle --dump-params: dump resolved config to JSON and exit
    if args.dump_params:
        import json
        dump_data = {
            "mode": args.mode,
            "sensitivity": args.sensitivity,
            "subs_language": args.subs_language,
            "language_code": language_code,
            "resolved_config": resolved_config,
            "cli_args": {
                "model": args.model,
                "ensemble": args.ensemble,
                "asr": getattr(args, 'asr', None),
                "vad": getattr(args, 'vad', None),
                "speech_segmenter": getattr(args, 'speech_segmenter', None),
                "transformers_two_pass": getattr(args, 'transformers_two_pass', False),
            }
        }
        try:
            with open(args.dump_params, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"Parameters dumped to: {args.dump_params}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to dump parameters: {e}")
            sys.exit(1)

    # Setup temp directory
    if args.temp_dir:
        temp_path = Path(args.temp_dir)
    else:
        temp_path = Path(tempfile.gettempdir()) / "whisperjav"
    temp_path.mkdir(parents=True, exist_ok=True)
    args.temp_dir = str(temp_path)
    logger.debug(f"Using temporary directory: {args.temp_dir}")
    
    # Discover media files
    discovery = MediaDiscovery()
    media_files = discovery.discover(args.input)
    
    if not media_files:
        logger.error(f"No valid media files found in the specified paths: {', '.join(args.input)}")
        sys.exit(1)
    
    logger.info(f"Found {len(media_files)} media file(s) to process:")
    for f in media_files:
        logger.info(f"  - {f['path']}")

    # Create parameter tracer for ensemble mode (must be created before try block)
    tracer = create_tracer(args.trace_params) if args.ensemble else None
    if tracer and args.trace_params:
        logger.info(f"Parameter tracing enabled: {args.trace_params}")

    try:
        # Check if two-pass ensemble mode
        if args.ensemble:
            # Use EnsembleOrchestrator for two-pass processing
            from whisperjav.ensemble.orchestrator import EnsembleOrchestrator
            import json

            # Parse pass 1 full params (custom mode) or overrides (deprecated)
            pass1_params = None
            if args.pass1_params:
                try:
                    pass1_params = json.loads(args.pass1_params)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass1-params: {e}")
                    sys.exit(1)
            elif args.pass1_overrides:
                # Deprecated: use overrides
                try:
                    pass1_params = json.loads(args.pass1_overrides)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass1-overrides: {e}")
                    sys.exit(1)

            # Parse pass 2 full params or overrides
            pass2_params = None
            if args.pass2_params:
                try:
                    pass2_params = json.loads(args.pass2_params)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass2-params: {e}")
                    sys.exit(1)
            elif args.pass2_overrides:
                try:
                    pass2_params = json.loads(args.pass2_overrides)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass2-overrides: {e}")
                    sys.exit(1)

            # Parse HuggingFace Transformers params (for pipeline=transformers)
            pass1_hf_params = None
            if args.pass1_hf_params:
                try:
                    pass1_hf_params = json.loads(args.pass1_hf_params)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass1-hf-params: {e}")
                    sys.exit(1)

            pass2_hf_params = None
            if args.pass2_hf_params:
                try:
                    pass2_hf_params = json.loads(args.pass2_hf_params)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in --pass2-hf-params: {e}")
                    sys.exit(1)

            # Build pass configurations
            # Full Configuration Snapshot approach:
            # - If params provided: use as full config (customized)
            # - If not: backend resolves from pipeline+sensitivity
            # - For transformers: use hf_params if provided
            pass1_config = {
                'pipeline': args.pass1_pipeline,
                'sensitivity': args.pass1_sensitivity,
                'scene_detector': args.pass1_scene_detector,
                'speech_segmenter': args.pass1_speech_segmenter,
                'speech_enhancer': args.pass1_speech_enhancer,
                'model': args.pass1_model,
                'params': pass1_params,  # None = use defaults, object = custom
                'hf_params': pass1_hf_params,  # For transformers pipeline
                'language': language_code,  # Source language code (e.g., 'en', 'ja')
                'device': args.device,  # Hardware override (None = auto-detect)
                'compute_type': args.compute_type,  # Compute type override (None = auto)
            }

            pass2_config = None
            if args.pass2_pipeline:
                pass2_config = {
                    'pipeline': args.pass2_pipeline,
                    'sensitivity': args.pass2_sensitivity,
                    'scene_detector': args.pass2_scene_detector,
                    'speech_segmenter': args.pass2_speech_segmenter,
                    'speech_enhancer': args.pass2_speech_enhancer,
                    'model': args.pass2_model,
                    'params': pass2_params,
                    'hf_params': pass2_hf_params,  # For transformers pipeline
                    'language': language_code,  # Source language code (e.g., 'en', 'ja')
                    'device': args.device,  # Hardware override (None = auto-detect)
                    'compute_type': args.compute_type,  # Compute type override (None = auto)
                }

            # Create orchestrator
            orchestrator = EnsembleOrchestrator(
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                keep_temp_files=args.keep_temp,
                save_metadata_json=getattr(args, 'debug', False),  # --debug enables metadata JSON preservation
                subs_language=args.subs_language,
                parameter_tracer=tracer,
                log_level=log_level,
            )

            # Process all files with batch processing for optimal VRAM usage
            # This loads each model only once for all files instead of per-file
            results = orchestrator.process_batch(
                media_files=media_files,
                pass1_config=pass1_config,
                pass2_config=pass2_config,
                merge_strategy=args.merge_strategy
            )

            # Report individual results
            failed_files = []
            successful_count = 0
            total_processing_time = 0.0

            for result in results:
                basename = result.get('input', {}).get('basename', 'unknown')
                if result.get('error') or result.get('status') == 'failed':
                    logger.error(f"Failed: {basename} - {result.get('error', 'Unknown error')}")
                    failed_files.append(basename)
                else:
                    output_path = result.get('summary', {}).get('final_output', 'unknown')
                    logger.info(f"Completed: {output_path}")
                    successful_count += 1
                    total_processing_time += result.get('summary', {}).get('total_processing_time_seconds', 0.0)

            # Print ensemble processing summary (matching standard pipeline format)
            print("\n" + "="*50)
            print("ENSEMBLE PROCESSING SUMMARY")
            print("="*50)
            print(f"Total files: {len(media_files)}")
            print(f"Successful: {successful_count}")
            print(f"Failed: {len(failed_files)}")
            if total_processing_time > 0:
                print(f"Total processing time: {total_processing_time:.2f}s")
                if successful_count > 0:
                    print(f"Average per file: {total_processing_time / successful_count:.2f}s")
            if failed_files:
                print("\nFailed files:")
                for f in failed_files:
                    print(f"  - {f}")
            print("="*50)

            # ============================================================
            # TRANSLATION: Translate successful ensemble outputs if requested
            # ============================================================
            if args.translate and successful_count > 0:
                print("\n" + "="*50)
                print("STARTING TRANSLATION")
                print("="*50)

                translation_success = 0
                translation_failed = 0
                extra_context = build_translation_context(args)

                for result in results:
                    if result.get('error') or result.get('status') == 'failed':
                        continue  # Skip failed transcriptions

                    output_path = result.get('summary', {}).get('final_output')
                    if not output_path:
                        continue

                    basename = result.get('input', {}).get('basename', 'unknown')

                    try:
                        logger.info(f"Translating: {basename}")
                        print(f"Translating: {basename}")

                        translated_path = translate_with_config(
                            input_path=output_path,
                            provider=args.translate_provider,
                            target_lang=args.translate_target,
                            tone=args.translate_tone,
                            api_key=getattr(args, 'translate_api_key', None),
                            model=getattr(args, 'translate_model', None),
                            debug=getattr(args, 'debug', False),
                            extra_context=extra_context,
                            n_gpu_layers=getattr(args, 'translate_gpu_layers', -1)
                        )

                        if translated_path:
                            logger.info(f"Translation saved: {translated_path}")
                            print(f"  -> {translated_path}")
                            translation_success += 1
                        else:
                            logger.warning(f"Translation returned no output for {basename}")
                            translation_failed += 1

                    except (TranslationError, ConfigurationError) as e:
                        logger.error(f"Translation failed for {basename}: {e}")
                        translation_failed += 1
                    except Exception as e:
                        logger.error(f"Unexpected translation error for {basename}: {e}")
                        translation_failed += 1

                # Print translation summary
                print("\n" + "-"*50)
                print("TRANSLATION SUMMARY")
                print("-"*50)
                print(f"Translated: {translation_success}")
                print(f"Failed: {translation_failed}")
                print("="*50)

            # Close parameter tracer for ensemble mode
            if tracer:
                tracer.close()

        # Choose sync or async processing for normal mode
        elif args.async_processing:
            process_files_async(media_files, args, resolved_config)
        else:
            process_files_sync(media_files, args, resolved_config)
            
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        if not args.keep_temp:
            logger.debug("Cleaning up temporary files...")
            cleanup_temp_directory(args.temp_dir)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if not args.keep_temp:
            logger.debug("Cleaning up temporary files...")
            cleanup_temp_directory(args.temp_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()