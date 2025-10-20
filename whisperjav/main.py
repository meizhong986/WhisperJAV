#!/usr/bin/env python3
"""WhisperJAV Main Entry Point - V3 Enhanced with all improvements."""

import os
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
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.config.transcription_tuner import TranscriptionTuner
from whisperjav.__version__ import __version__


from whisperjav.utils.preflight_check import enforce_cuda_requirement, run_preflight_checks
from whisperjav.utils.progress_aggregator import VerbosityLevel, create_progress_handler
from whisperjav.utils.async_processor import AsyncPipelineManager, ProcessingStatus
from whisperjav.config.manager import ConfigManager, quick_update_ui_preference



# --- UNCONDITIONAL CUDA CHECK ---
# This code runs the moment the module is loaded,
# ensuring the check is never bypassed.
# Bypass for help/version/check
args = sys.argv[1:]
bypass_flags = ['--check', '--help', '-h', '--version', '-v']
if not any(flag in args for flag in bypass_flags):
    enforce_cuda_requirement()
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
║   CUDA GPU Required - Optimized for Performance   ║
║                                                   ║
║   Available modes:                                ║
║   - faster: Direct transcription (fastest)        ║
║   - fast: Scene detection + standard Whisper      ║
║   - balanced: Scene + VAD-enhanced processing     ║
║                                                   ║
║   Run with --check for environment diagnostics    ║
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
    parser.add_argument("--mode", choices=["balanced", "fast", "faster"], default="balanced", 
                       help="Processing mode (default: balanced)")
    parser.add_argument("--model", default=None, 
                       help="Whisper model to use (e.g., large-v2, turbo, large). Overrides config default.")
    parser.add_argument("--config", default=None, help="Path to a JSON configuration file")
    parser.add_argument("--subs-language", choices=["japanese", "english-direct"], 
                       default="japanese", help="Output subtitle language")
    
    # Environment check
    parser.add_argument("--check", action="store_true", help="Run environment checks and exit")
    parser.add_argument("--check-verbose", action="store_true", help="Run verbose environment checks")
    
    # Path and logging
    path_group = parser.add_argument_group("Path and Logging Options")
    path_group.add_argument("--output-dir", default="./output", help="Output directory")
    path_group.add_argument("--temp-dir", default=None, help="Temporary directory")
    path_group.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    path_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                           default="INFO", help="Logging level")
    path_group.add_argument("--log-file", help="Log file path")
    path_group.add_argument("--stats-file", help="Save processing statistics to JSON")
    
    # Progress control
    progress_group = parser.add_argument_group("Progress Display Options")
    progress_group.add_argument("--no-progress", action="store_true", 
                               help="Disable progress bars")
    progress_group.add_argument("--verbosity", 
                               choices=["quiet", "summary", "normal", "verbose"],
                               default=None,
                               help="Console output verbosity (overrides config)")
    
    # Enhancement features
    enhancement_group = parser.add_argument_group("Optional Enhancement Features")
    enhancement_group.add_argument("--adaptive-classification", action="store_true")
    enhancement_group.add_argument("--adaptive-audio-enhancement", action="store_true")
    enhancement_group.add_argument("--smart-postprocessing", action="store_true")
    
    # Transcription tuning
    tuning_group = parser.add_argument_group("Transcription Tuning")
    tuning_group.add_argument("--sensitivity", 
                             choices=["conservative", "balanced", "aggressive"],
                             default="balanced", help="Transcription sensitivity")
    
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
        choices=["deepseek", "openrouter", "gemini", "claude", "gpt"],
        default="deepseek",
        help="Translation AI provider (default: deepseek)"
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
        help="Translation model override"
    )
    translation_group.add_argument(
        "--translate-quiet",
        action="store_true",
        help="Hide translation progress messages (default: show progress)"
    )

    parser.add_argument("--version", action="version", version=f"WhisperJAV {__version__}")

    return parser.parse_args()


def add_signatures_to_srt(srt_path: str, producer_credit: str = None, 
                          add_technical_sig: bool = True, 
                          mode: str = "balanced", 
                          sensitivity: str = "balanced",
                          version: str = __version__):
    """Add producer credit and/or technical signature to SRT file.
    
    Args:
        srt_path: Path to the SRT file to modify
        producer_credit: Optional producer credit text to add at beginning
        add_technical_sig: Whether to add technical signature at end
        mode: Processing mode used (faster/fast/balanced)
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
        "subs_language": args.subs_language,
        "resolved_config": resolved_config,
        "progress_display": progress,
        **enhancement_kwargs
    }
    
    # Select pipeline
    if args.mode == "faster":
        pipeline = FasterPipeline(**pipeline_args)
    elif args.mode == "fast":
        pipeline = FastPipeline(**pipeline_args)
    else:  # balanced
        pipeline = BalancedPipeline(**pipeline_args)
    
    all_stats, failed_files = [], []
    
    try:
        for i, media_info in enumerate(media_files, 1):
            file_path_str = media_info.get('path', 'Unknown File')
            file_name = Path(file_path_str).name
            
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

                        # Build command
                        cmd = [
                            "whisperjav-translate",
                            "-i", output_path,
                            "--provider", args.translate_provider,
                            "-t", args.translate_target,
                            "--tone", args.translate_tone,
                        ]

                        # Stream progress unless suppressed
                        if not getattr(args, 'no_progress', False) and not getattr(args, 'translate_quiet', False):
                            cmd.append("--stream")
                        elif getattr(args, 'no_progress', False) or getattr(args, 'translate_quiet', False):
                            cmd.append("--no-progress")

                        # Add optional args
                        if hasattr(args, 'translate_api_key') and args.translate_api_key:
                            cmd.extend(["--api-key", args.translate_api_key])
                        if hasattr(args, 'translate_model') and args.translate_model:
                            cmd.extend(["--model", args.translate_model])

                        # Run subprocess
                        # Default: show progress (stderr visible), quiet: hide everything
                        if hasattr(args, 'translate_quiet') and args.translate_quiet:
                            result = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=600
                            )
                        else:
                            # Verbose mode: show stderr (progress), capture stdout (output path)
                            result = subprocess.run(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=None,
                                text=True,
                                timeout=600
                            )

                        if result.returncode == 0:
                            # Parse output path from stdout
                            translated_path = result.stdout.strip()
                            metadata.setdefault("output_files", {})["translated_srt"] = translated_path
                            logger.info(f"Translation complete: {Path(translated_path).name}")
                        else:
                            # In quiet mode, show captured stderr; in verbose mode, already shown
                            if hasattr(args, 'translate_quiet') and args.translate_quiet and result.stderr:
                                logger.error(f"Translation failed: {result.stderr}")
                            else:
                                logger.error("Translation failed")

                    except subprocess.TimeoutExpired:
                        logger.error("Translation timed out after 10 minutes")
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
        progress.close()
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {len(media_files)}")
    print(f"Successful: {len(media_files) - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.stats_file}")
    
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
    
    # Update resolved config with runtime options
    resolved_config['output_dir'] = args.output_dir
    resolved_config['temp_dir'] = args.temp_dir
    resolved_config['keep_temp_files'] = args.keep_temp
    resolved_config['subs_language'] = args.subs_language
    
    # Enhancement options
    for opt in ['adaptive_classification', 'adaptive_audio_enhancement', 'smart_postprocessing']:
        resolved_config[opt] = getattr(args, opt)
    
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
    
    try:
        # Process files
        print(f"\nProcessing {len(media_files)} files asynchronously...")
        task_ids = manager.process_files(media_files, args.mode, resolved_config)
        
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

                                # Build command
                                cmd = [
                                    "whisperjav-translate",
                                    "-i", output_path,
                                    "--provider", args.translate_provider,
                                    "-t", args.translate_target,
                                    "--tone", args.translate_tone,
                                ]

                                # Stream progress unless suppressed
                                if not getattr(args, 'no_progress', False) and not getattr(args, 'translate_quiet', False):
                                    cmd.append("--stream")
                                elif getattr(args, 'no_progress', False) or getattr(args, 'translate_quiet', False):
                                    cmd.append("--no-progress")

                                # Add optional args
                                if hasattr(args, 'translate_api_key') and args.translate_api_key:
                                    cmd.extend(["--api-key", args.translate_api_key])
                                if hasattr(args, 'translate_model') and args.translate_model:
                                    cmd.extend(["--model", args.translate_model])

                                # Run subprocess
                                # Default: show progress (stderr visible), quiet: hide everything
                                if hasattr(args, 'translate_quiet') and args.translate_quiet:
                                    result = subprocess.run(
                                        cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=600
                                    )
                                else:
                                    # Verbose mode: show stderr (progress), capture stdout (output path)
                                    result = subprocess.run(
                                        cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=None,
                                        text=True,
                                        timeout=600
                                    )

                                if result.returncode == 0:
                                    # Parse output path from stdout
                                    translated_path = result.stdout.strip()
                                    task.result.setdefault("output_files", {})["translated_srt"] = translated_path
                                    logger.info(f"Translation complete: {Path(translated_path).name}")
                                else:
                                    # In quiet mode, show captured stderr; in verbose mode, already shown
                                    if hasattr(args, 'translate_quiet') and args.translate_quiet and result.stderr:
                                        logger.error(f"Translation failed: {result.stderr}")
                                    else:
                                        logger.error("Translation failed")

                            except subprocess.TimeoutExpired:
                                logger.error("Translation timed out after 10 minutes")
                            except Exception as e:
                                logger.error(f"Translation failed: {e}")
                                # Don't re-raise - continue with next file
        
        # Summarize results
        successful = sum(1 for t in tasks if t.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == ProcessingStatus.FAILED)
        cancelled = sum(1 for t in tasks if t.status == ProcessingStatus.CANCELLED)
        
        print("\n" + "="*50)
        print("ASYNC PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files: {len(media_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if cancelled > 0:
            print(f"Cancelled: {cancelled}")
        
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
    
    # Enforce CUDA requirement
    enforce_cuda_requirement()
    
    # Setup logging
    global logger
    logger = setup_logger("whisperjav", args.log_level, args.log_file)
    print_banner()
    
    if not args.input:
        logger.error("No input files specified. Use -h for help.")
        sys.exit(1)
    
    # Configuration management
    config_path = Path(args.config) if args.config else None
    
    # Update UI preferences if verbosity override provided
    if args.verbosity:
        quick_update_ui_preference('console_verbosity', args.verbosity, config_path)
    
    # V3 Configuration resolution
    tuner = TranscriptionTuner(config_path=config_path)
    task = 'translate' if args.subs_language == 'english-direct' else 'transcribe'
    
    try:
        resolved_config = tuner.resolve_params(
            pipeline_name=args.mode,
            sensitivity=args.sensitivity,
            task=task
        )
    except Exception as e:
        logger.error(f"Failed to resolve configuration: {e}")
        sys.exit(1)
    
    logger.debug(f"Resolved configuration for pipeline='{args.mode}', sensitivity='{args.sensitivity}', task='{task}'")
    
    # Apply model override if specified via CLI
    if args.model:
        logger.info(f"Overriding model with CLI argument: {args.model}")
        # Create a model configuration for the CLI-specified model
        override_model_config = {
            "provider": "openai_whisper",  # Default provider
            "model_name": args.model,
            "device": "cuda",
            "compute_type": "float16",
            "supported_tasks": ["transcribe", "translate"]
        }
        resolved_config["model"] = override_model_config
        logger.debug(f"Model override applied: {args.model}")
    
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
    
    logger.info(f"Found {len(media_files)} media file(s) to process")
    
    try:
        # Choose sync or async processing
        if args.async_processing:
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