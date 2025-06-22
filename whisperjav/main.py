#!/usr/bin/env python3
"""WhisperJAV - Japanese Adult Video Subtitle Generator

Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path
import json
import tempfile
from typing import Dict, List, Any 
import io
import shutil  
from copy import deepcopy

from whisperjav.utils.progress_display import ProgressDisplay, DummyProgress



# Fix stdout before any imports that might use logging
def fix_stdout():
    """Ensure stdout is available, create a wrapper if needed."""
    if sys.stdout is None or (hasattr(sys.stdout, 'closed') and sys.stdout.closed):
        # Redirect to a new text wrapper
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
# --- NEW: Import the TranscriptionTuner ---
from whisperjav.config.transcription_tuner import TranscriptionTuner

__version__ = "1.1.0"

def safe_print(message: str):
    """Safely print a message, handling closed stdout."""
    try:
        print(message)
    except (ValueError, AttributeError, OSError):
        # If printing fails, try to write directly to stderr or ignore
        try:
            sys.stderr.write(message + '\n')
        except:
            pass  # If all else fails, just continue

def print_banner():
    """Print application banner."""
    banner = f"""
╔═══════════════════════════════════════════════════╗
║          WhisperJAV v{__version__}                        ║
║   Japanese Adult Video Subtitle Generator         ║
║   Optimized for JAV content transcription         ║
║                                                   ║
║   Available modes:                                ║
║   - faster: Faster-whisper backend                ║
║   - fast: Standard whisper with scene detection   ║
║   - balanced: Scene detection + VAD-enhanced ASR  ║
╚═══════════════════════════════════════════════════╝
"""
    safe_print(banner)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    The `override` dictionary's values take precedence.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def cleanup_temp_directory(temp_dir: str):
    """Clean up the temporary directory after processing."""
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        return
    
    # Check if this is the WhisperJAV subdirectory in system temp
    if temp_path.name == "whisperjav" and temp_path.parent == Path(tempfile.gettempdir()):
        logger.debug(f"Cleaning up WhisperJAV temp directory: {temp_path}")
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
            logger.debug("Temp directory cleaned up successfully")
        except Exception as e:
            logger.error(f"Error removing temp directory: {e}")
    else:
        # For custom temp directories, just clean the contents but keep the directory
        logger.debug(f"Cleaning up temp directory contents: {temp_path}")
        try:
            # List subdirectories that should be cleaned
            subdirs_to_clean = ["scenes", "scene_srts", "raw_subs"]
            
            for subdir in subdirs_to_clean:
                subdir_path = temp_path / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path, ignore_errors=True)
                    logger.debug(f"Removed temp subdirectory: {subdir_path}")
            
            # Remove any remaining files in temp root
            for file in temp_path.glob("*"):
                if file.is_file():
                    file.unlink()
                    logger.debug(f"Removed temp file: {file}")
                    
            logger.info("Temp directory contents cleaned up successfully")
                
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV - Generate accurate subtitles for Japanese adult videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core Execution Arguments
    parser.add_argument("input", nargs="*", help="Input media file(s), directory, or wildcard pattern.")
    parser.add_argument("--mode", choices=["balanced", "fast", "faster"], default="balanced", help="Processing mode (default: balanced)")
    parser.add_argument("--config", default=None, help="Path to a JSON configuration file. Overrides defaults.")
    parser.add_argument(
        "--subs-language", 
        choices=["japanese", "english-direct"], 
        default="japanese",
        help="Specify the output language for the subtitle file. 'english-direct' will translate from audio."
    )
    
    # Path and Logging Options
    path_group = parser.add_argument_group("Path and Logging Options")
    path_group.add_argument("--output-dir", default="./output", help="Output directory for subtitles (default: ./output)")
    path_group.add_argument("--temp-dir", default=None, help="Temporary directory for processing (default: OS-specific temp folder)")
    path_group.add_argument("--keep-temp", action="store_true", help="Keep temporary files after processing")
    path_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level (default: INFO)")
    path_group.add_argument("--log-file", help="Log file path")
    path_group.add_argument("--stats-file", help="Save processing statistics to JSON file")  
    path_group.add_argument("--no-progress", action="store_true", 
                           help="Disable progress bars and use traditional scrolling output")

    
    # Optional Enhancement Features
    enhancement_group = parser.add_argument_group("Optional Enhancement Features")
    enhancement_group.add_argument("--adaptive-classification", action="store_true", help="Enable adaptive classification.")
    enhancement_group.add_argument("--adaptive-audio-enhancement", action="store_true", help="Enable adaptive audio enhancement.")
    enhancement_group.add_argument("--smart-postprocessing", action="store_true", help="Enable advanced subtitle refinement.")
    
    # Advanced Tuning Arguments
    tuning_group = parser.add_argument_group("Advanced Tuning (Overrides config file)")
    tuning_group.add_argument("--model", default='turbo', help="Whisper model size for balanced mode (e.g., 'large-v2').")
    tuning_group.add_argument("--vad-threshold", type=float, default=None, help="VAD threshold for speech detection.")
    # --- NEW: Added --sensitivity flag ---
    tuning_group.add_argument(
        "--sensitivity",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Set the transcription sensitivity to control the detail vs. noise trade-off."
    )
    
    parser.add_argument("--version", action="version", version=f"WhisperJAV {__version__}")
    
    return parser.parse_args()



def process_files(media_files: List[Dict], args: argparse.Namespace, resolved_params: Dict):
    """Process multiple files with the selected pipeline."""
    
    enhancement_kwargs = {
        "adaptive_classification": args.adaptive_classification,
        "adaptive_audio_enhancement": args.adaptive_audio_enhancement,
        "smart_postprocessing": args.smart_postprocessing
    }

    # Initialize progress display
    progress = ProgressDisplay(len(media_files), enabled=not args.no_progress)
    
    pipeline_args = {
        "output_dir": args.output_dir,
        "temp_dir": args.temp_dir,
        "keep_temp_files": args.keep_temp,
        "subs_language": args.subs_language,
        "resolved_params": resolved_params,
        "progress_display": progress,  # Pass progress display to pipeline
        **enhancement_kwargs
    }

    if args.mode == "faster":
        pipeline = FasterPipeline(**pipeline_args)
    elif args.mode == "fast":
        pipeline = FastPipeline(**pipeline_args)
    elif args.mode == "balanced":
        pipeline = BalancedPipeline(**pipeline_args)
    else:
        logger.error(f"Unknown mode '{args.mode}'")
        sys.exit(1)
        
    all_stats, failed_files = [], []
    total_files = len(media_files)
    
    try:
        for i, media_info in enumerate(media_files, 1):
            file_path_str = media_info.get('path', 'Unknown File')
            file_name = Path(file_path_str).name
            
            progress.set_current_file(file_path_str, i)
            
            try:
                metadata = pipeline.process(media_info)
                all_stats.append({"file": file_path_str, "status": "success", "metadata": metadata})
                
                # Show completion
                subtitle_count = metadata.get("summary", {}).get("final_subtitles_refined", 0)
                output_path = metadata.get("output_files", {}).get("final_srt", "")
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
    
    # Print summary (after progress bars are closed)
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {total_files}")
    print(f"Successful: {total_files - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
            
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.stats_file}")
    
    # Clean up temp directory if not keeping temp files
    if not args.keep_temp:
        if args.no_progress:
            logger.debug("Cleaning up temporary files...")
        cleanup_temp_directory(args.temp_dir)


def main():
    """Main entry point."""
    args = parse_arguments()

    global logger
    logger = setup_logger("whisperjav", args.log_level, args.log_file)
    print_banner()

    if not args.input:
        logger.error("No input files specified. Use -h for help.")
        sys.exit(1)

    # --- REVISED: Use TranscriptionTuner to handle all config logic ---
    
    # 1. Instantiate the tuner. It handles loading the base and user configs.
    config_path = Path(args.config) if args.config else None
    tuner = TranscriptionTuner(config_path=config_path)

    # 2. Get the fully resolved parameters for all workers.
    # The tuner handles the logic for mode and sensitivity.
    resolved_params = tuner.get_resolved_params(
        mode=args.mode,
        sensitivity=args.sensitivity
    )

    if not resolved_params:
        logger.error("Could not resolve transcription parameters. Check config file. Exiting.")
        sys.exit(1)
        
    # 3. Apply final CLI overrides for parameters not related to sensitivity profiles.
    # This logic is kept for backward compatibility and specific use cases.
    if args.model:
        if 'model_load_params' not in resolved_params: resolved_params['model_load_params'] = {}
        resolved_params['model_load_params']['model_name'] = args.model
        logger.debug(f"CLI override: model_name set to '{args.model}'")

    if args.vad_threshold is not None:
        if 'vad_options' not in resolved_params: resolved_params['vad_options'] = {}
        resolved_params['vad_options']['threshold'] = args.vad_threshold
        logger.debug(f"CLI override: vad_threshold set to '{args.vad_threshold}'")

    # 4. Set up temp directory
    if args.temp_dir:
        temp_path = Path(args.temp_dir)
    else:
        temp_path = Path(tempfile.gettempdir()) / "whisperjav"
    temp_path.mkdir(parents=True, exist_ok=True)
    args.temp_dir = str(temp_path)
    logger.debug(f"Using temporary directory: {args.temp_dir}")
        
    discovery = MediaDiscovery()
    media_files = discovery.discover(args.input)

    if not media_files:
        logger.error(f"No valid media files found in the specified paths: {', '.join(args.input)}")
        sys.exit(1)

    logger.info(f"Found {len(media_files)} media file(s) to process")

    try:
        # Pass the final, resolved parameters to the processing function
        process_files(media_files, args, resolved_params)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        # Clean up temp directory even on interrupt
        if not args.keep_temp:
            logger.debug("Cleaning up temporary files...")
            cleanup_temp_directory(args.temp_dir)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        # Clean up temp directory even on error
        if not args.keep_temp:
            logger.debug("Cleaning up temporary files...")
            cleanup_temp_directory(args.temp_dir)
        sys.exit(1)

if __name__ == "__main__":
    main()