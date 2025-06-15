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

from copy import deepcopy

from whisperjav.utils.logger import setup_logger, logger
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
# --- NEW: Import the TranscriptionTuner ---
from whisperjav.config.transcription_tuner import TranscriptionTuner

__version__ = "1.1.0"

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
    print(banner)


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

    pipeline_args = {
        "output_dir": args.output_dir,
        "temp_dir": args.temp_dir,
        "keep_temp_files": args.keep_temp,
        "subs_language": args.subs_language,
        "resolved_params": resolved_params, # Pass the single resolved params dictionary
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
    for i, media_info in enumerate(media_files, 1):
        file_path_str = media_info.get('path', 'Unknown File')
        file_name = Path(file_path_str).name
        logger.info(f"\nProcessing file {i}/{total_files}: {file_name}")
        try:
            metadata = pipeline.process(media_info)
            all_stats.append({"file": file_path_str, "status": "success", "metadata": metadata})
        except Exception as e:
            logger.error(f"Failed to process {file_path_str}: {e}", exc_info=True)
            failed_files.append(file_path_str)
            all_stats.append({"file": file_path_str, "status": "failed", "error": str(e)})
            
    logger.info("\n" + "="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {total_files - len(failed_files)}")
    logger.info(f"Failed: {len(failed_files)}")
    if failed_files:
        logger.warning("\nFailed files:")
        for file in failed_files:
            logger.warning(f"  - {file}")
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nStatistics saved to: {args.stats_file}")


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
    logger.info(f"Using temporary directory: {args.temp_dir}")
        
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
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

