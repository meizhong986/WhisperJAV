#!/usr/bin/env python3
"""WhisperJAV - Japanese Adult Video Subtitle Generator

Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import json

from whisperjav.utils.logger import setup_logger, logger
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline

__version__ = "1.1.0"

def print_banner():
    """Print application banner."""
    banner = f"""
╔═══════════════════════════════════════════════════╗
║          WhisperJAV v{__version__}                ║
║   Japanese Adult Video Subtitle Generator         ║
║   Optimized for JAV content transcription         ║
║                                                   ║
║   Available modes:                                ║
║   - faster: Faster-whisper backend                ║
║   - fast: Standard whisper with chunking          ║
║   - balanced: Coming soon                         ║
╚═══════════════════════════════════════════════════╝
"""
    print(banner)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV - Generate accurate subtitles for Japanese adult videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file in fast mode
  whisperjav "path/to/my video.mp4" --mode fast

  # Process all MKV files in a directory
  whisperjav "./videos/*.mkv" --mode faster --output-dir ./subs

  # Keep temporary files for debugging
  whisperjav video.mp4 --keep-temp
"""
    )
    
    parser.add_argument(
        "input",
        nargs="*",
        help="Input media file(s), directory, or wildcard pattern (e.g., \"*.mp4\")."
    )
    
    parser.add_argument(
        "--mode",
        choices=["balanced", "fast", "faster"],
        default="fast",
        help="Processing mode (default: fast)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for subtitles (default: ./output)"
    )
    
    parser.add_argument(
        "--temp-dir",
        default="./temp",
        help="Temporary directory for processing (default: ./temp)"
    )
    
    parser.add_argument(
        "--language",
        default="ja",
        help="Source language (default: ja)"
    )
    
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"WhisperJAV {__version__}"
    )
    
    parser.add_argument(
        "--stats-file",
        help="Save processing statistics to JSON file"
    )
    
    return parser.parse_args()

def process_files(media_files: List[Dict], args):
    """Process multiple files with the selected pipeline."""
    # Select pipeline based on mode
    if args.mode == "faster":
        pipeline = FasterPipeline(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            keep_temp_files=args.keep_temp
        )
    elif args.mode == "fast":
        pipeline = FastPipeline(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            keep_temp_files=args.keep_temp
        )
    elif args.mode == "balanced":
        logger.error(f"Mode 'balanced' not yet implemented. Please use 'faster' or 'fast' mode.")
        sys.exit(1)
    else:
        logger.error(f"Unknown mode '{args.mode}'")
        sys.exit(1)
        
    all_stats = []
    failed_files = []
    
    total_files = len(media_files)
    for i, media_info in enumerate(media_files, 1):
        file_path_str = media_info.get('path', 'Unknown File')
        file_name = Path(file_path_str).name
        
        logger.info(f"\nProcessing file {i}/{total_files}: {file_name}")
        
        try:
            # Pass the complete media_info dictionary, not just the path!
            metadata = pipeline.process(media_info)
            all_stats.append({"file": file_path_str, "status": "success", "metadata": metadata})
        except Exception as e:
            logger.error(f"Failed to process {file_path_str}: {e}", exc_info=True)
            failed_files.append(file_path_str)
            all_stats.append({"file": file_path_str, "status": "failed", "error": str(e)})
            
    # Summary
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
        
    # Step 1: Discover all media files ONCE at the beginning.
    discovery = MediaDiscovery()
    media_files = discovery.discover(args.input)

    if not media_files:
        logger.error(f"No valid media files found in the specified paths: {', '.join(args.input)}")
        sys.exit(1)

    logger.info(f"Found {len(media_files)} media file(s) to process")

    # Step 2: Process the discovered files by passing the rich metadata.
    try:
        process_files(media_files, args)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()