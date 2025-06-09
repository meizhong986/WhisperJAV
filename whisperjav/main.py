#!/usr/bin/env python3
"""WhisperJAV - Japanese Adult Video Subtitle Generator

Main entry point for the application.
"""

import argparse
import sys
from pathlib import Path
from typing import List
import json


from whisperjav.utils.logger import setup_logger, logger
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline

__version__ = "1.0.0"

def print_banner():
    """Print application banner."""
    banner = f"""
╔═══════════════════════════════════════════════════╗
║          WhisperJAV v{__version__}                      ║
║   Japanese Adult Video Subtitle Generator         ║
║   Optimized for JAV content transcription         ║
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
  # Process single file
  whisperjav video.mp4
  
  # Process with specific mode
  whisperjav video.mp4 --mode faster
  
  # Process multiple files with wildcards
  whisperjav "*.mp4" --output-dir ./subtitles
  
  # Keep temporary files for debugging
  whisperjav video.mp4 --keep-temp
"""
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        nargs="+",
        help="Input video file(s) or directory. Supports wildcards."
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["balanced", "fast", "faster"],
        default="balanced",
        help="Processing mode (default: balanced)"
    )
    
    # Output options
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
    
    # Processing options
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for turbo mode (default: 16)"
    )
    
    # Post-processing options
    parser.add_argument(
        "--no-remove-hallucinations",
        action="store_true",
        help="Disable hallucination removal"
    )
    
    parser.add_argument(
        "--no-remove-repetitions",
        action="store_true",
        help="Disable repetition removal"
    )
    
    parser.add_argument(
        "--repetition-threshold",
        type=int,
        default=2,
        help="Number of allowed repetitions (default: 2)"
    )
    
    # Logging options
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
    
    # Other options
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

def process_files(files: List[Path], args):
    """Process multiple files with the selected pipeline."""
    # Select pipeline based on mode
    if args.mode == "faster":
        pipeline = FasterPipeline(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            keep_temp_files=args.keep_temp
        )
    else:
        logger.error(f"Mode '{args.mode}' not yet implemented. Only 'faster' mode is available in v1.0")
        sys.exit(1)
        
    # Process each file
    all_stats = []
    failed_files = []
    
    for i, file_path in enumerate(files, 1):
        logger.info(f"\nProcessing file {i}/{len(files)}: {file_path.name}")
        
        try:
            metadata = pipeline.process(str(file_path))
            all_stats.append({
                "file": str(file_path),
                "status": "success",
                "metadata": metadata
            })
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed_files.append(str(file_path))
            all_stats.append({
                "file": str(file_path),
                "status": "failed",
                "error": str(e)
            })
            
    # Summary
    logger.info("\n" + "="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Successful: {len(files) - len(failed_files)}")
    logger.info(f"Failed: {len(failed_files)}")
    
    if failed_files:
        logger.warning("\nFailed files:")
        for file in failed_files:
            logger.warning(f"  - {file}")
            
    # Save statistics if requested
    if args.stats_file:
        with open(args.stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nStatistics saved to: {args.stats_file}")

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    global logger
    logger = setup_logger("whisperjav", args.log_level, args.log_file)
    
    # Print banner
    print_banner()
    
    # Discover media files
    discovery = MediaDiscovery()
    files = discovery.discover_media_files(args.input)
    
    if not files:
        logger.error("No media files found!")
        sys.exit(1)
        
    logger.info(f"Found {len(files)} media file(s) to process")
    
    # Process files
    try:
        process_files(files, args)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()