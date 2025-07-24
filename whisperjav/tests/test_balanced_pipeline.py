#!/usr/bin/env python3
"""Test script for the balanced pipeline implementation."""

from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.utils.logger import setup_logger, logger


def test_balanced_pipeline():
    """Test the balanced pipeline with a sample file."""
    
    # Setup logger
    global logger
    logger = setup_logger("test_balanced", "DEBUG")
    
    # Test media file path - update this to your test file
    test_file = Path("test_media/sample.mp4")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please update the test_file path to point to a valid media file")
        return
    
    # Discover media info
    logger.info("Discovering media information...")
    discovery = MediaDiscovery()
    media_files = discovery.discover([str(test_file)])
    
    if not media_files:
        logger.error("No media files discovered")
        return
        
    media_info = media_files[0]
    logger.info(f"Media info: {json.dumps(media_info, indent=2)}")
    
    # Create pipeline
    logger.info("\nCreating balanced pipeline...")
    pipeline = BalancedPipeline(
        output_dir="./test_output",
        temp_dir="./test_temp",
        keep_temp_files=True,  # Keep for inspection
        model_name="large-v2",
        vad_threshold=0.3,
        vad_chunk_threshold=4.0
    )
    
    # Process file
    try:
        logger.info("\nProcessing file...")
        metadata = pipeline.process(media_info)
        
        # Save metadata for inspection
        metadata_path = Path("./test_output") / f"{media_info['basename']}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\nProcessing completed successfully!")
        logger.info(f"Output SRT: {metadata['output_files']['final_srt']}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Print summary
        summary = metadata['summary']
        logger.info("\nProcessing Summary:")
        logger.info(f"- Total scenes detected: {summary['total_scenes_detected']}")
        logger.info(f"- Scenes processed: {summary['scenes_processed_successfully']}")
        logger.info(f"- Final subtitles: {summary['final_subtitles_refined']}")
        logger.info(f"- Processing time: {summary['total_processing_time_seconds']}s")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    test_balanced_pipeline()