#!/usr/bin/env python3
"""
Phase 1 Sanitization Test Script
Usage: python test_phase1_sanitizer.py input.srt [--output output.srt] [--config config.json] [--sensitivity balanced]
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('phase1_sanitization.log')
    ]
)

# Import the updated sanitizer
try:
    from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
    from whisperjav.config.sanitization_config import SanitizationConfig
    from whisperjav.utils.logger import logger
except ImportError as e:
    print(f"Error importing WhisperJAV modules: {e}")
    print("Make sure you're running from the correct directory and WhisperJAV is properly installed.")
    sys.exit(1)

def create_test_config(sensitivity: str = "balanced", language: str = "ja") -> SanitizationConfig:
    """Create a test configuration for Phase 1 testing"""
    
    config = SanitizationConfig(
        # Test configuration focused on Phase 1
        sensitivity_mode=sensitivity,
        primary_language=language,
        
        # Enable Phase 1 features
        enable_exact_matching=True,
        enable_repetition_cleaning=True,  # Only extreme cases in Phase 1
        
        # Disable Phase 2+ features for now
        enable_regex_matching=False,
        enable_fuzzy_matching=False,
        enable_cross_subtitle=False,
        
        # Output options for testing
        save_original=True,
        save_artifacts=True,
        preserve_original_file=True,  # Keep original, create new cleaned file
        artifact_detail_level="full",
        create_raw_subs_folder=True,
        
        # Debug options
        verbose=True,
        debug=True
    )
    
    return config

def main():
    parser = argparse.ArgumentParser(
        description="Test Phase 1 sanitization on SRT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_phase1_sanitizer.py test_input.srt
  python test_phase1_sanitizer.py test_input.srt --output cleaned_output.srt
  python test_phase1_sanitizer.py test_input.srt --sensitivity aggressive --language ja
  python test_phase1_sanitizer.py test_input.srt --config custom_config.json
        """
    )
    
    parser.add_argument(
        'input_srt', 
        type=str,
        help='Input SRT file to sanitize'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output SRT file (default: input_file.phase1_cleaned.srt)'
    )
    
    parser.add_argument(
        '--sensitivity', '-s',
        choices=['conservative', 'balanced', 'aggressive'],
        default='balanced',
        help='Sanitization sensitivity level (default: balanced)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='ja',
        help='Primary language for sanitization (default: ja)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Custom configuration JSON file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    input_path = Path(args.input_srt)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.srt':
        print(f"Warning: Input file '{input_path}' doesn't have .srt extension.")
    
    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}.phase1_cleaned.srt"
    
    print(f"📁 Input SRT: {input_path}")
    print(f"📁 Output SRT: {output_path}")
    print(f"⚙️  Sensitivity: {args.sensitivity}")
    print(f"🌐 Language: {args.language}")
    print("-" * 50)
    
    try:
        # Create configuration
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Config file '{config_path}' does not exist.")
                sys.exit(1)
            config = SanitizationConfig.from_file(config_path)
            print(f"📋 Using config file: {config_path}")
        else:
            config = create_test_config(args.sensitivity, args.language)
            print(f"📋 Using default test configuration")
        
        # Override output path in config
        config.preserve_original_file = True
        
        # Initialize sanitizer
        print("\n🔧 Initializing SubtitleSanitizer...")
        sanitizer = SubtitleSanitizer(config)
        
        # Process the file
        print(f"\n🧹 Starting Phase 1 sanitization...")
        print("=" * 60)
        
        result = sanitizer.process(input_path)
        
        print("=" * 60)
        print("✅ Phase 1 sanitization completed!")
        
        # Copy result to specified output path if different
        if result.sanitized_path != output_path:
            import shutil
            shutil.copy2(result.sanitized_path, output_path)
            print(f"📄 Copied result to: {output_path}")
        
        # Print summary
        stats = result.statistics
        print(f"\n📊 SANITIZATION SUMMARY:")
        print(f"   Original subtitles: {stats['original_subtitle_count']}")
        print(f"   Final subtitles: {stats['final_subtitle_count']}")
        print(f"   Total modifications: {stats['total_modifications']}")
        print(f"   Subtitles removed: {stats['removals']}")
        print(f"   Subtitles modified: {stats['modifications']}")
        print(f"   Reduction: {stats['reduction_percentage']:.1f}%")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        # Phase 1 specific stats
        if 'phase1_stats' in stats:
            p1_stats = stats['phase1_stats']
            print(f"\n🎯 PHASE 1 BREAKDOWN:")
            print(f"   Exact hallucination pass 1: {p1_stats['exact_hallucination_pass1_removed']} removed")
            print(f"   Extreme repetition cleaning: {p1_stats['extreme_repetition_cleaned']} cleaned")
            print(f"   Exact hallucination pass 2: {p1_stats['exact_hallucination_pass2_removed']} removed")
            print(f"   Empty subtitles purged: {p1_stats['empty_purged']} purged")
            
            # Hallucination database info
            if p1_stats['hallucination_database_size']:
                print(f"\n🗃️  HALLUCINATION DATABASE:")
                for lang, count in p1_stats['hallucination_database_size'].items():
                    print(f"   {lang}: {count} phrases")
        
        # Generated files
        print(f"\n📁 GENERATED FILES:")
        print(f"   🧹 Cleaned SRT: {output_path}")
        if result.original_backup_path:
            print(f"   💾 Original backup: {result.original_backup_path}")
        if result.artifacts_path:
            print(f"   🔍 Artifacts log: {result.artifacts_path}")
        print(f"   📋 Processing log: phase1_sanitization.log")
        
        print(f"\n🎉 Phase 1 test completed successfully!")
        print(f"You can now compare '{input_path}' with '{output_path}'")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during sanitization: {e}")
        logger.exception("Full error details:")
        return 1

if __name__ == "__main__":
    sys.exit(main())