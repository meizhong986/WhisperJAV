#!/usr/bin/env python3
"""
FIXED English Subtitle Sanitizer Test Script
Fixes the comparison logic to properly show what was changed

Usage: python test_english_sanitizer_fixed.py input.srt [options]
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import pysrt
import codecs
from datetime import datetime
import re

# Handle UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, TypeError):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Import the English sanitizer
try:
    from whisperjav.modules.subtitle_sanitizer_english import SimpleEnglishSanitizer, ENGLISH_CONSTANTS
except ImportError:
    print("Error: Could not import SimpleEnglishSanitizer")
    print("Make sure subtitle_sanitizer_english.py is in whisperjav/modules/")
    sys.exit(1)

class EnglishSanitizerTester:
    """Test harness for the English subtitle sanitizer with FIXED reporting"""
    
    def __init__(self, custom_constants: Optional[Dict[str, Any]] = None):
        self.custom_constants = custom_constants or ENGLISH_CONSTANTS.copy()
        self.stats = {
            'hallucinations_removed': 0,
            'high_cps_removed': 0,
            'repetitions_cleaned': 0,
            'duplicates_merged': 0,
            'timing_adjusted': 0,
            'empty_removed': 0
        }
    
    def analyze_artifacts(self, log_path: Path) -> Dict[str, Any]:
        """Parse the artifact log to extract detailed statistics"""
        stats = self.stats.copy()
        removed_items = []
        cleaned_items = []
        merged_items = []
        timing_items = []
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse different types of changes with details
            for line in content.split('\n'):
                if '[REMOVED]' in line:
                    if 'High CPS detected' in line:
                        stats['high_cps_removed'] += 1
                    elif 'hallucination' in line:
                        stats['hallucinations_removed'] += 1
                    else:
                        stats['empty_removed'] += 1
                    removed_items.append(line)
                    
                elif '[CLEANED]' in line:
                    stats['repetitions_cleaned'] += 1
                    cleaned_items.append(line)
                    
                elif '[MERGED]' in line:
                    stats['duplicates_merged'] += 1
                    merged_items.append(line)
                    
                elif '[TIMING ADJUSTED]' in line:
                    stats['timing_adjusted'] += 1
                    timing_items.append(line)
            
            # Extract merge details
            merge_pattern = r'\[MERGED\] Subs #(\d+) through #(\d+)'
            merges = re.findall(merge_pattern, content)
            if merges:
                total_subs_merged = sum(int(m[1]) - int(m[0]) + 1 for m in merges)
                stats['total_subs_in_merges'] = total_subs_merged
                
            # Store detailed items for reporting
            stats['removed_items'] = removed_items[:10]  # First 10
            stats['cleaned_items'] = cleaned_items[:10]
            stats['merged_items'] = merged_items[:10]
            stats['timing_items'] = timing_items[:10]
                
        except Exception as e:
            logging.warning(f"Could not analyze artifacts: {e}")
            
        return stats
    
    def create_test_report(self, input_path: Path, output_path: Path, 
                          sanitizer: SimpleEnglishSanitizer,
                          start_time: datetime, end_time: datetime) -> str:
        """Create a detailed test report with FIXED comparison logic"""
        
        # Load original and sanitized files
        try:
            original_subs = list(pysrt.open(str(input_path), encoding='utf-8'))
            sanitized_subs = list(pysrt.open(str(output_path), encoding='utf-8'))
        except Exception as e:
            return f"Error loading files for comparison: {e}"
        
        # Get artifact statistics
        log_path = output_path.parent / f"{output_path.stem.replace('_sanitized', '')}_sanitizer.log"
        stats = self.analyze_artifacts(log_path) if log_path.exists() else self.stats
        
        # Calculate CPS statistics
        original_cps_stats = self._calculate_cps_stats(original_subs)
        sanitized_cps_stats = self._calculate_cps_stats(sanitized_subs)
        
        # Build report
        report = []
        report.append("=" * 70)
        report.append("ENGLISH SUBTITLE SANITIZER TEST REPORT")
        report.append("=" * 70)
        report.append(f"Input file: {input_path.name}")
        report.append(f"Output file: {output_path.name}")
        report.append(f"Processing time: {(end_time - start_time).total_seconds():.2f} seconds")
        report.append(f"Hallucination phrases loaded: {len(sanitizer.hallucination_phrases)}")
        
        report.append("\n📊 OVERALL STATISTICS:")
        report.append(f"   Original subtitles: {len(original_subs)}")
        report.append(f"   Final subtitles: {len(sanitized_subs)}")
        report.append(f"   Total removed: {len(original_subs) - len(sanitized_subs)}")
        report.append(f"   Reduction: {((len(original_subs) - len(sanitized_subs)) / len(original_subs) * 100):.1f}%")
        
        report.append("\n🎯 CONTENT CLEANING BREAKDOWN:")
        report.append(f"   Hallucinations removed: {stats['hallucinations_removed']}")
        report.append(f"   High CPS lines removed: {stats['high_cps_removed']}")
        report.append(f"   Empty lines removed: {stats['empty_removed']}")
        report.append(f"   Lines with repetitions cleaned: {stats['repetitions_cleaned']}")
        if 'total_subs_in_merges' in stats:
            report.append(f"   Consecutive duplicates merged: {stats['total_subs_in_merges']} → {stats['duplicates_merged']} lines")
        else:
            report.append(f"   Merge operations: {stats['duplicates_merged']}")
        report.append(f"   Timing adjustments: {stats['timing_adjusted']}")
        
        report.append("\n📈 CPS (Characters Per Second) ANALYSIS:")
        report.append("   Original file:")
        report.append(f"     Average CPS: {original_cps_stats['avg']:.1f}")
        report.append(f"     Min CPS: {original_cps_stats['min']:.1f}")
        report.append(f"     Max CPS: {original_cps_stats['max']:.1f}")
        report.append(f"     Lines > {self.custom_constants['MAX_SAFE_CPS']} CPS: {original_cps_stats['high_cps_count']}")
        report.append("   Sanitized file:")
        report.append(f"     Average CPS: {sanitized_cps_stats['avg']:.1f}")
        report.append(f"     Min CPS: {sanitized_cps_stats['min']:.1f}")
        report.append(f"     Max CPS: {sanitized_cps_stats['max']:.1f}")
        report.append(f"     Lines > {self.custom_constants['MAX_SAFE_CPS']} CPS: {sanitized_cps_stats['high_cps_count']}")
        
        # Show samples of changes
        if 'removed_items' in stats and stats['removed_items']:
            report.append("\n🗑️ SAMPLE OF REMOVED CONTENT:")
            for item in stats['removed_items'][:5]:
                # Extract just the relevant part
                match = re.search(r"Sub #(\d+):.*Text: '([^']+)'", item)
                if match:
                    report.append(f"   • Sub #{match.group(1)}: {match.group(2)}")
        
        if 'cleaned_items' in stats and stats['cleaned_items']:
            report.append("\n✏️ SAMPLE OF CLEANED REPETITIONS:")
            for item in stats['cleaned_items'][:3]:
                # Extract the sub number
                match = re.search(r"Sub #(\d+):", item)
                if match:
                    sub_num = match.group(1)
                    # Look for the original and cleaned text in the next lines
                    report.append(f"   • Sub #{sub_num} had repetitions cleaned")
        
        report.append("\n⚙️ CONFIGURATION:")
        report.append(f"   Min safe CPS: {self.custom_constants['MIN_SAFE_CPS']}")
        report.append(f"   Max safe CPS: {self.custom_constants['MAX_SAFE_CPS']}")
        report.append(f"   Target CPS: {self.custom_constants['TARGET_CPS']}")
        report.append(f"   Min subtitle duration: {self.custom_constants['MIN_SUBTITLE_DURATION_S']}s")
        report.append(f"   Max subtitle duration: {self.custom_constants['MAX_SUBTITLE_DURATION_S']}s")
        
        report.append("\n📁 GENERATED FILES:")
        report.append(f"   ✅ Sanitized SRT: {output_path}")
        if log_path.exists():
            report.append(f"   📋 Detailed log: {log_path}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def _calculate_cps_stats(self, subtitles: List[pysrt.SubRipItem]) -> Dict[str, float]:
        """Calculate CPS statistics for a subtitle file"""
        cps_values = []
        high_cps_count = 0
        max_cps = self.custom_constants['MAX_SAFE_CPS']
        
        for sub in subtitles:
            text_len = len(sub.text.strip())
            duration_s = sub.duration.ordinal / 1000.0
            if text_len > 0 and duration_s > 0:
                cps = text_len / duration_s
                cps_values.append(cps)
                if cps > max_cps:
                    high_cps_count += 1
        
        if not cps_values:
            return {'avg': 0, 'min': 0, 'max': 0, 'high_cps_count': 0}
        
        return {
            'avg': sum(cps_values) / len(cps_values),
            'min': min(cps_values),
            'max': max(cps_values),
            'high_cps_count': high_cps_count
        }

def show_actual_changes(input_path: Path, output_path: Path, log_path: Path):
    """
    Show what actually changed in a clear way
    """
    try:
        print("\n🔍 WHAT ACTUALLY CHANGED:")
        print("-" * 50)
        
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # Group changes by type
            removed = []
            cleaned = []
            merged = []
            timing = []
            
            i = 0
            while i < len(log_lines):
                line = log_lines[i].strip()
                
                if '[REMOVED]' in line:
                    removed.append(line)
                elif '[CLEANED]' in line:
                    # Get the next lines for original/cleaned
                    details = [line]
                    if i + 1 < len(log_lines) and 'Original:' in log_lines[i + 1]:
                        details.append(log_lines[i + 1].strip())
                        if i + 2 < len(log_lines) and 'Cleaned:' in log_lines[i + 2]:
                            details.append(log_lines[i + 2].strip())
                    cleaned.append('\n    '.join(details))
                    i += len(details) - 1
                elif '[MERGED]' in line:
                    merged.append(line)
                elif '[TIMING ADJUSTED]' in line:
                    timing.append(line)
                
                i += 1
            
            # Show removed
            if removed:
                print(f"\n❌ REMOVED ({len(removed)} subtitles):")
                for item in removed[:5]:  # Show first 5
                    match = re.search(r"Sub #(\d+):.*?Text: '([^']+)'", item)
                    if match:
                        reason = "hallucination" if "hallucination" in item else "high CPS" if "CPS" in item else "empty"
                        print(f"   #{match.group(1)}: \"{match.group(2)}\" ({reason})")
                if len(removed) > 5:
                    print(f"   ... and {len(removed) - 5} more")
            
            # Show cleaned
            if cleaned:
                print(f"\n✏️ CLEANED ({len(cleaned)} subtitles):")
                for item in cleaned[:3]:  # Show first 3
                    print(f"   {item}")
                if len(cleaned) > 3:
                    print(f"   ... and {len(cleaned) - 3} more")
            
            # Show merged
            if merged:
                print(f"\n🔗 MERGED ({len(merged)} operations):")
                for item in merged[:3]:  # Show first 3
                    match = re.search(r"Subs #(\d+) through #(\d+)", item)
                    if match:
                        print(f"   Subtitles #{match.group(1)}-{match.group(2)} merged into one")
                if len(merged) > 3:
                    print(f"   ... and {len(merged) - 3} more")
        
        # Show first few remaining subtitles
        sanitized_subs = list(pysrt.open(str(output_path), encoding='utf-8'))
        if sanitized_subs:
            print(f"\n📝 FIRST FEW REMAINING SUBTITLES:")
            for i in range(min(5, len(sanitized_subs))):
                print(f"   #{i+1}: \"{sanitized_subs[i].text}\"")
                
    except Exception as e:
        print(f"Error showing changes: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Test the English subtitle sanitizer with FIXED comparison logic",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_srt', type=str, help='Input SRT file to sanitize')
    parser.add_argument('--output', '-o', type=str, help='Output SRT file (default: input_sanitized.srt)')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive settings')
    parser.add_argument('--custom-url', type=str, help='Custom hallucination list URL')
    parser.add_argument('--no-external', action='store_true', help='Skip loading external hallucination list')
    parser.add_argument('--custom-cps', type=float, help='Custom max CPS threshold')
    parser.add_argument('--custom-duration', type=float, help='Custom max duration in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    input_path = Path(args.input_srt)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    # Determine output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_sanitized.srt"
    
    print(f"🎬 English Subtitle Sanitizer Test (FIXED)")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print("-" * 50)
    
    try:
        # Configure constants
        constants = ENGLISH_CONSTANTS.copy()
        
        if args.aggressive:
            print("⚡ Using aggressive settings")
            constants['MAX_SAFE_CPS'] = 35.0
            constants['MIN_SAFE_CPS'] = 1.0
            constants['MERGE_MIN_SEQUENCE'] = 2
            
        if args.custom_cps:
            constants['MAX_SAFE_CPS'] = args.custom_cps
            print(f"🎯 Custom max CPS: {args.custom_cps}")
            
        if args.custom_duration:
            constants['MAX_SUBTITLE_DURATION_S'] = args.custom_duration
            print(f"⏱️  Custom max duration: {args.custom_duration}s")
        
        # Initialize tester
        tester = EnglishSanitizerTester(constants)
        
        # Create sanitizer
        print("\n🔧 Initializing English sanitizer...")
        sanitizer = SimpleEnglishSanitizer(constants=constants)
        
        # Process the file
        print(f"\n🧹 Starting sanitization process...")
        start_time = datetime.now()
        
        result_path = sanitizer.sanitize(input_path)
        
        end_time = datetime.now()
        
        # Move to specified output if needed
        if result_path != output_path:
            import shutil
            shutil.move(str(result_path), str(output_path))
        
        # Generate and print test report
        print("\n" + "=" * 70)
        report = tester.create_test_report(
            input_path, output_path, sanitizer, start_time, end_time
        )
        print(report)
        
        # Show actual changes (FIXED comparison)
        log_path = output_path.parent / f"{output_path.stem.replace('_sanitized', '')}_sanitizer.log"
        show_actual_changes(input_path, output_path, log_path)
        
        print("\n✅ Sanitization completed successfully!")
        print(f"\n💡 TIP: The sanitized file has been renumbered after removing subtitles.")
        print(f"   Original subtitle #8 might now be subtitle #1 if #1-7 were removed.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during sanitization: {e}")
        logging.exception("Full error details:")
        return 1

if __name__ == "__main__":
    sys.exit(main())