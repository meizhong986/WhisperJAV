#!/usr/bin/env python3
"""
Subtitle Diagnostic Tool
Helps debug what's in your subtitle files
"""

import pysrt
from pathlib import Path
import sys

def diagnose_srt(file_path: Path):
    """Show detailed info about an SRT file"""
    print(f"\n📄 Diagnosing: {file_path}")
    print("=" * 60)
    
    try:
        subs = list(pysrt.open(str(file_path), encoding='utf-8'))
        print(f"Total subtitles: {len(subs)}")
        
        # Show first 10 and last 5
        print("\n🔍 FIRST 10 SUBTITLES:")
        for i, sub in enumerate(subs[:10]):
            print(f"  #{sub.index} [{sub.start} --> {sub.end}]: \"{sub.text}\"")
        
        if len(subs) > 15:
            print("\n  ...")
            print("\n🔍 LAST 5 SUBTITLES:")
            for sub in subs[-5:]:
                print(f"  #{sub.index} [{sub.start} --> {sub.end}]: \"{sub.text}\"")
        elif len(subs) > 10:
            print("\n🔍 REMAINING SUBTITLES:")
            for sub in subs[10:]:
                print(f"  #{sub.index} [{sub.start} --> {sub.end}]: \"{sub.text}\"")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_srt.py file1.srt [file2.srt ...]")
        sys.exit(1)
    
    for arg in sys.argv[1:]:
        diagnose_srt(Path(arg))