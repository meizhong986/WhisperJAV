#!/usr/bin/env python
"""
CLI for exporting WhisperJAV metadata as SRT files for DaVinci Resolve.

Converts scene detection (Pass 1 & 2), VAD segments, and subtitles into
multiple SRT files that can be imported as subtitle tracks in Resolve.

Usage:
    python -m scripts.visualization.resolve_export_cli \
        --metadata ./temp/video_master.json \
        --srt ./output/video.ja.whisperjav.srt \
        --output-dir ./resolve_import/ \
        --prefix "video"

Output:
    resolve_import/video_scene_pass1.srt   - Coarse scene boundaries
    resolve_import/video_scene_pass2.srt   - Fine scene boundaries
    resolve_import/video_vad_segments.srt  - VAD speech segments
    resolve_import/video_subtitles.srt     - Original subtitles (copy)
    resolve_import/video_resolve_import.py - Optional automation script
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.visualization.resolve.metadata_to_srt import (
    export_all_layers,
    generate_resolve_import_script,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export WhisperJAV metadata as SRT files for DaVinci Resolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python -m scripts.visualization.resolve_export_cli \\
      --metadata ./temp/video_master.json \\
      --output-dir ./resolve_import/

  # With subtitles and audio path
  python -m scripts.visualization.resolve_export_cli \\
      --metadata ./temp/video_master.json \\
      --srt ./output/video.ja.whisperjav.srt \\
      --audio ./temp/video_extracted.wav \\
      --output-dir ./resolve_import/ \\
      --prefix "my_video" \\
      --generate-script

Output Files:
  {prefix}_scene_pass1.srt   - Coarse scene boundaries (Pass 1)
  {prefix}_scene_pass2.srt   - Fine scene boundaries (Pass 2)
  {prefix}_vad_segments.srt  - VAD speech segments
  {prefix}_subtitles.srt     - Original subtitles (copy)
  {prefix}_resolve_import.py - Resolve automation script (optional)
"""
    )

    parser.add_argument(
        "-m", "--metadata",
        type=Path,
        required=True,
        help="Path to WhisperJAV master metadata JSON file (*_master.json)"
    )

    parser.add_argument(
        "-s", "--srt",
        type=Path,
        default=None,
        help="Path to original subtitle SRT file (optional, will be copied)"
    )

    parser.add_argument(
        "-a", "--audio",
        type=Path,
        default=None,
        help="Path to audio file for Resolve import script (optional)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated SRT files"
    )

    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="whisperjav",
        help="Filename prefix for output files (default: whisperjav)"
    )

    parser.add_argument(
        "--no-duration",
        action="store_true",
        help="Don't include duration info in subtitle text"
    )

    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="Generate DaVinci Resolve Python import script"
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="WhisperJAV_Analysis",
        help="Project name for Resolve import script (default: WhisperJAV_Analysis)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    if args.srt and not args.srt.exists():
        print(f"Warning: SRT file not found: {args.srt}")
        print("         Subtitles will not be included in export.")
        args.srt = None

    if args.audio and not args.audio.exists():
        print(f"Warning: Audio file not found: {args.audio}")
        print("         Audio path will not be included in Resolve script.")
        args.audio = None

    # Print header
    print()
    print("=" * 60)
    print("WhisperJAV -> DaVinci Resolve Export")
    print("=" * 60)
    print()
    print(f"Metadata:   {args.metadata}")
    if args.srt:
        print(f"Subtitles:  {args.srt}")
    if args.audio:
        print(f"Audio:      {args.audio}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prefix:     {args.prefix}")
    print()

    # Export all layers
    print("Exporting layers as SRT files...")
    output_files = export_all_layers(
        metadata_path=args.metadata,
        srt_path=args.srt,
        output_dir=args.output_dir,
        prefix=args.prefix,
        include_duration=not args.no_duration
    )

    # Generate Resolve import script if requested
    if args.generate_script:
        print()
        print("Generating DaVinci Resolve import script...")
        script_content = generate_resolve_import_script(
            output_files=output_files,
            audio_path=args.audio,
            project_name=args.project_name
        )
        script_path = args.output_dir / f"{args.prefix}_resolve_import.py"
        script_path.write_text(script_content, encoding='utf-8')
        print(f"  Created: {script_path.name}")

    # Print summary
    print()
    print("=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print()
    print(f"Generated {len(output_files)} SRT files in: {args.output_dir}")
    print()
    print("To import into DaVinci Resolve:")
    print("  1. Open DaVinci Resolve 20")
    print("  2. Create or open your project")
    print("  3. Right-click in Media Pool > Import Media")
    print("  4. Select all the generated SRT files")
    print("  5. Create a timeline with your video/audio")
    print("  6. Drag each SRT from Media Pool to timeline")
    print("  7. Each SRT creates a new subtitle track")
    print()
    print("Tip: Rename tracks for clarity:")
    print("  - Track 1: 'P1-Coarse' (scene_pass1.srt)")
    print("  - Track 2: 'P2-Fine' (scene_pass2.srt)")
    print("  - Track 3: 'VAD' (vad_segments.srt)")
    print("  - Track 4: 'Subs' (subtitles.srt)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
