#!/usr/bin/env python3
"""
WhisperJAV Multi-Layer Visualization CLI

Standalone utility for visualizing WhisperJAV processing outputs.
Generates interactive HTML files showing:
- Layer 1: Waveform (16kHz mono)
- Layer 2: Scene Detection Pass 1 (coarse boundaries)
- Layer 3: Scene Detection Pass 2 (fine boundaries)
- Layer 4: VAD Segments (Silero speech regions)
- Layer 5: SRT Subtitles

Usage:
    python -m scripts.visualization.viz_cli \\
        --audio ./temp/video_extracted.wav \\
        --metadata ./temp/video_master.json \\
        --srt ./output/video.srt \\
        --output ./output/visualization.html

Prerequisites:
    pip install -r scripts/visualization/requirements.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-layer visualization of WhisperJAV processing outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full visualization with all layers (recommended)
  python -m scripts.visualization.viz_cli \\
      --temp-dir ./temp \\
      --basename sample \\
      --srt ./output/sample.ja.whisperjav.srt \\
      --output ./output/sample_viz.html

  # Explicit paths (all files specified manually)
  python -m scripts.visualization.viz_cli \\
      --audio ./temp/sample_extracted.wav \\
      --metadata ./temp/sample_master.json \\
      --srt ./output/sample.ja.whisperjav.srt \\
      --output ./output/sample_viz.html

  # Without subtitles (scenes + waveform only)
  python -m scripts.visualization.viz_cli \\
      --temp-dir ./temp \\
      --basename sample \\
      --output ./output/sample_viz.html
"""
    )

    # Input sources (individual files)
    parser.add_argument(
        "--audio", "-a",
        type=Path,
        help="Path to extracted WAV file (e.g., ./temp/video_extracted.wav)"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        help="Path to master metadata JSON (e.g., ./temp/video_master.json)"
    )
    parser.add_argument(
        "--srt", "-s",
        type=Path,
        help="Path to output SRT file (e.g., ./output/video.ja.whisperjav.srt)"
    )

    # Alternative: auto-discover audio/metadata from temp dir
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="WhisperJAV temp directory (auto-discovers audio and metadata)"
    )
    parser.add_argument(
        "--basename",
        type=str,
        help="Media basename for auto-discovery (e.g., 'sample' for sample_extracted.wav)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output HTML file path"
    )

    # Options
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Visualization title (default: derived from basename)"
    )
    parser.add_argument(
        "--downsample-points",
        type=int,
        default=10000,
        help="Target points for waveform downsampling (default: 10000)"
    )

    args = parser.parse_args()

    # Resolve input paths
    audio_path = args.audio
    metadata_path = args.metadata
    srt_path = args.srt

    # Auto-discovery mode
    if args.temp_dir and args.basename:
        temp_dir = args.temp_dir
        basename = args.basename

        if not audio_path:
            candidate = temp_dir / f"{basename}_extracted.wav"
            if candidate.exists():
                audio_path = candidate
                print(f"[FOUND] Audio: {audio_path}")
            else:
                print(f"[NOT FOUND] Audio: {candidate}")

        if not metadata_path:
            candidate = temp_dir / f"{basename}_master.json"
            if candidate.exists():
                metadata_path = candidate
                print(f"[FOUND] Metadata: {metadata_path}")
            else:
                print(f"[NOT FOUND] Metadata: {candidate}")

    # SRT validation - require explicit path for reliability
    if srt_path:
        if srt_path.exists():
            print(f"[FOUND] SRT: {srt_path}")
        else:
            print(f"[ERROR] SRT file not found: {srt_path}")
            sys.exit(1)
    else:
        print(f"[SKIP] SRT: Not specified (use --srt to include subtitles)")

    # Validate inputs
    if not any([audio_path, metadata_path, srt_path]):
        parser.error("At least one input source is required (--audio, --metadata, or --srt)")

    # Generate title
    title = args.title
    if not title:
        if args.basename:
            title = f"WhisperJAV Visualization - {args.basename}"
        elif metadata_path:
            title = f"WhisperJAV Visualization - {metadata_path.stem.replace('_master', '')}"
        else:
            title = "WhisperJAV Processing Visualization"

    # Run visualization
    try:
        generate_visualization(
            audio_path=audio_path,
            metadata_path=metadata_path,
            srt_path=srt_path,
            output_path=args.output,
            title=title,
            downsample_points=args.downsample_points
        )
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install visualization dependencies:")
        print("    pip install -r scripts/visualization/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def generate_visualization(
    audio_path: Optional[Path],
    metadata_path: Optional[Path],
    srt_path: Optional[Path],
    output_path: Path,
    title: str,
    downsample_points: int = 10000
):
    """
    Generate visualization from input files.

    Args:
        audio_path: Path to extracted WAV file
        metadata_path: Path to master metadata JSON
        srt_path: Path to output SRT file
        output_path: Output HTML file path
        title: Visualization title
        downsample_points: Target points for waveform downsampling
    """
    from .data_loader import load_all_data
    from .waveform_processor import downsample_waveform
    from .plotly_renderer import render_to_html

    import numpy as np

    print("=" * 60)
    print("WhisperJAV Multi-Layer Visualization")
    print("=" * 60)

    # Load all available data
    print("\nLoading data...")
    data = load_all_data(
        audio_path=audio_path,
        metadata_path=metadata_path,
        srt_path=srt_path
    )

    # Process waveform
    if data.waveform is not None and len(data.waveform) > 0:
        print("\nProcessing waveform...")
        time_array, min_envelope, max_envelope = downsample_waveform(
            data.waveform,
            data.sample_rate,
            target_points=downsample_points
        )
    else:
        print("\nNo waveform data - generating placeholder...")
        # Generate time array from duration
        duration = data.duration_seconds or 100.0
        time_array = np.linspace(0, duration, 1000)
        min_envelope = np.zeros(1000)
        max_envelope = np.zeros(1000)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    print("\nGenerating HTML visualization...")
    render_to_html(
        data=data,
        time_array=time_array,
        min_envelope=min_envelope,
        max_envelope=max_envelope,
        output_path=output_path,
        title=title
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print(f"Duration: {data.duration_seconds:.1f}s")
    print(f"Scenes: {len(data.scenes)}")
    if data.scenes:
        pass1_count = len([s for s in data.scenes if s.detection_pass == 1])
        pass2_count = len([s for s in data.scenes if s.detection_pass == 2])
        unknown_count = len([s for s in data.scenes if s.detection_pass is None])
        print(f"  - Pass 1 (coarse): {pass1_count}")
        print(f"  - Pass 2 (fine): {pass2_count}")
        if unknown_count:
            print(f"  - Legacy (no pass info): {unknown_count}")
    print(f"VAD Segments: {len(data.vad_segments)}")
    if data.vad_method:
        print(f"  - Method: {data.vad_method}")
    print(f"Subtitles: {len(data.subtitles)}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
