"""
Convert WhisperJAV metadata to SRT files for DaVinci Resolve.

Transforms scene detection, VAD segments, and other metadata layers
into SRT subtitle format for multi-track visualization in Resolve.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SrtEntry:
    """Single SRT subtitle entry."""
    index: int
    start_sec: float
    end_sec: float
    text: str


def format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_duration(seconds: float) -> str:
    """Format duration for display (e.g., '29.5s' or '1m 30s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def entries_to_srt(entries: List[SrtEntry]) -> str:
    """
    Convert list of SRT entries to SRT file content.

    Args:
        entries: List of SrtEntry objects

    Returns:
        Complete SRT file content as string
    """
    lines = []
    for entry in entries:
        lines.append(str(entry.index))
        start_ts = format_srt_timestamp(entry.start_sec)
        end_ts = format_srt_timestamp(entry.end_sec)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(entry.text)
        lines.append("")  # Blank line between entries
    return "\n".join(lines)


def convert_scenes_to_srt(
    scenes: List[Dict],
    pass_num: int,
    include_duration: bool = True
) -> str:
    """
    Convert scene detection data to SRT format.

    Args:
        scenes: List of scene dictionaries with start/end times
        pass_num: Pass number (1 for coarse, 2 for fine)
        include_duration: Whether to include duration in text

    Returns:
        SRT file content as string
    """
    entries = []

    for i, scene in enumerate(scenes):
        start = scene.get("start_time_seconds", 0.0)
        end = scene.get("end_time_seconds", 0.0)
        duration = end - start

        # Generate label
        if pass_num == 1:
            # Pass 1: A, B, C, etc.
            label = chr(65 + i) if i < 26 else f"S{i}"
        else:
            # Pass 2: A.1, A.2, B.1, etc.
            parent = chr(65 + (i // 10)) if i < 260 else str(i // 10)
            sub = (i % 10) + 1
            label = f"{parent}.{sub}"

        # Build text
        text_parts = [f"[{label}] Scene {i + 1}"]
        if include_duration:
            text_parts.append(f"| {format_duration(duration)}")

        entries.append(SrtEntry(
            index=i + 1,
            start_sec=start,
            end_sec=end,
            text=" ".join(text_parts)
        ))

    return entries_to_srt(entries)


def convert_vad_to_srt(
    vad_segments: List[Dict],
    include_duration: bool = True
) -> str:
    """
    Convert VAD segments to SRT format.

    Args:
        vad_segments: List of VAD segment dictionaries
        include_duration: Whether to include duration in text

    Returns:
        SRT file content as string
    """
    entries = []

    for i, seg in enumerate(vad_segments):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", 0.0)
        duration = end - start

        # Build text
        text_parts = [f"SPEECH #{i + 1}"]
        if include_duration:
            text_parts.append(f"| {format_duration(duration)}")

        entries.append(SrtEntry(
            index=i + 1,
            start_sec=start,
            end_sec=end,
            text=" ".join(text_parts)
        ))

    return entries_to_srt(entries)


def load_metadata(metadata_path: Path) -> Dict:
    """Load WhisperJAV master metadata JSON."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_all_layers(
    metadata_path: Path,
    srt_path: Optional[Path],
    output_dir: Path,
    prefix: str = "whisperjav",
    include_duration: bool = True
) -> Dict[str, Path]:
    """
    Export all visualization layers as SRT files.

    Args:
        metadata_path: Path to _master.json metadata file
        srt_path: Path to original subtitle SRT file (optional)
        output_dir: Directory to write output SRT files
        prefix: Filename prefix for output files
        include_duration: Whether to include duration info in text

    Returns:
        Dictionary mapping layer names to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Export Pass 1 (coarse boundaries)
    coarse = metadata.get("coarse_boundaries", [])
    if coarse:
        content = convert_scenes_to_srt(coarse, pass_num=1, include_duration=include_duration)
        path = output_dir / f"{prefix}_scene_pass1.srt"
        path.write_text(content, encoding='utf-8')
        output_files["scene_pass1"] = path
        print(f"  Created: {path.name} ({len(coarse)} scenes)")
    else:
        print(f"  Skipped: scene_pass1.srt (no coarse_boundaries in metadata)")

    # Export Pass 2 (fine scenes)
    scenes = metadata.get("scenes_detected", [])
    if scenes:
        content = convert_scenes_to_srt(scenes, pass_num=2, include_duration=include_duration)
        path = output_dir / f"{prefix}_scene_pass2.srt"
        path.write_text(content, encoding='utf-8')
        output_files["scene_pass2"] = path
        print(f"  Created: {path.name} ({len(scenes)} scenes)")
    else:
        print(f"  Skipped: scene_pass2.srt (no scenes_detected in metadata)")

    # Export VAD segments
    vad_segments = metadata.get("vad_segments", [])
    if vad_segments:
        content = convert_vad_to_srt(vad_segments, include_duration=include_duration)
        path = output_dir / f"{prefix}_vad_segments.srt"
        path.write_text(content, encoding='utf-8')
        output_files["vad_segments"] = path
        print(f"  Created: {path.name} ({len(vad_segments)} segments)")
    else:
        print(f"  Skipped: vad_segments.srt (no vad_segments in metadata)")

    # Copy original subtitles
    if srt_path and srt_path.exists():
        dest_path = output_dir / f"{prefix}_subtitles.srt"
        shutil.copy2(srt_path, dest_path)
        output_files["subtitles"] = dest_path
        print(f"  Copied:  {dest_path.name} (original subtitles)")
    elif srt_path:
        print(f"  Skipped: subtitles.srt (file not found: {srt_path})")

    return output_files


def generate_resolve_import_script(
    output_files: Dict[str, Path],
    audio_path: Optional[Path] = None,
    project_name: str = "WhisperJAV_Analysis"
) -> str:
    """
    Generate a Python script for DaVinci Resolve automation.

    Args:
        output_files: Dictionary of layer names to file paths
        audio_path: Optional path to audio file
        project_name: Name for the Resolve project

    Returns:
        Python script content as string
    """
    srt_paths = [str(p.resolve()) for p in output_files.values()]
    audio_str = f'"{audio_path.resolve()}"' if audio_path else "None"

    script = f'''#!/usr/bin/env python
"""
DaVinci Resolve Import Script - Generated by WhisperJAV
Project: {project_name}

SETUP INSTRUCTIONS:
1. Open DaVinci Resolve 20
2. Ensure scripting is enabled (Preferences > System > General > External scripting)
3. Set PYTHONPATH to include Resolve's scripting modules:
   - Windows: %PROGRAMDATA%\\Blackmagic Design\\DaVinci Resolve\\Support\\Developer\\Scripting\\Modules
   - Mac: /Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules
4. Run this script with: python resolve_import.py

MANUAL ALTERNATIVE:
1. Open DaVinci Resolve
2. Import the following SRT files to Media Pool:
{chr(10).join(f'   - {p}' for p in srt_paths)}
3. Create a timeline with the audio/video
4. Drag each SRT file to create subtitle tracks
"""

import sys

try:
    import DaVinciResolveScript as dvr_script
except ImportError:
    print("=" * 60)
    print("DaVinci Resolve Scripting API not found!")
    print()
    print("Please set PYTHONPATH to include Resolve's scripting modules:")
    print("  Windows: %PROGRAMDATA%\\\\Blackmagic Design\\\\DaVinci Resolve\\\\Support\\\\Developer\\\\Scripting\\\\Modules")
    print("  Mac: /Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules")
    print()
    print("Or manually import the SRT files into DaVinci Resolve.")
    print("=" * 60)
    sys.exit(1)


def main():
    project_name = "{project_name}"
    audio_path = {audio_str}
    srt_paths = {srt_paths}

    # Connect to Resolve
    resolve = dvr_script.scriptapp("Resolve")
    if not resolve:
        print("Please open DaVinci Resolve first.")
        return

    project_manager = resolve.GetProjectManager()
    media_storage = resolve.GetMediaStorage()

    # Create or load project
    project = project_manager.CreateProject(project_name)
    if not project:
        print(f"Project '{{project_name}}' may exist. Loading it...")
        project = project_manager.LoadProject(project_name)

    if not project:
        print("Unable to create or load project.")
        return

    media_pool = project.GetMediaPool()

    # Import files to Media Pool
    files_to_import = srt_paths.copy()
    if audio_path:
        files_to_import.insert(0, audio_path)

    print(f"Importing {{len(files_to_import)}} files to Media Pool...")
    added_items = media_storage.AddItemListToMediaPool(files_to_import)

    if added_items:
        print(f"Successfully imported {{len(added_items)}} items.")
    else:
        print("Import may have failed. Check Media Pool manually.")

    print()
    print("=" * 60)
    print("NEXT STEPS (Manual):")
    print("1. Create a timeline from your audio/video clip")
    print("2. Drag each SRT file from Media Pool to the timeline")
    print("3. Each SRT will create a new subtitle track")
    print("4. Rename tracks for clarity: P1-Scene, P2-Fine, VAD, Subs")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
    return script
