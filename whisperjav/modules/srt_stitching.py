#!/usr/bin/env python3
"""SRT stitching module for combining scene transcriptions."""

from pathlib import Path
from typing import List, Tuple
from datetime import timedelta
import srt

from whisperjav.utils.logger import logger


class SRTStitcher:
    """Handles stitching of multiple SRT files with time offset adjustments."""
    
    def __init__(self):
        """Initialize SRT stitcher."""
        pass
    
    def stitch(self, scene_srt_info: List[Tuple[Path, float]], output_path: Path) -> int:
        """
        Stitch multiple scene SRT files into a single output file.
        
        Args:
            scene_srt_info: List of tuples (srt_path, start_time_offset_seconds)
            output_path: Path for the final combined SRT file
            
        Returns:
            Number of subtitles in the final output
        """
        logger.info(f"Combining {len(scene_srt_info)} scene SRT files")
        
        all_subtitles = []
        global_index = 1
        
        # Sort by start time to ensure correct order
        scene_srt_info.sort(key=lambda x: x[1])
        
        for srt_path, start_offset in scene_srt_info:
            if not srt_path.exists() or srt_path.stat().st_size == 0:
                logger.warning(f"Scene SRT file {srt_path.name} is missing or empty. Skipping.")
                continue
            
            try:
                with open(srt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                scene_subs = list(srt.parse(content))
                
                for sub in scene_subs:
                    # Adjust timestamps by adding the scene's start offset
                    new_start = sub.start + timedelta(seconds=start_offset)
                    new_end = sub.end + timedelta(seconds=start_offset)
                    
                    # Ensure end time is after start time
                    if new_end <= new_start:
                        duration_ms = max(100, int((sub.end - sub.start).total_seconds() * 1000))
                        new_end = new_start + timedelta(milliseconds=duration_ms)
                    
                    # Create adjusted subtitle
                    adjusted_sub = srt.Subtitle(
                        index=global_index,
                        start=new_start,
                        end=new_end,
                        content=sub.content,
                        proprietary=sub.proprietary
                    )
                    
                    all_subtitles.append(adjusted_sub)
                    global_index += 1
                    
            except Exception as e:
                logger.error(f"Error processing scene SRT {srt_path.name}: {e}")
                continue
        
        # Write final combined SRT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(all_subtitles))
        
        logger.info(f"Successfully combined {len(all_subtitles)} subtitles into {output_path}")
        return len(all_subtitles)