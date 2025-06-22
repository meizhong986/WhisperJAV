#!/usr/bin/env python3
"""Audio extraction module using FFmpeg."""

import subprocess
from typing import Union
from pathlib import Path
from typing import Optional, Tuple

import shutil
from whisperjav.utils.logger import logger 

class AudioExtractor:
    """Extract audio from media files using FFmpeg."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: str = "mono",
                 audio_codec: str = "pcm_s16le",
                 ffmpeg_path: Optional[str] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_codec = audio_codec
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable in system PATH."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        logger.info(f"Found FFmpeg at: {ffmpeg}")
        return ffmpeg
        
    def extract(self, input_file: Union[str, Path], output_path: Union[str, Path]) -> Tuple[Path, float]:
        """Extract audio from media file.
        
        Returns:
            Tuple of (output_path, duration_seconds)
        """
        input_file = Path(input_file)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting audio from {input_file.name}")
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(input_file),
            "-vn",  # No video
            "-acodec", self.audio_codec,
            "-ar", str(self.sample_rate),
            "-ac", "1" if self.channels == "mono" else "2",
            "-y",  # Overwrite output
            str(output_path)
        ]
        
        try:
            # Run FFmpeg
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            # Get duration
            duration = self._get_audio_duration(output_path)
            
            logger.info(f"Audio extracted successfully: {output_path.name} (duration: {duration:.1f}s)")
            return output_path, duration
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")
            
    def _get_audio_duration(self, audio_file: Path) -> float:
        """Get duration of audio file in seconds."""
        cmd = [
            self.ffmpeg_path,
            "-i", str(audio_file),
            "-hide_banner",
            "-f", "null",
            "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse duration from FFmpeg output
        for line in result.stderr.split('\n'):
            if "Duration:" in line:
                # Extract duration in format: Duration: 00:01:23.45
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                parts = duration_str.split(":")
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
                
        return 0.0