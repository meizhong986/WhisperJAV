#!/usr/bin/env python3
"""Audio extraction module using FFmpeg."""

import subprocess
import time
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
        logger.debug(f"Found FFmpeg at: {ffmpeg}")
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

        # INFO, not debug (#429 group): on a long or slow-disk file this stage can run
        # for minutes with nothing else on screen, so the user needs to see it start.
        logger.info(f"Extracting the audio from {input_file.name}...")

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
            # Run FFmpeg. No timeout: extraction of a long file on a slow or sleeping
            # drive legitimately takes many minutes, and killing it would lose the run.
            started = time.monotonic()
            result = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  encoding='utf-8', errors='replace',
                                  check=True)
            elapsed = time.monotonic() - started

            # Get duration. Never by decoding the extracted file again: ffprobe
            # reads it out of the header, and failing that FFmpeg already printed
            # it while extracting.
            duration = self._get_audio_duration(output_path, result.stderr)

            logger.info(f"Audio ready: {duration:.1f} seconds of audio, "
                        f"extracted in {elapsed:.1f} seconds")
            return output_path, duration

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")

    def _get_audio_duration(self, audio_file: Path, extract_stderr: Optional[str] = None) -> float:
        """Get duration of audio file in seconds.

        The duration lives in the file header. Until v1.9.3 this ran FFmpeg a
        second time with "-f null -", which decodes the whole extracted file from
        end to end and throws the samples away, purely to read one line off the
        front of its log -- a second pass over every byte of audio, on every file,
        in every pipeline. Nothing here decodes any more:

        1. ffprobe reads the header and stops.
        2. If ffprobe is missing, the extraction run's own log is reused: FFmpeg
           prints the source's "Duration:" line before it starts work, so it costs
           nothing to read.

        Args:
            audio_file: The extracted audio file.
            extract_stderr: FFmpeg's output from the extraction run, if this is
                being called straight after one.

        Returns:
            Duration in seconds, or 0.0 if it could not be determined -- the
            same answer this returned on failure before.
        """
        ffprobe = self._find_ffprobe()
        if ffprobe:
            try:
                result = subprocess.run(
                    [ffprobe, "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     str(audio_file)],
                    capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=30)
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
            except (subprocess.SubprocessError, ValueError) as e:
                logger.debug(f"ffprobe could not report the duration: {e}")

        if extract_stderr:
            duration = self._parse_ffmpeg_duration(extract_stderr)
            if duration > 0.0:
                logger.debug("Duration taken from the extraction run's own output")
                return duration

        logger.debug(f"Could not determine the duration of {audio_file.name}")
        return 0.0

    def _find_ffprobe(self) -> Optional[str]:
        """Find ffprobe, which normally sits beside the ffmpeg we already found."""
        beside = Path(self.ffmpeg_path).with_name("ffprobe" + Path(self.ffmpeg_path).suffix)
        if beside.is_file():
            return str(beside)
        return shutil.which("ffprobe")

    @staticmethod
    def _parse_ffmpeg_duration(ffmpeg_output: str) -> float:
        """Read the 'Duration: 00:01:23.45' line out of FFmpeg's own output."""
        for line in ffmpeg_output.split('\n'):
            if "Duration:" not in line:
                continue
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            parts = duration_str.split(":")
            if len(parts) != 3:
                continue
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                # "Duration: N/A" for a stream with no known length.
                continue
        return 0.0
