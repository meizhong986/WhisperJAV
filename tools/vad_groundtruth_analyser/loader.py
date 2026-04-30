"""Audio and SRT loaders.

- `load_media_audio` accepts audio or video files; uses ffmpeg for video.
- `parse_srt` is a stdlib-only SRT reader (no pysrt dependency).
- `parse_srt_string` is exposed for testing.

All functions raise informative errors; no silent failures.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .models import GtSegment

logger = logging.getLogger("whisperjav.vad_gt")

# File extensions treated as audio (no ffmpeg extract needed)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

# Matches a single SRT entry: index \n timestamps \n text (possibly multi-line)
# Tolerant of CRLF endings and trailing whitespace; stops at blank line or EOF.
_SRT_ENTRY_RE = re.compile(
    r"""
    (?P<index>\d+)\s*\r?\n                  # 1) entry index
    (?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})  # 2) start timestamp
    \s*-->\s*
    (?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})    # 3) end timestamp
    \s*\r?\n
    (?P<text>.*?)                           # 4) text (lazy, multi-line)
    (?=\r?\n\s*\r?\n|\r?\n*\Z)              # stop at blank line or end-of-string
    """,
    re.DOTALL | re.VERBOSE,
)

_TIMESTAMP_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$"
)


def timestamp_to_sec(ts: str) -> float:
    """Convert SRT timestamp 'HH:MM:SS,mmm' (or '.') to float seconds.

    Accepts 1 or 2-digit hours and 1-3 digit milliseconds.
    Raises ValueError on malformed input.
    """
    m = _TIMESTAMP_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Malformed SRT timestamp: {ts!r}")
    h, mm, ss, ms = m.groups()
    # Zero-pad ms to 3 digits so '.5' means 500 ms, not 5 ms
    ms_padded = ms.ljust(3, "0")
    return int(h) * 3600 + int(mm) * 60 + int(ss) + int(ms_padded) / 1000.0


def parse_srt_string(content: str) -> List[GtSegment]:
    """Parse SRT content from a string. Robust to BOM, CRLF, malformed entries.

    Malformed individual entries are logged and skipped; the rest are returned.
    Returns a list (possibly empty). Never raises on input content alone.
    """
    # Strip UTF-8 BOM if present
    if content.startswith("﻿"):
        content = content[1:]

    segments: List[GtSegment] = []
    for m in _SRT_ENTRY_RE.finditer(content):
        try:
            idx = int(m.group("index"))
            start = timestamp_to_sec(m.group("start"))
            end = timestamp_to_sec(m.group("end"))
            text = m.group("text").strip().replace("\r\n", "\n")
            if end <= start:
                logger.warning(
                    "SRT entry %d has end<=start (%.3f vs %.3f), skipping",
                    idx, end, start,
                )
                continue
            segments.append(GtSegment(index=idx, start_sec=start, end_sec=end, text=text))
        except (ValueError, AttributeError) as e:
            logger.warning("Skipping malformed SRT entry: %s", e)
            continue
    return segments


def parse_srt(srt_path: Path) -> List[GtSegment]:
    """Load and parse an SRT file. Empty file → [].

    Tries utf-8 first, then cp932 (common for JA subs), then latin-1 fallback.
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    content = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp932", "latin-1"):
        try:
            content = srt_path.read_text(encoding=enc)
            logger.debug("Loaded SRT with encoding=%s", enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if content is None:
        raise ValueError(f"Could not decode SRT {srt_path}: {last_err}")

    segments = parse_srt_string(content)
    if not segments:
        logger.warning("SRT file %s yielded 0 segments (empty or malformed)", srt_path)
    return segments


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def _extract_audio_via_ffmpeg(
    media_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    timeout_sec: int = 600,
) -> None:
    """Run ffmpeg to extract mono PCM audio at target sample rate.

    Raises RuntimeError on ffmpeg failure with stderr captured.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-vn",                                 # no video
        "-acodec", "pcm_s16le",                # 16-bit PCM
        "-ar", str(sample_rate),               # target sample rate
        "-ac", "1",                            # mono
        "-loglevel", "error",                  # quiet unless error
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
        )
    except FileNotFoundError as e:
        raise ImportError(
            "ffmpeg is required to extract audio from video files. "
            "Install from https://ffmpeg.org/ and ensure it is on PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffmpeg extraction timed out after {timeout_sec}s for {media_path}"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) for {media_path}: "
            f"{result.stderr.strip()[:500]}"
        )


def load_media_audio(
    media_path: Path,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, int, float]:
    """Load mono float32 audio at target sample rate.

    For audio files: loads directly via soundfile.
    For video files: extracts with ffmpeg into a temp WAV, then loads.

    Returns:
        (audio_data, actual_sample_rate, duration_sec)

    Raises:
        FileNotFoundError: path missing
        ImportError: ffmpeg/soundfile missing
        RuntimeError: extraction failed
        ValueError: audio is empty / malformed
    """
    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to load audio. Install with: pip install soundfile"
        ) from e

    tmp_wav: Path | None = None
    try:
        if _is_audio_file(media_path):
            audio_path = media_path
        else:
            # Create a named tempfile; manually delete on exit (closing is not enough on Windows)
            fd_holder = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            fd_holder.close()
            tmp_wav = Path(fd_holder.name)
            logger.info("Extracting audio from %s to temp WAV", media_path.name)
            _extract_audio_via_ffmpeg(media_path, tmp_wav, sample_rate=sample_rate)
            audio_path = tmp_wav

        audio_data, actual_sr = sf.read(str(audio_path), dtype="float32")

        # Downmix to mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample if soundfile returned non-target rate (can happen for audio files)
        if actual_sr != sample_rate:
            from scipy import signal as _sig  # lazy; scipy is already a cli dep
            n_target = int(len(audio_data) * sample_rate / actual_sr)
            audio_data = _sig.resample(audio_data, n_target).astype(np.float32)
            logger.debug("Resampled %dHz → %dHz", actual_sr, sample_rate)
            actual_sr = sample_rate

        if len(audio_data) == 0:
            raise ValueError(f"Audio file {media_path} is empty")

        duration = len(audio_data) / actual_sr
        return audio_data, actual_sr, duration
    finally:
        if tmp_wav is not None and tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except OSError as e:
                logger.debug("Could not delete temp file %s: %s", tmp_wav, e)
