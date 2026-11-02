"""R-SCN picture evidence: stretches of black picture, found by FFmpeg's blackdetect in the media file and put on the
extracted audio's time base (owner W450 2g; proposal v6 7.3). Design: docs/plans/design/2026-11-02-rscn-design.md
section 7. Confirmation by the audio is in partition.py (it needs the change strength and the speech measure).
"""

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from whisperjav.modules.media_record import MediaRecord

from .settings import RSCNSettings

_BLACK = re.compile(r"black_start:\s*(-?[0-9.]+)\s+black_end:\s*(-?[0-9.]+)")


class PictureError(RuntimeError):
    """Black-picture detection was asked for and could not be done."""


@dataclass
class BlackStretch:
    container_start_s: float
    container_end_s: float
    start_s: float          # WAV time
    end_s: float
    confirmed: bool = False
    why: str = ""

    def as_dict(self):
        return {"container_start_s": round(self.container_start_s, 3), "container_end_s": round(self.container_end_s, 3),
                "start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3), "confirmed": self.confirmed,
                "why": self.why}


def decide(mode: str, media: Optional[MediaRecord]) -> Tuple[bool, str]:
    """Whether to look at the picture for this file, and why (one line for the log and the run record).
    Raises PictureError when the setting is ``on`` and the file has no real picture (W450 2g: an error)."""
    if mode == "off":
        return False, "picture analysis off by setting"
    if media is None:
        if mode == "on":
            raise PictureError("rscn_picture is on, but the scene detector was given no media record for this file")
        return False, "no media record given: picture analysis off for this file"
    if not media.probed:
        if mode == "on":
            raise PictureError(f"rscn_picture is on, but the media file could not be probed ({'; '.join(media.notes)})")
        return False, "the media file could not be probed: picture analysis off for this file"
    if not media.has_picture:
        if mode == "on":
            raise PictureError("rscn_picture is on, but the media file has no picture stream (audio only, or a cover "
                               "image only)")
        return False, "the media file has no picture stream: picture analysis off for this file"
    return True, "picture analysis on"


def black_stretches(media: MediaRecord, s: RSCNSettings, wav_duration_s: float) -> Tuple[List[BlackStretch], float]:
    """Black stretches of the media file's picture, in container time and in WAV time (clipped to the WAV; stretches
    wholly outside it are dropped). Returns them and the seconds taken. Raises PictureError when FFmpeg fails."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PictureError("FFmpeg not found")
    filters = []
    if s.black_analysis_width and int(s.black_analysis_width) > 0:
        filters.append(f"scale={int(s.black_analysis_width)}:-2")
    filters.append(f"blackdetect=d={s.black_min_s:g}:pix_th={s.black_pixel_threshold:g}"
                   f":pic_th={s.black_picture_threshold:g}")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-i", media.path, "-map", f"0:v:{media.picture_stream or 0}",
               "-vf", ",".join(filters), "-an", "-sn", "-dn", "-f", "null", "-"]
    started = time.time()
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        raise PictureError(f"FFmpeg could not be run: {e}") from e
    if result.returncode != 0:
        raise PictureError(f"FFmpeg's black-picture detection failed: {(result.stderr or '').strip()[-400:]}")
    out: List[BlackStretch] = []
    for a, b in _BLACK.findall(result.stderr or ""):
        ca, cb = float(a), float(b)
        wa, wb = media.to_wav_time(ca), media.to_wav_time(cb)
        wa, wb = max(0.0, wa), min(wav_duration_s, wb)
        if wb > wa:
            out.append(BlackStretch(ca, cb, wa, wb))
    return out, time.time() - started
