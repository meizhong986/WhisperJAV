"""The media record: what a later step may need to know about the media file itself (owner W450 2g).

Built once per media file by the pipeline from one ffprobe call, saved as JSON in the run's temp folder, and passed to
the scene detector as the optional argument ``media=`` (detectors that do not use it ignore it). R-SCN uses it to find
black picture in the media file and to put the picture's times on the extracted audio's time base.

Time base (checked 2026-11-02 on a synthetic file, design docs/plans/design/2026-11-02-rscn-design.md section 7):
FFmpeg reports times from the container's start, and WhisperJAV's extracted WAV starts at the sound's first sample
(modules/audio_extraction.py: aresample=async=1 without first_pts). So a container time t is at
t - audio_offset_s in the WAV, with audio_offset_s = sound start - container start.
"""

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperjav.utils.logger import logger

SCHEMA = "whisperjav-media-record-1"
# Codecs that hold a still image; with one frame, or marked as an attached picture, such a stream is a cover image.
STILL_IMAGE_CODECS = {"mjpeg", "png", "bmp", "gif", "webp", "tiff", "jpegls", "ljpeg"}


@dataclass
class MediaRecord:
    path: str
    probed: bool = False
    has_picture: bool = False                  # a real picture stream (a cover image is not one)
    picture_stream: Optional[int] = None       # ordinal among the video streams (for -map 0:v:<n>)
    picture_start_s: Optional[float] = None
    audio_start_s: Optional[float] = None
    container_start_s: Optional[float] = None
    duration_s: Optional[float] = None
    streams: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    schema: str = SCHEMA

    @property
    def audio_offset_s(self) -> float:
        """Sound start - container start (0 when unknown): subtract from a container time to get WAV time."""
        if self.audio_start_s is None:
            return 0.0
        return max(0.0, self.audio_start_s - (self.container_start_s or 0.0))

    def to_wav_time(self, container_s: float) -> float:
        return container_s - self.audio_offset_s

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["audio_offset_s"] = self.audio_offset_s
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "MediaRecord":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("audio_offset_s", None)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def find_ffprobe() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        beside = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
        if beside.is_file():
            return str(beside)
    return shutil.which("ffprobe")


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def probe(media_path: Path) -> MediaRecord:
    """The media record of one file. Never raises: a failed probe gives a record with ``probed=False`` and a note,
    on which the picture is treated as absent."""
    record = MediaRecord(path=str(Path(media_path)))
    ffprobe = find_ffprobe()
    if not ffprobe:
        record.notes.append("ffprobe not found")
        return record
    command = [ffprobe, "-v", "error", "-show_entries",
               "stream=index,codec_type,codec_name,start_time,duration,nb_frames,avg_frame_rate:"
               "stream_disposition=attached_pic:format=start_time,duration,format_name",
               "-of", "json", str(media_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=120)
        data = json.loads(result.stdout or "{}") if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        record.notes.append(f"ffprobe failed: {e}")
        return record
    if data is None:
        record.notes.append(f"ffprobe failed: {(result.stderr or '').strip()[:300]}")
        return record
    record.probed = True
    fmt = data.get("format") or {}
    record.container_start_s = _float(fmt.get("start_time"))
    record.duration_s = _float(fmt.get("duration"))
    video_ordinal = -1
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        entry = {"index": stream.get("index"), "type": kind, "codec": stream.get("codec_name"),
                 "start_s": _float(stream.get("start_time"))}
        if kind == "video":
            video_ordinal += 1
            cover = bool((stream.get("disposition") or {}).get("attached_pic"))
            codec = str(stream.get("codec_name") or "").lower()
            frames = _float(stream.get("nb_frames"))
            if codec in STILL_IMAGE_CODECS and (frames is not None and frames <= 1):
                cover = True
            entry["cover_image"] = cover
            if not cover and not record.has_picture:
                record.has_picture = True
                record.picture_stream = video_ordinal
                record.picture_start_s = entry["start_s"]
        elif kind == "audio" and record.audio_start_s is None:
            record.audio_start_s = entry["start_s"]
        record.streams.append(entry)
    if record.container_start_s is None:
        starts = [s["start_s"] for s in record.streams if s.get("start_s") is not None]
        record.container_start_s = min(starts) if starts else None
    logger.debug(f"Media record: {record.as_dict()}")
    return record
