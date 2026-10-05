"""Audio integrity check (1.9.4, REQ1): does the media's audio look damaged?

Many inputs are web downloads. Some carry damage a video player hides but
transcription does not: a stretch of the audio track that holds no sound
while the video plays on. Until 1.9.4 our extraction joined the audio on
either side of such a hole, so every subtitle after it came out early by the
hole's length (film A of docs/measurements/v1.9.4: five 2-second holes, subtitles up to 10 s early by
the last hour). Since 1.9.4 the extraction fills each hole with silence
(``audio_extraction.py``), so the times stay right; the sound in the hole is
still missing from the file, which is why the file is still reported.

This module only detects and describes. It never repairs, never stops a run
and never decides an outcome; the caller decides what to do with the report.
It prefers missing a problem to raising a false alarm: every threshold below
is a starting value, chosen so that ordinary files stay silent.

What is checked, and what each check costs:
  1. The stream list (ffprobe, header only): is there an audio track at all,
     and how long do the audio and video tracks say they are.
  2. The audio packet list (ffprobe, no decoding): holes between packets, holes
     hidden inside a packet whose stated length is far longer than the others
     (the film A of docs/measurements/v1.9.4 form, which the plain "gap between packets" rule misses),
     and timestamps that run backwards. Measured: 2.5-5 s on a 2-hour film when
     it runs after the extraction, as the file is then already read from disk.
  3. The extracted audio's length against the audio track's own timeline.
     Free: both numbers are known once the extraction has run.
  4. Errors FFmpeg printed while decoding the audio during the extraction.
     Free: the extraction already decodes every packet.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from whisperjav.utils.logger import logger

# Starting values (see the module docstring). A hole or a backwards jump this
# long moves every later subtitle by about as much, which users notice.
GAP_WARN_S = 0.5
BACKWARDS_WARN_S = 0.5
# The extracted audio may differ from the track's timeline by codec padding
# and rounding; a second is far above that.
LENGTH_WARN_S = 1.0
# Audio and video tracks of a sound file often end a little apart.
AV_LENGTH_WARN_S = 3.0
# Decoder errors: each damaged packet usually prints two lines.
DECODE_ERROR_LINES_WARN = 1
# How many individual places a report names before it summarises the rest.
MAX_PLACES = 5


class AudioIntegrityStop(RuntimeError):
    """Raised when a damaged audio track stops a file (--fail-on suspect)."""


@dataclass
class IntegrityFinding:
    kind: str                     # no_audio | hole | backwards | length | av_length | decode_errors
    detail: str                   # plain-language sentence
    at_s: Optional[float] = None  # where in the media, when there is one place
    size_s: float = 0.0           # how much time is involved


@dataclass
class IntegrityReport:
    findings: List[IntegrityFinding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # the measured values, for logs and the manifest
    seconds: float = 0.0                       # time the check added after the extraction
    checked: bool = True                       # False when ffprobe was missing or failed

    @property
    def suspect(self) -> bool:
        return bool(self.findings)

    def summary(self) -> str:
        """One line naming every finding, for the run summary."""
        if not self.findings:
            return ""
        return "audio integrity: " + "; ".join(f.detail for f in self.findings)


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Pure checks (no subprocesses; unit-tested)
# ---------------------------------------------------------------------------

def find_timeline_breaks(
    pts: Sequence[float],
    dur: Sequence[float],
    gap_warn_s: float = GAP_WARN_S,
    backwards_warn_s: float = BACKWARDS_WARN_S,
) -> List[IntegrityFinding]:
    """Holes and backwards jumps in an audio packet list (times in seconds).

    A hole is time the track's timeline covers but no sound fills. It shows in
    two forms: a gap between one packet's end and the next packet's start, or
    one packet that states a length far beyond the usual packet length (its
    stated length covers the gap, so the first rule sees nothing).
    """
    n = min(len(pts), len(dur))
    if n == 0:
        return []
    typical = sorted(dur[:n])[n // 2]
    holes: List[Tuple[float, float]] = []
    backwards: List[Tuple[float, float]] = []
    for i in range(n):
        hidden = dur[i] - typical
        if hidden >= gap_warn_s:
            holes.append((pts[i] + typical, hidden))
        if i + 1 < n:
            step = pts[i + 1] - (pts[i] + dur[i])
            if step >= gap_warn_s:
                holes.append((pts[i] + dur[i], step))
            elif pts[i + 1] <= pts[i] - backwards_warn_s:
                backwards.append((pts[i + 1], pts[i] - pts[i + 1]))
    out: List[IntegrityFinding] = []
    if holes:
        total = sum(h for _, h in holes)
        places = ", ".join(f"{h:.1f} s at {hms(t)}" for t, h in holes[:MAX_PLACES])
        more = f" and {len(holes) - MAX_PLACES} more" if len(holes) > MAX_PLACES else ""
        out.append(IntegrityFinding(
            "hole",
            f"{len(holes)} gap(s) in the audio track, {total:.1f} s in all ({places}{more})",
            at_s=holes[0][0], size_s=total))
    if backwards:
        places = ", ".join(f"{b:.1f} s at {hms(t)}" for t, b in backwards[:MAX_PLACES])
        more = f" and {len(backwards) - MAX_PLACES} more" if len(backwards) > MAX_PLACES else ""
        out.append(IntegrityFinding(
            "backwards",
            f"the audio track's time runs backwards {len(backwards)} time(s) ({places}{more})",
            at_s=backwards[0][0], size_s=max(b for _, b in backwards)))
    return out


def track_timeline_s(pts: Sequence[float], dur: Sequence[float]) -> Optional[float]:
    """Span of the audio track from its packets: first start to last end."""
    n = min(len(pts), len(dur))
    if n == 0:
        return None
    return max(p + d for p, d in zip(pts[:n], dur[:n])) - min(pts[:n])


def count_decode_errors(ffmpeg_stderr: str) -> int:
    """Error lines in FFmpeg output produced with ``-loglevel level+...``."""
    return sum(1 for line in (ffmpeg_stderr or "").splitlines()
               if "[error]" in line or "[fatal]" in line)


def assess(
    facts: dict,
    pts: Sequence[float] = (),
    dur: Sequence[float] = (),
    extracted_s: Optional[float] = None,
    decode_error_lines: int = 0,
) -> List[IntegrityFinding]:
    """All findings from measured values. ``facts`` holds the stream-list values:
    n_audio, audio_dur, video_dur, format_dur (None where unknown)."""
    findings: List[IntegrityFinding] = []
    if facts.get("n_audio") == 0:
        return [IntegrityFinding("no_audio", "the media has no audio track")]

    findings += find_timeline_breaks(pts, dur)
    holes_s = sum(f.size_s for f in findings if f.kind == "hole")

    timeline = track_timeline_s(pts, dur)
    if timeline is None:
        timeline = facts.get("audio_dur") or facts.get("format_dur")
    if extracted_s is not None and timeline:
        diff = extracted_s - timeline
        # Shorter by what the holes add up to: the holes already say it.
        explained = holes_s > 0 and diff < 0 and abs(-diff - holes_s) < LENGTH_WARN_S
        if abs(diff) >= LENGTH_WARN_S and not explained:
            word = "shorter" if diff < 0 else "longer"
            findings.append(IntegrityFinding(
                "length",
                f"the extracted audio is {abs(diff):.1f} s {word} than the audio track's timeline "
                f"({extracted_s:.1f} s against {timeline:.1f} s), so subtitle times may drift",
                size_s=abs(diff)))

    audio_len = facts.get("audio_dur") or timeline
    video_len = facts.get("video_dur") or (facts.get("format_dur") if facts.get("has_video") else None)
    if audio_len and video_len and video_len - audio_len >= AV_LENGTH_WARN_S:
        findings.append(IntegrityFinding(
            "av_length",
            f"the audio track ends {video_len - audio_len:.1f} s before the video "
            f"({hms(audio_len)} against {hms(video_len)})",
            at_s=audio_len, size_s=video_len - audio_len))

    if decode_error_lines >= DECODE_ERROR_LINES_WARN:
        findings.append(IntegrityFinding(
            "decode_errors",
            f"FFmpeg reported {decode_error_lines} error line(s) while decoding the audio; "
            f"parts of the audio may be missing or damaged",
            size_s=0.0))
    return findings


# ---------------------------------------------------------------------------
# Measuring (ffprobe)
# ---------------------------------------------------------------------------

def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_stream_facts(ffprobe: str, media: Union[str, Path]) -> dict:
    import json
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,start_time,duration:format=duration", "-of", "json", str(media)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    d = json.loads(r.stdout or "{}")
    streams = d.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video"]
    a0, v0 = (audio[0] if audio else {}), (video[0] if video else {})
    return {
        "n_audio": len(audio),
        "has_video": bool(video),
        "audio_start": _num(a0.get("start_time")),
        "video_start": _num(v0.get("start_time")),
        "audio_dur": _num(a0.get("duration")),
        "video_dur": _num(v0.get("duration")),
        "format_dur": _num((d.get("format") or {}).get("duration")),
    }


def read_audio_packets(ffprobe: str, media: Union[str, Path]) -> Tuple[List[float], List[float]]:
    """Start time and stated length of every packet of the first audio track.

    Packets flagged for discard (``D``: the lead-in an edit list cuts off, as
    in a stream-copied excerpt) are never played, so they are left out.
    """
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "packet=pts_time,duration_time,flags", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    pts: List[float] = []
    dur: List[float] = []
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or (len(parts) > 2 and "D" in parts[2]):
            continue
        p, q = _num(parts[0]), _num(parts[1])
        if p is not None and q is not None:
            pts.append(p)
            dur.append(q)
    return pts, dur


def _read_media(ffprobe: str, media: Union[str, Path]):
    facts = read_stream_facts(ffprobe, media)
    pts, dur = read_audio_packets(ffprobe, media) if facts.get("n_audio") else ([], [])
    return facts, pts, dur


class IntegrityProbe:
    """Reads the media's stream list and audio packet list in a background
    thread, started just before the extraction so that both read the file in
    the same pass. Read after the extraction, a large film that has left the
    disk cache took up to 87 s; read alongside it, no measurable time.

    ``finish`` waits for the reading, adds the extraction's own results and
    returns the report. Neither method raises: a check that cannot run leaves
    the report unchecked (``checked=False``), which callers must treat as
    "not assessed", never as "fine".
    """

    def __init__(self, ffprobe: Optional[str], media: Union[str, Path]):
        import threading
        self.media = media
        self._started = time.monotonic()
        self._result = None
        self._error: Optional[str] = None if ffprobe else "ffprobe not found"
        self._thread = None
        if ffprobe:
            self._thread = threading.Thread(target=self._read, args=(ffprobe,), daemon=True,
                                            name="audio-integrity-probe")
            self._thread.start()

    def _read(self, ffprobe: str) -> None:
        try:
            self._result = _read_media(ffprobe, self.media)
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._error = str(exc) or type(exc).__name__

    def finish(self, extracted_s: Optional[float], extraction_stderr: str = "") -> IntegrityReport:
        # The reading ran alongside the extraction; what the user waits for is
        # only what is left of it now, plus the assessment.
        self._started = time.monotonic()
        if self._thread is not None:
            self._thread.join()
        report = IntegrityReport()
        if self._result is None:
            report.checked = False
            report.facts["not_checked"] = self._error or "no result"
            logger.debug(f"Audio integrity check could not run on {self.media}: {report.facts['not_checked']}")
        else:
            try:
                facts, pts, dur = self._result
                errors = count_decode_errors(extraction_stderr)
                facts.update(packets=len(pts), track_timeline_s=track_timeline_s(pts, dur),
                             extracted_s=extracted_s, decode_error_lines=errors)
                report.facts = facts
                report.findings = assess(facts, pts, dur, extracted_s, errors)
            except Exception as exc:  # noqa: BLE001 - see the class docstring
                report.checked = False
                report.facts["not_checked"] = str(exc) or type(exc).__name__
        report.seconds = round(time.monotonic() - self._started, 2)
        return report


def check_audio_integrity(
    ffprobe: Optional[str],
    media: Union[str, Path],
    extracted_s: Optional[float],
    extraction_stderr: str = "",
) -> IntegrityReport:
    """Run every check on ``media`` after its audio was extracted (one call;
    the extraction path uses ``IntegrityProbe`` to overlap the reading)."""
    return IntegrityProbe(ffprobe, media).finish(extracted_s, extraction_stderr)


# ---------------------------------------------------------------------------
# What the user sees, and the stop switch
# ---------------------------------------------------------------------------

# Set by main() when --fail-on includes "suspect" (the GUI's "Treat 'suspect'
# files as failures"), so that it reaches every run path, the ensemble's worker
# processes included (they inherit the environment).
STOP_ENV = "WHISPERJAV_STOP_ON_AUDIO_SUSPECT"


def stop_requested() -> bool:
    import os
    return os.environ.get(STOP_ENV) == "1"


def console_message(report: IntegrityReport, media_name: str, stopping: bool = False) -> str:
    """The line printed during the run (wording approved 2026-10-05)."""
    if not report.checked:
        return f"Audio check: not run ({report.facts.get('not_checked', 'unknown reason')})."
    if not report.suspect:
        return f"Audio check: no problems found ({report.seconds:.1f} s)."
    hole = next((f for f in report.findings if f.kind == "hole"), None)
    others = [f.detail for f in report.findings if f is not hole]
    if hole is not None:
        text = f"Audio check: the audio track of {media_name} has " + \
               hole.detail.replace(" in the audio track", "", 1)
        if others:
            text += "; " + "; ".join(others)
        text += (". WhisperJAV filled the gaps with silence, so subtitle times stay in step with the video; "
                 "any speech inside the gaps is missing from the file.")
    else:
        text = f"Audio check: {media_name}: " + "; ".join(others) + "."
    if stopping:
        return text + " This file is stopped before transcription (--fail-on suspect)."
    return text + " Transcription continues."


def apply_to_run(report: Optional[IntegrityReport], degradations: Optional[List[str]],
                 in_summary: bool = True) -> None:
    """Called by a pipeline right after extraction.

    A damaged audio track adds its reason to the pipeline's ``degradations``
    list, which every run path turns into the ``suspect`` outcome with that
    reason. With --fail-on suspect it raises ``AudioIntegrityStop`` instead,
    which ends this file as ``failed``; the run goes on to the next file.
    ``in_summary`` is False for an ensemble's pass 2, which reads the same file
    as pass 1 and would only repeat the reason.
    """
    if report is None or not report.suspect:
        return
    if stop_requested():
        raise AudioIntegrityStop("stopped before transcription: " + report.summary())
    if in_summary and degradations is not None:
        degradations.append(report.summary())
