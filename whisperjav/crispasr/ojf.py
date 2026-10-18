"""CrispASR's two file formats: the cut-point file WhisperJAV writes, and the ``-ojf`` JSON it reads.

One CrispASR writer produces every engine's JSON, so reading it is shared (master plan 3.7, 4.2;
second architecture study 4.1, 4.2, 5.1, 5.5). What each engine's segments, words and tokens MEAN --
which times are native, which are estimated, what must be dropped -- is decided in each pipeline's
own parser, not here.

Facts this module relies on (CrispASR 0.8.43; the second study's runs S1-S9, E1, M1, P1):
- A cut-point file of kind ``chunks`` is used exactly as given: no re-cut, no VAD model
  loaded. Slice ``i`` of the file is ``chunk_id`` ``i`` of the output.
- ``chunk_id`` is omitted when the run had a single slice.
- Times in the output are absolute in the audio file passed with ``-f``. ``offsets`` are in
  milliseconds; ``t0``/``t1`` of words and tokens are in centiseconds.
- The top-level ``crispasr`` object echoes the back end, the ``-m`` value exactly as given, and the
  language.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The program's default --chunk-seconds is 30; writing the same value avoids its mismatch warning.
CHUNK_CS = 3000
SAMPLE_RATE = 16000

# How far a segment's start may lie outside the slice it names and still count as that slice's
# (master plan 4.2).
SLICE_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class CutPoint:
    """One slice WhisperJAV gives the program, in seconds, absolute in the audio passed with -f."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CrispToken:
    text: str
    start: float  # seconds, from offsets (ms)
    end: float
    p: Optional[float] = None


@dataclass(frozen=True)
class CrispWord:
    text: str
    start: float  # seconds, from offsets (ms)
    end: float


@dataclass
class CrispSegment:
    start: float  # seconds, from offsets (ms)
    end: float
    text: str
    chunk_id: int  # 0 when the program omitted it (a run with one slice)
    speaker: Optional[str] = None  # the raw "(Speaker N) " field; no pipeline uses it (W114 X7)
    words: list[CrispWord] = field(default_factory=list)
    tokens: list[CrispToken] = field(default_factory=list)


@dataclass
class CrispTranscript:
    backend: str
    model: str
    language: str
    segments: list[CrispSegment]
    malformed: int = 0  # segments, words or tokens skipped because they had no usable times


@dataclass
class SlicePlacement:
    """Segments tied to the slice they came from (master plan 4.2, 2.6)."""
    placed: list[tuple[int, CrispSegment]]  # (slice index, segment), in the program's order
    retied: int = 0  # chunk_id did not fit, but the segment's start lies in another slice: a note
    dropped: int = 0  # no slice contains it: dropped, and a degradation (Q21)


# ---------------------------------------------------------------------------------------------------
# The cut-point file (--vad-import, kind "chunks")
# ---------------------------------------------------------------------------------------------------

def write_vad_import(path: Path, cuts: list[CutPoint], sample_rate: int = SAMPLE_RATE,
                     audio_seconds: Optional[float] = None) -> None:
    """Write the cut points as a ``chunks`` import file, which the program uses exactly as given.

    ``cuts`` must already follow the rules every pipeline's slice rule guarantees (master plan 3.4):
    sorted, not overlapping, each longer than zero and inside the audio. They are checked here
    rather than repaired, because a slice the program drops would shift every later ``chunk_id``.
    """
    if not cuts:
        raise ValueError("no cut points to write")
    previous_end = 0.0
    for i, c in enumerate(cuts):
        if c.start < 0 or c.end <= c.start:
            raise ValueError(f"cut point {i} is empty or negative: {c.start:.3f}-{c.end:.3f} s")
        if c.start < previous_end:
            raise ValueError(f"cut point {i} starts before cut point {i - 1} ends "
                             f"({c.start:.3f} < {previous_end:.3f} s)")
        if audio_seconds is not None and c.end > audio_seconds + 1e-6:
            raise ValueError(f"cut point {i} ends after the audio ({c.end:.3f} > {audio_seconds:.3f} s)")
        previous_end = c.end

    slices = [{"start": int(round(c.start * sample_rate)), "end": int(round(c.end * sample_rate)),
               "t0_cs": int(round(c.start * 100)), "t1_cs": int(round(c.end * 100))} for c in cuts]
    doc = {"crispasr_vad": {"version": 1, "kind": "chunks", "sample_rate": sample_rate,
                            "chunk_cs": CHUNK_CS, "num_slices": len(slices), "slices": slices}}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# The -ojf JSON
# ---------------------------------------------------------------------------------------------------

def read_ojf(path: Path) -> Optional[dict]:
    """The program's JSON, or None when it is missing, does not parse, or has no transcription list.

    None counts as "no output" (master plan 4.2): a partial or damaged file is never used.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("transcription"), list):
        return None
    return doc


def _seconds(item: dict) -> Optional[tuple[float, float]]:
    """(start, end) in seconds from ``offsets`` (ms); from ``t0``/``t1`` (cs) when offsets are absent."""
    o = item.get("offsets")
    try:
        if isinstance(o, dict) and "from" in o and "to" in o:
            return float(o["from"]) / 1000.0, float(o["to"]) / 1000.0
        if "t0" in item and "t1" in item:
            return float(item["t0"]) / 100.0, float(item["t1"]) / 100.0
    except (TypeError, ValueError):
        return None
    return None


def _text(value) -> str:
    """A JSON text field: null reads as empty, never as the word 'None'."""
    return "" if value is None else str(value)


def parse_ojf(doc: dict) -> CrispTranscript:
    """Turn the program's JSON into segments, words and tokens, keeping everything it wrote.

    Nothing is cleaned here: blank tokens, zero-length tokens, sound labels and speaker tags all
    reach the pipeline, whose own parser decides what to drop (master plan 3.5).
    """
    top = doc.get("crispasr") if isinstance(doc.get("crispasr"), dict) else {}
    segments: list[CrispSegment] = []
    malformed = 0
    for s in doc.get("transcription") or []:
        if not isinstance(s, dict):
            malformed += 1
            continue
        times = _seconds(s)
        if times is None:
            malformed += 1
            continue
        words: list[CrispWord] = []
        for w in s.get("words") or []:
            wt = _seconds(w) if isinstance(w, dict) else None
            if wt is None:
                malformed += 1
                continue
            words.append(CrispWord(_text(w.get("text")), wt[0], wt[1]))
        tokens: list[CrispToken] = []
        for t in s.get("tokens") or []:
            tt = _seconds(t) if isinstance(t, dict) else None
            if tt is None:
                malformed += 1
                continue
            p = t.get("p")
            tokens.append(CrispToken(_text(t.get("text")), tt[0], tt[1],
                                     float(p) if isinstance(p, (int, float)) else None))
        raw_id = s.get("chunk_id", 0)
        # Only a whole number names a slice; anything else (1.5, true, "1") is placed by time instead.
        chunk_id = raw_id if type(raw_id) is int else -1
        speaker = s.get("speaker")
        segments.append(CrispSegment(start=times[0], end=times[1], text=_text(s.get("text")),
                                     chunk_id=chunk_id, speaker=speaker if isinstance(speaker, str) else None,
                                     words=words, tokens=tokens))
    return CrispTranscript(backend=str(top.get("backend", "")), model=str(top.get("model", "")),
                           language=str(top.get("language", "")), segments=segments, malformed=malformed)


def header_mismatch(transcript: CrispTranscript, backend: str, model: str,
                    language: Optional[str] = None) -> Optional[str]:
    """Why this JSON is not the requested run's output, or None when it is.

    The program echoes ``--backend``, the ``-m`` value as given and ``-l``. A JSON naming another back
    end, model or language counts as no output (master plan 4.2). ``language`` None skips that check.
    """
    if transcript.backend != backend:
        return f"the JSON names back end '{transcript.backend}', not '{backend}'"
    if language is not None and transcript.language.strip().lower() != language.strip().lower():
        return f"the JSON names language '{transcript.language}', not '{language}'"

    def norm(p: str) -> str:
        return os.path.normcase(os.path.normpath(p)) if p else ""

    if norm(transcript.model) != norm(model):
        return f"the JSON names model '{transcript.model}', not '{model}'"
    return None


def place_in_slices(segments: list[CrispSegment], cuts: list[CutPoint],
                    tolerance: float = SLICE_TOLERANCE_S) -> SlicePlacement:
    """Tie each segment to the slice it came from (master plan 4.2).

    A segment belongs to slice ``chunk_id`` when that index exists and the segment's start lies
    inside the slice (within ``tolerance``). Otherwise it is tied by its start time and counted as
    re-tied; when no slice contains its start it is dropped and counted.
    """
    result = SlicePlacement(placed=[])

    def inside(i: int, t: float) -> bool:
        return cuts[i].start - tolerance <= t <= cuts[i].end + tolerance

    for seg in segments:
        if 0 <= seg.chunk_id < len(cuts) and inside(seg.chunk_id, seg.start):
            result.placed.append((seg.chunk_id, seg))
            continue
        by_time = next((i for i in range(len(cuts)) if inside(i, seg.start)), None)
        if by_time is None:
            result.dropped += 1
        else:
            result.retied += 1
            result.placed.append((by_time, seg))
    return result
