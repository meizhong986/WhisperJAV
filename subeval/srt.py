"""SRT reading. Every block that cannot be read is reported, never skipped silently.

A block is a group of non-blank lines separated by blank lines. Its first line may be a number (the cue index); the
timing line is "HH:MM:SS,mmm --> HH:MM:SS,mmm" (a "." in place of the "," is accepted, and hours may have one or more
digits). Every following line is text; the lines are joined with "\\n". A block without a readable timing line, or
whose end time is before its start time, is reported as a problem and left out. A timing line inside a block (a
missing blank line) starts a new block there - with the number line just before it, if any - and is reported. A cue
keeps the block's own number from the file (or its position among the read blocks when the block has none).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_TIMING = re.compile(r"^\s*" + _TIME + r"\s*-->\s*" + _TIME)


@dataclass
class Cue:
    """One subtitle line (one SRT block): times in seconds, text as written."""
    start: float
    end: float
    text: str
    number: int  # the block's number in the file; its 1-based position among the read blocks if it has none


@dataclass
class SrtFile:
    path: str
    cues: list[Cue] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_text(text: str, path: str = "<text>") -> SrtFile:
    out = SrtFile(path=path)
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks, cur, start_line = [], [], 0
    for no, line in enumerate(text.split("\n"), 1):
        if line.strip():
            if not cur:
                start_line = no
            cur.append(line)
        elif cur:
            blocks.append((start_line, cur))
            cur = []
    if cur:
        blocks.append((start_line, cur))
    split = []
    for start_line, lines in blocks:  # a timing line after the first one in a block: a missing blank line
        cut = [k for k in range(1, len(lines)) if _TIMING.match(lines[k]) and k > 0
               and not (k == 1 and not _TIMING.match(lines[0]))]
        prev = 0
        for k in cut:
            k0 = k - 1 if lines[k - 1].strip().isdigit() and k - 1 > prev else k
            split.append((start_line + prev, lines[prev:k0]))
            out.problems.append(f"line {start_line + k0}: no blank line before this block; read as a new block")
            prev = k0
        split.append((start_line + prev, lines[prev:]))
    for start_line, lines in split:
        k = 0 if _TIMING.match(lines[0]) else 1
        m = _TIMING.match(lines[k]) if k < len(lines) else None
        if m and k == 1 and not lines[0].strip().isdigit():
            out.problems.append(f"line {start_line}: text before the timing line ignored: {lines[0]!r}")
        if not m:
            out.problems.append(f"line {start_line}: no timing line in block: {lines[:2]!r}")
            continue
        g = m.groups()
        start, end = _seconds(*g[:4]), _seconds(*g[4:])
        if end < start:
            out.problems.append(f"line {start_line}: end {end:.3f} before start {start:.3f}")
            continue
        number = int(lines[0].strip()) if k == 1 and lines[0].strip().isdigit() else len(out.cues) + 1
        out.cues.append(Cue(start=start, end=end, text="\n".join(lines[k + 1:]), number=number))
    return out


def read_srt(path: str | Path) -> SrtFile:
    path = Path(path)
    return parse_srt_text(path.read_text(encoding="utf-8-sig", errors="strict"), str(path))
