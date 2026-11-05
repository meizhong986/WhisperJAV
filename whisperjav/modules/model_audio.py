"""The shortest audio WhisperJAV hands to a model, in one place (owner W461, 2026-11-02).

Rule: no ASR model, aligner or timing model is given a piece of audio shorter than ``MIN_MODEL_AUDIO_S`` (0.2 s).

Why 0.2 s: Netflix's Japanese Timed Text Style Guide (section I.9, Duration) sets the shortest subtitle at 500 ms
(12 frames at 24 fps); the owner asked for a floor well below that, with a good margin (W461). It also lies above
the length band that stops the process inside CrispASR's DTW word timing (CRISPASR_ASSERT ``filter_width <
a->ne[2]``, crispasr.cpp:10327 in v0.8.44: a decoded window of 15 mel frames or fewer, about 0.15-0.16 s at 16 kHz;
the #498 crash and the owner's runs of 2026-10-31 and 2026-11-02).

How the rule is kept:
  - VAD groups (subtitle_pipeline/framers/vad_grouped.py): a group shorter than the floor is CUT, not sent
    (``SHORT_VAD_GROUPS = "cut"``). The owner's experience (W463 CL2): VAD groups under 200 ms are usually false
    positives. The other strategy is kept, ready: ``"widen"`` widens such a group to the floor into the real audio
    beside it, within the scene and without reaching into the piece before or after (``widen_frames``); it keeps
    short real speech (a quick 「はい」) at the cost of up to ~50 ms of the surrounding sound and up to ~0.1 s of
    timing shift (W433, W442: real audio is not removed). To change strategy, change ``SHORT_VAD_GROUPS``; the
    framers' own noise floor (0.1 s) applies under either;
  - SRT entries (subtitle_pipeline/framers/srt_source.py) are subtitles, not VAD guesses: always widened, never cut;
  - a piece still shorter than the floor (no room beside it: a scene that short, or neighbours flush against it) is
    not sent, and is counted in a log line: ChronosJAV's orchestrator (subtitle_pipeline/orchestrator.py, before
    its generator; an empty text then also skips its aligner and timing model; a frame that carries its own text
    keeps it) and the CrispASR chain (pipelines/crispasr_pipeline.py, VAD groups and whole scenes);
  - the CrispASR entry points refuse audio shorter than the floor as a last guard (``too_short``): the ASR engine and
    the aligner give no result, the timing model gives no timing, which by rule T3 changes nothing
    (subtitle_pipeline/timing_rules.py).
  Not covered: the legacy pipelines (balanced, fast, faster, fidelity) and XXL, whose Whisper engines pad every
  window to 30 s themselves.

Not covered by a length rule: inside a longer piece, CrispASR's Whisper may still decode a short window of its own
(a window ending at 0.14 s or earlier, or a short tail). That path needs the timing model in a child process or a
fix in CrispASR (W460 answer e).

Light: standard library only.
"""

import math
from typing import Any, List, Sequence, Tuple

MIN_MODEL_AUDIO_S = 0.2
# What a VAD group shorter than MIN_MODEL_AUDIO_S becomes: "cut" (not sent; owner W463 CL2, the default) or "widen"
# (widened to the floor into the audio beside it). See the module docstring.
SHORT_VAD_GROUPS = "cut"
# A piece is widened a little past the floor, so that cutting it at whole samples (int(start * rate)) never falls
# under it.
_WIDEN_MARGIN_S = 0.005


def min_samples(sample_rate: int) -> int:
    """The floor in samples at ``sample_rate``."""
    return int(math.ceil(MIN_MODEL_AUDIO_S * sample_rate - 1e-9))


def too_short(n_samples: int, sample_rate: int) -> bool:
    """True when ``n_samples`` at ``sample_rate`` is shorter than the floor: the piece is not given to a model."""
    return n_samples < min_samples(sample_rate)


def widen_short_spans(spans: Sequence[Tuple[float, float]], bound_start: float, bound_end: float,
                      min_s: float = MIN_MODEL_AUDIO_S) -> List[Tuple[float, float]]:
    """The spans (start, end), in order and not overlapping, with each one shorter than ``min_s`` widened to it.

    A short span grows by the same amount on both sides; where one side has no room (the scene's edge, or the span
    before or after), the other side takes the rest. It never reaches past ``bound_start``/``bound_end`` nor into
    its neighbours, and it never shrinks: a neighbour that overlaps it or lies out of order leaves no room on that
    side. A span with no room around it stays as it is (and is then not sent). Long spans are unchanged.
    """
    target = min_s + _WIDEN_MARGIN_S
    out: List[Tuple[float, float]] = []
    for i, (start, end) in enumerate(spans):
        length = end - start
        if length >= min_s:
            out.append((start, end))
            continue
        # Never past the span's own edges inward: an overlapping or out-of-order neighbour gives no room, no cut.
        left_limit = min(start, max(bound_start, out[-1][1] if out else bound_start))
        right_limit = max(end, min(bound_end, spans[i + 1][0] if i + 1 < len(spans) else bound_end))
        room_left = max(0.0, start - left_limit)
        room_right = max(0.0, right_limit - end)
        need = target - length
        take_left = min(room_left, need / 2)
        take_right = min(room_right, need - take_left)
        take_left = min(room_left, need - take_right)  # what the right side could not take
        # Land exactly on the edge or the neighbour where the room was used up (no float residue).
        new_start = left_limit if take_left >= room_left - 1e-9 else start - take_left
        new_end = right_limit if take_right >= room_right - 1e-9 else end + take_right
        out.append((new_start, new_end))
    return out


def widen_frames(frames: Sequence[Any], bound_end: float) -> int:
    """Widen, in place, each frame (an object with ``start`` and ``end``, seconds from the scene's start) shorter than
    the floor, within 0..``bound_end``; returns how many were widened. The frames are taken in the order given."""
    spans = widen_short_spans([(f.start, f.end) for f in frames], 0.0, bound_end)
    widened = 0
    for f, (start, end) in zip(frames, spans):
        if (start, end) != (f.start, f.end):
            f.start, f.end = start, end
            widened += 1
    return widened
