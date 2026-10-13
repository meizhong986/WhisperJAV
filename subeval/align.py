"""Character alignment of a test character stream to a reference character stream.

Each character carries the subtitle line it came from; each line has a start and end time. The alignment is an
edit-distance alignment (Levenshtein: substitution, deletion and insertion each cost 1) with two additions:

1. Time window. A reference character may be paired with a test character (as a hit or a substitution) only when
   the time gap between their two lines is at most `window` seconds. The gap between [a1, a2] and [b1, b2] is
   max(0, max(a1, b1) - min(a2, b2)): zero when the lines overlap.
2. Tie-breaking by time. Among all alignments with the fewest edits, the one with the smallest sum of gaps (in ms)
   over its paired characters is chosen; among those, the smallest sum of distances between line midpoints (in ms).
   This is an exact lexicographic minimum, computed with integer weights that cannot overflow into each other.
   Remaining ties are broken by a fixed move order: pair > reference character missing > test character extra,
   applied in the forward direction ("forward") or, as a check, on both streams reversed ("mirror").

Result: a list of operations in stream order, each (kind, i, j): kind "hit" (same character), "wrong"
(substitution), "missing" (reference character with no partner, j None) or "extra" (test character with no
partner, i None); i and j index the reference and test streams.
"""
from __future__ import annotations

PAIR, UP, LEFT = 1, 2, 3


def gap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


def _pair_table(ref_lines, test_lines, window):
    """Per (reference line, test line): None if pairing is not allowed, else (gap ms, midpoint distance ms)."""
    table = []
    for a in ref_lines:
        row = []
        ma = (a[0] + a[1]) / 2
        for b in test_lines:
            g = gap_seconds(a, b)
            if g > window + 1e-9:
                row.append(None)
            else:
                row.append((round(g * 1000), round(abs(ma - (b[0] + b[1]) / 2) * 1000)))
        table.append(row)
    return table


def _align(rc, rl, tc, tl, table):
    n, m = len(rc), len(tc)
    k = min(n, m)
    gmax = max((c[0] for row in table for c in row if c), default=0)
    mmax = max((c[1] for row in table for c in row if c), default=0)
    MS = mmax * k + 1
    E = (gmax * MS + mmax) * k + 1
    cost = [[None if c is None else c[0] * MS + c[1] for c in row] for row in table]
    prev = [j * E for j in range(m + 1)]
    back = [bytearray([LEFT]) * (m + 1)]
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        cur[0] = i * E
        bi = bytearray(m + 1)
        bi[0] = UP
        ch = rc[i - 1]
        crow = cost[rl[i - 1]]
        for j in range(1, m + 1):
            best = prev[j] + E
            mv = UP
            pc = crow[tl[j - 1]]
            if pc is not None:
                d = prev[j - 1] + pc + (0 if ch == tc[j - 1] else E)
                if d <= best:
                    best, mv = d, PAIR
            left = cur[j - 1] + E
            if left < best:
                best, mv = left, LEFT
            cur[j] = best
            bi[j] = mv
        back.append(bi)
        prev = cur
    ops = []
    i, j = n, m
    while i or j:
        mv = back[i][j]
        if mv == PAIR:
            ops.append(("hit" if rc[i - 1] == tc[j - 1] else "wrong", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == UP:
            ops.append(("missing", i - 1, None))
            i -= 1
        else:
            ops.append(("extra", None, j - 1))
            j -= 1
    ops.reverse()
    edits = sum(1 for o in ops if o[0] != "hit")
    return ops, edits


def align(ref_chars: str, ref_line_of: list[int], ref_lines: list[tuple[float, float]],
          test_chars: str, test_line_of: list[int], test_lines: list[tuple[float, float]],
          window: float = 1.0, direction: str = "forward"):
    """Time-aware alignment; returns (operations, number of edits). See the module docstring."""
    table = _pair_table(ref_lines, test_lines, window)
    if direction == "forward":
        return _align(ref_chars, ref_line_of, test_chars, test_line_of, table)
    if direction != "mirror":
        raise ValueError(direction)
    n, m = len(ref_chars), len(test_chars)
    ops, edits = _align(ref_chars[::-1], ref_line_of[::-1], test_chars[::-1], test_line_of[::-1], table)
    ops = [(k, None if i is None else n - 1 - i, None if j is None else m - 1 - j) for k, i, j in reversed(ops)]
    return ops, edits


def levenshtein(a: str, b: str) -> int:
    """Plain edit distance (substitution, deletion, insertion each cost 1), no time constraint."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]
