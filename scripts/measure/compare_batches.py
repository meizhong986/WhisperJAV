"""Compare the subtitle files of two measurement batches, pipeline by pipeline and film by film.

For each pipeline folder present in both batches and each .srt file present in both: the number of subtitle
lines, how many lines have the same text at the same position, and the largest start/end difference among
lines whose text is the same. Signature lines added by WhisperJAV are not treated specially: a version string
that differs would show as one changed line.

Usage (from the repository root):
    python scripts/measure/compare_batches.py D:\\WJ110_measure\\F1_stack D:\\WJ110_measure\\F2_tf5
"""
import re
import sys
from pathlib import Path

TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def read_srt(path):
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            m = TIME.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append((start, end, "\n".join(lines[i + 1:]).strip()))
                break
    return cues


def main(a, b):
    a, b = Path(a), Path(b)
    all_same = True
    for pa in sorted(p for p in a.iterdir() if p.is_dir()):
        pb = b / pa.name
        if not pb.is_dir():
            continue
        for fa in sorted(pa.glob("*.srt")):
            fb = pb / fa.name
            if not fb.exists():
                print(f"{pa.name:28} {fa.name[:44]:44} missing in {b.name}")
                all_same = False
                continue
            ca, cb = read_srt(fa), read_srt(fb)
            same_pos = sum(1 for x, y in zip(ca, cb) if x[2] == y[2])
            dt = [max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(ca, cb) if x[2] == y[2]]
            identical = len(ca) == len(cb) == same_pos and (max(dt) if dt else 0) == 0
            all_same &= identical
            print(f"{pa.name:28} {fa.name[:44]:44} lines {len(ca):3} vs {len(cb):3}; same text at same place "
                  f"{same_pos:3}; max time diff among those {max(dt) if dt else 0:.3f} s"
                  + ("  IDENTICAL" if identical else ""))
    print("RESULT:", "all identical" if all_same else "differences found")


if __name__ == "__main__":
    main(*sys.argv[1:3])
