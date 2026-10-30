"""Time today's extraction against the gap-filling one on the same film, alternating, after one warm-up read,
so both read the film from memory. Usage: python fix_cost.py <work dir> <film> [<film> ...]"""
import statistics
import subprocess
import sys
import time
from pathlib import Path

FF = "ffmpeg"
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
wav = work / "cost.wav"
VARIANTS = {"today": [], "fixed": ["-af", "aresample=async=1:first_pts=0"]}


def extract(media, extra):
    t0 = time.monotonic()
    subprocess.run([FF, "-loglevel", "level+info", "-i", media, "-vn"] + extra +
                   ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(wav)],
                   capture_output=True, check=True)
    return time.monotonic() - t0


for media in sys.argv[2:]:
    warm = extract(media, [])
    times = {k: [] for k in VARIANTS}
    for _ in range(3):
        for k, extra in VARIANTS.items():
            times[k].append(extract(media, extra))
    print(f"{Path(media).name}: warm-up {warm:.1f}s | today " + " ".join(f"{t:.2f}" for t in times["today"]) +
          f" (median {statistics.median(times['today']):.2f}s) | fixed " + " ".join(f"{t:.2f}" for t in times["fixed"]) +
          f" (median {statistics.median(times['fixed']):.2f}s)", flush=True)
wav.unlink(missing_ok=True)
