"""Cost of the audio packet scan next to the extraction, on files not yet read this session.
mode 'after': extraction (cold) then scan.  mode 'parallel': both started together.
Usage: python scan_cost.py <media> <after|parallel> <work dir>"""
import subprocess
import sys
import time
from pathlib import Path

FF = "ffmpeg"
FP = "ffprobe"
media, mode, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
work.mkdir(parents=True, exist_ok=True)
wav = work / "cost.wav"
ext = [FF, "-i", str(media), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(wav)]
scan = [FP, "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=pts_time,duration_time",
        "-of", "csv=p=0", str(media)]
print(f"{media.name}  {media.stat().st_size / 1e9:.2f} GB  mode={mode}")
t0 = time.monotonic()
if mode == "after":
    subprocess.run(ext, capture_output=True, check=True)
    t1 = time.monotonic()
    out = subprocess.run(scan, capture_output=True, text=True, check=True).stdout
    t2 = time.monotonic()
    print(f"extraction {t1 - t0:6.1f}s   scan after it {t2 - t1:6.1f}s   packets {len(out.splitlines())}")
else:
    pe = subprocess.Popen(ext, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps = subprocess.Popen(scan, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out, _ = ps.communicate()
    ts = time.monotonic()
    pe.wait()
    te = time.monotonic()
    print(f"scan done at {ts - t0:6.1f}s   extraction done at {te - t0:6.1f}s   packets {len(out.splitlines())}")
wav.unlink(missing_ok=True)
