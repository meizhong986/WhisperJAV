"""Extract a film's audio as WhisperJAV does today, and with the gap-filling filter; compare length, time taken,
and where audio cut by timestamp (ffmpeg -ss) sits in each WAV.
Usage: python fix_test.py <media> <work dir> [t1 t2 ...]"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

FF = "ffmpeg"
FP = "ffprobe"
SR = 16000
media = Path(sys.argv[1])
work = Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
times = [float(x) for x in sys.argv[3:]]

r = subprocess.run([FP, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration,start_time",
                    "-of", "csv=p=0", str(media)], capture_output=True, text=True, check=True)
print("audio stream start,duration:", r.stdout.strip())

variants = {
    "today": [],
    "fixed": ["-af", "aresample=async=1:first_pts=0"],
}
wavs = {}
for name, extra in variants.items():
    out = work / f"{media.stem}.{name}.wav"
    cmd = [FF, "-i", str(media), "-vn"] + extra + ["-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", "-y", str(out)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.monotonic() - t0
    info = sf.info(str(out))
    print(f"{name:6s} exit {p.returncode}  took {dt:6.1f}s  WAV length {info.frames / info.samplerate:10.3f}s")
    wavs[name] = out

for name, out in wavs.items():
    wav, _ = sf.read(str(out), dtype="float32")
    for t in times:
        q = np.frombuffer(subprocess.run([FF, "-v", "error", "-ss", f"{t}", "-i", str(media), "-t", "8", "-vn", "-ac", "1",
                                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout,
                          dtype=np.float32)
        lo = max(int((t - 14) * SR), 0)
        seg = wav[lo:int((t + 4) * SR) + len(q)]
        nfft = 1 << (len(seg) + len(q) - 1).bit_length()
        c = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(q, nfft)), nfft)[: len(seg) - len(q) + 1]
        e = np.sqrt(np.convolve(seg * seg, np.ones(len(q)), "valid")) * np.sqrt(np.sum(q * q)) + 1e-12
        nc = c / e
        k = int(np.argmax(nc))
        print(f"  {name:6s} media t={t:8.1f}s  offset {(lo + k) / SR - t:+7.3f}s  peak {nc[k]:.3f}", flush=True)
