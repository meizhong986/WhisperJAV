"""Extract with WhisperJAV's own AudioExtractor (as the pipelines do) and find where audio cut from the film by
timestamp sits in its WAV. Usage: python extractor_check.py <film> <out wav> t1 t2 ..."""
import subprocess
import sys

import numpy as np
import soundfile as sf

from whisperjav.modules.audio_extraction import AudioExtractor

FF = "ffmpeg"
media, out = sys.argv[1], sys.argv[2]
times = [float(x) for x in sys.argv[3:]]
SR = 16000
path, duration = AudioExtractor(sample_rate=SR).extract(media, out)
print(f"extracted {duration:.3f} s")
wav, _ = sf.read(str(path), dtype="float32")
for t in times:
    q = np.frombuffer(subprocess.run([FF, "-v", "error", "-ss", str(t), "-i", media, "-t", "8", "-vn", "-ac", "1",
                                      "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout,
                      dtype=np.float32)
    lo = max(int((t - 14) * SR), 0)
    seg = wav[lo:int((t + 4) * SR) + len(q)]
    nfft = 1 << (len(seg) + len(q) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(q, nfft)), nfft)[: len(seg) - len(q) + 1]
    e = np.sqrt(np.convolve(seg * seg, np.ones(len(q)), "valid")) * np.sqrt(np.sum(q * q)) + 1e-12
    k = int(np.argmax(c / e))
    print(f"  film t={t:8.1f}s  offset {(lo + k) / SR - t:+7.3f}s  peak {float((c / e)[k]):.3f}")
