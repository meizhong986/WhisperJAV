"""For several media times t, cut 8 s of audio by timestamp (ffmpeg -ss) and find where it sits in the extracted WAV.
Prints t, best-matching WAV time, offset (WAV time - t) and correlation peak."""
import subprocess
import numpy as np
import soundfile as sf

FF = "ffmpeg"
MEDIA = r"<media library>\film_A.mp4"
WAV = r"<work folder>\probe\film A.probe.wav"
SR = 16000
wav, sr = sf.read(WAV, dtype="float32")
assert sr == SR

for t in [1000.0, 2100.0, 2200.0, 2900.0, 3700.0, 4400.0, 5100.0, 6500.0]:
    r = subprocess.run([FF, "-v", "error", "-ss", f"{t}", "-i", MEDIA, "-t", "8", "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    q = np.frombuffer(r.stdout, dtype=np.float32)
    lo = int((t - 14) * SR)
    hi = int((t + 4) * SR) + len(q)
    seg = wav[max(lo, 0):hi]
    # normalised cross-correlation via FFT
    n = len(seg) + len(q)
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(q, nfft)), nfft)[: len(seg) - len(q) + 1]
    e = np.sqrt(np.convolve(seg * seg, np.ones(len(q)), "valid")) * np.sqrt(np.sum(q * q)) + 1e-12
    nc = c / e
    k = int(np.argmax(nc))
    wt = (max(lo, 0) + k) / SR
    print(f"media t={t:8.1f}s  found in WAV at {wt:9.3f}s  offset {wt - t:+7.3f}s  peak {nc[k]:.3f}", flush=True)
