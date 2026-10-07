"""Audio that starts later (or earlier) than the video: where does the audio land in the WAV, today vs fixed?
Builds two remuxes of a clip with the audio shifted by +1.5 s and -1.5 s (stream copy), then for each extraction
finds the clip's original audio (cut from the unshifted source at time t) in the WAV.
Expected for subtitles to line up with the video: WAV time = t + shift."""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FF = "ffmpeg"
FP = "ffprobe"
SRC = str(Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix" / "The.Naked.Director.S01E04.Scene.2.mkv")
W = Path(r"<work folder>\startoff")
W.mkdir(exist_ok=True)
SR = 16000


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)


def cut(path, t, n=6):
    r = subprocess.run([FF, "-v", "error", "-ss", str(t), "-i", path, "-t", str(n), "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def locate(wav, q, guess):
    lo = max(int((guess - 5) * SR), 0)
    seg = wav[lo:int((guess + 5) * SR) + len(q)]
    nfft = 1 << (len(seg) + len(q) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(q, nfft)), nfft)[: len(seg) - len(q) + 1]
    e = np.sqrt(np.convolve(seg * seg, np.ones(len(q)), "valid")) * np.sqrt(np.sum(q * q)) + 1e-12
    k = int(np.argmax(c / e))
    return (lo + k) / SR, float((c / e)[k])


for shift in (1.5, -1.5):
    m = W / f"shift{shift:+.1f}.mkv"
    run([FF, "-v", "error", "-y", "-i", SRC, "-itsoffset", str(shift), "-i", SRC, "-map", "0:v:0", "-map", "1:a:0",
         "-c", "copy", str(m)])
    starts = run([FP, "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "csv=p=0", str(m)]).stdout
    print(f"== audio shifted {shift:+.1f} s; stream starts: {starts.split()}")
    for name, extra in (("today", []), ("fixed", ["-af", "aresample=async=1:first_pts=0"])):
        out = W / f"{m.stem}.{name}.wav"
        run([FF, "-v", "error", "-i", str(m), "-vn"] + extra + ["-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
                                                               "-y", str(out)])
        wav, _ = sf.read(str(out), dtype="float32")
        for t in (20.0, 60.0):
            where, peak = locate(wav, cut(SRC, t), t + shift)
            print(f"  {name}: source audio at {t:5.1f}s is in the WAV at {where:7.3f}s "
                  f"(video-aligned would be {t + shift:6.3f}s; off by {where - (t + shift):+.3f}s, peak {peak:.3f})")
