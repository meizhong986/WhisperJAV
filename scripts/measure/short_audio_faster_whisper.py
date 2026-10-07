"""Pass a very short piece of audio (n samples at 16 kHz) to faster-whisper large-v2 with Balanced/aggressive
parameters, word timestamps on, and report whether it completes.

Usage: python short_audio_faster_whisper.py <n_samples> [noise|tone|silence]
Run once per length in its own process, so a native crash shows as the process exit code. Needs a CUDA GPU and the
faster-whisper of the WhisperJAV install being tested.
"""
import sys

import numpy as np
from faster_whisper import WhisperModel

n = int(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "noise"
rng = np.random.default_rng(0)
if kind == "noise":
    audio = (rng.standard_normal(n) * 0.05).astype(np.float32)
elif kind == "tone":
    t = np.arange(n) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
else:
    audio = np.zeros(n, dtype=np.float32)

model = WhisperModel("large-v2", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    audio,
    language="ja",
    task="transcribe",
    beam_size=2,
    best_of=2,
    patience=1.0,
    temperature=[0.0],
    compression_ratio_threshold=2.2,
    no_speech_threshold=0.72,
    condition_on_previous_text=False,
    word_timestamps=True,
    without_timestamps=False,
    max_initial_timestamp=1.0,
    suppress_blank=True,
    chunk_length=30,
    vad_filter=False,
)
segs = list(segments)
print(f"OK n={n} kind={kind} duration={info.duration:.4f}s segments={len(segs)}", flush=True)
