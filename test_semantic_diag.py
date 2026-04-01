"""
Semantic Scene Detection — Standalone Diagnostic Test
=====================================================
This script reproduces the exact audio processing steps that the semantic
scene detector performs, outside of WhisperJAV's pipeline.  It helps
isolate whether a hang is caused by a system-level library issue or by
something specific to WhisperJAV's subprocess architecture.

Usage (from the WhisperJAV install folder):
    python test_semantic_diag.py <media_file>

Example:
    python test_semantic_diag.py "C:\Videos\my_video.mp4"
    python test_semantic_diag.py "D:\test_clip.wav"

The script will:
  1. Print library versions (numpy, librosa, numba, soundfile)
  2. Extract audio from the media file (via ffmpeg)
  3. Read the audio with soundfile
  4. Run librosa feature extraction (mfcc, rms, chroma, etc.)
  5. Report timing at each step

If the script hangs at a specific step, that tells us exactly which
library operation is the problem on your system.
"""

import sys
import os
import time
import tempfile
import subprocess
import shutil


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_semantic_diag.py <media_file>")
        print('Example: python test_semantic_diag.py "C:\\Videos\\my_video.mp4"')
        sys.exit(1)

    media_file = sys.argv[1]
    if not os.path.isfile(media_file):
        print(f"Error: File not found: {media_file}")
        sys.exit(1)

    print("=" * 60)
    print("Semantic Scene Detection — Diagnostic Test")
    print("=" * 60)
    print(f"Input: {media_file}")
    print()

    # ── Step 0: Library versions ──────────────────────────────
    print("[Step 0] Checking library versions...")
    for name in ("numpy", "librosa", "numba", "soundfile", "scipy"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {name} = {ver}")
        except ImportError:
            print(f"  {name} = NOT INSTALLED")
            if name in ("numpy", "librosa", "soundfile"):
                print(f"  ERROR: {name} is required. Cannot continue.")
                sys.exit(1)

    # ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            print(f"  ffmpeg = {r.stdout.splitlines()[0] if r.stdout else '?'}")
        except Exception as e:
            print(f"  ffmpeg = error ({e})")
    else:
        print("  ffmpeg = NOT FOUND")
        print("  ERROR: ffmpeg is required for audio extraction.")
        sys.exit(1)

    print()

    # ── Step 1: Extract audio with ffmpeg ─────────────────────
    print("[Step 1] Extracting audio with ffmpeg (16kHz, mono)...")
    t0 = time.time()

    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        cmd = [
            "ffmpeg", "-y", "-i", media_file,
            "-ar", "16000", "-ac", "1", "-vn",
            "-loglevel", "error",
            temp_wav,
        ]
        subprocess.run(cmd, check=True, timeout=120)
        print(f"  OK ({time.time() - t0:.1f}s) -> {temp_wav}")
    except subprocess.TimeoutExpired:
        print("  FAILED: ffmpeg timed out after 120s")
        _cleanup(temp_wav)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: ffmpeg error: {e}")
        _cleanup(temp_wav)
        sys.exit(1)

    print()

    # ── Step 2: Read audio with soundfile ─────────────────────
    print("[Step 2] Reading audio with soundfile...")
    t0 = time.time()

    import soundfile as sf

    try:
        info = sf.info(temp_wav)
        print(f"  Format: {info.format} {info.subtype}")
        print(f"  Sample rate: {info.samplerate} Hz")
        print(f"  Channels: {info.channels}")
        print(f"  Duration: {info.duration:.1f}s")
        print(f"  OK ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        _cleanup(temp_wav)
        sys.exit(1)

    print()

    # ── Step 3: Read first 60s block with sf.blocks ───────────
    print("[Step 3] Reading first 60s audio block with sf.blocks()...")
    t0 = time.time()

    import numpy as np

    block_size = 60 * info.samplerate  # 60 seconds
    block = None
    try:
        for block in sf.blocks(temp_wav, blocksize=block_size, always_2d=True):
            break  # only need first block
        if block is not None:
            print(f"  Block shape: {block.shape}")
            y = block[:, 0] if block.shape[1] == 1 else np.mean(block, axis=1)
            print(f"  Audio array: {len(y)} samples, dtype={y.dtype}")
            print(f"  OK ({time.time() - t0:.2f}s)")
        else:
            print("  FAILED: No blocks returned")
            _cleanup(temp_wav)
            sys.exit(1)
    except Exception as e:
        print(f"  FAILED: {e}")
        _cleanup(temp_wav)
        sys.exit(1)

    print()

    # ── Step 4: librosa.feature.mfcc ──────────────────────────
    print("[Step 4] Running librosa.feature.mfcc()...")
    print("  (This is where the semantic engine hangs on some systems.)")
    print("  Computing...", end="", flush=True)
    t0 = time.time()

    import librosa

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        elapsed = time.time() - t0
        print(f" OK ({elapsed:.2f}s)")
        print(f"  MFCC shape: {mfcc.shape}")
    except Exception as e:
        print(f" FAILED: {e}")
        _cleanup(temp_wav)
        sys.exit(1)

    print()

    # ── Step 5: Remaining librosa features ────────────────────
    print("[Step 5] Running remaining feature extraction...")

    features = {"mfcc": mfcc}
    for name, fn in [
        ("delta",              lambda: librosa.feature.delta(mfcc)),
        ("rms",                lambda: librosa.feature.rms(y=y)),
        ("zero_crossing_rate", lambda: librosa.feature.zero_crossing_rate(y=y)),
        ("spectral_contrast",  lambda: librosa.feature.spectral_contrast(y=y, sr=16000)),
        ("chroma_stft",        lambda: librosa.feature.chroma_stft(y=y, sr=16000)),
    ]:
        print(f"  {name}...", end="", flush=True)
        t0 = time.time()
        try:
            result = fn()
            print(f" OK ({time.time() - t0:.2f}s, shape={result.shape})")
            features[name] = result
        except Exception as e:
            print(f" FAILED: {e}")

    print()

    # ── Done ──────────────────────────────────────────────────
    _cleanup(temp_wav)

    print("=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("The semantic scene detection libraries work on your system.")
    print("If WhisperJAV still hangs with semantic scene detection,")
    print("the issue may be specific to the subprocess worker context.")
    print()
    print("In that case, try running WhisperJAV with these environment")
    print("variables set before launching:")
    print()
    print("  set OMP_NUM_THREADS=1")
    print("  set MKL_NUM_THREADS=1")
    print()
    print("Then run WhisperJAV as usual.")


def _cleanup(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
