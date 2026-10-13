"""Run whisperjav.modules.audio_integrity on real files: extract as today (with tagged log lines), then check.
An extraction that fails is reported with FFmpeg's last error lines and the sweep goes on."""
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

from whisperjav.modules.audio_integrity import check_audio_integrity

FF = "ffmpeg"
FP = "ffprobe"
work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
for media in sys.argv[2:]:
    wav = work / "integrity.wav"
    wav.unlink(missing_ok=True)
    t0 = time.monotonic()
    r = subprocess.run([FF, "-loglevel", "level+info", "-i", media, "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                        "-ac", "1", "-y", str(wav)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    t_ext = time.monotonic() - t0
    if r.returncode != 0 or not wav.exists():
        errs = [l for l in r.stderr.splitlines() if "[error]" in l or "[fatal]" in l][-3:]
        print(f"== {Path(media).name}: EXTRACTION FAILED (exit {r.returncode}) after {t_ext:.1f}s: {errs}", flush=True)
        continue
    info = sf.info(str(wav))
    rep = check_audio_integrity(FP, media, info.frames / info.samplerate, r.stderr)
    print(f"== {Path(media).name}: extraction {t_ext:.1f}s, check {rep.seconds:.1f}s, checked={rep.checked}, "
          f"suspect={rep.suspect}", flush=True)
    print("   " + (rep.summary() or "(no findings)"), flush=True)
    wav.unlink(missing_ok=True)
