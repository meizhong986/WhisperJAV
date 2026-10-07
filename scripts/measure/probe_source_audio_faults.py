"""Measure source-audio properties of one media file, and how long each measurement takes.

Checks: stream list and stated start times and durations (ffprobe); an audio packet scan (holes, overlaps, packets
much longer than usual, timestamps running backwards); extraction to 16 kHz mono as WhisperJAV does, counting
ffmpeg's error lines; extracted length; runs of digital silence longer than 2 s; left/right correlation and level
difference on 40 two-second stereo windows.

Usage:
    python probe_source_audio_faults.py <media> [--work <folder>] [--ffmpeg-dir <folder with ffmpeg and ffprobe>]
Prints one line per check (name, seconds taken, result) and writes <work>/<media stem>.probe.json.
The extracted WAV is kept in <work>. ffmpeg/ffprobe are taken from --ffmpeg-dir, else from PATH.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("media", type=Path)
_ap.add_argument("--work", type=Path, default=Path("."))
_ap.add_argument("--ffmpeg-dir", type=Path)
_args = _ap.parse_args()
FFMPEG = str(_args.ffmpeg_dir / "ffmpeg") if _args.ffmpeg_dir else "ffmpeg"
FFPROBE = str(_args.ffmpeg_dir / "ffprobe") if _args.ffmpeg_dir else "ffprobe"
media = _args.media
work = _args.work
work.mkdir(parents=True, exist_ok=True)
results = {}


def timed(name):
    def deco(fn):
        t0 = time.monotonic()
        out = fn()
        dt = time.monotonic() - t0
        results[name] = {"seconds": round(dt, 2), "result": out}
        print(f"{name:<34} {dt:7.2f}s  {json.dumps(out, ensure_ascii=False)[:300]}", flush=True)
        return out
    return deco


@timed("0 streams+format (ffprobe)")
def streams():
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=index,codec_type,codec_name,start_time,duration,channels,sample_rate,nb_frames:"
                        "stream_tags=language,title:stream_disposition=default,comment:format=duration,start_time",
                        "-of", "json", str(media)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    audio = [s for s in d["streams"] if s["codec_type"] == "audio"]
    video = [s for s in d["streams"] if s["codec_type"] == "video"]
    a0 = audio[0] if audio else {}
    v0 = video[0] if video else {}
    f = lambda x: float(x) if x not in (None, "N/A") else None
    return {
        "n_audio_streams": len(audio),
        "audio_start": f(a0.get("start_time")), "video_start": f(v0.get("start_time")),
        "audio_dur": f(a0.get("duration")), "video_dur": f(v0.get("duration")),
        "format_dur": f(d["format"].get("duration")),
        "channels": a0.get("channels"),
    }


@timed("1 audio packet scan (ffprobe)")
def packets():
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "packet=pts_time,duration_time", "-of", "csv=p=0", str(media)],
                       capture_output=True, text=True, check=True)
    pts, dur = [], []
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and parts[0] not in ("", "N/A") and parts[1] not in ("", "N/A"):
            pts.append(float(parts[0]))
            dur.append(float(parts[1]))
    pts, dur = np.array(pts), np.array(dur)
    order_breaks = int(np.sum(np.diff(pts) < 0))
    typical = float(np.median(dur)) if len(dur) else 0.0
    ends = pts + dur
    gaps = pts[1:] - ends[:-1]              # >0: hole; <0: overlap
    long_pkts = np.where(dur > 10 * typical)[0]
    holes = [(round(float(pts[i + 1]), 3), round(float(g), 3)) for i, g in enumerate(gaps) if g > 0.1]
    overlaps = int(np.sum(gaps < -0.1))
    span = float(ends[-1] - pts[0]) if len(pts) else 0.0
    return {"packets": len(pts), "typical_dur": round(typical, 4), "backwards": order_breaks,
            "holes_gt_0.1s": holes[:10], "n_holes": len(holes), "overlaps_gt_0.1s": overlaps,
            "long_packets": [(round(float(pts[i]), 3), round(float(dur[i]), 3)) for i in long_pkts[:10]],
            "sum_dur": round(float(dur.sum()), 3), "ts_span": round(span, 3)}


wav = work / (media.stem + ".probe.wav")


@timed("2 extraction as today (ffmpeg)")
def extract():
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(media), "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                        "-ac", "1", "-y", str(wav)], capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if l.strip()]
    return {"exit": r.returncode, "decode_error_lines": len(errs), "first_errors": errs[:3]}


@timed("3 extracted length vs stream")
def length_check():
    info = sf.info(str(wav))
    return {"wav_seconds": round(info.frames / info.samplerate, 3)}


@timed("4 digital-silence runs >2s (numpy)")
def silence():
    x, sr = sf.read(str(wav), dtype="int16")
    zero = (np.abs(x) <= 1).astype(np.int8)
    edges = np.diff(np.concatenate([[0], zero, [0]]))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = [(round(s / sr, 2), round((e - s) / sr, 2)) for s, e in zip(starts, stops) if (e - s) / sr > 2.0]
    return {"n_runs": len(runs), "runs": runs[:10], "total_s": round(sum(r[1] for r in runs), 1)}


@timed("5 stereo phase/channel (40 windows)")
def stereo():
    info = results["0 streams+format (ffprobe)"]["result"]
    if (info.get("channels") or 1) < 2:
        return {"skipped": "mono"}
    total = info.get("audio_dur") or info.get("format_dur") or 0
    corrs, lr_ratio = [], []
    for k in range(40):
        t = total * (k + 0.5) / 40
        r = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.2f}", "-i", str(media), "-t", "2", "-vn",
                            "-ac", "2", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True)
        s = np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2)
        if len(s) < 1000:
            continue
        l, rr = s[:, 0], s[:, 1]
        el, er = float(np.mean(l * l)), float(np.mean(rr * rr))
        if el + er < 1e-8:
            continue
        corrs.append(float(np.corrcoef(l, rr)[0, 1]) if el > 1e-10 and er > 1e-10 else 0.0)
        lr_ratio.append(10 * np.log10((el + 1e-12) / (er + 1e-12)))
    return {"windows": len(corrs), "median_LR_corr": round(float(np.median(corrs)), 3) if corrs else None,
            "min_LR_corr": round(float(np.min(corrs)), 3) if corrs else None,
            "median_L_minus_R_dB": round(float(np.median(lr_ratio)), 1) if lr_ratio else None}


(work / (media.stem + ".probe.json")).write_text(json.dumps(results, indent=1), encoding="utf-8")
