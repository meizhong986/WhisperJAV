#!/usr/bin/env python3
"""Scene Inspector - run WhisperJAV scene detectors on a media file and report.

WHAT IT DOES
------------
Runs one or more of WhisperJAV's scene-detection backends (auditok, silero,
semantic, none) on a media file exactly as the pipelines do - audio extracted with
the project's AudioExtractor, detector built by SceneDetectorFactory with the
backend's own default parameters - and writes, into ``<media folder>/scenes_info/``
by default, for every backend B:

    <name>.B.scenes.csv          one row per scene: index, hh:mm:ss.mmm start/end, duration,
                                 gap before, detection pass, loudness (RMS / peak dBFS),
                                 speech seconds + ratio (--speech-ratio), subtitle cues (--srt),
                                 screenshot paths, backend metadata
    <name>.B.scenes.json         the full record (media facts, effective parameters, statistics,
                                 every scene, file names)
    <name>.B.summary.md          the human-readable summary
    <name>.B.chapters.ffmeta     FFmetadata chapters, one per scene (mux with
                                 ffmpeg -i media -i file.ffmeta -map_metadata 1 -codec copy out.mkv)
    <name>.B.contact_sheet_pNN.jpg  begin frame of every scene with index / time / duration
    screenshots/B/               begin / middle / end frame per scene, named
                                 <name>__scene0001__begin__00_12_03.450.jpg
    <name>.compare.md / .json    when more than one backend is given: counts, coverage,
                                 duration percentiles and pairwise boundary agreement

Only the media file is read; nothing is transcribed. No WhisperJAV configuration
files are touched.

USAGE
-----
    python tools/scene_inspector.py MEDIA [MEDIA ...]
        [--backend auditok[,silero,semantic,none]]   default: auditok; several = comparison
        [--sensitivity conservative|balanced|aggressive]  YAML presets; default: backend defaults
        [--scene-threshold FLOAT]                    semantic scene-change threshold (default 18)
        [--param KEY=VALUE ...]                      any other backend parameter
        [--speech-ratio [SEGMENTER]]                 speech seconds per scene (default firered-vad)
        [--srt FILE]                                 count subtitle cues per scene
        [--output-dir DIR]                           default: <media folder>/scenes_info
        [--no-screenshots] [--image-format jpg|png] [--keep-scene-audio]
        [--list-params BACKEND]

    python tools/scene_inspector.py movie.mp4
    python tools/scene_inspector.py movie.mp4 --backend semantic --scene-threshold 10
    python tools/scene_inspector.py movie.mp4 --backend auditok,semantic --speech-ratio --srt movie.ja.whisperjav.srt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKENDS = ("auditok", "silero", "semantic", "none")
DEFAULT_BACKEND = "auditok"          # the legacy pipelines' default (main.py --scene-detection-method)
SENSITIVITIES = ("conservative", "balanced", "aggressive")
DEFAULT_SPEECH_SEGMENTER = "firered-vad"   # the v1.9.2 Balanced default
OUTPUT_SUBFOLDER = "scenes_info"     # owner P2
SCREENSHOT_SUBFOLDER = "screenshots"
FRAME_POSITIONS = ("begin", "middle", "end")
END_FRAME_BACKOFF_S = 0.10           # step back from the cut so the last frame belongs to this scene
DURATION_BUCKETS_S = (5, 10, 20, 30, 60, 120)   # histogram edges
ASR_WINDOW_S = 29.0                  # one Whisper encoder window
ALIGNER_LIMIT_S = 180.0              # ChronosJAV ForcedAligner ceiling
BOUNDARY_TOLERANCE_S = 1.0           # two backends "agree" on a cut within this distance
SHEET_COLS, SHEET_TILE_W, SHEET_PAGE = 5, 320, 60   # contact sheet layout
SILENCE_DBFS = -120.0

# Parameters each backend's constructor reads (from the backends' _build_config_from_kwargs).
BACKEND_PARAMS: Dict[str, List[str]] = {
    "auditok": [
        "max_duration_s", "min_duration_s",
        "pass1_min_duration", "pass1_max_duration", "pass1_max_silence", "energy_threshold",
        "pass2_min_duration", "pass2_max_duration", "pass2_max_silence", "pass2_energy_threshold",
        "assist_processing", "bandpass_low_hz", "bandpass_high_hz",
        "drc_threshold_db", "drc_ratio", "drc_attack_ms", "drc_release_ms", "skip_assist_on_loud_dbfs",
        "brute_force_fallback", "brute_force_chunk_s",
    ],
    "silero": [
        "max_duration_s", "min_duration_s",
        "pass1_min_duration", "pass1_max_duration", "pass1_max_silence", "energy_threshold",
        "pass2_min_duration", "pass2_max_duration",
        "silero_threshold", "silero_neg_threshold", "silero_min_silence_ms", "silero_min_speech_ms",
        "silero_max_speech_s", "silero_min_silence_at_max", "silero_speech_pad_ms",
        "brute_force_chunk_s",
    ],
    "semantic": [
        "min_duration", "max_duration", "snap_window", "clustering_threshold", "sample_rate",
        "preserve_original_sr", "visualize",
    ],
    "none": [],
}
YAML_TOOL_NAMES = {
    "auditok": "auditok-scene-detection",
    "silero": "silero-scene-detection",
    "semantic": "semantic-scene-detection",
}


# ----------------------------------------------------------------------------- small helpers
def hms(seconds: float, sep: str = ":") -> str:
    """00:12:03.450 (sep=':') or 00_12_03.450 (sep='_')."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}{sep}{m:02d}{sep}{s:06.3f}"


def parse_param(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--param expects KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    key, raw = key.strip(), raw.strip()
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return key, True
    if low in ("false", "no", "off"):
        return key, False
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw


def parse_backends(text: str) -> List[str]:
    names = [b.strip() for b in text.split(",") if b.strip()]
    bad = [b for b in names if b not in BACKENDS]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown backend(s) {bad}; choose from {', '.join(BACKENDS)}")
    seen: List[str] = []
    for b in names:
        if b not in seen:
            seen.append(b)
    return seen


def dbfs(x: float) -> float:
    return round(20.0 * math.log10(x), 2) if x > 0 else SILENCE_DBFS


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bucket_label(edges: Tuple[int, ...], i: int) -> str:
    if i == 0:
        return f"0-{edges[0]}s"
    if i == len(edges):
        return f">{edges[-1]}s"
    return f"{edges[i-1]}-{edges[i]}s"


def ffprobe_media(path: Path) -> Dict[str, Any]:
    """Container duration, whether a real video stream exists, its geometry and fps."""
    ffprobe = shutil.which("ffprobe")
    info: Dict[str, Any] = {"container_duration_s": None, "has_video": False, "width": None,
                            "height": None, "fps": None, "size_bytes": path.stat().st_size}
    if not ffprobe:
        return info
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        ).stdout
        data = json.loads(out or "{}")
        fmt = data.get("format") or {}
        if fmt.get("duration"):
            info["container_duration_s"] = round(float(fmt["duration"]), 3)
        for st in data.get("streams") or []:
            if st.get("codec_type") == "video" and st.get("disposition", {}).get("attached_pic", 0) != 1:
                info["has_video"] = True
                info["width"], info["height"] = st.get("width"), st.get("height")
                rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or ""
                if "/" in rate:
                    num, den = rate.split("/")
                    if float(den or 0):
                        info["fps"] = round(float(num) / float(den), 3)
                break
    except Exception:  # noqa: BLE001 - probing is best-effort
        pass
    return info


def grab_frame(ffmpeg: str, media: Path, t: float, out_path: Path) -> bool:
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(media),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except Exception:  # noqa: BLE001
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def load_backend_yaml(backend: str) -> Dict[str, Any]:
    name = YAML_TOOL_NAMES.get(backend)
    if not name:
        return {}
    path = REPO_ROOT / "whisperjav" / "config" / "v4" / "ecosystems" / "tools" / f"{name}.yaml"
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def print_params(backend: str) -> None:
    print(f"Backend '{backend}' - parameters accepted via --param KEY=VALUE:")
    if not BACKEND_PARAMS.get(backend):
        print("  (none)")
        return
    doc = load_backend_yaml(backend)
    spec, presets = doc.get("spec") or {}, doc.get("presets") or {}
    for key in BACKEND_PARAMS[backend]:
        default = "(backend default)"
        for cand in (key, key + "_s", key[:-2] if key.endswith("_s") else key,
                     "pass1_" + key if key == "energy_threshold" else key):
            if cand in spec:
                default = spec[cand]
                break
        print(f"  {key:<28} default: {default}")
    if backend == "semantic":
        print("\n  --scene-threshold FLOAT is a shortcut for --param clustering_threshold=FLOAT")
    if presets:
        print("\nNamed presets in the tool YAML (--sensitivity uses conservative/balanced/aggressive):")
        for pname, pvals in presets.items():
            if pvals:
                print(f"  {pname}: {json.dumps(pvals)}")
    print(f"\nSource: whisperjav/config/v4/ecosystems/tools/{YAML_TOOL_NAMES.get(backend, '')}.yaml")


def sensitivity_params(backend: str, sensitivity: Optional[str]) -> Dict[str, Any]:
    """Only the named preset's own keys from the backend's tool YAML.

    Deliberately NOT the YAML ``spec`` block or ConfigManager's merged tool config:
    the auditok spec disagrees with the backend constructor defaults (e.g.
    pass2_max_duration_s 1800 vs 28 s, pass2_max_silence_s 1.8 vs 0.94 s), and the
    merged config carries global ``decode.*`` / ``silero.*`` keys. Applying the
    whole spec produced a 222 s scene where the defaults produce 28 s scenes.
    Owner P7: defaults are the backend's own; a preset adjusts a few keys on top.
    """
    if not sensitivity or backend not in YAML_TOOL_NAMES:
        return {}
    presets = (load_backend_yaml(backend).get("presets") or {})
    preset = presets.get(sensitivity) or {}
    return {k: v for k, v in preset.items() if not isinstance(v, (dict, list))}


# ----------------------------------------------------------------------------- statistics
def compute_stats(scenes: List[Dict[str, Any]], media_duration: float) -> Dict[str, Any]:
    durations = sorted(s["duration_s"] for s in scenes)
    n, total = len(scenes), sum(durations)
    gaps: List[float] = []
    prev_end = 0.0
    covered = 0.0   # union of the scene intervals: padded scenes (semantic) can overlap
    for s in scenes:  # time order
        gap = max(0.0, s["start_s"] - prev_end)
        s["gap_before_s"] = round(gap, 3)
        gaps.append(gap)
        covered += max(0.0, s["end_s"] - max(s["start_s"], prev_end))
        prev_end = max(prev_end, s["end_s"])
    trailing = max(0.0, media_duration - prev_end)
    hist = [0] * (len(DURATION_BUCKETS_S) + 1)
    for d in durations:
        i = 0
        while i < len(DURATION_BUCKETS_S) and d >= DURATION_BUCKETS_S[i]:
            i += 1
        hist[i] += 1
    out: Dict[str, Any] = {
        "scene_count": n,
        "media_duration_s": round(media_duration, 3),
        "scenes_total_s": round(total, 3),
        "covered_s": round(covered, 3),
        "overlap_s": round(max(0.0, total - covered), 3),
        "coverage_ratio": round(min(1.0, covered / media_duration), 4) if media_duration else None,
        "uncovered_s": round(max(0.0, media_duration - covered), 3),
        "scenes_per_minute": round(n / (media_duration / 60.0), 3) if media_duration else None,
        "duration": {
            "min_s": round(durations[0], 3) if n else 0.0,
            "p10_s": round(percentile(durations, 0.10), 3),
            "median_s": round(percentile(durations, 0.50), 3),
            "mean_s": round(total / n, 3) if n else 0.0,
            "p90_s": round(percentile(durations, 0.90), 3),
            "max_s": round(durations[-1], 3) if n else 0.0,
            "stdev_s": round(statistics.pstdev(durations), 3) if n > 1 else 0.0,
        },
        "duration_histogram": {bucket_label(DURATION_BUCKETS_S, i): c for i, c in enumerate(hist)},
        "scenes_over_asr_window_29s": sum(1 for d in durations if d > ASR_WINDOW_S),
        "scenes_over_aligner_limit_180s": sum(1 for d in durations if d > ALIGNER_LIMIT_S),
        "gaps": {
            "count_over_1s": sum(1 for g in gaps if g > 1.0),
            "total_s": round(sum(gaps) + trailing, 3),
            "longest_s": round(max(gaps + [trailing]) if (gaps or trailing) else 0.0, 3),
            "leading_s": round(gaps[0], 3) if gaps else 0.0,
            "trailing_s": round(trailing, 3),
        },
    }
    # Loudness (suggestion 2)
    rms = sorted(s["rms_dbfs"] for s in scenes if s.get("rms_dbfs") is not None)
    if rms:
        quiet = min(scenes, key=lambda s: s["rms_dbfs"])
        loud = max(scenes, key=lambda s: s["rms_dbfs"])
        out["loudness"] = {
            "rms_dbfs_min": rms[0], "rms_dbfs_median": round(percentile(rms, 0.5), 2), "rms_dbfs_max": rms[-1],
            "quietest_scene": quiet["scene_index"], "loudest_scene": loud["scene_index"],
        }
    # Speech ratio (suggestion 3)
    if any("speech_s" in s for s in scenes):
        speech_total = sum(s.get("speech_s") or 0.0 for s in scenes)
        out["speech"] = {
            "speech_total_s": round(speech_total, 3),
            "speech_ratio_of_scenes": round(speech_total / total, 4) if total else None,
            "speech_ratio_of_media": round(speech_total / media_duration, 4) if media_duration else None,
            "scenes_without_speech": [s["scene_index"] for s in scenes if (s.get("speech_s") or 0.0) == 0.0],
        }
    # Subtitle overlay (suggestion 7)
    if any("cues" in s for s in scenes):
        out["subtitles"] = {
            "cues_in_scenes": sum(s.get("cues") or 0 for s in scenes),
            "scenes_without_cues": [s["scene_index"] for s in scenes if (s.get("cues") or 0) == 0],
        }
    return out


def scene_loudness(audio, sr: int, start_s: float, end_s: float) -> Tuple[float, float]:
    import numpy as np
    a, b = int(max(0.0, start_s) * sr), int(max(0.0, end_s) * sr)
    x = audio[a:b]
    if x.size == 0:
        return SILENCE_DBFS, SILENCE_DBFS
    x = x.astype("float64")
    return dbfs(float(np.sqrt(np.mean(x * x)))), dbfs(float(np.max(np.abs(x))))


def read_srt_cues(path: Path) -> List[Tuple[float, float]]:
    import pysrt
    subs = pysrt.open(str(path), encoding="utf-8")
    return [(s.start.ordinal / 1000.0, s.end.ordinal / 1000.0) for s in subs]


def boundaries_of(scenes: List[Dict[str, Any]]) -> List[float]:
    pts = set()
    for s in scenes:
        pts.add(round(s["start_s"], 3))
        pts.add(round(s["end_s"], 3))
    return sorted(pts)


def agreement(a: List[float], b: List[float], tol: float = BOUNDARY_TOLERANCE_S) -> Optional[float]:
    """Fraction of A's boundaries that have a B boundary within tol seconds."""
    if not a:
        return None
    import bisect
    hits = 0
    for t in a:
        i = bisect.bisect_left(b, t)
        near = [b[j] for j in (i - 1, i) if 0 <= j < len(b)]
        if near and min(abs(t - x) for x in near) <= tol:
            hits += 1
    return round(hits / len(a), 3)


# ----------------------------------------------------------------------------- outputs
def write_chapters(path: Path, scenes: List[Dict[str, Any]], media_name: str) -> None:
    lines = [";FFMETADATA1", f"title={media_name} scenes", ""]
    for s in scenes:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(round(s['start_s'] * 1000))}", f"END={int(round(s['end_s'] * 1000))}",
                  f"title=Scene {s['scene_index']:04d} {s['start_hms']}-{s['end_hms']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_contact_sheets(out_dir: Path, base: str, scenes: List[Dict[str, Any]]) -> List[str]:
    """Begin frame of every scene, labelled, SHEET_PAGE tiles per page."""
    from PIL import Image, ImageDraw, ImageFont
    tiles = [(s, out_dir / s["screenshots"]["begin"]) for s in scenes
             if s.get("screenshots", {}).get("begin")]
    if not tiles:
        return []
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    label_h = 22
    written: List[str] = []
    pages = [tiles[i:i + SHEET_PAGE] for i in range(0, len(tiles), SHEET_PAGE)]
    for pno, page in enumerate(pages, start=1):
        thumbs = []
        for s, path in page:
            try:
                im = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            h = max(1, int(im.height * SHEET_TILE_W / im.width))
            thumbs.append((s, im.resize((SHEET_TILE_W, h))))
        if not thumbs:
            continue
        tile_h = max(t.height for _, t in thumbs) + label_h
        rows = math.ceil(len(thumbs) / SHEET_COLS)
        sheet = Image.new("RGB", (SHEET_COLS * SHEET_TILE_W, rows * tile_h), "black")
        draw = ImageDraw.Draw(sheet)
        for i, (s, t) in enumerate(thumbs):
            x, y = (i % SHEET_COLS) * SHEET_TILE_W, (i // SHEET_COLS) * tile_h
            sheet.paste(t, (x, y))
            draw.text((x + 4, y + t.height + 4),
                      f"#{s['scene_index']} {s['start_hms']} ({s['duration_s']:.1f}s)", fill="white", font=font)
        name = f"{base}.contact_sheet_p{pno:02d}.jpg"
        sheet.save(out_dir / name, quality=85)
        written.append(name)
    return written


def write_outputs(out_dir: Path, base: str, rec: Dict[str, Any]) -> None:
    (out_dir / f"{base}.scenes.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with open(out_dir / f"{base}.scenes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scene_index", "start_hms", "end_hms", "start_s", "end_s", "duration_s", "gap_before_s",
                    "detection_pass", "rms_dbfs", "peak_dbfs", "speech_s", "speech_ratio", "cues", "cues_per_min",
                    "screenshot_begin", "screenshot_middle", "screenshot_end", "metadata_json"])
        for s in rec["scenes"]:
            shots = s.get("screenshots") or {}
            w.writerow([s["scene_index"], s["start_hms"], s["end_hms"], s["start_s"], s["end_s"], s["duration_s"],
                        s.get("gap_before_s", 0.0), s["detection_pass"], s.get("rms_dbfs", ""), s.get("peak_dbfs", ""),
                        s.get("speech_s", ""), s.get("speech_ratio", ""), s.get("cues", ""), s.get("cues_per_min", ""),
                        shots.get("begin", ""), shots.get("middle", ""), shots.get("end", ""),
                        json.dumps(s["metadata"], ensure_ascii=False) if s["metadata"] else ""])

    st, m, b = rec["stats"], rec["media"], rec["backend"]
    dur = st["duration"]
    sc = rec["screenshots"]
    lines = [
        f"# Scene inspection - {m['name']} - {b['used']}",
        "",
        f"Generated {rec['generated_at']} by tools/scene_inspector.py",
        "",
        "## Media",
        f"- Audio duration: {hms(m['duration_s'])} ({m['duration_s']} s); container header {m.get('container_duration_s')} s; "
        f"size {round((m['size_bytes'] or 0) / 2**20, 1)} MiB",
        f"- Video: {'yes, ' + str(m['width']) + 'x' + str(m['height']) + ' @ ' + str(m['fps']) + ' fps' if m['has_video'] else 'no video stream'}",
        "",
        "## Detector",
        f"- Backend: {b['used']} (requested {b['requested']}); sensitivity preset: {b['sensitivity'] or 'none (backend defaults)'}; "
        f"detection {b['processing_time_s']} s, audio extraction {b['audio_extraction_s']} s",
        f"- Parameter overrides: {json.dumps(b['overrides']) if b['overrides'] else 'none'}",
        "",
        "## Scenes",
        f"- Count: {st['scene_count']}  ({st['scenes_per_minute']} per minute of media)",
        f"- Covered: {st['covered_s']} s of {st['media_duration_s']} s = {round((st['coverage_ratio'] or 0) * 100, 1)} %; "
        f"uncovered {st['uncovered_s']} s; scene time {st['scenes_total_s']} s of which {st['overlap_s']} s overlaps between scenes",
        f"- Duration: min {dur['min_s']} / p10 {dur['p10_s']} / median {dur['median_s']} / mean {dur['mean_s']} / p90 {dur['p90_s']} / max {dur['max_s']} s (stdev {dur['stdev_s']})",
        f"- Over the {ASR_WINDOW_S:.0f} s Whisper window: {st['scenes_over_asr_window_29s']}; over the {ALIGNER_LIMIT_S:.0f} s aligner limit: {st['scenes_over_aligner_limit_180s']}",
        f"- Gaps: {st['gaps']['count_over_1s']} longer than 1 s; total {st['gaps']['total_s']} s; longest {st['gaps']['longest_s']} s; "
        f"leading {st['gaps']['leading_s']} s; trailing {st['gaps']['trailing_s']} s",
    ]
    if "loudness" in st:
        L = st["loudness"]
        lines.append(f"- Loudness (RMS dBFS): min {L['rms_dbfs_min']} (scene {L['quietest_scene']}) / median {L['rms_dbfs_median']} / "
                     f"max {L['rms_dbfs_max']} (scene {L['loudest_scene']})")
    if "speech" in st:
        S = st["speech"]
        lines.append(f"- Speech ({rec['speech_segmenter']}): {S['speech_total_s']} s = {round((S['speech_ratio_of_scenes'] or 0) * 100, 1)} % of scene time, "
                     f"{round((S['speech_ratio_of_media'] or 0) * 100, 1)} % of the media; scenes without speech: "
                     f"{len(S['scenes_without_speech'])} {S['scenes_without_speech'][:20]}")
    if "subtitles" in st:
        T = st["subtitles"]
        lines.append(f"- Subtitles ({rec['srt']['name']}): {rec['srt']['cues_total']} cues, {T['cues_in_scenes']} inside scenes, "
                     f"{rec['srt']['cues_outside_scenes']} in gaps; scenes without cues: {len(T['scenes_without_cues'])} {T['scenes_without_cues'][:20]}")
    lines += ["", "### Duration histogram", "| bucket | scenes |", "|---|---|",
              *[f"| {k} | {v} |" for k, v in st["duration_histogram"].items()], "",
              "## Files",
              f"- `{base}.scenes.csv`, `{base}.scenes.json`, `{base}.chapters.ffmeta`"]
    if sc["written"]:
        lines.append(f"- {sc['written']} frames in `{sc['folder']}/` (begin / middle / end per scene)"
                     + (f", {sc['failed']} failed" if sc["failed"] else ""))
        if rec.get("contact_sheets"):
            lines.append("- Contact sheet(s): " + ", ".join(f"`{n}`" for n in rec["contact_sheets"]))
    else:
        lines.append(f"- Screenshots: none ({sc['skipped_reason'] or 'disabled with --no-screenshots'})")
    head = "| # | start | end | duration s | gap s | pass | RMS dBFS | speech s | cues |"
    lines += ["", "## Scene table", head, "|---|---|---|---|---|---|---|---|---|"]
    for s in rec["scenes"]:
        lines.append(f"| {s['scene_index']} | {s['start_hms']} | {s['end_hms']} | {s['duration_s']} | {s.get('gap_before_s', 0.0)} | "
                     f"{s['detection_pass']} | {s.get('rms_dbfs', '')} | {s.get('speech_s', '')} | {s.get('cues', '')} |")
    (out_dir / f"{base}.summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(out_dir: Path, stem: str, records: List[Dict[str, Any]]) -> None:
    rows = []
    bounds = {r["backend"]["used"]: boundaries_of(r["scenes"]) for r in records}
    for r in records:
        st, b = r["stats"], r["backend"]["used"]
        rows.append({
            "backend": b, "scenes": st["scene_count"], "coverage_pct": round((st["coverage_ratio"] or 0) * 100, 1),
            "median_s": st["duration"]["median_s"], "p90_s": st["duration"]["p90_s"], "max_s": st["duration"]["max_s"],
            "over_29s": st["scenes_over_asr_window_29s"], "gaps_over_1s": st["gaps"]["count_over_1s"],
            "detection_s": r["backend"]["processing_time_s"],
            "boundary_agreement": {other: agreement(bounds[b], bounds[other]) for other in bounds if other != b},
        })
    data = {"media": records[0]["media"]["name"], "generated_at": records[0]["generated_at"],
            "boundary_tolerance_s": BOUNDARY_TOLERANCE_S, "backends": rows}
    (out_dir / f"{stem}.compare.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    names = [r["backend"] for r in rows]
    lines = [f"# Scene detector comparison - {data['media']}", "", f"Generated {data['generated_at']}", "",
             "| backend | scenes | coverage % | median s | p90 s | max s | >29 s | gaps >1 s | detection s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['backend']} | {r['scenes']} | {r['coverage_pct']} | {r['median_s']} | {r['p90_s']} | {r['max_s']} | "
                     f"{r['over_29s']} | {r['gaps_over_1s']} | {r['detection_s']} |")
    lines += ["", f"## Boundary agreement (share of the row's cuts that the column also has within {BOUNDARY_TOLERANCE_S} s)", "",
              "| row \\ column | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in rows:
        cells = ["-" if n == r["backend"] else str(r["boundary_agreement"].get(n)) for n in names]
        lines.append(f"| {r['backend']} | " + " | ".join(cells) + " |")
    lines += ["", "Per-backend detail: `<name>.<backend>.summary.md`."]
    (out_dir / f"{stem}.compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- core
def inspect_media(media: Path, backends: List[str], sensitivity: Optional[str], params: Dict[str, Any],
                  out_dir: Path, *, screenshots: bool, image_format: str, keep_scene_audio: bool,
                  speech_segmenter: Optional[str], srt_path: Optional[Path]) -> List[Dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from whisperjav.modules.audio_extraction import AudioExtractor
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
    from whisperjav.utils.logger import logger

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = media.stem
    started = datetime.now().isoformat(timespec="seconds")
    media_info = ffprobe_media(media)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    cues = read_srt_cues(srt_path) if srt_path else None

    work = Path(tempfile.mkdtemp(prefix="scene_inspector_"))
    records: List[Dict[str, Any]] = []
    try:
        # 1. Audio, the way the pipelines extract it (16 kHz mono PCM).
        t0 = time.time()
        wav, audio_duration = AudioExtractor(sample_rate=16000).extract(media, work / f"{stem}.wav")
        extract_s = time.time() - t0
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        media_duration = round(float(audio_duration), 3)
        media_info["duration_s"] = media_duration
        logger.info("Audio extracted: %.1f s in %.1f s", audio_duration, extract_s)
        if media_info["container_duration_s"] and abs(media_info["container_duration_s"] - audio_duration) > 1.0:
            logger.warning("Container duration %.1f s differs from extracted audio %.1f s; statistics use the audio.",
                           media_info["container_duration_s"], audio_duration)

        segmenter = None
        if speech_segmenter:
            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
            segmenter = SpeechSegmenterFactory.create(speech_segmenter)
            logger.info("Speech ratio: segmenter %s", segmenter.name)

        for backend in backends:
            base = f"{stem}.{backend}"
            # 2. Detector: backend defaults (P7) < YAML sensitivity preset (6) < --param / --scene-threshold.
            effective = {**sensitivity_params(backend, sensitivity), **params}
            detector = SceneDetectorFactory.create(backend, **effective)
            scene_dir = (out_dir / f"{base}_scene_audio") if keep_scene_audio else (work / f"scenes_{backend}")
            scene_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            result = detector.detect_scenes(Path(wav), scene_dir, stem)
            detect_s = time.time() - t0
            try:
                detector.cleanup()
            except Exception:  # noqa: BLE001
                pass

            scenes: List[Dict[str, Any]] = []
            for idx, sc in enumerate(sorted(result.scenes, key=lambda s: s.start_sec), start=1):
                rms, peak = scene_loudness(audio, sr, sc.start_sec, sc.end_sec)
                row: Dict[str, Any] = {
                    "scene_index": idx,
                    "start_s": round(sc.start_sec, 3), "end_s": round(sc.end_sec, 3),
                    "duration_s": round(sc.duration_sec, 3),
                    "start_hms": hms(sc.start_sec), "end_hms": hms(sc.end_sec),
                    "detection_pass": sc.detection_pass,
                    "rms_dbfs": rms, "peak_dbfs": peak,
                    "metadata": sc.metadata or {},
                    "scene_audio": sc.scene_path.name if (keep_scene_audio and sc.scene_path) else None,
                }
                # 3. Speech seconds per scene (suggestion 3), on the scene WAV the detector wrote.
                if segmenter is not None and sc.scene_path and Path(sc.scene_path).exists():
                    try:
                        seg = segmenter.segment(Path(sc.scene_path))
                        speech = sum(max(0.0, min(g.end_sec, sc.duration_sec) - max(0.0, g.start_sec)) for g in seg.segments)
                    except Exception as exc:  # noqa: BLE001 - one scene must not stop the report
                        logger.warning("Speech ratio failed on scene %d: %s", idx, exc)
                        speech = 0.0
                    row["speech_s"] = round(speech, 3)
                    row["speech_ratio"] = round(speech / sc.duration_sec, 4) if sc.duration_sec else 0.0
                # 4. Subtitle cues per scene (suggestion 7): a cue belongs to the scene holding its midpoint.
                if cues is not None:
                    n_cues = sum(1 for a, b_ in cues if sc.start_sec <= (a + b_) / 2.0 < sc.end_sec)
                    row["cues"] = n_cues
                    row["cues_per_min"] = round(n_cues / (sc.duration_sec / 60.0), 2) if sc.duration_sec else 0.0
                scenes.append(row)
            stats = compute_stats(scenes, media_duration)

            # 5. Screenshots (P5/P6) and contact sheets (suggestion 4).
            shot_rel = f"{SCREENSHOT_SUBFOLDER}/{backend}"
            shot_dir = out_dir / SCREENSHOT_SUBFOLDER / backend
            shots_written = shots_failed = 0
            if screenshots and media_info["has_video"]:
                shot_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                for s in scenes:
                    times = {"begin": s["start_s"], "middle": (s["start_s"] + s["end_s"]) / 2.0,
                             "end": max(s["start_s"], s["end_s"] - END_FRAME_BACKOFF_S)}
                    s["screenshots"] = {}
                    for pos in FRAME_POSITIONS:
                        fname = f"{stem}__scene{s['scene_index']:04d}__{pos}__{hms(times[pos], '_')}.{image_format}"
                        if grab_frame(ffmpeg, media, times[pos], shot_dir / fname):
                            s["screenshots"][pos] = f"{shot_rel}/{fname}"
                            shots_written += 1
                        else:
                            shots_failed += 1
                logger.info("Screenshots (%s): %d written, %d failed, %.1f s", backend, shots_written, shots_failed, time.time() - t0)
            elif screenshots:
                logger.info("No video stream in %s - screenshots skipped", media.name)
            sheets = write_contact_sheets(out_dir, base, scenes) if shots_written else []

            # 6. Chapters (suggestion 5).
            write_chapters(out_dir / f"{base}.chapters.ffmeta", scenes, media.name)

            rec = {
                "tool": "scene_inspector", "generated_at": started,
                "media": {"path": str(media), "name": media.name, **media_info},
                "backend": {"requested": backend, "used": result.method, "sensitivity": sensitivity,
                            "parameters": result.parameters or effective, "overrides": params,
                            "processing_time_s": round(detect_s, 3), "audio_extraction_s": round(extract_s, 3)},
                "speech_segmenter": segmenter.name if segmenter is not None else None,
                "srt": ({"name": srt_path.name, "cues_total": len(cues),
                         "cues_outside_scenes": len(cues) - sum(s.get("cues", 0) for s in scenes)}
                        if cues is not None else None),
                "stats": stats,
                "screenshots": {"enabled": bool(screenshots), "written": shots_written, "failed": shots_failed,
                                "folder": shot_rel if shots_written else None,
                                "skipped_reason": None if (not screenshots or media_info["has_video"]) else "no video stream"},
                "contact_sheets": sheets,
                "chapters": f"{base}.chapters.ffmeta",
                "coarse_boundaries": result.coarse_boundaries,
                "scenes": scenes,
            }
            write_outputs(out_dir, base, rec)
            records.append(rec)

        if segmenter is not None:
            try:
                segmenter.cleanup()
            except Exception:  # noqa: BLE001
                pass
        if len(records) > 1:
            write_comparison(out_dir, stem, records)
        return records
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scene_inspector",
        description="Run WhisperJAV scene detectors on media and write scene statistics (CSV, JSON, Markdown), "
                    "FFmetadata chapters, begin/middle/end screenshots and contact sheets per scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[0],
    )
    p.add_argument("media", nargs="*", type=Path, help="Media file(s) - video or audio.")
    p.add_argument("--backend", type=parse_backends, default=[DEFAULT_BACKEND], metavar="NAME[,NAME...]",
                   help=f"Scene detector backend(s): {', '.join(BACKENDS)}. Default {DEFAULT_BACKEND} "
                        "(the legacy pipelines' default). Several names = run each and write a comparison.")
    p.add_argument("--sensitivity", choices=SENSITIVITIES, default=None,
                   help="Apply the backend's YAML sensitivity preset (as the pipelines do). Default: the backend's own defaults.")
    p.add_argument("--scene-threshold", type=float, default=None, metavar="FLOAT",
                   help="Semantic detector: clustering distance that separates scenes (default 18; lower tends to give "
                        "more, shorter scenes). Shortcut for --param clustering_threshold=FLOAT.")
    p.add_argument("--param", action="append", type=parse_param, default=[], metavar="KEY=VALUE",
                   help="Override one backend parameter; repeatable. See --list-params BACKEND.")
    p.add_argument("--speech-ratio", nargs="?", const=DEFAULT_SPEECH_SEGMENTER, default=None, metavar="SEGMENTER",
                   help=f"Measure speech seconds per scene with a WhisperJAV speech segmenter (default {DEFAULT_SPEECH_SEGMENTER}; "
                        "e.g. ten, silero-v3.1, whisperseg).")
    p.add_argument("--srt", type=Path, default=None, metavar="FILE",
                   help="Count the cues of this subtitle file per scene (a cue belongs to the scene holding its midpoint).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Target folder (default: <media folder>/{OUTPUT_SUBFOLDER}).")
    p.add_argument("--no-screenshots", action="store_true", help="Do not extract frames (also skips the contact sheet).")
    p.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    p.add_argument("--keep-scene-audio", action="store_true",
                   help="Also keep the per-scene WAV files the detector writes (in <target>/<name>.<backend>_scene_audio/).")
    p.add_argument("--list-params", choices=BACKENDS, metavar="BACKEND",
                   help="Print the parameters a backend accepts and exit.")
    p.add_argument("--quiet", action="store_true", help="Warnings only on the console.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_params:
        print_params(args.list_params)
        return 0
    if not args.media:
        print("error: give at least one media file (or --list-params BACKEND)", file=sys.stderr)
        return 2
    if args.srt is not None and not args.srt.exists():
        print(f"error: --srt file not found: {args.srt}", file=sys.stderr)
        return 2
    from whisperjav.utils.logger import setup_logger
    setup_logger("whisperjav", "WARNING" if args.quiet else "INFO")

    params = dict(args.param)
    if args.scene_threshold is not None:
        params["clustering_threshold"] = float(args.scene_threshold)
        if "semantic" not in args.backend:
            print("warning: --scene-threshold only affects the semantic backend, which is not selected", file=sys.stderr)
    for backend in args.backend:
        unknown = [k for k in params if k not in BACKEND_PARAMS.get(backend, [])]
        if unknown and len(args.backend) == 1:
            print(f"warning: --param key(s) not read by the {backend} backend and will be ignored: {unknown}", file=sys.stderr)

    status = 0
    for media in args.media:
        if not media.exists():
            print(f"error: not found: {media}", file=sys.stderr)
            status = 1
            continue
        out_dir = args.output_dir or (media.parent / OUTPUT_SUBFOLDER)
        try:
            records = inspect_media(media.resolve(), args.backend, args.sensitivity, params, out_dir,
                                    screenshots=not args.no_screenshots, image_format=args.image_format,
                                    keep_scene_audio=args.keep_scene_audio, speech_segmenter=args.speech_ratio,
                                    srt_path=args.srt.resolve() if args.srt else None)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next file
            print(f"error: {media.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
            continue
        for rec in records:
            st = rec["stats"]
            extra = ""
            if "speech" in st:
                extra += f", speech {round((st['speech']['speech_ratio_of_media'] or 0) * 100, 1)} % of media"
            if "subtitles" in st:
                extra += f", {len(st['subtitles']['scenes_without_cues'])} scenes without cues"
            print(f"{media.name} [{rec['backend']['used']}]: {st['scene_count']} scenes, coverage "
                  f"{round((st['coverage_ratio'] or 0) * 100, 1)} %, median {st['duration']['median_s']} s, "
                  f"max {st['duration']['max_s']} s{extra}; screenshots {rec['screenshots']['written']} -> {out_dir}")
        if len(records) > 1:
            print(f"{media.name}: comparison -> {out_dir / (media.stem + '.compare.md')}")
    return status


if __name__ == "__main__":
    sys.exit(main())
