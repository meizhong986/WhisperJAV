#!/usr/bin/env python3
"""Scene Inspector - run a WhisperJAV scene detector on a media file and report.

WHAT IT DOES
------------
Runs one of WhisperJAV's scene-detection backends (auditok, silero, semantic, none)
on a media file exactly as the pipelines do - audio extracted with the project's
AudioExtractor, detector built by SceneDetectorFactory with the backend's own
default parameters - and writes, into ``<media folder>/scenes_info/`` by default:

    <name>.scenes.csv       one row per scene (index, start/end/duration, hh:mm:ss.mmm,
                            detection pass, gap to the previous scene, backend metadata)
    <name>.scenes.json      the full record: media facts, backend + parameters, aggregate
                            statistics, every scene, the screenshot file names
    <name>.summary.md       the human-readable summary of the same statistics
    screenshots/            three frames per scene (beginning / middle / end), named
                            <name>__scene0001__begin__00_12_03.450.jpg   (--no-screenshots to skip)

Only the media file is read; nothing is transcribed. No WhisperJAV configuration
files are touched.

USAGE
-----
    python tools/scene_inspector.py MEDIA [MEDIA ...]
        [--backend auditok|silero|semantic|none]     default: auditok
        [--param KEY=VALUE ...]                      backend parameter overrides
        [--output-dir DIR]                           default: <media folder>/scenes_info
        [--no-screenshots] [--image-format jpg|png]
        [--keep-scene-audio]                         also keep the per-scene WAVs
        [--list-params BACKEND]                      show the parameters a backend accepts

    python tools/scene_inspector.py movie.mp4
    python tools/scene_inspector.py movie.mp4 --backend semantic --param clustering_threshold=10
    python tools/scene_inspector.py --list-params auditok
"""
from __future__ import annotations

import argparse
import csv
import json
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
OUTPUT_SUBFOLDER = "scenes_info"     # owner P2
SCREENSHOT_SUBFOLDER = "screenshots"
FRAME_POSITIONS = ("begin", "middle", "end")
END_FRAME_BACKOFF_S = 0.10           # step back from the cut so the last frame belongs to this scene
DURATION_BUCKETS_S = (5, 10, 20, 30, 60, 120)   # histogram edges
ASR_WINDOW_S = 29.0                  # one Whisper encoder window
ALIGNER_LIMIT_S = 180.0              # ChronosJAV ForcedAligner ceiling

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
    "auditok": "auditok-scene-detection.yaml",
    "silero": "silero-scene-detection.yaml",
    "semantic": "semantic-scene-detection.yaml",
}


# ----------------------------------------------------------------------------- helpers
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
    key = key.strip()
    raw = raw.strip()
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return key, True
    if low in ("false", "no", "off"):
        return key, False
    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        return key, raw


def ffprobe_media(path: Path) -> Dict[str, Any]:
    """Container duration, whether a video stream exists, and its geometry/fps."""
    ffprobe = shutil.which("ffprobe")
    info: Dict[str, Any] = {"duration_s": None, "has_video": False, "width": None,
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
            info["duration_s"] = float(fmt["duration"])
        for st in data.get("streams") or []:
            if st.get("codec_type") == "video" and st.get("disposition", {}).get("attached_pic", 0) != 1:
                info["has_video"] = True
                info["width"] = st.get("width")
                info["height"] = st.get("height")
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


def compute_stats(scenes: List[Dict[str, Any]], media_duration: Optional[float]) -> Dict[str, Any]:
    durations = sorted(s["duration_s"] for s in scenes)
    n = len(scenes)
    total = sum(durations)
    gaps: List[float] = []
    prev_end = 0.0
    for s in scenes:  # scenes are in time order
        gap = max(0.0, s["start_s"] - prev_end)
        s["gap_before_s"] = round(gap, 3)
        gaps.append(gap)
        prev_end = s["end_s"]
    trailing = max(0.0, (media_duration or prev_end) - prev_end)
    hist = [0] * (len(DURATION_BUCKETS_S) + 1)
    for d in durations:
        i = 0
        while i < len(DURATION_BUCKETS_S) and d >= DURATION_BUCKETS_S[i]:
            i += 1
        hist[i] += 1
    return {
        "scene_count": n,
        "media_duration_s": round(media_duration, 3) if media_duration else None,
        "scenes_total_s": round(total, 3),
        "coverage_ratio": round(total / media_duration, 4) if media_duration else None,
        "uncovered_s": round(max(0.0, (media_duration or total) - total), 3),
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
        "scenes_per_minute": round(n / (media_duration / 60.0), 3) if media_duration else None,
    }


def load_backend_yaml(backend: str) -> Dict[str, Any]:
    name = YAML_TOOL_NAMES.get(backend)
    if not name:
        return {}
    path = REPO_ROOT / "whisperjav" / "config" / "v4" / "ecosystems" / "tools" / name
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
    spec = doc.get("spec") or {}
    presets = doc.get("presets") or {}
    for key in BACKEND_PARAMS[backend]:
        default = "(backend default)"
        for cand in (key, key + "_s", key[:-2] if key.endswith("_s") else key,
                     "pass1_" + key if key == "energy_threshold" else key):
            if cand in spec:
                default = spec[cand]
                break
        print(f"  {key:<28} default: {default}")
    if presets:
        print("\nNamed presets in the tool YAML (for reference; pass their values with --param):")
        for pname, pvals in presets.items():
            if pvals:
                print(f"  {pname}: {json.dumps(pvals)}")
    print("\nSource: whisperjav/config/v4/ecosystems/tools/" + YAML_TOOL_NAMES.get(backend, ""))


# ----------------------------------------------------------------------------- core
def inspect_media(media: Path, backend: str, params: Dict[str, Any], out_dir: Path,
                  screenshots: bool, image_format: str, keep_scene_audio: bool) -> Dict[str, Any]:
    from whisperjav.modules.audio_extraction import AudioExtractor
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
    from whisperjav.utils.logger import logger

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = media.stem
    started = datetime.now()
    media_info = ffprobe_media(media)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    work = Path(tempfile.mkdtemp(prefix="scene_inspector_"))
    try:
        # 1. Audio, the way the pipelines extract it (16 kHz mono PCM).
        t0 = time.time()
        wav, audio_duration = AudioExtractor(sample_rate=16000).extract(media, work / f"{stem}.wav")
        extract_s = time.time() - t0
        logger.info("Audio extracted: %.1f s of audio in %.1f s", audio_duration, extract_s)
        # The detector works on the extracted audio; container headers can disagree
        # (a re-muxed clip may keep the original file's duration), so statistics use
        # the audio duration and the container value is reported alongside.
        media_info["container_duration_s"] = media_info.pop("duration_s")
        media_info["duration_s"] = round(float(audio_duration), 3)
        if media_info["container_duration_s"] and abs(media_info["container_duration_s"] - audio_duration) > 1.0:
            logger.warning("Container duration %.1f s differs from extracted audio %.1f s; statistics use the audio.",
                           media_info["container_duration_s"], audio_duration)

        # 2. Detector with the backend's own defaults plus explicit overrides (owner P7).
        detector = SceneDetectorFactory.create(backend, **params)
        scene_dir = (out_dir / f"{stem}_scene_audio") if keep_scene_audio else (work / "scenes")
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
            scenes.append({
                "scene_index": idx,
                "start_s": round(sc.start_sec, 3),
                "end_s": round(sc.end_sec, 3),
                "duration_s": round(sc.duration_sec, 3),
                "start_hms": hms(sc.start_sec),
                "end_hms": hms(sc.end_sec),
                "detection_pass": sc.detection_pass,
                "metadata": sc.metadata or {},
                "scene_audio": sc.scene_path.name if (keep_scene_audio and sc.scene_path) else None,
            })
        stats = compute_stats(scenes, media_info["duration_s"])

        # 3. Screenshots: beginning / middle / end of every scene (owner P5/P6).
        shot_dir = out_dir / SCREENSHOT_SUBFOLDER
        shots_written = 0
        shots_failed = 0
        if screenshots and media_info["has_video"]:
            shot_dir.mkdir(exist_ok=True)
            t0 = time.time()
            for s in scenes:
                begin = s["start_s"]
                end = max(s["start_s"], s["end_s"] - END_FRAME_BACKOFF_S)
                times = {"begin": begin, "middle": (s["start_s"] + s["end_s"]) / 2.0, "end": end}
                s["screenshots"] = {}
                for pos in FRAME_POSITIONS:
                    t = times[pos]
                    fname = f"{stem}__scene{s['scene_index']:04d}__{pos}__{hms(t, '_')}.{image_format}"
                    if grab_frame(ffmpeg, media, t, shot_dir / fname):
                        s["screenshots"][pos] = f"{SCREENSHOT_SUBFOLDER}/{fname}"
                        shots_written += 1
                    else:
                        shots_failed += 1
            logger.info("Screenshots: %d written, %d failed, %.1f s", shots_written, shots_failed, time.time() - t0)
        elif screenshots:
            logger.info("No video stream in %s — screenshots skipped", media.name)

        record = {
            "tool": "scene_inspector",
            "generated_at": started.isoformat(timespec="seconds"),
            "media": {"path": str(media), "name": media.name, **media_info},
            "backend": {"requested": backend, "used": result.method, "parameters": result.parameters or params,
                        "overrides": params, "processing_time_s": round(detect_s, 3),
                        "audio_extraction_s": round(extract_s, 3)},
            "stats": stats,
            "screenshots": {"enabled": bool(screenshots), "written": shots_written, "failed": shots_failed,
                            "folder": SCREENSHOT_SUBFOLDER if shots_written else None,
                            "skipped_reason": None if (not screenshots or media_info["has_video"]) else "no video stream"},
            "coarse_boundaries": result.coarse_boundaries,
            "scenes": scenes,
        }
        write_outputs(out_dir, stem, record)
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)


def write_outputs(out_dir: Path, stem: str, rec: Dict[str, Any]) -> None:
    (out_dir / f"{stem}.scenes.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with open(out_dir / f"{stem}.scenes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scene_index", "start_hms", "end_hms", "start_s", "end_s", "duration_s",
                    "gap_before_s", "detection_pass", "screenshot_begin", "screenshot_middle",
                    "screenshot_end", "metadata_json"])
        for s in rec["scenes"]:
            shots = s.get("screenshots") or {}
            w.writerow([s["scene_index"], s["start_hms"], s["end_hms"], s["start_s"], s["end_s"],
                        s["duration_s"], s.get("gap_before_s", 0.0), s["detection_pass"],
                        shots.get("begin", ""), shots.get("middle", ""), shots.get("end", ""),
                        json.dumps(s["metadata"], ensure_ascii=False) if s["metadata"] else ""])

    st, m, b = rec["stats"], rec["media"], rec["backend"]
    dur = st["duration"]
    lines = [
        f"# Scene inspection — {m['name']}",
        "",
        f"Generated {rec['generated_at']} by tools/scene_inspector.py",
        "",
        "## Media",
        f"- Audio duration: {hms(m['duration_s'] or 0)} ({m['duration_s']} s); container header {m.get('container_duration_s')} s; size {round((m['size_bytes'] or 0) / 2**20, 1)} MiB",
        f"- Video: {'yes, ' + str(m['width']) + 'x' + str(m['height']) + ' @ ' + str(m['fps']) + ' fps' if m['has_video'] else 'no video stream'}",
        "",
        "## Detector",
        f"- Backend: {b['used']} (requested {b['requested']}); detection {b['processing_time_s']} s, audio extraction {b['audio_extraction_s']} s",
        f"- Parameter overrides: {json.dumps(b['overrides']) if b['overrides'] else 'none (backend defaults)'}",
        "",
        "## Scenes",
        f"- Count: {st['scene_count']}  ({st['scenes_per_minute']} per minute of media)",
        f"- Covered: {st['scenes_total_s']} s of {st['media_duration_s']} s = {round((st['coverage_ratio'] or 0) * 100, 1)} %; uncovered {st['uncovered_s']} s",
        f"- Duration: min {dur['min_s']} / p10 {dur['p10_s']} / median {dur['median_s']} / mean {dur['mean_s']} / p90 {dur['p90_s']} / max {dur['max_s']} s (stdev {dur['stdev_s']})",
        f"- Over the {ASR_WINDOW_S:.0f} s Whisper window: {st['scenes_over_asr_window_29s']}; over the {ALIGNER_LIMIT_S:.0f} s aligner limit: {st['scenes_over_aligner_limit_180s']}",
        f"- Gaps: {st['gaps']['count_over_1s']} gaps longer than 1 s; total {st['gaps']['total_s']} s; longest {st['gaps']['longest_s']} s; leading {st['gaps']['leading_s']} s; trailing {st['gaps']['trailing_s']} s",
        "",
        "### Duration histogram",
        "| bucket | scenes |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in st["duration_histogram"].items()],
        "",
        "## Screenshots",
        (f"- {rec['screenshots']['written']} frames written to `{SCREENSHOT_SUBFOLDER}/` (3 per scene: begin / middle / end)"
         + (f", {rec['screenshots']['failed']} failed" if rec['screenshots']['failed'] else ""))
        if rec["screenshots"]["written"] else
        f"- none ({rec['screenshots']['skipped_reason'] or 'disabled with --no-screenshots'})",
        "",
        "## Scene table",
        "| # | start | end | duration s | gap before s | pass |", "|---|---|---|---|---|---|",
        *[f"| {s['scene_index']} | {s['start_hms']} | {s['end_hms']} | {s['duration_s']} | {s.get('gap_before_s', 0.0)} | {s['detection_pass']} |"
          for s in rec["scenes"]],
        "",
        f"Files: `{stem}.scenes.csv`, `{stem}.scenes.json`",
    ]
    (out_dir / f"{stem}.summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scene_inspector",
        description="Run a WhisperJAV scene detector on media and write scene statistics "
                    "(CSV, JSON, Markdown) plus begin/middle/end screenshots per scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[0],
    )
    p.add_argument("media", nargs="*", type=Path, help="Media file(s) - video or audio.")
    p.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                   help=f"Scene detector backend (default: {DEFAULT_BACKEND}, the legacy pipelines' default).")
    p.add_argument("--param", action="append", type=parse_param, default=[], metavar="KEY=VALUE",
                   help="Override one backend parameter; repeatable. See --list-params BACKEND.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Target folder (default: <media folder>/{OUTPUT_SUBFOLDER}).")
    p.add_argument("--no-screenshots", action="store_true", help="Do not extract frames.")
    p.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    p.add_argument("--keep-scene-audio", action="store_true",
                   help="Also keep the per-scene WAV files the detector writes (in <target>/<name>_scene_audio/).")
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
    from whisperjav.utils.logger import setup_logger
    setup_logger("whisperjav", "WARNING" if args.quiet else "INFO")

    params = dict(args.param)
    unknown = [k for k in params if BACKEND_PARAMS.get(args.backend) is not None and k not in BACKEND_PARAMS[args.backend]]
    if unknown:
        print(f"warning: --param key(s) not read by the {args.backend} backend and will be ignored: {unknown}",
              file=sys.stderr)

    status = 0
    for media in args.media:
        if not media.exists():
            print(f"error: not found: {media}", file=sys.stderr)
            status = 1
            continue
        out_dir = args.output_dir or (media.parent / OUTPUT_SUBFOLDER)
        try:
            rec = inspect_media(media.resolve(), args.backend, params, out_dir,
                                screenshots=not args.no_screenshots, image_format=args.image_format,
                                keep_scene_audio=args.keep_scene_audio)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next file
            print(f"error: {media.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
            continue
        st = rec["stats"]
        print(f"{media.name}: {st['scene_count']} scenes ({rec['backend']['used']}), "
              f"coverage {round((st['coverage_ratio'] or 0) * 100, 1)} %, median {st['duration']['median_s']} s, "
              f"max {st['duration']['max_s']} s; screenshots {rec['screenshots']['written']} -> {out_dir}")
    return status


if __name__ == "__main__":
    sys.exit(main())
