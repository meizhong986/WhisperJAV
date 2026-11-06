#!/usr/bin/env python
"""VibeVoice-ASR through CrispASR: a reference way to run, measure and compare VibeVoice models and settings.

Standalone research utility. It imports nothing from WhisperJAV. It needs the crispasr package (CrispASR's Python
binding), numpy, ffmpeg, and for the GPU figures nvidia-smi; huggingface_hub for listing and downloading models.
A run can be scored with subeval (the repository's subtitle scoring tool) against a reference SRT.

  python tools/vibevoice_crispasr_ref.py gpu
  python tools/vibevoice_crispasr_ref.py models
  python tools/vibevoice_crispasr_ref.py download q8_0
  python tools/vibevoice_crispasr_ref.py run --model q8_0 --input clip.mkv --piece-seconds 60 \
      --reference clip.ja.srt --out D:\\runs\\vv_q8

Every command is described in tools/vibevoice_crispasr_ref.md.

Exit status: 0 done; 2 a setup or input problem (message printed); 3 not enough free GPU memory for the model
(nothing loaded); 4 CrispASR could not open the model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

TOOL_VERSION = "1 (2026-11-06)"
REPO_ROOT = Path(__file__).resolve().parents[1]
SR = 16000  # what the binding's Session.transcribe takes; CrispASR converts to VibeVoice's 24 kHz itself

# Short names for the VibeVoice files on Hugging Face that CrispASR's readme names (checked 2026-11-06).
# Any other file: a local path, or hf:<owner>/<repo>/<file>.
MODELS = {
    "q4_k": ("cstr/vibevoice-asr-GGUF", "vibevoice-asr-q4_k.gguf", "VibeVoice-ASR 7B, Q4_K"),
    "q8_0": ("cstr/vibevoice-asr-GGUF", "vibevoice-asr-q8_0.gguf", "VibeVoice-ASR 7B, Q8_0"),
    "f16": ("cstr/vibevoice-asr-GGUF", "vibevoice-asr-f16.gguf", "VibeVoice-ASR 7B, F16"),
    "streaming-1.5b-q4_k": ("cstr/vibevoice-asr-streaming-1.5b-GGUF", "vibevoice-asr-streaming-1.5b-q4_k.gguf",
                            "VibeVoice-ASR-Streaming 1.5B, Q4_K"),
    "bitnet-tq2": ("cstr/vibevoice-asr-bitnet-GGUF", "vibevoice-asr-bitnet-tq2.gguf",
                   "VibeVoice-ASR-BitNet, ternary language model (TQ2_0)"),
}

# Settings a variant may carry, and the binding call each one goes to. Which of them CrispASR 0.8.50 applies to
# VibeVoice is listed in tools/vibevoice_crispasr_ref.md (from CrispASR's source): context and max_new_tokens only.
SETTING_KEYS = ("context", "max_new_tokens", "language", "temperature", "seed", "do_sample", "top_k", "top_p",
                "beam_size", "best_of", "frequency_penalty", "repetition_penalty", "ask")
PIECE_KEYS = ("piece_seconds", "segments", "whole")
LEFTOVER_NAMES = ("python.exe", "pythonw.exe", "python", "python3")


def say(*parts):
    print(*parts, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------------------------------------------

def _smi(args):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_info(index=0):
    out = _smi(["-i", str(index), "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return None
    name, total, used, free, driver = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    return {"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "driver": driver}


def gpu_used_mib(index=0):
    out = _smi(["-i", str(index), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(out.strip().splitlines()[0])
    except (AttributeError, ValueError, IndexError):
        return None


def gpu_processes(index=0):
    """The processes nvidia-smi lists on the card. On Windows each one's memory is usually reported as N/A."""
    out = _smi(["-i", str(index), "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    procs = []
    for line in (out or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            mem = int(parts[2]) if parts[2].isdigit() else None
            procs.append({"pid": int(parts[0]), "name": parts[1], "used_mib": mem})
    return procs


def leftover_processes(procs):
    """Python (and WhisperJAV) processes on the card other than this one and its parents: the only kind this tool
    offers to end. Desktop and other programs are never offered."""
    try:
        import psutil
    except ImportError:
        return []
    own = {os.getpid()}
    try:
        own.update(p.pid for p in psutil.Process().parents())
    except Exception:
        pass
    found = []
    for pr in procs:
        if pr["pid"] in own:
            continue
        try:
            p = psutil.Process(pr["pid"])
            name = p.name().lower()
            if name in LEFTOVER_NAMES or name.startswith("whisperjav"):
                cmd = " ".join(p.cmdline())[:200]
                found.append({"pid": pr["pid"], "name": p.name(), "cmdline": cmd})
        except Exception:
            continue
    return found


def end_leftovers(found):
    """Ask, one by one, whether to end each leftover process. Ends nothing without a typed y."""
    if not found:
        say("No leftover Python or WhisperJAV process holds the GPU.")
        return 0
    if not sys.stdin or not sys.stdin.isatty():
        say("Leftover processes found, but there is no terminal to answer y/n: nothing ended.")
        return 0
    import psutil
    ended = 0
    for f in found:
        answer = input(f"End process {f['pid']} {f['name']} ({f['cmdline']})? [y/N] ").strip().lower()
        if answer != "y":
            continue
        try:
            p = psutil.Process(f["pid"])
            p.terminate()
            try:
                p.wait(10)
            except psutil.TimeoutExpired:
                p.kill()
            ended += 1
            say(f"  ended {f['pid']}")
        except Exception as e:
            say(f"  could not end {f['pid']}: {e}")
    return ended


class GpuWatch(threading.Thread):
    """Samples the card's used memory every half second: the peak, and whether the card was full (near its total,
    where the Windows NVIDIA driver may move memory into ordinary RAM instead of failing, and runs become slow)."""

    def __init__(self, index, total_mib, full_margin_mib=200):
        super().__init__(daemon=True)
        self.index, self.total, self.margin = index, total_mib, full_margin_mib
        self.peak = None
        self.full = False
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = gpu_used_mib(self.index)
            if v is not None:
                if self.peak is None or v > self.peak:
                    self.peak = v
                if self.total and v >= self.total - self.margin and not self.full:
                    self.full = True
                    say(f"WARNING: the GPU is full ({v} of {self.total} MiB in use). The driver may now be using "
                        f"ordinary RAM: this run may be very slow and its speed figures do not hold.")
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(5)


def print_gpu(index=0):
    info = gpu_info(index)
    if not info:
        say("nvidia-smi gave no answer: no NVIDIA GPU figures.")
        return None
    say(f"GPU {index}: {info['name']}, driver {info['driver']}")
    say(f"  memory: {info['total_mib']} MiB in all, {info['used_mib']} MiB in use, {info['free_mib']} MiB free")
    procs = gpu_processes(index)
    if procs:
        say(f"  programs on the GPU ({len(procs)}; Windows usually reports each one's memory as N/A):")
        for p in procs:
            mem = f"{p['used_mib']} MiB" if p["used_mib"] is not None else "N/A"
            say(f"    {p['pid']:>7}  {mem:>9}  {p['name']}")
    left = leftover_processes(procs)
    if left:
        say("  leftover Python / WhisperJAV processes on the GPU (the tool offers to end these with --end-leftover):")
        for f in left:
            say(f"    {f['pid']:>7}  {f['name']}  {f['cmdline']}")
    else:
        say("  no leftover Python or WhisperJAV process on the GPU")
    return info


# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------

def parse_model_spec(spec):
    if spec in MODELS:
        return MODELS[spec][0], MODELS[spec][1]
    if spec.startswith("hf:"):
        rest = spec[3:].split("/")
        if len(rest) >= 3:
            return "/".join(rest[:2]), "/".join(rest[2:])
    return None


def hf_listing(repo):
    """Hugging Face's file listing of a repository: {file: (size, sha256)}, and the commit it describes."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        sha = json.load(r).get("sha")
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{sha}", timeout=60) as r:
        tree = json.load(r)
    files = {e["path"]: (e.get("size"), (e.get("lfs") or {}).get("oid")) for e in tree if e.get("type") == "file"}
    return files, sha


def cached_file(repo, filename):
    """The file in the Hugging Face cache, whatever revision it was downloaded at, or None."""
    try:
        from huggingface_hub import constants
        root = Path(constants.HF_HUB_CACHE)
    except Exception:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    snaps = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    hits = [p for p in (s / filename for s in snaps.iterdir()) if p.exists()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def resolve_model(spec):
    p = Path(spec)
    if p.is_file():
        return p, None
    rf = parse_model_spec(spec)
    if not rf:
        raise SystemExit(f"Unknown model '{spec}': give a short name ({', '.join(MODELS)}), a local .gguf path, "
                         f"or hf:<owner>/<repo>/<file>.")
    c = cached_file(*rf)
    if not c:
        raise SystemExit(f"'{spec}' ({rf[0]}/{rf[1]}) is not downloaded. Run: download {spec}")
    return c, rf


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def cmd_models(args):
    by_repo = {}
    for short, (repo, fn, desc) in MODELS.items():
        by_repo.setdefault(repo, []).append((short, fn, desc))
    for repo, items in by_repo.items():
        try:
            files, sha = hf_listing(repo)
            say(f"{repo} (Hugging Face commit {sha[:12]})")
        except Exception as e:
            files = {}
            say(f"{repo} (listing not read: {e})")
        for short, fn, desc in items:
            size = files.get(fn, (None, None))[0]
            gb = f"{size / 1e9:6.2f} GB" if size else "    ? GB"
            local = cached_file(repo, fn)
            say(f"  {short:<22} {gb}  {'downloaded' if local else 'not downloaded':<15} {fn}  ({desc})")
    return 0


def cmd_download(args):
    rf = parse_model_spec(args.model)
    if not rf:
        raise SystemExit(f"Unknown model '{args.model}'.")
    repo, fn = rf
    files, sha = hf_listing(repo)
    if fn not in files:
        raise SystemExit(f"{fn} is not in {repo} at commit {sha}.")
    size, oid = files[fn]
    say(f"Downloading {repo}/{fn} at commit {sha} ({size / 1e9:.2f} GB) into the Hugging Face cache ...")
    from huggingface_hub import hf_hub_download
    t = time.time()
    path = Path(hf_hub_download(repo, fn, revision=sha))
    say(f"  done in {time.time() - t:.0f} s: {path}")
    got = sha256_of(path)
    ok = (oid is None or got == oid) and path.stat().st_size == size
    say(f"  size {path.stat().st_size} bytes, SHA-256 {got}: "
        f"{'matches Hugging Face' if ok else 'DOES NOT MATCH Hugging Face (' + str(oid) + ')'}")
    return 0 if ok else 2


# ----------------------------------------------------------------------------------------------------------------
# Audio and pieces
# ----------------------------------------------------------------------------------------------------------------

def find_ffmpeg(given):
    exe = given or shutil.which("ffmpeg")
    if not exe or not Path(exe).exists() and not shutil.which(exe):
        raise SystemExit("ffmpeg not found: put it on PATH or pass --ffmpeg.")
    return exe


def load_audio(ffmpeg, media, start=0.0, duration=None):
    """16 kHz mono audio as float32, extracted as WhisperJAV extracts it (aresample=async=1, 16-bit PCM)."""
    import numpy as np
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(media)]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-af", "aresample=async=1", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise SystemExit(f"ffmpeg could not read {media}: {r.stderr.decode('utf-8', 'replace')[-500:]}")
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0


def quiet_pieces(audio, piece_s, search_share=0.25, min_tail_s=2.0):
    """Pieces of at most piece_s seconds, each cut at the quietest 100 ms in the last quarter of the piece."""
    import numpy as np
    n, length, frame = len(audio), int(piece_s * SR), SR // 10
    pieces, b = [], 0
    while n - b > length:
        lo, hi = b + int(length * (1 - search_share)), b + length
        seg = audio[lo:hi]
        k = len(seg) // frame
        if k >= 1:
            energy = (seg[:k * frame].reshape(k, frame) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * frame + frame // 2
        else:
            cut = hi
        pieces.append((b, cut))
        b = cut
    if pieces and n - b < min_tail_s * SR:
        pieces[-1] = (pieces[-1][0], n)
    elif n > b:
        pieces.append((b, n))
    return [(s / SR, e / SR) for s, e in pieces]


def read_segments(path, offset):
    """start/end pairs (seconds of the media file) from a .jsonl or .json file, made relative to the loaded audio."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("segments") or rows.get("groups") or []
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for r in rows:
        s, e = float(r["start"]) - offset, float(r["end"]) - offset
        if e > s and e > 0:
            out.append((max(0.0, s), e))
    return out


# ----------------------------------------------------------------------------------------------------------------
# One variant
# ----------------------------------------------------------------------------------------------------------------

def srt_time(t):
    ms = max(0, int(round(t * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, entries):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for n, (s, e, text) in enumerate(entries, 1):
            fh.write(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{text}\n\n")


def apply_settings(session, v, previous_extra):
    """Apply a variant's settings to the session; context and the token limit are cleared when a variant leaves
    them out. Returns {setting: 'applied' | 'error: ...'} and the extra settings now possibly in force."""
    done = {}

    def call(key, fn, *a):
        try:
            fn(*a)
            done[key] = "applied (no error)"
        except Exception as e:
            done[key] = f"error: {type(e).__name__}: {e}"

    call("context", session.set_hotwords, str(v.get("context") or ""))
    call("max_new_tokens", session.set_max_new_tokens, int(v.get("max_new_tokens") or 0))
    extra = set()
    if "temperature" in v or "seed" in v:
        call("temperature", session.set_temperature, float(v.get("temperature", 0.0)), int(v.get("seed", 0)))
        extra.add("temperature")
    simple = {"do_sample": (session.set_do_sample, bool), "top_k": (session.set_top_k, int),
              "top_p": (session.set_top_p, float), "beam_size": (session.set_beam_size, int),
              "best_of": (session.set_best_of, int), "frequency_penalty": (session.set_frequency_penalty, float),
              "repetition_penalty": (session.set_repetition_penalty, float), "ask": (session.set_ask, str)}
    for key, (fn, kind) in simple.items():
        if key in v:
            call(key, fn, kind(v[key]))
            extra.add(key)
    stale = sorted(previous_extra - extra)
    if stale:
        done["note"] = (f"set by an earlier variant and not cleared (the binding has no clear call): "
                        f"{', '.join(stale)}")
    return done, previous_extra | extra


def run_variant(session, audio, offset, v, out_dir, watch, args):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    total_s = len(audio) / SR
    if v.get("segments"):
        pieces, how = read_segments(v["segments"], offset), f"segments from {v['segments']}"
    elif v.get("piece_seconds"):
        pieces, how = quiet_pieces(audio, float(v["piece_seconds"])), f"pieces of at most {v['piece_seconds']} s"
    else:
        pieces, how = [(0.0, total_s)], "the whole audio in one call"
    language = v.get("language", "ja")
    language = None if language in (None, "", "none") else language
    applied, args._extra = apply_settings(session, v, getattr(args, "_extra", set()))
    given = ", ".join(f"{k}={v[k]!r}" for k in SETTING_KEYS if k in v) or "binding defaults"
    say(f"\n== variant '{v['name']}': {len(pieces)} call(s), {how}; settings: {given}")

    model_entries, edge_entries, calls, errors, records = [], [], [], 0, []
    used_before = gpu_used_mib(args.gpu_index)
    with open(out_dir / "pieces.jsonl", "w", encoding="utf-8") as fh:
        for i, (s, e) in enumerate(pieces):
            pcm = np.ascontiguousarray(audio[int(round(s * SR)):int(round(e * SR))])
            t = time.perf_counter()
            rec = {"i": i, "start": round(offset + s, 3), "end": round(offset + e, 3)}
            try:
                segs = session.transcribe(pcm, sample_rate=SR, language=language)
                dt = time.perf_counter() - t
                calls.append(dt)
                rec["call_s"] = round(dt, 3)
                rec["segments"] = []
                for sg in segs:
                    toks = list(getattr(sg, "tokens", None) or ())
                    confs = [tk.confidence for tk in toks if getattr(tk, "confidence", None) is not None]
                    item = {"text": str(sg.text or ""), "start": sg.start, "end": sg.end,
                            "speaker": str(getattr(sg, "speaker", "") or ""),
                            "words": len(sg.words or ()), "tokens": len(toks),
                            "token_confidence_mean": round(float(np.mean(confs)), 4) if confs else None}
                    if args.keep_tokens:
                        item["token_list"] = [[str(tk.text), tk.start, tk.end, tk.confidence] for tk in toks]
                    rec["segments"].append(item)
            except Exception as ex:
                rec["call_s"] = round(time.perf_counter() - t, 3)
                rec["error"] = f"{type(ex).__name__}: {ex}"
                errors += 1
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            texts = []
            for sg in rec.get("segments", []):
                x = sg["text"].strip()
                if not x:
                    continue
                texts.append(x)
                a = offset + s + (sg["start"] if sg["start"] is not None else 0.0)
                b = offset + s + (sg["end"] if sg["end"] is not None else e - s)
                label = (sg["speaker"] if args.srt_speaker else "")
                model_entries.append((a, max(a, b), label + x))
            if texts:
                edge_entries.append((offset + s, offset + e, "\n".join(texts)))
            if i == 0 or (i + 1) % max(1, args.progress_every) == 0 or i + 1 == len(pieces):
                say(f"  call {i + 1}/{len(pieces)}: {e - s:6.1f} s of audio in {rec['call_s']:6.2f} s"
                    + (f"  ERROR {rec['error']}" if "error" in rec else f", {len(rec['segments'])} utterance(s)"))
    model_entries.sort(key=lambda x: (x[0], x[1]))
    write_srt(out_dir / "model_times.srt", model_entries)
    write_srt(out_dir / "piece_edges.srt", edge_entries)

    audio_s = sum(r["end"] - r["start"] for r in records if "error" not in r)
    summary = {
        "variant": v, "pieces_how": how, "calls": len(records), "calls_failed": errors,
        "first_errors": [r["error"] for r in records if "error" in r][:3],
        "settings_applied": applied, "language_passed": language,
        "audio_s": round(audio_s, 1), "asr_s": round(sum(calls), 1),
        "speed_x_realtime": round(audio_s / sum(calls), 2) if calls else None,
        "call_s_max": round(max(calls), 2) if calls else None,
        "longest_piece_s": round(max((e - s for s, e in pieces), default=0.0), 1),
        "utterances": sum(len(r.get("segments", [])) for r in records),
        "pieces_without_text": sum(1 for r in records if not "".join(x["text"] for x in r.get("segments", [])).strip()),
        "chars": sum(len(t) for _, _, t in edge_entries),
        "speakers": sorted({x["speaker"].strip() for r in records for x in r.get("segments", []) if x["speaker"]}),
        "gpu_used_before_variant_mib": used_before, "gpu_peak_so_far_mib": watch.peak if watch else None,
        "gpu_was_full": watch.full if watch else None,
    }
    if args.reference:
        summary["score_model_times"] = score(args.reference, out_dir / "model_times.srt", out_dir / "subeval_model_times",
                                             args.timing_threshold)
        summary["score_piece_edges"] = score(args.reference, out_dir / "piece_edges.srt", out_dir / "subeval_piece_edges",
                                             args.timing_threshold)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------------------------------------------
# Scoring (subeval)
# ----------------------------------------------------------------------------------------------------------------

def score(ref, srt, out, threshold):
    if not (REPO_ROOT / "subeval").is_dir():
        return {"error": f"subeval not found in {REPO_ROOT}"}
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "subeval", "score", str(ref), str(srt), "--out", str(out)],
                       cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    res = Path(out) / "result.json"
    if not res.exists():
        return {"error": f"subeval exit {r.returncode}: {r.stderr[-800:]}"}
    d = json.loads(res.read_text(encoding="utf-8"))
    s = d["summary"]
    keep = ("ref_lines", "ref_chars", "test_lines", "cer", "wrong_rate", "missing_rate", "extra_rate",
            "missed_lines", "hallucination_lines", "start_measured", "end_measured", "start_abs_median",
            "end_abs_median")
    out_d = {k: s.get(k) for k in keep}
    # Timing beyond a stated threshold, early and late, for starts and ends (subeval: + means our line is later).
    starts = [x["start_err"] for c in d["clips"] for x in c["ref"] if x.get("start_err") is not None]
    ends = [x["end_err"] for c in d["clips"] for x in c["ref"] if x.get("end_err") is not None]
    out_d.update({"threshold_s": threshold,
                  "starts_early": sum(1 for x in starts if x < -threshold),
                  "starts_late": sum(1 for x in starts if x > threshold),
                  "ends_early": sum(1 for x in ends if x < -threshold),
                  "ends_late": sum(1 for x in ends if x > threshold)})
    return out_d


# ----------------------------------------------------------------------------------------------------------------
# The run command
# ----------------------------------------------------------------------------------------------------------------

def variants_from(args):
    if args.variants:
        rows = json.loads(Path(args.variants).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise SystemExit("--variants: a JSON list of objects, each with a 'name'.")
        for i, r in enumerate(rows):
            r.setdefault("name", f"v{i + 1}")
            unknown = set(r) - set(SETTING_KEYS) - set(PIECE_KEYS) - {"name"}
            if unknown:
                raise SystemExit(f"--variants: unknown key(s) {sorted(unknown)} in '{r['name']}'. "
                                 f"Known: {', '.join(SETTING_KEYS + PIECE_KEYS)}")
        return rows
    v = {"name": args.name}
    for key in SETTING_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            v[key] = val
    if args.segments:
        v["segments"] = args.segments
    elif args.piece_seconds:
        v["piece_seconds"] = args.piece_seconds
    return [v]


def fmt(x, spec="{:.3f}"):
    if x is None:
        return "–"
    return spec.format(x) if isinstance(x, float) else str(x)


def write_results_md(out, run, summaries):
    lines = [f"# VibeVoice through CrispASR: {run['model_label']}", "",
             f"Tool `tools/vibevoice_crispasr_ref.py` version {TOOL_VERSION}; run {run['when']}.", "",
             "| Fact | Value |", "|---|---|"]
    for k in ("model_file", "model_size_bytes", "model_sha256", "sha256_vs_hugging_face", "crispasr", "backend",
              "input", "audio_window", "reference", "crispasr_cutting", "gpu", "gpu_free_before_mib",
              "model_open_s", "gpu_after_open_mib", "gpu_peak_mib", "gpu_over_before_mib", "gpu_was_full"):
        lines.append(f"| {k} | {run.get(k, '–')} |")
    lines += ["", "## Calls", "",
              "| Variant | How the audio was cut | Calls | Failed | Longest piece (s) | Audio (s) | ASR time (s) | "
              "x real time | Longest call (s) | Utterances | Pieces without text | Characters |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['variant']['name']} | {s['pieces_how']} | {s['calls']} | {s['calls_failed']} | "
                     f"{s['longest_piece_s']} | {s['audio_s']} | {s['asr_s']} | {fmt(s['speed_x_realtime'], '{:.2f}')} | "
                     f"{fmt(s['call_s_max'], '{:.2f}')} | {s['utterances']} | {s['pieces_without_text']} | {s['chars']} |")
    if any("score_piece_edges" in s for s in summaries):
        lines += ["", "## Text (subeval against the reference; each piece's text at the piece's edges, so the model's "
                      "own times do not affect it)", "",
                  "| Variant | CER | Wrong | Missing | Extra | Missed lines | Hallucination lines | Test lines |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in summaries:
            sc = s.get("score_piece_edges") or {}
            lines.append(f"| {s['variant']['name']} | {fmt(sc.get('cer'))} | {fmt(sc.get('wrong_rate'))} | "
                         f"{fmt(sc.get('missing_rate'))} | {fmt(sc.get('extra_rate'))} | {fmt(sc.get('missed_lines'))} | "
                         f"{fmt(sc.get('hallucination_lines'))} | {fmt(sc.get('test_lines'))} |")
        thr = summaries[0].get("score_model_times", {}).get("threshold_s")
        lines += ["", f"## The model's own times (subeval; one subtitle per utterance at the model's start and end)", "",
                  f"Early / late = more than {thr} s before / after the reference line's start or end, counted only "
                  f"where subeval measures that edge.", "",
                  "| Variant | CER | Missed lines | Hallucination lines | Starts measured | Starts early | Starts late | "
                  "Ends measured | Ends early | Ends late | Start median abs error (s) | End median abs error (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in summaries:
            sc = s.get("score_model_times") or {}
            lines.append(f"| {s['variant']['name']} | {fmt(sc.get('cer'))} | {fmt(sc.get('missed_lines'))} | "
                         f"{fmt(sc.get('hallucination_lines'))} | {fmt(sc.get('start_measured'))} | "
                         f"{fmt(sc.get('starts_early'))} | {fmt(sc.get('starts_late'))} | {fmt(sc.get('end_measured'))} | "
                         f"{fmt(sc.get('ends_early'))} | {fmt(sc.get('ends_late'))} | {fmt(sc.get('start_abs_median'))} | "
                         f"{fmt(sc.get('end_abs_median'))} |")
    lines += ["", "## Settings as applied", ""]
    for s in summaries:
        lines.append(f"- {s['variant']['name']}: " + "; ".join(f"{k}: {v}" for k, v in s["settings_applied"].items()))
    lines += ["", "CER, wrong, missing, extra, missed lines and hallucination lines: as defined in "
                  "`subeval/METHOD.md` (subeval 0.1.0). Files per variant: `pieces.jsonl` (every call's result as "
                  "returned), `model_times.srt`, `piece_edges.srt`, `summary.json`, the two subeval folders.", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_run(args):
    if args.crisp_cutting == "off":
        os.environ["CRISPASR_SESSION_AUTOCHUNK"] = "0"
    elif args.crisp_cutting.isdigit():
        os.environ["CRISPASR_SESSION_CHUNK_SECONDS"] = args.crisp_cutting
    cutting = {"default": "CrispASR's own (audio over 30 s cut into pieces of at most 30 s at quiet points)",
               "off": "off (CRISPASR_SESSION_AUTOCHUNK=0): each call's audio goes to the model whole"}.get(
        args.crisp_cutting, f"CrispASR cuts audio over {args.crisp_cutting} s (CRISPASR_SESSION_CHUNK_SECONDS)")

    path, rf = resolve_model(args.model)
    size = path.stat().st_size
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = variants_from(args)
    media = Path(args.input)
    if not media.is_file():
        raise SystemExit(f"Input not found: {media}")
    if args.reference and not Path(args.reference).is_file():
        raise SystemExit(f"Reference not found: {args.reference}")
    run = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "model_label": args.model, "model_file": str(path),
           "model_size_bytes": size, "input": str(media), "reference": args.reference or "–",
           "crispasr_cutting": cutting}

    # GPU room, before anything is loaded
    info = None if args.cpu else gpu_info(args.gpu_index)
    if info:
        need = int(size / 2 ** 20) + args.margin_mib
        budget = info["total_mib"] - args.reserve_mib
        run["gpu"] = f"{info['name']}, {info['total_mib']} MiB, driver {info['driver']}"
        say(f"GPU: {info['free_mib']} MiB free of {info['total_mib']} MiB. The model file is {size / 2 ** 20:.0f} MiB; "
            f"with a margin of {args.margin_mib} MiB it needs about {need} MiB. Budget (card less "
            f"{args.reserve_mib} MiB kept for the system): {budget} MiB.")
        if need > budget and not args.force:
            say("Not enough room in the budget for this model; nothing loaded (use --force to try anyway).")
            return 3
        if info["free_mib"] < need:
            left = leftover_processes(gpu_processes(args.gpu_index))
            if args.end_leftover:
                end_leftovers(left)
                time.sleep(2)
                info = gpu_info(args.gpu_index)
            if info["free_mib"] < need and not args.force:
                say(f"Only {info['free_mib']} MiB free; nothing loaded. What is on the GPU now:")
                print_gpu(args.gpu_index)
                say("Close programs you do not need (Windows does not say how much each one holds), "
                    "or use --end-leftover for leftover Python processes, or --force to try anyway.")
                return 3
        run["gpu_free_before_mib"] = info["free_mib"]
    elif not args.cpu:
        say("No NVIDIA GPU figures (nvidia-smi gave no answer); the run goes on without the memory check.")

    if not args.no_hash:
        say(f"Checking the model file's SHA-256 ({size / 1e9:.2f} GB) ...")
        run["model_sha256"] = sha256_of(path)
        if rf:
            try:
                files, _ = hf_listing(rf[0])
                oid = files.get(rf[1], (None, None))[1]
                run["sha256_vs_hugging_face"] = ("matches" if oid == run["model_sha256"] else f"differs ({oid})") \
                    if oid else "not listed"
            except Exception as e:
                run["sha256_vs_hugging_face"] = f"not checked ({type(e).__name__})"

    ffmpeg = find_ffmpeg(args.ffmpeg)
    audio = load_audio(ffmpeg, media, args.start, args.duration)
    run["audio_window"] = f"from {args.start} s, {len(audio) / SR:.1f} s of audio"
    say(f"Audio: {len(audio) / SR:.1f} s from {media.name}")

    import crispasr
    run["crispasr"] = getattr(crispasr, "__version__", "?")
    watch = GpuWatch(args.gpu_index, info["total_mib"]) if info else None
    base = gpu_used_mib(args.gpu_index) if info else None
    if watch:
        watch.start()
    t = time.time()
    try:
        session = crispasr.Session(str(path), n_threads=args.threads, use_gpu=not args.cpu,
                                   flash_attn=not args.no_flash_attn)
    except Exception as e:
        if watch:
            watch.stop()
        say(f"CrispASR could not open the model: {type(e).__name__}: {e}")
        return 4
    run["model_open_s"] = round(time.time() - t, 1)
    run["backend"] = getattr(session, "backend", "?")
    run["gpu_after_open_mib"] = gpu_used_mib(args.gpu_index) if info else None
    say(f"Model open as backend '{run['backend']}' in {run['model_open_s']} s"
        + (f"; GPU now {run['gpu_after_open_mib']} MiB in use" if info else ""))

    summaries = []
    try:
        for v in variants:
            summaries.append(run_variant(session, audio, args.start, v, out / v["name"], watch, args))
            write_results_md(out, run, summaries)
    finally:
        session.close()
        del session
        gc.collect()
        if watch:
            watch.stop()
            run["gpu_peak_mib"] = watch.peak
            run["gpu_over_before_mib"] = (watch.peak - base) if watch.peak and base else None
            run["gpu_was_full"] = watch.full
            time.sleep(1)
            run["gpu_after_close_mib"] = gpu_used_mib(args.gpu_index)
    (out / "results.json").write_text(json.dumps({"run": run, "variants": summaries}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    write_results_md(out, run, summaries)
    say(f"\nModel closed. GPU peak {run.get('gpu_peak_mib')} MiB ({run.get('gpu_over_before_mib')} MiB over the "
        f"start), now {run.get('gpu_after_close_mib')} MiB. Results: {out / 'results.md'}")
    return 0


# ----------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="vibevoice_crispasr_ref.py",
        description="VibeVoice-ASR through CrispASR's Python binding: show the GPU, list and download VibeVoice "
                    "models, and run a model on a media file with chosen settings, scored against a reference SRT.")
    p.add_argument("--gpu-index", type=int, default=0, help="Which NVIDIA GPU nvidia-smi reports on (default 0).")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("gpu", help="Show the GPU's memory, the programs on it, and leftover Python processes.")
    sub.add_parser("models", help="List the known VibeVoice files: size on Hugging Face, downloaded or not.")
    d = sub.add_parser("download", help="Download one model into the Hugging Face cache and check its SHA-256.")
    d.add_argument("model", help=f"Short name ({', '.join(MODELS)}) or hf:<owner>/<repo>/<file>.")

    r = sub.add_parser("run", help="Run a model on a media file with one or more variants of settings.")
    r.add_argument("--model", required=True, help="Short name, local .gguf path, or hf:<owner>/<repo>/<file> "
                                                  "(downloaded first with 'download').")
    r.add_argument("--input", required=True, help="Media file (any file ffmpeg reads).")
    r.add_argument("--out", required=True, help="Output folder (one subfolder per variant).")
    r.add_argument("--reference", help="Reference SRT: score each variant with subeval.")
    r.add_argument("--start", type=float, default=0.0, help="Start of the audio window in seconds (default 0).")
    r.add_argument("--duration", type=float, help="Length of the audio window in seconds (default: to the end).")
    r.add_argument("--piece-seconds", type=float,
                   help="Cut the audio into pieces of at most this many seconds, at the quietest 100 ms of each "
                        "piece's last quarter; one call per piece. Default: the whole window in one call.")
    r.add_argument("--segments", help="A .jsonl/.json file of {start, end} in seconds of the media file: one call "
                                      "per entry (for example the VAD groups of a WhisperJAV --debug run).")
    r.add_argument("--variants", help="A JSON list of variants, each an object with a 'name' and any of: "
                                      + ", ".join(SETTING_KEYS + PIECE_KEYS) + ". The model is loaded once.")
    r.add_argument("--name", default="run", help="Name of the single variant when --variants is not given.")
    g = r.add_argument_group("settings (for a single variant; see the .md for which ones CrispASR applies to "
                             "VibeVoice)")
    g.add_argument("--context", help="Context / hotword text, put into VibeVoice's prompt (binding set_hotwords).")
    g.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, help="Token limit per call (0 = model default).")
    g.add_argument("--language", default="ja", help="Language passed with each call (default ja; 'none' for none).")
    g.add_argument("--temperature", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--do-sample", dest="do_sample", type=int, choices=(0, 1))
    g.add_argument("--top-k", dest="top_k", type=int)
    g.add_argument("--top-p", dest="top_p", type=float)
    g.add_argument("--beam-size", dest="beam_size", type=int)
    g.add_argument("--best-of", dest="best_of", type=int)
    g.add_argument("--frequency-penalty", dest="frequency_penalty", type=float)
    g.add_argument("--repetition-penalty", dest="repetition_penalty", type=float)
    g.add_argument("--ask", help="Free-form prompt (binding set_ask).")
    e = r.add_argument_group("engine and GPU")
    e.add_argument("--crisp-cutting", default="default",
                   help="CrispASR's own cutting of long audio: 'default' (over 30 s: pieces of at most 30 s), 'off', "
                        "or a number of seconds. Set once per run.")
    e.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)), help="CPU threads (default up to 8).")
    e.add_argument("--cpu", action="store_true", help="Run on the CPU (no GPU).")
    e.add_argument("--no-flash-attn", action="store_true", help="Open the model without flash attention.")
    e.add_argument("--reserve-mib", type=int, default=2048,
                   help="GPU memory kept for the system and other programs (default 2048 MiB).")
    e.add_argument("--margin-mib", type=int, default=512,
                   help="Memory counted on top of the model file for its working memory (default 512 MiB).")
    e.add_argument("--end-leftover", action="store_true",
                   help="When the GPU is short of room, offer to end leftover Python / WhisperJAV processes on it, "
                        "asking y/n for each. No other program is ever offered.")
    e.add_argument("--force", action="store_true", help="Load the model even when the memory check says no.")
    o = r.add_argument_group("output")
    o.add_argument("--ffmpeg", help="Path to ffmpeg (default: from PATH).")
    o.add_argument("--timing-threshold", type=float, default=0.2,
                   help="Seconds beyond which a start or end counts as early or late (default 0.2).")
    o.add_argument("--srt-speaker", action="store_true", help="Put the speaker label in front of each subtitle.")
    o.add_argument("--keep-tokens", action="store_true", help="Keep every token in pieces.jsonl.")
    o.add_argument("--progress-every", type=int, default=10, help="Print a line every N calls (default 10).")
    o.add_argument("--no-hash", action="store_true", help="Skip the model file's SHA-256 check.")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.command == "gpu":
        return 0 if print_gpu(args.gpu_index) else 2
    if args.command == "models":
        return cmd_models(args)
    if args.command == "download":
        return cmd_download(args)
    return cmd_run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
