#!/usr/bin/env python
"""Test any GGUF model through CrispASR before WhisperJAV takes it (owner W568, W569, 2026-11-06).

Seven stages, each a command, each writing its evidence under --out ROOT/<model>/:

  catalog   what this CrispASR can run: its registry and backends; which files WhisperJAV ships; what is new
  fetch     download a model at a pinned commit, check size and SHA-256, read the licence
  open      open it on the GPU and the CPU: backend, open time, GPU memory, memory freed after closing
  payload   is the object CrispASR returns consistent, and which of its fields carry usable times (document B:
            payload disassembly)? fixed test audio, checks, a proposed pipeline row
  measure   the Netflix clips (or any clips) scored with subeval: the model's own times, the VAD groups' edges,
            and the subtitles WhisperJAV's own unchanged code would build (read-only import, W569)
  listen    files for a hand test: SRTs side by side and a list of the lines that differ
  report    one results file: every stage's facts, a draft manifest entry and pipeline row

  python tools/crispasr_model_test.py catalog --out D:\\runs
  python tools/crispasr_model_test.py fetch haas-ja-v1.0-q8_0
  python tools/crispasr_model_test.py open --model haas-ja-v1.0-q8_0 --out D:\\runs
  python tools/crispasr_model_test.py payload --model haas-ja-v1.0-q8_0 --input clip.mkv --reference clip.ja.srt --out D:\\runs
  python tools/crispasr_model_test.py measure --model haas-ja-v1.0-q8_0 --clips test_media\\Ground_Truths\\Netflix \\
      --groups-from D:\\runs\\pipeline_parakeet --out D:\\runs
  python tools/crispasr_model_test.py listen --model haas-ja-v1.0-q8_0 --baseline parakeet-tdt-0.6b-ja --out D:\\runs
  python tools/crispasr_model_test.py report --model haas-ja-v1.0-q8_0 --out D:\\runs

Every command is described in tools/crispasr_model_test.md. The models are described in tools/crispasr_models.json.
The model work of every stage runs in a child process (the hidden command _worker), so a native crash of CrispASR is
recorded as a finding. Nothing in WhisperJAV is changed. Stages catalog to payload import nothing from WhisperJAV;
measure imports its speech segmenter, engine.segments_from_binding, crispasr/lines.py and crispasr/cleanup.py,
unchanged, and records the commit.

Exit status: 0 done; 2 a setup or input problem; 3 not enough free GPU memory (nothing loaded); 4 CrispASR could not
open the model; 5 the child process ended abnormally (a crash; what it wrote before is kept).
"""
from __future__ import annotations

import argparse
import ast
import difflib
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
from types import SimpleNamespace

TOOL_VERSION = "1 (2026-11-06)"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = Path(__file__).resolve().with_name("crispasr_models.json")
MANIFEST = REPO_ROOT / "whisperjav" / "crispasr" / "manifest.py"
NETFLIX = REPO_ROOT / "test_media" / "Ground_Truths" / "Netflix"
SR = 16000          # what the binding's Session.transcribe takes
TOL_S = 0.05        # a time this far outside the call's audio still counts as inside (rounding)
MEDIA_EXT = (".mkv", ".mp4", ".wav", ".flac", ".mp3", ".m4a", ".webm", ".avi", ".mov")
LEFTOVER_NAMES = ("python.exe", "pythonw.exe", "python", "python3")


def say(*parts):
    print(*parts, flush=True)


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break   # a line cut off by a crash
    return out


# ----------------------------------------------------------------------------------------------------------------
# GPU (as tools/vibevoice_crispasr_ref.py)
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
    out = _smi(["-i", str(index), "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    procs = []
    for line in (out or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "name": parts[1],
                          "used_mib": int(parts[2]) if parts[2].isdigit() else None})
    return procs


def leftover_processes(procs):
    """Python and WhisperJAV processes on the card other than this one and its parents."""
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
                found.append({"pid": pr["pid"], "name": p.name(), "cmdline": " ".join(p.cmdline())[:200]})
        except Exception:
            continue
    return found


class GpuWatch(threading.Thread):
    """Samples the card's used memory every half second: the peak, and whether the card was full."""

    def __init__(self, index, total_mib, full_margin_mib=200):
        super().__init__(daemon=True)
        self.index, self.total, self.margin = index, total_mib, full_margin_mib
        self.peak = None
        self.full = False
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            v = gpu_used_mib(self.index)
            if v is not None:
                if self.peak is None or v > self.peak:
                    self.peak = v
                if self.total and v >= self.total - self.margin and not self.full:
                    self.full = True
                    say(f"WARNING: the GPU is full ({v} of {self.total} MiB in use); the driver may now use ordinary "
                        f"RAM, and speed figures do not hold.")
            self._halt.wait(0.5)

    def stop(self):
        self._halt.set()
        self.join(5)


def print_gpu(index=0):
    info = gpu_info(index)
    if not info:
        say("nvidia-smi gave no answer: no NVIDIA GPU figures.")
        return None
    say(f"GPU {index}: {info['name']}, driver {info['driver']}")
    say(f"  memory: {info['total_mib']} MiB in all, {info['used_mib']} MiB in use, {info['free_mib']} MiB free")
    for p in gpu_processes(index):
        say(f"    {p['pid']:>7}  {p['used_mib'] if p['used_mib'] is not None else 'N/A':>6}  {p['name']}")
    left = leftover_processes(gpu_processes(index))
    say("  leftover Python / WhisperJAV processes: " + (", ".join(f"{f['pid']} {f['name']}" for f in left) or "none"))
    return info


def gpu_room(args, size_bytes):
    """None when there is room (or no NVIDIA figures, or --cpu), else a message; and the GPU info."""
    if getattr(args, "cpu", False):
        return None, None
    info = gpu_info(args.gpu_index)
    if not info:
        say("No NVIDIA GPU figures (nvidia-smi gave no answer); no memory check.")
        return None, None
    need = int(size_bytes / 2 ** 20) + args.margin_mib
    budget = info["total_mib"] - args.reserve_mib
    say(f"GPU: {info['free_mib']} MiB free of {info['total_mib']} MiB; the model file is {size_bytes / 2 ** 20:.0f} MiB, "
        f"about {need} MiB with a margin of {args.margin_mib} MiB; budget {budget} MiB.")
    if args.force:
        return None, info
    if need > budget:
        return "Not enough room in the budget for this model; nothing loaded (--force to try anyway).", info
    if info["free_mib"] < need:
        left = leftover_processes(gpu_processes(args.gpu_index))
        names = ", ".join(f"{f['pid']} {f['name']}" for f in left) or "none"
        return (f"Only {info['free_mib']} MiB free; nothing loaded. Leftover Python / WhisperJAV processes on the GPU: "
                f"{names}. Close what you do not need, or --force."), info
    return None, info


# ----------------------------------------------------------------------------------------------------------------
# Models: the data file, Hugging Face, the cache
# ----------------------------------------------------------------------------------------------------------------

def load_models():
    return load_json(MODELS_FILE)


def model_entry(name, models=None):
    models = models or load_models()
    entry = models["models"].get(name)
    if entry is None:
        raise SystemExit(f"Unknown model '{name}'. Known: {', '.join(models['models'])}. Add an entry to "
                         f"{MODELS_FILE.name}, or give a local .gguf path.")
    return entry


def hf_api(path):
    with urllib.request.urlopen(f"https://huggingface.co/api/{path}", timeout=60) as r:
        return json.load(r)


def hf_listing(repo, revision=None):
    """{file: (size, sha256)}, the commit, and the repository's card data (licence)."""
    info = hf_api(f"models/{repo}" + (f"/revision/{revision}" if revision else ""))
    sha = info.get("sha") or revision
    tree = hf_api(f"models/{repo}/tree/{sha}")
    files = {e["path"]: (e.get("size"), (e.get("lfs") or {}).get("oid")) for e in tree if e.get("type") == "file"}
    card = info.get("cardData") or {}
    return files, sha, card


def hf_cache_root():
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def cached_file(repo, filename, revision=None):
    """The file in the Hugging Face cache: at ``revision`` when given, else the newest snapshot that has it."""
    snaps = hf_cache_root() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    if revision:
        p = snaps / revision / filename
        return p if p.exists() else None
    hits = [p for p in (s / filename for s in snaps.iterdir()) if p.exists()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def resolve_model(spec):
    """(path, name, entry) for a data-file name or a local .gguf path."""
    p = Path(spec)
    if p.suffix.lower() == ".gguf" and p.is_file():
        return p, p.stem, {"label": p.name, "backend": None, "row": None}
    entry = model_entry(spec)
    c = cached_file(entry["repo"], entry["file"], entry.get("revision"))
    if not c:
        raise SystemExit(f"'{spec}' ({entry['repo']}/{entry['file']}) is not downloaded. Run: fetch {spec}")
    return c, spec, entry


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def manifest_files(path=MANIFEST):
    """{filename: (short name, repo_id)} of the GGUF files WhisperJAV ships, read from manifest.py's text (no import)."""
    out = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ModelFile" and len(node.args) >= 5:
            vals = [a.value if isinstance(a, ast.Constant) else None for a in node.args[:5]]
            if all(isinstance(v, str) for v in (vals[0], vals[3], vals[4])):
                out[vals[4]] = (vals[0], vals[3])
    return out


def is_speech_to_text(name, models):
    rule = models.get("not_speech_to_text", {})
    return name not in set(rule.get("names", ())) and not any(name.startswith(p) for p in rule.get("prefixes", ()))


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "whisperjav"], cwd=str(REPO_ROOT),
                               capture_output=True, text=True).stdout.strip()
        return r.stdout.strip() + (" (whisperjav/ has uncommitted changes)" if dirty else "")
    except OSError:
        return "?"


# ----------------------------------------------------------------------------------------------------------------
# Audio, pieces, groups, SRT
# ----------------------------------------------------------------------------------------------------------------

def find_ffmpeg(given=None):
    exe = given or shutil.which("ffmpeg")
    if not exe or (not Path(exe).exists() and not shutil.which(exe)):
        raise SystemExit("ffmpeg not found: put it on PATH or pass --ffmpeg.")
    return exe


def load_audio(ffmpeg, media, start=0.0, duration=None):
    """16 kHz mono float32, extracted as WhisperJAV extracts it (modules/audio_extraction.py: aresample=async=1,
    16-bit PCM, 16 kHz, mono)."""
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
    """Pieces of at most piece_s seconds, each cut at the quietest 100 ms of its last quarter (seconds)."""
    import numpy as np
    n, length, frame = len(audio), int(piece_s * SR), SR // 10
    pieces, b = [], 0
    while n - b > length:
        lo, hi = b + int(length * (1 - search_share)), b + length
        seg = audio[lo:hi]
        k = len(seg) // frame
        cut = lo + int(np.argmin((seg[:k * frame].reshape(k, frame) ** 2).mean(axis=1))) * frame + frame // 2 \
            if k >= 1 else hi
        pieces.append((b, cut))
        b = cut
    if pieces and n - b < min_tail_s * SR:
        pieces[-1] = (pieces[-1][0], n)
    elif n > b:
        pieces.append((b, n))
    return [(s / SR, e / SR) for s, e in pieces]


def pipeline_groups(path):
    """A WhisperJAV --debug per-group record (raw_subs/<name>.<pipeline>.groups.jsonl): each VAD group's scene, its
    start and end in media-file time, and the segments the pipeline built its subtitles from."""
    rows = read_jsonl(path)
    return [{"scene": int(r.get("scene", 0)), "start": float(r["start"]), "end": float(r["end"]),
             "aligned": bool(r.get("aligned")), "segments": r.get("segments") or []} for r in rows]


def srt_time(t):
    ms = max(0, int(round(t * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, entries):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for n, (s, e, text) in enumerate(entries, 1):
            fh.write(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{text}\n\n")


def read_srt(path):
    """[(start, end, text)] of an SRT file."""
    def secs(t):
        h, m, rest = t.strip().split(":")
        s, ms = rest.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    out = []
    blocks = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").split("\n\n")
    for b in blocks:
        lines = [x for x in b.strip().split("\n") if x.strip() != ""]
        for i, line in enumerate(lines):
            if "-->" in line:
                a, z = line.split("-->")
                out.append((secs(a), secs(z.split()[0]), "\n".join(lines[i + 1:])))
                break
    return out


def find_clips(folder):
    """(media, reference SRT) pairs in a folder: <stem>.<media ext> with <stem>.ja.srt or <stem>.srt beside it."""
    pairs = []
    for m in sorted(Path(folder).iterdir()):
        if m.suffix.lower() not in MEDIA_EXT or m.suffix.lower() == ".wav" and m.with_suffix(".mkv").exists():
            continue
        for ref in (m.with_name(m.stem + ".ja.srt"), m.with_suffix(".srt")):
            if ref.exists():
                pairs.append((m, ref))
                break
    return pairs


# ----------------------------------------------------------------------------------------------------------------
# The child process: open the model, run the calls, write every result as returned
# ----------------------------------------------------------------------------------------------------------------

def segment_record(sg, keep_raw=True):
    words = [[str(w.text), w.start, w.end, float(w.confidence)] for w in (sg.words or ())]
    toks = []
    for t in (getattr(sg, "tokens", None) or ()):
        item = [str(t.text), t.start, t.end, float(t.confidence)]
        if keep_raw:
            item.append((getattr(t, "raw_text", b"") or b"").hex())
        toks.append(item)
    return {"text": str(sg.text or ""), "start": float(sg.start), "end": float(sg.end),
            "no_speech_prob": float(getattr(sg, "no_speech_prob", -1.0)),
            "speaker": str(getattr(sg, "speaker", "") or ""), "words": words, "tokens": toks}


def apply_settings(session, settings):
    done = {}
    setters = {"context": ("set_hotwords", str), "max_new_tokens": ("set_max_new_tokens", int),
               "do_sample": ("set_do_sample", bool), "top_k": ("set_top_k", int), "top_p": ("set_top_p", float),
               "beam_size": ("set_beam_size", int), "best_of": ("set_best_of", int),
               "frequency_penalty": ("set_frequency_penalty", float),
               "repetition_penalty": ("set_repetition_penalty", float), "ask": ("set_ask", str)}
    for key, val in (settings or {}).items():
        try:
            if key == "temperature":
                session.set_temperature(float(val), int(settings.get("seed", 0)))
            elif key == "seed":
                continue
            else:
                name, kind = setters[key]
                getattr(session, name)(kind(val))
            done[key] = "applied (no error)"
        except Exception as e:
            done[key] = f"error: {type(e).__name__}: {e}"
    return done


def worker(job_path):
    """Runs one job (a JSON file) and writes one JSON line per event to job['out']: open, each call, close."""
    import numpy as np
    job = load_json(job_path)
    out = open(job["out"], "w", encoding="utf-8")

    def emit(rec):
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()

    import crispasr
    version = getattr(crispasr, "__version__", "?")
    gidx = job.get("gpu_index", 0)
    audio_of = {}
    for media in {c["media"] for c in job["calls"] if c.get("media")}:
        audio_of[media] = load_audio(job["ffmpeg"], media)
    for round_no in range(int(job.get("open_times", 1))):
        before = gpu_used_mib(gidx)
        t = time.time()
        try:
            session = crispasr.Session(job["model"], n_threads=job.get("threads", 8), use_gpu=job.get("use_gpu", True),
                                       flash_attn=job.get("flash_attn", True))
        except Exception as e:
            emit({"kind": "open_failed", "round": round_no, "error": f"{type(e).__name__}: {e}", "crispasr": version})
            out.close()
            return 4
        opened = {"kind": "open", "round": round_no, "backend": getattr(session, "backend", "?"),
                  "open_s": round(time.time() - t, 2), "crispasr": version, "gpu_before_mib": before,
                  "gpu_after_open_mib": gpu_used_mib(gidx), "use_gpu": job.get("use_gpu", True),
                  "settings": apply_settings(session, job.get("settings")),
                  "env": {k: v for k, v in os.environ.items() if k.startswith("CRISPASR_")}}
        emit(opened)
        if round_no == 0:
            for i, c in enumerate(job["calls"]):
                if c.get("zeros") is not None:
                    pcm = np.zeros(int(round(float(c["zeros"]) * SR)), dtype=np.float32)
                else:
                    a = audio_of[c["media"]]
                    pcm = np.ascontiguousarray(a[int(round(c["start"] * SR)):int(round(c["end"] * SR))])
                rec = {"kind": "call", "i": i, "id": c["id"], "seconds": round(len(pcm) / SR, 3),
                       "language": c.get("language", "ja")}
                for k in ("clip", "group", "scene", "start", "end", "probe", "run"):
                    if k in c:
                        rec[k] = c[k]
                t = time.perf_counter()
                try:
                    segs = session.transcribe(pcm, sample_rate=SR, language=c.get("language", "ja"))
                    rec["call_s"] = round(time.perf_counter() - t, 4)
                    rec["segments"] = [segment_record(sg) for sg in segs]
                except Exception as e:
                    rec["call_s"] = round(time.perf_counter() - t, 4)
                    rec["error"] = f"{type(e).__name__}: {e}"
                emit(rec)
                every = max(1, int(job.get("progress_every", 25)))
                if i == 0 or (i + 1) % every == 0 or i + 1 == len(job["calls"]):
                    say(f"  call {i + 1}/{len(job['calls'])}: {rec['seconds']:.1f} s in {rec['call_s']:.2f} s"
                        + (f"  ERROR {rec['error']}" if "error" in rec else ""))
        session.close()
        del session
        gc.collect()
        time.sleep(1.0)
        emit({"kind": "closed", "round": round_no, "gpu_after_close_mib": gpu_used_mib(gidx)})
    out.close()
    return 0


def run_job(job, folder, env_extra=None, gpu_index=0, total_mib=None):
    """Runs a job in a child process; returns (events, exit code, GPU peak, card was full)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    job = dict(job)
    job.setdefault("gpu_index", gpu_index)
    job["out"] = str(folder / "events.jsonl")
    dump(folder / "job.json", job)
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("CRISPASR_SESSION_") and k not in (env_extra or {}):
            del env[k]   # each job states its own CrispASR switches
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    env["PYTHONIOENCODING"] = "utf-8"
    watch = GpuWatch(gpu_index, total_mib) if total_mib else None
    if watch:
        watch.start()
    t = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "_worker", str(folder / "job.json")],
                       env=env)
    if watch:
        watch.stop()
    events = read_jsonl(folder / "events.jsonl")
    meta = {"exit_code": r.returncode, "wall_s": round(time.time() - t, 1), "gpu_peak_mib": watch.peak if watch else None,
            "gpu_was_full": watch.full if watch else None, "env": env_extra or {}}
    if r.returncode not in (0, 4):
        calls = [e for e in events if e.get("kind") == "call"]
        meta["crash"] = (f"the child process ended with exit code {r.returncode} after {len(calls)} of "
                         f"{len(job['calls'])} calls (next call: {job['calls'][len(calls)]['id'] if len(calls) < len(job['calls']) else 'none'})")
        say("CRASH: " + meta["crash"])
    dump(folder / "run.json", meta)
    return events, r.returncode, meta


def base_job(path, args, use_gpu=True):
    return {"model": str(path), "use_gpu": use_gpu and not getattr(args, "cpu", False), "threads": args.threads,
            "flash_attn": not args.no_flash_attn, "ffmpeg": find_ffmpeg(getattr(args, "ffmpeg", None)),
            "settings": {}, "calls": [], "progress_every": getattr(args, "progress_every", 25)}


# ----------------------------------------------------------------------------------------------------------------
# Checks on the returned object (pure; tests/test_crispasr_model_test.py)
# ----------------------------------------------------------------------------------------------------------------

def rebuild_token_text(tokens):
    """The tokens' text, from their bytes where the binding gave them (a UTF-8 character may span two tokens)."""
    raw = b""
    texts = []
    for t in tokens:
        hexed = t[4] if len(t) > 4 else ""
        if hexed:
            raw += bytes.fromhex(hexed)
        else:
            if raw:
                texts.append(raw.decode("utf-8", "replace"))
                raw = b""
            texts.append(str(t[0]))
    if raw:
        texts.append(raw.decode("utf-8", "replace"))
    return "".join(texts)


def _norm(text):
    return "".join(str(text).split())


def call_checks(rec, tol=TOL_S):
    """The facts of one call's result: bounds, order, overlap, which units carry times, text rebuilt from tokens,
    confidence values."""
    dur = float(rec.get("seconds", 0.0))
    f = {"error": rec.get("error"), "segments": 0, "text_chars": 0, "seg_out_of_bounds": 0, "seg_end_before_start": 0,
         "seg_unordered": 0, "seg_overlap_max_s": 0.0, "words": 0, "words_start": 0, "words_end": 0,
         "word_conf_all_one": None, "tokens": 0, "tokens_start": 0, "tokens_end": 0, "tokens_out_of_bounds": 0,
         "tokens_backwards": 0, "token_conf_none": 0, "token_text_equal": None, "speakers": 0,
         "no_speech_prob": 0}
    if rec.get("error"):
        return f
    segs = rec.get("segments") or []
    f["segments"] = len(segs)
    prev = None
    word_confs, rebuilt_ok = [], []
    for s in segs:
        f["text_chars"] += len(_norm(s["text"]))
        a, b = s["start"], s["end"]
        if a < -tol or b > dur + tol:
            f["seg_out_of_bounds"] += 1
        if b < a:
            f["seg_end_before_start"] += 1
        if prev is not None:
            if a < prev[0] - 1e-6:
                f["seg_unordered"] += 1
            f["seg_overlap_max_s"] = max(f["seg_overlap_max_s"], round(prev[1] - a, 3))
        prev = (a, b)
        if s.get("speaker"):
            f["speakers"] += 1
        if s.get("no_speech_prob", -1.0) >= 0:
            f["no_speech_prob"] += 1
        for w in s.get("words") or ():
            f["words"] += 1
            f["words_start"] += w[1] is not None
            f["words_end"] += w[2] is not None
            word_confs.append(w[3])
        last = None
        for t in s.get("tokens") or ():
            f["tokens"] += 1
            st, en = t[1], t[2]
            f["tokens_start"] += st is not None
            f["tokens_end"] += en is not None
            if t[3] is None or t[3] < 0:
                f["token_conf_none"] += 1
            for x in (st, en):
                if x is not None and (x < -tol or x > dur + tol):
                    f["tokens_out_of_bounds"] += 1
                    break
            if st is not None:
                if last is not None and st < last - 1e-6:
                    f["tokens_backwards"] += 1
                last = st
        if s.get("tokens"):
            rebuilt_ok.append(_norm(rebuild_token_text(s["tokens"])) == _norm(s["text"]))
    f["seg_overlap_max_s"] = max(0.0, f["seg_overlap_max_s"])
    if word_confs:
        f["word_conf_all_one"] = all(abs(c - 1.0) < 1e-9 for c in word_confs)
    if rebuilt_ok:
        f["token_text_equal"] = all(rebuilt_ok)
    return f


def time_kind(n, starts, ends, share=0.95):
    """'start and end', 'start only', 'none' or 'some' for a kind of unit (tokens or words)."""
    if not n:
        return "none (not returned)"
    if starts >= share * n and ends >= share * n:
        return "start and end"
    if starts >= share * n:
        return "start only"
    if starts == 0 and ends == 0:
        return "none (no times)"
    return f"some ({starts} starts, {ends} ends of {n})"


def total(checks):
    keys = [k for k, v in checks[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)] if checks else []
    out = {k: sum(c[k] for c in checks if isinstance(c.get(k), (int, float))) for k in keys}
    if checks:
        out["seg_overlap_max_s"] = max(c["seg_overlap_max_s"] for c in checks)
        tte = [c["token_text_equal"] for c in checks if c["token_text_equal"] is not None]
        out["token_text_equal"] = None if not tte else all(tte)
        wco = [c["word_conf_all_one"] for c in checks if c["word_conf_all_one"] is not None]
        out["word_conf_all_one"] = None if not wco else all(wco)
        out["calls_failed"] = sum(1 for c in checks if c["error"])
    return out


def propose_row(t, segments_per_call_max):
    """The WhisperJAV pipeline row the returned object supports (pipelines/crispasr_pipeline.py EngineRow, the line
    rules of crispasr/lines.py), with the reason."""
    tk = time_kind(t.get("tokens", 0), t.get("tokens_start", 0), t.get("tokens_end", 0))
    wk = time_kind(t.get("words", 0), t.get("words_start", 0), t.get("words_end", 0))
    if tk == "start and end":
        return {"line_rule": "tokens", "onsets": False, "aligner": None,
                "why": "every token has a start and an end (as Parakeet-CrispASR, Kotoba, Whisper)"}
    if tk == "start only":
        return {"line_rule": "tokens", "onsets": True, "aligner": None,
                "why": "tokens have starts only (as ReazonSpeech-CrispASR: ends at the next start, the last at the VAD "
                       "group's end)"}
    if wk == "start and end":
        return {"line_rule": "tokens", "onsets": False, "aligner": None,
                "why": "words have times but tokens do not: lines.py reads the words instead (counted as "
                       "segments_without_token_times); check its subtitles"}
    if segments_per_call_max > 1:
        return {"line_rule": "segments", "onsets": False, "aligner": None,
                "why": "no unit times, but several timed segments per call (as MOSS-CrispASR)"}
    return {"line_rule": "group", "onsets": False, "aligner": "wav2vec2-ja",
            "why": "no usable times inside a call: one subtitle per VAD group, timed by the CrispASR aligner (as "
                   "Cohere-CrispASR)"}


def text_of(rec):
    return "".join(s["text"] for s in rec.get("segments") or ())


def _all_times(rec):
    """Every time of a result in order: each segment's start and end, its words' and its tokens' starts and ends."""
    out = []
    for s in rec.get("segments") or ():
        out += [s["start"], s["end"]]
        for w in s.get("words") or ():
            out += [w[1], w[2]]
        for t in s.get("tokens") or ():
            out += [t[1], t[2]]
    return out


def _max_diff(xa, xb):
    """The largest difference between two equally long lists of times (None where either has no time), or None
    when the lists differ in length (different structure)."""
    if len(xa) != len(xb):
        return None
    d = [abs(x - y) for x, y in zip(xa, xb) if x is not None and y is not None]
    none_mismatch = sum(1 for x, y in zip(xa, xb) if (x is None) != (y is None))
    return round(max(d), 4) if d and not none_mismatch else (0.0 if not none_mismatch else None)


def compare_records(a, b):
    """Two results of the same audio: text equal, text similarity, the largest token-start difference, and the
    largest difference over every time (segments, words, tokens; None when the structure differs)."""
    ta, tb = _norm(text_of(a)), _norm(text_of(b))
    sa = [t[1] for s in a.get("segments") or () for t in s.get("tokens") or ()]
    sb = [t[1] for s in b.get("segments") or () for t in s.get("tokens") or ()]
    return {"text_equal": ta == tb, "similarity": round(difflib.SequenceMatcher(None, ta, tb).ratio(), 4),
            "token_start_max_diff_s": _max_diff(sa, sb), "time_max_diff_s": _max_diff(_all_times(a), _all_times(b)),
            "a_chars": len(ta), "b_chars": len(tb)}


# ----------------------------------------------------------------------------------------------------------------
# Stage 1: catalog
# ----------------------------------------------------------------------------------------------------------------

def cmd_catalog(args):
    import crispasr
    from crispasr import _binding as b
    models = load_models()
    version = getattr(crispasr, "__version__", "?")
    built = b.Session.available_backends()
    names = list(dict.fromkeys(b.list_known_models()))
    shipped = manifest_files()
    rows = []
    for n in names:
        try:
            e = b.registry_lookup(n)
        except Exception as ex:
            e = None
            say(f"  {n}: lookup failed ({ex})")
        rows.append({"name": n, "file": e.filename if e else None, "url": e.url if e else None,
                     "approx_size": e.approx_size if e else None, "speech_to_text": is_speech_to_text(n, models),
                     "shipped_by_whisperjav": shipped.get(e.filename, (None,))[0] if e else None})
    cat = {"crispasr": version, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "backends_built": built,
           "registry": rows, "whisperjav_files": {k: v[0] for k, v in shipped.items()}}
    out = Path(args.out) / f"catalog_crispasr-{version}.json"
    dump(out, cat)
    stt = [r for r in rows if r["speech_to_text"]]
    say(f"CrispASR {version}: {len(built)} backends built in; {len(names)} names in its registry, {len(stt)} of them "
        f"speech-to-text by the data file's list.")
    say(f"WhisperJAV ships {len(shipped)} GGUF files (manifest.py); in the registry by file name: "
        f"{len({r['file'] for r in rows if r['shipped_by_whisperjav']})} files under "
        f"{sum(1 for r in rows if r['shipped_by_whisperjav'])} names.")
    say("\nSpeech-to-text names in the registry (file, size, shipped by WhisperJAV):")
    for r in stt:
        mark = f"shipped as {r['shipped_by_whisperjav']}" if r["shipped_by_whisperjav"] else ""
        say(f"  {r['name']:<34} {r['file'] or '-':<52} {r['approx_size'] or '':<9} {mark}")
    if args.compare:
        old = load_json(args.compare)
        before = {r["name"] for r in old.get("registry", ())}
        now = {r["name"] for r in rows}
        new, gone = sorted(now - before), sorted(before - now)
        say(f"\nCompared with {args.compare} (CrispASR {old.get('crispasr')}): new {len(new)}: {', '.join(new) or 'none'}; "
            f"no longer listed {len(gone)}: {', '.join(gone) or 'none'}")
        bnew = sorted(set(built) - set(old.get("backends_built", ())))
        say(f"Backends new in this build: {', '.join(bnew) or 'none'}")
    say(f"\nWritten: {out}")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# Stage 2: fetch
# ----------------------------------------------------------------------------------------------------------------

def cmd_fetch(args):
    entry = model_entry(args.model)
    repo, fn = entry["repo"], entry["file"]
    files, sha, card = hf_listing(repo, entry.get("revision"))
    if fn not in files:
        raise SystemExit(f"{fn} is not in {repo} at commit {sha}. Files there: {', '.join(sorted(files))}")
    size, oid = files[fn]
    path = cached_file(repo, fn, sha)
    if path and path.stat().st_size == size:
        say(f"Already in the Hugging Face cache: {path}")
    else:
        say(f"Downloading {repo}/{fn} at commit {sha} ({size / 1e9:.2f} GB) ...")
        from huggingface_hub import hf_hub_download
        t = time.time()
        path = Path(hf_hub_download(repo, fn, revision=sha))
        say(f"  done in {time.time() - t:.0f} s: {path}")
    got = sha256_of(path)
    ok = (oid is None or got == oid) and path.stat().st_size == size
    licence_hf = card.get("license")
    try:
        from crispasr import _binding as b
        bundle = b.registry_lookup_by_filename(fn)
        registry = {"file": bundle.filename, "url": bundle.url, "approx_size": bundle.approx_size} if bundle else None
    except Exception as e:
        registry = f"not read ({type(e).__name__})"
    facts = {"model": args.model, "repo": repo, "file": fn, "commit": sha, "size": size, "sha256_local": got,
             "sha256_hugging_face": oid, "matches": ok, "path": str(path), "licence_repository": licence_hf,
             "licence_data_file": entry.get("licence"), "base_model": entry.get("base_model"), "registry": registry,
             "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    facts["draft_manifest_entry"] = (
        f'ModelFile("{args.model}", "{entry.get("label", args.model)}", "{entry.get("backend")}",\n'
        f'          "{repo}", "{fn}",\n          "{sha}", {size:_},\n          "{got}",\n'
        f'          "{entry.get("licence") or licence_hf}",\n          vram_mib=<from the open stage>)')
    root = Path(args.out) / args.model if args.out else None
    if root:
        dump(root / "fetch.json", facts)
    say(f"  size {path.stat().st_size} bytes, SHA-256 {got}: {'matches Hugging Face' if ok else 'DOES NOT MATCH (' + str(oid) + ')'}")
    say(f"  licence: repository card {licence_hf!r}; data file {entry.get('licence')!r}")
    return 0 if ok else 2


# ----------------------------------------------------------------------------------------------------------------
# Stage 3: open
# ----------------------------------------------------------------------------------------------------------------

def cmd_open(args):
    path, name, entry = resolve_model(args.model)
    root = Path(args.out) / name / "open"
    msg, info = gpu_room(args, path.stat().st_size)
    if msg:
        say(msg)
        return 3
    results = {"model": name, "file": str(path), "size": path.stat().st_size, "expected_backend": entry.get("backend")}
    for label, use_gpu in (("gpu", True), ("cpu", False)):
        if label == "gpu" and (args.cpu or not info):
            continue
        if label == "cpu" and args.no_cpu:
            continue
        job = base_job(path, args, use_gpu=use_gpu)
        job["open_times"] = 2
        job["calls"] = [{"id": "zeros_5s", "zeros": 5.0}]
        say(f"\n== open on the {label.upper()} (twice, one 5 s call of silence in the first)")
        events, code, meta = run_job(job, root / label, gpu_index=args.gpu_index,
                                     total_mib=info["total_mib"] if info and use_gpu else None)
        opens = [e for e in events if e["kind"] == "open"]
        closes = [e for e in events if e["kind"] == "closed"]
        failed = [e for e in events if e["kind"] == "open_failed"]
        r = {"exit_code": code, "crash": meta.get("crash"), "open_failed": failed[0]["error"] if failed else None}
        if opens:
            o = opens[0]
            r.update({"backend": o["backend"], "backend_as_expected": entry.get("backend") in (None, o["backend"]),
                      "crispasr": o["crispasr"], "open_s": [x["open_s"] for x in opens]})
            if use_gpu and o.get("gpu_before_mib") is not None:
                r["gpu_before_mib"] = o["gpu_before_mib"]
                r["gpu_open_over_before_mib"] = [x["gpu_after_open_mib"] - x["gpu_before_mib"] for x in opens
                                                 if x.get("gpu_after_open_mib") is not None]
                r["gpu_peak_over_before_mib"] = (meta["gpu_peak_mib"] - o["gpu_before_mib"]) if meta.get("gpu_peak_mib") else None
                r["gpu_after_close_over_before_mib"] = [c["gpu_after_close_mib"] - o["gpu_before_mib"] for c in closes
                                                        if c.get("gpu_after_close_mib") is not None]
        calls = [e for e in events if e["kind"] == "call"]
        if calls:
            r["silence_call"] = {"call_s": calls[0].get("call_s"), "text": text_of(calls[0])[:200],
                                 "error": calls[0].get("error")}
        results[label] = r
        say(json.dumps(r, ensure_ascii=False))
    dump(root / "open.json", results)
    say(f"\nWritten: {root / 'open.json'}")
    return 0 if all(results.get(k, {}).get("exit_code", 0) == 0 for k in ("gpu", "cpu")) else 5


# ----------------------------------------------------------------------------------------------------------------
# Stage 4: payload
# ----------------------------------------------------------------------------------------------------------------

def probe_calls(media, ref_lines, clip_s, noise_at=None, speech_lines=3):
    """The fixed test audio: silence, VAD-group-sized speech around reference lines, long pieces, and noise."""
    calls = []
    for s in (0.2, 0.5, 5.0):
        calls.append({"id": f"zeros_{s}s", "zeros": s, "probe": "silence"})
    starts = [a for a, _, _ in ref_lines if a + 6.5 < clip_s][:speech_lines] or [0.0, 10.0, 20.0][:speech_lines]
    for n, a in enumerate(starts):
        for length in (3.0, 6.0):
            b = max(0.0, a - 0.2)
            calls.append({"id": f"speech{n}_{length:g}s", "media": str(media), "start": round(b, 3),
                          "end": round(min(clip_s, b + length), 3), "probe": f"speech {length:g} s"})
    for length in (30.0, 60.0, 120.0):
        if length <= clip_s:
            calls.append({"id": f"long_{length:g}s", "media": str(media), "start": 0.0, "end": length,
                          "probe": f"long {length:g} s"})
    if noise_at is not None:
        calls.append({"id": "noise_8s", "media": str(media), "start": noise_at, "end": min(clip_s, noise_at + 8.0),
                      "probe": "music or noise"})
    return calls


def cmd_payload(args):
    path, name, entry = resolve_model(args.model)
    root = Path(args.out) / name / "payload"
    msg, info = gpu_room(args, path.stat().st_size)
    if msg:
        say(msg)
        return 3
    media = Path(args.input)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    clip_s = len(load_audio(ffmpeg, media)) / SR
    ref = read_srt(args.reference) if args.reference else []
    probes = probe_calls(media, ref, clip_s, args.noise_at)
    total_mib = info["total_mib"] if info else None

    # A: every probe twice, then the speech probes once more with no language
    job = base_job(path, args)
    job["settings"] = entry.get("settings") or {}
    for run_no in (0, 1):
        for c in probes:
            job["calls"].append(dict(c, id=f"{c['id']}#run{run_no}", run=run_no))
    for c in probes:
        if c["probe"].startswith("speech"):
            job["calls"].append(dict(c, id=f"{c['id']}#nolang", run=0, language=None))
    say(f"\n== payload A: {len(job['calls'])} calls (each probe twice; the speech probes again with no language)")
    ev_a, code_a, meta_a = run_job(job, root / "a_probes", env_extra=entry.get("env"), gpu_index=args.gpu_index,
                                   total_mib=total_mib)
    # B: the long probes with CrispASR's own cutting off
    longs = [c for c in probes if c["probe"].startswith("long")]
    ev_b, meta_b = [], {}
    if longs:
        job_b = base_job(path, args)
        job_b["settings"] = job["settings"]
        job_b["calls"] = [dict(c, id=f"{c['id']}#cutoff", run=0) for c in longs]
        say(f"\n== payload B: {len(longs)} long calls with CrispASR's cutting off (CRISPASR_SESSION_AUTOCHUNK=0)")
        ev_b, _, meta_b = run_job(job_b, root / "b_cutting_off", env_extra=dict(entry.get("env") or {},
                                  CRISPASR_SESSION_AUTOCHUNK="0"), gpu_index=args.gpu_index, total_mib=total_mib)
    # C: the speech probes and the 30 s probe on the CPU
    ev_c, meta_c = [], {}
    if not args.no_cpu and not args.cpu:
        job_c = base_job(path, args, use_gpu=False)
        job_c["settings"] = job["settings"]
        job_c["calls"] = [dict(c, id=f"{c['id']}#cpu", run=0) for c in probes
                          if c["probe"].startswith("speech") or c["id"] == "long_30s"]
        say(f"\n== payload C: {len(job_c['calls'])} calls on the CPU")
        ev_c, _, meta_c = run_job(job_c, root / "c_cpu", env_extra=entry.get("env"), gpu_index=args.gpu_index)

    facts = payload_facts(ev_a, ev_b, ev_c, probes)
    facts.update({"model": name, "file": str(path), "input": str(media), "reference": args.reference,
                  "crashes": [m.get("crash") for m in (meta_a, meta_b, meta_c) if m.get("crash")],
                  "gpu_peak_mib": meta_a.get("gpu_peak_mib"), "tool": TOOL_VERSION})
    dump(root / "payload.json", facts)
    write_payload_md(root / "payload.md", facts)
    say(f"\nWritten: {root / 'payload.md'}")
    return 0 if code_a == 0 else 5


def payload_facts(ev_a, ev_b, ev_c, probes):
    calls = {e["id"]: e for e in ev_a + ev_b + ev_c if e.get("kind") == "call"}
    opened = [e for e in ev_a if e.get("kind") == "open"]
    first = {e["id"].split("#")[0]: e for e in ev_a if e.get("kind") == "call" and e["id"].endswith("#run0")}
    checks = {pid: call_checks(rec) for pid, rec in first.items()}
    allc = list(checks.values())
    tot = total(allc) if allc else {}
    spc = max((c["segments"] for c in allc), default=0)
    facts = {"crispasr": opened[0]["crispasr"] if opened else None, "backend": opened[0]["backend"] if opened else None,
             "per_probe": {pid: dict(c, seconds=first[pid]["seconds"], call_s=first[pid].get("call_s"),
                                     text=text_of(first[pid])[:300]) for pid, c in checks.items()},
             "totals": tot, "segments_per_call_max": spc,
             "token_times": time_kind(tot.get("tokens", 0), tot.get("tokens_start", 0), tot.get("tokens_end", 0)),
             "word_times": time_kind(tot.get("words", 0), tot.get("words_start", 0), tot.get("words_end", 0))}
    facts["proposed_row"] = propose_row(tot, spc) if allc else None
    facts["silence_text"] = {pid: text_of(first[pid]) for pid in first if pid.startswith("zeros") and text_of(first[pid]).strip()}
    rep, lang, cut, cpu = {}, {}, {}, {}
    for c in probes:
        pid = c["id"]
        a, b = calls.get(f"{pid}#run0"), calls.get(f"{pid}#run1")
        if a and b and not a.get("error") and not b.get("error"):
            rep[pid] = compare_records(a, b)
        n = calls.get(f"{pid}#nolang")
        if a and n and not n.get("error"):
            lang[pid] = compare_records(a, n)
        o = calls.get(f"{pid}#cutoff")
        if a and o and not o.get("error"):
            cut[pid] = compare_records(a, o)
        p = calls.get(f"{pid}#cpu")
        if a and p and not p.get("error"):
            cpu[pid] = dict(compare_records(a, p), gpu_call_s=a.get("call_s"), cpu_call_s=p.get("call_s"))
    facts.update({"repeat_same": rep, "no_language_vs_ja": lang, "cutting_off_vs_default": cut, "cpu_vs_gpu": cpu})
    return facts


def write_payload_md(path, f):
    L = [f"# The returned object: {f['model']}", "",
         f"CrispASR {f['crispasr']}, backend `{f['backend']}`; tool version {f['tool']}. Input `{f['input']}`.", "",
         f"- Token times: **{f['token_times']}**. Word times: **{f['word_times']}**. Most segments in one call: "
         f"{f['segments_per_call_max']}.",
         f"- Proposed pipeline row: `{json.dumps(f['proposed_row'], ensure_ascii=False)}`.",
         f"- Text on silence: {json.dumps(f['silence_text'], ensure_ascii=False) if f['silence_text'] else 'none'}.",
         f"- Crashes: {'; '.join(f['crashes']) if f['crashes'] else 'none'}.", "",
         "## Each probe (first run)", "",
         "| Probe | Audio (s) | Call (s) | Segments | Chars | Seg. out of bounds | Seg. unordered | Overlap max (s) | "
         "Tokens | with start | with end | out of bounds | backwards | conf. none | token text = text | Words | "
         "with start | Error |", "|" + "---|" * 18]
    for pid, c in f["per_probe"].items():
        L.append(f"| {pid} | {c['seconds']} | {c['call_s']} | {c['segments']} | {c['text_chars']} | "
                 f"{c['seg_out_of_bounds']} | {c['seg_unordered']} | {c['seg_overlap_max_s']} | {c['tokens']} | "
                 f"{c['tokens_start']} | {c['tokens_end']} | {c['tokens_out_of_bounds']} | {c['tokens_backwards']} | "
                 f"{c['token_conf_none']} | {c['token_text_equal']} | {c['words']} | {c['words_start']} | "
                 f"{c['error'] or ''} |")
    for title, key in (("The same call twice", "repeat_same"), ("No language against ja", "no_language_vs_ja"),
                       ("CrispASR's cutting off against its default", "cutting_off_vs_default"),
                       ("CPU against GPU", "cpu_vs_gpu")):
        L += ["", f"## {title}", ""]
        rows = f.get(key) or {}
        if not rows:
            L.append("Not run.")
            continue
        L += ["| Probe | Text equal | Similarity | Token start max diff (s) | Every time max diff (s) | Chars | "
              "Chars (other) | Extra |", "|---|---|---|---|---|---|---|---|"]
        for pid, r in rows.items():
            extra = f"GPU {r['gpu_call_s']} s, CPU {r['cpu_call_s']} s" if "cpu_call_s" in r else ""
            L.append(f"| {pid} | {r['text_equal']} | {r['similarity']} | {r['token_start_max_diff_s']} | "
                     f"{r.get('time_max_diff_s')} | {r['a_chars']} | {r['b_chars']} | {extra} |")
    L += ["", "Checks per call run on each probe's first run. 'Every time': segment, word and token starts and ends "
              "(None: the two results differ in structure).", ""]
    L += ["", "Texts of each probe: payload.json (per_probe.text) and the events.jsonl files.", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# Stage 5: measure
# ----------------------------------------------------------------------------------------------------------------

def score(ref, srt, out):
    if not (REPO_ROOT / "subeval").is_dir():
        return {"error": f"subeval not found in {REPO_ROOT}"}
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "subeval", "score", str(ref), str(srt), "--out", str(out)],
                       cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    res = Path(out) / "result.json"
    if not res.exists():
        return {"error": f"subeval exit {r.returncode}: {r.stderr[-800:]}"}
    s = load_json(res)["summary"]
    keep = ("ref_lines", "ref_chars", "test_lines", "cer", "edits", "wrong_rate", "missing_rate", "extra_rate",
            "missed_lines", "hallucination_lines", "start_measured", "end_measured", "start_abs_median",
            "end_abs_median")
    return {k: s.get(k) for k in keep}


def pooled(scores):
    ok = [s for s in scores if s and "error" not in s and s.get("ref_chars")]
    if not ok:
        return {}
    ref_chars = sum(s["ref_chars"] for s in ok)
    out = {"clips": len(ok), "ref_chars": ref_chars, "cer": sum(s["edits"] for s in ok) / ref_chars}
    for k in ("wrong_rate", "missing_rate", "extra_rate"):
        out[k] = sum(s[k] * s["ref_chars"] for s in ok) / ref_chars
    for k in ("missed_lines", "hallucination_lines", "test_lines", "ref_lines", "start_measured"):
        out[k] = sum(s.get(k) or 0 for s in ok)
    return out


def as_segments(rec):
    """A call record's segments as WhisperJAV's own types, through engine.segments_from_binding (read-only)."""
    from whisperjav.crispasr.engine import segments_from_binding
    fake = []
    for s in rec.get("segments") or ():
        words = [SimpleNamespace(text=w[0], start=w[1], end=w[2], confidence=w[3]) for w in s.get("words") or ()]
        toks = [SimpleNamespace(text=t[0], start=t[1], end=t[2], confidence=t[3],
                                raw_text=bytes.fromhex(t[4]) if len(t) > 4 and t[4] else b"")
                for t in s.get("tokens") or ()]
        fake.append(SimpleNamespace(text=s["text"], start=s["start"], end=s["end"], words=words, tokens=toks))
    return segments_from_binding(fake)


def whisperjav_subtitles(groups_with_recs, row, folder, stem):
    """The subtitles WhisperJAV's unchanged code builds from these results with this row: crispasr/lines.py build()
    per VAD group, each scene's subtitles by start, the scenes in order, then crispasr/cleanup.py SrtCleanup with the
    row's stages. Returns (final SRT path, raw count, final count, clean-up counts)."""
    from whisperjav.crispasr import cleanup as wj_cleanup
    from whisperjav.crispasr import lines as wj_lines
    stats = wj_lines.LineStats()
    by_scene = {}
    for g, rec in groups_with_recs:
        segs = as_segments(rec) if not rec.get("error") else []
        subs = wj_lines.build(row["line_rule"], g["start"], g["end"], segs, stats, wj_lines.DEFAULT_CUES,
                              onsets=row.get("onsets", False), cut_at_pauses=row.get("cut_at_pauses", True),
                              end_at_group=row.get("onsets", False))
        by_scene.setdefault(g.get("scene", 0), []).extend(subs)
    entries = []
    for sc in sorted(by_scene):
        entries += [(s, e, t) for t, s, e in sorted(by_scene[sc], key=lambda x: x[1])]
    stitched = Path(folder) / f"{stem}.whisperjav.srt"
    write_srt(stitched, entries)
    stages = {"srt": wj_cleanup.SRT_STAGES, "moss": wj_cleanup.MOSS_STAGES, "none": ()}[row.get("cleanup", "srt")]
    st, count = wj_cleanup.SrtCleanup(stages, label="crispasr_model_test").run(
        stitched, Path(folder) / "raw_subs", stem, len(entries))
    if count == 0:
        stitched.write_text("", encoding="utf-8")
    return stitched, len(entries), count, dict(st, segments_without_token_times=stats.fallback_segments,
                                               clamped_times=stats.clamped_times)


def cmd_measure(args):
    path, name, entry = resolve_model(args.model)
    row = json.loads(args.row) if args.row else entry.get("row")
    if not row:
        raise SystemExit("No pipeline row for this model: add 'row' to its data-file entry or pass --row.")
    root = Path(args.out) / name / (args.name or "measure")
    clips = []
    if args.clips:
        clips = find_clips(args.clips)
    if args.input:
        clips.append((Path(args.input), Path(args.reference) if args.reference else None))
    if args.only:
        clips = [c for c in clips if any(o in c[0].name for o in args.only.split(","))]
    if not clips:
        raise SystemExit("No clips: give --clips FOLDER or --input FILE.")
    msg, info = gpu_room(args, path.stat().st_size)
    if msg:
        say(msg)
        return 3
    ffmpeg = find_ffmpeg(args.ffmpeg)

    # The VAD groups of each clip, found before the model loads
    plan = []
    for media, ref in clips:
        stem = media.stem
        if args.groups_from:
            gfile = Path(args.groups_from) / "raw_subs" / f"{stem}.{args.groups_pipeline}.groups.jsonl"
            if not gfile.exists():
                raise SystemExit(f"No per-group record for {stem}: {gfile} (run WhisperJAV with --debug)")
            groups, how = pipeline_groups(gfile), f"VAD groups of WhisperJAV's {args.groups_pipeline} run ({gfile})"
        elif args.vad:
            groups, how = vad_groups(ffmpeg, media, args), f"{args.vad} VAD groups, {args.vad_like} settings"
        else:
            audio = load_audio(ffmpeg, media)
            groups = [{"scene": 0, "start": s, "end": e} for s, e in quiet_pieces(audio, args.piece_seconds)]
            how = f"pieces of at most {args.piece_seconds} s"
        plan.append((media, ref, groups, how))
        say(f"{stem}: {len(groups)} calls ({how})")

    job = base_job(path, args)
    job["settings"] = entry.get("settings") or {}
    for ci, (media, ref, groups, how) in enumerate(plan):
        for gi, g in enumerate(groups):
            job["calls"].append({"id": f"{media.stem}#{gi}", "clip": ci, "group": gi, "scene": g.get("scene", 0),
                                 "media": str(media), "start": g["start"], "end": g["end"]})
    say(f"\n== measure: {len(job['calls'])} calls over {len(plan)} clip(s), model loaded once")
    events, code, meta = run_job(job, root / "_run", env_extra=entry.get("env"), gpu_index=args.gpu_index,
                                 total_mib=info["total_mib"] if info else None)
    opened = next((e for e in events if e.get("kind") == "open"), {})
    recs = {(e["clip"], e["group"]): e for e in events if e.get("kind") == "call"}

    results = []
    for ci, (media, ref, groups, how) in enumerate(plan):
        stem, folder = media.stem, root / media.stem
        folder.mkdir(parents=True, exist_ok=True)
        pairs = [(g, recs.get((ci, gi), {"error": "not run (crash before this call)"})) for gi, g in enumerate(groups)]
        with open(folder / "pieces.jsonl", "w", encoding="utf-8") as fh:
            for g, r in pairs:
                fh.write(json.dumps({**r, "vad_group": {k: g[k] for k in ("scene", "start", "end")}},
                                    ensure_ascii=False) + "\n")
        model_times, edges = [], []
        for g, r in pairs:
            texts = []
            for s in r.get("segments") or ():
                x = s["text"].strip()
                if x:
                    texts.append(x)
                    model_times.append((g["start"] + s["start"], max(g["start"] + s["start"], g["start"] + s["end"]), x))
            if texts:
                edges.append((g["start"], g["end"], "".join(texts)))
        write_srt(folder / "model_times.srt", sorted(model_times))
        write_srt(folder / "group_edges.srt", edges)
        wj_srt, raw_n, final_n, cl = whisperjav_subtitles(pairs, row, folder, stem)
        ok = [r for _, r in pairs if not r.get("error")]
        audio_s = sum(r["seconds"] for r in ok)
        asr_s = sum(r.get("call_s") or 0 for r in ok)
        res = {"clip": stem, "media": str(media), "reference": str(ref) if ref else None, "calls": len(pairs),
               "calls_failed": len(pairs) - len(ok), "groups_how": how, "audio_s": round(audio_s, 1),
               "asr_s": round(asr_s, 2), "x_realtime": round(audio_s / asr_s, 1) if asr_s else None,
               "without_text": sum(1 for r in ok if not text_of(r).strip()),
               "whisperjav_raw_subtitles": raw_n, "whisperjav_subtitles": final_n, "cleanup": cl}
        if ref:
            res["score_whisperjav"] = score(ref, wj_srt, folder / "subeval_whisperjav")
            res["score_model_times"] = score(ref, folder / "model_times.srt", folder / "subeval_model_times")
            res["score_group_edges"] = score(ref, folder / "group_edges.srt", folder / "subeval_group_edges")
        if args.compare_pipeline:
            res["pipeline_check"] = pipeline_check(args, stem, pairs, wj_srt)
        results.append(res)
        say(f"{stem}: {final_n} subtitles" + (f", CER {fmt(res['score_whisperjav'].get('cer'))}" if ref else ""))
    summary = {"model": name, "file": str(path), "row": row, "crispasr": opened.get("crispasr"),
               "backend": opened.get("backend"), "whisperjav_commit": git_commit(), "tool": TOOL_VERSION,
               "when": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": code, "crash": meta.get("crash"),
               "gpu_before_mib": opened.get("gpu_before_mib"), "gpu_after_open_mib": opened.get("gpu_after_open_mib"),
               "gpu_peak_mib": meta.get("gpu_peak_mib"), "gpu_was_full": meta.get("gpu_was_full"), "clips": results}
    for kind in ("whisperjav", "model_times", "group_edges"):
        summary[f"pooled_{kind}"] = pooled([r.get(f"score_{kind}") for r in results])
    a, t = sum(r["audio_s"] for r in results), sum(r["asr_s"] for r in results)
    summary["x_realtime"] = round(a / t, 1) if t else None
    dump(root / "measure.json", summary)
    write_measure_md(root / "measure.md", summary)
    say(f"\nWritten: {root / 'measure.md'}")
    return 0 if code == 0 else 5


def vad_groups(ffmpeg, media, args):
    """VAD groups by WhisperJAV's own speech segmenter, as the CrispASR pipelines find them (read-only import), over
    the whole clip as one scene."""
    import numpy as np
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from whisperjav.crispasr.pass_settings import segmenter_settings
    from whisperjav.modules.subtitle_pipeline.framers.vad_grouped import VadGroupedFramer
    audio = load_audio(ffmpeg, media)
    s = segmenter_settings(args.vad_like, args.vad, None)
    framer = VadGroupedFramer(segmenter_backend=s.backend, max_group_duration_s=s.max_group_duration_s,
                              chunk_threshold_s=s.chunk_threshold_s, min_frame_duration_s=0.1,
                              segmenter_config=dict(s.config))
    try:
        res = framer.frame(np.ascontiguousarray(audio), SR)
    finally:
        framer.cleanup()
    dur = len(audio) / SR
    return [{"scene": 0, "start": max(0.0, f.start), "end": min(dur, f.end)} for f in res.frames
            if min(dur, f.end) - max(0.0, f.start) >= 0.2]


def pipeline_check(args, stem, pairs, wj_srt):
    """The tool against a WhisperJAV run on the same VAD groups: each group's text and every time (segments, words,
    tokens) as the pipeline received them (its --debug record), and its final subtitles."""
    gfile = Path(args.compare_pipeline) / "raw_subs" / f"{stem}.{args.groups_pipeline}.groups.jsonl"
    theirs = pipeline_groups(gfile) if gfile.exists() else []
    out = {"groups_record": str(gfile), "groups": len(theirs), "groups_in_tool": len(pairs)}
    if theirs and len(theirs) == len(pairs):
        cmp = [compare_records({"segments": [dict(s, tokens=[t + [""] for t in s.get("tokens", [])])
                                             for s in g["segments"]]}, r) for g, (_, r) in zip(theirs, pairs)]
        out.update({"text_equal": sum(c["text_equal"] for c in cmp), "similarity_min": min(c["similarity"] for c in cmp),
                    "group_time_max_diff_s": max((c["time_max_diff_s"] or 0.0) for c in cmp),
                    "groups_structure_differs": [i for i, c in enumerate(cmp) if c["time_max_diff_s"] is None],
                    "groups_time_differs": [i for i, c in enumerate(cmp) if (c["time_max_diff_s"] or 0) > 0.0005],
                    "groups_text_differs": [i for i, c in enumerate(cmp) if not c["text_equal"]][:20]})
    final = sorted(Path(args.compare_pipeline).glob(args.pipeline_srt.format(stem=stem)))
    if final:
        a, b = read_srt(final[0]), read_srt(wj_srt) if Path(wj_srt).stat().st_size else []
        same = sum(1 for x, y in zip(a, b) if _norm(x[2]) == _norm(y[2])
                   and max(abs(x[0] - y[0]), abs(x[1] - y[1])) < 0.0005)
        dt = max((max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(a, b)), default=0.0)
        out.update({"pipeline_srt": str(final[0]), "pipeline_subtitles": len(a), "tool_subtitles": len(b),
                    "same_subtitles_in_order": same, "srt_time_max_diff_s": round(dt, 3),
                    "srt_identical": len(a) == len(b) == same})
    return out


def fmt(x, spec="{:.3f}"):
    if x is None:
        return "–"
    return spec.format(x) if isinstance(x, float) else str(x)


def write_measure_md(path, s):
    L = [f"# Measured: {s['model']}", "",
         f"CrispASR {s['crispasr']}, backend `{s['backend']}`; WhisperJAV code {s['whisperjav_commit']} (read only); "
         f"tool {s['tool']}; {s['when']}. Row: `{json.dumps(s['row'])}`.",
         f"GPU: {s['gpu_before_mib']} MiB before, {s['gpu_after_open_mib']} after opening, peak {s['gpu_peak_mib']}"
         f"{' (the card was full)' if s['gpu_was_full'] else ''}. Speed {s['x_realtime']} x real time (ASR calls only)."
         + (f" CRASH: {s['crash']}" if s.get("crash") else ""), "",
         "## Pooled over the clips (subeval; CER = edits / reference characters)", "",
         "| Subtitles | Clips | CER | Wrong | Missing | Extra | Missed lines | Hallucination lines | Test lines |",
         "|---|---|---|---|---|---|---|---|---|"]
    for kind, label in (("whisperjav", "as WhisperJAV builds them"), ("model_times", "model's own times"),
                        ("group_edges", "VAD group edges")):
        p = s.get(f"pooled_{kind}") or {}
        L.append(f"| {label} | {p.get('clips', '–')} | {fmt(p.get('cer'))} | {fmt(p.get('wrong_rate'))} | "
                 f"{fmt(p.get('missing_rate'))} | {fmt(p.get('extra_rate'))} | {p.get('missed_lines', '–')} | "
                 f"{p.get('hallucination_lines', '–')} | {p.get('test_lines', '–')} |")
    L += ["", "## Each clip (as WhisperJAV builds them)", "",
          "| Clip | Calls | Failed | Without text | x real time | Subtitles (raw / final) | CER | Missed lines | "
          "Halluc. lines | Start |err| med (s) | End |err| med (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["clips"]:
        sc = r.get("score_whisperjav") or {}
        L.append(f"| {r['clip']} | {r['calls']} | {r['calls_failed']} | {r['without_text']} | {r['x_realtime']} | "
                 f"{r['whisperjav_raw_subtitles']} / {r['whisperjav_subtitles']} | {fmt(sc.get('cer'))} | "
                 f"{sc.get('missed_lines', '–')} | {sc.get('hallucination_lines', '–')} | "
                 f"{fmt(sc.get('start_abs_median'))} | {fmt(sc.get('end_abs_median'))} |")
    checks = [r for r in s["clips"] if r.get("pipeline_check")]
    if checks:
        L += ["", "## Against the WhisperJAV run on the same VAD groups", "",
              "| Clip | Groups (pipeline / tool) | Same text | Lowest similarity | Groups with a time difference "
              "(largest, s) | Structure differs | Pipeline subtitles | Tool subtitles | Same subtitles in order "
              "(text and times) | SRT identical |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in checks:
            c = r["pipeline_check"]
            L.append(f"| {r['clip']} | {c.get('groups')} / {c.get('groups_in_tool')} | {c.get('text_equal', '–')} | "
                     f"{c.get('similarity_min', '–')} | {len(c.get('groups_time_differs', []))} "
                     f"({c.get('group_time_max_diff_s', '–')}) | {len(c.get('groups_structure_differs', []))} | "
                     f"{c.get('pipeline_subtitles', '–')} | {c.get('tool_subtitles', '–')} | "
                     f"{c.get('same_subtitles_in_order', '–')} | {c.get('srt_identical', '–')} |")
        L += ["", "Group times compared: every segment, word and token start and end of each group's result, as the "
                  "pipeline's --debug record has them.", ""]
    L += ["", "Files per clip: pieces.jsonl (every call as returned), <clip>.whisperjav.srt, model_times.srt, "
              "group_edges.srt, raw_subs/ (the clean-up's original and artifacts), subeval_*/. Definitions: "
              "subeval/METHOD.md.", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# Stage 6: listen
# ----------------------------------------------------------------------------------------------------------------

def subeval_lines(folder):
    res = Path(folder) / "result.json"
    if not res.exists():
        return None
    c = load_json(res)["clips"][0]
    return c["ref"], c["test"]


def cmd_listen(args):
    root = Path(args.out)
    mine = root / args.model / (args.name or "measure")
    base = root / args.baseline / (args.baseline_name or args.name or "measure") if args.baseline else None
    s = load_json(mine / "measure.json")
    to = mine / "listen"
    to.mkdir(parents=True, exist_ok=True)
    L = [f"# Hand test: {args.model}" + (f" against {args.baseline}" if base else ""), "",
         f"Open each clip in PotPlayer and load the subtitle files below (Subtitles > Load subtitle). Folder: `{to}`.", ""]
    for r in s["clips"]:
        stem = r["clip"]
        shutil.copy2(mine / stem / f"{stem}.whisperjav.srt", to / f"{stem}.{args.model}.srt")
        L += [f"## {stem}", "", f"Clip: `{r['media']}`", f"- `{stem}.{args.model}.srt`"]
        b_lines = None
        if base and (base / stem / f"{stem}.whisperjav.srt").exists():
            shutil.copy2(base / stem / f"{stem}.whisperjav.srt", to / f"{stem}.{args.baseline}.srt")
            L.append(f"- `{stem}.{args.baseline}.srt`")
            b_lines = subeval_lines(base / stem / "subeval_whisperjav")
        m_lines = subeval_lines(mine / stem / "subeval_whisperjav")
        if m_lines:
            ref, test = m_lines
            missed = {x["idx"] for x in ref if x.get("missed")}
            b_missed = {x["idx"] for x in b_lines[0] if x.get("missed")} if b_lines else None
            L += ["", "| Time | Reference line | This model | Baseline |", "|---|---|---|---|"]
            for x in ref:
                mm, bm = x["idx"] in missed, (x["idx"] in b_missed) if b_missed is not None else None
                if mm or bm:
                    L.append(f"| {srt_time(x['start'])} | {x['text'].replace(chr(10), ' ')} | "
                             f"{'missed' if mm else 'got it'} | {'–' if bm is None else ('missed' if bm else 'got it')} |")
            hall = [t for t in test if t.get("hallucination")]
            if hall:
                L += ["", "Lines of this model with no reference line under them (suspected hallucination):", ""]
                L += [f"- {srt_time(t['start'])} {t['text'].replace(chr(10), ' ')}" for t in hall]
        L.append("")
    (to / "review.md").write_text("\n".join(L), encoding="utf-8")
    say(f"Written: {to / 'review.md'} and the SRT files beside it")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# Stage 7: report
# ----------------------------------------------------------------------------------------------------------------

def cmd_report(args):
    root = Path(args.out)
    mdir = root / args.model
    entry = load_models()["models"].get(args.model, {})
    get = lambda p: load_json(p) if Path(p).exists() else None
    fetch, opened = get(mdir / "fetch.json"), get(mdir / "open" / "open.json")
    payload, meas = get(mdir / "payload" / "payload.json"), get(mdir / (args.name or "measure") / "measure.json")
    baseline = entry.get("baseline")
    bmeas = get(root / baseline / (args.name or "measure") / "measure.json") if baseline else None
    L = [f"# {entry.get('label', args.model)} through CrispASR: results", "",
         f"Tool `tools/crispasr_model_test.py` {TOOL_VERSION}; written {time.strftime('%Y-%m-%d %H:%M')}. "
         f"Limits: one CrispASR build (the one installed), one GPU; the Netflix clips are drama, not JAV.", ""]
    L += ["## Stage 2: the file", ""]
    if fetch:
        L += [f"- `{fetch['repo']}` `{fetch['file']}`, commit `{fetch['commit']}`, {fetch['size']:,} bytes, SHA-256 "
              f"`{fetch['sha256_local']}` ({'matches' if fetch['matches'] else 'DOES NOT match'} Hugging Face).",
              f"- Licence: repository card `{fetch['licence_repository']}`; data file `{fetch['licence_data_file']}`."]
    else:
        L.append("Not run.")
    L += ["", "## Stage 3: opening", ""]
    if opened:
        for k in ("gpu", "cpu"):
            o = opened.get(k)
            if o:
                L.append(f"- {k.upper()}: backend `{o.get('backend')}` (as expected: {o.get('backend_as_expected')}), "
                         f"open {o.get('open_s')} s; GPU over before: open {o.get('gpu_open_over_before_mib')}, peak "
                         f"{o.get('gpu_peak_over_before_mib')}, after closing {o.get('gpu_after_close_over_before_mib')} "
                         f"MiB; exit {o.get('exit_code')}{', CRASH ' + o['crash'] if o.get('crash') else ''}.")
    else:
        L.append("Not run.")
    L += ["", "## Stage 4: the returned object", ""]
    if payload:
        L += [f"- Token times: {payload['token_times']}; word times: {payload['word_times']}; most segments per call "
              f"{payload['segments_per_call_max']}.",
              f"- Text on silence: {payload['silence_text'] or 'none'}.",
              f"- Same call twice: {sum(r['text_equal'] for r in payload['repeat_same'].values())} of "
              f"{len(payload['repeat_same'])} identical. No language vs ja: "
              f"{sum(r['text_equal'] for r in payload['no_language_vs_ja'].values())} of {len(payload['no_language_vs_ja'])} "
              f"identical. Cutting off vs default: {sum(r['text_equal'] for r in payload['cutting_off_vs_default'].values())} "
              f"of {len(payload['cutting_off_vs_default'])}. CPU vs GPU: "
              f"{sum(r['text_equal'] for r in payload['cpu_vs_gpu'].values())} of {len(payload['cpu_vs_gpu'])}.",
              f"- Crashes: {payload['crashes'] or 'none'}. Details: `payload/payload.md`.",
              f"- Proposed row: `{json.dumps(payload['proposed_row'], ensure_ascii=False)}`."]
    else:
        L.append("Not run.")
    L += ["", "## Stage 5: the clips", ""]
    if meas:
        L += ["| Model | Subtitles | CER | Wrong | Missing | Extra | Missed lines | Halluc. lines | Test lines | x real time |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for label, m in ((args.model, meas), (baseline, bmeas)):
            if not m:
                continue
            for kind, kl in (("whisperjav", "as WhisperJAV"), ("model_times", "own times"), ("group_edges", "group edges")):
                p = m.get(f"pooled_{kind}") or {}
                L.append(f"| {label} | {kl} | {fmt(p.get('cer'))} | {fmt(p.get('wrong_rate'))} | {fmt(p.get('missing_rate'))} | "
                         f"{fmt(p.get('extra_rate'))} | {p.get('missed_lines', '–')} | {p.get('hallucination_lines', '–')} | "
                         f"{p.get('test_lines', '–')} | {m.get('x_realtime')} |")
        L += ["", "Per clip: `measure/measure.md`."]
    else:
        L.append("Not run.")
    L += ["", "## Draft for adoption (if the owner decides so)", "",
          "vram_mib: the card's peak over its use before the model was opened, during the measuring stage's VAD groups "
          "(as manifest.py's figures were measured).", "", "```python"]
    vram = None
    if meas and meas.get("gpu_peak_mib") and meas.get("gpu_before_mib") is not None:
        vram = meas["gpu_peak_mib"] - meas["gpu_before_mib"]
    L.append(fetch["draft_manifest_entry"].replace("<from the open stage>", str(vram or "<measure not run>"))
             if fetch else "# fetch not run")
    pr = (payload or {}).get("proposed_row") or entry.get("row") or {}
    L.append(f'ENGINE = EngineRow(line_rule="{pr.get("line_rule")}", audio_analytics=True'
             + (', onsets=True' if pr.get("onsets") else '') + (f', aligner="{pr["aligner"]}"' if pr.get("aligner") else '')
             + ")")
    L += ["```", ""]
    out = mdir / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    say(f"Written: {out}")
    return 0


# ----------------------------------------------------------------------------------------------------------------

def common_engine(p):
    e = p.add_argument_group("engine and GPU")
    e.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)), help="CPU threads (default up to 8).")
    e.add_argument("--cpu", action="store_true", help="Run on the CPU only.")
    e.add_argument("--no-flash-attn", action="store_true", help="Open the model without flash attention.")
    e.add_argument("--reserve-mib", type=int, default=2048, help="GPU memory kept for the system (default 2048).")
    e.add_argument("--margin-mib", type=int, default=512, help="Memory counted on top of the model file (default 512).")
    e.add_argument("--force", action="store_true", help="Load the model even when the memory check says no.")
    e.add_argument("--ffmpeg", help="Path to ffmpeg (default: from PATH).")
    e.add_argument("--progress-every", type=int, default=25, help="A line every N calls (default 25).")


def build_parser():
    p = argparse.ArgumentParser(prog="crispasr_model_test.py",
                                description="Test a GGUF model through CrispASR before WhisperJAV takes it: catalog, "
                                            "fetch, open, payload, measure, listen, report.")
    p.add_argument("--gpu-index", type=int, default=0, help="Which NVIDIA GPU nvidia-smi reports on (default 0).")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("gpu", help="Show the GPU's memory and the programs on it.")
    sub.add_parser("models", help="List the models of tools/crispasr_models.json and whether they are downloaded.")

    c = sub.add_parser("catalog", help="Stage 1: what this CrispASR can run, and what WhisperJAV ships.")
    c.add_argument("--out", required=True, help="Folder for catalog_crispasr-<version>.json.")
    c.add_argument("--compare", help="An earlier catalog file: show what is new or gone.")

    f = sub.add_parser("fetch", help="Stage 2: download a model at a pinned commit and check it.")
    f.add_argument("model", help="A name from tools/crispasr_models.json.")
    f.add_argument("--out", help="Root folder: writes <out>/<model>/fetch.json.")

    o = sub.add_parser("open", help="Stage 3: open the model on the GPU and the CPU, twice each.")
    o.add_argument("--model", required=True, help="A data-file name or a local .gguf path.")
    o.add_argument("--out", required=True, help="Root folder.")
    o.add_argument("--no-cpu", action="store_true", help="Skip the CPU.")
    common_engine(o)

    k = sub.add_parser("payload", help="Stage 4: check the object CrispASR returns, on fixed test audio.")
    k.add_argument("--model", required=True)
    k.add_argument("--input", required=True, help="A clip with speech (for example a Netflix clip).")
    k.add_argument("--reference", help="Its reference SRT: the speech probes start at its first lines.")
    k.add_argument("--noise-at", type=float, help="Start (s) of 8 s of music or noise in the clip, for a probe.")
    k.add_argument("--out", required=True)
    k.add_argument("--no-cpu", action="store_true", help="Skip the CPU comparison.")
    common_engine(k)

    m = sub.add_parser("measure", help="Stage 5: run the model on clips and score it with subeval.")
    m.add_argument("--model", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--name", help="Name of this measurement (default 'measure').")
    m.add_argument("--clips", help=f"A folder of clips with reference SRTs beside them (for example {NETFLIX}).")
    m.add_argument("--only", help="Only clips whose file name contains one of these (comma-separated).")
    m.add_argument("--input", help="One clip (with --reference, or without for a hand test only).")
    m.add_argument("--reference", help="The reference SRT of --input.")
    m.add_argument("--groups-from", help="A WhisperJAV output folder of a --debug run: use its VAD groups "
                                         "(raw_subs/<clip>.<pipeline>.groups.jsonl).")
    m.add_argument("--groups-pipeline", default="parakeet-crispasr", help="Whose record (default parakeet-crispasr).")
    m.add_argument("--compare-pipeline", help="A WhisperJAV output folder to compare with (the self-check): its "
                                              "per-group record and its final SRT.")
    m.add_argument("--pipeline-srt", default="{stem}.ja.pass1.srt",
                   help="Glob of that run's final SRT, {stem} for the clip (default {stem}.ja.pass1.srt).")
    m.add_argument("--vad", help="Without --groups-from: WhisperJAV's speech segmenter, for example ten.")
    m.add_argument("--vad-like", default="parakeet-crispasr", help="Whose segmenter settings (default parakeet-crispasr).")
    m.add_argument("--piece-seconds", type=float, default=30.0, help="Without groups or VAD: pieces of at most N s.")
    m.add_argument("--row", help="The pipeline row as JSON, instead of the data file's.")
    common_engine(m)

    li = sub.add_parser("listen", help="Stage 6: SRTs side by side and a list of the lines that differ.")
    li.add_argument("--model", required=True)
    li.add_argument("--baseline", help="The model to put beside it (its measure folder under --out).")
    li.add_argument("--out", required=True)
    li.add_argument("--name", help="Measurement name (default 'measure').")
    li.add_argument("--baseline-name", help="The baseline's measurement name (default: as --name).")

    r = sub.add_parser("report", help="Stage 7: one results file and the draft for adoption.")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--name", help="Measurement name (default 'measure').")

    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("job")
    return p


def cmd_models(_args):
    models = load_models()
    for n, e in models["models"].items():
        c = cached_file(e["repo"], e["file"], e.get("revision"))
        say(f"  {n:<26} {'downloaded' if c else 'not downloaded':<15} {e['repo']}/{e['file']}  backend {e.get('backend')}")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.command == "_worker":
        return worker(args.job)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return {"gpu": lambda a: 0 if print_gpu(a.gpu_index) else 2, "models": cmd_models, "catalog": cmd_catalog,
            "fetch": cmd_fetch, "open": cmd_open, "payload": cmd_payload, "measure": cmd_measure,
            "listen": cmd_listen, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
