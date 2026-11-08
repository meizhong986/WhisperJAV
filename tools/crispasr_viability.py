#!/usr/bin/env python
"""CrispASR viability suite: one stage of WhisperJAV's chain swapped, everything else as shipped (owner W581, W582, W584).

Phase A swaps one stage of Parakeet-CrispASR, run as WhisperJAV ships it:
  A1  the speech enhancer   (CrispASR separators and audio-to-audio speech models, against WhisperJAV's own enhancers)
  A2  the speech segmenter  (CrispASR VAD engines, against WhisperJAV's own segmenters)
Every other step (audio extraction, semantic scenes, the enhancer's scene loop, the VAD grouping, parakeet-tdt-0.6b-ja,
line building, clean-up) is WhisperJAV's unchanged code, imported read-only in a child process.

  python tools/crispasr_viability.py fetch --all --out D:\\runs
  python tools/crispasr_viability.py run crisp-silero --out D:\\runs
  python tools/crispasr_viability.py run crisp-htdemucs-f16 --track vad --out D:\\runs
  python tools/crispasr_viability.py bench crisp-htdemucs-f16 --out D:\\runs
  python tools/crispasr_viability.py rnnoise --out D:\\runs
  python tools/crispasr_viability.py table --out D:\\runs
  python tools/crispasr_viability.py selfcheck --out D:\\runs

The candidates are in tools/crispasr_viability.json; the commands are described in tools/crispasr_viability.md; the
charter and the design in docs/plans/design/2026-11-07-phase-a-charter-and-viability-suite.md (private).

Exit status: 0 done; 2 a setup or input problem; 5 the child process ended abnormally (what it wrote before is kept).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOL_VERSION = "1 (2026-11-07)"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path(__file__).resolve().with_name("crispasr_viability.json")
NETFLIX = REPO_ROOT / "test_media" / "Ground_Truths" / "Netflix"
CHAIN_PIPELINE = "parakeet-crispasr"
SR = 16000
SEPARATOR_SR = 44100      # CrispASR's separators take 44.1 kHz stereo (_binding.py: Session.separate)
MARGIN_FLOOR = 0.005      # 0.5 percentage points of CER (charter 1.6)


def _load_model_test_tool():
    """tools/crispasr_model_test.py, for its scoring, Hugging Face and GPU helpers (unchanged, read only)."""
    spec = importlib.util.spec_from_file_location("crispasr_model_test", Path(__file__).with_name("crispasr_model_test.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmt = _load_model_test_tool()
say, dump, load_json, read_jsonl = cmt.say, cmt.dump, cmt.load_json, cmt.read_jsonl


# ----------------------------------------------------------------------------------------------------------------
# Candidates and model files
# ----------------------------------------------------------------------------------------------------------------

def load_candidates():
    return load_json(DATA_FILE)["candidates"]


def candidate(name, cands=None):
    cands = cands or load_candidates()
    if name not in cands:
        raise SystemExit(f"Unknown candidate {name!r}. Known: {', '.join(sorted(cands))}")
    return cands[name]


def file_spec(name, cands=None):
    """The entry that names the model file of a candidate (itself, or the one it shares the file with)."""
    cands = cands or load_candidates()
    spec = candidate(name, cands)
    return candidate(spec["same_file_as"], cands) if spec.get("same_file_as") else spec


def model_path(name, cands=None):
    """The model file of a candidate: a fixed name (WebRTC needs none), the cached Hugging Face file, or None."""
    spec = file_spec(name, cands)
    if spec.get("path"):
        return spec["path"]
    if not spec.get("repo"):
        return None
    p = cmt.cached_file(spec["repo"], spec["file"], spec.get("revision"))
    return str(p) if p else None


def needs_file(name, cands=None):
    spec = file_spec(name, cands)
    return bool(spec.get("repo"))


# ----------------------------------------------------------------------------------------------------------------
# The two swapped stages (run inside the child process)
# ----------------------------------------------------------------------------------------------------------------

class CrispEnhancer:
    """A CrispASR model as WhisperJAV's speech enhancer (the SpeechEnhancer protocol of speech_enhancement/base.py).

    Separators: the mono scene is duplicated into 44.1 kHz stereo (as WhisperJAV's htdemucs does, htdemucs.py:296-303),
    Session.separate splits it, the named stem is averaged back to mono. Audio-to-audio speech models (Sidon, VoxCPM2
    AudioVAE): Session.speech_to_speech on 16 kHz mono; they answer at their own rate (48 kHz). WhisperJAV's own
    enhance_scenes() resamples every answer to 16 kHz and handles a failed scene as it always does."""

    def __init__(self, name, spec, path, device=None, use_gpu=None, threads=8, record=None):
        self._name, self.spec, self.path = name, spec, path
        self.use_gpu = (str(device or "auto") != "cpu") if use_gpu is None else bool(use_gpu)
        self.threads = threads
        self.session = None
        self.record = record if record is not None else {}
        self.record.update({"scenes": 0, "failed": 0, "audio_s": 0.0, "enhance_s": 0.0, "open_s": None, "errors": []})

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return f"CrispASR {self.spec.get('backend')} ({Path(str(self.path)).name})"

    def _ensure_initialized(self):
        if self.session is None:
            import crispasr
            t = time.time()
            self.session = crispasr.Session(self.path, n_threads=self.threads, backend=self.spec.get("backend"),
                                            use_gpu=self.use_gpu, flash_attn=True)
            self.record["open_s"] = round(time.time() - t, 2)
            self.record["backend_opened"] = getattr(self.session, "backend", None)
        return True

    def get_preferred_sample_rate(self):
        return SEPARATOR_SR if self.spec["kind"] == "crisp-separator" else SR

    def get_output_sample_rate(self):
        return SEPARATOR_SR if self.spec["kind"] == "crisp-separator" else 48000

    def enhance(self, audio, sample_rate, **kwargs):
        import numpy as np
        from whisperjav.modules.speech_enhancement.base import EnhancementResult, create_failed_result
        t = time.time()
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1).astype(np.float32)
        self.record["scenes"] += 1
        self.record["audio_s"] += len(mono) / float(sample_rate)
        try:
            self._ensure_initialized()
            out, out_sr = self.process(mono, sample_rate)
            self.record["enhance_s"] += time.time() - t
            return EnhancementResult(audio=out, sample_rate=out_sr, method=self._name, parameters={},
                                     processing_time_sec=time.time() - t, success=True)
        except Exception as e:  # one scene's failure is reported, as WhisperJAV's own enhancers report theirs
            self.record["failed"] += 1
            self.record["enhance_s"] += time.time() - t
            if len(self.record["errors"]) < 5:
                self.record["errors"].append(f"{type(e).__name__}: {e}")
            return create_failed_result(audio=mono, sample_rate=sample_rate, method=self._name,
                                        error_message=f"{self._name}: {e}", processing_time_sec=time.time() - t)

    def process(self, mono, sample_rate):
        """(output audio, its rate) for one scene."""
        import numpy as np
        if self.spec["kind"] == "crisp-separator":
            if sample_rate != SEPARATOR_SR:
                raise ValueError(f"separator input must be {SEPARATOR_SR} Hz, got {sample_rate}")
            stereo = np.repeat(mono[:, None], 2, axis=1)
            stems = self.session.separate(stereo)
            stem = self.spec.get("stem", "vocals")
            if stem not in stems:
                raise RuntimeError(f"no '{stem}' stem; stems: {sorted(stems)}")
            out = stems[stem].reshape(-1, 2).mean(axis=1).astype(np.float32)
            return out, SEPARATOR_SR
        if sample_rate != SR:
            self.session.set_pcm_sample_rate(int(sample_rate))
        out, _transcript = self.session.speech_to_speech(mono)
        out = np.asarray(out, dtype=np.float32)
        return out, output_rate(len(mono), sample_rate, len(out))

    def cleanup(self):
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None


def output_rate(n_in, sr_in, n_out, rates=(16000, 22050, 24000, 32000, 44100, 48000)):
    """The output rate of an audio-to-audio model, from the lengths in and out (the nearest common rate)."""
    if not n_in:
        return sr_in
    est = n_out / float(n_in) * sr_in
    return min(rates, key=lambda r: abs(r - est))


def crisp_vad_settings(spec, config):
    """The settings a CrispASR VAD candidate runs with (charter 1.2): 'twin' = the values WhisperJAV resolved for its
    twin in Parakeet-CrispASR (threshold, shortest speech and silence, start and end pads); 'own' = CrispASR's own
    defaults (threshold left to the model, 250 ms, 100 ms, 30 ms pads). Grouping and the longest speech segment are
    always the pipeline's (from the resolved settings)."""
    if spec.get("settings") == "own":
        s = {"threshold": 0.0, "min_speech_ms": 250, "min_silence_ms": 100, "start_pad_ms": 30, "end_pad_ms": 30}
    else:
        pad = config.get("speech_pad_ms", 0) or 0
        s = {"threshold": 0.0 if spec.get("model_threshold") else float(config.get("threshold") or 0.0),
             "min_speech_ms": int(config.get("min_speech_duration_ms") or 0),
             "min_silence_ms": int(config.get("min_silence_duration_ms") or 0),
             "start_pad_ms": int(config.get("start_pad_ms", pad) or 0),
             "end_pad_ms": int(config.get("end_pad_ms", pad) or 0)}
    s.update({"max_speech_s": float(config.get("max_speech_duration_s") or config.get("max_group_duration_s") or 0.0),
              "max_group_duration_s": float(config.get("max_group_duration_s") or 29.0),
              "chunk_threshold_s": float(config.get("chunk_threshold_s") if config.get("chunk_threshold_s") is not None
                                         else 1.0),
              "min_group_duration_s": float(config.get("min_group_duration_s") or 0.0)})
    return s


def pad_spans(spans, start_pad, end_pad, duration):
    """Pad speech spans (seconds); clamp to the audio; spans that would overlap meet in the middle of their gap."""
    out = []
    for s, e in sorted(spans):
        out.append([max(0.0, s - start_pad), min(duration, e + end_pad), s, e])
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][1]:
            mid = max(out[i - 1][3], min(out[i][2], (out[i - 1][3] + out[i][2]) / 2.0))
            out[i - 1][1] = max(out[i - 1][0], mid)
            out[i][0] = min(out[i][1], mid)
    return [(a, b) for a, b, _s, _e in out if b > a]


class CrispSegmenter:
    """A CrispASR VAD as WhisperJAV's speech segmenter backend (the SpeechSegmenter protocol of
    speech_segmentation/base.py): crispasr.vad_slices finds the speech spans, the suite pads them, WhisperJAV's own
    group_segments() (backends/ten.py) makes the VAD groups."""

    def __init__(self, name, spec, path, record=None, threads=8, **config):
        self._name, self.spec, self.path, self.threads = name, spec, path, threads
        self.settings = crisp_vad_settings(spec, config)
        self.record = record if record is not None else {}
        self.record.update({"scenes": 0, "failed": 0, "vad_s": 0.0, "audio_s": 0.0, "spans": 0, "errors": [],
                            "settings": self.settings})

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return f"CrispASR VAD {self._name}"

    def segment(self, audio, sample_rate=16000, **kwargs):
        import numpy as np
        import crispasr
        from whisperjav.modules.speech_segmentation.base import SegmentationResult, SpeechSegment
        from whisperjav.modules.speech_segmentation.backends.ten import group_segments
        t = time.time()
        pcm = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        if pcm.ndim > 1:
            pcm = np.ascontiguousarray(pcm.mean(axis=1), dtype=np.float32)
        if sample_rate != SR:
            from whisperjav.modules.speech_enhancement.pipeline_helper import resample_audio
            pcm = np.ascontiguousarray(resample_audio(pcm, sample_rate, SR), dtype=np.float32)
        duration = len(pcm) / SR
        st = self.settings
        self.record["scenes"] += 1
        self.record["audio_s"] += duration
        try:
            spans = crispasr.vad_slices(pcm, self.path, sample_rate=SR, threshold=st["threshold"],
                                        min_speech_ms=st["min_speech_ms"], min_silence_ms=st["min_silence_ms"],
                                        speech_pad_ms=0, max_chunk_duration_s=st["max_speech_s"],
                                        n_threads=self.threads)
            padded = pad_spans([(sp.start, sp.end) for sp in spans], st["start_pad_ms"] / 1000.0,
                               st["end_pad_ms"] / 1000.0, duration)
            segs = [SpeechSegment(start_sec=a, end_sec=b, start_sample=int(a * SR), end_sample=int(b * SR))
                    for a, b in padded]
            groups = group_segments(segs, st["max_group_duration_s"], st["chunk_threshold_s"],
                                    st["min_group_duration_s"])
            self.record["spans"] += len(spans)
            self.record["vad_s"] += time.time() - t
            return SegmentationResult(segments=segs, groups=groups, method=self._name, audio_duration_sec=duration,
                                      parameters=dict(st), processing_time_sec=time.time() - t)
        except Exception as e:
            self.record["failed"] += 1
            self.record["vad_s"] += time.time() - t
            if len(self.record["errors"]) < 5:
                self.record["errors"].append(f"{type(e).__name__}: {e}")
            return SegmentationResult(segments=[], groups=[], method=self._name, audio_duration_sec=duration,
                                      parameters=dict(st), processing_time_sec=time.time() - t,
                                      error=f"{type(e).__name__}: {e}")

    def cleanup(self):
        pass

    def get_supported_sample_rates(self):
        return [SR]


# ----------------------------------------------------------------------------------------------------------------
# The child process: Parakeet-CrispASR with one stage swapped
# ----------------------------------------------------------------------------------------------------------------

# The steps of the CrispASR chain whose windows are recorded (crispasr_pipeline.py: process()), in order: model check,
# audio extraction, scene detection, speech enhancer, speech segmenter, ASR, subtitle building, clean-up (W586).
CHAIN_STEPS = ("ready", "extract_audio", "detect_scenes", "enhance", "speech_groups", "transcribe", "build_subtitles",
               "postprocess")
STEP_LABELS = {"ready": "model check", "extract_audio": "audio extraction", "detect_scenes": "scene detection",
               "enhance": "speech enhancer", "speech_groups": "speech segmenter", "transcribe": "ASR (CrispASR)",
               "build_subtitles": "subtitle building", "postprocess": "clean-up"}


def chain_settings(name, spec, track):
    """(speech enhancer name, speech segmenter name the pipeline resolves its settings for, enhance for VAD)."""
    kind = spec["kind"]
    enhancer = spec.get("enhancer", "none")
    segmenter = spec.get("segmenter") or spec.get("twin_segmenter") or "ten"
    if kind in ("crisp-separator", "crisp-s2s", "wj-enhancer-swap"):
        enhancer = name
    if track == "vad" and enhancer in ("", "none"):
        raise SystemExit(f"{name}: --track vad needs a speech enhancer")
    return enhancer, segmenter, track == "vad"


def worker(job_path):
    job = load_json(job_path)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    out = open(job["events"], "a", encoding="utf-8")

    def emit(rec):
        rec["t"] = round(time.time(), 3)
        out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out.flush()

    name, spec = job["candidate"], job["spec"]
    kind = spec["kind"]
    import crispasr
    emit({"kind": "start", "crispasr": getattr(crispasr, "__version__", "?"), "pid": os.getpid()})

    # Measuring only: the time of each speech segmenter call (framer.frame), for every segmenter alike.
    from whisperjav.modules.subtitle_pipeline.framers import vad_grouped
    vad_clock = {"s": 0.0}
    orig_frame = vad_grouped.VadGroupedFramer.frame

    def timed_frame(self, audio, sample_rate, **kw):
        t = time.perf_counter()
        try:
            return orig_frame(self, audio, sample_rate, **kw)
        finally:
            vad_clock["s"] += time.perf_counter() - t
    vad_grouped.VadGroupedFramer.frame = timed_frame

    stage_record = {}
    # A1 swap point: the one enhancer name the pipeline is given is answered here.
    if kind in ("crisp-separator", "crisp-s2s", "wj-enhancer-swap"):
        import whisperjav.modules.speech_enhancement as se
        orig_ced = se.create_enhancer_direct

        def create_enhancer_direct(backend, model=None, **kw):
            if backend == name:
                if kind == "wj-enhancer-swap":
                    return orig_ced(spec["swap_enhancer"], model, **kw)
                return CrispEnhancer(name, spec, job["model_path"], device=kw.get("device"), record=stage_record)
            return orig_ced(backend, model, **kw)
        se.create_enhancer_direct = create_enhancer_direct
    # A2 swap point: the one segmenter name is answered here, with the settings WhisperJAV resolved for the twin.
    if kind in ("crisp-vad", "wj-segmenter-swap"):
        from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
        orig_create = SpeechSegmenterFactory.create

        def create(backend, config=None, **kw):
            if backend == name:
                params = {**(config or {}), **kw}
                if kind == "wj-segmenter-swap":
                    return orig_create(spec["twin_segmenter"], None, **params)
                return CrispSegmenter(name, spec, job["model_path"], record=stage_record, **params)
            return orig_create(backend, config, **kw)
        SpeechSegmenterFactory.create = staticmethod(create)

    import dataclasses
    from whisperjav.pipelines.parakeet_crispasr_pipeline import ParakeetCrispASRPipeline
    enhancer, segmenter, enhance_for_vad = chain_settings(name, spec, job.get("track"))
    run_dir = Path(job["run_dir"])
    # As the Ensemble pass worker builds a CrispASR pass (pass_worker.py: crispasr_kwargs) for a pass that names only
    # its pipeline, its speech enhancer and its speech segmenter.
    pipeline = ParakeetCrispASRPipeline(
        output_dir=str(run_dir), temp_dir=str(run_dir / "_temp"), keep_temp_files=False, save_metadata_json=True,
        progress_display=None, model_name=None, speech_segmenter=segmenter, language="ja", subs_language="native",
        sensitivity=None, device=None, vad_threshold=None, speech_pad_ms=None, scene_detector=None,
        enhance_for_vad=enhance_for_vad, word_timestamps=None,
        speech_enhancer=None if enhancer in ("", "none") else enhancer, speech_enhancer_model=None,
        segmenter_overrides={}, scene_params={}, model_params={})
    if kind in ("crisp-vad", "wj-segmenter-swap"):
        pipeline.segmenter = dataclasses.replace(pipeline.segmenter, backend=name)
        pipeline.speech_segmenter = name
    pipeline.raw_subs_dir = str(run_dir / "raw_subs")
    pipeline.pass_tag = "pass1"
    emit({"kind": "pipeline", "enhancer": enhancer, "enhance_for_vad": enhance_for_vad,
          "segmenter": pipeline.speech_segmenter, "segmenter_settings": pipeline.segmenter.as_dict(),
          "model": pipeline.model.label})

    # Each step's window (wall clock), for its duration and for the parent's GPU and RAM samples (W586).
    for stage in CHAIN_STEPS:
        orig = getattr(pipeline, "_" + stage)

        def wrapped(*a, _orig=orig, _stage=stage, **k):
            t0 = time.time()
            try:
                return _orig(*a, **k)
            finally:
                emit({"kind": "stage", "stage": _stage, "t0": round(t0, 3), "t1": round(time.time(), 3)})
        setattr(pipeline, "_" + stage, wrapped)

    code = 0
    try:
        for media in job["media"]:
            media = Path(media)
            vad_clock["s"] = 0.0
            before = dict(stage_record)
            t = time.time()
            try:
                md = pipeline.process({"path": media, "basename": media.stem})
            except Exception as e:
                import traceback
                emit({"kind": "file_failed", "clip": media.stem, "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc()[-3000:]})
                code = 1
                continue
            emit({"kind": "file", "clip": media.stem, "seconds": round(time.time() - t, 2),
                  "final_srt": md.get("srt_path"), "stages": md.get("stages"), "summary": md.get("summary"),
                  "vad_s": round(vad_clock["s"], 3), "stage_record": dict(stage_record),
                  "stage_record_before": before})
    finally:
        try:
            pipeline.cleanup()
        except Exception as e:
            emit({"kind": "cleanup_error", "error": str(e)})
    emit({"kind": "end", "stage_record": dict(stage_record)})
    out.close()
    return code


# ----------------------------------------------------------------------------------------------------------------
# The parent: run, score, record
# ----------------------------------------------------------------------------------------------------------------

def tree_rss_mib(pid):
    """Resident memory (RAM) of a process and all its descendants, MiB; None when it is gone."""
    try:
        import psutil
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except Exception:
        return None
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except Exception:
            pass
    return round(total / 2 ** 20, 1)


class Sampler(threading.Thread):
    """Every half second, with the time: the card's memory in use (nvidia-smi, MiB) and the RAM of the watched child
    process and its descendants (psutil RSS, MiB). samples: [(time, gpu_mib or None, ram_mib or None)]."""

    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.index, self.samples, self.pid, self._stop = index, [], None, threading.Event()

    def watch(self, pid):
        self.pid = pid

    def run(self):
        while not self._stop.is_set():
            g = cmt.gpu_used_mib(self.index)
            r = tree_rss_mib(self.pid) if self.pid else None
            self.samples.append((time.time(), g if g is not None and g >= 0 else None, r))
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(5)

    def gpu_peak(self):
        vals = [g for _, g, _ in self.samples if g is not None]
        return max(vals) if vals else None

    def ram_peak(self):
        vals = [r for _, _, r in self.samples if r is not None]
        return max(vals) if vals else None


GpuSampler = Sampler  # the name used before W586


def clips_of(args):
    clips = cmt.find_clips(args.clips or NETFLIX)
    if getattr(args, "only", None):
        clips = [c for c in clips if any(o in c[0].name for o in args.only.split(","))]
    if not clips:
        raise SystemExit("No clips (give --clips FOLDER with <stem>.ja.srt or <stem>.srt beside each media file).")
    return clips


def run_name(name, track, given=None):
    return given or (name + ("_vadtrack" if track == "vad" else ""))


def spawn(job, folder, log_name="run.log", sampler=None):
    """The child process; returns its exit code. A sampler, when given, watches its RAM."""
    dump(folder / "job.json", job)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(folder / log_name, "a", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), job["worker"],
                                 str(folder / "job.json")], cwd=str(REPO_ROOT), env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        if sampler is not None:
            sampler.watch(proc.pid)
        code = proc.wait()
        if sampler is not None:
            sampler.watch(None)
    return code


def run_candidate(name, out_root, clips, track=None, given_name=None, gpu_index=0):
    cands = load_candidates()
    spec = candidate(name, cands)
    path = model_path(name, cands)
    if needs_file(name, cands) and not path:
        raise SystemExit(f"{name}: the model file is not downloaded; run: fetch {name} --out ...")
    folder = Path(out_root) / run_name(name, track, given_name)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    say(f"\n== {folder.name}: {len(clips)} clip(s); {spec.get('note') or spec['kind']}")
    groups = [[m for m, _ in clips]] if not spec.get("one_file_per_process") else [[m] for m, _ in clips]
    sampler = Sampler(gpu_index)
    gpu_before = cmt.gpu_used_mib(gpu_index)
    sampler.start()
    began = time.time()
    codes = []
    for media_list in groups:
        job = {"worker": "_worker", "candidate": name, "spec": spec, "model_path": path, "track": track,
               "run_dir": str(folder), "events": str(folder / "events.jsonl"), "media": [str(m) for m in media_list]}
        codes.append(spawn(job, folder, sampler=sampler))
    wall = time.time() - began
    sampler.stop()
    dump(folder / "samples.json", sampler.samples)
    events = read_jsonl(folder / "events.jsonl")
    res = summarise(name, spec, track, folder, clips, events, codes, wall, gpu_before, sampler.samples, path)
    dump(folder / "result.json", res)
    write_result_md(folder / "result.md", res)
    p = res.get("pooled") or {}
    say(f"{folder.name}: CER {fmt_pct(p.get('cer'))}, {res['files_done']}/{len(clips)} files, {wall:.0f} s"
        + (f"; CRASH/FAIL exit codes {codes}" if any(codes) else ""))
    return res


def step_figures(samples, events, step, gpu_before, audio_s=None):
    """One chain step over the whole run (W586): its calls, total seconds, x real time, and from the half-second
    samples inside its windows (0.6 s either side): the GPU memory's peak over the use before the run, the largest
    rise of GPU memory and of RAM within one call (over the last sample before it began), the RAM peak of the run's
    process tree. Only samples inside a call's own window count (0.3 s after its end, for nvidia-smi's lag), so a step
    shorter than the half-second sampling step shows no memory figure rather than a neighbour's."""
    wins = [(e["t0"], e["t1"]) for e in events if e.get("kind") == "stage" and e.get("stage") == step]
    if not wins:
        return None
    gpu_in, ram_in, g_rise, r_rise = [], [], [], []
    for a, b in wins:
        inside = [(g, r) for t, g, r in samples if a <= t <= b + 0.3]
        before = [(g, r) for t, g, r in samples if t < a]
        g0 = next((g for g, _ in reversed(before) if g is not None), None)
        r0 = next((r for _, r in reversed(before) if r is not None), None)
        gs = [g for g, _ in inside if g is not None]
        rs = [r for _, r in inside if r is not None]
        gpu_in += gs
        ram_in += rs
        if gs and g0 is not None:
            g_rise.append(max(gs) - g0)
        if rs and r0 is not None:
            r_rise.append(max(rs) - r0)
    secs = sum(b - a for a, b in wins)
    return {"calls": len(wins), "seconds": round(secs, 2),
            "x_realtime": round(audio_s / secs, 1) if audio_s and secs > 0 else None,
            "gpu_peak_over_run_start_mib": (max(gpu_in) - gpu_before) if gpu_in and gpu_before is not None else None,
            "gpu_rise_mib": max(g_rise) if g_rise else None,
            "ram_peak_mib": max(ram_in) if ram_in else None,
            "ram_rise_mib": round(max(r_rise), 1) if r_rise else None}


def stage_gpu_peak(samples, events, stage, before):
    f = step_figures(samples, events, stage, before)
    return f["gpu_peak_over_run_start_mib"] if f else None


def vad_metrics(groups_file, ref, duration):
    """The VAD groups (as sent to the model) against the reference lines: frame precision, recall, F1 at 10 ms
    (tools/vad_groundtruth_analyser/metrics.py, unchanged), plus the frame counts for pooling."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.vad_groundtruth_analyser.metrics import segments_to_mask
    from tools.vad_groundtruth_analyser.models import GtSegment, VadSegment
    gt = [GtSegment(i + 1, s, e, t) for i, (s, e, t) in enumerate(cmt.read_srt(ref))]
    test = [VadSegment(float(r["start"]), float(r["end"])) for r in read_jsonl(groups_file)] if Path(groups_file).exists() \
        else []
    g = segments_to_mask(gt, duration, 10)
    s = segments_to_mask(test, duration, 10)
    tp, gp, sp = int((g & s).sum()), int(g.sum()), int(s.sum())
    return {"tp": tp, "ref_frames": gp, "test_frames": sp, "groups": len(test),
            "group_seconds": round(sum(x.end_sec - x.start_sec for x in test), 1)}


def prf(tp, ref_frames, test_frames):
    p = tp / test_frames if test_frames else 0.0
    r = tp / ref_frames if ref_frames else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def media_seconds(media):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def summarise(name, spec, track, folder, clips, events, codes, wall, gpu_before, samples, path):
    files = {e["clip"]: e for e in events if e.get("kind") == "file"}
    failed = {e["clip"]: e["error"] for e in events if e.get("kind") == "file_failed"}
    start = next((e for e in events if e.get("kind") == "start"), {})
    pipe = next((e for e in events if e.get("kind") == "pipeline"), {})
    per_clip, scores, vad = [], [], []
    for media, ref in clips:
        stem = media.stem
        f = files.get(stem)
        row = {"clip": stem, "done": bool(f), "error": failed.get(stem)}
        if f:
            st = f.get("stages") or {}
            summ = f.get("summary") or {}
            srt = Path(f["final_srt"]) if f.get("final_srt") else None
            row.update({"seconds": f["seconds"], "scenes": (st.get("scene_detection") or {}).get("scenes"),
                        "groups": (st.get("groups") or {}).get("count"),
                        "group_seconds": (st.get("groups") or {}).get("seconds"),
                        "failed_scenes_vad": (st.get("groups") or {}).get("failed_scenes"),
                        "enhance_s": (st.get("enhancement") or {}).get("time_sec"), "vad_s": f.get("vad_s"),
                        "asr_s": (st.get("asr") or {}).get("seconds"),
                        "subtitles": summ.get("final_subtitles_refined"), "degradations": summ.get("degradations"),
                        "notes": summ.get("notes")})
            if srt and srt.exists() and ref:
                sc = cmt.score(ref, srt, folder / "_scores" / stem)
                res_json = folder / "_scores" / stem / "result.json"
                if res_json.exists():
                    sc["cer_plain"] = load_json(res_json)["summary"].get("cer_plain")
                row["score"] = sc
                scores.append(sc)
            gfile = folder / "raw_subs" / f"{stem}.{CHAIN_PIPELINE}.groups.jsonl"
            dur = media_seconds(media)
            if ref and dur:
                vm = vad_metrics(gfile, ref, dur)
                row["vad"] = dict(vm, **dict(zip(("precision", "recall", "f1"), prf(vm["tp"], vm["ref_frames"],
                                                                                     vm["test_frames"]))))
                vad.append(vm)
        per_clip.append(row)
    pooled = cmt.pooled(scores) if len(scores) == len(clips) else {}
    if pooled:
        c = sum(s["ref_chars"] for s in scores)
        pooled["cer_plain"] = sum((s.get("cer_plain") or 0) * s["ref_chars"] for s in scores) / c if c else None
    vad_pooled = None
    if vad and len(vad) == len(clips):
        tp, gf, tf = (sum(v[k] for v in vad) for k in ("tp", "ref_frames", "test_frames"))
        p, r, f1 = prf(tp, gf, tf)
        vad_pooled = {"precision": p, "recall": r, "f1": f1, "groups": sum(v["groups"] for v in vad),
                      "group_seconds": round(sum(v["group_seconds"] for v in vad), 1)}
    done = [r for r in per_clip if r["done"]]
    audio_s = sum(media_seconds(m) or 0 for m, _ in clips)
    total = lambda k: round(sum((r.get(k) or 0) for r in done), 2)  # noqa: E731
    end = next((e for e in reversed(events) if e.get("kind") == "end"), {})
    return {
        "run": folder.name, "candidate": name, "area": spec.get("area"), "kind": spec["kind"], "track": track or "one",
        "model_file": path, "spec": spec, "crispasr": start.get("crispasr"), "whisperjav_commit": cmt.git_commit(),
        "tool": TOOL_VERSION, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_codes": codes,
        "pipeline": pipe, "files_done": len(done), "files": len(clips), "per_clip": per_clip, "pooled": pooled,
        "vad_pooled": vad_pooled, "wall_s": round(wall, 1), "audio_s": round(audio_s, 1),
        "enhance_s": total("enhance_s"), "vad_s": total("vad_s"), "asr_s": total("asr_s"),
        "gpu_before_mib": gpu_before, "gpu_peak_over_before_mib": (max(g for _, g, _ in samples if g is not None)
                                                                   - gpu_before)
        if any(g is not None for _, g, _ in samples) and gpu_before is not None else None,
        "ram_peak_mib": max((r for _, _, r in samples if r is not None), default=None),
        "components": {s: step_figures(samples, events, s, gpu_before, audio_s) for s in CHAIN_STEPS},
        "gpu_enhance_peak_mib": stage_gpu_peak(samples, events, "enhance", gpu_before),
        "gpu_vad_peak_mib": stage_gpu_peak(samples, events, "speech_groups", gpu_before),
        "gpu_asr_peak_mib": stage_gpu_peak(samples, events, "transcribe", gpu_before),
        "stage_record": end.get("stage_record") or {},
    }


def fmt_pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f} %"


def fmt(x, spec="{:.2f}"):
    return "–" if x is None else (spec.format(x) if isinstance(x, (int, float)) else str(x))


def write_result_md(path, r):
    p = r.get("pooled") or {}
    L = [f"# {r['run']}", "",
         f"Candidate `{r['candidate']}` ({r['kind']}, area {r['area']}, track {r['track']}); CrispASR {r['crispasr']}; "
         f"WhisperJAV code {r['whisperjav_commit']} (read only); suite {r['tool']}; {r['when']}.",
         f"Chain: {CHAIN_PIPELINE} as shipped; enhancer `{r['pipeline'].get('enhancer')}` (enhance for VAD: "
         f"{r['pipeline'].get('enhance_for_vad')}); segmenter `{r['pipeline'].get('segmenter')}`; model "
         f"{r['pipeline'].get('model')}. Segmenter settings: `{json.dumps(r['pipeline'].get('segmenter_settings'))}`.",
         f"Files done: {r['files_done']} of {r['files']}; exit codes {r['exit_codes']}.", "",
         "## Pooled (subeval; CER = all edits / all reference characters)", "",
         "| CER | CER plain | Wrong | Missing | Extra | Missed lines | Susp. halluc. lines | Lines |",
         "|---|---|---|---|---|---|---|---|",
         f"| {fmt_pct(p.get('cer'))} | {fmt_pct(p.get('cer_plain'))} | {fmt_pct(p.get('wrong_rate'))} | "
         f"{fmt_pct(p.get('missing_rate'))} | {fmt_pct(p.get('extra_rate'))} | {p.get('missed_lines', '–')} | "
         f"{p.get('hallucination_lines', '–')} | {p.get('test_lines', '–')} |", ""]
    v = r.get("vad_pooled")
    if v:
        L += [f"VAD groups against the reference lines (10 ms frames): precision {fmt_pct(v['precision'])}, recall "
              f"{fmt_pct(v['recall'])}, F1 {fmt_pct(v['f1'])}; {v['groups']} groups, {v['group_seconds']} s.", ""]
    L += [f"Time: {r['wall_s']} s for {r['audio_s']} s of audio; enhancer {r['enhance_s']} s, speech segmenter "
          f"{r['vad_s']} s, ASR {r['asr_s']} s. GPU over the {r['gpu_before_mib']} MiB in use before: peak "
          f"{r['gpu_peak_over_before_mib']} MiB; enhancer {r['gpu_enhance_peak_mib']}, segmenter {r['gpu_vad_peak_mib']}, "
          f"ASR {r['gpu_asr_peak_mib']}. RAM peak of the run's process tree: {r.get('ram_peak_mib')} MiB.", "",
          "## Each step of the chain (W586)", "",
          "Seconds and x real time over all files; GPU memory over the use before the run, and its largest rise within "
          "one call; RAM of the run's process tree (RSS): its peak, and its largest rise within one call. Sampled every "
          "half second, so a step shorter than that may show no figure. A step's first call includes its model's load.",
          "", "| Step | Calls | Seconds | x real time | GPU peak over start (MiB) | GPU rise in a call (MiB) | "
          "RAM peak (MiB) | RAM rise in a call (MiB) |", "|---|---|---|---|---|---|---|---|"]
    for s in CHAIN_STEPS:
        c = (r.get("components") or {}).get(s)
        if c:
            L.append(f"| {STEP_LABELS[s]} | {c['calls']} | {c['seconds']} | {fmt(c['x_realtime'], '{:.1f}')} | "
                     f"{fmt(c['gpu_peak_over_run_start_mib'], '{:.0f}')} | {fmt(c['gpu_rise_mib'], '{:.0f}')} | "
                     f"{fmt(c['ram_peak_mib'], '{:.0f}')} | {fmt(c['ram_rise_mib'], '{:.0f}')} |")
    L += ["", "## Each clip", "",
          "| Clip | Done | Scenes | VAD groups | Group s | Subtitles | CER | Missing | Missed lines | Halluc. lines | "
          "VAD F1 | Enhancer s | VAD s | ASR s | Notes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in r["per_clip"]:
        sc = c.get("score") or {}
        notes = "; ".join((c.get("degradations") or []) + ([c["error"]] if c.get("error") else []))
        L.append(f"| {c['clip']} | {c['done']} | {c.get('scenes', '–')} | {c.get('groups', '–')} | "
                 f"{fmt(c.get('group_seconds'), '{:.1f}')} | {c.get('subtitles', '–')} | {fmt_pct(sc.get('cer'))} | "
                 f"{fmt_pct(sc.get('missing_rate'))} | {sc.get('missed_lines', '–')} | "
                 f"{sc.get('hallucination_lines', '–')} | {fmt_pct((c.get('vad') or {}).get('f1'))} | "
                 f"{fmt(c.get('enhance_s'))} | {fmt(c.get('vad_s'))} | {fmt(c.get('asr_s'))} | {notes[:300]} |")
    sr = r.get("stage_record") or {}
    if sr:
        L += ["", f"Swapped stage record: `{json.dumps(sr, ensure_ascii=False)[:1500]}`"]
    L += ["", "Files: <clip>.ja.whisperjav.srt (final), raw_subs/ (per-group record, raw copy, artifacts), _scores/<clip>/ "
              "(subeval), events.jsonl, run.log. Definitions: subeval/METHOD.md.", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------

def cmd_fetch(args):
    cands = load_candidates()
    names = sorted(n for n in cands if needs_file(n, cands) and not cands[n].get("same_file_as")) if args.all \
        else args.names
    out = Path(args.out) / "_fetch"
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    bad = 0
    for n in names:
        spec = file_spec(n, cands)
        if not spec.get("repo"):
            say(f"{n}: no model file to fetch")
            continue
        files, sha, card = cmt.hf_listing(spec["repo"], spec.get("revision"))
        size, oid = files.get(spec["file"], (None, None))
        say(f"{n}: {spec['repo']}/{spec['file']} @ {sha[:7]} ({(size or 0) / 1e6:.1f} MB)")
        p = Path(hf_hub_download(spec["repo"], spec["file"], revision=sha))
        got = p.stat().st_size
        digest = cmt.sha256_of(p) if oid else None
        ok = (size is None or got == size) and (oid is None or digest == oid)
        bad += 0 if ok else 1
        dump(out / f"{n}.json", {"candidate": n, "repo": spec["repo"], "file": spec["file"], "revision": sha,
                                 "path": str(p), "size": got, "listed_size": size, "sha256": digest, "listed_sha256": oid,
                                 "ok": ok, "licence_card": card.get("license"), "when": time.strftime("%Y-%m-%d %H:%M:%S")})
        say(f"   {'OK' if ok else 'MISMATCH'}: {p}")
    return 0 if not bad else 2


def cmd_run(args):
    clips = clips_of(args)
    res = run_candidate(args.name_, args.out, clips, track=args.track, given_name=args.run_name,
                        gpu_index=args.gpu_index)
    return 0 if res["files_done"] == len(clips) and not any(res["exit_codes"]) else 5


def cmd_rnnoise(args):
    """CrispASR's RNNoise once, on one second of silence: its answer is the evidence (research 3.4)."""
    code = ("import numpy as np, crispasr\n"
            "from crispasr import _binding as b\n"
            "try:\n"
            "    y = b.enhance_audio_rnnoise(np.zeros(16000, dtype=np.float32))\n"
            "    print('RESULT ok', len(y))\n"
            "except Exception as e:\n"
            "    print('RESULT error', type(e).__name__, e)\n")
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = Path(args.out) / "_rnnoise.json"
    rec = {"exit_code": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:],
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(out, rec)
    say(r.stdout.strip() or r.stderr.strip()[-500:])
    return 0


# --- bench: the swapped stage alone, on the GPU and on the CPU (M3) ---------------------------------------------

def bench_worker(job_path):
    job = load_json(job_path)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    name, spec, device = job["candidate"], job["spec"], job["device"]
    rec = {"candidate": name, "device": device}
    import whisperjav.modules.speech_enhancement  # noqa: F401  (WhisperJAV's imports done before the start RAM)
    import whisperjav.modules.speech_segmentation  # noqa: F401
    rec["ram_start_mib"] = tree_rss_mib(os.getpid())
    piece_s = float(job["piece_seconds"])
    if spec["area"] == "A1":
        from whisperjav.modules.speech_enhancement.pipeline_helper import resample_audio
        raw = subprocess.run([job["ffmpeg"], "-nostdin", "-loglevel", "error", "-i", job["media"], "-t",
                              str(job["seconds"]), "-vn", "-af", "aresample=async=1", "-acodec", "pcm_s16le", "-ar",
                              "48000", "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if spec["kind"].startswith("crisp"):
            enh = CrispEnhancer(name, spec, job["model_path"], use_gpu=(device != "cpu"))
        else:
            from whisperjav.modules.speech_enhancement import create_enhancer_direct
            enh = create_enhancer_direct(backend=spec["enhancer"], model=None, device=device)
        sr = enh.get_preferred_sample_rate()
        a = resample_audio(audio, 48000, sr) if sr != 48000 else audio
        t = time.time()
        if hasattr(enh, "_ensure_initialized"):
            enh._ensure_initialized()
        rec["open_s"] = round(time.time() - t, 2)
        n = int(piece_s * sr)
        t = time.time()
        ok = 0
        pieces = [a[i:i + n] for i in range(0, len(a), n)]
        for piece in pieces:
            r = enh.enhance(piece, sr)
            ok += int(bool(r.success))
        rec.update({"stage_s": round(time.time() - t, 2), "pieces": len(pieces), "pieces_ok": ok})
        enh.cleanup()
    else:
        from whisperjav.crispasr.pass_settings import segmenter_settings
        raw = subprocess.run([job["ffmpeg"], "-nostdin", "-loglevel", "error", "-i", job["media"], "-t",
                              str(job["seconds"]), "-vn", "-af", "aresample=async=1", "-acodec", "pcm_s16le", "-ar",
                              str(SR), "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        twin = spec.get("segmenter") or spec.get("twin_segmenter") or "ten"
        s = segmenter_settings(CHAIN_PIPELINE, twin, None)
        params = {**s.config, "max_group_duration_s": s.max_group_duration_s, "chunk_threshold_s": s.chunk_threshold_s}
        if spec["kind"] == "crisp-vad":
            seg = CrispSegmenter(name, spec, job["model_path"], **params)
        else:
            from whisperjav.modules.speech_segmentation.factory import SpeechSegmenterFactory
            seg = SpeechSegmenterFactory.create(twin, None, **params)
        n = int(piece_s * SR)
        t = time.time()
        pieces = [audio[i:i + n] for i in range(0, len(audio), n)]
        ok = 0
        for piece in pieces:
            r = seg.segment(np.ascontiguousarray(piece), sample_rate=SR)
            ok += int(not getattr(r, "error", None))
        rec.update({"stage_s": round(time.time() - t, 2), "pieces": len(pieces), "pieces_ok": ok,
                    "note": "includes the model's first load"})
        seg.cleanup()
    rec["audio_s"] = round(len(audio) / (48000 if spec["area"] == "A1" else SR), 1)
    rec["x_realtime"] = round(rec["audio_s"] / rec["stage_s"], 1) if rec.get("stage_s") else None
    dump(Path(job["out"]), rec)
    return 0


def cmd_bench(args):
    cands = load_candidates()
    name = args.name_
    spec = candidate(name, cands)
    if spec.get("area") not in ("A1", "A2", "A"):
        raise SystemExit("bench takes an A1 or A2 candidate, or the baseline (its TEN VAD)")
    path = model_path(name, cands)
    if needs_file(name, cands) and not path:
        raise SystemExit(f"{name}: the model file is not downloaded")
    media = cmt.find_clips(args.clips or NETFLIX)
    media = max(media, key=lambda c: media_seconds(c[0]) or 0)[0]
    folder = Path(args.out) / "_bench" / name
    folder.mkdir(parents=True, exist_ok=True)
    devices = ["cuda", "cpu"]
    if spec["area"] in ("A2", "A") and not spec["kind"].startswith("crisp"):
        devices = ["cpu"]   # WhisperJAV's VAD engines in Parakeet-CrispASR run on the CPU (whisperseg: CPU on NVIDIA installs)
    if spec["area"] == "A2" and spec["kind"].startswith("crisp"):
        devices = ["as built"]  # crispasr.vad_slices takes no GPU switch
    if args.devices:
        devices = [d for d in devices if d in args.devices.split(",")] or devices
    results = []
    for dev in devices:
        job = {"worker": "_bench", "candidate": name, "spec": spec, "model_path": path, "device": dev,
               "media": str(media), "seconds": args.cpu_seconds if dev == "cpu" else args.seconds,
               "piece_seconds": args.piece_seconds,
               "ffmpeg": cmt.find_ffmpeg(None), "out": str(folder / f"{dev.replace(' ', '_')}.json")}
        before = cmt.gpu_used_mib(args.gpu_index)
        sampler = Sampler(args.gpu_index)
        sampler.start()
        code = spawn(job, folder, log_name=f"{dev.replace(' ', '_')}.log", sampler=sampler)
        sampler.stop()
        rec = load_json(job["out"]) if Path(job["out"]).exists() else {"error": f"exit code {code}"}
        g = sampler.gpu_peak()
        # RAM: the child's peak, and its rise over the child before the model (worker records it, ram_start_mib).
        rec.update({"exit_code": code, "gpu_peak_over_before_mib": (g - before) if g is not None and before is not None
                    else None, "ram_peak_mib": sampler.ram_peak(), "media": str(media)})
        if rec.get("ram_peak_mib") is not None and rec.get("ram_start_mib") is not None:
            rec["ram_rise_mib"] = round(rec["ram_peak_mib"] - rec["ram_start_mib"], 1)
        dump(Path(job["out"]), rec)
        results.append(rec)
        say(f"{name} on {dev}: {rec.get('stage_s')} s for {rec.get('audio_s')} s ({rec.get('x_realtime')} x real time), "
            f"pieces ok {rec.get('pieces_ok')}/{rec.get('pieces')}, GPU +{rec['gpu_peak_over_before_mib']} MiB, "
            f"RAM peak {rec.get('ram_peak_mib')} MiB (+{rec.get('ram_rise_mib')} over the process before the model)"
            + (f", exit {code}" if code else ""))
    dump(folder / "bench.json", results)
    return 0


# --- table ------------------------------------------------------------------------------------------------------

def verdict(r, ref, margin):
    if not r.get("pooled") or not ref.get("pooled"):
        return "not scored"
    d = r["pooled"]["cer"] - ref["pooled"]["cer"]
    dm = r["pooled"]["missing_rate"] - ref["pooled"]["missing_rate"]
    if d < -margin:
        return "better"
    if d <= margin and dm <= margin:
        return "keeps quality"
    return "worse"


def cmd_table(args):
    root = Path(args.out)
    cands = load_candidates()
    runs = {}
    for f in sorted(root.glob("*/result.json")):
        r = load_json(f)
        runs[r["run"]] = r
    base = runs.get("baseline")
    if not base or not base.get("pooled"):
        raise SystemExit("No scored baseline run (run baseline first).")
    rep = runs.get("baseline-repeat")
    diff = abs(rep["pooled"]["cer"] - base["pooled"]["cer"]) if rep and rep.get("pooled") else None
    margin = max(MARGIN_FLOOR, 2 * diff) if diff is not None else MARGIN_FLOOR
    rows = []
    for name, r in runs.items():
        spec = cands.get(r["candidate"], {})
        twin = spec.get("twin")
        twin_run = None
        if twin:
            twin_run = runs.get(twin + ("_vadtrack" if r["track"] == "vad" else ""))
        rows.append({"run": name, "area": r["area"], "candidate": r["candidate"], "track": r["track"],
                     "pooled": r.get("pooled"), "vad": r.get("vad_pooled"), "files": f"{r['files_done']}/{r['files']}",
                     "vs_baseline": verdict(r, base, margin),
                     "vs_twin": verdict(r, twin_run, margin) if twin_run else None,
                     "twin_run": twin_run["run"] if twin_run else None,
                     "d_cer_pp": round((r["pooled"]["cer"] - base["pooled"]["cer"]) * 100, 2) if r.get("pooled") else None,
                     "enhance_s": r["enhance_s"], "vad_s": r["vad_s"], "asr_s": r["asr_s"], "wall_s": r["wall_s"],
                     "gpu_enhance": r["gpu_enhance_peak_mib"], "gpu_vad": r["gpu_vad_peak_mib"],
                     "degraded_files": sum(1 for c in r["per_clip"] if c.get("degradations")),
                     "simplicity": spec.get("simplicity"),
                     "stage_fig": (r.get("components") or {}).get("enhance" if r["area"] == "A1" else "speech_groups"),
                     "asr_fig": (r.get("components") or {}).get("transcribe"),
                     "components": r.get("components"), "ram_peak": r.get("ram_peak_mib")})
    order = {"A": 0, "A1": 1, "A2": 2, "check": 3}
    rows.sort(key=lambda x: (order.get(x["area"], 9), x["pooled"]["cer"] if x["pooled"] else 9))
    benches = {}
    for f in sorted((root / "_bench").glob("*/bench.json")):
        benches[f.parent.name] = load_json(f)
    out = {"margin": margin, "baseline_cer": base["pooled"]["cer"], "baseline_repeat_cer":
           rep["pooled"]["cer"] if rep and rep.get("pooled") else None, "rows": rows, "bench": benches,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(root / "table.json", out)
    L = [f"# Phase A table ({out['when']})", "",
         f"Baseline CER {fmt_pct(base['pooled']['cer'], 2)}; repeat {fmt_pct(out['baseline_repeat_cer'], 2)}; margin "
         f"{margin * 100:.2f} percentage points (charter 1.6: the larger of 0.5 and twice the two baseline runs' difference).",
         "", "| Run | Area | Track | Files | CER | ΔCER (pp) | Wrong | Missing | Extra | Missed lines | Halluc. lines | "
         "Lines | VAD F1 | vs baseline | vs twin (run) | Enhancer s | VAD s | ASR s | GPU enhancer / VAD (MiB) | "
         "Degraded files |", "|---" * 20 + "|"]
    for x in rows:
        p = x["pooled"] or {}
        L.append(f"| {x['run']} | {x['area']} | {x['track']} | {x['files']} | {fmt_pct(p.get('cer'), 2)} | "
                 f"{fmt(x['d_cer_pp'])} | {fmt_pct(p.get('wrong_rate'))} | {fmt_pct(p.get('missing_rate'))} | "
                 f"{fmt_pct(p.get('extra_rate'))} | {p.get('missed_lines', '–')} | {p.get('hallucination_lines', '–')} | "
                 f"{p.get('test_lines', '–')} | {fmt_pct((x['vad'] or {}).get('f1'))} | {x['vs_baseline']} | "
                 f"{(x['vs_twin'] + ' (' + x['twin_run'] + ')') if x['vs_twin'] else '–'} | {fmt(x['enhance_s'], '{:.0f}')} | "
                 f"{fmt(x['vad_s'], '{:.1f}')} | {fmt(x['asr_s'], '{:.0f}')} | {x['gpu_enhance']} / {x['gpu_vad']} | "
                 f"{x['degraded_files']} |")
    L += ["", "## The swapped stage and the ASR step in each run (W586)", "",
          "Over the seven clips. Stage: the speech enhancer (A1) or the speech segmenter (A2; the baseline's TEN), its "
          "model load included. GPU: rise within one call over the last sample before it. RAM: the run's process tree "
          "(RSS); rise within one call. Sampled every half second.", "",
          "| Run | Stage s | Stage x real time | Stage GPU rise (MiB) | Stage RAM rise (MiB) | Stage RAM peak (MiB) | "
          "ASR s | ASR GPU rise (MiB) | ASR RAM rise (MiB) | Run RAM peak (MiB) | Run s |", "|---" * 11 + "|"]
    for x in rows:
        s, a = x.get("stage_fig") or {}, x.get("asr_fig") or {}
        L.append(f"| {x['run']} | {fmt(s.get('seconds'), '{:.1f}')} | {fmt(s.get('x_realtime'), '{:.0f}')} | "
                 f"{fmt(s.get('gpu_rise_mib'), '{:.0f}')} | {fmt(s.get('ram_rise_mib'), '{:.0f}')} | "
                 f"{fmt(s.get('ram_peak_mib'), '{:.0f}')} | {fmt(a.get('seconds'), '{:.1f}')} | "
                 f"{fmt(a.get('gpu_rise_mib'), '{:.0f}')} | {fmt(a.get('ram_rise_mib'), '{:.0f}')} | "
                 f"{fmt(x.get('ram_peak'), '{:.0f}')} | {fmt(x.get('wall_s'), '{:.0f}')} |")
    bc = next((x for x in rows if x["run"] == "baseline"), None)
    if bc and bc.get("components"):
        L += ["", "## Every step of the baseline (Parakeet-CrispASR as shipped)", "",
              "| Step | Calls | Seconds | x real time | GPU rise in a call (MiB) | RAM rise in a call (MiB) | RAM peak (MiB) |",
              "|---|---|---|---|---|---|---|"]
        for st in CHAIN_STEPS:
            c = bc["components"].get(st)
            if c:
                L.append(f"| {STEP_LABELS[st]} | {c['calls']} | {c['seconds']} | {fmt(c['x_realtime'], '{:.1f}')} | "
                         f"{fmt(c['gpu_rise_mib'], '{:.0f}')} | {fmt(c['ram_rise_mib'], '{:.0f}')} | "
                         f"{fmt(c['ram_peak_mib'], '{:.0f}')} |")
    if benches:
        L += ["", "## The swapped stage alone (bench)", "", "| Candidate | Device | Audio s | Stage s | x real time | "
              "Pieces ok | GPU (MiB) | RAM peak (MiB) | RAM rise (MiB) |", "|---|---|---|---|---|---|---|---|---|"]
        for n, recs in benches.items():
            for b in recs:
                L.append(f"| {n} | {b.get('device')} | {b.get('audio_s')} | {b.get('stage_s')} | {b.get('x_realtime')} | "
                         f"{b.get('pieces_ok')}/{b.get('pieces')} | {b.get('gpu_peak_over_before_mib')} | "
                         f"{fmt(b.get('ram_peak_mib'), '{:.0f}')} | {fmt(b.get('ram_rise_mib'), '{:.0f}')} |")
    L += ["", "## Simplicity facts (data file)", "", "| Candidate | Replaces | Packages no longer needed | Why |",
          "|---|---|---|---|"]
    for n, spec in cands.items():
        s = spec.get("simplicity")
        if s:
            L.append(f"| {n} | {s.get('replaces') or '– (new)'} | {', '.join(s.get('packages_removed') or []) or 'none'} | "
                     f"{s.get('why')} |")
    (root / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    say(f"Written: {root / 'table.md'}")
    return 0


# --- selfcheck ---------------------------------------------------------------------------------------------------

def same_srt(a, b):
    """(identical, detail): the two SRT files, subtitle by subtitle, text and times (1 ms)."""
    x = cmt.read_srt(a) if Path(a).exists() and Path(a).stat().st_size else []
    y = cmt.read_srt(b) if Path(b).exists() and Path(b).stat().st_size else []
    same = sum(1 for p, q in zip(x, y) if p[2].strip() == q[2].strip() and abs(p[0] - q[0]) < 0.0005
               and abs(p[1] - q[1]) < 0.0005)
    return len(x) == len(y) == same and len(x) > 0, f"{len(x)} vs {len(y)} subtitles, {same} identical in order"


def cmd_selfcheck(args):
    root = Path(args.out) / "_selfcheck"
    root.mkdir(parents=True, exist_ok=True)
    clips = [c for c in cmt.find_clips(args.clips or NETFLIX) if args.clip in c[0].name]
    if len(clips) != 1:
        raise SystemExit(f"--clip {args.clip!r} must name exactly one clip")
    media, ref = clips[0]
    stem = media.stem
    checks = []
    # 1. The suite's baseline against WhisperJAV's own command line
    cli = root / "cli"
    if cli.exists():
        shutil.rmtree(cli)
    cli.mkdir()
    cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", str(media), "--ensemble", "--pass1-pipeline",
           CHAIN_PIPELINE, "--merge-strategy", "pass1_primary", "--subs-language", "native", "--language", "japanese",
           "--debug", "--output-dir", str(cli)]
    say("selfcheck 1: WhisperJAV's command line, " + " ".join(cmd[3:]))
    with open(cli / "cli.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT).returncode
    base = run_candidate("baseline", root, clips, given_name="baseline")
    ok, detail = same_srt(cli / f"{stem}.ja.pass1.srt", root / "baseline" / f"{stem}.ja.whisperjav.srt")
    checks.append(("1 suite baseline = WhisperJAV command line (exit %d)" % code, ok, detail))
    # 2. The swap points change nothing but the stage
    swap = run_candidate("check-ten-swap", root, clips)
    ok, detail = same_srt(root / "baseline" / f"{stem}.ja.whisperjav.srt", root / "check-ten-swap" / f"{stem}.ja.whisperjav.srt")
    checks.append(("2a TEN through the A2 swap point = baseline", ok, detail))
    run_candidate("wj-ffmpeg-dsp", root, clips)
    run_candidate("check-ffmpeg-dsp-swap", root, clips)
    ok, detail = same_srt(root / "wj-ffmpeg-dsp" / f"{stem}.ja.whisperjav.srt",
                          root / "check-ffmpeg-dsp-swap" / f"{stem}.ja.whisperjav.srt")
    checks.append(("2b ffmpeg-dsp through the A1 swap point = ffmpeg-dsp", ok, detail))
    # 3. One CrispASR candidate per area, end to end
    for n in args.crisp.split(","):
        if needs_file(n) and not model_path(n):
            checks.append((f"3 {n} end to end", False, "model file not downloaded"))
            continue
        r = run_candidate(n, root, clips)
        c = r["per_clip"][0]
        sr = r.get("stage_record") or {}
        ok = bool(c.get("done") and (c.get("subtitles") or 0) > 0 and not sr.get("failed"))
        checks.append((f"3 {n} end to end", ok, f"{c.get('subtitles')} subtitles, CER {fmt_pct((c.get('score') or {}).get('cer'))}, "
                       f"stage: {sr.get('scenes')} scenes, {sr.get('failed')} failed"))
    L = [f"# Self-check ({time.strftime('%Y-%m-%d %H:%M:%S')}), clip {stem}", "", "| Check | Result | Detail |",
         "|---|---|---|"] + [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    (root / "selfcheck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    say("\n".join(L))
    return 0 if all(ok for _, ok, _ in checks) else 1


# ----------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="crispasr_viability", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download candidate model files at their commits; check size and SHA-256")
    f.add_argument("names", nargs="*")
    f.add_argument("--all", action="store_true")
    f.add_argument("--out", required=True)
    r = sub.add_parser("run", help="Parakeet-CrispASR with one stage swapped, over the clips; scored")
    r.add_argument("name_", metavar="CANDIDATE")
    r.add_argument("--out", required=True)
    r.add_argument("--track", choices=("one", "vad"), default=None,
                   help="A1: one = the enhanced audio to the segmenter and the model (default); vad = enhance for VAD")
    r.add_argument("--run-name", default=None, help="the run's folder name (default: the candidate)")
    r.add_argument("--clips", default=None, help=f"folder of clips with references (default: {NETFLIX})")
    r.add_argument("--only", default=None, help="keep the clips whose names contain one of these (comma-separated)")
    r.add_argument("--gpu-index", type=int, default=0)
    b = sub.add_parser("bench", help="the candidate's stage alone on the GPU and the CPU")
    b.add_argument("name_", metavar="CANDIDATE")
    b.add_argument("--out", required=True)
    b.add_argument("--clips", default=None)
    b.add_argument("--seconds", type=float, default=120.0, help="audio on the GPU (default 120)")
    b.add_argument("--cpu-seconds", type=float, default=30.0, help="audio on the CPU (default 30)")
    b.add_argument("--piece-seconds", type=float, default=30.0)
    b.add_argument("--devices", default=None, help="only these of cuda,cpu (default: both for A1)")
    b.add_argument("--gpu-index", type=int, default=0)
    n = sub.add_parser("rnnoise", help="call CrispASR's RNNoise once and record its answer")
    n.add_argument("--out", required=True)
    t = sub.add_parser("table", help="one comparison table of every run under --out")
    t.add_argument("--out", required=True)
    s = sub.add_parser("selfcheck", help="the suite against WhisperJAV's own runs (design 2.3)")
    s.add_argument("--out", required=True)
    s.add_argument("--clips", default=None)
    s.add_argument("--clip", default="S01E04.Scene.1")
    s.add_argument("--crisp", default="crisp-silero,crisp-htdemucs-f16")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return worker(argv[1])
    if argv and argv[0] == "_bench":
        return bench_worker(argv[1])
    args = build_parser().parse_args(argv)
    return {"fetch": cmd_fetch, "run": cmd_run, "bench": cmd_bench, "rnnoise": cmd_rnnoise, "table": cmd_table,
            "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
