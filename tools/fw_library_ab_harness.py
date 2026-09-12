#!/usr/bin/env python3
"""
fw_library_ab_harness.py — library-level A/B harness for the balanced pipeline's
faster-whisper stack.

PURPOSE
-------
Answer, with numbers you can compare across environments, whether swapping the
faster-whisper library underneath the balanced pipeline changes (a) speed,
(b) speech coverage, (c) word-timestamp availability — and which CALL PATTERN or
PARAMETER RECIPE is responsible for what.

The script deliberately imports NOTHING from whisperjav. It uses only
faster_whisper, numpy and the standard library (torch / pynvml are optional), so
the SAME file runs unchanged in every environment under test:

    WJ    faster-whisper 1.2.1   + CTranslate2 4.6.2   (WhisperJAV dev env)
    FW2   faster-whisper2 2.1.1  + CTranslate2 4.6.2   (candidate fork, isolated env)
    FW2b  faster-whisper2 2.1.1  + CTranslate2 4.8.x   (candidate + newest CT2)
    V07B  faster-whisper 1.0.2   + CTranslate2 4.4.0   (the popular v0.7b notebook stack)

Every run writes a manifest that fingerprints the library, CTranslate2, the
bundled Silero VAD asset, the model files and the GPU, so two result folders can
be compared without guessing what was actually loaded.

ARMS (call pattern x parameter recipe)
--------------------------------------
  wj_native_scene    WhisperJAV balanced recipe + native-VAD preset, ONE
                     transcribe(vad_filter=True) call PER SCENE (scenes <= 29 s,
                     emulated from silence, or supplied via --scenes-json).
                     == the current "internal VAD" balanced path (Problem 2 config).
  wj_native_file     Same recipe, ONE call for the WHOLE FILE. Isolates the
                     per-scene call pattern from the recipe.
  wj_external_group  Emulates the external-segmenter path: Silero speech chunks
                     grouped to <= 9 s with 0.1 s gap tolerance (WhisperJAV's
                     "Test-D" grouping), ONE transcribe(vad_filter=False) call PER
                     GROUP. == the Problem 1 call pattern (P1H1).
  lib_default_file   Library defaults + vad_filter=True (default VadOptions),
                     whole file. Isolates "WhisperJAV over-parameterises" (M3).
  v07b_file          The v0.7b notebook recipe, whole file. The community
                     reference the users compare us against.
  wj_batched_file    BatchedInferencePipeline with the WhisperJAV recipe, whole
                     file. The batched path the fork claims to have fixed.

RECIPES (--recipe; applies to the four wj_* arms only)
------------------------------------------------------
  wj_v192     (default, unchanged behaviour) WhisperJAV v1.9.2 balanced-pipeline
              transcribe kwargs for --sensitivity balanced|aggressive, plus the
              pipeline's native VadOptions. Model loaded float16, which is what
              balanced resolves on CUDA.
  fwdefaults  faster-whisper's OWN defaults for every decoding parameter, with
              two deliberate exceptions: temperature pinned to 0.0 (no fallback
              ladder) and word_timestamps=True. The native-VAD arms use the
              library's default VadOptions. Model loaded compute_type=auto.
              Answers: is it WhisperJAV's parameter recipe, rather than the call
              pattern, that costs coverage or speed? (M3 crossed with P1H1.)

  lib_default_file and v07b_file carry their own fixed recipes and ignore --recipe.
  The Silero parameters used to SPLIT the audio for the per-scene and per-group
  arms are WhisperJAV's preset for --sensitivity in EVERY recipe, so the spans
  those arms transcribe stay comparable when the recipe changes: only the
  decoding parameters move.

OUTPUT (per run directory)
--------------------------
  manifest.json                 environment + model fingerprint + arm recipes
  summary.csv                   one row per (file, arm): timing, VRAM, counts,
                                coverage vs ground truth, CER, log counters
  <stem>/<arm>.segments.jsonl   every segment with logprob / no_speech / CR / temp / words
  <stem>/<arm>.srt              the arm's output as SRT
  <stem>/<arm>.timeline.csv     30 s bins: ground-truth speech, output speech, covered
  <stem>/<arm>.log              faster_whisper DEBUG log captured during the arm
  <stem>/<arm>.calls.jsonl      one line per transcribe() call (span, wall, segments)

Ground truth: an SRT next to the media file (<stem>.srt or <stem>.ja.srt), or
--gt-dir <dir> holding SRTs with the same stem.

USAGE
-----
  python tools/fw_library_ab_harness.py --label WJ \
      --files "test_media/Ground_Truths/Netflix/*.mkv" \
      --out F:/fw_ab_results

  python tools/fw_library_ab_harness.py --label WJ --recipe fwdefaults \
      --arms wj_native_file,wj_external_group \
      --files "F:/MEDIA_DLNA/EKAI-023/EKAI-023.mp4" \
      --out "F:/MEDIA_DLNA/EKAI-023/ab_results_WJ"

  See tools/fw_library_ab_harness.md for the full procedure.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import hashlib
import inspect
import io
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Console hygiene (Windows legacy code pages choke on Japanese)
# --------------------------------------------------------------------------- #
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - very old Pythons
        pass

# CUDA runtime DLLs on Windows. CTranslate2's wheel bundles only the cuDNN loader
# stub (cudnn64_9.dll); the split cuDNN 9 libraries (cudnn_ops64_9.dll, ...) and
# cuBLAS must come from somewhere else. Two sources are supported:
#   1. torch, when installed: importing it registers its bundled DLL folder. This is
#      what WhisperJAV and the Colab notebooks rely on, so we mirror it.
#   2. the NVIDIA pip packages nvidia-cudnn-cu12 / nvidia-cublas-cu12: their bin
#      folders are registered with os.add_dll_directory when present.
_DLL_DIRS_REGISTERED: List[str] = []
if os.name == "nt":
    import site as _site

    for _sp in {*_site.getsitepackages(), _site.getusersitepackages()}:
        for _sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
            _d = Path(_sp) / _sub
            if _d.is_dir():
                try:
                    os.add_dll_directory(str(_d))
                    os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")
                    _DLL_DIRS_REGISTERED.append(str(_d))
                except Exception:  # pragma: no cover
                    pass

try:  # noqa: SIM105
    import torch  # type: ignore

    _TORCH_VERSION = getattr(torch, "__version__", "unknown")
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_VERSION = None

import numpy as np  # noqa: E402

import faster_whisper  # noqa: E402
from faster_whisper import WhisperModel, decode_audio  # noqa: E402
from faster_whisper.vad import get_speech_timestamps  # noqa: E402

try:
    from faster_whisper import BatchedInferencePipeline  # type: ignore
except Exception:  # faster-whisper 1.0.x has no batched pipeline
    BatchedInferencePipeline = None  # type: ignore

try:
    import ctranslate2  # noqa: E402
except Exception:  # pragma: no cover
    ctranslate2 = None  # type: ignore

SAMPLE_RATE = 16000
SCENE_MAX_S = 29.0          # WhisperJAV balanced: scene_detection.max_duration_s
SCENE_GAP_S = 2.5           # WhisperJAV balanced: pass1_max_silence_s (scene boundary)
GROUP_MAX_S = 9.0           # WhisperJAV "Test-D" grouping for external segmenters
GROUP_GAP_S = 0.1           # WhisperJAV "Test-D" chunk_threshold_s
TIMELINE_BIN_S = 30.0

log = logging.getLogger("fw_ab")

# Objects that must outlive main(): the CTranslate2 model's destructor is what crashes the
# process (research README, section 7b), and it would run when main()'s locals are released.
_KEEPALIVE: List[Any] = []


# --------------------------------------------------------------------------- #
# Parameter recipes
# --------------------------------------------------------------------------- #
def wj_recipe(sensitivity: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """WhisperJAV v1.9.2 balanced-pipeline transcribe kwargs + native VadOptions.

    Values are the resolved output of
      python -m whisperjav.main --dump-params <file> --mode balanced --sensitivity <s>
    (2026-09-08) after FasterWhisperProASR._prepare_whisper_params() renaming
    (logprob_threshold -> log_prob_threshold, list temperature -> float/tuple,
    None values dropped). WhisperJAV-side post filters (logprob_margin,
    drop_nonverbal_vocals, post_model_filter) are NOT library parameters and are
    therefore not part of a library A/B.
    """
    if sensitivity == "balanced":
        decode = dict(
            task="transcribe", language="ja",
            beam_size=2, best_of=2, patience=1.2,
            # max_initial_timestamp: 0.0 until 2026-09-11, now 1.0, mirroring the
            # change to the shipped preset. At 0.0 a window has exactly one legal
            # opening timestamp; with beam search the second beam then has no legal
            # alternative and is returned carrying the score CTranslate2 gives a beam
            # at initialisation, decoding to a run of '!' that the compression-ratio
            # check discards. Measured here as 0 subtitles on 7 of 7 clips at float32.
            suppress_blank=True, without_timestamps=False, max_initial_timestamp=1.0,
            temperature=0.0,
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.65,
            condition_on_previous_text=False, word_timestamps=True,
            repetition_penalty=1.5, no_repeat_ngram_size=3,
            multilingual=False, log_progress=False,
        )
        vad = dict(threshold=0.40, min_speech_duration_ms=100, max_speech_duration_s=15.0,
                   min_silence_duration_ms=300, speech_pad_ms=400)
    elif sensitivity == "aggressive":
        decode = dict(
            task="transcribe", language="ja",
            beam_size=3, best_of=2, patience=1.3,
            suppress_blank=True, without_timestamps=False, max_initial_timestamp=0.0,
            temperature=(0.0, 0.2),
            compression_ratio_threshold=2.6, log_prob_threshold=-1.0, no_speech_threshold=0.72,
            condition_on_previous_text=False, word_timestamps=True,
            chunk_length=30,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            multilingual=False, log_progress=False,
        )
        vad = dict(threshold=0.25, min_speech_duration_ms=30, max_speech_duration_s=9.0,
                   min_silence_duration_ms=300, speech_pad_ms=300)
    else:
        raise ValueError(f"unknown sensitivity {sensitivity!r} (balanced|aggressive)")
    return decode, vad


def v07b_recipe() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The v0.7b notebook's transcribe() call (notebook/WhisperJAV_v0_7b.ipynb, cell 8/9).

    Whole-file single call; faster-whisper 1.0.2 defaults for everything not listed.
    initial_prompt (new_vocabulary) is omitted: it is a user-supplied string in the
    notebook and empty by default.
    """
    decode = dict(
        task="transcribe", language="ja",
        condition_on_previous_text=False, word_timestamps=True,
        temperature=0, beam_size=2, best_of=2, patience=2,
        repetition_penalty=1.5, no_repeat_ngram_size=2,
    )
    vad = dict(threshold=0.35, max_speech_duration_s=4)
    return decode, vad


def lib_default_recipe() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Library defaults. Only language (skip detection) and word_timestamps (benefit BC) set."""
    return dict(language="ja", word_timestamps=True), {}


def fw_defaults_recipe() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """faster-whisper's own decoding defaults, written out key by key.

    Every value below IS the library default in faster-whisper 1.2.1, verified
    against WhisperModel.transcribe's signature on 2026-09-09, EXCEPT two:

      temperature=0.0        the library default is the fallback ladder
                             [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. Pinning 0.0 disables
                             temperature fallback, so compression_ratio_threshold
                             and log_prob_threshold can no longer trigger a
                             re-decode: they only mark a window as failed.
      word_timestamps=True   the library default is False; the harness needs word
                             timestamps for the words_* columns (benefit BC).

    hallucination_silence_threshold is deliberately ABSENT (library default None
    = disabled). language and task are set so no per-call language detection runs;
    they are not decoding parameters. Everything else the library decides:
    best_of=5, without_timestamps=False, max_initial_timestamp=1.0,
    prompt_reset_on_temperature=0.5, multilingual=False, log_progress=False --
    on the PLAIN transcribe path. BatchedInferencePipeline overrides several of
    them in its body whatever it is passed (in 1.2.1: condition_on_previous_text,
    hallucination_silence_threshold, max_initial_timestamp,
    prompt_reset_on_temperature, and only the first temperature), so
    wj_batched_file cannot carry this recipe faithfully. The harness reads that
    list out of the installed library and reports it per arm in
    recipe_keys_not_honoured; it is NOT visible in kwargs_dropped_by_library,
    which only inspects the signature.

    compute_type=auto is NOT the library's default (that is "default", which
    resolves to float16 here); auto asks CTranslate2 for the fastest supported
    type, an int8-quantised model on current GPUs. It is part of this recipe
    because the parameter set was specified that way, but it is a model-load
    setting: it applies to every arm in the run, not just the wj_* arms.

    The empty VadOptions dict means: when an arm uses vad_filter=True, pass NO
    vad_parameters, so the library applies its own default VadOptions.
    """
    decode = dict(
        task="transcribe", language="ja",
        beam_size=5,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        suppress_blank=True,
        word_timestamps=True,
    )
    return decode, {}


# --------------------------------------------------------------------------- #
# Recipe registry: named, frozen parameter sets for the four wj_* arms.
# Adding a recipe never changes an existing one, so a result folder written by an
# earlier run stays readable: the recipe id and version are in its manifest, in
# its summary.csv and in its run-folder name.
# --------------------------------------------------------------------------- #
RECIPES: Dict[str, Dict[str, Any]] = {
    "wj_v192": {
        "version": "wj_v192-balanced-20260908",
        "build": wj_recipe,
        "sensitivity_aware": True,
        "compute_type": "float16",
        "description": "WhisperJAV v1.9.2 balanced-pipeline transcribe kwargs + its native "
                       "VadOptions, per --sensitivity (the harness's original behaviour).",
    },
    "fwdefaults": {
        "version": "fwdefaults-v1-20260909",
        "build": lambda _sensitivity: fw_defaults_recipe(),
        "sensitivity_aware": False,
        "compute_type": "auto",
        "description": "faster-whisper's own defaults, temperature pinned to 0.0 and "
                       "word_timestamps=True; library-default VadOptions; compute_type=auto.",
    },
}
DEFAULT_RECIPE = "wj_v192"


# --------------------------------------------------------------------------- #
# Environment fingerprint
# --------------------------------------------------------------------------- #
def _md5(path: Path, limit: Optional[int] = None) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def _nvidia_smi(query: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def environment_fingerprint() -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "faster_whisper_version": getattr(faster_whisper, "__version__", "unknown"),
        "faster_whisper_file": str(Path(faster_whisper.__file__).resolve()),
        "faster_whisper_has_batched": BatchedInferencePipeline is not None,
        "ctranslate2_version": getattr(ctranslate2, "__version__", None) if ctranslate2 else None,
        "torch_version": _TORCH_VERSION,
        "cuda_dll_dirs_registered": _DLL_DIRS_REGISTERED,
        "gpu_name": _nvidia_smi("name"),
        "gpu_driver": _nvidia_smi("driver_version"),
        "gpu_memory_total_mib": _nvidia_smi("memory.total"),
    }
    # Distribution name (faster-whisper vs faster-whisper2 share the import name)
    try:
        from importlib import metadata as _md

        dists = []
        for name in ("faster-whisper", "faster-whisper2", "ctranslate2", "onnxruntime",
                     "onnxruntime-gpu", "numpy", "av", "torch"):
            try:
                dists.append(f"{name}=={_md.version(name)}")
            except _md.PackageNotFoundError:
                pass
        fp["installed_distributions"] = dists
    except Exception:
        fp["installed_distributions"] = None
    if ctranslate2 is not None:
        try:
            fp["ct2_cuda_device_count"] = ctranslate2.get_cuda_device_count()
            fp["ct2_supported_compute_types_cuda"] = sorted(
                ctranslate2.get_supported_compute_types("cuda")
            )
        except Exception as e:  # pragma: no cover
            fp["ct2_probe_error"] = repr(e)
    # Silero VAD asset fingerprint (v4 / v5 / v6 differ by file name and hash)
    try:
        from faster_whisper.utils import get_assets_path

        assets = Path(get_assets_path())
        fp["vad_assets"] = {
            p.name: {"bytes": p.stat().st_size, "md5": _md5(p)}
            for p in sorted(assets.glob("*.onnx"))
        }
    except Exception as e:  # pragma: no cover
        fp["vad_assets"] = {"error": repr(e)}
    return fp


def model_fingerprint(model: Any, model_arg: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"model_arg": model_arg}
    try:
        # WhisperModel keeps no public path; resolve the way the library does.
        from faster_whisper.utils import download_model

        p = Path(download_model(model_arg, local_files_only=True)) if "/" in model_arg or not Path(model_arg).exists() else Path(model_arg)
        info["model_path"] = str(p)
        cfg = p / "config.json"
        binf = p / "model.bin"
        if cfg.exists():
            info["config_md5"] = _md5(cfg)
            cfg_json = json.loads(cfg.read_text(encoding="utf-8"))
            info["config"] = {k: v for k, v in cfg_json.items()
                              if k not in ("lang_ids", "suppress_ids", "alignment_heads", "suppress_ids_begin")}
        if binf.exists():
            info["model_bin_bytes"] = binf.stat().st_size
            info["model_bin_md5_first16MiB"] = _md5(binf, limit=16 << 20)
    except Exception as e:
        info["fingerprint_error"] = repr(e)
    try:
        info["ct2_compute_type_effective"] = getattr(model.model, "compute_type", None)
        info["ct2_device"] = getattr(model.model, "device", None)
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# VRAM sampler
# --------------------------------------------------------------------------- #
class VramSampler:
    """Samples GPU memory.used (MiB) on a thread; reports peak over a window."""

    def __init__(self, interval_s: float = 0.25, gpu_index: int = 0):
        self.interval = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._peak = 0
        self._samples = 0
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._nvml = None

    def _read_process(self) -> Optional[int]:
        """This process's own GPU memory (MiB), when the driver exposes it."""
        pid = os.getpid()
        if self._nvml is not None:
            try:
                for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                    if proc.pid == pid and proc.usedGpuMemory not in (None, 0):
                        return int(proc.usedGpuMemory // (1 << 20))
            except Exception:
                pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid and parts[1].isdigit():
                    return int(parts[1])
        except Exception:
            pass
        return None

    def _read(self) -> Optional[int]:
        """GPU memory.used for the whole device (MiB)."""
        if self._nvml is not None:
            try:
                return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used // (1 << 20))
            except Exception:
                return None
        v = _nvidia_smi("memory.used")
        try:
            return int(float(v)) if v is not None else None
        except ValueError:
            return None

    def read_now(self) -> Optional[int]:
        return self._read()

    def read_process_now(self) -> Optional[int]:
        return self._read_process()

    def start(self) -> None:
        self._stop.clear()
        self._peak = 0
        self._peak_proc = 0
        self._samples = 0
        self._baseline = self._read()            # device total just before this arm
        self._baseline_proc = self._read_process()

        def _loop():
            while not self._stop.is_set():
                v = self._read()
                if v is not None:
                    self._peak = max(self._peak, v)
                    self._samples += 1
                vp = self._read_process()
                if vp is not None:
                    self._peak_proc = max(self._peak_proc, vp)
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        ok = self._samples > 0
        return {
            "samples": self._samples,
            "vram_device_baseline_mib": self._baseline if self._baseline is not None else "",
            "vram_device_peak_mib": self._peak if ok else "",
            "vram_device_delta_mib": (self._peak - self._baseline) if (ok and self._baseline is not None) else "",
            "vram_process_baseline_mib": self._baseline_proc if self._baseline_proc is not None else "",
            "vram_process_peak_mib": self._peak_proc if self._peak_proc else "",
        }


# --------------------------------------------------------------------------- #
# faster_whisper log capture
# --------------------------------------------------------------------------- #
LOG_COUNTERS = {
    "no_speech_skips": "No speech threshold is met",
    "compression_ratio_fallbacks": "Compression ratio threshold is not met",
    "log_prob_fallbacks": "Log probability threshold is not met",
    "prompt_resets": "Reset prompt",
    "windows_processed": "Processing segment at",
    "vad_filter_removed_msgs": "VAD filter removed",
}


class FwLogCapture:
    def __init__(self, path: Path):
        self.path = path
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger = logging.getLogger("faster_whisper")
        self._prev_level = self.logger.level

    def __enter__(self):
        self._prev_propagate = self.logger.propagate
        self.logger.propagate = False  # keep DEBUG chatter out of the console
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._prev_level)
        self.logger.propagate = self._prev_propagate
        text = self.buffer.getvalue()
        self.path.write_text(text, encoding="utf-8")
        self.counts = {k: text.count(v) for k, v in LOG_COUNTERS.items()}
        return False


# --------------------------------------------------------------------------- #
# SRT / ground truth
# --------------------------------------------------------------------------- #
_TS_RE = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _ts_to_s(ts: str) -> float:
    m = _TS_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp {ts!r}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    cues: List[Dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n+", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # index line optional
        tl_idx = 0 if "-->" in lines[0] else 1
        if tl_idx >= len(lines) or "-->" not in lines[tl_idx]:
            continue
        a, b = lines[tl_idx].split("-->")[:2]
        body = " ".join(re.sub(r"<[^>]+>", "", ln).strip() for ln in lines[tl_idx + 1:])
        try:
            cues.append({"start": _ts_to_s(a), "end": _ts_to_s(b), "text": body})
        except ValueError:
            continue
    return cues


def find_ground_truth(media: Path, gt_dir: Optional[Path]) -> Optional[Path]:
    stems = [media.stem]
    dirs = [gt_dir] if gt_dir else []
    dirs.append(media.parent)
    for d in dirs:
        if d is None:
            continue
        for stem in stems:
            for cand in (d / f"{stem}.ja.srt", d / f"{stem}.srt"):
                if cand.exists():
                    return cand
    return None


def _fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, segs: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}\n{s['text'].strip()}\n\n")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _union(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _total(intervals: Sequence[Tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in intervals))


def _intersect(x: Sequence[Tuple[float, float]], y: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    i = j = 0
    while i < len(x) and j < len(y):
        a = max(x[i][0], y[j][0])
        b = min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def _clip(intervals: Sequence[Tuple[float, float]], lo: float, hi: float) -> float:
    return _total(_intersect(list(intervals), [(lo, hi)]))


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(
        ch for ch in s
        if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "S", "Z", "C"))
    )


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def coverage_metrics(
    out_segs: Sequence[Dict[str, Any]],
    gt_cues: Optional[Sequence[Dict[str, Any]]],
    audio_s: float,
    spans: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out_iv = _union([(s["start"], s["end"]) for s in out_segs if s["end"] > s["start"]])
    m: Dict[str, Any] = {
        "out_segments": len(out_segs),
        "out_speech_s": round(_total(out_iv), 3),
        "out_mean_seg_s": round(_total(out_iv) / len(out_segs), 3) if out_segs else 0.0,
        "out_chars": sum(len(normalise_text(s["text"])) for s in out_segs),
    }
    timeline: List[Dict[str, Any]] = []
    n_bins = int(np.ceil(audio_s / TIMELINE_BIN_S)) if audio_s > 0 else 0
    if gt_cues is None:
        for k in range(n_bins):
            lo, hi = k * TIMELINE_BIN_S, min((k + 1) * TIMELINE_BIN_S, audio_s)
            timeline.append({"bin_start_s": lo, "gt_s": "", "out_s": round(_clip(out_iv, lo, hi), 3), "covered_s": ""})
        m.update({"gt_cues": "", "gt_speech_s": "", "gt_speech_in_spans_pct": "", "gt_covered_pct": "",
                  "gt_cues_hit_pct": "",
                  "out_outside_gt_pct": "", "cer": "", "longest_uncovered_speech_bins": "",
                  "longest_uncovered_span_s": ""})
        return m, timeline

    gt_iv = _union([(c["start"], c["end"]) for c in gt_cues])
    inter = _intersect(gt_iv, out_iv)
    gt_total = _total(gt_iv)
    gt_in_spans = _total(_intersect(gt_iv, _union(list(spans)))) if spans else gt_total
    out_total = _total(out_iv)
    covered = _total(inter)
    hit = 0
    for c in gt_cues:
        ov = _clip(out_iv, c["start"], c["end"])
        dur = max(c["end"] - c["start"], 1e-6)
        if ov >= min(0.2, dur * 0.5) or ov / dur >= 0.2:
            hit += 1
    gt_text = normalise_text(" ".join(c["text"] for c in gt_cues))
    out_text = normalise_text(" ".join(s["text"] for s in out_segs))
    cer = levenshtein(gt_text, out_text) / max(len(gt_text), 1)

    longest = cur = 0                      # consecutive speech-bearing bins with zero coverage
    longest_span = 0.0                     # elapsed seconds from the first to the last bin of that run
    run_start: Optional[float] = None
    for k in range(n_bins):
        lo, hi = k * TIMELINE_BIN_S, min((k + 1) * TIMELINE_BIN_S, audio_s)
        g = _clip(gt_iv, lo, hi)
        o = _clip(out_iv, lo, hi)
        c = _clip(inter, lo, hi)
        timeline.append({"bin_start_s": lo, "gt_s": round(g, 3), "out_s": round(o, 3), "covered_s": round(c, 3)})
        if g >= 3.0 and c == 0.0:
            cur += 1
            if run_start is None:
                run_start = lo
            longest = max(longest, cur)
            longest_span = max(longest_span, hi - run_start)
        elif g >= 3.0:
            cur = 0
            run_start = None
    m.update({
        "gt_cues": len(gt_cues),
        "gt_speech_s": round(gt_total, 3),
        "gt_speech_in_spans_pct": round(100.0 * gt_in_spans / gt_total, 2) if gt_total else "",
        "gt_covered_pct": round(100.0 * covered / gt_total, 2) if gt_total else "",
        "gt_cues_hit_pct": round(100.0 * hit / len(gt_cues), 2) if gt_cues else "",
        "out_outside_gt_pct": round(100.0 * (out_total - covered) / out_total, 2) if out_total else "",
        "cer": round(cer, 4),
        "longest_uncovered_speech_bins": longest,
        "longest_uncovered_span_s": round(longest_span, 1),
    })
    return m, timeline


# --------------------------------------------------------------------------- #
# Segmentation helpers (emulations — labelled as such in the manifest)
# --------------------------------------------------------------------------- #
def silero_chunks(audio: np.ndarray, vad: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Speech chunks from the library's own bundled Silero VAD, in seconds."""
    chunks = get_speech_timestamps(audio, **vad)
    return [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in chunks]


def emulate_scenes(chunks: Sequence[Tuple[float, float]], audio_s: float) -> List[Tuple[float, float]]:
    """Pack speech chunks into scenes <= SCENE_MAX_S, splitting at gaps >= SCENE_GAP_S
    or when the cap would be exceeded (a chunk longer than the cap is split hard).
    This EMULATES WhisperJAV's silence-based scene detector; it is not the detector."""
    scenes: List[Tuple[float, float]] = []
    cur: Optional[List[float]] = None
    for a, b in chunks:
        while b - a > SCENE_MAX_S:  # hard split of an over-long chunk
            if cur is not None:
                scenes.append((cur[0], cur[1]))
                cur = None
            scenes.append((a, a + SCENE_MAX_S))
            a += SCENE_MAX_S
        if cur is None:
            cur = [a, b]
        elif a - cur[1] >= SCENE_GAP_S or b - cur[0] > SCENE_MAX_S:
            scenes.append((cur[0], cur[1]))
            cur = [a, b]
        else:
            cur[1] = b
    if cur is not None:
        scenes.append((cur[0], cur[1]))
    return [(max(0.0, s), min(audio_s, e)) for s, e in scenes if e > s]


def emulate_groups(chunks: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Test-D grouping: merge chunks separated by < GROUP_GAP_S, cap groups at GROUP_MAX_S."""
    groups: List[Tuple[float, float]] = []
    cur: Optional[List[float]] = None
    for a, b in chunks:
        if cur is None:
            cur = [a, b]
        elif a - cur[1] < GROUP_GAP_S and b - cur[0] <= GROUP_MAX_S:
            cur[1] = b
        else:
            groups.append((cur[0], cur[1]))
            cur = [a, b]
    if cur is not None:
        groups.append((cur[0], cur[1]))
    return groups


def load_scenes_json(path: Path, audio_s: float) -> List[Tuple[float, float]]:
    """[{"start": s, "end": e}, ...] or [[s, e], ...] in seconds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # WhisperJAV master metadata or a wrapper object
        data = data.get("scenes_detected") or data.get("scenes") or data.get("coarse_boundaries") or []
    key_pairs = (("start", "end"), ("start_abs", "end_abs"), ("start_time", "end_time"),
                 ("start_time_seconds", "end_time_seconds"), ("start_sec", "end_sec"))
    out: List[Tuple[float, float]] = []
    for item in data:
        if isinstance(item, dict):
            for ka, kb in key_pairs:
                if ka in item and kb in item:
                    out.append((float(item[ka]), float(item[kb])))
                    break
            else:
                raise ValueError(f"scene entry without a recognised start/end pair: {list(item)[:6]}")
        else:
            out.append((float(item[0]), float(item[1])))
    return [(max(0.0, a), min(audio_s, b)) for a, b in sorted(out) if b > a]


# --------------------------------------------------------------------------- #
# Transcribe wrappers
# --------------------------------------------------------------------------- #
def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Drop kwargs this library version's transcribe() does not accept; report them."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs), []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(k for k in kwargs if k not in params)
    return kept, dropped


def batched_forced_options() -> Dict[str, Any]:
    """What BatchedInferencePipeline.transcribe hardcodes REGARDLESS of its arguments.

    _filter_kwargs only sees the signature, so a key the batched path accepts and
    then ignores would be reported as honoured. faster-whisper 1.2.1 builds its
    TranscriptionOptions with literal constants for condition_on_previous_text,
    hallucination_silence_threshold, max_initial_timestamp and
    prompt_reset_on_temperature, and passes only temperature[:1]. Rather than
    hardcode that list, read it out of the installed source, so a different library
    version reports its own behaviour.

    Returns {option_name: forced_value}, plus _temperature_first_only when the
    batched path truncates a temperature ladder.
    """
    if BatchedInferencePipeline is None:
        return {}
    try:
        import ast
        import textwrap

        code = textwrap.dedent(inspect.getsource(BatchedInferencePipeline.transcribe))
        tree = ast.parse(code)
    except Exception:  # pragma: no cover - source unavailable
        return {}
    forced: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TranscriptionOptions":
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    forced[kw.arg] = kw.value.value
    if "temperature[:1]" in code.replace(" ", ""):
        forced["_temperature_first_only"] = True
    return forced


BATCHED_FORCED: Dict[str, Any] = batched_forced_options()


def batched_conflicts(kwargs: Dict[str, Any]) -> List[str]:
    """Recipe keys the batched path will not honour, with what it substitutes.

    Covers both keys the recipe states explicitly and keys it leaves to the library:
    the batched path overrides some of those too, so 'not passed' does not mean
    'library default' there.
    """
    if not BATCHED_FORCED:
        return []
    try:
        plain_defaults = {k: v.default for k, v in
                          inspect.signature(WhisperModel.transcribe).parameters.items()}
    except (TypeError, ValueError):  # pragma: no cover
        plain_defaults = {}
    out: List[str] = []
    for key, forced in sorted(BATCHED_FORCED.items()):
        if key == "_temperature_first_only":
            temp = kwargs.get("temperature")
            if isinstance(temp, (list, tuple)) and len(temp) > 1:
                out.append(f"temperature: asked {tuple(temp)} -> only {temp[0]!r} is used")
            continue
        if key in kwargs:
            if kwargs[key] != forced:
                out.append(f"{key}: asked {kwargs[key]!r} -> forced {forced!r}")
        elif key in plain_defaults and plain_defaults[key] != forced:
            out.append(f"{key}: left to the library ({plain_defaults[key]!r}) -> forced {forced!r}")
    return out


def _seg_to_dict(seg: Any, offset: float) -> Dict[str, Any]:
    words = getattr(seg, "words", None)
    return {
        "start": round(float(seg.start) + offset, 3),
        "end": round(float(seg.end) + offset, 3),
        "text": seg.text,
        "avg_logprob": round(float(getattr(seg, "avg_logprob", float("nan"))), 4),
        "no_speech_prob": round(float(getattr(seg, "no_speech_prob", float("nan"))), 4),
        "compression_ratio": round(float(getattr(seg, "compression_ratio", float("nan"))), 4),
        "temperature": getattr(seg, "temperature", None),
        "seek": getattr(seg, "seek", None),
        "n_words": len(words) if words else 0,
        "words": [
            {"start": round(float(w.start) + offset, 3), "end": round(float(w.end) + offset, 3),
             "word": w.word, "probability": round(float(w.probability), 4)}
            for w in (words or [])
        ],
    }


@dataclasses.dataclass
class CallRecord:
    index: int
    span_start_s: float
    span_end_s: float
    wall_s: float
    segments: int
    duration_after_vad_s: Optional[float]
    error: Optional[str] = None


def run_calls(
    transcribe_fn: Any,
    audio: np.ndarray,
    spans: Sequence[Tuple[float, float]],
    kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[CallRecord]]:
    """One transcribe() per span; consume the generator inside the timed region."""
    segs: List[Dict[str, Any]] = []
    calls: List[CallRecord] = []
    for i, (a, b) in enumerate(spans):
        piece = audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
        t0 = time.perf_counter()
        err = None
        n_before = len(segs)
        dav: Optional[float] = None
        try:
            # copy the VAD dict: upstream 1.2.1's batched path mutates it in place
            kw = dict(kwargs)
            if isinstance(kw.get("vad_parameters"), dict):
                kw["vad_parameters"] = dict(kw["vad_parameters"])
            gen, info = transcribe_fn(piece, **kw)
            dav = float(getattr(info, "duration_after_vad", float("nan")))
            for seg in gen:
                segs.append(_seg_to_dict(seg, a))
        except Exception as e:  # record, do not abort the arm
            err = f"{type(e).__name__}: {e}"
            log.error("call %d [%.2f-%.2f] failed: %s", i, a, b, err)
        calls.append(CallRecord(i, a, b, time.perf_counter() - t0, len(segs) - n_before, dav, err))
    return segs, calls


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
ARM_ORDER = ["wj_native_scene", "wj_native_file", "wj_external_group",
             "lib_default_file", "v07b_file", "wj_batched_file"]


WJ_ARMS = ("wj_native_scene", "wj_native_file", "wj_external_group", "wj_batched_file")


def build_arm(name: str, recipe: str, sensitivity: str, batch_size: int) -> Dict[str, Any]:
    """One arm = one call pattern x one parameter recipe.

    --recipe selects the parameter set for the four wj_* arms. lib_default_file and
    v07b_file ARE their own recipes and ignore it (recorded as recipe "fixed").

    The Silero parameters that SPLIT the audio for the per-scene and per-group arms
    always come from WhisperJAV's preset for --sensitivity, whatever the recipe:
    those arms exist to emulate WhisperJAV's segmentation, so holding the split
    constant keeps the spans comparable and leaves the decoding parameters as the
    only thing --recipe moves.
    """
    spec = RECIPES[recipe]
    _, split_vad = wj_recipe(sensitivity)          # segmentation, held constant across recipes
    tag = dict(recipe=recipe, recipe_version=spec["version"])

    def wj(**extra: Any) -> Dict[str, Any]:
        d, native_vad = spec["build"](sensitivity)
        decode: Dict[str, Any] = {**d, **extra}
        if decode.get("vad_filter") and native_vad:
            decode["vad_parameters"] = native_vad   # empty dict => library default VadOptions
        return dict(name=name, decode=decode, **tag)

    if name == "wj_native_scene":
        return dict(**wj(vad_filter=True),
                    pattern="per_scene", vad_for_split=split_vad, batched=False)
    if name == "wj_native_file":
        return dict(**wj(vad_filter=True),
                    pattern="whole_file", batched=False)
    if name == "wj_external_group":
        return dict(**wj(vad_filter=False),
                    pattern="per_group", vad_for_split=split_vad, batched=False)
    if name == "wj_batched_file":
        return dict(**wj(vad_filter=True, batch_size=batch_size),
                    pattern="whole_file", batched=True)
    if name == "lib_default_file":
        d, _v = lib_default_recipe()
        return dict(name=name, decode={**d, "vad_filter": True}, pattern="whole_file",
                    batched=False, recipe="fixed", recipe_version="lib_default")
    if name == "v07b_file":
        d, v = v07b_recipe()
        return dict(name=name, decode={**d, "vad_filter": True, "vad_parameters": v},
                    pattern="whole_file", batched=False, recipe="fixed", recipe_version="v07b")
    raise ValueError(f"unknown arm {name!r}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_model(model_arg: str, device: str, compute_type: str) -> Tuple[Any, str, bool]:
    """Load the model; if the library rejects a bare alias (faster-whisper2 dropped
    large-v1/large-v2/distil-large-v2), retry with the Systran repo id and say so."""
    try:
        return WhisperModel(model_arg, device=device, compute_type=compute_type), model_arg, False
    except ValueError as e:
        if "/" in model_arg or "Invalid model size" not in str(e):
            raise
        alt = f"Systran/faster-whisper-{model_arg}"
        log.warning("library rejected alias %r (%s); retrying with %r", model_arg, e, alt)
        return WhisperModel(alt, device=device, compute_type=compute_type), alt, True


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+", required=True, help="media files or globs")
    p.add_argument("--out", type=Path, required=True, help="results root; a run folder is created inside")
    p.add_argument("--label", default=None, help="environment label, e.g. WJ, FW2, V07B (default: fw-<version>)")
    p.add_argument("--model", default="Systran/faster-whisper-large-v2",
                   help="model name, HF repo id, or local path (default: the repo id WhisperJAV's "
                        "large-v2 alias resolves to — identical files in every environment)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default=None,
                   help="model compute type. Default: whatever the chosen --recipe asks for "
                        "(wj_v192 -> float16, as WhisperJAV balanced resolves on CUDA; "
                        "fwdefaults -> auto). Passing this flag overrides the recipe. NOTE: "
                        "'auto' lets CTranslate2 pick the fastest supported type, which on a "
                        "modern GPU is int8_float16, NOT float16; the manifest records what it "
                        "resolved to (model.ct2_compute_type_effective).")
    p.add_argument("--recipe", default=DEFAULT_RECIPE, choices=sorted(RECIPES),
                   help="parameter recipe for the wj_* arms: "
                        + "; ".join(f"{k} = {v['description']}" for k, v in sorted(RECIPES.items()))
                        + f". Default {DEFAULT_RECIPE}. lib_default_file and v07b_file ignore it.")
    p.add_argument("--sensitivity", default="balanced", choices=["balanced", "aggressive"])
    p.add_argument("--arms", default="all", help="comma list of arms or 'all'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gt-dir", type=Path, default=None, help="folder holding <stem>.srt / <stem>.ja.srt")
    p.add_argument("--scenes-json", type=Path, default=None,
                   help="real scene boundaries for wj_native_scene (JSON list of {start,end} seconds); "
                        "applies to a single --files entry")
    p.add_argument("--max-seconds", type=float, default=None, help="truncate audio (smoke tests)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--shuffle-arms", action="store_true",
                   help="randomise arm order within each repetition (guards against first-arm effects)")
    p.add_argument("--repeat", type=int, default=1,
                   help="run every arm N times (rows carry a 'rep' column); GPU fp16 beam search is "
                        "not bit-reproducible, so differences smaller than the spread between reps "
                        "are noise, not findings")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files: List[Path] = []
    for pat in args.files:
        matches = sorted(glob.glob(pat))
        files.extend(Path(m) for m in matches) if matches else files.append(Path(pat))
    files = [f for f in files if f.exists()]
    if not files:
        log.error("no input files found")
        return 2
    if args.scenes_json and len(files) != 1:
        log.error("--scenes-json applies to exactly one input file")
        return 2

    arms = ARM_ORDER if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_ORDER:
            log.error("unknown arm %r; known: %s", a, ", ".join(ARM_ORDER))
            return 2

    recipe_spec = RECIPES[args.recipe]
    compute_type = args.compute_type or recipe_spec["compute_type"]
    compute_type_source = "cli" if args.compute_type else f"recipe:{args.recipe}"
    log.info("recipe %s (%s) | compute_type %s (from %s)",
             args.recipe, recipe_spec["version"], compute_type, compute_type_source)
    if compute_type in ("auto", "default"):
        log.warning("compute_type %r lets CTranslate2 choose the model's numeric precision "
                    "('auto' resolves to int8_float16 on a modern GPU, not float16). It applies "
                    "to the WHOLE run - every arm, including lib_default_file and v07b_file, "
                    "which otherwise ignore --recipe. Speed and VRAM from this run cannot be "
                    "compared with a float16 run, and a coverage or CER difference carries the "
                    "quantisation with it. Pass --compute-type float16 to A/B the decoding "
                    "parameters alone. The manifest records what it resolved to.", compute_type)
    if not recipe_spec["sensitivity_aware"] and any(a in WJ_ARMS for a in arms):
        splits = [a for a in arms if a in ("wj_native_scene", "wj_external_group")]
        if splits:
            log.info("recipe %r has no per-sensitivity variant; --sensitivity %s changes nothing "
                     "but the Silero parameters that split the audio for %s",
                     args.recipe, args.sensitivity, ", ".join(splits))
        else:
            log.warning("recipe %r has no per-sensitivity variant and none of the selected arms "
                        "splits the audio, so --sensitivity %s has NO effect on this run; the "
                        "summary still records it", args.recipe, args.sensitivity)

    env = environment_fingerprint()
    label = args.label or f"fw-{env['faster_whisper_version']}"
    run_dir = args.out / f"{label}_{args.recipe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run dir: %s", run_dir)

    vram = VramSampler(gpu_index=args.gpu_index)
    vram_idle = vram.read_now()

    t0 = time.perf_counter()
    model, model_used, alias_fallback = load_model(args.model, args.device, compute_type)
    load_s = time.perf_counter() - t0
    vram_after_load = vram.read_now()
    batched = BatchedInferencePipeline(model=model) if BatchedInferencePipeline is not None else None

    arm_specs = {a: build_arm(a, args.recipe, args.sensitivity, args.batch_size) for a in arms}
    manifest: Dict[str, Any] = {
        "label": label,
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "environment": env,
        "model": model_fingerprint(model, model_used),
        "model_alias_fallback": alias_fallback,
        "model_load_s": round(load_s, 2),
        "vram_device_idle_before_load_mib": vram_idle,
        "vram_device_after_load_mib": vram_after_load,
        "vram_process_after_load_mib": vram.read_process_now(),
        "device": args.device,
        "compute_type_requested": compute_type,
        "compute_type_source": compute_type_source,
        "recipe": args.recipe,
        "recipe_version": recipe_spec["version"],
        "recipe_description": recipe_spec["description"],
        "recipe_applies_to": list(WJ_ARMS),
        "recipes_available": {k: {kk: vv for kk, vv in v.items() if kk != "build"}
                              for k, v in sorted(RECIPES.items())},
        "sensitivity": args.sensitivity,
        "arms": arm_specs,
        "constants": dict(SCENE_MAX_S=SCENE_MAX_S, SCENE_GAP_S=SCENE_GAP_S,
                          GROUP_MAX_S=GROUP_MAX_S, GROUP_GAP_S=GROUP_GAP_S,
                          TIMELINE_BIN_S=TIMELINE_BIN_S),
        "notes": [
            "wj_native_scene scenes are EMULATED from Silero speech chunks unless --scenes-json is given.",
            "wj_external_group groups are EMULATED with the library's Silero, not WhisperJAV's external segmenter; "
            "it reproduces the per-group CALL PATTERN, not the segmenter's exact boundaries.",
            "Recipe wj_v192 reflects WhisperJAV v1.9.2 --dump-params output for --mode balanced "
            "(2026-09-08); recipe fwdefaults is faster-whisper's own defaults with temperature "
            "pinned to 0.0 and word_timestamps=True (verified against the 1.2.1 signature 2026-09-09).",
            "--recipe changes the DECODING parameters of the wj_* arms only. The Silero parameters "
            "used to split the audio for the per-scene / per-group arms are WhisperJAV's preset for "
            "--sensitivity in every recipe, so those arms transcribe comparable spans.",
            "compute_type 'auto' is not float16: CTranslate2 picks the fastest supported type "
            "(int8_float16 on this class of GPU). Read model.ct2_compute_type_effective before "
            "comparing a run against one loaded float16.",
        ],
        "files": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_warmup:
        log.info("warm-up call")
        warm = decode_audio(str(files[0]), sampling_rate=SAMPLE_RATE)[: SAMPLE_RATE * 8]
        try:
            list(model.transcribe(warm, language="ja", beam_size=1)[0])
        except Exception as e:  # pragma: no cover
            log.warning("warm-up failed: %s", e)

    summary_path = run_dir / "summary.csv"
    summary_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    for media in files:
        stem = media.stem
        fdir = run_dir / stem
        fdir.mkdir(exist_ok=True)
        log.info("=== %s", media.name)
        audio = decode_audio(str(media), sampling_rate=SAMPLE_RATE)
        if args.max_seconds:
            audio = audio[: int(args.max_seconds * SAMPLE_RATE)]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        audio_s = len(audio) / SAMPLE_RATE
        gt_path = find_ground_truth(media, args.gt_dir)
        gt_cues = parse_srt(gt_path) if gt_path else None
        if gt_cues is not None and args.max_seconds:
            gt_cues = [c for c in gt_cues if c["start"] < audio_s]
            for c in gt_cues:
                c["end"] = min(c["end"], audio_s)
        manifest["files"].append({"path": str(media), "audio_s": round(audio_s, 3),
                                  "ground_truth": str(gt_path) if gt_path else None,
                                  "gt_cues": len(gt_cues) if gt_cues else None})

        _order: List[Tuple[int, str]] = []
        for r in range(max(1, args.repeat)):
            _arms = list(arms)
            if args.shuffle_arms:
                random.shuffle(_arms)
            _order.extend((r, a) for a in _arms)
        for rep, arm_name in _order:
            spec = arm_specs[arm_name]
            tag = arm_name if args.repeat <= 1 else f"{arm_name}.rep{rep}"
            row: Dict[str, Any] = {
                "label": label, "file": media.name, "arm": arm_name, "rep": rep, "pattern": spec["pattern"],
                "audio_s": round(audio_s, 3), "sensitivity": args.sensitivity,
                "fw_version": env["faster_whisper_version"], "ct2_version": env["ctranslate2_version"],
                "model": model_used, "compute_type": compute_type,
                "compute_type_effective": manifest["model"].get("ct2_compute_type_effective"),
                "recipe": spec.get("recipe", "fixed"),
                "recipe_version": spec.get("recipe_version", ""),
            }
            if spec["batched"] and batched is None:
                row.update(status="skipped", error="BatchedInferencePipeline not in this library version")
                summary_rows.append(row)
                log.warning("[%s] skipped: no BatchedInferencePipeline", arm_name)
                continue

            fn = batched.transcribe if spec["batched"] else model.transcribe
            kwargs, dropped = _filter_kwargs(fn, spec["decode"])
            row["kwargs_dropped_by_library"] = ";".join(dropped)
            conflicts = batched_conflicts(kwargs) if spec["batched"] else []
            row["recipe_keys_not_honoured"] = ";".join(conflicts)
            if conflicts:
                log.warning("[%s] the batched path overrides the recipe: %s",
                            arm_name, "; ".join(conflicts))

            # spans to call
            t_split = time.perf_counter()
            split_info: Dict[str, Any] = {}
            try:
                if spec["pattern"] == "whole_file":
                    spans = [(0.0, audio_s)]
                elif spec["pattern"] == "per_scene":
                    if args.scenes_json:
                        spans = load_scenes_json(args.scenes_json, audio_s)
                        split_info["scene_source"] = "scenes_json"
                    else:
                        chunks = silero_chunks(audio, spec["vad_for_split"])
                        spans = emulate_scenes(chunks, audio_s)
                        split_info["scene_source"] = "emulated"
                        split_info["silero_chunks"] = len(chunks)
                elif spec["pattern"] == "per_group":
                    chunks = silero_chunks(audio, spec["vad_for_split"])
                    spans = emulate_groups(chunks)
                    split_info["silero_chunks"] = len(chunks)
                else:  # pragma: no cover
                    raise ValueError(spec["pattern"])
            except Exception as e:
                row.update(status="failed", error=f"split: {type(e).__name__}: {e}")
                summary_rows.append(row)
                log.error("[%s] split failed: %s", arm_name, e)
                continue
            split_s = time.perf_counter() - t_split
            if not spans:
                spans = []

            log.info("[%s] %d call(s), span total %.1fs", arm_name, len(spans),
                     sum(b - a for a, b in spans))

            vram.start()
            t_arm = time.perf_counter()
            with FwLogCapture(fdir / f"{tag}.log") as cap:
                segs, calls = run_calls(fn, audio, spans, kwargs)
            wall = time.perf_counter() - t_arm
            vram_stats = vram.stop()

            segs.sort(key=lambda s: (s["start"], s["end"]))
            with open(fdir / f"{tag}.segments.jsonl", "w", encoding="utf-8") as f:
                for s in segs:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(fdir / f"{tag}.calls.jsonl", "w", encoding="utf-8") as f:
                for c in calls:
                    f.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")
            write_srt(fdir / f"{tag}.srt", segs)

            metrics, timeline = coverage_metrics(segs, gt_cues, audio_s, spans)
            with open(fdir / f"{tag}.timeline.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["bin_start_s", "gt_s", "out_s", "covered_s"])
                w.writeheader()
                w.writerows(timeline)

            n_err = sum(1 for c in calls if c.error)
            row.update(
                status="ok" if n_err == 0 else f"partial({n_err} call errors)",
                error=";".join(c.error for c in calls if c.error)[:500],
                n_calls=len(calls),
                span_total_s=round(sum(b - a for a, b in spans), 3),
                split_s=round(split_s, 3),
                decode_wall_s=round(wall, 3),
                wall_s=round(wall + split_s, 3),
                x_realtime=round(audio_s / (wall + split_s), 3) if (wall + split_s) > 0 else "",
                **vram_stats,
                duration_after_vad_s=round(sum(c.duration_after_vad_s or 0.0 for c in calls), 3),
                words_with_timestamps=sum(s["n_words"] for s in segs),
                words_nonempty=sum(1 for s in segs for w in s["words"] if w["word"].strip()),
                words_empty=sum(1 for s in segs for w in s["words"] if not w["word"].strip()),
                mean_no_speech_prob=round(float(np.mean([s["no_speech_prob"] for s in segs])), 4) if segs else "",
                mean_avg_logprob=round(float(np.mean([s["avg_logprob"] for s in segs])), 4) if segs else "",
                max_compression_ratio=round(max(s["compression_ratio"] for s in segs), 3) if segs else "",
                **split_info,
                **metrics,
                **cap.counts,
            )
            summary_rows.append(row)
            _tw = wall + split_s
            log.info("[%s] wall %.1fs incl. split (%.2fx RT) | segs %d | covered %s%% | CER %s | process VRAM peak %s MiB",
                     arm_name, _tw, audio_s / _tw if _tw else 0, len(segs),
                     metrics.get("gt_covered_pct", ""), metrics.get("cer", ""),
                     vram_stats.get("vram_process_peak_mib") or "?")

            for r in summary_rows:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(summary_rows)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("done. summary: %s", summary_path)
    _KEEPALIVE.extend([model, batched])   # never let the model's destructor run in this process
    return 0


if __name__ == "__main__":
    _rc = main()
    # Every result file is closed by now. With CTranslate2 4.6.2, destroying the CUDA Whisper
    # model (explicitly via del, when main()'s locals are released, or at interpreter teardown)
    # kills the process with STATUS_STACK_BUFFER_OVERRUN (0xC0000409 / 3221226505; Git Bash
    # shows 127). This is WhisperJAV issue #125 — the same crash main.py's "nuclear exit" and
    # balanced_pipeline.py's _IMMORTAL_ASR_REFERENCE exist for. Reproduced 2026-09-08 in the
    # WhisperJAV conda env, a clean faster-whisper 1.2.1 venv and the faster-whisper2 venv
    # alike; CTranslate2 4.8.1 and 4.8.2 in the same venv survive. main() parks the model in
    # _KEEPALIVE so the destructor cannot run before we terminate.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        # os._exit still runs DLL detach routines on Windows (ExitProcess), and the
        # CTranslate2 4.6.2 static destructors crash there just the same (observed:
        # exit 127 after "done" whenever the warm-up call had run). TerminateProcess
        # skips DLL detach entirely; every file is already flushed and closed.
        import ctypes
        import ctypes.wintypes as _wt

        _k32 = ctypes.windll.kernel32
        _k32.GetCurrentProcess.restype = _wt.HANDLE
        _k32.TerminateProcess.argtypes = (_wt.HANDLE, ctypes.c_uint)
        _k32.TerminateProcess.restype = _wt.BOOL
        _ok = _k32.TerminateProcess(_k32.GetCurrentProcess(), int(_rc))
        log.error("TerminateProcess returned %s (GetLastError=%s); falling back to os._exit", _ok, _k32.GetLastError())
    os._exit(_rc)
