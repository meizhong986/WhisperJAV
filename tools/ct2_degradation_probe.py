#!/usr/bin/env python3
"""Probe for CTranslate2 instance degradation — the failure tracked in issue #394.

WHAT THIS ANSWERS
-----------------
Several users have reported that the recognizer stops producing output part-way
through a long file and never recovers, while the run still reports success.
@AlanZ-Git found the sharpest clue: feeding a known-good short clip to the
*already-loaded* model returns nothing, while a fresh process transcribes the
same clip correctly. The loaded instance, not the audio, is what has gone wrong.

The open question is whether that belongs to WhisperJAV or to faster-whisper /
CTranslate2 underneath it. This tool answers that by running the same experiment
two ways:

    --engine bare        faster-whisper directly, no WhisperJAV involved
    --engine whisperjav  through WhisperJAV's ASR module

Both arms run WhisperJAV's *shipped* decode configuration by default, resolved
live from the installed package (--profile shipped). That matters: with the
tool's own defaults instead (--profile raw), the bare arm ran with VAD off, no
repetition penalty and no word timestamps -- a configuration WhisperJAV never
uses -- so a clean result said nothing about the configuration people actually
run. Results collected before 2026-09-02 were all --profile raw.

Same audio, same model, same decode parameters, same probe. **The difference
between the two runs is the answer**, and running both from one harness removes
any argument about the configurations differing.

    Degrades on `bare`                  -> upstream; belongs to CTranslate2
    Clean on `bare`, degrades on ours   -> ours; bisect our parameters
    Clean on both                       -> not reproducible on demand

HOW IT WORKS
------------
One model instance is loaded and kept alive for the whole run. A short reference
clip *you supply* is transcribed first to establish a baseline. Then, repeatedly:
a chunk of your problem audio is transcribed, and the reference clip is
transcribed again. When the reference stops matching its own baseline, the
instance has degraded, and the tool reports the iteration at which that happened.

YOU SUPPLY THE REFERENCE CLIP
-----------------------------
Use 5-15 seconds of clear speech that you have already confirmed transcribes
correctly. It should come from the same kind of material you normally process.
Nothing is uploaded anywhere; everything stays on your machine.

USAGE
-----
    python ct2_degradation_probe.py --audio PROBLEM.wav --reference GOOD.wav

Standalone-installer users can run it with the bundled Python::

    %LOCALAPPDATA%\\WhisperJAV\\python.exe ct2_degradation_probe.py --audio ... --reference ...

Output is a JSONL file, one record per iteration, plus a verdict on stdout.
Please attach the JSONL to issue #394.

This tool is read-only with respect to your installation. It imports nothing
from WhisperJAV unless --engine whisperjav is requested.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROBE_VERSION = "2026-09-02"

DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_ITERATIONS = 200

# A degraded CTranslate2 instance has been observed emitting runs of punctuation
# ("!!!!!") instead of text. Treat output that carries almost no word characters
# as degraded even when it is not empty.
_WORDISH = re.compile(r"[^\W_]", re.UNICODE)


def _log(msg: str) -> None:
    """Print without ever failing on the console's code page.

    The transcripts this tool prints are Japanese. A Windows console on a legacy
    code page raises UnicodeEncodeError on the first line of output, which killed
    the run before it started. Reconfigure once if we can, and degrade to an
    ASCII-safe rendering if we cannot.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"),
              flush=True)


def _make_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - older Python, or a wrapped stream
            pass


def _memory_snapshot() -> dict:
    snap = {"cuda_allocated_mb": None, "cuda_reserved_mb": None, "rss_mb": None}
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1048576, 1)
            snap["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576, 1)
    except Exception:
        pass
    try:
        import psutil

        snap["rss_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 1)
    except Exception:
        pass
    return snap


def _wordiness(text: str) -> float:
    """Fraction of characters that are letters or digits."""
    if not text:
        return 0.0
    return len(_WORDISH.findall(text)) / len(text)


def _similar(a: str, b: str) -> float:
    """Rough similarity, 0..1. Deliberately crude: we are detecting collapse,
    not measuring quality."""
    import difflib

    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _read_audio(path: Path):
    """Return (samples, sample_rate) as float32 mono at 16 kHz."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        _log(f"ERROR: needs numpy and soundfile ({exc}). "
             f"Install with: pip install numpy soundfile")
        raise SystemExit(2) from exc

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        try:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            _log(f"WARNING: {path.name} is {sr} Hz and librosa is unavailable; "
                 f"passing through unresampled.")
    return data, sr


def resolve_shipped_params(args) -> dict:
    """Return WhisperJAV's resolved runtime configuration for --mode balanced.

    Imports only the config layer, which pulls in no GPU stack. Returns a dict
    with 'model', 'params' and 'task', the exact structure the pipelines hand to
    FasterWhisperProASR. Raises SystemExit with an actionable message rather
    than a traceback if the installed package cannot provide it.
    """
    try:
        from whisperjav.config.legacy import resolve_legacy_pipeline
    except ImportError as exc:
        _log(f"ERROR: --profile shipped needs WhisperJAV importable ({exc}).\n"
             f"Run this with the Python that has WhisperJAV installed, or pass\n"
             f"--profile raw to use this tool's own decode defaults.")
        raise SystemExit(2) from exc

    try:
        cfg = resolve_legacy_pipeline(
            "balanced", args.sensitivity, task="transcribe",
            device=args.device, compute_type=args.compute_type,
        )
    except Exception as exc:  # noqa: BLE001 - any resolver failure is fatal here
        _log(f"ERROR: could not resolve the balanced configuration ({exc}).\n"
             f"Pass --profile raw to fall back to this tool's own defaults.")
        raise SystemExit(2) from exc

    # Two steps in main.py stand between the resolver and what actually runs,
    # and skipping either gives a configuration WhisperJAV never uses.
    #
    #   1. main.py:2014-2023 -- balanced with no explicit --speech-segmenter
    #      defaults to "faster-whisper", i.e. the native internal VAD.
    #   2. main.py:2083 -- apply_balanced_vad_defaults() then overlays the VAD
    #      preset for THAT backend. Without step 1 the overlay does not fire and
    #      the threshold stays on the external-Silero scale (0.28 / 5.0 s)
    #      instead of the native faster-whisper one (0.40 / 15.0 s).
    #
    # This was caught by running the tool, not by reading it.
    params = cfg.setdefault("params", {})
    params.setdefault("speech_segmenter", {})
    if not params["speech_segmenter"].get("backend"):
        params["speech_segmenter"]["backend"] = args.speech_segmenter
    try:
        from whisperjav.config.legacy import apply_balanced_vad_defaults

        apply_balanced_vad_defaults(cfg, sensitivity=args.sensitivity, is_balanced=True)
    except ImportError:
        _log("WARNING: this WhisperJAV predates apply_balanced_vad_defaults; the "
             "VAD parameters below are the resolver's, not the shipped overlay's.")

    model_cfg = dict(cfg["model"])
    if args.model:
        model_cfg["model_name"] = args.model
    return {"model": model_cfg, "params": cfg["params"], "task": cfg.get("task", "transcribe")}


def _accepted_kwargs(func, candidate: dict) -> dict:
    """Drop keys the callee does not accept.

    WhisperJAV's params carry entries that belong to WhisperJAV and not to
    faster-whisper (logprob_margin, drop_nonverbal_vocals, ...), and
    faster-whisper's own signature moves between versions. Filtering keeps this
    tool runnable against whatever the user has installed.
    """
    import inspect

    try:
        allowed = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k in allowed}


def _vad_parameters(vad: dict) -> dict:
    """Filter WhisperJAV's vad params down to what this faster-whisper accepts."""
    try:
        import dataclasses

        from faster_whisper.vad import VadOptions

        allowed = {f.name for f in dataclasses.fields(VadOptions)}
    except Exception:  # noqa: BLE001 - fall back to the long-stable subset
        allowed = {"threshold", "min_speech_duration_ms", "max_speech_duration_s",
                   "min_silence_duration_ms", "speech_pad_ms"}
    return {k: v for k, v in vad.items() if k in allowed and v is not None}


class BareEngine:
    """faster-whisper directly. No WhisperJAV code is involved."""

    name = "bare"

    def __init__(self, args):
        from faster_whisper import WhisperModel

        if args.profile == "shipped":
            resolved = resolve_shipped_params(args)
            model_name = resolved["model"].get("model_name", args.model)
            device = resolved["model"].get("device", args.device)
            compute_type = resolved["model"].get("compute_type", args.compute_type)
            candidate = dict(resolved["params"].get("decoder", {}))
            candidate.update(resolved["params"].get("provider", {}))
            candidate["vad_filter"] = True
            candidate["vad_parameters"] = _vad_parameters(
                resolved["params"].get("vad", {}) or {})
            candidate.pop("task", None)
            opts = _accepted_kwargs(WhisperModel.transcribe, candidate)
        else:
            model_name, device, compute_type = args.model, args.device, args.compute_type
            opts = dict(
                beam_size=args.beam_size,
                best_of=args.best_of,
                temperature=[float(t) for t in args.temperature.split(",")],
                language=args.language or None,
                vad_filter=args.vad_filter,
            )

        _log(f"Loading model '{model_name}' on {device} ({compute_type})...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.opts = opts
        self.config = {"profile": args.profile, "model": model_name,
                       "device": device, "compute_type": compute_type,
                       "decode_options": {k: str(v) for k, v in sorted(opts.items())}}
        _log(f"Profile: {args.profile}. Decode options: {self.opts}")

    def transcribe(self, samples, sample_rate):
        segments, _info = self.model.transcribe(samples, **self.opts)
        out = []
        for seg in segments:  # generator: consuming it is what does the work
            out.append({
                "text": seg.text,
                "temperature": getattr(seg, "temperature", None),
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
            })
        return out


class WhisperJavEngine:
    """WhisperJAV's ASR module, for the comparison arm.

    Imported lazily and only in this mode, so the tool stays decoupled from any
    particular WhisperJAV version when run with --engine bare.
    """

    name = "whisperjav"

    def __init__(self, args):
        try:
            from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
        except ImportError as exc:
            _log(f"ERROR: --engine whisperjav needs WhisperJAV importable ({exc}).\n"
                 f"Run this with the Python that has WhisperJAV installed, or use "
                 f"--engine bare.")
            raise SystemExit(2) from exc

        # The configuration is built by WhisperJAV's own resolver rather than
        # assembled here, so this arm cannot drift away from what the pipeline
        # runs. An earlier version of this file passed model_name/device/
        # compute_type directly; that signature has never existed and the arm
        # could not start. See issue #394.
        resolved = resolve_shipped_params(args)
        model_cfg, params = resolved["model"], resolved["params"]

        _log(f"Loading WhisperJAV ASR: model '{model_cfg.get('model_name')}', "
             f"sensitivity '{args.sensitivity}', "
             f"segmenter '{params.get('speech_segmenter', {}).get('backend')}'...")
        self.asr = FasterWhisperProASR(
            model_config=model_cfg, params=params, task=resolved["task"],
        )
        self.config = {
            "profile": "shipped",
            "model": model_cfg.get("model_name"),
            "device": model_cfg.get("device"),
            "compute_type": model_cfg.get("compute_type"),
            "sensitivity": args.sensitivity,
            "speech_segmenter": params.get("speech_segmenter", {}).get("backend"),
            "decode_options": {
                k: str(v) for k, v in sorted(
                    {**params.get("decoder", {}), **params.get("provider", {})}.items())
            },
        }

    def transcribe(self, samples, sample_rate):
        # Write a temp wav; the module's public entry point takes a path.
        import tempfile

        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "probe.wav"
            sf.write(str(wav), samples, sample_rate)
            result = self.asr.transcribe(wav)
        segs = result.get("segments", []) if isinstance(result, dict) else []
        return [{"text": s.get("text", ""),
                 "temperature": s.get("temperature"),
                 "avg_logprob": s.get("avg_logprob"),
                 "compression_ratio": s.get("compression_ratio"),
                 "no_speech_prob": s.get("no_speech_prob")} for s in segs]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe for CTranslate2 instance degradation (issue #394).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audio", required=True, type=Path,
                   help="The problem audio: a file that has shown the failure.")
    p.add_argument("--reference", required=True, type=Path,
                   help="A short clip (5-15s) you have verified transcribes correctly.")
    p.add_argument("--engine", choices=["bare", "whisperjav"], default="bare",
                   help="bare = faster-whisper directly; whisperjav = through our ASR "
                        "module. Run BOTH: the difference between them is the answer.")
    p.add_argument("--profile", choices=["shipped", "raw"], default="shipped",
                   help="shipped (default) = use WhisperJAV's resolved balanced decode "
                        "parameters in both arms, so the comparison is about the engine "
                        "and not about settings. raw = this tool's own defaults "
                        "(what runs collected before 2026-09-02 used).")
    p.add_argument("--sensitivity", default="balanced",
                   choices=["conservative", "balanced", "aggressive"],
                   help="Sensitivity to resolve for --profile shipped (default: balanced).")
    p.add_argument("--speech-segmenter", default="faster-whisper",
                   help="Segmenter to resolve for --profile shipped. Default "
                        "'faster-whisper' is the v1.9.0 balanced default (native "
                        "internal VAD). Pass silero-v3.1 to probe the external "
                        "per-group path instead.")
    p.add_argument("--model", default="large-v2",
                   help="Overrides the resolved model name under --profile shipped.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default="ja")
    p.add_argument("--beam-size", type=int, default=5,
                   help="--profile raw only.")
    p.add_argument("--best-of", type=int, default=1,
                   help="--profile raw only.")
    p.add_argument("--temperature", default="0.0,0.2,0.4,0.6,0.8,1.0",
                   help="Comma-separated fallback ladder.")
    p.add_argument("--vad-filter", action="store_true",
                   help="--profile raw only. Under 'shipped' the VAD setting comes "
                        "from the resolved configuration.")
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                   help="Maximum chunks to process before stopping.")
    p.add_argument("--similarity-threshold", type=float, default=0.5,
                   help="Reference output below this similarity to its own "
                        "baseline counts as degraded.")
    p.add_argument("--out", type=Path, default=Path("ct2_probe.jsonl"))
    return p


def main(argv=None) -> int:
    _make_stdout_utf8()
    args = build_parser().parse_args(argv)

    for path in (args.audio, args.reference):
        if not path.exists():
            _log(f"ERROR: file not found: {path}")
            return 2

    engine = (BareEngine if args.engine == "bare" else WhisperJavEngine)(args)

    ref_samples, ref_sr = _read_audio(args.reference)
    _log(f"Reference clip: {args.reference.name} "
         f"({len(ref_samples) / max(ref_sr, 1):.1f}s)")

    baseline_segs = engine.transcribe(ref_samples, ref_sr)
    baseline_text = " ".join(s["text"] for s in baseline_segs).strip()
    if not baseline_text or _wordiness(baseline_text) < 0.3:
        _log("ERROR: the reference clip did not transcribe cleanly even on a fresh\n"
             "instance, so it cannot serve as a baseline. Please choose a clip with\n"
             "clear speech that you have confirmed works.")
        return 2
    _log(f"Baseline reference output ({len(baseline_text)} chars): "
         f"{baseline_text[:120]}")

    audio, sr = _read_audio(args.audio)
    chunk = int(args.chunk_seconds * sr)
    total_chunks = max(1, len(audio) // chunk)
    n = min(args.iterations, total_chunks)
    _log(f"Problem audio: {args.audio.name} ({len(audio) / sr:.1f}s) "
         f"-> {n} chunk(s) of {args.chunk_seconds:.0f}s\n")

    # First record describes the run, so no JSONL can be read out of context.
    config_record = {"record": "config", "engine": engine.name,
                     "probe_version": PROBE_VERSION}
    config_record.update(getattr(engine, "config", {}))
    config_record["reference_clip_s"] = round(len(ref_samples) / max(ref_sr, 1), 2)
    config_record["audio_s"] = round(len(audio) / max(sr, 1), 2)
    config_record["chunk_seconds"] = args.chunk_seconds

    records, degraded_at = [config_record], None
    for i in range(n):
        piece = audio[i * chunk:(i + 1) * chunk]
        if len(piece) == 0:
            break

        t0 = time.time()
        work_segs = engine.transcribe(piece, sr)
        work_wall = time.time() - t0

        t0 = time.time()
        probe_segs = engine.transcribe(ref_samples, ref_sr)
        probe_wall = time.time() - t0
        probe_text = " ".join(s["text"] for s in probe_segs).strip()

        similarity = _similar(baseline_text, probe_text)
        is_degraded = (
            not probe_text
            or _wordiness(probe_text) < 0.3
            or similarity < args.similarity_threshold
        )

        temps = [s["temperature"] for s in work_segs
                 if isinstance(s.get("temperature"), (int, float))]
        rec = {
            "engine": engine.name,
            "iteration": i,
            "work_segments": len(work_segs),
            "work_wall_s": round(work_wall, 3),
            "work_rtf": round(work_wall / max(len(piece) / sr, 1e-6), 3),
            "work_max_temperature": max(temps) if temps else None,
            "work_fallback_segments": sum(1 for t in temps if t > 0.0),
            "probe_segments": len(probe_segs),
            "probe_wall_s": round(probe_wall, 3),
            "probe_similarity": round(similarity, 3),
            "probe_degraded": bool(is_degraded),
            "probe_text": probe_text[:200],
        }
        rec.update(_memory_snapshot())
        records.append(rec)

        flag = "  <-- DEGRADED" if is_degraded else ""
        _log(f"[{i:>3}] work {rec['work_wall_s']:>6.2f}s "
             f"(rtf {rec['work_rtf']:>5.2f}, {rec['work_segments']:>3} segs) | "
             f"probe sim {similarity:.2f}{flag}")

        if is_degraded and degraded_at is None:
            degraded_at = i
            _log("\nThe reference clip stopped transcribing correctly from the "
                 "already-loaded instance.\nContinuing a little further to show "
                 "whether it recovers...")
            if i + 3 < n:
                n = i + 4

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _log(f"\nWrote {len(records) - 1} chunk record(s) plus a config header "
             f"to {args.out}")
    except Exception as exc:
        _log(f"\nWARNING: could not write {args.out}: {exc}")

    _log("\n" + "=" * 68)
    if degraded_at is not None:
        _log(f"VERDICT: instance degraded after {degraded_at} chunk(s), "
             f"engine={engine.name}, profile={args.profile}")
        if engine.name == "bare":
            _log("This reproduced WITHOUT WhisperJAV, which points at\n"
                 "faster-whisper / CTranslate2 rather than at WhisperJAV.")
        else:
            _log("This reproduced through WhisperJAV. Please also run with\n"
                 "--engine bare: if that stays clean, the cause is on our side.")
    else:
        _log(f"VERDICT: no degradation in {len(records) - 1} chunk(s), "
             f"engine={engine.name}, profile={args.profile}")
        _log("Either this material does not trigger it, or more chunks are needed.\n"
             "Raising --iterations, or using audio known to have failed, may help.\n"
             "A clean result on ONE arm settles nothing on its own: the comparison\n"
             "between --engine bare and --engine whisperjav is what discriminates.")
    _log("=" * 68)
    _log(f"\nPlease attach {args.out} to https://github.com/meizhong986/WhisperJAV/issues/394")
    return 0


if __name__ == "__main__":
    sys.exit(main())
