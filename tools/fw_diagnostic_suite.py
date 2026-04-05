#!/usr/bin/env python3
"""
Faster-Whisper Diagnostic Suite — Isolates FW's per-chunk behavior.

Purpose
-------
WhisperJAV's balanced pipeline (Faster-Whisper backend) returns empty output for
~89% of ground-truth subtitles on JAV content, while the fidelity pipeline
(OpenAI Whisper) captures ~84% with identical thresholds.

This script reproduces the EXACT production audio flow (semantic scene
detection + Silero-v6.2 VAD grouping via WhisperJAV modules), then calls
Faster-Whisper DIRECTLY on each VAD group with:
  - Explicit capture of every TranscriptionInfo field
  - Per-segment avg_logprob, no_speech_prob, compression_ratio, temperature
  - "empty" vs "output" determination (n_segments == 0)
  - Multiple parameter variants to identify WHICH param causes the empty output

Outputs (per run)
-----------------
  <outdir>/run_<X>_results.csv          One row per VAD group
  <outdir>/run_<X>_summary.json         Aggregate stats
  <outdir>/run_<X>_fw_verbose.log       Raw FW stdout/stderr
  <outdir>/vad_groups.json              VAD group metadata (saved once)
  <outdir>/scenes.json                  Scene metadata (saved once)
  <outdir>/scenes/*.wav                 Scene WAVs (saved once)

Usage
-----
  python fw_diagnostic_suite.py --run all
  python fw_diagnostic_suite.py --run A,C --input <media>
  python fw_diagnostic_suite.py --run all --skip-prep   # re-use cached scenes/vad
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_INPUT = Path(r"C:\BIN\git\WhisperJav_V1_Minami_Edition\test_media\293sec-S01E04-scene4.mkv")
DEFAULT_OUTDIR = Path(r"F:\MEDIA_DLNA\SONE-853\DIAG_FW")
DEFAULT_MODEL = "large-v2"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "ja"


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
def make_logger(verbose: bool = True) -> logging.Logger:
    lg = logging.getLogger("fw_diag")
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)
    if lg.handlers:
        return lg
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s",
                                     "%H:%M:%S"))
    lg.addHandler(h)
    return lg


log = make_logger()


def enable_verbose_dependencies() -> None:
    """Enable DEBUG logging on Faster-Whisper and WhisperJAV loggers.

    E0a: Faster-Whisper does NOT support a ``verbose=True`` flag on its
    ``transcribe()`` method (unlike OpenAI Whisper). Instead, it emits debug
    messages through the ``faster_whisper`` Python logger. The critical
    "No speech threshold is met (%f > %f)" message (transcribe.py line 1222)
    is ONLY visible at DEBUG level. We enable it here.

    E0b: WhisperJAV modules (AudioExtractor, scene detectors, speech
    segmenters) use the ``whisperjav`` logger. Enabling DEBUG exposes their
    internal decisions (VAD group counts, scene boundaries, etc.).

    Both loggers are attached to the root handler so their output flows
    to stdout alongside ``fw_diag`` messages.
    """
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    for logger_name in ("faster_whisper", "whisperjav"):
        dep_logger = logging.getLogger(logger_name)
        dep_logger.setLevel(logging.DEBUG)
        # Remove any existing handlers to avoid duplicate output
        for existing in list(dep_logger.handlers):
            dep_logger.removeHandler(existing)
        dep_logger.addHandler(handler)
        dep_logger.propagate = False  # don't double-log via root logger
        log.info(f"[E0] Enabled DEBUG logging for '{logger_name}' logger")


# Enable dependency verbose logging at import time so early phases also emit debug
enable_verbose_dependencies()


# ============================================================================
# Phase 1 — Audio Extraction (uses WhisperJAV module)
# ============================================================================
def extract_audio(input_media: Path, out_wav: Path) -> Tuple[Path, float]:
    from whisperjav.modules.audio_extraction import AudioExtractor
    log.info("=" * 70)
    log.info("PHASE 1: Audio extraction (WhisperJAV AudioExtractor)")
    log.info("=" * 70)
    log.info(f"  input:  {input_media}")
    log.info(f"  output: {out_wav}")
    extractor = AudioExtractor(sample_rate=16000, channels="mono", audio_codec="pcm_s16le")
    out_path, duration = extractor.extract(input_media, out_wav)
    log.info(f"  duration: {duration:.2f}s  ({out_path})")
    return out_path, duration


# ============================================================================
# Phase 2 — Semantic Scene Detection (uses WhisperJAV factory)
# ============================================================================
def detect_scenes(audio_wav: Path, scenes_dir: Path, basename: str) -> List[Dict[str, Any]]:
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
    log.info("=" * 70)
    log.info("PHASE 2: Semantic scene detection")
    log.info("=" * 70)

    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Semantic detector — matches production "balanced" pipeline.
    # If semantic is unavailable (sklearn missing), fall back to auditok.
    available, hint = SceneDetectorFactory.is_backend_available("semantic")
    if not available:
        log.warning(f"Semantic backend unavailable ({hint}) — falling back to 'auditok'.")
        detector = SceneDetectorFactory.create("auditok", max_duration=29.0, min_duration=0.3)
    else:
        detector = SceneDetectorFactory.create("semantic")

    log.info(f"  detector: {detector.name}")
    result = detector.detect_scenes(audio_wav, scenes_dir, basename)
    log.info(f"  scenes:   {result.num_scenes}  "
             f"(coverage={result.coverage_ratio:.1%})")

    scenes_meta = []
    for i, scene in enumerate(result.scenes):
        scenes_meta.append({
            "scene_idx": i,
            "start_sec": scene.start_sec,
            "end_sec": scene.end_sec,
            "duration_sec": scene.duration_sec,
            "scene_path": str(scene.scene_path) if scene.scene_path else None,
        })
    return scenes_meta


# ============================================================================
# Phase 3 — Speech Segmentation (VAD grouping) per scene
# ============================================================================
def segment_scenes(
    scenes_meta: List[Dict[str, Any]],
    segmenter_backend: str = "silero-v6.2",
) -> List[Dict[str, Any]]:
    """Run the speech segmenter on each scene WAV, returning a flat list of
    VAD groups with scene/group indices and absolute timestamps."""
    from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
    import soundfile as sf

    log.info("=" * 70)
    log.info(f"PHASE 3: Speech segmentation ({segmenter_backend}) per scene")
    log.info("=" * 70)

    # Fallback if silero-v6.2 unavailable
    available, hint = SpeechSegmenterFactory.is_backend_available(segmenter_backend)
    if not available:
        log.warning(f"Segmenter '{segmenter_backend}' unavailable ({hint}) — "
                    f"falling back to 'silero' (v4.0).")
        segmenter_backend = "silero"

    # Mirror production balanced-pipeline grouping knobs.
    seg_config = {
        "chunk_threshold_s": 1.0,
        "max_group_duration_s": 29.0,
    }
    segmenter = SpeechSegmenterFactory.create(segmenter_backend, config=seg_config)
    log.info(f"  segmenter: {segmenter.name}")

    vad_groups_flat = []
    for scene in scenes_meta:
        scene_idx = scene["scene_idx"]
        scene_path = Path(scene["scene_path"])
        scene_start = scene["start_sec"]

        # Load scene WAV as numpy (mono 16kHz from scene detector).
        audio, sr = sf.read(str(scene_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        seg_result = segmenter.segment(audio, sample_rate=sr)
        log.info(f"  scene {scene_idx:02d}: {seg_result.num_segments} segments, "
                 f"{seg_result.num_groups} groups "
                 f"({seg_result.speech_coverage_ratio:.1%} speech)")

        for grp_idx, grp in enumerate(seg_result.groups):
            if not grp:
                continue
            grp_start_rel = grp[0].start_sec
            grp_end_rel = grp[-1].end_sec
            vad_groups_flat.append({
                "scene_idx": scene_idx,
                "group_idx": grp_idx,
                "scene_wav": str(scene_path),
                "scene_start_abs": scene_start,
                "group_start_rel": grp_start_rel,
                "group_end_rel": grp_end_rel,
                "group_duration": grp_end_rel - grp_start_rel,
                "group_start_abs": scene_start + grp_start_rel,
                "group_end_abs": scene_start + grp_end_rel,
                "n_sub_segments": len(grp),
            })

    log.info(f"  total VAD groups across all scenes: {len(vad_groups_flat)}")
    try:
        segmenter.cleanup()
    except Exception:
        pass
    return vad_groups_flat


# ============================================================================
# Phase 4 — Parameter variants (the actual diagnostic)
# ============================================================================
def _base_params(language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """faster-whisper transcribe() parameters used as a baseline for variants.

    This MIRRORS EXACTLY the kwargs construct that WhisperJAV's balanced
    pipeline passes to model.transcribe() in production. Captured from
    production debug logs (R6 run):

        faster_whisper_pro_asr.py assembles whisper_params from:
        - decoder_params (task, language, beam_size, best_of, patience,
          suppress_blank, without_timestamps, max_initial_timestamp)
        - provider_params (temperature, compression_ratio_threshold,
          logprob_threshold, no_speech_threshold, drop_nonverbal_vocals [popped],
          condition_on_previous_text, word_timestamps, repetition_penalty,
          no_repeat_ngram_size, multilingual, log_progress, chunk_length)

        Then adds at call time:
        - vad_filter=False (always — WhisperJAV feeds pre-VAD'd audio)
        - vad_parameters={...} (passed but unused when vad_filter=False)
        - logprob_threshold renamed to log_prob_threshold

    Each PARAM_VARIANT below then overrides beam_size/best_of/patience/
    temperature/log_prob_threshold/no_speech_threshold/compression_ratio_threshold
    to isolate which parameter causes the empty-output behavior.
    """
    return {
        # === Decoder (matches decoder_params in production) ===
        "language": language,
        "task": "transcribe",
        "suppress_blank": True,
        "without_timestamps": False,
        "max_initial_timestamp": 0.0,
        # === Transcriber / Engine (matches provider_params in production) ===
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "chunk_length": 30,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        "multilingual": False,
        "log_progress": False,
        # === VAD (critical: we feed pre-VAD'd audio, vad_filter MUST be False) ===
        "vad_filter": False,
        # vad_parameters is unused when vad_filter=False but WhisperJAV still
        # passes it in production kwargs. Included here for EXACT kwargs parity.
        "vad_parameters": {
            "threshold": 0.05,
            "neg_threshold": 0.08,
            "min_speech_duration_ms": 30,
            "max_speech_duration_s": 14.0,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 300,
        },
    }


PARAM_VARIANTS: Dict[str, Dict[str, Any]] = {
    # Run A1 — HYPOTHESIS: Faster-Whisper has a hard floor at logprob=-1.0.
    # Going below (to -1.3) causes catastrophic empty output because the skip
    # condition `avg_logprob <= logprob_threshold` is too easily met for
    # noisy JAV content whose avg_logprob naturally sits around -1.0 to -1.4.
    # A1 tests: keep no_speech=0.9 (permissive), raise logprob to -1.0.
    # If A1 captures significantly more than A, the -1.0 floor is confirmed.
    "A1": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": 0.0,
        "log_prob_threshold": -1.0,   # HYPOTHESIS: hard floor for Faster-Whisper
        "no_speech_threshold": 0.9,
        "compression_ratio_threshold": 2.4,
    },
    # Run A2 — HYPOTHESIS: Combines -1.0 logprob with tighter no_speech=0.6.
    # If A2 still captures well, confirms -1.0 is the critical threshold,
    # NOT no_speech_threshold. If A2 captures less than A1, no_speech=0.9 helps.
    "A2": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": 0.0,
        "log_prob_threshold": -1.0,   # HYPOTHESIS: hard floor for Faster-Whisper
        "no_speech_threshold": 0.6,   # tighter — closer to library default
        "compression_ratio_threshold": 2.4,
    },
    # Run A — WhisperJAV's current aggressive parameters
    "A": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": 0.0,
        "log_prob_threshold": -1.3,
        "no_speech_threshold": 0.9,
        "compression_ratio_threshold": 2.4,
    },
    # Run B — Reduce beam / best_of / patience (greedy-ish)
    "B": {
        **_base_params(),
        "beam_size": 1,
        "best_of": 1,
        "patience": 1.0,
        "temperature": 0.0,
        "log_prob_threshold": -1.3,
        "no_speech_threshold": 0.9,
        "compression_ratio_threshold": 2.4,
    },
    # Run C — Temperature fallback (whisper default)
    "C": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "log_prob_threshold": -1.3,
        "no_speech_threshold": 0.9,
        "compression_ratio_threshold": 2.4,
    },
    # Run D — condition_on_previous_text=True
    "D": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": 0.0,
        "log_prob_threshold": -1.3,
        "no_speech_threshold": 0.9,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": True,
    },
    # Run E — Very high no_speech_threshold (force speech-mode)
    "E": {
        **_base_params(),
        "beam_size": 4,
        "best_of": 3,
        "patience": 2.5,
        "temperature": 0.0,
        "log_prob_threshold": -1.3,
        "no_speech_threshold": 0.99,
        "compression_ratio_threshold": 2.4,
    },
    # Run F — Faster-Whisper library defaults (sanity reference)
    "F": {
        **_base_params(),
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
    },
}


# ============================================================================
# Phase 5 — Direct Faster-Whisper invocation with full capture
# ============================================================================
@dataclass
class GroupResult:
    scene_idx: int
    group_idx: int
    start_abs: float
    end_abs: float
    duration: float
    audio_peak: float = 0.0
    audio_rms: float = 0.0
    detected_language: str = ""
    language_probability: float = 0.0
    info_duration: float = 0.0
    info_duration_after_vad: float = 0.0
    n_segments_yielded: int = 0
    raw_text_concat: str = ""
    first_seg_avg_logprob: Optional[float] = None
    first_seg_no_speech_prob: Optional[float] = None
    first_seg_compression_ratio: Optional[float] = None
    first_seg_temperature: Optional[float] = None
    all_avg_logprob: List[float] = field(default_factory=list)
    all_no_speech_prob: List[float] = field(default_factory=list)
    all_compression_ratio: List[float] = field(default_factory=list)
    all_temperature: List[float] = field(default_factory=list)
    error: str = ""

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "scene_idx": self.scene_idx,
            "group_idx": self.group_idx,
            "start_abs": round(self.start_abs, 3),
            "end_abs": round(self.end_abs, 3),
            "duration": round(self.duration, 3),
            "audio_peak": round(self.audio_peak, 4),
            "audio_rms": round(self.audio_rms, 4),
            "detected_language": self.detected_language,
            "language_probability": round(self.language_probability, 4),
            "info_duration_after_vad": round(self.info_duration_after_vad, 3),
            "n_segments_yielded": self.n_segments_yielded,
            "first_seg_avg_logprob": self.first_seg_avg_logprob,
            "first_seg_no_speech_prob": self.first_seg_no_speech_prob,
            "first_seg_compression_ratio": self.first_seg_compression_ratio,
            "first_seg_temperature": self.first_seg_temperature,
            "raw_text_concat": self.raw_text_concat.replace("\n", " ").replace("\r", " ")[:500],
            "error": self.error[:200],
        }


def run_variant(
    run_name: str,
    model,  # WhisperModel
    vad_groups: List[Dict[str, Any]],
    params: Dict[str, Any],
    outdir: Path,
) -> None:
    import soundfile as sf

    log.info("=" * 70)
    log.info(f"RUN {run_name}: transcribing {len(vad_groups)} VAD groups")
    log.info("=" * 70)
    compact_params = {k: v for k, v in params.items() if k != "temperature"}
    log.info(f"  params: {json.dumps(compact_params, indent=2)}")
    log.info(f"  temperature: {params.get('temperature')}")

    fw_log_path = outdir / f"run_{run_name}_fw_verbose.log"
    csv_path = outdir / f"run_{run_name}_results.csv"
    summary_path = outdir / f"run_{run_name}_summary.json"

    results: List[GroupResult] = []
    scene_cache: Dict[str, Tuple[Any, int]] = {}

    fw_log = open(fw_log_path, "w", encoding="utf-8")
    t_start = time.time()
    try:
        for i, grp in enumerate(vad_groups):
            scene_wav = grp["scene_wav"]
            if scene_wav not in scene_cache:
                audio, sr = sf.read(scene_wav, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                scene_cache[scene_wav] = (audio, sr)
            audio, sr = scene_cache[scene_wav]

            start_sample = int(grp["group_start_rel"] * sr)
            end_sample = int(grp["group_end_rel"] * sr)
            grp_audio = audio[start_sample:end_sample]

            gr = GroupResult(
                scene_idx=grp["scene_idx"],
                group_idx=grp["group_idx"],
                start_abs=grp["group_start_abs"],
                end_abs=grp["group_end_abs"],
                duration=grp["group_duration"],
            )

            if grp_audio.size > 0:
                gr.audio_peak = float(abs(grp_audio).max())
                gr.audio_rms = float((grp_audio ** 2).mean() ** 0.5)
            else:
                gr.error = "EMPTY_AUDIO_SLICE"
                results.append(gr)
                continue

            buf_out = io.StringIO()
            buf_err = io.StringIO()
            header = (f"\n\n========== grp {i+1}/{len(vad_groups)} "
                      f"scene={gr.scene_idx} grp={gr.group_idx} "
                      f"abs={gr.start_abs:.2f}-{gr.end_abs:.2f}s "
                      f"dur={gr.duration:.2f}s ==========\n")
            fw_log.write(header)
            fw_log.flush()

            try:
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    segments_gen, info = model.transcribe(grp_audio, **params)
                    segments = list(segments_gen)

                gr.detected_language = info.language or ""
                gr.language_probability = float(info.language_probability or 0.0)
                gr.info_duration = float(getattr(info, "duration", 0.0) or 0.0)
                gr.info_duration_after_vad = float(
                    getattr(info, "duration_after_vad", gr.info_duration) or 0.0
                )

                gr.n_segments_yielded = len(segments)
                if segments:
                    texts = []
                    for s in segments:
                        texts.append(s.text.strip())
                        gr.all_avg_logprob.append(float(s.avg_logprob))
                        gr.all_no_speech_prob.append(float(s.no_speech_prob))
                        gr.all_compression_ratio.append(float(s.compression_ratio))
                        gr.all_temperature.append(float(s.temperature))
                    gr.raw_text_concat = " / ".join(texts)
                    gr.first_seg_avg_logprob = gr.all_avg_logprob[0]
                    gr.first_seg_no_speech_prob = gr.all_no_speech_prob[0]
                    gr.first_seg_compression_ratio = gr.all_compression_ratio[0]
                    gr.first_seg_temperature = gr.all_temperature[0]

            except Exception as e:
                gr.error = f"{type(e).__name__}: {e}"
                log.error(f"  grp {i}: transcribe failed: {gr.error}")
                traceback.print_exc(file=fw_log)

            captured = buf_out.getvalue() + buf_err.getvalue()
            if captured:
                fw_log.write(captured)
            fw_log.write(
                f"[result] n_segments={gr.n_segments_yielded} "
                f"peak={gr.audio_peak:.3f} rms={gr.audio_rms:.4f} "
                f"no_speech[0]={gr.first_seg_no_speech_prob} "
                f"avg_logprob[0]={gr.first_seg_avg_logprob} "
                f"text={gr.raw_text_concat[:120]!r}\n"
            )
            fw_log.flush()

            results.append(gr)

            if (i + 1) % 10 == 0 or (i + 1) == len(vad_groups):
                empties = sum(1 for r in results if r.n_segments_yielded == 0)
                log.info(f"  [{i+1:3d}/{len(vad_groups)}] empties={empties}")

    finally:
        fw_log.close()

    elapsed = time.time() - t_start

    fieldnames = list(GroupResult(0, 0, 0, 0, 0).to_csv_row().keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.to_csv_row())

    total = len(results)
    empty = sum(1 for r in results if r.n_segments_yielded == 0)
    withtxt = total - empty
    all_lp = [lp for r in results for lp in r.all_avg_logprob]
    all_ns = [ns for r in results for ns in r.all_no_speech_prob]
    summary = {
        "run_name": run_name,
        "params": {k: (v if not callable(v) else str(v)) for k, v in params.items()},
        "total_groups": total,
        "groups_with_output": withtxt,
        "groups_empty": empty,
        "empty_pct": round(100.0 * empty / total, 2) if total else 0.0,
        "mean_avg_logprob": round(sum(all_lp) / len(all_lp), 4) if all_lp else None,
        "mean_no_speech_prob": round(sum(all_ns) / len(all_ns), 4) if all_ns else None,
        "n_segments_total": sum(r.n_segments_yielded for r in results),
        "elapsed_sec": round(elapsed, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("-" * 70)
    log.info(f"RUN {run_name} DONE  ({elapsed:.1f}s)")
    log.info(f"  total groups:       {total}")
    log.info(f"  with output:        {withtxt}")
    log.info(f"  empty:              {empty}  ({summary['empty_pct']}%)")
    log.info(f"  mean avg_logprob:   {summary['mean_avg_logprob']}")
    log.info(f"  mean no_speech_prob:{summary['mean_no_speech_prob']}")
    log.info(f"  CSV:                {csv_path}")
    log.info(f"  summary:            {summary_path}")
    log.info(f"  fw log:             {fw_log_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Faster-Whisper diagnostic suite using WhisperJAV modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Runs:
  A  current aggressive (beam=4, best_of=3, patience=2.5, logprob=-1.3, no_speech=0.9)
  B  greedy (beam=1, best_of=1, patience=1.0, else same as A)
  C  A + temperature fallback [0.0..1.0]
  D  A + condition_on_previous_text=True
  E  A + no_speech_threshold=0.99
  F  faster-whisper library defaults (reference)
""",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Input media file (default: {DEFAULT_INPUT})")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--run", default="A",
                   help="Comma-separated run IDs or 'all' (e.g. 'A', 'A,C,E', 'all')")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE)
    p.add_argument("--language", default=DEFAULT_LANGUAGE)
    p.add_argument("--segmenter", default="silero-v6.2",
                   help="Speech segmenter backend (default silero-v6.2)")
    p.add_argument("--skip-prep", action="store_true",
                   help="Reuse cached scenes + vad_groups.json from outdir")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    scenes_dir = args.outdir / "scenes"
    audio_wav = args.outdir / f"{args.input.stem}_extracted.wav"
    vad_path = args.outdir / "vad_groups.json"
    scenes_path = args.outdir / "scenes.json"

    # Prep
    if args.skip_prep and vad_path.exists() and scenes_path.exists():
        log.info(f"Skipping prep, reusing {vad_path}")
        vad_groups = json.loads(vad_path.read_text(encoding="utf-8"))
        scenes_meta = json.loads(scenes_path.read_text(encoding="utf-8"))
    else:
        if not args.input.exists():
            log.error(f"Input media not found: {args.input}")
            return 2
        if not audio_wav.exists():
            extract_audio(args.input, audio_wav)
        else:
            log.info(f"Re-using extracted WAV: {audio_wav}")
        scenes_meta = detect_scenes(audio_wav, scenes_dir, args.input.stem)
        scenes_path.write_text(json.dumps(scenes_meta, indent=2), encoding="utf-8")
        vad_groups = segment_scenes(scenes_meta, segmenter_backend=args.segmenter)
        vad_path.write_text(json.dumps(vad_groups, indent=2), encoding="utf-8")
        log.info(f"  saved: {scenes_path}")
        log.info(f"  saved: {vad_path}")

    if not vad_groups:
        log.error("No VAD groups produced — cannot continue.")
        return 3

    # Load FW model
    from faster_whisper import WhisperModel
    log.info("=" * 70)
    log.info(f"Loading Faster-Whisper model: {args.model} "
             f"({args.device}, {args.compute_type})")
    log.info("=" * 70)
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=0,
        num_workers=1,
    )
    log.info("  model loaded.")

    # Resolve runs
    if args.run.lower() == "all":
        runs = list(PARAM_VARIANTS.keys())
    else:
        runs = [r.strip().upper() for r in args.run.split(",") if r.strip()]
        for r in runs:
            if r not in PARAM_VARIANTS:
                log.error(f"Unknown run: {r}  (available: {list(PARAM_VARIANTS.keys())})")
                return 4

    for key in PARAM_VARIANTS:
        PARAM_VARIANTS[key]["language"] = args.language

    # Execute
    for run_name in runs:
        params = dict(PARAM_VARIANTS[run_name])
        run_variant(run_name, model, vad_groups, params, args.outdir)

    log.info("=" * 70)
    log.info("ALL RUNS COMPLETE")
    log.info(f"Outputs in: {args.outdir}")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
