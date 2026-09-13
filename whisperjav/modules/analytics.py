"""Audio analytics: tell the user which parts of their film are acoustically awkward.

Why this exists
---------------
A user ran a 25-minute film through Balanced and got 25 subtitles. The run reported
success and 96% coverage. In fact Silero VAD had kept 77 seconds of the 1,500 and
two thirds of the dialogue was never transcribed -- lowering the VAD threshold to
0.15 recovered 45 more lines of ordinary Japanese speech. Nothing in the run said so.

This module says so. It reports, once per file, which scenes look likely to lose
speech and which look acoustically hard, with time offsets so the user can check
those places and decide whether to re-run with different settings.

What it measures, and what it deliberately does not
---------------------------------------------------
Everything comes from ONE Silero VAD pass per scene, at two thresholds. The semantic
scene detector supplies only the scene boundaries and their start times.

Two signals ship, both calibrated against measured outcomes:

  QUIET       For the FILE as a whole: speech found at 0.15 divided by speech
              found at the run's threshold, >= 2.0. The film above measures 2.54;
              nine files that transcribe correctly measure 1.13 to 1.71.
              It is a RATIO, not a loudness, because loudness does not separate
              them: scenes that lost dialogue measured -50.5..-39.7 dB and scenes
              that did not measured -49.6..-39.8 dB. A ratio is also
              self-normalising, so an unusually quiet or loud film cannot fool it.

              This is deliberately reported for the whole file and NOT per scene.
              Per scene the measure does not work: healthy films contain single
              scenes scoring up to 4.25, higher than five of the seven scenes that
              actually lost dialogue in the film above, and the scene that lost the
              most lines of all scored only 1.83. A quiet passage inside a healthy
              film is normal and costs the user nothing. Only the whole-file figure
              separates a film that loses speech from one that does not.

              (An earlier per-scene calibration put the scenes that lost dialogue at
              2.01-3.59. Those were measured on the unpadded scene times from the
              detector's JSON. The spans actually handed to the recogniser carry
              about 0.3 s of padding at each end, and on those the same scenes give
              1.83-4.46. The padded figures are the ones that stand, because they
              are the audio the recogniser is given.)

  DIFFICULT   signal-to-noise ratio <= 3 dB, per scene. Against the seven Netflix
              clips that have human Japanese subtitles, SNR tracks measured
              character error rate with a Spearman correlation of -0.89, and a 3 dB
              line flags exactly the three worst-transcribed clips and nothing else.
              The speech literature puts "difficult" nearer 10 dB, but every file
              of this kind measures between 2 and 14 dB, so a 10 dB line would
              flag everything and mean nothing.
              Note the calibration was done a clip at a time, each clip scored as a
              whole against its own subtitles; applying it per scene extends it to a
              shorter span than it was measured on. SNR is a local property of the
              audio, so that extension is reasonable, but it is an extension.

Three further ideas were tried and dropped for want of evidence, and should not be
revived without new measurements: a noise anomaly (no case where it predicted a
worse transcript); a hallucination anomaly from the share of a scene that is not
speech (rank correlation +0.50 on seven clips, with a clear counterexample); and a
whole-film loudness warning (a film that transcribes correctly and one that fails
have the same noise floor, so it cannot work).

The scene classifier's own labels -- silence, ambient_noise, high_energy and the
rest -- are NOT used in anything shown to a user. They are calibrated against each
film's own levels, and `ambient_noise` is its fallback bucket: it was assigned to a
clip that turned out to be the second best in the set, with 56 subtitles and the
highest SNR measured anywhere. Reporting it as "background noise, little speech"
would have been plainly wrong.

Structure
---------
Findings are produced as records first and rendered to text second. Nothing here
decides anything: it reads audio, appends records, and returns them. `report()`
wraps the whole thing so that a fault in analytics can never disturb a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from whisperjav.utils.logger import logger

# --- thresholds, each with the measurement that set it -----------------------

QUIET_RATIO = 2.0          # reference speech ratio; at or above this the file looks quiet
QUIET_MIN_SPEECH_S = 2.0   # below this much speech in the WHOLE file, do not judge it
DIFFICULT_SNR_DB = 3.0     # <= this looks acoustically hard

# Both thresholds are FIXED, and deliberately not the ones the run transcribes with.
# The run's threshold is 0.50/0.40/0.30 on Balanced and 0.41/0.28/0.18 on Fidelity,
# and this module measures with the Silero build faster-whisper bundles, while the
# run may transcribe with another one installed in a worker process. Using the run's
# threshold would therefore compare two numbers produced by different instruments,
# and only 0.40 was ever calibrated. Fixing both ends makes the ratio a property of
# the audio, measured the same way every time, which is what the calibration
# supports. The remedy printed to the user stays directionally right either way.
REFERENCE_THRESHOLD = 0.40
PROBE_THRESHOLD = 0.15

SAMPLE_RATE = 16000
_WINDOW_S = 0.1            # window for the speech/background level comparison


@dataclass
class Finding:
    """One observation about one scene, as data rather than as a sentence.

    Rendering reads these; so can anything added later -- a pre-transcription
    analysis mode, a settings adviser, the GUI, or the run manifest -- without
    having to parse English.
    """

    kind: str                     # "quiet" | "difficult"
    scene_index: Optional[int]    # None when the finding is about the whole file
    start_sec: float
    end_sec: float
    measurements: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "scene_index": self.scene_index,
            "scope": "file" if self.scene_index is None else "scene",
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "measurements": {k: round(v, 4) for k, v in self.measurements.items()},
        }


@dataclass
class AudioAnalytics:
    """Everything measured for one input file."""

    filename: str
    scene_count: int = 0
    total_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    findings: List[Finding] = field(default_factory=list)
    scenes_measured: int = 0

    @property
    def speech_share(self) -> float:
        if self.total_duration_sec <= 0:
            return 0.0
        return self.speech_duration_sec / self.total_duration_sec

    def of_kind(self, kind: str) -> List[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "scene_count": self.scene_count,
            "scenes_measured": self.scenes_measured,
            "total_duration_sec": round(self.total_duration_sec, 3),
            "speech_duration_sec": round(self.speech_duration_sec, 3),
            "speech_share": round(self.speech_share, 4),
            "findings": [f.to_dict() for f in self.findings],
        }


# --- measurement -------------------------------------------------------------


def _mmss(seconds: float) -> str:
    """m:ss, or h:mm:ss past an hour -- a feature film needs the hour."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _speech_seconds(audio, threshold: float) -> Tuple[float, Optional[Any]]:
    """Seconds of speech Silero VAD finds, and the mask of where it found it."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=100,
        max_speech_duration_s=6.0,
        min_silence_duration_ms=300,
        speech_pad_ms=400,
    )
    spans = get_speech_timestamps(audio, options)
    if not spans:
        return 0.0, None
    mask = np.zeros(len(audio), dtype=bool)
    for span in spans:
        mask[span["start"]:span["end"]] = True
    total = sum(s["end"] - s["start"] for s in spans) / SAMPLE_RATE
    return float(total), mask


def _snr_db(audio, mask) -> Optional[float]:
    """Speech level minus background level, in dB.

    Medians of short windows rather than means, so one loud bang does not stand in
    for the background. Returns None when the scene is all speech or all silence,
    where the comparison has no meaning.
    """
    import numpy as np

    window = int(SAMPLE_RATE * _WINDOW_S)
    count = len(audio) // window
    if count < 4:
        return None
    speech, background = [], []
    for i in range(count):
        chunk = slice(i * window, (i + 1) * window)
        level = float(np.sqrt(np.mean(audio[chunk].astype(np.float64) ** 2)))
        (speech if mask[chunk].mean() > 0.5 else background).append(level)
    if not speech or not background:
        return None
    return _db(float(np.median(speech))) - _db(float(np.median(background)))


def _load_audio(path: Path):
    """Read an extracted audio file as mono float32 at 16 kHz.

    AudioExtractor always writes 16 kHz mono pcm_s16le, so soundfile handles this
    without resampling. faster-whisper's decoder is the fallback for anything else.
    """
    import numpy as np

    try:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if rate == SAMPLE_RATE:
            return np.ascontiguousarray(data, dtype=np.float32)
    except Exception:
        pass

    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def analyse(
    audio_path: Path,
    scenes: Sequence[Tuple[int, float, float]],
    vad_threshold: Optional[float] = 0.40,
) -> AudioAnalytics:
    """Measure one file. `scenes` is (index, start_sec, end_sec) per scene.

    `vad_threshold` is the Silero VAD threshold THIS RUN will transcribe with. Pass
    None when the run does not use Silero at a threshold -- the Qwen and
    anime-whisper pipelines segment with WhisperSeg by default. The quiet check is
    then skipped, because it measures whether a Silero threshold is under-detecting
    and would otherwise recommend a setting that run does not use. The
    signal-to-noise check still applies: it describes the audio, not the tool.

    Raises on a genuine failure; `report()` is the wrapped entry point callers use.
    """
    measure_quiet = vad_threshold is not None
    shipped_total = 0.0
    probe_total = 0.0
    audio_path = Path(audio_path)
    audio = _load_audio(audio_path)

    result = AudioAnalytics(filename=audio_path.name, scene_count=len(scenes))
    result.total_duration_sec = len(audio) / SAMPLE_RATE

    for index, start, end in scenes:
        lo = max(0, int(start * SAMPLE_RATE))
        hi = min(len(audio), int(end * SAMPLE_RATE))
        if hi - lo < SAMPLE_RATE:          # under a second, nothing to say
            continue
        chunk = audio[lo:hi]
        result.scenes_measured += 1

        shipped_s, mask = _speech_seconds(chunk, REFERENCE_THRESHOLD)
        result.speech_duration_sec += shipped_s

        if measure_quiet and shipped_s > 0:
            probe_s, _ = _speech_seconds(chunk, PROBE_THRESHOLD)
            shipped_total += shipped_s
            probe_total += probe_s

        if mask is not None:
            snr = _snr_db(chunk, mask)
            if snr is not None and snr <= DIFFICULT_SNR_DB:
                result.findings.append(Finding(
                    "difficult", index, start, end, {"snr_db": snr},
                ))

    # The quiet judgement is made once, for the whole file. scene_index -1 says so.
    if measure_quiet and shipped_total >= QUIET_MIN_SPEECH_S:
        ratio = probe_total / shipped_total
        if ratio >= QUIET_RATIO:
            result.findings.append(Finding(
                "quiet", None, 0.0, result.total_duration_sec,
                {"ratio": ratio, "speech_sec": shipped_total,
                 "speech_sec_permissive": probe_total},
            ))

    return result


# --- rendering ---------------------------------------------------------------

_WIDTH = 66
_RULE = "-" * _WIDTH


def _columns(findings: List[Finding], per_row: int = 2) -> List[str]:
    cells = [
        f"scene {f.scene_index:<3d} {_mmss(f.start_sec):>8} - {_mmss(f.end_sec)}"
        for f in sorted(findings, key=lambda x: x.scene_index)
    ]
    width = max((len(c) for c in cells), default=0) + 4
    rows = []
    for i in range(0, len(cells), per_row):
        rows.append("     " + "".join(c.ljust(width) for c in cells[i:i + per_row]).rstrip())
    return rows


def render(result: AudioAnalytics) -> List[str]:
    """Turn the measurements into the lines a user reads. Never raises."""
    quiet = result.of_kind("quiet")
    difficult = result.of_kind("difficult")

    lines = [_RULE, f"  Audio analytics - {result.filename}", _RULE]
    lines.append(
        f"  {result.scene_count} scenes, {_mmss(result.total_duration_sec)} total. "
        f"Speech detected in {100 * result.speech_share:.0f}% of the running time."
    )

    if quiet:
        # Always a single whole-file judgement: the per-scene figure does not
        # distinguish a film that loses speech from a quiet passage in a healthy one.
        lines.append("")
        lines.append("  This file appears to be unusually quiet for speech detection.")
        lines.append("  Subtitles may be missing or sparse throughout.")

    if difficult:
        lines.append("")
        n = len(difficult)
        lines.append(f"  {n} scene{'' if n == 1 else 's'} appear{'s' if n == 1 else ''} "
                     f"to be acoustically difficult -")
        lines.append("  speech close in level to everything else around it:")
        lines.append("")
        lines.extend(_columns(difficult))

    if quiet:
        lines.append("")
        # Name flags that EXIST. `--speech-enhancement` never did: argparse rejects it
        # with exit 2, and on a single-pass Balanced or Fidelity run there is no way to
        # turn enhancement on at all -- no CLI flag, and the GUI's Speech Enhancer
        # control lives only in the Ensemble tab. Advice a user cannot follow is worse
        # than no advice. (Found 2026-09-12 while writing the release notes.)
        lines.append("  If the subtitles do look sparse, this may help:")
        lines.append("     --vad-threshold 0.15   pick up quieter speech")
        lines.append("")
        lines.append("  Raising the level first also helps, but it is only available on")
        lines.append("  two-pass runs (--pass1-speech-enhancer ffmpeg-dsp, or the Speech")
        lines.append("  Enhancer column in the GUI's Ensemble tab) and on --mode qwen")
        lines.append("  (--qwen-enhancer ffmpeg-dsp).")

    lines.append(_RULE)
    return lines


# --- the entry point pipelines call ------------------------------------------


def report(
    audio_path: Path,
    scenes: Sequence[Tuple[int, float, float]],
    scene_method: str,
    vad_threshold: Optional[float] = 0.40,
) -> Optional[AudioAnalytics]:
    """Measure and print, for the semantic scene detector only.

    Never raises and never changes anything about the run. A fault here costs the
    user a report, not their subtitles -- so every failure is swallowed after a
    debug line, deliberately.
    """
    try:
        if str(scene_method).lower() != "semantic":
            logger.debug(
                "Audio analytics: skipped, needs the semantic scene detector "
                "(this run used %s).", scene_method,
            )
            return None
        if not scenes:
            return None

        result = analyse(Path(audio_path), scenes, vad_threshold=vad_threshold)
        for line in render(result):
            logger.info(line)
        return result
    except Exception as exc:                                   # noqa: BLE001
        logger.debug("Audio analytics skipped: %s: %s", type(exc).__name__, exc)
        return None
