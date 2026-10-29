"""What an Ensemble pass running a CrispASR pipeline accepts, refuses and uses (1.10.0).

The CrispASR pipelines (parakeet-crispasr, cohere-crispasr) are Ensemble passes only, with no --mode
value (owner W118 2). Of a pass's settings they use: the model (--passN-model, a short name from the
manifest), the speech segmenter (--passN-speech-segmenter), the sensitivity (--passN-sensitivity),
--passN-vad-threshold and --passN-speech-pad-ms (owner W208), the source language (--language),
--device, and since design 22.9.1 (owner W212 D3, W214 2, W232 8) the scene detector
(--passN-scene-detector, semantic by default), the speech enhancer (--passN-speech-enhancer),
--passN-enhance-for-vad and the Customize values of the Segmenter and Scene tabs (--passN-params).
Only --passN-hf-params and --passN-qwen-params, the transformers and ChronosJAV settings, are not used.

The speech segmenter settings follow the pipelines the owner named (W199 DIR1: "For cohere, uses
similar presets and settings as qwenasr. And for prakeet use similar settings and presets as fidelity
whisper pipeline"; W202): ``segmenter_settings()`` resolves them through those pipelines' own functions.

Used by three callers, so the rules have one home:
  - whisperjav/main.py refuses a pass that cannot run (validate_crispasr_passes), as a usage error
    with exit status 2, and shows the resolved settings in --dump-params;
  - whisperjav/ensemble/pass_worker.py names, in one log line, the pass settings the user set that
    the pipeline does not use (unused_pass_settings), and warns about --compute-type;
  - whisperjav/pipelines/crispasr_pipeline.py runs the speech segmenter with segmenter_settings().

Light: the manifest (standard library only) and the segmenter presets; no ASR import.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from whisperjav.config.segmenter_presets import CRISPASR_DEFAULT_SEGMENTERS, CRISPASR_DEFAULT_SENSITIVITY
from whisperjav.crispasr import manifest
from whisperjav.crispasr.manifest import CRISPASR_PIPELINES, JAPANESE_NAMES

# The scene detector the CrispASR pipelines run when the pass names none (owner W214 2).
CRISPASR_SCENE_DETECTOR = "semantic"

# --- Texts (route B design 12, T11 and T12; W193 5, W203 I2) --------------------------------------

# Usage errors; main.py prints them as "error: <text>" and exits with status 2.
MODEL_PROBLEM = "--pass{n}-model: {why}"
LANGUAGE_PROBLEM = ("--pass{n}-pipeline {pipeline} transcribes Japanese only, and --language is "
                    "{language}. Use --language japanese, or choose another pipeline for pass {n}.")
SEGMENTER_NONE_PROBLEM = ("--pass{n}-speech-segmenter none is not available when --pass{n}-pipeline is "
                          "{pipeline}: it transcribes the VAD groups the speech segmenter finds. Leave the "
                          "flag out to use {default}, or name another speech segmenter.")

# One info line from the pass worker, e.g.
# "Parakeet-CrispASR does not use: --pass1-scene-detector, --pass1-speech-enhancer".
UNUSED_SETTINGS_MESSAGE = "{label} does not use: {flags}"

# One warning from the pass worker, e.g. "Parakeet-CrispASR ignores --compute-type int8: ...".
IGNORED_HARDWARE_MESSAGE = "{label} ignores {flags}: CrispASR's model files come in one number format (8-bit)."

# The model short names in --passN-model's help, e.g.
# "parakeet-crispasr: parakeet-tdt-0.6b-ja (default); cohere-crispasr: cohere-asr-ja-v0.1 (default), cohere-asr-ja".
MODELS_HELP_DEFAULT = " (default)"

# --- The speech segmenter settings (W199 DIR1, W202) ----------------------------------------------

# Which pipeline's presets each CrispASR pipeline takes.
SEGMENTER_STYLE = {"parakeet-crispasr": "fidelity", "cohere-crispasr": "qwen"}
# Parakeet-CrispASR's longest VAD group and longest speech segment, at every sensitivity (W193 CLA3,
# W194 2 "(a)", confirmed by W202 1 "1-a").
PARAKEET_LONGEST_S = 6.0
# The qwen pipeline's own group and pad settings, which win over the segmenter's preset there:
# QwenPipeline's defaults (whisperjav/pipelines/qwen_pipeline.py, segmenter_max_group_duration,
# segmenter_chunk_threshold, segmenter_start_pad_ms, segmenter_end_pad_ms). Kept equal by
# tests/test_crispasr_pipelines.py, which reads them from that file.
QWEN_MAX_GROUP_S = 3.0
QWEN_CHUNK_THRESHOLD_S = 0.3
QWEN_START_PAD_MS = 100
QWEN_END_PAD_MS = 100
QWEN_GENERATOR = "qwen3"  # "qwenasr" (W199 DIR1): Qwen3-ASR, ChronosJAV's default generator


@dataclass(frozen=True)
class SegmenterSettings:
    """What the speech segmenter of one CrispASR pass runs with."""
    backend: str
    sensitivity: str
    max_group_duration_s: Optional[float]
    chunk_threshold_s: Optional[float]
    config: Dict[str, Any] = field(default_factory=dict)  # everything else, as the segmenter factory takes it

    def as_dict(self) -> Dict[str, Any]:
        return {"speech_segmenter": self.backend, "sensitivity": self.sensitivity,
                "max_group_duration_s": self.max_group_duration_s,
                "chunk_threshold_s": self.chunk_threshold_s, **self.config}


def default_segmenter(pipeline: str) -> str:
    return CRISPASR_DEFAULT_SEGMENTERS[pipeline]


def segmenter_settings(pipeline: str, backend: Optional[str], sensitivity: Optional[str], *,
                       vad_threshold: Optional[float] = None,
                       speech_pad_ms: Optional[int] = None) -> SegmenterSettings:
    """The speech segmenter settings of a CrispASR pass, resolved as the named pipeline resolves them.

    parakeet-crispasr, as the fidelity pipeline: the backend's own preset at the sensitivity
        (resolve_segmenter_sensitivity, as fidelity's ensemble pass), with the longest VAD group and
        the longest speech segment 6 s at every sensitivity.
    cohere-crispasr, as the qwen pipeline with Qwen3-ASR: resolve_chronosjav_segmenter_config (the
        backend's preset, ChronosJAV's longest speech segment for TEN and FireRedVAD, its WhisperSeg
        values), then the qwen pipeline's own groups (3.0 s, gap 0.3 s) and pads (100 ms; the end pad
        chronosjav_end_pad_default() when it gives one) on top, as QwenPipeline sets them on its framer.

    ``vad_threshold`` and ``speech_pad_ms`` are --passN-vad-threshold and --passN-speech-pad-ms (owner
    W208: "make the flags work"), applied as those pipelines apply them (pass_worker.py): the threshold,
    clamped to 0-1, wins over the preset in both; the pad is the segmenter's speech_pad_ms for Parakeet,
    as fidelity, only when the backend has that setting (speech_pad_ignored() says when it has not),
    and both the start and the end pad for Cohere, as the qwen pipeline.
    """
    backend = (backend or default_segmenter(pipeline)).strip().lower()
    sensitivity = sensitivity or CRISPASR_DEFAULT_SENSITIVITY[pipeline]
    style = SEGMENTER_STYLE[pipeline]
    user: Dict[str, Any] = {}
    if vad_threshold is not None:
        user["threshold"] = max(0.0, min(1.0, float(vad_threshold)))
    if style == "fidelity":
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity

        if speech_pad_ms is not None and not speech_pad_ignored(pipeline, backend, speech_pad_ms):
            user["speech_pad_ms"] = max(0, int(speech_pad_ms))
        config = dict(resolve_segmenter_sensitivity(backend, sensitivity, {
            **user, "max_group_duration_s": PARAKEET_LONGEST_S, "max_speech_duration_s": PARAKEET_LONGEST_S}))
        max_group = config.pop("max_group_duration_s", PARAKEET_LONGEST_S)
        chunk = config.pop("chunk_threshold_s", None)
        return SegmenterSettings(backend, sensitivity, max_group, chunk, config)
    from whisperjav.config.chronosjav_vad import chronosjav_end_pad_default, resolve_chronosjav_segmenter_config

    config = dict(resolve_chronosjav_segmenter_config(QWEN_GENERATOR, backend, sensitivity, user or None))
    config.pop("max_group_duration_s", None)
    config.pop("chunk_threshold_s", None)
    end_pad = chronosjav_end_pad_default(QWEN_GENERATOR, backend)
    config["start_pad_ms"] = QWEN_START_PAD_MS
    config["end_pad_ms"] = QWEN_END_PAD_MS if end_pad is None else end_pad
    if speech_pad_ms is not None:
        config["start_pad_ms"] = config["end_pad_ms"] = max(0, int(speech_pad_ms))
    return SegmenterSettings(backend, sensitivity, QWEN_MAX_GROUP_S, QWEN_CHUNK_THRESHOLD_S, config)


def speech_pad_ignored(pipeline: str, backend: Optional[str], speech_pad_ms: Optional[int]) -> bool:
    """True when --passN-speech-pad-ms is given for Parakeet-CrispASR with a backend that has no
    speech_pad_ms setting (firered-vad, ten): fidelity ignores it there, with a warning."""
    if speech_pad_ms is None or SEGMENTER_STYLE.get(pipeline) != "fidelity":
        return False
    from whisperjav.config.segmenter_presets import segmenter_accepts

    return not segmenter_accepts((backend or default_segmenter(pipeline)).strip().lower(), "speech_pad_ms")


# --- Usage checks ---------------------------------------------------------------------------------

def crispasr_pass_problem(n: int, pipeline: str, *, model: Optional[str],
                          speech_segmenter: Optional[str], language: Optional[str]) -> Optional[str]:
    """Why pass ``n`` with this CrispASR pipeline cannot run, in plain words; None when it can.

    ``model`` is --passN-model, ``speech_segmenter`` --passN-speech-segmenter and ``language``
    the --language value (japanese, korean, chinese or english); None means the flag was left out.
    """
    language = str(language or "japanese").strip().lower()
    if language not in JAPANESE_NAMES:
        # The models transcribe Japanese only.
        return LANGUAGE_PROBLEM.format(n=n, pipeline=pipeline, language=language)
    try:
        manifest.model_for(pipeline, model)
    except ValueError as why:
        return MODEL_PROBLEM.format(n=n, why=why)
    if speech_segmenter is not None and str(speech_segmenter).strip().lower() == "none":
        return SEGMENTER_NONE_PROBLEM.format(n=n, pipeline=pipeline, default=default_segmenter(pipeline))
    return None


def validate_crispasr_passes(args: Any) -> None:
    """Refuse an Ensemble pass that names a CrispASR pipeline and cannot run. Raises ValueError.

    ``args`` is main.py's parsed command line. Like main.py's other pass checks, it looks at every
    pass that names a CrispASR pipeline, with or without --ensemble.
    """
    for n in (1, 2):
        pipeline = getattr(args, f"pass{n}_pipeline", None)
        if pipeline not in CRISPASR_PIPELINES:
            continue
        problem = crispasr_pass_problem(
            n, pipeline,
            model=getattr(args, f"pass{n}_model", None),
            speech_segmenter=getattr(args, f"pass{n}_speech_segmenter", None),
            language=getattr(args, "language", None),
        )
        if problem:
            raise ValueError(problem)


def unused_pass_settings(pass_config: Mapping[str, Any], pass_number: int) -> List[str]:
    """The flags of the pass settings the user set that a CrispASR pipeline does not use.

    ``pass_config`` is the pass configuration main.py (or the GUI, through main.py) builds. Since
    design 22.9.1 the scene detector, the speech enhancer, enhance-for-vad and --passN-params are
    used, so only the transformers and ChronosJAV settings (--passN-hf-params, --passN-qwen-params)
    can be listed. --compute-type is ignored_hardware_settings'.
    """
    n = pass_number
    unused = []
    for key, flag in (("hf_params", "hf-params"), ("qwen_params", "qwen-params")):
        if pass_config.get(key):
            unused.append(f"--pass{n}-{flag}")
    return unused


def unused_settings_message(label: str, flags: List[str]) -> str:
    """The pass worker's one line naming what a CrispASR pipeline does not use."""
    return UNUSED_SETTINGS_MESSAGE.format(label=label, flags=", ".join(flags))


def ignored_hardware_settings(pass_config: Mapping[str, Any]) -> List[str]:
    """--compute-type as given, when it names anything but auto. --device is used (owner W194 1)."""
    value = pass_config.get("compute_type")
    if value and str(value).strip().lower() != "auto":
        return [f"--compute-type {value}"]
    return []


def ignored_hardware_message(label: str, flags: List[str]) -> str:
    """The pass worker's warning naming the --compute-type value a CrispASR pass ignores."""
    return IGNORED_HARDWARE_MESSAGE.format(label=label, flags=" and ".join(flags))


def models_help() -> str:
    """The model short names of every CrispASR pipeline, for --passN-model's help."""
    parts = []
    for pipeline in CRISPASR_PIPELINES:
        default = manifest.PIPELINE_DEFAULT_MODEL[pipeline]
        names = [name + (MODELS_HELP_DEFAULT if name == default else "")
                 for name in manifest.PIPELINE_MODELS[pipeline]]
        parts.append(f"{pipeline}: {', '.join(names)}")
    return "; ".join(parts)
