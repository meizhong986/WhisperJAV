"""What an Ensemble pass running a CrispASR pipeline accepts, refuses and uses (1.10.0; 1.10.1).

The CrispASR pipelines (manifest.CRISPASR_PIPELINES: parakeet-crispasr, cohere-crispasr and, since 1.10.1,
reazonspeech-crispasr, kotoba-crispasr, whisper-turbo-crispasr, whisper-large-crispasr, moss-crispasr) are
Ensemble passes only, with no --mode value (owner W118 2). Of a pass's settings they use: the model (--passN-model, a short name from the
manifest), the speech segmenter (--passN-speech-segmenter), the sensitivity (--passN-sensitivity),
--passN-vad-threshold and --passN-speech-pad-ms (owner W208), the source language (--language),
--device, and since design 22.9.1 (owner W212 D3, W214 2, W232 8) the scene detector
(--passN-scene-detector, semantic by default), the speech enhancer (--passN-speech-enhancer),
--passN-enhance-for-vad and the Customize values of the Segmenter and Scene tabs (--passN-params), and since
1.10.1 --passN-word-timestamps (owner W411-W414; used by these pipelines only).
Only --passN-hf-params and --passN-qwen-params, the transformers and ChronosJAV settings, are not used.

The speech segmenter settings follow the pipelines the owner named (W199 DIR1: "For cohere, uses
similar presets and settings as qwenasr. And for prakeet use similar settings and presets as fidelity
whisper pipeline"; W202): ``segmenter_settings()`` resolves them through those pipelines' own functions.
The 1.10.1 pipelines (W401, pre-approved choices of step 2) take Cohere's settings, except ReazonSpeech-CrispASR,
whose model is a transducer like Parakeet's and takes Parakeet's, and MOSS-CrispASR, which has its own: FireRedVAD
VAD groups that end at a silence over 5 s and are at most 240 s, in semantic scenes of 90-240 s (owner W482; before it
Silero VAD v6.2 with 90-120 s stretches in silero scenes of up to 420 s, W425, W426).

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

# The scene detector the CrispASR pipelines run when the pass names none (owner W214 2), the shortest semantic scene
# (ChronosJAV's 12 s, W202 5d) and the longest scene.
# MOSS-CrispASR (owner W482; docs/plans/design/2026-11-03-moss-default-semantic-firered.md): semantic scenes of 90 to
# 240 s. The 90 s is semantic's real merge floor (shorter pieces join a neighbour; no audio is dropped); auditok and
# silero keep their own minimum. The 240 s is the longest scene the pass asks for, whatever scene detector it runs;
# semantic can pass it when its final clean-up joins a piece shorter than 90 s to a neighbour (it logs a warning). The
# VAD groups, what MOSS hears, stay at most 240 s (MOSS_MAX_GROUP_S), except with the speech segmenter none. Before W482:
# silero with a 420 s maximum (W426 4), whose first pass ends a scene at a short silence, so a short scene gave MOSS a
# short call, and audio between its scenes reached no scene (the Netflix clip, 2026-11-03: 25.9 s of 397 s).
CRISPASR_SCENE_DETECTOR = "semantic"
CRISPASR_SCENE_MIN_S = 12
CRISPASR_SCENE_MAX_S = 48
SCENE_DETECTOR_FOR = {"moss-crispasr": "semantic"}
SCENE_MIN_S_FOR = {"moss-crispasr": 90}
SCENE_MAX_S_FOR = {"moss-crispasr": 240}


def scene_detector_for(pipeline: str) -> str:
    """The scene detector a CrispASR pass runs when it names none."""
    return SCENE_DETECTOR_FOR.get(pipeline, CRISPASR_SCENE_DETECTOR)


def scene_min_s_for(pipeline: str) -> float:
    """The shortest semantic scene of a CrispASR pass (auditok and silero keep their own minimum)."""
    return SCENE_MIN_S_FOR.get(pipeline, CRISPASR_SCENE_MIN_S)


def scene_max_s_for(pipeline: str) -> float:
    """The longest scene of a CrispASR pass (semantic, auditok and silero alike)."""
    return SCENE_MAX_S_FOR.get(pipeline, CRISPASR_SCENE_MAX_S)

# --- Texts (route B design 12, T11 and T12; W193 5, W203 I2) --------------------------------------

# Usage errors; main.py prints them as "error: <text>" and exits with status 2.
MODEL_PROBLEM = "--pass{n}-model: {why}"
LANGUAGE_PROBLEM = ("--pass{n}-pipeline {pipeline} transcribes Japanese only, and --language is "
                    "{language}. Use --language japanese, or choose another pipeline for pass {n}.")

# One info line from the pass worker, e.g.
# "Parakeet-CrispASR does not use: --pass1-scene-detector, --pass1-speech-enhancer".
UNUSED_SETTINGS_MESSAGE = "{label} does not use: {flags}"

# One warning from the pass worker, e.g. "Parakeet-CrispASR ignores --compute-type int8: ...".
IGNORED_HARDWARE_MESSAGE = ("{label} ignores {flags}: each model file has its own number format (f16, q8_0, ...); "
                            "choose the file with --passN-model.")

# The model short names in --passN-model's help, e.g.
# "parakeet-crispasr: parakeet-tdt-0.6b-ja (default); cohere-crispasr: cohere-asr-ja-v0.1 (default), cohere-asr-ja".
MODELS_HELP_DEFAULT = " (default)"

# --- The speech segmenter settings (W199 DIR1, W202) ----------------------------------------------

# Which pipeline's presets each CrispASR pipeline takes.
SEGMENTER_STYLE = {"parakeet-crispasr": "fidelity", "cohere-crispasr": "qwen",
                   "reazonspeech-crispasr": "fidelity", "kotoba-crispasr": "qwen", "whisper-turbo-crispasr": "qwen",
                   "whisper-large-crispasr": "qwen", "moss-crispasr": "moss"}
# MOSS-CrispASR's VAD groups, what MOSS hears in one call (owner W482): speech segments less than 5 s apart are one
# group; a silence longer than 5 s marks a change of sequence and ends the group; a group is at most 240 s; a short
# group followed by a long silence stays alone (no shortest group). The same at every sensitivity (W428 I1 2); the
# sensitivity picks the segmenter's other values. --passN-vad-threshold and the Customize Segmenter tab change them for
# one pass. MOSS_THRESHOLD, a Silero speech probability (W426 2), applies only when the user chooses a Silero segmenter
# for MOSS; the default segmenter is FireRedVAD with its own threshold (CRISPASR_DEFAULT_SEGMENTERS, W482).
# Before W482 (W426): Silero v6.2 at 0.7, groups of 70-120 s.
MOSS_THRESHOLD = 0.7
MOSS_CHUNK_THRESHOLD_S = 5.0
MOSS_MAX_GROUP_S = 240.0
MOSS_MIN_GROUP_S = 0.0   # off: the shortest-group join (W426 3) is not used by the default
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


# The speech segmenter "none" (owner W457, W458): no speech detection; each scene goes to the model whole, as one VAD
# group, so the scene detector's longest scene is the longest call. No group length, no gap, no settings.
NO_SEGMENTER = "none"


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

    moss-crispasr (owner W482; before it W425, W426, W428 I1 2): the backend's own preset at the sensitivity, then
        MOSS's group values (MOSS_CHUNK_THRESHOLD_S, MOSS_MAX_GROUP_S, MOSS_MIN_GROUP_S, and MOSS_THRESHOLD for a
        Silero segmenter) at every sensitivity. A shortest group, when the user sets one, works with the segmenters
        that use the shared grouping (silero-v6.2, ten, firered-vad, whisperseg).
        The longest speech segment stays the preset's: a speech segment cut there is followed at once by the next,
        which the grouping (gaps up to 5 s) puts into the same group. --passN-vad-threshold wins over the threshold;
        --passN-speech-pad-ms is the segmenter's speech_pad_ms where the backend has that setting.
    none (owner W457, W458): no segmenter settings at all; the pipeline sends each scene whole.
    """
    backend = (backend or default_segmenter(pipeline)).strip().lower()
    sensitivity = sensitivity or CRISPASR_DEFAULT_SENSITIVITY[pipeline]
    if backend == NO_SEGMENTER:
        return SegmenterSettings(NO_SEGMENTER, sensitivity, None, None, {})
    style = SEGMENTER_STYLE[pipeline]
    user: Dict[str, Any] = {}
    if vad_threshold is not None:
        user["threshold"] = max(0.0, min(1.0, float(vad_threshold)))
    if style == "moss":
        from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity, segmenter_accepts

        moss = {"max_group_duration_s": MOSS_MAX_GROUP_S, "chunk_threshold_s": MOSS_CHUNK_THRESHOLD_S,
                "min_group_duration_s": MOSS_MIN_GROUP_S}
        if backend.startswith("silero"):
            # 0.7 is a Silero speech probability (W425, W426 2); other segmenters keep their own threshold.
            moss["threshold"] = MOSS_THRESHOLD
        if speech_pad_ms is not None and segmenter_accepts(backend, "speech_pad_ms"):
            user["speech_pad_ms"] = max(0, int(speech_pad_ms))
        config = dict(resolve_segmenter_sensitivity(backend, sensitivity, {**moss, **user}))
        max_group = config.pop("max_group_duration_s", MOSS_MAX_GROUP_S)
        chunk = config.pop("chunk_threshold_s", MOSS_CHUNK_THRESHOLD_S)
        return SegmenterSettings(backend, sensitivity, max_group, chunk, config)
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
    # The speech segmenter "none" is accepted (owner W457, W458; it replaced route B's refusal, Q14): each scene goes
    # to the model whole (segmenter_settings, NO_SEGMENTER).
    return None


def validate_crispasr_passes(args: Any) -> None:
    """Refuse an Ensemble pass that names a CrispASR pipeline and cannot run. Raises ValueError.

    ``args`` is main.py's parsed command line. Like main.py's other pass checks, it looks at every
    pass that names a CrispASR pipeline, with or without --ensemble.
    """
    for n in (1, 2):
        pipeline = getattr(args, f"pass{n}_pipeline", None)
        problem = rscn_problem(n, pipeline, getattr(args, f"pass{n}_scene_detector", None),
                               getattr(args, f"pass{n}_params", None))
        if problem:
            raise ValueError(problem)
        if pipeline not in CRISPASR_PIPELINES:
            continue
        problem = crispasr_pass_problem(
            n, pipeline,
            model=getattr(args, f"pass{n}_model", None),
            speech_segmenter=getattr(args, f"pass{n}_speech_segmenter", None),
            language=getattr(args, "language", None),
        ) or model_params_problem(n, pipeline, getattr(args, f"pass{n}_params", None))
        if problem:
            raise ValueError(problem)


# R-SCN (owner W448-W452; docs/plans/design/2026-11-02-rscn-design.md): built for MOSS-CrispASR, but not offered in
# 1.10.1 (owner W482 Q3): rscn is not among the --passN-scene-detector choices (main.py) nor in the GUI (app.js
# rscnPipelines, kept equal to this by tests/test_gui_rscn.py). The code stays, unused, until the new scene detector
# R replaces it (W481). Its settings would be the rscn_ keys of --passN-params.
RSCN_PIPELINES = ()
RSCN_PIPELINE_PROBLEM = ("--pass{n}-scene-detector rscn is available only when --pass{n}-pipeline is "
                         "{pipelines}; pass {n} is {pipeline}.")
RSCN_NOT_OFFERED = "--pass{n}-scene-detector rscn is not offered in this version."
RSCN_PARAMS_PROBLEM = "--pass{n}-params: {why}"


def rscn_problem(n: int, pipeline: Optional[str], scene_detector: Optional[str],
                 params_json: Optional[str]) -> Optional[str]:
    """Why pass ``n`` cannot run the rscn scene detector, in plain words; None when it can or does not name it.
    The rscn_ settings in --passN-params are checked here (a JSON error is left to main.py's own check)."""
    if str(scene_detector or "").strip().lower() != "rscn" or not pipeline:
        return None
    if not RSCN_PIPELINES:
        return RSCN_NOT_OFFERED.format(n=n)
    if pipeline not in RSCN_PIPELINES:
        return RSCN_PIPELINE_PROBLEM.format(n=n, pipeline=pipeline, pipelines=", ".join(RSCN_PIPELINES))
    if not params_json:
        return None
    import json

    try:
        params = json.loads(params_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(params, dict):
        return None
    from whisperjav.modules.scene_detection_backends.rscn.settings import RSCNSettings

    from whisperjav.modules.scene_detection_backends.rscn.speech import known_backend

    try:
        settings = RSCNSettings.for_pipeline(pipeline, params)
    except ValueError as why:
        return RSCN_PARAMS_PROBLEM.format(n=n, why=why)
    if not known_backend(settings.speech_measure_name):
        return RSCN_PARAMS_PROBLEM.format(n=n, why=f"rscn_speech_measure '{settings.speech_measure}' is not a "
                                                f"speech segmenter backend; name one (firered-vad, silero-v6.2, "
                                                f"ten, ...) or none")
    return None


MODEL_PARAMS_PROBLEM = "--pass{n}-params: {why}"


def model_params_problem(n: int, pipeline: str, params_json: Optional[str]) -> Optional[str]:
    """Why pass ``n``'s model settings in --passN-params cannot be used (MOSS-CrispASR's MOSS settings,
    crispasr/moss_prompt.py), in plain words; None when they can, or when the pipeline has none. A JSON error is
    left to main.py's own check of --passN-params."""
    if pipeline != "moss-crispasr" or not params_json:
        return None
    import json

    from whisperjav.crispasr.moss_prompt import MossSettings

    try:
        params = json.loads(params_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(params, dict):
        return None
    try:
        MossSettings.from_params(params)
    except ValueError as why:
        return MODEL_PARAMS_PROBLEM.format(n=n, why=why)
    return None


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


def unused_by_other_pipelines(pass_config: Mapping[str, Any], pass_number: int) -> List[str]:
    """The CrispASR-only flags the user set on a pass whose pipeline is not a CrispASR pipeline.

    Today --passN-word-timestamps (1.10.1, Part B; owner W413): only the CrispASR pipelines use it; the pass
    worker names it in the same "does not use" line (unused_settings_message) for every other pipeline.
    """
    if pass_config.get("pipeline") in CRISPASR_PIPELINES:
        return []
    if pass_config.get("word_timestamps"):
        return [f"--pass{pass_number}-word-timestamps"]
    return []


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
