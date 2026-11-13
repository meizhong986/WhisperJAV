#!/usr/bin/env python3
"""
Which timestamp mode a ChronosJAV run with Qwen3-ASR or anime-whisper uses when it names none (1.10.1).

ONE decision, used by every place that needs it, so they cannot disagree:

    - whisperjav/main.py                    (standalone --mode qwen, both models)
    - whisperjav/ensemble/pass_worker.py    (Ensemble passes; the GUI's ChronosJAV rows)
    - tests/test_chronosjav_timing_wiring.py

The rule (owner W498 D1, W501 1): the default is the timestamp mode vad_only with no timing model -- no aligner is
loaded and each subtitle takes its VAD group's edges, ChronosJAV's timing of 1.9.5. A run that names an aligner
(--qwen-aligner, or an Ensemble pass's aligner_id / aligner_backend crispasr or qwen3) but no timestamp mode gets
aligner_vad_fallback instead, so the aligner it names is the aligner that runs, as in 1.10.0. The timing model stays
off unless it is named too. A timestamp mode the run names always wins.

The GUI is not affected: a customized ChronosJAV row sends its timestamp mode with its aligner, and an uncustomized
row sends no aligner. The dormant Cohere-in-ChronosJAV is not covered (it keeps 1.10.0's values).

1.10.2 (owner W624 C, W626): which files a ChronosJAV pass uses -- the CrispASR aligner, the PyTorch aligner, the
timing model -- is decided here too, ONE place for:

    - whisperjav/ensemble/pass_worker.py    (the settings merge: prepare_qwen_params, _build_pipeline)
    - whisperjav/pipelines/qwen_pipeline.py (which aligner and timing model it builds)
    - whisperjav/main.py                    (--mode qwen's settings; the start-of-run check, which downloads the
                                             files before any audio is read; the --dump-params block)

pass_start_values + pass_timing give an Ensemble pass's five timing settings, single_pass_timing those of
--mode qwen, and chronosjav_files the files a pipeline built with them uses. The dormant Cohere-in-ChronosJAV goes
through the same merge as before (its own branch in the pass worker is unchanged) and stays outside the start check
(W626 2).

Light: no imports at module level (the CrispASR aligner names are read from crispasr/manifest.py when needed).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

DEFAULT_TIMESTAMP_MODE = "vad_only"          # owner W498 D1
DEFAULT_TIMING_MODEL = "none"                # owner W498 D1
ALIGNER_TIMESTAMP_MODE = "aligner_vad_fallback"   # what a named aligner brings (owner W501 1)
ALIGNER_BACKENDS = ("crispasr", "qwen3")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def aligner_named(aligner_id: Optional[Any] = None, aligner_backend: Optional[Any] = None,
                  use_aligner: Optional[Any] = None) -> bool:
    """True when the run names an aligner to use: an aligner model other than "none", or the aligner backend
    crispasr or qwen3. False when it turns the aligner off (aligner backend "none", use_aligner false)."""
    if use_aligner is False or _text(aligner_backend) == "none":
        return False
    if _text(aligner_backend) in ALIGNER_BACKENDS:
        return True
    return _text(aligner_id) not in ("", "none")


def timestamp_mode_for(named_mode: Optional[str], aligner_is_named: bool) -> str:
    """The run's timestamp mode: the one it names, else aligner_vad_fallback when it names an aligner, else
    vad_only."""
    if named_mode:
        return named_mode
    return ALIGNER_TIMESTAMP_MODE if aligner_is_named else DEFAULT_TIMESTAMP_MODE


def timestamp_mode_for_pass(qwen_params: Optional[dict]) -> Optional[str]:
    """For an Ensemble pass's ChronosJAV values (--passN-qwen-params, both key forms): the mode to set when the pass
    names none, or None when it names one (then its own value stands)."""
    p = qwen_params or {}
    if p.get("timestamp_mode") or p.get("qwen_timestamp_mode"):
        return None
    named = aligner_named(p.get("aligner_id", p.get("qwen_aligner")),
                          p.get("aligner_backend", p.get("qwen_aligner_backend")),
                          p.get("use_aligner", p.get("qwen_use_aligner")))
    return timestamp_mode_for(None, named)


# --- 1.10.2: the files a ChronosJAV pass uses (owner W624 C, W626) -------------------------------------------------

# QwenPipeline's timestamp modes (modules/subtitle_pipeline/types.py TimestampMode); an unknown value is taken as
# aligner_vad_fallback, with a warning, by QwenPipeline.
TIMESTAMP_MODES = ("aligner_interpolation", "aligner_vad_fallback", "aligner_only", "vad_only")
UNKNOWN_MODE_TAKEN_AS = "aligner_vad_fallback"
PYTORCH_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"   # the PyTorch aligner's model ID when the pass names none

# An Ensemble pass's starting values, the pass worker's DEFAULT_QWEN_PARAMS entries: the shared 1.10.0 values (owner
# W254), still taken by the dormant Cohere-in-ChronosJAV; Qwen3-ASR and anime-whisper get vad_only and no timing
# model in pass_timing (W498 D1).
PASS_START = {
    "qwen_timestamps": "word",
    "qwen_aligner": "wav2vec2-ja",            # the CrispASR aligner (design note 2026-10-28)
    "qwen_timing_model": "whisper-small",     # the CrispASR timing model, step 8
    "qwen_timestamp_mode": "aligner_vad_fallback",
}
# The --passN-qwen-params keys that carry them, with the pass worker's names (part of its prepare_qwen_params mapping).
TIMING_KEYS = {
    "timestamps": "qwen_timestamps",
    "aligner_id": "qwen_aligner",
    "use_aligner": "qwen_use_aligner",
    "aligner_backend": "qwen_aligner_backend",
    "timing_model": "qwen_timing_model",
    "timestamp_mode": "qwen_timestamp_mode",
}
PASS_ALIGNER_BACKENDS = ("crispasr", "qwen3", "none")   # the values the pass worker hands to QwenPipeline


def apply_aligner_choice(params: Dict[str, Any], qwen_params: Optional[dict]) -> None:
    """A pass's aligner choice applied, in place, to its qwen_* values (prepare_qwen_params; moved here unchanged).

    use_aligner false or aligner_backend "none" turn the aligner off -- and, unless the pass names them, bring
    vad_only and no step-down (the GUI's triple flip, for qwen_params written by hand too); "crispasr" / "qwen3" take
    that backend, with its default model when the aligner named is not one of its own (an older GUI preset may pair
    "qwen3" with nothing, or "crispasr" with the Hugging Face ID). Absent, the backend follows from the aligner value.
    """
    qp = qwen_params or {}
    if qp.get("use_aligner") is False:
        params["qwen_aligner"] = None
    aligner_backend = qp.get("aligner_backend")
    if aligner_backend == "none" or qp.get("use_aligner") is False:
        params["qwen_aligner"] = None
        params["qwen_timestamps"] = "none"  # the pipeline takes use_aligner from timestamps
        if "timestamp_mode" not in qp and "qwen_timestamp_mode" not in qp:
            params["qwen_timestamp_mode"] = "vad_only"
        if "stepdown" not in qp and "qwen_stepdown" not in qp:
            params["qwen_stepdown"] = False
    elif aligner_backend in ("crispasr", "qwen3"):
        from whisperjav.crispasr.manifest import ALIGNER_MODELS, DEFAULT_ALIGNER
        is_crispasr_name = params.get("qwen_aligner") in ALIGNER_MODELS
        if aligner_backend == "crispasr" and not is_crispasr_name:
            params["qwen_aligner"] = DEFAULT_ALIGNER
        elif aligner_backend == "qwen3" and (is_crispasr_name or not params.get("qwen_aligner")):
            params["qwen_aligner"] = PYTORCH_ALIGNER


def pass_start_values(qwen_params: Optional[dict]) -> Dict[str, Any]:
    """An Ensemble pass's qwen_* timing values, as the pass worker's prepare_qwen_params makes them: PASS_START, the
    pass's own keys in their order (both forms, "aligner_id" and "qwen_aligner"), then apply_aligner_choice."""
    qp = qwen_params or {}
    params = dict(PASS_START)
    names = set(TIMING_KEYS.values())
    for key, value in qp.items():
        if key in TIMING_KEYS:
            params[TIMING_KEYS[key]] = value
        elif key in names:
            params[key] = value
    apply_aligner_choice(params, qp)
    return params


def generator_for_pass(qwen_params: Optional[dict]) -> str:
    """The pass's ASR model as the pass worker takes it: generator_backend or qwen_generator_backend, the later key
    winning (prepare_qwen_params), else qwen3."""
    generator = "qwen3"
    for key, value in (qwen_params or {}).items():
        if key in ("generator_backend", "qwen_generator_backend"):
            generator = value
    return generator


def pass_timing(values: Dict[str, Any], qwen_params: Optional[dict], generator: str) -> Dict[str, Any]:
    """The five timing keyword arguments the pass worker builds an Ensemble pass's QwenPipeline with.

    ``values`` are the pass's qwen_* values (prepare_qwen_params, or pass_start_values); ``qwen_params`` what the pass
    names itself. Qwen3-ASR and anime-whisper then get the 1.10.1 defaults (vad_only unless the pass names a mode or
    an aligner, no timing model unless it names one; W498 D1, W501 1). The dormant Cohere-in-ChronosJAV branch of the
    pass worker sets its own timestamp mode after this (unchanged).
    """
    backend = values.get("qwen_aligner_backend")
    kwargs = {
        "timestamps": values.get("qwen_timestamps", PASS_START["qwen_timestamps"]),
        "aligner_id": values.get("qwen_aligner", PASS_START["qwen_aligner"]),
        "aligner_backend": backend if backend in PASS_ALIGNER_BACKENDS else None,
        "timing_model": values.get("qwen_timing_model", PASS_START["qwen_timing_model"]),
        "timestamp_mode": values.get("qwen_timestamp_mode", PASS_START["qwen_timestamp_mode"]),
    }
    if generator in ("qwen3", "anime-whisper"):
        named = qwen_params or {}
        mode = timestamp_mode_for_pass(named)
        if mode is not None:
            kwargs["timestamp_mode"] = mode
        if "timing_model" not in named and "qwen_timing_model" not in named:
            kwargs["timing_model"] = DEFAULT_TIMING_MODEL
    return kwargs


def single_pass_timing(timestamps: Optional[str], aligner: Optional[str], timing_model: Optional[str],
                       timestamp_mode: Optional[str]) -> Dict[str, Any]:
    """The five timing keyword arguments --mode qwen builds QwenPipeline with, from --qwen-timestamps,
    --qwen-aligner, --qwen-timing-model and --qwen-timestamp-mode (main.process_files_sync)."""
    return {
        "timestamps": timestamps if timestamps is not None else "word",
        # None = not given: the aligner an aligner mode uses is then wav2vec2-ja
        "aligner_id": aligner or PASS_START["qwen_aligner"],
        "aligner_backend": None,          # from the aligner value (aligner_backend_for)
        "timing_model": timing_model if timing_model is not None else DEFAULT_TIMING_MODEL,
        "timestamp_mode": timestamp_mode_for(timestamp_mode, aligner_named(aligner)),
    }


def aligner_backend_for(aligner_backend: Optional[str], aligner_id: Optional[str]) -> str:
    """Which aligner backend runs: the one named, else the one the model value belongs to -- a CrispASR short name
    (manifest.ALIGNER_MODELS) means "crispasr", anything else (a PyTorch model ID) "qwen3", nothing "none"."""
    if aligner_backend:
        return aligner_backend
    if not aligner_id:
        return "none"
    from whisperjav.crispasr.manifest import ALIGNER_MODELS
    return "crispasr" if aligner_id in ALIGNER_MODELS else "qwen3"


@dataclass(frozen=True)
class ChronosFiles:
    """What a ChronosJAV pipeline built with given timing settings uses: the timestamp mode it runs, the aligner
    (backend "crispasr" with a CrispASR short name, or any other backend -- "qwen3" -- with a PyTorch model ID;
    None when it builds none) and the timing model (None when off)."""
    timestamp_mode: str
    aligner_backend: Optional[str]
    aligner: Optional[str]
    timing_model: Optional[str]

    @property
    def uses_files(self) -> bool:
        return bool(self.aligner or self.timing_model)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chronosjav_files(*, timestamps: Optional[str], aligner_id: Optional[str], aligner_backend: Optional[str],
                     timing_model: Optional[str], timestamp_mode: Any) -> ChronosFiles:
    """The rule QwenPipeline builds its aligner and timing model by (owner W624 C): no aligner in vad_only, with the
    aligner backend "none", or with timestamps other than "word"; else the aligner of its backend. The timing model
    whenever one other than "none" is named, at every timestamp mode, with or without an aligner (rule T5)."""
    mode = getattr(timestamp_mode, "value", timestamp_mode)
    if mode not in TIMESTAMP_MODES:
        mode = UNKNOWN_MODE_TAKEN_AS
    backend = aligner_backend_for(aligner_backend, aligner_id)
    if mode == "vad_only" or backend == "none" or timestamps != "word":
        backend, aligner = None, None
    else:
        aligner = aligner_id
    timing = timing_model if (timing_model or "none") != "none" else None
    return ChronosFiles(mode, backend, aligner, timing)
