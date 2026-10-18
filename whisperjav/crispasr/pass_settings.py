"""What an Ensemble pass running a CrispASR pipeline accepts, refuses and does not use (1.10.0).

The CrispASR pipelines (parakeet-crispasr, cohere-crispasr) are Ensemble passes only, with no --mode
value (master plan 2.4, owner W118 2 and 5). Of a pass's settings they use three: the model
(--passN-model, a short name from the manifest), the speech segmenter (--passN-speech-segmenter)
and the source language (--language). Everything else about the run is fixed by the pipeline
(plan 2.2): semantic scenes, no speech enhancer, no sensitivity presets, no custom parameters.

Used by two callers, so the rules have one home:
  - whisperjav/main.py refuses a pass that cannot run (validate_crispasr_passes), as a usage error
    with exit status 2, before any audio is read or anything is downloaded;
  - whisperjav/ensemble/pass_worker.py names, in one log line, the pass settings the user set that
    the pipeline does not use (unused_pass_settings). They are not refused: the GUI and saved
    Ensemble presets carry them for every pipeline.

Light: the manifest (standard library only) and the segmenter presets; no ASR import.
"""

from typing import Any, List, Mapping, Optional

from whisperjav.config.segmenter_presets import CRISPASR_DEFAULT_SEGMENTER
from whisperjav.crispasr import manifest
from whisperjav.crispasr.manifest import CRISPASR_PIPELINES

# The scene detector both CrispASR pipelines run, always (plan 2.2: "fixed at Semantic").
CRISPASR_SCENE_DETECTOR = "semantic"

# --- Texts. DRAFT, not yet approved by the owner: user-visible wording is his (CLAUDE.md). -------

# Usage errors; main.py prints them as "error: <text>" and exits with status 2.
MODEL_PROBLEM = "--pass{n}-model: {why}"
LANGUAGE_PROBLEM = ("--pass{n}-pipeline {pipeline} transcribes Japanese only, and --language is "
                    "{language}. Use --language japanese, or choose another pipeline for pass {n}.")
SEGMENTER_NONE_PROBLEM = ("--pass{n}-speech-segmenter none is not available when --pass{n}-pipeline "
                          "is {pipeline}: its slices follow the speech that the segmenter finds. Leave "
                          "the flag out to use {default}, or name another speech segmenter.")

# One info line from the pass worker, e.g.
# "Parakeet-CrispASR does not use: --pass1-scene-detector, --pass1-speech-enhancer".
UNUSED_SETTINGS_MESSAGE = "{label} does not use: {flags}"

# The model short names in --passN-model's help (the offered ones only), e.g.
# "parakeet-crispasr: parakeet-tdt-0.6b-ja (default); cohere-crispasr: cohere-asr-ja-v0.1, cohere-asr-ja ...".
MODELS_HELP_DEFAULT = " (default)"
MODELS_HELP_NO_DEFAULT = " (no default yet: name one)"


def crispasr_pass_problem(n: int, pipeline: str, *, model: Optional[str],
                          speech_segmenter: Optional[str], language: Optional[str]) -> Optional[str]:
    """Why pass ``n`` with this CrispASR pipeline cannot run, in plain words; None when it can.

    ``model`` is --passN-model, ``speech_segmenter`` --passN-speech-segmenter and ``language``
    the --language value (japanese, korean, chinese or english); None means the flag was left out.
    """
    language = str(language or "japanese").strip().lower()
    if language != "japanese":
        # Q19: the models transcribe Japanese only.
        return LANGUAGE_PROBLEM.format(n=n, pipeline=pipeline, language=language)
    try:
        manifest.model_for(pipeline, model)
    except ValueError as why:
        # An unknown name, or no model named for a pipeline without a default (plan 2.4).
        return MODEL_PROBLEM.format(n=n, why=why)
    if speech_segmenter is not None and str(speech_segmenter).strip().lower() == "none":
        # Q14: the slices follow speech, so there is no "none".
        return SEGMENTER_NONE_PROBLEM.format(n=n, pipeline=pipeline, default=CRISPASR_DEFAULT_SEGMENTER)
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

    ``pass_config`` is the pass configuration main.py (or the GUI, through main.py) builds. A
    setting equal to what the pipeline does anyway -- the semantic scene detector, the speech
    enhancer "none" -- is not listed. Sensitivity is left out: main.py fills it in for every pass
    (apply_ensemble_pass_defaults), so it cannot tell whether the user set it. The enhance-for-vad
    flag is named in its per-pass form even when it came from --enhance-for-vad, which main.py
    folds into every pass.
    """
    n = pass_number
    unused = []
    scene_detector = pass_config.get("scene_detector")
    if scene_detector and str(scene_detector).strip().lower() != CRISPASR_SCENE_DETECTOR:
        unused.append(f"--pass{n}-scene-detector")
    enhancer = pass_config.get("speech_enhancer")
    if enhancer and str(enhancer).split(":", 1)[0].strip().lower() != "none":
        unused.append(f"--pass{n}-speech-enhancer")
    if pass_config.get("enhance_for_vad"):
        unused.append(f"--pass{n}-enhance-for-vad")
    if pass_config.get("vad_threshold") is not None:
        unused.append(f"--pass{n}-vad-threshold")
    if pass_config.get("speech_pad_ms") is not None:
        unused.append(f"--pass{n}-speech-pad-ms")
    for key, flag in (("params", "params"), ("hf_params", "hf-params"), ("qwen_params", "qwen-params")):
        if pass_config.get(key):
            unused.append(f"--pass{n}-{flag}")
    return unused


def unused_settings_message(label: str, flags: List[str]) -> str:
    """The pass worker's one line naming what a CrispASR pipeline does not use."""
    return UNUSED_SETTINGS_MESSAGE.format(label=label, flags=", ".join(flags))


def models_help() -> str:
    """The model short names of every CrispASR pipeline, for --passN-model's help."""
    parts = []
    for pipeline in CRISPASR_PIPELINES:
        default = manifest.PIPELINE_DEFAULT_MODEL[pipeline]
        names = [name + (MODELS_HELP_DEFAULT if name == default else "")
                 for name in manifest.PIPELINE_MODELS[pipeline]]
        parts.append(f"{pipeline}: {', '.join(names)}" + ("" if default else MODELS_HELP_NO_DEFAULT))
    return "; ".join(parts)
