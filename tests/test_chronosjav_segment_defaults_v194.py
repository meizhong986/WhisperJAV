"""v1.9.4 ChronosJAV longest-segment defaults (owner, 2026-10-05).

With the WhisperSeg speech segmenter, the longest single segment (the WhisperSeg
`max_speech_duration_s`, the cap that ends most subtitle lines) is the SAME at every
sensitivity, per ASR model:

    Qwen3-ASR      4.0 s   (was 6 / 5 / 4 s from the WhisperSeg YAML presets)
    anime-whisper  3.0 s   (was 6 / 5 / 4 s), and grow floor 0.15 on aggressive
                           (the only row that runs the offline decoder; was 0.05)

Group cap and group gap are unchanged. Evidence and limits:
docs/plans/MEASUREMENTS_v194_REQ1_REQ2.md section 2.3.

These tests resolve the values through the same functions the two entry points call
(main.py standalone --mode qwen, ensemble/pass_worker.py), check that a user's own
value still wins, that nothing outside ChronosJAV + WhisperSeg changed, and that the
GUI Customize dialog shows the values that will run.
"""

import pytest

from whisperjav.config.anime_whisper_vad import (
    anime_whisperseg_defaults,
    apply_anime_segmenter_defaults,
)
from whisperjav.config.qwen3_whisperseg_vad import (
    QWEN3_WHISPERSEG_DEFAULTS,
    apply_qwen3_segmenter_defaults,
)
from whisperjav.ensemble.pass_worker import resolve_qwen_sensitivity

SENSITIVITIES = ("conservative", "balanced", "aggressive")

# (sensitivity, longest segment, group cap, group gap, decoder, grow floor)
ANIME = [
    ("conservative", 3.0, 3.0, 0.3, None, None),
    ("balanced", 3.0, 2.5, 0.25, None, None),
    ("aggressive", 3.0, 2.0, 0.2, "offline", 0.15),
]


def _resolve(generator, sensitivity, user=None):
    """What a run resolves for a qwen-pipeline pass using WhisperSeg."""
    overrides = dict(user or {})
    if generator == "anime-whisper":
        apply_anime_segmenter_defaults(overrides, sensitivity)
    else:
        apply_qwen3_segmenter_defaults(overrides)
    return resolve_qwen_sensitivity("whisperseg", sensitivity, overrides)


@pytest.mark.parametrize("sens", SENSITIVITIES)
def test_qwen3_longest_segment_is_4s_at_every_sensitivity(sens):
    cfg = _resolve("qwen3", sens)
    assert cfg["max_speech_duration_s"] == 4.0
    assert cfg["threshold"] == 0.25


@pytest.mark.parametrize("sens,max_s,group_cap,group_gap,decoder,floor", ANIME)
def test_anime_whisper_values_at_every_sensitivity(sens, max_s, group_cap, group_gap, decoder, floor):
    cfg = _resolve("anime-whisper", sens)
    assert cfg["max_speech_duration_s"] == max_s
    assert cfg.get("segmentation_decoder") == decoder
    if floor is not None:
        assert cfg["grow_floor"] == floor
    row = anime_whisperseg_defaults(sens)
    assert row["max_group_duration_s"] == group_cap     # group settings unchanged
    assert row["chunk_threshold_s"] == group_gap


def test_qwen3_group_settings_unchanged():
    import inspect
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline
    params = inspect.signature(QwenPipeline.__init__).parameters
    assert params["segmenter_max_group_duration"].default == 3.0
    assert params["segmenter_chunk_threshold"].default == 0.3


@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_a_users_own_value_still_wins(generator):
    cfg = _resolve(generator, "balanced", user={"max_speech_duration_s": 5.5, "threshold": 0.4})
    assert cfg["max_speech_duration_s"] == 5.5
    assert cfg["threshold"] == 0.4


def test_nothing_outside_chronosjav_whisperseg_changed():
    # WhisperSeg with no ChronosJAV injection (Cohere, other pipelines) keeps the YAML steps.
    plain = {s: resolve_qwen_sensitivity("whisperseg", s, None)["max_speech_duration_s"] for s in SENSITIVITIES}
    assert plain == {"conservative": 6, "balanced": 5, "aggressive": 4}
    # Another segmenter keeps its own preset.
    assert resolve_qwen_sensitivity("ten", "balanced", None).get("max_speech_duration_s") == 5


def test_single_source_values():
    assert QWEN3_WHISPERSEG_DEFAULTS == {"threshold": 0.25, "max_speech_duration_s": 4.0}


@pytest.mark.parametrize("sens", SENSITIVITIES)
@pytest.mark.parametrize("generator,expected", [("qwen3", 4.0), ("anime-whisper", 3.0)])
def test_gui_customize_dialog_shows_the_value_that_runs(sens, generator, expected):
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    schema = WhisperJAVAPI.get_qwen_schema(api, sens, generator)
    audio = schema["schema"]["audio"]
    assert audio["max_speech_duration"]["default"] == expected
    assert expected == _resolve(generator, sens)["max_speech_duration_s"]
    if generator == "anime-whisper" and sens == "aggressive":
        assert audio["vad_grow_floor"]["default"] == 0.15


@pytest.mark.parametrize("sens", SENSITIVITIES)
@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_gui_customize_dialog_decoder_is_the_one_that_runs(sens, generator):
    """Apply in Customize sends every shown value back as the user's choice, so the
    decoder shown must be the one the run uses: hysteresis (WhisperSeg's own default)
    unless the anime-whisper aggressive row pins "offline"."""
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    shown = WhisperJAVAPI.get_qwen_schema(api, sens, generator)["schema"]["audio"]["vad_decoder"]
    runs = _resolve(generator, sens).get("segmentation_decoder") or "hysteresis"
    assert shown["default"] == runs
    assert not any("(default)" in o["label"] for o in shown["options"])
