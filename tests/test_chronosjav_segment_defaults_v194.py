"""v1.9.4 ChronosJAV longest-segment defaults (owner, 2026-10-05) and a Customize dialog that shows what runs.

With the WhisperSeg speech segmenter, the longest single segment (the WhisperSeg
`max_speech_duration_s`, the cap that ends most subtitle lines) is the SAME at every
sensitivity, per ASR model:

    Qwen3-ASR      4.0 s   (was 6 / 5 / 4 s from the WhisperSeg YAML presets)
    anime-whisper  4.0 s   (was 6 / 5 / 4 s; option B), and grow floor 0.15 on aggressive
                           (the only row that runs the offline decoder; was 0.05)

With TEN and FireRedVAD (owner, 2026-10-08) the longest segment is also 4.0 s at every
sensitivity for both models (their presets: 6 / 5 / 4 s and 7 / 6 / 5 s); the rest of their
presets is unchanged, and other pipelines keep the presets.

With WhisperSeg, Qwen3-ASR's end pad is 0 ms at every sensitivity (owner, 2026-10-09; was the
pipeline's 100 ms, which other segmenters keep).

Group cap and group gap are unchanged. Evidence and limits:
docs/measurements/v1.9.4/MEASUREMENTS_v194_REQ1_REQ2.md section 2.

The decision "which ChronosJAV defaults apply" lives in config/chronosjav_vad.py and is
called by main.py (--mode qwen), ensemble/pass_worker.py and the GUI dialog
(webview_gui/api.py get_qwen_schema); these tests call the same function, so the
WhisperSeg-only rule itself is exercised.
"""

import inspect
from pathlib import Path

import pytest

from whisperjav.config.anime_whisper_vad import anime_whisperseg_defaults
from whisperjav.config.chronosjav_vad import (
    apply_chronosjav_segmenter_defaults,
    chronosjav_end_pad_default,
    resolve_chronosjav_segmenter_config,
)
from whisperjav.config.qwen3_whisperseg_vad import QWEN3_WHISPERSEG_DEFAULTS
from whisperjav.config.segmenter_presets import resolve_segmenter_sensitivity

SENSITIVITIES = ("conservative", "balanced", "aggressive")
REPO = Path(__file__).resolve().parents[1]

# (sensitivity, longest segment, group cap, group gap, decoder, grow floor)
ANIME = [
    ("conservative", 4.0, 3.0, 0.3, None, None),
    ("balanced", 4.0, 2.5, 0.25, None, None),
    ("aggressive", 4.0, 2.0, 0.2, "offline", 0.15),
]


def _resolve(generator, sensitivity, segmenter="whisperseg", user=None):
    return resolve_chronosjav_segmenter_config(generator, sensitivity=sensitivity, segmenter=segmenter,
                                               overrides=user)


# ---- the values a run resolves --------------------------------------------------------------------

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
    from whisperjav.pipelines.qwen_pipeline import QwenPipeline
    params = inspect.signature(QwenPipeline.__init__).parameters
    assert params["segmenter_max_group_duration"].default == 3.0
    assert params["segmenter_chunk_threshold"].default == 0.3


@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_a_users_own_value_still_wins(generator):
    cfg = _resolve(generator, "balanced", user={"max_speech_duration_s": 5.5, "threshold": 0.4})
    assert cfg["max_speech_duration_s"] == 5.5
    assert cfg["threshold"] == 0.4


@pytest.mark.parametrize("segmenter", ["silero-v6.2", "silero", "whisper-vad"])
@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_other_segmenters_keep_their_own_presets(generator, segmenter):
    # The gate itself: no ChronosJAV value is injected for these segmenters.
    assert apply_chronosjav_segmenter_defaults({}, generator, segmenter, "balanced") == {}
    assert _resolve(generator, "balanced", segmenter) == resolve_segmenter_sensitivity(segmenter, "balanced", None)


# TEN and FireRedVAD (owner, 2026-10-08): longest segment 4 s at every sensitivity, nothing else changed.
@pytest.mark.parametrize("sens", SENSITIVITIES)
@pytest.mark.parametrize("segmenter", ["ten", "firered-vad"])
@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_ten_and_firered_longest_segment_is_4s(generator, segmenter, sens):
    assert apply_chronosjav_segmenter_defaults({}, generator, segmenter, sens) == {"max_speech_duration_s": 4.0}
    cfg = _resolve(generator, sens, segmenter)
    preset = resolve_segmenter_sensitivity(segmenter, sens, None)
    assert cfg["max_speech_duration_s"] == 4.0
    assert {k: v for k, v in cfg.items() if k != "max_speech_duration_s"} == \
           {k: v for k, v in preset.items() if k != "max_speech_duration_s"}


@pytest.mark.parametrize("segmenter", ["ten", "firered-vad"])
def test_ten_and_firered_presets_untouched_elsewhere(segmenter):
    # Other pipelines read the presets directly; Cohere is not a ChronosJAV model here.
    for sens, expected in zip(SENSITIVITIES, (6, 5, 4) if segmenter == "ten" else (7, 6, 5)):
        assert resolve_segmenter_sensitivity(segmenter, sens, None)["max_speech_duration_s"] == expected
        assert _resolve("cohere", sens, segmenter)["max_speech_duration_s"] == expected


@pytest.mark.parametrize("segmenter", ["ten", "firered-vad"])
def test_ten_and_firered_users_own_value_still_wins(segmenter):
    assert _resolve("qwen3", "conservative", segmenter, user={"max_speech_duration_s": 6.5})[
        "max_speech_duration_s"] == 6.5


def test_cohere_keeps_the_yaml_steps():
    for sens, expected in zip(SENSITIVITIES, (6, 5, 4)):
        assert _resolve("cohere", sens)["max_speech_duration_s"] == expected


def test_single_source_values():
    assert QWEN3_WHISPERSEG_DEFAULTS == {"threshold": 0.25, "max_speech_duration_s": 4.0}


@pytest.mark.parametrize("path", ["whisperjav/main.py", "whisperjav/ensemble/pass_worker.py"])
def test_both_entry_points_use_the_shared_decision(path):
    src = (REPO / path).read_text(encoding="utf-8")
    assert "apply_chronosjav_segmenter_defaults(" in src
    assert "apply_qwen3_segmenter_defaults(" not in src      # no private copy of the rule left
    assert "apply_anime_segmenter_defaults(" not in src


# ---- the Customize dialog shows (and Apply sends) what the run uses -------------------------------

def _dialog(sens, generator, segmenter="whisperseg"):
    from whisperjav.webview_gui.api import WhisperJAVAPI
    api = WhisperJAVAPI.__new__(WhisperJAVAPI)
    return WhisperJAVAPI.get_qwen_schema(api, sens, generator, segmenter)["schema"]["audio"]


@pytest.mark.parametrize("sens", SENSITIVITIES)
@pytest.mark.parametrize("generator", ["qwen3", "anime-whisper"])
def test_dialog_whisperseg_values_are_the_run_values(sens, generator):
    audio = _dialog(sens, generator)
    runs = _resolve(generator, sens)
    assert audio["max_speech_duration"]["default"] == runs["max_speech_duration_s"]
    assert audio["vad_threshold"]["default"] == runs["threshold"]
    assert audio["vad_decoder"]["default"] == (runs.get("segmentation_decoder") or "hysteresis")
    assert not any("(default)" in o["label"] for o in audio["vad_decoder"]["options"])
    if generator == "anime-whisper" and sens == "aggressive":
        assert audio["vad_grow_floor"]["default"] == 0.15


@pytest.mark.parametrize("sens", SENSITIVITIES)
def test_dialog_for_ten_shows_tens_own_values_and_no_whisperseg_levers(sens):
    # The GUI's default pass 2 is Qwen3-ASR with TEN.
    audio = _dialog(sens, "qwen3", "ten")
    runs = resolve_segmenter_sensitivity("ten", sens, None)
    assert audio["vad_threshold"]["default"] == runs["threshold"]
    assert audio["max_speech_duration"]["default"] == 4.0          # ChronosJAV longest segment (2026-10-08)
    for lever in ("vad_decoder", "vad_grow_floor", "vad_gap_merge_ms"):
        assert lever not in audio


def test_dialog_hides_controls_a_segmenter_does_not_have():
    silero = _dialog("balanced", "qwen3", "silero")          # Silero v4.0 ignores the longest-segment cap
    assert "max_speech_duration" not in silero and "vad_decoder" not in silero
    none = _dialog("balanced", "qwen3", "none")
    for key in ("vad_threshold", "max_speech_duration", "vad_decoder", "vad_grow_floor", "vad_gap_merge_ms"):
        assert key not in none


def test_dialog_anime_pipeline_values_follow_the_table():
    for sens in SENSITIVITIES:
        audio, row = _dialog(sens, "anime-whisper"), anime_whisperseg_defaults(sens)
        assert audio["chunk_threshold_ms"]["default"] == int(round(row["chunk_threshold_s"] * 1000))
        assert audio["max_group_duration"]["default"] == row["max_group_duration_s"]
        assert audio["vad_start_pad"]["default"] == row["start_pad_ms"]
        assert audio["vad_end_pad"]["default"] == row["end_pad_ms"]


def test_dialog_js_takes_segmenter_values_from_the_schema():
    js = (REPO / "whisperjav/webview_gui/assets/app.js").read_text(encoding="utf-8")
    assert "passState.speechSegmenter || 'whisperseg')" in js          # the segmenter is sent
    assert js.count("QwenManager.applySegmenterDefaults(") == 2         # open and Reset
    assert "defaults.chunk_threshold_ms = 300;" not in js               # no fixed anime values in Reset


# ---- end pad: Qwen3-ASR with WhisperSeg 0 ms (owner, 2026-10-09) ----------------------------------

@pytest.mark.parametrize("generator,segmenter,expected", [
    ("qwen3", "whisperseg", 0), ("qwen3", None, 0), ("qwen3", " WhisperSeg ", 0),
    ("qwen3", "ten", None), ("qwen3", "firered-vad", None), ("qwen3", "silero-v6.2", None), ("qwen3", "none", None),
    ("anime-whisper", "whisperseg", None), ("cohere", "whisperseg", None)])
def test_end_pad_default(generator, segmenter, expected):
    assert chronosjav_end_pad_default(generator, segmenter) == expected


@pytest.mark.parametrize("sens", SENSITIVITIES)
def test_dialog_end_pad_is_the_run_value(sens):
    assert _dialog(sens, "qwen3")["vad_end_pad"]["default"] == 0
    assert _dialog(sens, "qwen3", "ten")["vad_end_pad"]["default"] == 100
    assert _dialog(sens, "anime-whisper")["vad_end_pad"]["default"] == anime_whisperseg_defaults(sens)["end_pad_ms"]


def test_both_entry_points_apply_the_end_pad_default_unless_the_user_set_one():
    main = (REPO / "whisperjav/main.py").read_text(encoding="utf-8")
    assert "chronosjav_end_pad_default(_gen_backend, _qwen_segmenter)" in main
    i = main.index("chronosjav_end_pad_default(_gen_backend, _qwen_segmenter)")
    block = main[i:i + 400]
    assert "_vad_end_pad is None and _vad_pad_legacy is None" in block   # args, so abbreviations count
    assert 'qwen_kwargs["segmenter_end_pad_ms"] = _end_pad_default' in block
    # it comes after the user's pad flags are read, and the flags gate it
    assert main.index("_vad_end_pad = getattr(args, 'qwen_vad_end_pad', None)") < i
    worker = (REPO / "whisperjav/ensemble/pass_worker.py").read_text(encoding="utf-8")
    j = worker.index('chronosjav_end_pad_default(_gen_backend, qwen_pipeline_params["speech_segmenter"])')
    block = worker[j:j + 300]
    assert '"vad_padding" not in _user_qwen and "vad_end_pad" not in _user_qwen' in block
    # the user's GUI / qwen-params / speech_pad_ms values are applied after it, so they win
    assert worker.index('_end_pad = qwen_defaults.get("qwen_vad_end_pad")') > j
    assert worker.index('_legacy_pad = pass_config.get("speech_pad_ms")') > j
