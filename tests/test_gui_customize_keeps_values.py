"""G1 (owner W491, step 4): Customize Apply keeps the values the user did not touch.

The GUI sends every control's value on Apply, and a range slider holds only min + k * step inside [min, max]: a value
outside the range is clamped, a value between steps is moved to the nearest step, and either is then sent as the
user's choice (seen in #499's log: gap 0.5 s instead of 0.3 s, scenes 60 s instead of 48 s). This test opens every
CrispASR and ChronosJAV Customize window the way app.js does (same API calls; openCrispasrCustomize seeds the controls
with the pass's own values, openQwenCustomize with each control's default) and checks that each slider and dropdown
holds the value the run uses, for every pipeline, sensitivity, speech segmenter and scene detector. Light: no window,
no model.
"""
from decimal import Decimal, ROUND_FLOOR

import pytest

from whisperjav.crispasr.manifest import CRISPASR_PIPELINES
from whisperjav.webview_gui.api import WhisperJAVAPI

SENSITIVITIES = ("conservative", "balanced", "aggressive")
SCENE_DETECTORS = ("semantic", "auditok", "silero")


def held_by(schema, value):
    """Whether the control can hold ``value`` exactly, as app.js renders it (generateSchemaControl)."""
    kind = schema.get("type")
    if kind == "slider":
        lo, hi = Decimal(str(schema.get("min") or 0)), Decimal(str(schema.get("max") or 1))
        step = Decimal(str(schema.get("step") or "0.01"))
        v = Decimal(str(value))
        top = lo + ((hi - lo) / step).to_integral_value(rounding=ROUND_FLOOR) * step
        return lo <= v <= top and ((v - lo) / step) == ((v - lo) / step).to_integral_value()
    if kind == "dropdown":
        options = [str(o["value"]) if isinstance(o, dict) else str(o) for o in schema.get("options") or []]
        return str(value) in options
    return True


@pytest.fixture(scope="module")
def api():
    return WhisperJAVAPI()


@pytest.fixture(scope="module")
def segmenters(api):
    return [b["name"] if isinstance(b, dict) else b for b in api.get_speech_segmenter_backends()["backends"]]


def test_the_slider_rule_is_the_browsers():
    gap = {"type": "slider", "min": 0.5, "max": 30.0, "step": 0.1}
    assert not held_by(gap, 0.3)                                                    # clamped to 0.5 before G1
    assert not held_by({"type": "slider", "min": 0.1, "max": 0.9, "step": 0.05}, 0.32)   # moved to 0.3
    assert held_by({"type": "slider", "min": 0.1, "max": 0.9, "step": 0.01}, 0.32)
    assert not held_by({"type": "slider", "min": 20, "max": 300, "step": 5}, 48)         # moved to 50
    assert held_by({"type": "slider", "min": 10, "max": 1200, "step": 1}, 48)


@pytest.mark.parametrize("pipeline", CRISPASR_PIPELINES)
def test_a_crispasr_pass_window_holds_the_values_the_pass_runs_with(api, segmenters, pipeline):
    wrong, seen = [], 0
    for sensitivity in SENSITIVITIES:
        for segmenter in [""] + segmenters:
            for scene in SCENE_DETECTORS:
                opened = api.get_crispasr_customize_defaults(pipeline, sensitivity, segmenter, scene)
                if not opened.get("success"):
                    continue
                tabs = ((api.get_segmenter_schema(opened["segmenter_backend"]), opened["segmenter"]),
                        (api.get_scene_detector_schema(scene, pipeline), opened["scene"]))
                for schema, values in tabs:
                    for name, control in (schema.get("parameters") or {}).items():
                        if name in values:
                            seen += 1
                            if not held_by(control, values[name]):
                                wrong.append((sensitivity, opened["segmenter_backend"], scene, name, values[name],
                                              [control.get("min"), control.get("max"), control.get("step")]))
    assert seen and not wrong, wrong[:10]


@pytest.mark.parametrize("model", ("qwen3", "anime-whisper"))
def test_the_chronosjav_window_holds_the_values_the_pass_runs_with(api, segmenters, model):
    wrong, seen = [], 0
    for sensitivity in SENSITIVITIES:
        for segmenter in ["whisperseg"] + segmenters:
            opened = api.get_qwen_schema(sensitivity, model, segmenter)
            if not opened.get("success"):
                continue
            for tab, controls in opened["schema"].items():
                for name, control in (controls or {}).items():
                    if isinstance(control, dict) and control.get("default") is not None and "type" in control:
                        seen += 1
                        if not held_by(control, control["default"]):
                            wrong.append((sensitivity, segmenter, tab, name, control["default"],
                                          [control.get("min"), control.get("max"), control.get("step")]))
    assert seen and not wrong, wrong[:10]


def test_the_three_cases_of_the_report_are_held(api):
    """#499's log (2026-11-01): Cohere-CrispASR's gap 0.3 s, Parakeet-CrispASR's longest group 6 s, 48 s scenes."""
    cohere = api.get_crispasr_customize_defaults("cohere-crispasr", "balanced", "", "semantic")
    parakeet = api.get_crispasr_customize_defaults("parakeet-crispasr", "balanced", "", "semantic")
    firered = api.get_segmenter_schema("firered-vad")["parameters"]
    ten = api.get_segmenter_schema("ten")["parameters"]
    semantic = api.get_scene_detector_schema("semantic", "cohere-crispasr")["parameters"]
    assert cohere["segmenter"]["chunk_threshold_s"] == 0.3 and held_by(firered["chunk_threshold_s"], 0.3)
    assert parakeet["segmenter"]["max_group_duration_s"] == 6.0 and held_by(ten["max_group_duration_s"], 6.0)
    assert cohere["scene"]["max_duration"] == 48 and held_by(semantic["max_duration"], 48)
