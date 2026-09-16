#!/usr/bin/env python3
"""The Ensemble tab's model lists must agree with each other.

`app.js` holds three tables that all describe which ASR model a pass may use:

  legacyModels                 every model the Ensemble ROW can offer
  pipelineModelCompatibility   which of those each pipeline can actually run
  allModelOptions              the table the Customize dialog filters (generateModelTab)

They are filtered in a chain, and a model present in the compatibility list but absent from
`allModelOptions` cannot be shown in the dialog. `generateModelTab` then falls back to the
first allowed model, and `applyCustomization` syncs that substitute back over the row —
silently, because the substitute IS a legitimate row option. That is how
`whisper-ja-1.5B-ct2` came to be added to Balanced and be unselectable in the same commit.

This check previously lived only in a scratchpad file outside the repository, so nothing
stopped the same mistake at the next model addition. It parses `app.js` rather than running
it: there is no JavaScript test runner here, and the tables are plain literals.
"""
import json
import re
from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parents[1]
          / "whisperjav" / "webview_gui" / "assets" / "app.js")


def _js_array_of_objects(source, anchor):
    """Values from a `[{ value: 'x', label: 'y' }, ...]` literal following `anchor`."""
    i = source.index(anchor)
    start = source.index("[", i)
    depth, j = 0, start
    for j in range(start, len(source)):
        if source[j] == "[":
            depth += 1
        elif source[j] == "]":
            depth -= 1
            if depth == 0:
                break
    return set(re.findall(r"value:\s*'([^']+)'", source[start:j + 1]))


def _compatibility_map(source):
    i = source.index("pipelineModelCompatibility: {")
    start = source.index("{", i)
    depth, j = 0, start
    for j in range(start, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                break
    body = source[start:j + 1]
    out = {}
    for m in re.finditer(r"'?([A-Za-z0-9_-]+)'?\s*:\s*\[([^\]]*)\]", body):
        out[m.group(1)] = set(re.findall(r"'([^']+)'", m.group(2)))
    return out


@pytest.fixture(scope="module")
def js():
    return APP_JS.read_text(encoding="utf-8")


def test_tables_were_found(js):
    """Guard the parser itself: a silent parse failure would make every test below vacuous."""
    assert _js_array_of_objects(js, "legacyModels:"), "legacyModels not parsed"
    assert _js_array_of_objects(js, "const allModelOptions = ["), "allModelOptions not parsed"
    compat = _compatibility_map(js)
    assert len(compat) >= 4, "pipelineModelCompatibility not parsed: %r" % compat
    assert "balanced" in compat and "fidelity" in compat


def test_every_compatible_model_can_be_shown_in_the_dialog(js):
    """The invariant that failed: compatibility ⊆ the dialog's own table."""
    allowed_anywhere = set()
    for models in _compatibility_map(js).values():
        allowed_anywhere |= models
    dialog = _js_array_of_objects(js, "const allModelOptions = [")
    missing = allowed_anywhere - dialog
    assert not missing, (
        "these models are offered by a pipeline but are absent from generateModelTab's "
        "allModelOptions, so the dialog cannot show them and will substitute another "
        "model over the user's choice: %s" % sorted(missing))


def test_every_compatible_model_can_be_shown_in_the_row(js):
    """Compatibility ⊆ legacyModels, or the row cannot offer it either."""
    row = _js_array_of_objects(js, "legacyModels:")
    for pipeline, models in _compatibility_map(js).items():
        if pipeline == "kotoba-faster-whisper":
            continue  # not offered in the row's pipeline dropdown
        missing = models - row
        assert not missing, (
            "%s is allowed models the row cannot offer: %s" % (pipeline, sorted(missing)))


def test_the_engine_constraints_hold(js):
    """faster-whisper has no turbo; OpenAI Whisper cannot load a CTranslate2 checkpoint."""
    compat = _compatibility_map(js)
    ct2 = "TransWithAI/whisper-ja-1.5B-ct2"
    for pipeline in ("balanced", "fast", "faster"):
        assert "turbo" not in compat[pipeline], (
            "%s runs faster-whisper, which has no 'turbo' build" % pipeline)
    assert "turbo" in compat["fidelity"], "fidelity runs OpenAI Whisper, which has turbo"
    assert ct2 not in compat["fidelity"], (
        "fidelity runs OpenAI Whisper and cannot load a CTranslate2 checkpoint")
    assert ct2 in compat["balanced"], "whisper-ja-1.5B is a Balanced option (owner, 2026-09-16)"


def test_the_html_row_options_match_legacy_models(js):
    """index.html ships a static list used until the pipeline dropdown is first touched."""
    html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
    i = html.index('id="pass1-model"')
    block = html[i:html.index("</select>", i)]
    shipped = set(re.findall(r'<option value="([^"]+)"', block))
    row = _js_array_of_objects(js, "legacyModels:")
    assert shipped <= row, (
        "index.html offers models legacyModels does not know: %s" % sorted(shipped - row))
