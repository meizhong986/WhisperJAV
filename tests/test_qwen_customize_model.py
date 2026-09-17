#!/usr/bin/env python3
"""
Tests for the Qwen Customize window and the row agreeing on which model runs.

The window used to show its own model while the row was what actually ran
(pass_worker sets qwen_model_id from the row). So a model chosen in the window
was displayed as accepted and then silently discarded. Owner, 2026-09-17, on the
fix: make the window show the row's model and write back to it, so there is one
answer and the row is always right.

These read the shipped app.js. A GUI change is not verified until the owner
clicks it -- this only catches the wiring being removed or reverted.

Run with: pytest tests/test_qwen_customize_model.py -v
"""

from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui"
          / "assets" / "app.js")


@pytest.fixture(scope="module")
def source():
    return APP_JS.read_text(encoding="utf-8")


class TestTheWindowShowsWhatWillRun:
    def test_it_reads_the_models_model(self, source):
        window = source[source.index("generateQwenModelTab(tabId"):]
        window = window[:window.index("// Language dropdown")]
        assert "const rowModel = passState && passState.model;" in window
        assert "rowModel || currentValues.model_id || modelDefault" in window

    def test_a_model_the_window_does_not_list_is_shown_anyway(self, source):
        """
        The row offers models this window's list does not. Substituting one of
        the window's own is how the two came to disagree, so the row's value is
        added to the list instead.
        """
        window = source[source.index("generateQwenModelTab(tabId"):]
        window = window[:window.index("// Language dropdown")]
        assert "!modelOptions.some(o => o.value === rowModel)" in window
        assert "modelOptions.concat([{ value: rowModel, label: rowModel }])" in window


class TestTheWindowWritesBackToTheRow:
    def test_a_qwen_model_chosen_in_the_window_reaches_the_row(self, source):
        assert "if (fullParams.model_id && passState.isQwen) {" in source
        block = source[source.index("if (fullParams.model_id && passState.isQwen) {"):]
        block = block[:block.index("// Sync framer from modal to state")]
        assert "modelDropdown.value = fullParams.model_id;" in block
        assert "this.state[passKey].model = fullParams.model_id;" in block

    def test_a_value_the_row_cannot_show_is_reported_not_lost(self, source):
        """
        Assigning an absent value to a <select> does nothing, so syncing without
        the guard would leave the row and the state disagreeing silently.
        """
        block = source[source.index("if (fullParams.model_id && passState.isQwen) {"):]
        block = block[:block.index("// Sync framer from modal to state")]
        assert "const offered = Array.from(modelDropdown.options)" in block
        assert "ConsoleManager.log(" in block
        assert "'warn'" in block

    def test_the_legacy_path_is_untouched(self, source):
        """The legacy branch uses model_name and must keep its own behaviour."""
        assert ("if (fullParams.model_name && !passState.isTransformers && "
                "!passState.isQwen) {") in source


class TestWhatIsNotDone:
    def test_transformers_is_recorded_as_still_having_the_old_behaviour(self, source):
        """
        The Transformers window has the same shape and has not been changed. The
        comment must say so, rather than leaving a later reader to assume both
        were fixed.
        """
        assert "Transformers" in source
        assert "Same shape, not yet done." in source


class TestTheRowShowsWhatWasApplied:
    """
    Owner's GUI test A4, 2026-09-17: tick "Use enhanced audio for VAD framing
    only" inside the Customize window, Apply, and the row's box stayed UNTICKED
    while the run used the value from the window.

    The repaint function was already correct -- it sets the box from the stored
    value. Nothing called it after Apply. Apply wrote the state and refreshed
    only the badges, so the row was never redrawn.
    """

    def test_apply_repaints_the_row(self, source):
        start = source.index("applyCustomization")
        end = source.index("showApplyFeedback();", start)
        apply_body = source[start:end]
        assert "this.state[passKey].enhanceForVad = efvCheck.checked" in apply_body
        assert "this.updateEnhanceForVadCheckbox(passKey)" in apply_body, (
            "Apply stores the value but never asks the row to redraw")

    def test_the_repaint_sets_the_box_from_the_stored_value(self, source):
        start = source.index("updateEnhanceForVadCheckbox(passId)")
        body = source[start:start + 2000]
        assert "box.checked = !!this.state[passId]?.enhanceForVad" in body, (
            "the row must show the value that will actually be used")

    def test_balanced_still_clears_it(self, source):
        # The refusal must survive: balanced offers no VAD-only split, and a
        # value left from another pipeline must not travel into a run.
        start = source.index("updateEnhanceForVadCheckbox(passId)")
        body = source[start:start + 2000]
        assert "this.state[passId].enhanceForVad = false" in body
