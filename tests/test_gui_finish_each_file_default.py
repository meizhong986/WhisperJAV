"""1.10.1 (owner W493 1): the Ensemble option "Finish each file" is ticked by default in the GUI.

Three hops: the checkbox starts ticked (index.html), the page's state starts true (app.js, the value sent as
``serial_mode`` when the run starts), and the API turns ``serial_mode`` into ``--ensemble-serial``. Unticked, nothing is
sent. The command line's own default is not changed here (open for the owner). Light: no window, no model.
"""
import re
from pathlib import Path

import pytest

from whisperjav.webview_gui.api import WhisperJAVAPI

ASSETS = Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets"


def test_checkbox_starts_ticked():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    tag = re.search(r"<input[^>]*\bid=\"ensemble-serial\"[^>]*>", html)
    assert tag, "the Finish each file checkbox is missing"
    assert re.search(r"\schecked(\s|>|$)", tag.group(0)), tag.group(0)


def test_page_state_starts_true():
    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert re.search(r"^\s*serialMode: true,", js, re.M)
    assert not re.search(r"^\s*serialMode: false,", js, re.M)
    # the start of a run sends the state as serial_mode
    assert "serial_mode: this.state.serialMode," in js


def test_label_and_tooltip_unchanged():
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    assert "<span>Finish each file</span>" in html
    assert ('title="Complete each file fully before starting the next. Slower (reloads models per file) but delivers '
            'results incrementally."') in html


@pytest.mark.parametrize("serial, expected", [(True, True), (False, False)])
def test_api_sends_the_flag_only_when_ticked(serial, expected):
    config = {"inputs": [r"C:\media\a.mp4"], "pass1": {"pipeline": "balanced", "sensitivity": "balanced"},
              "pass2": {"enabled": False}, "merge_strategy": "pass1_primary", "serial_mode": serial}
    args = WhisperJAVAPI()._build_twopass_args(config)
    assert ("--ensemble-serial" in args) is expected
    assert "--ensemble" in args
