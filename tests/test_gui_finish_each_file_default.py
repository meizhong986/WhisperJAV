"""1.10.1 (owner W493 1, W501): the Ensemble option "Finish each file" is on by default, in the GUI and on the command
line.

Hops: the checkbox starts ticked (index.html), the page's state starts true (app.js, the value sent as ``serial_mode``
when the run starts), the API turns ``serial_mode`` into ``--ensemble-serial`` and an unticked box into
``--no-ensemble-serial`` (the command line's default is on since W501), and "Remember settings" keeps the box (W501).
Light: no window, no model.
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


@pytest.mark.parametrize("serial, sent, not_sent", [(True, "--ensemble-serial", "--no-ensemble-serial"),
                                                    (False, "--no-ensemble-serial", "--ensemble-serial")])
def test_api_sends_on_or_off(serial, sent, not_sent):
    config = {"inputs": [r"C:\media\a.mp4"], "pass1": {"pipeline": "balanced", "sensitivity": "balanced"},
              "pass2": {"enabled": False}, "merge_strategy": "pass1_primary", "serial_mode": serial}
    args = WhisperJAVAPI()._build_twopass_args(config)
    assert sent in args and not_sent not in args
    assert "--ensemble" in args


@pytest.mark.parametrize("argv, expected", [([], True), (["--ensemble-serial"], True),
                                            (["--no-ensemble-serial"], False)])
def test_the_command_line_default_is_on(monkeypatch, argv, expected):
    from whisperjav import main as wj_main

    monkeypatch.setattr("sys.argv", ["whisperjav", "a.mp4", "--ensemble", *argv])
    assert wj_main.parse_arguments().ensemble_serial is expected


def test_the_command_line_help_names_both():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "whisperjav.main", "--help"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert done.returncode == 0
    out = " ".join(done.stdout.split())
    assert "--ensemble-serial, --no-ensemble-serial" in out
    assert "before the next one starts (default: on)" in out


def test_remember_settings_keeps_the_box():
    """W501: the box is a remembered field (page side), mapped in the API, with a default that matches the page."""
    from whisperjav.settings.gui_settings import DEFAULT_GUI_SETTINGS

    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert "'ensemble-serial':  { key: 'ensembleSerial',      prop: 'checked' }," in js
    assert WhisperJAVAPI._GUI_SETTINGS_MAP["ensemble_serial"] == "ensembleSerial"
    assert DEFAULT_GUI_SETTINGS["ensemble_serial"] is True


def test_a_saved_unticked_box_comes_back_unticked(tmp_path, monkeypatch):
    from whisperjav.settings import gui_settings

    monkeypatch.setattr(gui_settings, "get_gui_settings_path", lambda: tmp_path / "gui_settings.json")
    assert gui_settings.save_gui_settings({"ensemble_serial": False, "remember_settings": True})
    assert gui_settings.load_gui_settings()["ensemble_serial"] is False
