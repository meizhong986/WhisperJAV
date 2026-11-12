"""whisperjav-translate's utility commands (1.10.2, still-open item 6 of the 1.10.1 notes).

Two faults: (1) --configure stopped with a TypeError for ollama and custom, which have no key variable
(os.getenv(None)); (2) -i/--input was required by argparse, so --configure, --show-settings and
--create-default-settings alone stopped with exit status 2 for every provider before they ran.

The settings file lives under %APPDATA% on Windows: every test points APPDATA at its own temporary folder, so the
user's real translation settings are never read or written.
"""
import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from whisperjav.translate import configure
from whisperjav.translate.providers import PROVIDER_CONFIGS

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("provider, line", [
    ("ollama", "Ollama runs on this computer and needs no API key."),
    ("custom", "If your server needs a key, give it with --api-key when you translate."),
])
def test_configure_without_a_key_variable_runs_to_the_end(appdata, monkeypatch, capsys, provider, line):
    choice = str(list(PROVIDER_CONFIGS).index(provider) + 1)
    asked = []

    def answer(prompt=""):
        asked.append(prompt)
        return choice if "Select provider" in prompt else ""   # Enter everywhere else: defaults, save = yes

    monkeypatch.setattr(builtins, "input", answer)
    assert configure.interactive_configure() is True
    out = capsys.readouterr().out
    assert line in out
    assert not any("API key" in p for p in asked)                 # no key prompt for these two
    assert (appdata / "WhisperJAV" / "translate" / "settings.json").is_file()


def test_configure_with_a_key_variable_still_asks_for_the_key(appdata, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    choice = str(list(PROVIDER_CONFIGS).index("deepseek") + 1)
    asked = []

    def answer(prompt=""):
        asked.append(prompt)
        return choice if "Select provider" in prompt else ""

    monkeypatch.setattr(builtins, "input", answer)
    assert configure.interactive_configure() is True
    assert any("Enter API key" in p for p in asked)


def _cli(appdata, *args):
    env = dict(os.environ, APPDATA=str(appdata), HOME=str(appdata))
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "whisperjav.translate.cli", *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                          cwd=str(REPO), env=env)


@pytest.mark.slow
def test_show_settings_runs_without_input(appdata):
    done = _cli(appdata, "--show-settings")
    assert done.returncode == 0, done.stderr[-1500:]
    assert "Settings file:" in done.stdout


@pytest.mark.slow
def test_create_default_settings_runs_without_input(appdata):
    done = _cli(appdata, "--create-default-settings")
    assert done.returncode == 0, done.stderr[-1500:]
    assert (appdata / "WhisperJAV" / "translate" / "settings.json").is_file()


@pytest.mark.slow
def test_a_translation_without_input_still_stops_with_status_2(appdata):
    done = _cli(appdata, "--provider", "claude")
    assert done.returncode == 2
    assert "the following arguments are required: -i/--input" in done.stderr
