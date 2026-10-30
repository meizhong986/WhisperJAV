"""v1.10.0 (REQ2, owner W22 (1), W42 (2), W43): llama-cpp ('local') is removed.

- A saved 'local' choice -- the translate settings file's provider, and the two GUI tab rows
  kept in the same file (#435) -- is read as 'ollama', with the owner's one console line, once.
- 'local' typed after --provider / --translate-provider stops with the owner's error, exit 1.

The settings file is a temporary one (APPDATA / HOME point at tmp_path).
"""
import json
import subprocess
import sys

import pytest

from whisperjav.translate import settings as settings_mod
from whisperjav.translate.providers import SAVED_LOCAL_NOTICE, PROVIDER_CONFIGS

NOTICE = "Local translation with llama-cpp was removed in 1.10; your saved choice now uses Ollama."
ERROR_TRANSLATE = ("Error: --provider local (llama-cpp) was removed in WhisperJAV 1.10. "
                   "Use --provider ollama for local translation.")
ERROR_MAIN = ("Error: --translate-provider local (llama-cpp) was removed in WhisperJAV 1.10. "
              "Use --translate-provider ollama for local translation.")


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(settings_mod, "_saved_local_notice_printed", False)
    monkeypatch.setattr(settings_mod, "_saved_local_seen", False)
    path = settings_mod.get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(data):
        full = dict(settings_mod.DEFAULT_SETTINGS)
        full.update(data)
        path.write_text(json.dumps(full), encoding="utf-8")
        return path

    return write


def test_the_texts_are_the_owners():
    assert SAVED_LOCAL_NOTICE == NOTICE
    assert "local" not in PROVIDER_CONFIGS


def test_saved_local_is_read_as_ollama_with_one_line(settings_file, capsys):
    settings_file({"provider": "local", "model": "llama-8b"})
    loaded = settings_mod.load_settings()
    assert loaded["provider"] == "ollama"
    assert loaded["model"] is None  # a llama-cpp model name means nothing to Ollama
    settings_mod.load_settings()  # read again: no second line
    err = capsys.readouterr().err
    assert err.count(NOTICE) == 1


def test_saved_tab_rows_are_read_as_ollama(settings_file, capsys):
    settings_file({"provider": "deepseek", "gui_tab_state": {
        "ensemble": {"provider": "local", "model": "gemma-9b"},
        "srt": {"provider": "local", "customModel": "auto", "targetLang": "english"},
    }})
    tabs = settings_mod.load_settings()["gui_tab_state"]
    assert tabs["ensemble"] == {"provider": "ollama"}
    assert tabs["srt"] == {"provider": "ollama", "targetLang": "english"}
    assert capsys.readouterr().err.count(NOTICE) == 1


def test_other_saved_choices_are_left_alone(settings_file, capsys):
    settings_file({"provider": "ollama", "model": "gemma3:12b"})
    loaded = settings_mod.load_settings()
    assert (loaded["provider"], loaded["model"]) == ("ollama", "gemma3:12b")
    assert NOTICE not in capsys.readouterr().err
    assert settings_mod.saved_local_provider_notice() == ""


def test_a_choice_seen_only_in_browser_storage_gets_the_line_too(settings_file, capsys):
    settings_file({"provider": "deepseek"})
    settings_mod.load_settings()
    assert settings_mod.saved_local_provider_notice(seen_elsewhere=True) == NOTICE
    assert settings_mod.saved_local_provider_notice(seen_elsewhere=True) == NOTICE
    assert capsys.readouterr().err.count(NOTICE) == 1


@pytest.mark.parametrize("cmd, expected", [
    ([sys.executable, "-m", "whisperjav.translate.cli", "--provider", "local", "-i", "x.srt"], ERROR_TRANSLATE),
    ([sys.executable, "-m", "whisperjav.main", "x.srt", "--translate", "--translate-provider", "local"], ERROR_MAIN),
])
def test_local_typed_on_a_command_line_stops_with_the_owners_error(tmp_path, cmd, expected):
    (tmp_path / "x.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, timeout=300,
                       env={**__import__("os").environ, "APPDATA": str(tmp_path), "PYTHONIOENCODING": "utf-8"})
    err = r.stderr.decode("utf-8", errors="replace")
    assert r.returncode == 1, err[-500:]
    assert expected in err
    assert not list(tmp_path.glob("x.*.srt"))
