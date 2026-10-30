"""1.10.0, item 5 (REQ5; #449, #451, #452): Faster Whisper XXL as pass 1, and the chosen XXL model reaches XXL.

Owner decisions: W24 2 (XXL default model large-v2), W92 (an XXL failure is never silent), W271 1-4 (a failed pass 1
still fails the file; the Model column offers large-v2, large-v3, large-v3-turbo, medium; both passes may be XXL; no
--model in the Extra Args default, a typed one removed with a console warning).

The pass worker is not imported here (it loads the ASR pipelines); its XXL lines are read from the source.
"""
import inspect
import json
import re
import shlex
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from whisperjav.byop.xxl_runner import (DEFAULT_XXL_MODEL, SIDE_FILE_NAME, failure_note, failure_reason,
                                        model_override_note, run_xxl, strip_model_args)
from whisperjav.webview_gui.api import WhisperJAVAPI

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "whisperjav" / "webview_gui" / "assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
JS = (ASSETS / "app.js").read_text(encoding="utf-8")
WORKER = (ROOT / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")


@pytest.fixture
def api():
    return WhisperJAVAPI()


@pytest.fixture
def fake_exe(tmp_path):
    exe = tmp_path / "faster-whisper-xxl.exe"
    exe.write_bytes(b"not a real program")
    return exe


def _select(html, select_id):
    return re.search(r'<select[^>]*id="%s"[^>]*>(.*?)</select>' % re.escape(select_id), html, re.S).group(1)


def _twopass(tmp_path, pass1, pass2=None):
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    return {"inputs": [str(inp)], "output_dir": str(tmp_path / "out"), "source_language": "japanese",
            "subs_language": "native", "pass1": pass1, "pass2": pass2 or {"enabled": False}}


def _value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def _xxl_pass(exe, model, enabled=None):
    p = {"pipeline": "xxl", "isXxl": True, "xxlExe": str(exe), "model": model,
         # what the row still holds for the greyed controls; none of it may be sent
         "sensitivity": "aggressive", "sceneDetector": "semantic", "speechSegmenter": "whisperseg",
         "speechEnhancer": "none"}
    if enabled is not None:
        p["enabled"] = enabled
    return p


# ---------------------------------------------------------------------------------------------------------------
# The model: large-v2 by default (W24 2); a --model in Extra Args is removed (#272) and reported (W271 4)
# ---------------------------------------------------------------------------------------------------------------

def test_the_default_model_is_large_v2():
    assert DEFAULT_XXL_MODEL == "large-v2"
    assert inspect.signature(run_xxl).parameters["model"].default == "large-v2"
    assert 'pass_config.get("model") or DEFAULT_XXL_MODEL' in WORKER
    assert '"large-v3"' not in WORKER


@pytest.mark.parametrize("args, kept, removed", [
    ("--model large-v3 --verbose True --standard_asia", ["--verbose", "True", "--standard_asia"], ["--model large-v3"]),
    ("--verbose True --model=large-v3", ["--verbose", "True"], ["--model=large-v3"]),
    ("-m medium --standard_asia", ["--standard_asia"], ["-m medium"]),
    ("-mlarge-v3 --standard_asia", ["--standard_asia"], ["-mlarge-v3"]),
    ("--standard_asia --model", ["--standard_asia"], ["--model"]),
    ("--model_dir D:/models --standard_asia", ["--model_dir", "D:/models", "--standard_asia"], []),
    # a quoted value is one argument: never cut, its spaces kept
    ('--initial_prompt "say -m now"  --standard_asia', ["--initial_prompt", "say -m now", "--standard_asia"], []),
    ('--initial_prompt "a  b"', ["--initial_prompt", "a  b"], []),
    ("", [], []),
])
def test_strip_model_args(args, kept, removed):
    assert strip_model_args(shlex.split(args)) == (kept, removed)


def test_the_override_note_names_extra_args_and_the_side_file(tmp_path, fake_exe):
    assert model_override_note(1, "large-v2", "--verbose True --standard_asia", str(fake_exe)) is None
    (tmp_path / SIDE_FILE_NAME).write_text("-m medium", encoding="utf-8")
    note = model_override_note(2, "large-v3", "--model large-v2 --standard_asia", str(fake_exe))
    assert note == ('Pass 2: XXL extra arguments contain "--model large-v2, -m medium (whisperjav_xxl.txt)", '
                    "which is not used. XXL runs large-v3, the model chosen for pass 2 (the Model column in the "
                    "GUI, --pass2-model on the command line).")
    # arguments that cannot be split: no note here; run_xxl fails per file with the reason
    assert model_override_note(1, "large-v2", '--initial_prompt "open', str(fake_exe)) is None


def test_the_failure_note_gives_the_reason_and_a_true_outcome():
    error = RuntimeError("XXL failed (exit 2, 0.1s):\nusage: ...\nError: unrecognized arguments: --bogus")
    assert failure_reason(error) == "XXL failed (exit 2, 0.1s) — Error: unrecognized arguments: --bogus"
    assert failure_note(1, "film", error) == (
        "Pass 1 (XXL Faster Whisper) failed for film: XXL failed (exit 2, 0.1s) — Error: unrecognized "
        "arguments: --bogus. This file is marked failed.")
    # pass 2 runs even when pass 1 failed for the file, so it promises nothing
    assert failure_note(2, "film", FileNotFoundError("XXL executable not found: x.exe")) == (
        "Pass 2 (XXL Faster Whisper) failed for film: XXL executable not found: x.exe. "
        "The file keeps pass 1's subtitles, if pass 1 made them.")
    assert failure_reason(RuntimeError()) == "RuntimeError"


def test_the_worker_uses_the_notes():
    assert "model_override_note(pass_number, xxl_model, xxl_extra_args, xxl_exe)" in WORKER
    assert "failure_note(pass_number, basename, xxl_error)" in WORKER
    assert 'error=f"{failure_reason(xxl_error)}\\n' in WORKER


# ---------------------------------------------------------------------------------------------------------------
# What the GUI sends
# ---------------------------------------------------------------------------------------------------------------

def test_an_xxl_pass_1_sends_the_program_and_the_model_only(api, tmp_path, fake_exe):
    args = api._build_twopass_args(_twopass(tmp_path, _xxl_pass(fake_exe, "large-v2")))
    assert _value(args, "--pass1-pipeline") == "xxl"
    assert _value(args, "--xxl-exe") == str(fake_exe)
    assert _value(args, "--pass1-model") == "large-v2"
    for flag in ("--pass1-sensitivity", "--pass1-scene-detector", "--pass1-speech-segmenter",
                 "--pass1-speech-enhancer", "--pass1-qwen-params", "--pass1-params"):
        assert flag not in args, flag


def test_an_xxl_pass_2_sends_its_model(api, tmp_path, fake_exe):
    """#449: the GUI sent no model for XXL, and XXL ran large-v3."""
    pass1 = {"pipeline": "fidelity", "sensitivity": "aggressive", "model": "large-v2"}
    args = api._build_twopass_args(_twopass(tmp_path, pass1, _xxl_pass(fake_exe, "large-v2", enabled=True)))
    assert _value(args, "--pass2-pipeline") == "xxl"
    assert _value(args, "--pass2-model") == "large-v2"
    assert _value(args, "--xxl-exe") == str(fake_exe)
    assert "--pass2-sensitivity" not in args


def test_both_passes_xxl_send_one_program_and_a_model_each(api, tmp_path, fake_exe):
    args = api._build_twopass_args(_twopass(tmp_path, _xxl_pass(fake_exe, "large-v2"),
                                            _xxl_pass(fake_exe, "large-v3", enabled=True)))
    assert args.count("--xxl-exe") == 1
    assert _value(args, "--pass1-model") == "large-v2"
    assert _value(args, "--pass2-model") == "large-v3"


# ---------------------------------------------------------------------------------------------------------------
# The command line accepts it (--dump-params runs main.py's checks without ASR)
# ---------------------------------------------------------------------------------------------------------------

def _main(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "whisperjav.main", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300, cwd=str(cwd))


def test_whisperjav_accepts_the_command_line_the_gui_builds(api, tmp_path, fake_exe):
    args = api._build_twopass_args(_twopass(tmp_path, _xxl_pass(fake_exe, "large-v2"),
                                            _xxl_pass(fake_exe, "large-v3", enabled=True)))
    dump = tmp_path / "dump.json"
    p = _main(*args, "--dump-params", str(dump))
    assert p.returncode == 0, p.stderr[-2000:]
    ensemble = json.loads(dump.read_text(encoding="utf-8"))["ensemble_config"]
    assert ensemble["pass1"]["pipeline"] == "xxl"
    assert ensemble["pass1"]["model"] == "large-v2"
    assert ensemble["pass1"]["xxl_exe"] == str(fake_exe)
    assert ensemble["pass2"]["pipeline"] == "xxl"
    assert ensemble["pass2"]["model"] == "large-v3"


def test_xxl_as_pass_1_without_the_program_stops(tmp_path):
    inp = tmp_path / "a.mp4"
    inp.write_bytes(b"x")
    p = _main(str(inp), "--ensemble", "--pass1-pipeline", "xxl", "--dump-params", str(tmp_path / "d.json"))
    assert p.returncode == 1
    assert "--xxl-exe is required when --pass1-pipeline is xxl" in (p.stdout + p.stderr)


def test_help_names_xxl_for_pass_1():
    p = _main("--help")
    assert p.returncode == 0
    text = " ".join(p.stdout.split())
    assert "'xxl' = BYOP XXL Faster Whisper (requires --xxl-exe)" in text
    assert "For xxl: the Whisper model XXL runs (default large-v2)." in text
    assert "required when --pass1-pipeline or --pass2-pipeline is xxl" in text


# ---------------------------------------------------------------------------------------------------------------
# The page and the shipped settings
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("select_id", ["pass1-pipeline", "pass2-pipeline"])
def test_both_passes_offer_xxl(select_id):
    assert '<option value="xxl">XXL Faster Whisper</option>' in _select(HTML, select_id)


def test_the_xxl_model_list_is_the_owners_with_large_v2_first():
    block = re.search(r"xxlModels:\s*\[(.*?)\]", JS, re.S).group(1)
    assert re.findall(r"value:\s*'([^']+)'", block) == ["large-v2", "large-v3", "large-v3-turbo", "medium"]


def test_the_extra_args_box_carries_no_model():
    box = re.search(r'<input[^>]*id="xxl-extra-args"[^>]*>', HTML, re.S).group(0)
    assert 'value="--verbose True --standard_asia"' in box
    assert "--model large" not in box
    assert "extraArgs: '--verbose True --standard_asia'" in JS


def test_the_shipped_settings_hold_no_xxl_path_and_no_model():
    byop = json.loads((ROOT / "whisperjav" / "config" / "asr_config.json").read_text(encoding="utf-8"))[
        "ui_preferences"]["byop"]
    assert "xxl_exe_path" not in byop
    assert strip_model_args(shlex.split(byop.get("xxl_extra_args", "")))[1] == []


def _build_release():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_release", ROOT / "installer" / "build_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_build_refuses_a_path_from_this_computer_in_the_shipped_settings():
    """Owner W275: 1.9.5 shipped the developer's XXL path; the release build now stops on it."""
    problems = _build_release().shipped_settings_problems
    assert problems({"ui_preferences": {"byop": {"xxl_extra_args": "--verbose True --standard_asia"},
                                        "theme": "default", "last_language": "japanese"}}) == []
    found = problems({"ui_preferences": {"byop": {"xxl_exe_path": "faster-whisper-xxl.exe"},
                                         "folders": ["C:\\Users\\me\\Videos", "\\\\nas\\share", "/home/me/x"],
                                         "note": "a / b"}})
    assert found == ["ui_preferences.byop.xxl_exe_path: 'faster-whisper-xxl.exe'",
                     "ui_preferences.folders[0]: 'C:\\\\Users\\\\me\\\\Videos'",
                     "ui_preferences.folders[1]: '\\\\\\\\nas\\\\share'",
                     "ui_preferences.folders[2]: '/home/me/x'"]


def test_the_shipped_settings_pass_the_build_check():
    p = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "installer" / "build_release.py"), "--check-settings"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, cwd=str(ROOT))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "holds no path from this computer" in p.stdout


def test_the_page_keeps_one_xxl_setting_for_both_passes():
    assert "xxlExePath" not in JS and "xxlExtraArgs" not in JS
    # the late preference reload wrote here before 1.10.0; AppState has no state object
    assert not re.search(r"AppState\.state\.[\w.]+\s*=", JS)


# ---------------------------------------------------------------------------------------------------------------
# End to end with a stand-in XXL program: main.py -> ensemble -> pass worker -> run_xxl -> the program.
# Covers what --dump-params cannot: the pass configuration the run itself builds (program, model, extra args).
# ---------------------------------------------------------------------------------------------------------------

_STAND_IN = '''import json, sys
from pathlib import Path
here = Path(__file__).parent
args = sys.argv[1:]
with (here / "calls.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\\n")
if (here / "FAIL").exists():
    print("Error: the stand-in XXL was told to fail", file=sys.stderr)
    sys.exit(3)
out = Path(args[args.index("--output_dir") + 1])
(out / (Path(args[0]).stem + ".srt")).write_text("1\\n00:00:00,000 --> 00:00:00,800\\nテスト\\n\\n", encoding="utf-8")
'''


@pytest.fixture
def stand_in_xxl(tmp_path):
    folder = tmp_path / "xxl"
    folder.mkdir()
    (folder / "stand_in.py").write_text(_STAND_IN, encoding="utf-8")
    exe = folder / "faster-whisper-xxl.cmd"
    exe.write_text(f'@"{sys.executable}" -X utf8 "{folder / "stand_in.py"}" %*\r\n', encoding="utf-8")
    return exe


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return path


def _calls(exe):
    return [json.loads(line) for line in (exe.parent / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.skipif(sys.platform != "win32", reason="the stand-in XXL is a Windows .cmd file")
def test_an_xxl_pass_1_run_gets_its_model_and_the_extra_args(tmp_path, stand_in_xxl, clip):
    from whisperjav.main import _get_xxl_extra_args_from_config
    (stand_in_xxl.parent / SIDE_FILE_NAME).write_text("--model large-v3", encoding="utf-8")
    p = _main(str(clip), "--ensemble", "--pass1-pipeline", "xxl", "--pass1-model", "large-v3-turbo",
              "--xxl-exe", str(stand_in_xxl), "--output-dir", str(tmp_path / "out"), "--temp-dir", str(tmp_path / "t"))
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-3000:]
    [call] = _calls(stand_in_xxl)
    assert call.count("--model") == 1 and call[call.index("--model") + 1] == "large-v3-turbo"
    assert "large-v3" not in call
    extra = strip_model_args(shlex.split(_get_xxl_extra_args_from_config()))[0]
    assert call[len(call) - len(extra):] == extra or not extra
    assert f'contain "--model large-v3 ({SIDE_FILE_NAME})", which is not used. XXL runs large-v3-turbo' in log
    assert (tmp_path / "out" / "clip.ja.pass1.srt").is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="the stand-in XXL is a Windows .cmd file")
def test_a_failing_xxl_in_both_passes_says_so_truly(tmp_path, stand_in_xxl, clip):
    (stand_in_xxl.parent / "FAIL").write_text("", encoding="utf-8")
    p = _main(str(clip), "--ensemble", "--pass1-pipeline", "xxl", "--pass2-pipeline", "xxl", "--pass2-model",
              "large-v3", "--xxl-exe", str(stand_in_xxl), "--output-dir", str(tmp_path / "out"),
              "--temp-dir", str(tmp_path / "t"))
    log = p.stdout + p.stderr
    assert p.returncode == 1, log[-3000:]
    assert ("Pass 1 (XXL Faster Whisper) failed for clip: XXL failed (exit 3" in log
            and "Error: the stand-in XXL was told to fail. This file is marked failed." in log)
    assert ("Pass 2 (XXL Faster Whisper) failed for clip:" in log
            and "The file keeps pass 1's subtitles, if pass 1 made them." in log)
    assert [c[c.index("--model") + 1] for c in _calls(stand_in_xxl)] == ["large-v2", "large-v3"]
