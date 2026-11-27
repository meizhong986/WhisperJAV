"""Drag and drop: the Python side tells folders from files, and the page filters by tab."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from whisperjav.webview_gui.api import describe_dropped_paths, srt_files_in

APP_JS = Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui" / "assets" / "app.js"


@pytest.fixture
def drop(tmp_path):
    """A folder with a dot in its name holding two .srt files and a subfolder, plus a video and a text file."""
    folder = tmp_path / "SONE-853.1080p"
    (folder / "sub").mkdir(parents=True)
    for name in ("b.srt", "a.srt", "notes.txt", "sub/c.srt"):
        (folder / name).write_text("x", encoding="utf-8")
    empty = tmp_path / "empty.folder"
    empty.mkdir()
    video = tmp_path / "film.mp4"
    text = tmp_path / "readme.txt"
    for f in (video, text):
        f.write_text("x", encoding="utf-8")
    return {"folder": str(folder), "empty": str(empty), "video": str(video), "text": str(text)}


def test_srt_files_in_lists_only_the_folder_itself(drop):
    folder = Path(drop["folder"])
    assert srt_files_in(folder) == [str(folder / "a.srt"), str(folder / "b.srt")]


def test_folder_with_a_dot_is_a_folder(drop):
    entries = describe_dropped_paths([drop["folder"], drop["video"]])
    assert entries[0] == {"path": drop["folder"], "is_dir": True, "srt_files": srt_files_in(drop["folder"])}
    assert entries[1] == {"path": drop["video"], "is_dir": False, "is_media": True}


def test_media_uses_the_add_files_list(tmp_path):
    # .m2ts and .vob are in the Add File(s) list, so a drop takes them too.
    names = ["a.M2TS", "b.vob", "c.txt"]
    entries = describe_dropped_paths([str(tmp_path / n) for n in names])
    assert [e["is_media"] for e in entries] == [True, True, False]


# Runs the real addDroppedFiles from app.js with stub app state, then prints what was added and logged.
_DROP_JS = r"""
const s = require("fs").readFileSync(process.argv[1], "utf8");
const grab = (re) => { const m = s.match(re); if (!m) throw new Error("not found: " + re); return m[1]; };
const body = grab(/addDroppedFiles\(entries\) \{([\s\S]*?)\r?\n    \},/);
const [tab, entries] = JSON.parse(process.argv[2]);
const logs = [];
global.AppState = {activeTab: tab, selectedFiles: []};
global.ConsoleManager = {log: (msg, level) => logs.push(level)};
const flm = {render() {}, addDroppedFiles: new Function("entries", body)};
flm.addDroppedFiles(entries);
console.log(JSON.stringify({added: AppState.selectedFiles, logs}));
"""


def _run_drop(tab, entries):
    arg = json.dumps([tab, entries])
    result = subprocess.run(["node", "-e", _DROP_JS, str(APP_JS), arg], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


@needs_node
def test_media_tab_adds_the_folder_and_the_video(drop):
    paths = [drop["folder"], drop["video"], drop["text"]]
    out = _run_drop("tab1", describe_dropped_paths(paths))
    assert out["added"] == [drop["folder"], drop["video"]]
    assert "warning" in out["logs"]  # the text file is refused


@needs_node
def test_srt_tab_adds_the_srt_files_of_a_folder(drop):
    out = _run_drop("tab4", describe_dropped_paths([drop["folder"], drop["video"]]))
    assert out["added"] == srt_files_in(drop["folder"])
    assert "warning" in out["logs"]  # the video is refused


@needs_node
def test_srt_tab_warns_about_a_folder_without_srt_files(drop):
    out = _run_drop("tab4", describe_dropped_paths([drop["empty"]]))
    assert out["added"] == []
    assert out["logs"] == ["warning"]
