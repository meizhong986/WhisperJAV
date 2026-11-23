"""The media files a run takes and the Add File(s) window offers: one list (1.10.3, owner W750 A7).

.ts and .m2ts (MPEG transport streams) and .vob (DVD video) are taken by runs since 1.10.3; before, the window offered
.ts while a run refused it ("Unsupported file extension"), and a run took .mov, .m4v and .wma, which the window did
not offer. No media is decoded: discovery chooses files by extension (ffprobe only adds details).

Run with: python -m pytest tests/test_media_file_types.py -q
"""
import re
from pathlib import Path

from whisperjav.modules.media_discovery import (AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS,
                                                MediaDiscovery, media_file_filter)

BEFORE_1103 = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
               '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.m4b', '.opus'}
REPO = Path(__file__).resolve().parents[1]


def test_new_types_are_taken_and_none_dropped():
    assert {'.ts', '.m2ts', '.vob'} <= VIDEO_EXTENSIONS
    assert BEFORE_1103 <= SUPPORTED_EXTENSIONS
    assert SUPPORTED_EXTENSIONS == VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def test_the_window_offers_exactly_what_a_run_takes():
    m = re.fullmatch(r"Media Files \((.+)\)", media_file_filter())
    assert m, media_file_filter()
    offered = {p[1:] for p in m.group(1).split(";")}
    assert offered == set(SUPPORTED_EXTENSIONS)
    assert " " not in m.group(1)          # pywebview: semicolon-separated, no spaces


def test_the_gui_builds_its_filter_from_the_one_list():
    api = (REPO / "whisperjav/webview_gui/api.py").read_text(encoding="utf-8")
    assert "media_file_filter()" in api
    assert "*.mp4;*.mkv" not in api


def test_a_folder_input_finds_the_new_types(tmp_path):
    for name in ("a.ts", "b.M2TS", "c.vob", "d.mov", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    found = {Path(f["path"]).name for f in MediaDiscovery().discover([str(tmp_path)])}
    assert found == {"a.ts", "b.M2TS", "c.vob", "d.mov"}


def test_a_single_ts_file_is_taken(tmp_path):
    f = tmp_path / "recording.ts"
    f.write_bytes(b"")
    assert [Path(x["path"]).name for x in MediaDiscovery().discover([str(f)])] == ["recording.ts"]
