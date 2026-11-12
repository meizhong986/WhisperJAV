"""BasePipeline.cleanup_temp_files (1.10.2, still-open item 1 of the 1.10.1 notes).

Before: the raw_subs pattern "<name>*" also matched ChronosJAV's per-file folder raw_subs/<name>/, and unlink() on a
folder failed, so every ChronosJAV file without --keep-temp ended with "Error during temporary file cleanup"; a name
starting with "[" matched nothing (a glob character class); "Film*" also matched "Film 2"'s files.
"""
from pathlib import Path

from whisperjav.pipelines.base_pipeline import BasePipeline


class _Pipeline(BasePipeline):
    def get_mode_name(self):
        return "test"

    def process(self, media_info):
        return {}


def _make(root: Path, rel: str, folder: bool = False) -> Path:
    path = root / rel
    if folder:
        path.mkdir(parents=True, exist_ok=True)
        (path / "scene_0000_diagnostics.json").write_text("{}", encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return path


def _pipeline(tmp_path, keep=False):
    return _Pipeline(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "tmp"), keep_temp_files=keep)


def test_chronosjav_folder_is_removed_and_no_error_is_logged(tmp_path, caplog):
    p = _pipeline(tmp_path)
    tmp = p.temp_dir
    folder = _make(tmp, "raw_subs/Film", folder=True)
    stitched = _make(tmp, "raw_subs/Film.qwen_stitched.original.srt")
    other = _make(tmp, "raw_subs/Film 2", folder=True)
    other_file = _make(tmp, "raw_subs/Film 2_stitched.original.srt")
    with caplog.at_level("ERROR"):
        p.cleanup_temp_files("Film")
    assert not folder.exists() and not stitched.exists()
    assert other.exists() and other_file.exists()          # another media file's own temp files stay
    assert "Error during temporary file cleanup" not in caplog.text


def test_a_name_with_brackets_is_matched_literally(tmp_path):
    p = _pipeline(tmp_path)
    tmp = p.temp_dir
    name = "[ABC-123] Film"
    files = [_make(tmp, f"scenes/{name}_scene_0001.wav"), _make(tmp, f"scene_srts/{name}_scene_0001.srt"),
             _make(tmp, f"raw_subs/{name}_stitched.original.srt"), _make(tmp, f"raw_subs/{name}", folder=True),
             _make(tmp, f"{name}_extracted.wav")]
    keep = _make(tmp, "scenes/A_scene_0001.wav")            # "[ABC-123]" read as a class would match "A"
    p.cleanup_temp_files(name)
    assert not any(f.exists() for f in files)
    assert keep.exists()


def test_other_folders_are_left_alone(tmp_path):
    p = _pipeline(tmp_path)
    nested = _make(p.temp_dir, "raw_subs/Film_extra", folder=True)
    p.cleanup_temp_files("Film")
    assert nested.exists()


def test_keep_temp_deletes_nothing(tmp_path):
    p = _pipeline(tmp_path, keep=True)
    folder = _make(p.temp_dir, "raw_subs/Film", folder=True)
    wav = _make(p.temp_dir, "Film_extracted.wav")
    p.cleanup_temp_files("Film")
    assert folder.exists() and wav.exists()
