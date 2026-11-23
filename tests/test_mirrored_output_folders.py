"""Under a chosen output folder, the input's subfolders are kept (1.10.3, owner W750 B8, W756 3 "always"; #126).

Before 1.10.3, with a chosen output folder, A/x.mp4 and B/x.mp4 of one folder input both wrote x.ja.whisperjav.srt
into that folder: the second overwrote the first, or --skip-existing skipped it. Now a file found in a subfolder of a
folder input writes into the same subfolder under the chosen folder. Files directly in the folder input, files given
one by one, and "source" (beside each file) are as before. The orchestrator and main.py are checked by their source:
importing the ensemble worker can hang (CLAUDE.md).

Run with: python -m pytest tests/test_mirrored_output_folders.py -q
"""
from pathlib import Path

from whisperjav.modules.media_discovery import MediaDiscovery, output_folder_for

REPO = Path(__file__).resolve().parents[1]


def _tree(root: Path):
    for rel in ("A/x.mp4", "B/x.mp4", "B/deeper/z.mkv", "y.wav"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_a_folder_inputs_subfolders_are_kept_under_the_chosen_folder(tmp_path):
    _tree(tmp_path / "in")
    out = tmp_path / "out"
    found = {Path(m["path"]).relative_to((tmp_path / "in").absolute()).as_posix(): output_folder_for(m, str(out))
             for m in MediaDiscovery().discover([str(tmp_path / "in")])}
    assert found == {"A/x.mp4": out / "A", "B/x.mp4": out / "B", "B/deeper/z.mkv": out / "B" / "deeper", "y.wav": out}


def test_two_files_of_the_same_name_no_longer_share_an_output_folder(tmp_path):
    _tree(tmp_path / "in")
    folders = [output_folder_for(m, str(tmp_path / "out")) for m in MediaDiscovery().discover([str(tmp_path / "in")])
               if Path(m["path"]).name == "x.mp4"]
    assert len(folders) == 2 and folders[0] != folders[1]


def test_source_and_single_files_are_as_before(tmp_path):
    _tree(tmp_path / "in")
    for m in MediaDiscovery().discover([str(tmp_path / "in")]):
        assert output_folder_for(m, "source") == Path(m["path"]).parent
        assert output_folder_for(m, "SOURCE ") == Path(m["path"]).parent
    single = MediaDiscovery().discover([str(tmp_path / "in" / "A" / "x.mp4")])[0]
    assert "input_folder" not in single
    assert output_folder_for(single, str(tmp_path / "out")) == tmp_path / "out"


def test_a_file_outside_its_recorded_folder_goes_to_the_chosen_folder(tmp_path):
    info = {"path": str(tmp_path / "elsewhere" / "a.mp4"), "input_folder": str(tmp_path / "in")}
    assert output_folder_for(info, str(tmp_path / "out")) == tmp_path / "out"


def test_every_entry_point_takes_the_folder_from_the_one_helper():
    main = (REPO / "whisperjav/main.py").read_text(encoding="utf-8")
    assert main.count("output_folder_for(") == 4          # sync run, async run, async and ensemble --skip-existing
    assert "expected_dir = Path(args.output_dir)" not in main
    orch = (REPO / "whisperjav/ensemble/orchestrator.py").read_text(encoding="utf-8")
    i = orch.index("def _resolve_file_output_dir")
    assert "output_folder_for(media_info, str(self.output_dir))" in orch[i:i + 900]
