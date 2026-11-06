"""#501: a partial alignment in ChronosJAV is a note, not "suspect"; notes reach the run summary on every path."""
from pathlib import Path

from whisperjav.utils.run_outcome import classify_output, format_summary

ROOT = Path(__file__).resolve().parents[1] / "whisperjav"
NOTE = ("ChronosJAV's aligner (wav2vec2-ja) gave no times for 6 of 1624 VAD group(s): each keeps its text as one "
        "unsplit subtitle, timed from its VAD group (the timing model may still move its start).")


def _srt(path: Path, n: int, step: float) -> Path:
    def ts(s):
        return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)):03d}"
    path.write_text("".join(f"{i + 1}\n{ts(i * step)} --> {ts(i * step + 1.5)}\nセリフ{i}\n\n" for i in range(n)),
                    encoding="utf-8")
    return path


def test_a_note_leaves_a_full_file_done_and_is_printed(tmp_path):
    out = _srt(tmp_path / "film.ja.whisperjav.srt", 1125, 6.0)
    outcome = classify_output(str(tmp_path / "film.mp4"), str(out), 6750.0)
    outcome.notes = [NOTE]
    assert outcome.state == "done"
    assert "gave no times for 6 of 1624" in format_summary([outcome], (), 0)


def test_main_attaches_notes_on_the_single_mode_paths_too():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert src.count('outcome.notes = list(') >= 3   # sync single mode, async single mode, Ensemble
