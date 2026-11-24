"""The audio check's finding reaches the run summary (1.10.3, owner W749 2): a pipeline that extracts the audio hands
the audio check's report to apply_to_run right after the extraction, so a damaged audio track makes the file 'suspect'
with its reason, or stops it with --fail-on suspect. Until 1.10.2 the transformers pipeline printed the check's line
and kept the file 'done'.

Read from the source, as tests/test_speech_enhancer_spec.py does: importing the pipelines loads the ASR libraries.
Not covered by 1.10.3 (seen, his decision): fast, faster and decoupled do not hand the report on either.
"""
import re
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"


@pytest.mark.parametrize("module", ["transformers_pipeline.py", "balanced_pipeline.py", "fidelity_pipeline.py",
                                    "qwen_pipeline.py", "crispasr_pipeline.py"])
def test_the_report_is_applied_right_after_the_extraction(module):
    src = (PIPELINES / module).read_text(encoding="utf-8")
    extract = src.index("self.audio_extractor.extract(")
    after = src[extract:extract + 900]
    assert re.search(r"apply_audio_integrity\(self\.audio_extractor\.last_integrity, self\.degradations,\s*"
                     r"in_summary=self\.audio_integrity_in_summary\)", after), module


def test_transformers_resets_its_list_per_file_and_logs_a_stop_without_a_traceback():
    src = (PIPELINES / "transformers_pipeline.py").read_text(encoding="utf-8")
    assert src.index("self.degradations = []") < src.index("self.audio_extractor.extract(")
    assert "from whisperjav.modules.audio_integrity import AudioIntegrityStop, apply_to_run as apply_audio_integrity" in src
    assert 'logger.error(f"Pipeline error: {e}", exc_info=not isinstance(e, AudioIntegrityStop))' in src
    assert 'master_metadata["summary"]["degradations"] = list(getattr(self, "degradations", None) or [])' in src
