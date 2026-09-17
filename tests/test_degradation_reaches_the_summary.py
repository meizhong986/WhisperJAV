#!/usr/bin/env python3
"""
Cross-cutting rule 1 of the agreed error-handling table (owner, 2026-09-17):
anything that quietly fell short is named in the RUN SUMMARY, not left in a log
line nobody reads.

This is what makes degrading acceptable instead of silent. It also gives one
switch -- ``--fail-on suspect`` -- to anyone who wants no degradation at all.

The channel is the ``degraded`` / ``degraded_reason`` pair that classify_output
already had for a failed pass 2; nothing new was invented for it.

Run with: pytest tests/test_degradation_reaches_the_summary.py -v
"""

import pytest

from whisperjav.utils.run_outcome import (
    FAIL_ON_CHOICES,
    classify_output,
    exit_status,
    format_summary,
    parse_fail_on,
)

SHORTFALL = "2 of 3 scenes went through without being cleaned up by ClearVoice"


@pytest.fixture
def srt(tmp_path):
    """A subtitle file long enough to be judged a good result on its own."""
    path = tmp_path / "clip.ja.srt"
    lines = []
    for n in range(30):
        start, end = n * 10, n * 10 + 9
        lines.append(
            f"{n + 1}\n"
            f"00:{start // 60:02d}:{start % 60:02d},000 --> "
            f"00:{end // 60:02d}:{end % 60:02d},000\n"
            f"line {n + 1}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestAShortfallIsVisible:
    def test_without_one_the_file_is_simply_done(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0)
        assert outcome.state == "done"

    def test_with_one_the_file_becomes_suspect(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        assert outcome.state == "suspect"

    def test_the_reason_is_carried_not_summarised_away(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        assert SHORTFALL in outcome.detail

    def test_it_appears_in_the_printed_summary(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        text = format_summary([outcome], fail_on=(), status=0, manifest_path=None)
        assert "suspect" in text.lower()
        assert "cleaned up" in text

    def test_the_subtitles_are_still_delivered(self, srt):
        # The point of degrading rather than failing: the work is not thrown away.
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        assert outcome.output == str(srt)
        assert outcome.subtitle_count == 30


class TestTheUserCanMakeItFatal:
    def test_by_default_a_shortfall_still_exits_zero(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        assert exit_status([outcome], parse_fail_on(None)) == 0

    def test_fail_on_suspect_turns_it_into_a_failure(self, srt):
        outcome = classify_output("clip.mp4", srt, 300.0,
                                  degraded=True, degraded_reason=SHORTFALL)
        assert exit_status([outcome], parse_fail_on(["suspect"])) == 1

    def test_suspect_is_offered_as_a_choice(self):
        assert "suspect" in FAIL_ON_CHOICES


class TestEveryPipelineCollectsThem:
    """
    Each pipeline hands enhance_scenes a list to write into and leaves it on
    itself for main.py to read after process(). A pipeline that forgets would
    lose the shortfall silently, which is the whole failure this rule exists to
    prevent -- and it cannot be caught by running, because it only shows up when
    a clean-up half-fails.
    """

    @pytest.mark.parametrize("module", [
        "fidelity_pipeline", "balanced_pipeline", "qwen_pipeline",
        "transformers_pipeline", "decoupled_pipeline",
    ])
    def test_it_passes_the_list_and_resets_it_per_file(self, module):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
                  / f"{module}.py").read_text(encoding="utf-8")

        assert "self.degradations = []" in source, f"{module} never resets the list"

        # The import reads "enhance_scenes," so only real calls carry a "(".
        calls = source.count("enhance_scenes(")
        passed = source.count("degradations=self.degradations")
        assert calls > 0, f"{module} does not call enhance_scenes at all"
        assert passed == calls, (
            f"{module} calls enhance_scenes {calls} time(s) but passes "
            f"degradations {passed} time(s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
