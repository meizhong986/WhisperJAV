#!/usr/bin/env python3
"""
A template can carry a note to its maintainer without shipping it to the user.

Why this exists. The installer README's "WHAT'S NEW IN v{{VERSION}}" section is
the first thing a user reads after installing, and only the version number is
substituted -- the prose is not. So up to v1.9.2 it still listed the v1.9.0
features as "NEW" under whatever version was being built, because there was
nowhere to leave a reminder beside the text that needed rewriting. Found in the
owner's acceptance pass, 2026-09-17.

A line whose first non-blank characters are ``{{!`` is for whoever maintains the
template and is stripped before the file is generated.

Run with: pytest tests/test_installer_template_comments.py -v
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "installer" / "templates"
GENERATED = REPO / "installer" / "generated"

sys.path.insert(0, str(REPO / "installer"))


@pytest.fixture(scope="module")
def strip():
    from build_release import ReleaseBuilder
    return ReleaseBuilder._strip_template_comments


class TestTheStripper:
    def test_a_comment_line_is_removed(self, strip):
        assert strip("keep me\n{{! not this\nkeep me too\n") == "keep me\nkeep me too\n"

    def test_indented_comments_are_removed_too(self, strip):
        assert strip("a\n    {{! hidden\nb\n") == "a\nb\n"

    def test_ordinary_placeholders_are_untouched(self, strip):
        # {{VERSION}} must survive; only {{! is a comment.
        text = "WhisperJAV v{{VERSION}}\n"
        assert strip(text) == text

    def test_a_brace_in_the_middle_of_a_line_is_not_a_comment(self, strip):
        text = "see {{! this stays because it is not at the start\n"
        assert strip(text) == text

    def test_nothing_else_changes(self, strip):
        text = "one\ntwo\nthree\n"
        assert strip(text) == text


class TestNothingLeaksToTheUser:
    """The generated files are what a user actually opens."""

    def test_no_template_comment_reaches_a_generated_file(self):
        generated = list(GENERATED.glob("*")) if GENERATED.exists() else []
        if not generated:
            pytest.skip("installer/generated is empty; run build_release.py first")

        offenders = []
        for path in generated:
            if path.suffix.lower() in (".exe", ".whl", ".ico"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("{{!"):
                    offenders.append(f"{path.name}:{number}")
        assert offenders == [], f"maintainer notes reached the user: {offenders}"

    def test_the_readme_does_not_still_advertise_the_old_release(self):
        readme = next(GENERATED.glob("README_INSTALLER_v*.txt"), None)
        if readme is None:
            pytest.skip("no generated installer README; run build_release.py first")
        text = readme.read_text(encoding="utf-8", errors="replace")

        start = text.index("WHAT'S NEW IN v")
        end = text.index("WHAT THIS INSTALLER DOES", start)
        whats_new = text[start:end]

        # The v1.9.0 headings that were still being sold as new in v1.9.2.
        for stale in ("QWEN3-ASR PIPELINE (NEW)",
                      "ADAPTIVE STEP-DOWN ARCHITECTURE (NEW)",
                      "ALIGNMENT SENTINEL (NEW)"):
            assert stale not in whats_new, (
                f"the installer README still advertises {stale!r} as new")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
