#!/usr/bin/env python3
"""
--hf-endpoint sends Hugging Face downloads to an address the user chooses.

For machines that cannot reach huggingface.co. huggingface_hub reads HF_ENDPOINT
when it is IMPORTED, so the flag cannot wait for argparse: the entry points scan
the raw argument list first, exactly as --offline does. If that scan is wrong, the
flag is accepted and silently does nothing, which is the failure worth testing
for.

WhisperJAV names no mirror. The flag takes whatever address the user gives it.

Run with: pytest tests/test_hf_endpoint_flag.py -v
"""

import pytest

from whisperjav.utils.offline_mode import (
    enable_hf_endpoint,
    hf_endpoint,
    hf_endpoint_requested,
)

MIRROR = "https://example-mirror.invalid"


class TestReadingItOffTheCommandLine:
    def test_separate_argument(self):
        assert hf_endpoint_requested(["video.mp4", "--hf-endpoint", MIRROR]) == MIRROR

    def test_equals_form(self):
        assert hf_endpoint_requested(["video.mp4", f"--hf-endpoint={MIRROR}"]) == MIRROR

    def test_absent(self):
        assert hf_endpoint_requested(["video.mp4", "--mode", "fidelity"]) is None

    def test_a_prefix_argparse_would_accept(self):
        # argparse takes any unambiguous prefix, and this scan runs before it.
        assert hf_endpoint_requested(["--hf-end", MIRROR]) == MIRROR

    def test_the_flag_with_nothing_after_it(self):
        # argparse will reject this; the scan must not crash or invent a value.
        assert hf_endpoint_requested(["video.mp4", "--hf-endpoint"]) is None

    def test_an_empty_value(self):
        assert hf_endpoint_requested(["--hf-endpoint", ""]) is None
        assert hf_endpoint_requested(["--hf-endpoint="]) is None

    def test_it_does_not_claim_another_flag(self):
        assert hf_endpoint_requested(["--offline"]) is None
        assert hf_endpoint_requested(["--help"]) is None


class TestApplyingIt:
    def test_it_reaches_the_environment(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        enable_hf_endpoint(MIRROR)
        assert hf_endpoint() == MIRROR

    def test_a_trailing_slash_is_dropped(self, monkeypatch):
        # huggingface_hub builds URLs by appending, so a trailing slash would
        # produce a double slash in every request.
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        enable_hf_endpoint(MIRROR + "/")
        assert hf_endpoint() == MIRROR

    def test_nothing_set_means_the_default(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        assert hf_endpoint() == ""


class TestOfflineIsUntouched:
    """The two flags are independent: a mirror is not offline mode."""

    def test_the_endpoint_scan_does_not_trigger_offline(self):
        from whisperjav.utils.offline_mode import offline_requested
        assert offline_requested(["--hf-endpoint", MIRROR]) is False

    def test_the_offline_scan_does_not_trigger_the_endpoint(self):
        assert hf_endpoint_requested(["--offline"]) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
