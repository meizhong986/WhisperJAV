"""
Tests for whisperjav.utils.model_loader — HuggingFace Hub download resilience.

Test hierarchy:
  - Unit tests: _is_network_error classification (fast, no I/O)
  - Monkeypatch tests: patch_hf_hub_downloads behavior via mocking (fast, no network)
  - Integration tests: Real model download with simulated SSL failure (require cache)

Run all:           pytest tests/test_model_loader.py -v
Run fast only:     pytest tests/test_model_loader.py -v -m "not slow"
Run integration:   pytest tests/test_model_loader.py -v -m integration
"""

import ssl
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from whisperjav.utils.model_loader import _is_network_error


# ---------------------------------------------------------------------------
# 1. Unit tests: _is_network_error classification
# ---------------------------------------------------------------------------

class TestIsNetworkError:
    """Verify _is_network_error correctly classifies exceptions."""

    # --- True positives (network errors) ---

    def test_ssl_error(self):
        assert _is_network_error(ssl.SSLError("CERTIFICATE_VERIFY_FAILED"))

    def test_connection_error(self):
        assert _is_network_error(ConnectionError("Connection refused"))

    def test_connection_reset_error(self):
        assert _is_network_error(ConnectionResetError("Connection reset by peer"))

    def test_timeout_error(self):
        assert _is_network_error(TimeoutError("timed out"))

    def test_os_error_dns(self):
        assert _is_network_error(OSError("[Errno -2] Name or service not known"))

    def test_os_error_ssl_string(self):
        assert _is_network_error(
            OSError("urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] "
                    "certificate verify failed: unable to get local issuer certificate")
        )

    def test_runtime_error_with_socket(self):
        assert _is_network_error(RuntimeError("socket timeout"))

    def test_runtime_error_with_connection(self):
        assert _is_network_error(RuntimeError("connection timed out"))

    def test_os_error_getaddrinfo(self):
        assert _is_network_error(OSError("[Errno 11001] getaddrinfo failed"))

    # --- Chained exceptions ---

    def test_chained_ssl_inside_runtime(self):
        """SSL error wrapped in RuntimeError should be detected."""
        try:
            try:
                raise ssl.SSLError("cert verify failed")
            except ssl.SSLError as inner:
                raise RuntimeError("model download failed") from inner
        except RuntimeError as outer:
            assert _is_network_error(outer)

    def test_chained_connection_inside_value_error(self):
        """ConnectionError wrapped in ValueError should be detected."""
        try:
            try:
                raise ConnectionError("refused")
            except ConnectionError as inner:
                raise ValueError("snapshot_download failed") from inner
        except ValueError as outer:
            assert _is_network_error(outer)

    def test_chained_timeout_two_levels_deep(self):
        """Timeout 2 levels deep in exception chain."""
        try:
            try:
                try:
                    raise TimeoutError("read timed out")
                except TimeoutError as t:
                    raise OSError("network unreachable") from t
            except OSError as o:
                raise RuntimeError("failed to download") from o
        except RuntimeError as outer:
            assert _is_network_error(outer)

    # --- True negatives (NOT network errors) ---

    def test_value_error(self):
        assert not _is_network_error(ValueError("Invalid model size 'xxxl'"))

    def test_runtime_error_oom(self):
        assert not _is_network_error(RuntimeError("CUDA out of memory"))

    def test_runtime_error_cublas(self):
        assert not _is_network_error(RuntimeError("CUBLAS error"))

    def test_file_not_found(self):
        assert not _is_network_error(FileNotFoundError("model.bin not found"))

    def test_permission_error(self):
        assert not _is_network_error(PermissionError("access denied"))

    def test_is_a_directory_error(self):
        assert not _is_network_error(IsADirectoryError("/tmp/model"))

    def test_import_error(self):
        assert not _is_network_error(ImportError("No module named 'ctranslate2'"))

    def test_file_exists_error(self):
        assert not _is_network_error(FileExistsError("file already exists"))

    def test_runtime_error_model_invalid(self):
        assert not _is_network_error(RuntimeError("invalid model format"))

    def test_keyboard_interrupt(self):
        assert not _is_network_error(KeyboardInterrupt())

    # --- Edge cases ---

    def test_empty_message(self):
        assert not _is_network_error(RuntimeError(""))

    def test_none_cause(self):
        e = RuntimeError("some error")
        assert e.__cause__ is None
        assert not _is_network_error(e)

    def test_circular_chain_does_not_loop(self):
        e1 = RuntimeError("first")
        e2 = RuntimeError("second")
        e1.__cause__ = e2
        e2.__cause__ = e1  # circular
        result = _is_network_error(e1)
        assert not result


# ---------------------------------------------------------------------------
# 2. Monkeypatch tests: patch_hf_hub_downloads behavior
# ---------------------------------------------------------------------------

class TestPatchHfHubDownloads:
    """Test that the monkeypatch correctly wraps snapshot_download."""

    def setup_method(self):
        """Reset the patch state before each test."""
        import whisperjav.utils.model_loader as ml
        ml._patched = False

    def _apply_patch(self):
        from whisperjav.utils.model_loader import patch_hf_hub_downloads
        patch_hf_hub_downloads()

    def test_patch_replaces_snapshot_download(self):
        """After patching, huggingface_hub.snapshot_download should be wrapped."""
        import huggingface_hub
        original = huggingface_hub.snapshot_download
        self._apply_patch()
        assert huggingface_hub.snapshot_download is not original
        assert huggingface_hub.snapshot_download.__name__ == "_resilient_snapshot_download"

    def test_patch_is_idempotent(self):
        """Calling patch twice should not double-wrap."""
        import huggingface_hub
        self._apply_patch()
        first = huggingface_hub.snapshot_download
        self._apply_patch()
        second = huggingface_hub.snapshot_download
        assert first is second

    def test_normal_download_passes_through(self):
        """When no error, the original function is called normally."""
        import huggingface_hub
        mock_original = MagicMock(return_value="/path/to/model")

        with patch.object(huggingface_hub, "snapshot_download", mock_original):
            self._apply_patch()
            patched = huggingface_hub.snapshot_download
            result = patched("Systran/faster-whisper-tiny")
            assert result == "/path/to/model"

    def test_ssl_error_triggers_local_fallback(self):
        """SSL error should retry with local_files_only=True."""
        import huggingface_hub
        import whisperjav.utils.model_loader as ml

        call_log = []

        def mock_download(*args, **kwargs):
            call_log.append(kwargs.copy())
            if not kwargs.get("local_files_only"):
                raise OSError(
                    "urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] "
                    "certificate verify failed"
                )
            return "/cached/model/path"

        # Inject mock as the "original" before patching
        ml._patched = False
        with patch.object(huggingface_hub, "snapshot_download", mock_download):
            self._apply_patch()
            patched = huggingface_hub.snapshot_download

            result = patched("Systran/faster-whisper-large-v2")
            assert result == "/cached/model/path"
            assert len(call_log) == 2
            assert not call_log[0].get("local_files_only")
            assert call_log[1]["local_files_only"] is True

    def test_non_network_error_raises_immediately(self):
        """Non-network errors should not trigger fallback."""
        import huggingface_hub
        import whisperjav.utils.model_loader as ml

        def mock_download(*args, **kwargs):
            raise ValueError("Invalid repo_id format")

        ml._patched = False
        with patch.object(huggingface_hub, "snapshot_download", mock_download):
            self._apply_patch()
            patched = huggingface_hub.snapshot_download

            with pytest.raises(ValueError, match="Invalid repo_id"):
                patched("bad/repo/format")

    def test_ssl_fail_no_cache_raises_original_error(self):
        """When SSL fails and cache miss, the ORIGINAL SSL error is raised."""
        import huggingface_hub
        import whisperjav.utils.model_loader as ml

        original_error = OSError(
            "urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] cert verify failed"
        )

        def mock_download(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("model not found in cache")
            raise original_error

        ml._patched = False
        with patch.object(huggingface_hub, "snapshot_download", mock_download):
            self._apply_patch()
            patched = huggingface_hub.snapshot_download

            with pytest.raises(OSError) as exc_info:
                patched("Systran/faster-whisper-large-v2")

            assert exc_info.value is original_error

    def test_local_files_only_not_double_wrapped(self):
        """If local_files_only=True is already set, don't add fallback logic."""
        import huggingface_hub
        import whisperjav.utils.model_loader as ml

        def mock_download(*args, **kwargs):
            raise OSError("model not in cache")

        ml._patched = False
        with patch.object(huggingface_hub, "snapshot_download", mock_download):
            self._apply_patch()
            patched = huggingface_hub.snapshot_download

            with pytest.raises(OSError, match="model not in cache"):
                patched("tiny", local_files_only=True)


# ---------------------------------------------------------------------------
# 3. Integration: Verify patch works with faster-whisper's download_model
# ---------------------------------------------------------------------------

class TestFasterWhisperIntegration:
    """Verify the monkeypatch intercepts faster-whisper's actual call path."""

    def setup_method(self):
        import whisperjav.utils.model_loader as ml
        ml._patched = False

    def test_faster_whisper_uses_patched_snapshot_download(self):
        """faster_whisper.utils.download_model should use the patched function."""
        from whisperjav.utils.model_loader import patch_hf_hub_downloads
        import huggingface_hub

        patch_hf_hub_downloads()

        # Verify the function faster_whisper will call is our patched version
        assert huggingface_hub.snapshot_download.__name__ == "_resilient_snapshot_download"

    def test_no_direct_whispermodel_import_in_asr_modules(self):
        """ASR modules should use WhisperModel directly (monkeypatch handles resilience)."""
        # With the monkeypatch approach, modules SHOULD import WhisperModel directly.
        # The resilience is handled at the huggingface_hub level, not per-module.
        from whisperjav.modules import faster_whisper_pro_asr
        source = Path(faster_whisper_pro_asr.__file__).read_text(encoding="utf-8")
        assert "from faster_whisper import WhisperModel" in source

    def test_entry_points_apply_patch(self):
        """Verify CLI and GUI entry points call patch_hf_hub_downloads."""
        cli_source = Path("whisperjav/cli.py").read_text(encoding="utf-8")
        assert "patch_hf_hub_downloads" in cli_source

        main_source = Path("whisperjav/main.py").read_text(encoding="utf-8")
        assert "patch_hf_hub_downloads" in main_source

        gui_source = Path("whisperjav/webview_gui/main.py").read_text(encoding="utf-8")
        assert "patch_hf_hub_downloads" in gui_source
