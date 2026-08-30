"""Regression tests for the small verified fixes in v1.9.2.

Each test names the issue it guards. They are grouped here rather than scattered
because they share nothing but their size — the alternative was five new files of
one test each.
"""

import pytest

# ---------------------------------------------------------------------------
# #340 — Path.resolve() raises WinError 1005 on CloudDrive2 virtual volumes
# ---------------------------------------------------------------------------

class TestCanonicalPathGuard:
    """#340: an unresolvable path must not abort discovery before it starts."""

    def test_falls_back_when_resolve_raises_oserror(self, tmp_path, monkeypatch):
        from pathlib import Path

        from whisperjav.modules import media_discovery

        target = tmp_path / "clip.mp4"
        target.write_bytes(b"")

        def _boom(self, *args, **kwargs):
            raise OSError(
                1005, "The volume does not contain a recognized file system."
            )

        monkeypatch.setattr(Path, "resolve", _boom)

        result = media_discovery._canonical_path(target)

        assert result.is_absolute()
        assert result.name == "clip.mp4"

    def test_falls_back_on_valueerror(self, tmp_path, monkeypatch):
        from pathlib import Path

        from whisperjav.modules import media_discovery

        target = tmp_path / "clip.mp4"
        target.write_bytes(b"")
        monkeypatch.setattr(
            Path, "resolve", lambda self, *a, **k: (_ for _ in ()).throw(ValueError("bad"))
        )

        assert media_discovery._canonical_path(target).is_absolute()

    def test_normal_path_still_resolves(self, tmp_path):
        from whisperjav.modules import media_discovery

        target = tmp_path / "clip.mp4"
        target.write_bytes(b"")
        assert media_discovery._canonical_path(target) == target.resolve()

    def test_deduplication_still_works_under_fallback(self, tmp_path, monkeypatch):
        """The fallback must still collapse two spellings of the same file."""
        from pathlib import Path

        from whisperjav.modules import media_discovery

        target = tmp_path / "clip.mp4"
        target.write_bytes(b"")
        monkeypatch.setattr(
            Path, "resolve", lambda self, *a, **k: (_ for _ in ()).throw(OSError(1005, "x"))
        )

        a = media_discovery._canonical_path(target)
        b = media_discovery._canonical_path(Path(str(tmp_path) + "/./clip.mp4"))
        assert a == b, "abspath fallback should normalise './' segments"


# ---------------------------------------------------------------------------
# #341 — min_batch_size must be less than max_batch_size
# ---------------------------------------------------------------------------

class TestBatchSizeFloor:
    """#341: small-context Ollama models produced an unusable batch window.

    PySubtrans defaults ``min_batch_size`` to 10 and raises if it exceeds
    ``max_batch_size``.  WhisperJAV caps the maximum to fit the context window
    but never touched the minimum, so any model at <=4096 context produced
    max=5 and a guaranteed ValueError.
    """

    @pytest.mark.parametrize("n_ctx", [2048, 4096])
    def test_small_contexts_produce_a_usable_floor(self, n_ctx):
        from whisperjav.translate.core import (
            cap_batch_size_for_context,
            resolve_batch_window,
        )

        max_batch = cap_batch_size_for_context(30, n_ctx)
        min_batch, max_batch = resolve_batch_window(max_batch)
        assert min_batch < max_batch, (
            f"n_ctx={n_ctx} still yields min={min_batch} >= max={max_batch}"
        )
        assert min_batch >= 1

    @pytest.mark.parametrize("n_ctx", [8192, 16384, 32768])
    def test_large_contexts_keep_the_default_floor(self, n_ctx):
        """Where the default floor already fits, it must be left alone."""
        from whisperjav.translate.core import (
            PYSUBTRANS_DEFAULT_MIN_BATCH,
            cap_batch_size_for_context,
            resolve_batch_window,
        )

        max_batch = cap_batch_size_for_context(30, n_ctx)
        min_batch, _ = resolve_batch_window(max_batch)
        assert min_batch == PYSUBTRANS_DEFAULT_MIN_BATCH

    def test_degenerate_max_is_clamped(self):
        """A ceiling of 1 leaves no window; the floor must still be valid."""
        from whisperjav.translate.core import resolve_batch_window

        min_batch, max_batch = resolve_batch_window(1)
        assert min_batch >= 1
        assert min_batch <= max_batch

    def test_fallback_mirrors_the_curated_list(self):
        """The fallback must not drift from config/ollama_models.json."""
        import json
        from pathlib import Path

        from whisperjav.webview_gui.api import WhisperJAVAPI

        curated_path = (
            Path(__file__).resolve().parent.parent
            / "whisperjav" / "config" / "ollama_models.json"
        )
        curated = {m["model"] for m in json.loads(curated_path.read_text(encoding="utf-8"))}
        fallback = {m["model"] for m in WhisperJAVAPI._CURATED_MODELS_FALLBACK}
        assert fallback <= curated, (
            "the Ollama fallback offers models absent from the curated list: "
            f"{sorted(fallback - curated)}"
        )


# ---------------------------------------------------------------------------
# Curated Ollama fallback must not recommend a thinking model
# ---------------------------------------------------------------------------

class TestCuratedModelsFallback:
    """The fallback list is served when ollama_models.json cannot be read.

    It still led with a Qwen3-based model — the class removed from the curated
    list in v1.8.11 because chain-of-thought leaks into the SRT. #305's reporter
    was running exactly that model when they reported English output.
    """

    def test_fallback_contains_no_thinking_models(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI

        fallback = WhisperJAVAPI._CURATED_MODELS_FALLBACK
        assert fallback, "fallback list must not be empty"

        banned = ("qwen3", "shisa", "reasoner", "-r1", "deepseek-r")
        for entry in fallback:
            model = entry["model"].lower()
            for token in banned:
                assert token not in model, (
                    f"{entry['model']!r} is a reasoning/thinking model; these "
                    f"emit chain-of-thought into subtitles (v1.8.11 curation)"
                )

    def test_fallback_entries_are_well_formed(self):
        from whisperjav.webview_gui.api import WhisperJAVAPI

        for entry in WhisperJAVAPI._CURATED_MODELS_FALLBACK:
            assert {"model", "size", "label"} <= set(entry)


# ---------------------------------------------------------------------------
# #306 — an unknown speech-enhancer name silently became "none"
# ---------------------------------------------------------------------------

class TestUnknownEnhancerIsRejectedAtTheBoundary:
    """#306: `--pass2-speech-enhancer zipenhance` (a typo for `zipenhancer`) was
    accepted, silently downgraded to no enhancement, and the user spent a
    multi-hour run believing enhancement was active.

    The fallback in pipeline_helper is correct for a *known* backend whose
    dependencies are missing, so it is left alone.  An unknown *name* is a user
    error and belongs at the argument boundary, which is where `--qwen-enhancer`
    already caught it via `choices=`.
    """

    @pytest.mark.parametrize("flag", [
        "--pass1-speech-enhancer",
        "--pass2-speech-enhancer",
        "--qwen-enhancer",
    ])
    def test_typo_is_rejected_by_argparse(self, flag):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4", flag, "zipenhance"],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode != 0, f"{flag} accepted an unknown backend name"
        combined = result.stdout + result.stderr
        assert "invalid choice" in combined
        assert "zipenhancer" in combined, "the error should name the real backend"

    def test_cli_choices_match_the_backend_registry(self):
        """The CLI list is hardcoded to keep `--help` cheap; this catches drift."""
        from whisperjav.main import SPEECH_ENHANCER_CHOICES
        from whisperjav.modules.speech_enhancement.factory import (
            SpeechEnhancerFactory,
        )

        assert sorted(SPEECH_ENHANCER_CHOICES) == sorted(
            SpeechEnhancerFactory.list_backends()
        ), (
            "SPEECH_ENHANCER_CHOICES in main.py has drifted from "
            "_BACKEND_REGISTRY in speech_enhancement/factory.py"
        )

    def test_missing_dependency_still_falls_back_quietly(self):
        """A known backend with absent deps must still degrade, not raise."""
        from whisperjav.modules.speech_enhancement.factory import (
            SpeechEnhancerFactory,
        )

        ok, hint = SpeechEnhancerFactory.is_backend_available("clearvoice")
        # Whatever the answer on this machine, it must be a usable pair and must
        # not be reported as an unknown backend.
        assert isinstance(ok, bool)
        assert "Unknown backend" not in hint
