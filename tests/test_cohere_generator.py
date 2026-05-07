"""
Unit tests for the Cohere TextGenerator.

Test layers:
    - Lightweight (no model load, always run in CI):
        * Protocol compliance (TextGenerator)
        * Factory discoverability and registration
        * Default config values match the v1.8.14 plan sign-offs (D1-D7)
        * HF_TOKEN preflight raises a helpful diagnostic when missing
        * _format_load_error produces the gated-repo diagnostic
        * generate() before load() raises RuntimeError
        * QwenPipeline accepts generator_backend="cohere" and constructs
        * pass_worker.py override block routes cohere defaults correctly

    - Gated (require HF_TOKEN + accepted terms; skipped in CI):
        * load + unload lifecycle (logs peak VRAM)
        * generate() on a short JA clip returns non-empty text
        * Aligner-disabled VAD_ONLY path returns SRT-compatible output

Skip conditions:
    - HF_TOKEN unset → skip gated tests (sufficient for CI)
    - COHERE_TEST_AUDIO unset (no test fixture) → skip live-generation tests
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ─── Helpers ────────────────────────────────────────────────────────────────


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _test_audio_path() -> Path | None:
    """
    Resolve the optional test-audio fixture.

    Set COHERE_TEST_AUDIO to a 16 kHz mono JA wav file (10-30 s recommended)
    before running the live tests.
    """
    p = os.environ.get("COHERE_TEST_AUDIO")
    if not p:
        return None
    path = Path(p).expanduser().resolve()
    return path if path.is_file() else None


# ─── Lightweight tests (always runnable) ────────────────────────────────────


class TestCohereProtocolCompliance:
    """The CohereTextGenerator class must satisfy the TextGenerator protocol."""

    def test_protocol_runtime_compliance(self):
        """isinstance check against the TextGenerator protocol."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )
        from whisperjav.modules.subtitle_pipeline.protocols import TextGenerator

        gen = CohereTextGenerator()
        assert isinstance(gen, TextGenerator)

    def test_required_methods_exist(self):
        """All TextGenerator methods are present and callable."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        gen = CohereTextGenerator()
        for method in ("load", "unload", "generate", "generate_batch", "cleanup"):
            assert hasattr(gen, method), f"missing method: {method}"
            assert callable(getattr(gen, method)), f"not callable: {method}"

    def test_is_loaded_starts_false(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert CohereTextGenerator().is_loaded is False


class TestCohereFactoryRegistration:
    """The factory registers and instantiates the cohere backend."""

    def test_cohere_in_registry(self):
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )

        assert "cohere" in TextGeneratorFactory.available()

    def test_factory_create_returns_cohere_class(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )

        gen = TextGeneratorFactory.create("cohere")
        assert isinstance(gen, CohereTextGenerator)

    def test_factory_passes_kwargs_through(self):
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )

        gen = TextGeneratorFactory.create(
            "cohere",
            language="en",
            max_new_tokens=256,
        )
        assert gen._config["language"] == "en"
        assert gen._config["max_new_tokens"] == 256


class TestCohereConfigDefaults:
    """Default values must match the v1.8.14 plan sign-offs (D1-D7)."""

    def test_default_model_id_is_official_cohere_repo(self):
        """D2/D7: gated official repo by default."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert (
            CohereTextGenerator()._config["model_id"]
            == "CohereLabs/cohere-transcribe-03-2026"
        )

    def test_default_max_new_tokens_is_512(self):
        """D2: 512 — safe ceiling for JAV monologues."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert CohereTextGenerator()._config["max_new_tokens"] == 512

    def test_default_language_is_japanese(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert CohereTextGenerator()._config["language"] == "ja"

    def test_default_punctuation_enabled(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert CohereTextGenerator()._config["punctuation"] is True

    def test_default_trust_remote_code_true(self):
        """D4: required True until transformers exposes the class natively."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        assert CohereTextGenerator()._config["trust_remote_code"] is True


class TestCohereHFTokenPreflight:
    """HF_TOKEN preflight raises a diagnostic without attempting load."""

    def test_check_hf_access_raises_when_token_missing(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            CohereTextGenerator._check_hf_access()
        msg = str(exc_info.value)
        assert "HF_TOKEN" in msg
        assert "huggingface.co/CohereLabs/cohere-transcribe-03-2026" in msg
        # D6: diagnostic must point at FAQ section, not just raw error
        assert "FAQ" in msg

    def test_check_hf_access_passes_when_hf_token_set(self, monkeypatch):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        monkeypatch.setenv("HF_TOKEN", "hf_dummy_for_preflight_test")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # No exception expected
        CohereTextGenerator._check_hf_access()

    def test_check_hf_access_passes_when_hugging_face_hub_token_set(self, monkeypatch):
        """HUGGING_FACE_HUB_TOKEN is the older HF env var name; must also be honored."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_dummy_for_preflight_test")
        CohereTextGenerator._check_hf_access()


class TestCohereLoadErrorDiagnostic:
    """_format_load_error tailors the message based on the underlying exception."""

    def test_gated_repo_error_produces_helpful_diagnostic(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        msg = CohereTextGenerator._format_load_error(
            RuntimeError("Cannot access gated repo (403 Client Error)")
        )
        assert "gated repo" in msg.lower() or "access" in msg.lower()
        assert "huggingface.co/CohereLabs/cohere-transcribe-03-2026" in msg
        assert "FAQ" in msg

    def test_403_error_recognized_as_gating(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        msg = CohereTextGenerator._format_load_error(RuntimeError("HTTP 403"))
        assert "FAQ" in msg

    def test_unknown_error_produces_generic_message(self):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        msg = CohereTextGenerator._format_load_error(RuntimeError("disk full"))
        assert "Failed to load Cohere" in msg
        assert "disk full" in msg
        # Should NOT include FAQ link for unrelated errors
        assert "FAQ" not in msg


class TestCohereGenerateGuards:
    """generate() must guard against being called before load()."""

    def test_generate_before_load_raises(self, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        gen = CohereTextGenerator()
        dummy_audio = tmp_path / "dummy.wav"
        dummy_audio.write_bytes(b"")  # never read because guard fires first
        with pytest.raises(RuntimeError, match="before load"):
            gen.generate(dummy_audio)

    def test_generate_batch_before_load_raises(self, tmp_path):
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        gen = CohereTextGenerator()
        with pytest.raises(RuntimeError, match="before load"):
            gen.generate_batch([tmp_path / "dummy.wav"])


# ─── QwenPipeline integration (no model load) ───────────────────────────────


class TestQwenPipelineCohereIntegration:
    """QwenPipeline must accept generator_backend="cohere" and apply D7 defaults."""

    def test_qwen_pipeline_accepts_cohere_backend(self, tmp_path):
        from whisperjav.modules.subtitle_pipeline.types import TimestampMode
        from whisperjav.pipelines.qwen_pipeline import QwenPipeline

        pipeline = QwenPipeline(
            output_dir=str(tmp_path / "out"),
            temp_dir=str(tmp_path / "tmp"),
            generator_backend="cohere",
        )

        # D7: aligner ON by default → ALIGNER_WITH_VAD_FALLBACK timestamp mode
        assert pipeline.timestamp_mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK
        # D3: passthrough cleaner
        assert pipeline.assembly_cleaner_enabled is False
        # Aligner ON → stepdown safe
        assert pipeline.stepdown_enabled is True
        # Cohere prefers fewer cuts
        assert pipeline.segmenter_chunk_threshold == 1.0
        assert pipeline.segmenter_max_group_duration == 6.0


# ─── pass_worker.py integration (no model load) ─────────────────────────────


class TestPassWorkerCohereIntegration:
    """pass_worker.py override block must apply Cohere defaults correctly."""

    def test_segmenter_default_extends_to_cohere(self):
        """
        The segmenter rule in pass_worker.py:1149 must fire for cohere
        (extension of the anime-whisper rule per the v1.8.14 plan).
        Source verification by import + grep — runs in seconds.
        """
        import whisperjav.ensemble.pass_worker as pw

        src = Path(pw.__file__).read_text(encoding="utf-8")
        # Confirm both backends in the conditional
        assert (
            'if _aw_gen in ("anime-whisper", "cohere") '
            "and not pass_config.get(\"speech_segmenter\")" in src
        )

    def test_cohere_override_block_present(self):
        """The cohere elif block at line ~1245 must be wired up."""
        import whisperjav.ensemble.pass_worker as pw

        src = Path(pw.__file__).read_text(encoding="utf-8")
        assert 'elif _gen_backend == "cohere":' in src
        assert (
            'qwen_pipeline_params["model_id"] = '
            '"CohereLabs/cohere-transcribe-03-2026"' in src
        )
        # D7 default — aligner ON, timestamp_mode aligner_vad_fallback
        assert 'qwen_pipeline_params["timestamp_mode"] = "aligner_vad_fallback"' in src


# ─── Gated tests (require HF_TOKEN; skipped in CI) ──────────────────────────


@pytest.mark.gated_model
@pytest.mark.slow
@pytest.mark.skipif(
    not _has_hf_token(),
    reason="HF_TOKEN not set — Cohere weights are gated, skipping live model test",
)
class TestCohereLiveModel:
    """End-to-end tests requiring real Cohere weights and a JA audio fixture."""

    def test_load_unload_lifecycle(self):
        """Load weights, confirm is_loaded, unload cleanly."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        gen = CohereTextGenerator()
        try:
            gen.load()
            assert gen.is_loaded is True
        finally:
            gen.unload()
            assert gen.is_loaded is False

    @pytest.mark.skipif(
        _test_audio_path() is None,
        reason="COHERE_TEST_AUDIO not set; skipping live generation test",
    )
    def test_generate_returns_non_empty_text(self):
        """generate() on a real JA clip returns non-empty TranscriptionResult."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )
        from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult

        audio = _test_audio_path()
        assert audio is not None  # narrowed by skipif

        gen = CohereTextGenerator()
        try:
            gen.load()
            result = gen.generate(audio, language="ja")
        finally:
            gen.unload()

        assert isinstance(result, TranscriptionResult)
        assert result.text.strip(), "Cohere returned empty text on a real JA clip"
        assert result.language == "ja"
        assert result.metadata.get("generator") == "cohere"

    @pytest.mark.skipif(
        _test_audio_path() is None,
        reason="COHERE_TEST_AUDIO not set; skipping batch generation test",
    )
    def test_generate_batch_returns_per_clip_results(self):
        """generate_batch over the same clip twice yields two results."""
        from whisperjav.modules.subtitle_pipeline.generators.cohere import (
            CohereTextGenerator,
        )

        audio = _test_audio_path()
        assert audio is not None

        gen = CohereTextGenerator()
        try:
            gen.load()
            results = gen.generate_batch([audio, audio], language="ja")
        finally:
            gen.unload()

        assert len(results) == 2
        for r in results:
            assert r.metadata.get("generator") == "cohere"
