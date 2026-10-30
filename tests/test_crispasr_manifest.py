"""
The model files WhisperJAV downloads for its CrispASR pipelines, pinned (whisperjav/crispasr/manifest.py).

These tests keep its entries well formed and its choices consistent, so an edit at release time cannot
leave a model that no code path can use, or a pipeline offering a model that does not exist. CrispASR
itself is pinned in uv.lock (route B), not here.

Run with: python -m pytest tests/test_crispasr_manifest.py -q
"""
import re

import pytest

from whisperjav.crispasr import manifest
from whisperjav.crispasr.manifest import (
    CRISPASR_PIPELINES,
    JAPANESE_NAMES,
    MODELS,
    PIPELINE_DEFAULT_MODEL,
    PIPELINE_MODELS,
    model_for,
)

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class TestModels:
    def test_every_model_file_is_pinned(self):
        assert MODELS
        for name, m in MODELS.items():
            assert m.name == name
            assert HEX40.fullmatch(m.revision), m
            assert HEX64.fullmatch(m.sha256), m
            assert m.size > 0, m
            assert m.filename.endswith(".gguf"), m
            assert re.fullmatch(r"[\w.-]+/[\w.-]+", m.repo_id), m
            assert m.label and m.backend and m.licence, m

    def test_every_pipeline_model_exists_and_runs_on_its_back_end(self):
        backends = {"parakeet-crispasr": "parakeet", "cohere-crispasr": "cohere"}
        assert set(PIPELINE_MODELS) == set(backends)
        for pipeline, names in PIPELINE_MODELS.items():
            assert names, pipeline
            for name in names:
                assert MODELS[name].backend == backends[pipeline], (pipeline, name)

    def test_each_default_is_one_of_its_pipelines_models(self):
        assert set(PIPELINE_DEFAULT_MODEL) == set(PIPELINE_MODELS)
        for pipeline, default in PIPELINE_DEFAULT_MODEL.items():
            assert default in PIPELINE_MODELS[pipeline], pipeline
        assert CRISPASR_PIPELINES == tuple(PIPELINE_MODELS)

    def test_the_defaults(self):
        # Cohere v0.1 (W129 1)
        assert PIPELINE_DEFAULT_MODEL == {"parakeet-crispasr": "parakeet-tdt-0.6b-ja",
                                          "cohere-crispasr": "cohere-asr-ja-v0.1"}

    def test_the_offered_models_without_the_ctc_model(self):
        # The Parakeet CTC model is not offered (W129 2).
        assert PIPELINE_MODELS == {"parakeet-crispasr": ("parakeet-tdt-0.6b-ja",),
                                   "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja")}
        assert set(MODELS) == {"parakeet-tdt-0.6b-ja", "cohere-asr-ja-v0.1", "cohere-asr-ja"}

    def test_nothing_of_the_program_route_is_left(self):
        for name in ("PROGRAM_BUILDS", "CRISPASR_VERSION", "CRISPASR_GIT", "CUDA13_RUNTIME_FILES",
                     "MEASUREMENT_MODELS", "builds_for_this_machine"):
            assert not hasattr(manifest, name), name

    def test_the_japanese_names(self):
        assert JAPANESE_NAMES == {"ja", "jp", "jpn", "japanese"}


class TestModelFor:
    @pytest.mark.parametrize("pipeline, default", [("parakeet-crispasr", "parakeet-tdt-0.6b-ja"),
                                                   ("cohere-crispasr", "cohere-asr-ja-v0.1")])
    def test_a_pipeline_without_a_named_model_gets_its_default(self, pipeline, default):
        assert model_for(pipeline, None) is MODELS[default]
        assert model_for(pipeline, "") is MODELS[default]

    @pytest.mark.parametrize("pipeline, name", [
        ("parakeet-crispasr", "parakeet-tdt-0.6b-ja"),
        ("cohere-crispasr", "cohere-asr-ja-v0.1"),
        ("cohere-crispasr", "cohere-asr-ja"),
    ])
    def test_a_named_model(self, pipeline, name):
        assert model_for(pipeline, name) is MODELS[name]

    def test_an_unknown_name_is_refused_with_the_choices(self):
        with pytest.raises(ValueError) as caught:
            model_for("parakeet-crispasr", "parakeet-ctc-1.1b-ja")
        assert str(caught.value) == ("'parakeet-ctc-1.1b-ja' is not a model of parakeet-crispasr; "
                                     "choose one of: parakeet-tdt-0.6b-ja")

    def test_a_model_of_another_pipeline_is_refused(self):
        with pytest.raises(ValueError, match="is not a model of parakeet-crispasr"):
            model_for("parakeet-crispasr", "cohere-asr-ja")


class TestAlignerModels:
    """ChronosJAV's CrispASR aligners (item 12 C1; design note 2026-10-28, section 4.3; W248 2, W255)."""

    def test_every_aligner_file_is_pinned(self):
        assert set(manifest.ALIGNER_MODELS) == {"wav2vec2-ja", "parakeet-ctc-ja", "qwen3-fa-gguf"}
        for name, m in manifest.ALIGNER_MODELS.items():
            assert m.name == name
            assert HEX40.fullmatch(m.revision), m
            assert HEX64.fullmatch(m.sha256), m
            assert m.size > 0 and m.filename.endswith(".gguf"), m
            assert re.fullmatch(r"[\w.-]+/[\w.-]+", m.repo_id), m
            assert m.label and m.backend and m.licence, m

    def test_the_file_names_pick_the_aligner_family_crispasr_reads_from_them(self):
        # crispasr/_binding.py align_words: "forced-aligner" -> Qwen3-ForcedAligner, "wav2vec2" -> CTC with the
        # model's vocabulary, anything else -> canary-ctc.
        names = {n: m.filename for n, m in manifest.ALIGNER_MODELS.items()}
        assert "wav2vec2" in names["wav2vec2-ja"]
        assert "forced-aligner" in names["qwen3-fa-gguf"]
        assert not any(w in names["parakeet-ctc-ja"] for w in ("wav2vec2", "forced-aligner", "qwen3-fa"))

    def test_the_default_is_wav2vec2_ja(self):
        assert manifest.DEFAULT_ALIGNER == "wav2vec2-ja"
        assert manifest.aligner_model_for(None) is manifest.ALIGNER_MODELS["wav2vec2-ja"]
        assert manifest.aligner_model_for("") is manifest.ALIGNER_MODELS["wav2vec2-ja"]

    def test_an_unknown_aligner_is_refused_with_the_choices(self):
        with pytest.raises(ValueError) as caught:
            manifest.aligner_model_for("canary-ctc-aligner")
        assert str(caught.value) == ("'canary-ctc-aligner' is not a CrispASR aligner; choose one of: "
                                     "wav2vec2-ja, parakeet-ctc-ja, qwen3-fa-gguf")

    def test_only_the_parakeet_aligner_carries_a_warning(self):
        assert set(manifest.ALIGNER_WARNINGS) == {"parakeet-ctc-ja"}
        assert set(manifest.ALIGNER_WARNINGS) <= set(manifest.ALIGNER_MODELS)

    def test_the_aligners_are_not_pipeline_models(self):
        assert not set(manifest.ALIGNER_MODELS) & set(MODELS)
