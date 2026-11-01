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
    menu_label,
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
            # GGUF files; the Whisper models are whisper.cpp files (1.10.1)
            assert m.filename.endswith(".bin" if m.backend == "whisper" else ".gguf"), m
            assert re.fullmatch(r"[\w.-]+/[\w.-]+", m.repo_id), m
            assert m.label and m.backend and m.licence, m
            # W400: every offered model has its measured GPU memory, and its label names the quantization
            assert m.vram_mib > 0, m
            assert re.fullmatch(r".+ \((f16|q8_0|q5_0)\)", m.label), m

    def test_every_pipeline_model_exists_and_runs_on_its_back_end(self):
        # the model family CrispASR 0.8.44 reported opening each file (2026-10-31); ReazonSpeech opens as parakeet
        backends = {"parakeet-crispasr": "parakeet", "cohere-crispasr": "cohere", "reazonspeech-crispasr": "parakeet",
                    "kotoba-crispasr": "whisper", "whisper-turbo-crispasr": "whisper",
                    "whisper-large-crispasr": "whisper", "moss-crispasr": "moss-diarize"}
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
        # Cohere v0.1 (W129 1); the 1.10.0 defaults stay (W401 1); each new pipeline's largest file (W400)
        assert PIPELINE_DEFAULT_MODEL == {"parakeet-crispasr": "parakeet-tdt-0.6b-ja",
                                          "cohere-crispasr": "cohere-asr-ja-v0.1",
                                          "reazonspeech-crispasr": "reazonspeech-nemo-v2-f16",
                                          "kotoba-crispasr": "kotoba-whisper-v2.0-f16",
                                          "whisper-turbo-crispasr": "whisper-large-v3-turbo-f16",
                                          "whisper-large-crispasr": "whisper-large-v2-f16",
                                          "moss-crispasr": "moss-transcribe-diarize-0.9b-f16"}

    def test_each_new_default_is_a_full_size_file(self):
        # W400: the largest files; Whisper-Large offers two full-size files, large-v2 (the default) and large-v3
        for pipeline in CRISPASR_PIPELINES[2:]:
            default = MODELS[PIPELINE_DEFAULT_MODEL[pipeline]]
            assert default.label.endswith("(f16)"), pipeline
            biggest = max(MODELS[n].size for n in PIPELINE_MODELS[pipeline])
            assert default.size > 0.99 * biggest, pipeline

    def test_every_offered_model_fits_14_gb(self):
        # W400: 8-16 GB cards with 2 GB taken; the largest files that fit 6-14 GB
        for names in PIPELINE_MODELS.values():
            for name in names:
                assert MODELS[name].vram_mib <= 14 * 1024, name

    def test_the_offered_models_without_the_ctc_model(self):
        # The Parakeet CTC model is not offered (W129 2); the full-size files join (W401 1).
        assert PIPELINE_MODELS == {
            "parakeet-crispasr": ("parakeet-tdt-0.6b-ja", "parakeet-tdt-0.6b-ja-f16"),
            "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja-v0.1-f16", "cohere-asr-ja", "cohere-asr-ja-f16"),
            "reazonspeech-crispasr": ("reazonspeech-nemo-v2-f16", "reazonspeech-nemo-v2-q8_0"),
            "kotoba-crispasr": ("kotoba-whisper-v2.0-f16", "kotoba-whisper-v2.0-q5_0"),
            "whisper-turbo-crispasr": ("whisper-large-v3-turbo-f16", "whisper-large-v3-turbo-q8_0"),
            "whisper-large-crispasr": ("whisper-large-v2-f16", "whisper-large-v2-q8_0",
                                       "whisper-large-v3-f16", "whisper-large-v3-q5_0"),
            "moss-crispasr": ("moss-transcribe-diarize-0.9b-f16", "moss-transcribe-diarize-0.9b-q8_0")}
        assert set(MODELS) == {name for names in PIPELINE_MODELS.values() for name in names}

    def test_the_menu_label_names_quantization_and_gpu_memory(self):
        # W400; rounded up to a tenth of a GB
        model = MODELS["whisper-large-v2-f16"]
        gb = model.vram_mib / 1024
        assert menu_label(model).startswith("Whisper large-v2 (f16), ~") and menu_label(model).endswith(" GB VRAM")
        shown = float(menu_label(model).split("~")[1].split(" ")[0])
        assert gb <= shown < gb + 0.1 + 1e-9

    def test_the_aligned_pipelines(self):
        # Cohere-CrispASR times its text with the CrispASR aligner; the others use their models' own times (owner
        # W411, W412; MOSS-CrispASR its own segment times since 1.10.1, W426)
        assert manifest.ALIGNED_PIPELINES == {"cohere-crispasr"}

    def test_the_word_timestamps_rule(self):
        # 1.10.1 (Part B; owner W411-W414): the pipeline's default, or the user's choice
        assert manifest.WORD_TIMESTAMPS == ("model", "aligner")
        for pipeline in manifest.CRISPASR_PIPELINES:
            default = "aligner" if pipeline in manifest.ALIGNED_PIPELINES else "model"
            assert manifest.word_timestamps_for(pipeline) == default
            assert manifest.word_timestamps_for(pipeline, "") == default
            assert manifest.word_timestamps_for(pipeline, "model") == "model"
            assert manifest.word_timestamps_for(pipeline, " Aligner ") == "aligner"
        with pytest.raises(ValueError, match="choose one of: model, aligner"):
            manifest.word_timestamps_for("kotoba-crispasr", "dtw")

    def test_nothing_of_the_program_route_is_left(self):
        for name in ("PROGRAM_BUILDS", "CRISPASR_VERSION", "CRISPASR_GIT", "CUDA13_RUNTIME_FILES",
                     "MEASUREMENT_MODELS", "builds_for_this_machine"):
            assert not hasattr(manifest, name), name

    def test_the_japanese_names(self):
        assert JAPANESE_NAMES == {"ja", "jp", "jpn", "japanese"}


class TestModelFor:
    @pytest.mark.parametrize("pipeline, default", [("parakeet-crispasr", "parakeet-tdt-0.6b-ja"),
                                                   ("cohere-crispasr", "cohere-asr-ja-v0.1"),
                                                   ("whisper-large-crispasr", "whisper-large-v2-f16")])
    def test_a_pipeline_without_a_named_model_gets_its_default(self, pipeline, default):
        assert model_for(pipeline, None) is MODELS[default]
        assert model_for(pipeline, "") is MODELS[default]

    @pytest.mark.parametrize("pipeline, name", [
        ("parakeet-crispasr", "parakeet-tdt-0.6b-ja"),
        ("cohere-crispasr", "cohere-asr-ja-v0.1"),
        ("cohere-crispasr", "cohere-asr-ja"),
        ("cohere-crispasr", "cohere-asr-ja-f16"),
        ("whisper-large-crispasr", "whisper-large-v3-q5_0"),
    ])
    def test_a_named_model(self, pipeline, name):
        assert model_for(pipeline, name) is MODELS[name]

    def test_an_unknown_name_is_refused_with_the_choices(self):
        with pytest.raises(ValueError) as caught:
            model_for("parakeet-crispasr", "parakeet-ctc-1.1b-ja")
        assert str(caught.value) == ("'parakeet-ctc-1.1b-ja' is not a model of parakeet-crispasr; "
                                     "choose one of: parakeet-tdt-0.6b-ja, parakeet-tdt-0.6b-ja-f16")

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
