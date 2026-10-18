"""
What WhisperJAV downloads for its CrispASR pipelines, pinned (whisperjav/crispasr/manifest.py;
master plan 1.1-1.2, 3.3).

The manifest is the program's pin, as uv.lock is the Python packages'. These tests keep its entries
well formed and its choices consistent, so an edit at release time cannot leave a build or a model
that no code path can use, or a pipeline offering a model that does not exist.

Run with: python -m pytest tests/test_crispasr_manifest.py -q
"""
import platform
import re
import sys

import pytest

from whisperjav.crispasr.manifest import (
    CRISPASR_GIT,
    CRISPASR_PIPELINES,
    CRISPASR_VERSION,
    CUDA13_RUNTIME_FILES,
    MEASUREMENT_MODELS,
    MODELS,
    PIPELINE_DEFAULT_MODEL,
    PIPELINE_MODELS,
    PROGRAM_BUILDS,
    RELEASE_BASE_URL,
    build_by_key,
    builds_for_this_machine,
    model_for,
)

CTC = "parakeet-ctc-1.1b-ja"  # measured in step 2b, offered only if step 2b supports it (plan 3.3, Q13)

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class TestProgramBuilds:
    def test_windows_x86_64_gets_the_cuda13_build_first_then_the_cpu_build(self):
        builds = builds_for_this_machine("win32", "AMD64")
        assert [b.key for b in builds] == ["cuda13", "cpu"]
        gpu, cpu = builds
        assert (gpu.gpu, cpu.gpu) == ("cuda13", None)
        assert gpu.needs_avx2 and cpu.needs_avx2
        assert gpu.exe_name == cpu.exe_name == "crispasr.exe"
        assert gpu.extra_args == cpu.extra_args == ()
        assert gpu.asset == "crispasr-windows-x86_64-cuda13-non-cuda.zip"
        assert cpu.asset == "crispasr-windows-x86_64-cpu.zip"

    def test_apple_silicon_gets_the_cpu_build_run_without_the_gpu(self):
        (build,) = builds_for_this_machine("darwin", "arm64")
        assert (build.key, build.gpu) == ("cpu", None)
        assert build.extra_args == ("--no-gpu",)
        assert not build.needs_avx2
        assert build.exe_name == "crispasr"
        assert build.asset == "crispasr-macos-arm64.tar.gz"

    @pytest.mark.parametrize("sys_platform, machine, asset", [
        ("darwin", "x86_64", "crispasr-macos-x86_64.tar.gz"),
        ("linux", "x86_64", "crispasr-linux-x86_64.tar.gz"),
    ])
    def test_the_other_x86_64_machines_get_their_cpu_build(self, sys_platform, machine, asset):
        (build,) = builds_for_this_machine(sys_platform, machine)
        assert (build.key, build.gpu, build.asset) == ("cpu", None, asset)
        assert build.needs_avx2
        assert build.exe_name == "crispasr"

    @pytest.mark.parametrize("sys_platform, machine", [
        ("win32", "ARM64"), ("linux", "aarch64"), ("freebsd14", "amd64"), ("cygwin", "x86_64"), ("aix", "powerpc"),
    ])
    def test_a_machine_without_a_build_gets_none(self, sys_platform, machine):
        assert builds_for_this_machine(sys_platform, machine) == []

    def test_this_machine_is_the_default(self):
        assert builds_for_this_machine() == builds_for_this_machine(sys.platform, platform.machine())

    def test_a_build_by_its_key(self):
        assert build_by_key("cpu", "win32", "AMD64").asset == "crispasr-windows-x86_64-cpu.zip"
        assert build_by_key("cuda13", "win32", "AMD64").gpu == "cuda13"
        assert build_by_key("cuda13", "darwin", "arm64") is None

    def test_every_build_is_pinned_by_size_and_sha256_at_the_release_address(self):
        assert RELEASE_BASE_URL == f"https://github.com/CrispStrobe/CrispASR/releases/download/v{CRISPASR_VERSION}/"
        for b in PROGRAM_BUILDS:
            assert HEX64.fullmatch(b.sha256), b
            assert b.size > 0, b
            assert b.url == RELEASE_BASE_URL + b.asset
            assert b.asset.endswith((".zip", ".tar.gz")), b
            assert b.machines, b

    def test_no_two_builds_for_one_machine_share_a_folder(self):
        seen = set()
        for b in PROGRAM_BUILDS:
            for machine in b.machines:
                where = (b.platform, machine, b.key)
                assert where not in seen, b
                seen.add(where)

    def test_the_cuda13_runtime_files(self):
        assert CUDA13_RUNTIME_FILES == ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")

    def test_the_version_and_git_pin(self):
        assert CRISPASR_VERSION == "0.8.43"
        assert re.fullmatch(r"[0-9a-f]{7,40}", CRISPASR_GIT)


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

    def test_every_pipeline_model_exists_and_runs_on_a_fitting_back_end(self):
        backends = {"parakeet-crispasr": {"parakeet", "fastconformer-ctc"}, "cohere-crispasr": {"cohere"}}
        assert set(PIPELINE_MODELS) == set(backends)
        for pipeline, names in PIPELINE_MODELS.items():
            assert names, pipeline
            for name in names:
                assert name in MODELS, (pipeline, name)
                assert MODELS[name].backend in backends[pipeline], (pipeline, name)

    def test_each_default_is_one_of_its_pipelines_models(self):
        assert set(PIPELINE_DEFAULT_MODEL) == set(PIPELINE_MODELS)
        for pipeline, default in PIPELINE_DEFAULT_MODEL.items():
            assert default is None or default in PIPELINE_MODELS[pipeline], pipeline
        assert CRISPASR_PIPELINES == tuple(PIPELINE_MODELS)

    def test_the_defaults(self):
        # Cohere v0.1 on the step 2b scores (W129 1)
        assert PIPELINE_DEFAULT_MODEL == {"parakeet-crispasr": "parakeet-tdt-0.6b-ja",
                                          "cohere-crispasr": "cohere-asr-ja-v0.1"}

    def test_the_ctc_model_switches_off_the_punctuation_download(self):
        # Without it the program fetches a 58 MB punctuation model by itself (run P1).
        assert MODELS[CTC].extra_args == ("--punc-model", "none")

    def test_the_offered_models(self):
        assert PIPELINE_MODELS == {"parakeet-crispasr": ("parakeet-tdt-0.6b-ja",),
                                   "cohere-crispasr": ("cohere-asr-ja-v0.1", "cohere-asr-ja")}


class TestTheMeasuredOnlyModels:
    """Models step 2b measures but 1.10.0 does not offer: only the pipeline constructor reaches them."""

    def test_the_parakeet_ctc_is_measured_but_not_offered(self):
        assert MEASUREMENT_MODELS == {"parakeet-crispasr": (CTC,), "cohere-crispasr": ()}
        assert all(CTC not in names for names in PIPELINE_MODELS.values())
        assert CTC not in PIPELINE_DEFAULT_MODEL.values()

    def test_every_measured_model_is_pinned_runs_on_its_pipelines_back_end_and_is_not_also_offered(self):
        backends = {"parakeet-crispasr": {"parakeet", "fastconformer-ctc"}, "cohere-crispasr": {"cohere"}}
        assert set(MEASUREMENT_MODELS) == set(PIPELINE_MODELS)
        for pipeline, names in MEASUREMENT_MODELS.items():
            for name in names:
                assert name in MODELS, (pipeline, name)
                assert MODELS[name].backend in backends[pipeline], (pipeline, name)
                assert name not in PIPELINE_MODELS[pipeline], (pipeline, name)


class TestModelFor:
    def test_parakeet_has_a_default(self):
        assert model_for("parakeet-crispasr", None) is MODELS["parakeet-tdt-0.6b-ja"]
        assert model_for("parakeet-crispasr", "") is MODELS["parakeet-tdt-0.6b-ja"]

    def test_cohere_has_a_default(self):
        assert model_for("cohere-crispasr", None) is MODELS["cohere-asr-ja-v0.1"]
        assert model_for("cohere-crispasr", "") is MODELS["cohere-asr-ja-v0.1"]

    @pytest.mark.parametrize("pipeline, name", [
        ("parakeet-crispasr", "parakeet-tdt-0.6b-ja"),
        ("cohere-crispasr", "cohere-asr-ja-v0.1"),
        ("cohere-crispasr", "cohere-asr-ja"),
    ])
    def test_a_named_model(self, pipeline, name):
        assert model_for(pipeline, name) is MODELS[name]
        assert model_for(pipeline, name, include_measurement=True) is MODELS[name]

    def test_the_measured_only_model_is_refused_unless_measurement_models_are_asked_for(self):
        with pytest.raises(ValueError) as caught:
            model_for("parakeet-crispasr", CTC)
        text = str(caught.value)
        assert text == f"'{CTC}' is not a model of parakeet-crispasr; choose one of: parakeet-tdt-0.6b-ja"
        assert model_for("parakeet-crispasr", CTC, include_measurement=True) is MODELS[CTC]

    def test_asking_for_measurement_models_changes_no_default_and_opens_no_other_pipeline(self):
        assert model_for("parakeet-crispasr", None, include_measurement=True) is MODELS["parakeet-tdt-0.6b-ja"]
        assert model_for("cohere-crispasr", None, include_measurement=True) is MODELS["cohere-asr-ja-v0.1"]
        with pytest.raises(ValueError, match="is not a model of cohere-crispasr"):
            model_for("cohere-crispasr", CTC, include_measurement=True)
        with pytest.raises(ValueError) as caught:
            model_for("cohere-crispasr", "nope", include_measurement=True)
        assert CTC not in str(caught.value)  # the choices named are the offered ones only

    def test_an_unknown_name_is_refused_with_the_choices(self):
        with pytest.raises(ValueError) as caught:
            model_for("parakeet-crispasr", "parakeet-tdt-1.1b-ja")
        text = str(caught.value)
        assert "'parakeet-tdt-1.1b-ja'" in text
        assert all(name in text for name in PIPELINE_MODELS["parakeet-crispasr"])

    def test_a_model_of_another_pipeline_is_refused(self):
        with pytest.raises(ValueError, match="is not a model of parakeet-crispasr"):
            model_for("parakeet-crispasr", "cohere-asr-ja")

    def test_a_pipeline_without_a_default_says_what_to_choose(self, monkeypatch):
        # No pipeline is without one today; the refusal stays for a pipeline added before its default is picked.
        monkeypatch.setitem(PIPELINE_DEFAULT_MODEL, "cohere-crispasr", None)
        with pytest.raises(ValueError) as caught:
            model_for("cohere-crispasr", None)
        text = str(caught.value)
        assert "no default model yet" in text
        assert all(name in text for name in PIPELINE_MODELS["cohere-crispasr"])
