"""
Whether CrispASR can run here, and on which device (whisperjav/crispasr/install.py prepare_runtime and
choose_device; route B design 4.2, 9.1; owner W194 1, A + Metal).

The platform, the installed package, the processor, CrispASR's libraries and WhisperJAV's own device
check are all stand-ins: CrispASR is never loaded and torch is never imported.

Run with: python -m pytest tests/test_crispasr_runtime.py -q
"""
import types

import pytest

from whisperjav.crispasr import install
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import MODELS

PARAKEET = MODELS["parakeet-tdt-0.6b-ja"]
COHERE = MODELS["cohere-asr-ja-v0.1"]


@pytest.fixture
def machine(monkeypatch):
    """A computer CrispASR runs on, with the CUDA build installed and an NVIDIA GPU, every part replaceable."""
    state = types.SimpleNamespace(supported=True, installed=True, version="0.8.44+cuda", avx2=True,
                                  backends=["parakeet", "cohere", "whisper"], best="cuda", unusable=None,
                                  loads=[], torch="2.11.0+cu130")
    monkeypatch.setattr(install, "platform_supported", lambda *a: state.supported)
    monkeypatch.setattr(install, "crispasr_installed", lambda: state.installed)
    monkeypatch.setattr(install, "has_avx2", lambda: state.avx2)

    def backends():
        state.loads.append("libraries")
        if isinstance(state.backends, Exception):
            raise CrispASRSetupError(install.NOT_LOADED_TEXT.format(reason=state.backends))
        return list(state.backends)

    monkeypatch.setattr(install, "available_backends", backends)
    monkeypatch.setattr(install, "_best_device", lambda: (state.best, state.unusable))

    from importlib import metadata

    def version(name):
        if name == "torch":
            return state.torch
        assert name == "crispasr"
        if not state.installed:
            raise metadata.PackageNotFoundError(name)
        return state.version

    monkeypatch.setattr(metadata, "version", version)
    return state


class TestInstalledBuild:
    @pytest.mark.parametrize("version, build", [("0.8.44+cuda", "cuda"), ("0.8.44+CUDA128", "cuda"),
                                                ("0.8.44", "cpu"), ("0.8.43", "cpu"), ("0.8.44+vulkan", "vulkan")])
    def test_the_build_is_read_from_the_version_label(self, machine, version, build):
        machine.version = version
        assert install.installed_build() == (version, build)

    def test_a_missing_package_is_a_setup_error(self, machine):
        machine.installed = False
        with pytest.raises(CrispASRSetupError, match="not installed"):
            install.installed_build()


class TestPrepareRuntime:
    def test_everything_in_place_runs_on_the_nvidia_gpu(self, machine):
        choice = install.prepare_runtime(None, PARAKEET)
        assert (choice.use_gpu, choice.device, choice.build, choice.version) == (True, "cuda", "cuda", "0.8.44+cuda")
        assert choice.reason is None and not choice.note_in_summary

    def test_no_build_for_this_computer_stops_before_the_package_is_looked_for(self, machine, monkeypatch):
        machine.supported = False
        monkeypatch.setattr(install, "crispasr_installed", lambda: pytest.fail("the package was looked for"))
        with pytest.raises(CrispASRSetupError, match="no build for this computer"):
            install.prepare_runtime(None, PARAKEET)

    def test_a_missing_package_says_to_run_the_installer_again(self, machine):
        machine.installed = False
        with pytest.raises(CrispASRSetupError) as e:
            install.prepare_runtime(None, PARAKEET)
        assert str(e.value) == install.NOT_INSTALLED_TEXT

    @pytest.mark.parametrize("torch", ["2.11.0+cpu", "2.11.0"])
    def test_a_cpu_only_install_without_crispasr_also_says_run_the_installer_again(self, machine, torch):
        # 1.10.0 (design 22.3.2): every install of a supported computer has CrispASR, the CPU-only
        # one included, so a missing CrispASR is a damaged install there too (T10i is gone).
        machine.installed, machine.torch = False, torch
        with pytest.raises(CrispASRSetupError) as e:
            install.prepare_runtime(None, PARAKEET)
        assert str(e.value) == install.NOT_INSTALLED_TEXT
        assert not hasattr(install, "CPU_VARIANT_TEXT") and not hasattr(install, "is_cpu_variant")

    def test_a_processor_without_avx2_stops_before_the_libraries_load(self, machine):
        machine.avx2 = False
        with pytest.raises(CrispASRSetupError, match="AVX2"):
            install.prepare_runtime(None, PARAKEET)
        assert machine.loads == []

    def test_avx2_that_cannot_be_told_is_no_stop(self, machine):
        machine.avx2 = None
        assert install.prepare_runtime(None, PARAKEET).use_gpu

    def test_a_build_without_the_models_back_end_stops_with_its_name(self, machine):
        machine.backends = ["parakeet"]
        with pytest.raises(CrispASRSetupError) as e:
            install.prepare_runtime(None, COHERE)
        assert "'cohere'" in str(e.value) and COHERE.label in str(e.value) and "0.8.44+cuda" in str(e.value)

    def test_libraries_that_will_not_load_are_a_setup_error(self, machine):
        machine.backends = OSError("[WinError 126] The specified module could not be found")
        with pytest.raises(CrispASRSetupError, match="could not be loaded"):
            install.prepare_runtime(None, PARAKEET)


class TestTheDeviceRule:
    """Route B design 9.1: the --device value, the installed build and WhisperJAV's own device check."""

    @pytest.mark.parametrize("device", [None, "auto", "AUTO"])
    def test_cuda_build_auto_with_a_usable_nvidia_gpu(self, machine, device):
        choice = install.choose_device(device, "cuda", "0.8.44+cuda")
        assert (choice.use_gpu, choice.device) == (True, "cuda")

    def test_cuda_build_auto_without_an_nvidia_gpu_runs_on_the_processor_with_a_console_note(self, machine):
        machine.best = "cpu"
        choice = install.choose_device(None, "cuda", "0.8.44+cuda")
        assert (choice.use_gpu, choice.device, choice.reason) == (False, "cpu", install.REASON_NO_NVIDIA)
        assert not choice.note_in_summary

    def test_cuda_build_auto_with_a_card_pytorch_cannot_use_also_says_so_in_the_summary(self, machine):
        machine.best, machine.unusable = "cpu", "GTX 1060 has compute capability 6.1, ..."
        choice = install.choose_device(None, "cuda", "0.8.44+cuda")
        assert (choice.use_gpu, choice.reason, choice.note_in_summary) == (False, machine.unusable, True)

    def test_cuda_build_device_cuda_with_a_gpu(self, machine):
        assert install.choose_device("cuda", "cuda", "v").use_gpu

    def test_cuda_build_device_cuda_without_a_gpu_stops(self, machine):
        machine.best = "cpu"
        with pytest.raises(CrispASRSetupError) as e:
            install.choose_device("cuda", "cuda", "v")
        assert str(e.value) == install.DEVICE_CUDA_TEXT.format(reason=install.REASON_NO_NVIDIA)

    @pytest.mark.parametrize("build", ["cuda", "cpu", "vulkan"])
    def test_device_cpu_always_forces_the_processor(self, machine, build):
        choice = install.choose_device("cpu", build, "v")
        assert (choice.use_gpu, choice.reason, choice.note_in_summary) == (False, install.REASON_DEVICE_CPU, False)

    def test_cpu_build_on_apple_silicon_with_metal(self, machine):
        machine.best = "mps"
        choice = install.choose_device(None, "cpu", "0.8.44", sys_platform="darwin")
        assert (choice.use_gpu, choice.device) == (True, "metal")
        assert "not yet tested on a Mac" in choice.device_line("Parakeet-CrispASR")

    def test_cpu_build_on_a_mac_without_a_gpu(self, machine):
        machine.best = "cpu"
        choice = install.choose_device(None, "cpu", "0.8.44", sys_platform="darwin")
        assert (choice.use_gpu, choice.reason) == (False, install.REASON_NO_GPU)

    def test_cpu_build_elsewhere_runs_on_the_processor(self, machine):
        choice = install.choose_device(None, "cpu", "0.8.44", sys_platform="win32")
        assert (choice.use_gpu, choice.reason, choice.note_in_summary) == (False, install.REASON_CPU_BUILD, False)

    @pytest.mark.parametrize("build, platform", [("cpu", "win32"), ("cpu", "darwin"), ("vulkan", "win32")])
    def test_device_cuda_without_the_cuda_build_stops(self, machine, build, platform):
        with pytest.raises(CrispASRSetupError, match="--device cuda"):
            install.choose_device("cuda", build, "v", sys_platform=platform)

    def test_the_vulkan_build_asks_for_the_gpu(self, machine):
        # The GPU-through-Vulkan install (design 22.3.2, owner W234 D-A): auto asks the build for its GPU.
        choice = install.choose_device(None, "vulkan", "0.8.44+vulkan", sys_platform="win32")
        assert (choice.use_gpu, choice.device, choice.reason) == (True, "vulkan", None)
        assert choice.device_line("Parakeet-CrispASR") == (
            "Parakeet-CrispASR asks CrispASR's Vulkan build to use this PC's GPU (CrispASR 0.8.44+vulkan). "
            "GPUs through Vulkan are experimental in 1.10.0 and have not been tested.")

    def test_the_vulkan_build_with_device_cuda_says_what_to_use(self, machine):
        with pytest.raises(CrispASRSetupError) as e:
            install.choose_device("cuda", "vulkan", "0.8.44+vulkan", sys_platform="win32")
        assert str(e.value) == install.DEVICE_CUDA_VULKAN_TEXT

    def test_the_vulkan_build_with_device_cpu_runs_on_the_cpu(self, machine):
        choice = install.choose_device("cpu", "vulkan", "0.8.44+vulkan", sys_platform="win32")
        assert (choice.use_gpu, choice.device) == (False, "cpu")


class TestTheDeviceLine:
    def test_the_nvidia_gpu_is_worded_as_a_request(self):
        choice = install.RuntimeChoice(True, "cuda", None, False, "0.8.44+cuda", "cuda")
        assert choice.device_line("Parakeet-CrispASR") == \
            "Parakeet-CrispASR asks CrispASR to use the NVIDIA GPU (CUDA) (CrispASR 0.8.44+cuda)."

    def test_the_processor_with_its_reason(self):
        choice = install.RuntimeChoice(False, "cpu", "no NVIDIA GPU was found", False, "0.8.44+cuda", "cuda")
        assert choice.device_line("Cohere-CrispASR") == \
            ("Cohere-CrispASR runs on the CPU (not accelerated), several times slower than on a GPU: "
             "no NVIDIA GPU was found.")
