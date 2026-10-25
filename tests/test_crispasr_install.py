"""
Making CrispASR's model files ready, and the processor check (whisperjav/crispasr/install.py; route B).

Nothing here reaches the network or a real cache. WhisperJAV's cache folder, the home folder and the
Hugging Face cache are all moved into the test's temporary folder; the Hugging Face download itself is
replaced; the processor and PyTorch are stand-ins. CrispASR is never loaded and torch is never imported.
Whether CrispASR can run and on which device is tests/test_crispasr_runtime.py.

Run with: python -m pytest tests/test_crispasr_install.py -q
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants

from whisperjav.crispasr import install, manifest
from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import ModelFile

REPO = Path(__file__).resolve().parents[1]
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the Windows processor-feature call")

# The real Hugging Face download step, kept before the autouse fixture below replaces it in every test;
# TestTheHuggingFaceDownload calls it with huggingface_hub's own download function replaced.
REAL_HF_DOWNLOAD = install._hf_download


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def places(tmp_path, monkeypatch):
    """Every folder the code could write to, moved into this test's temporary folder."""
    home = tmp_path / "home"
    home.mkdir()
    cache_root = home / ".cache" / "whisperjav"
    hub = tmp_path / "hf" / "hub"
    monkeypatch.setattr(install, "whisperjav_cache_root", lambda: cache_root)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    # huggingface_hub read the variables when it was imported; its cache lookups use this value.
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", install.MODEL_DIR_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    def no_real_download(model, force=False):
        raise AssertionError("a test reached the real Hugging Face download")

    monkeypatch.setattr(install, "_hf_download", no_real_download)  # the hf_download fixture replaces it
    return types.SimpleNamespace(tmp=tmp_path, home=home, cache_root=cache_root, hub=hub)


def _no_torch(monkeypatch):
    """`import torch` fails, so nothing can load the real one."""
    monkeypatch.setitem(sys.modules, "torch", None)


class _TorchImportRecorder:
    """An import finder that records, and refuses, every import of torch."""

    def __init__(self):
        self.asked = []

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            self.asked.append(name)
            raise ImportError(f"{name} must not be imported here")
        return None


@pytest.fixture
def torch_imports(monkeypatch):
    """Every `import torch` from here on is recorded (and fails), even one inside a try block."""
    recorder = _TorchImportRecorder()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    return recorder

def _fake_torch(monkeypatch, capability="AVX2", capability_error=None, cuda=True, cuda_version="13.0"):
    torch = types.ModuleType("torch")

    def get_cpu_capability():
        if capability_error is not None:
            raise capability_error
        return capability

    def is_available():
        if isinstance(cuda, Exception):
            raise cuda
        return cuda

    torch.backends = types.SimpleNamespace(cpu=types.SimpleNamespace(get_cpu_capability=get_cpu_capability))
    torch.cuda = types.SimpleNamespace(is_available=is_available)
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch

# ---------------------------------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------------------------------

def _on(monkeypatch, sys_platform, machine="x86_64"):
    monkeypatch.setattr(install, "sys", types.SimpleNamespace(platform=sys_platform))
    monkeypatch.setattr(install.platform, "machine", lambda: machine)


def _cpuinfo(monkeypatch, tmp_path, text):
    """/proc/cpuinfo, read from a file of the test's (missing when text is None)."""
    path = tmp_path / "cpuinfo"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    real_path = install.Path
    monkeypatch.setattr(install, "Path",
                        lambda p, *rest: real_path(path) if p == "/proc/cpuinfo" else real_path(p, *rest))


def _sysctl(monkeypatch, output):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


CPUINFO_FLAGS = ("processor\t: 0\nvendor_id\t: AuthenticAMD\n"
                 "flags\t\t: fpu vme de pse tsc msr pae sse sse2 ssse3 fma cx16 sse4_1 sse4_2 popcnt aes "
                 "xsave avx f16c rdrand {avx2}bmi1 bmi2\n")


class TestTheProcessor:
    @windows_only
    def test_a_yes_from_windows_never_imports_torch(self, monkeypatch, torch_imports):
        import ctypes
        monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(ctypes.windll.kernel32, "IsProcessorFeaturePresent", lambda feature: 1)
        assert install.has_avx2() is True
        assert torch_imports.asked == []

    @windows_only
    def test_the_recorder_sees_the_import_when_pytorch_is_asked(self, monkeypatch, torch_imports):
        # Without the flag's yes, PyTorch is the second opinion: the recorder sees that import, so the
        # test above would fail if a yes from Windows still imported torch.
        import ctypes
        monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(ctypes.windll.kernel32, "IsProcessorFeaturePresent", lambda feature: 0)
        assert install.has_avx2() is None
        assert torch_imports.asked == ["torch"]

    @pytest.mark.parametrize("flag, expected", [("avx2 ", True), ("", False)])
    def test_an_answer_from_linux_never_imports_torch(self, monkeypatch, tmp_path, torch_imports, flag, expected):
        _on(monkeypatch, "linux")
        _cpuinfo(monkeypatch, tmp_path, CPUINFO_FLAGS.format(avx2=flag))
        assert install.has_avx2() is expected
        assert torch_imports.asked == []

    def test_an_answer_from_an_intel_mac_never_imports_torch(self, monkeypatch, torch_imports):
        _on(monkeypatch, "darwin")
        _sysctl(monkeypatch, "RDWRFSGS SMEP ERMS\n")
        assert install.has_avx2() is False
        assert torch_imports.asked == []

    @pytest.mark.parametrize("machine", ["arm64", "aarch64", "ARM64"])
    def test_a_processor_that_is_not_x86_64_cannot_be_told(self, monkeypatch, machine):
        monkeypatch.setattr(install.platform, "machine", lambda: machine)
        assert install.has_avx2() is None

    @windows_only
    def test_windows_answers_from_the_processor_feature_flag(self, monkeypatch):
        import ctypes
        asked = []
        monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(ctypes.windll.kernel32, "IsProcessorFeaturePresent",
                            lambda feature: asked.append(feature) or 1)
        _no_torch(monkeypatch)
        assert install.has_avx2() is True
        assert asked == [40]  # PF_AVX2_INSTRUCTIONS_AVAILABLE

    @windows_only
    @pytest.mark.parametrize("capability, expected", [("AVX2", True), ("AVX512", True), ("DEFAULT", False)])
    def test_windows_without_the_flag_asks_pytorch(self, monkeypatch, capability, expected):
        import ctypes
        monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(ctypes.windll.kernel32, "IsProcessorFeaturePresent", lambda feature: 0)
        _fake_torch(monkeypatch, capability=capability)
        assert install.has_avx2() is expected

    @windows_only
    def test_windows_without_the_flag_or_pytorch_cannot_be_told(self, monkeypatch):
        import ctypes
        monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(ctypes.windll.kernel32, "IsProcessorFeaturePresent", lambda feature: 0)
        _no_torch(monkeypatch)
        assert install.has_avx2() is None

    @pytest.mark.parametrize("flag, expected", [("avx2 ", True), ("", False)])
    def test_linux_reads_the_cpu_flags(self, monkeypatch, tmp_path, flag, expected):
        _on(monkeypatch, "linux")
        _cpuinfo(monkeypatch, tmp_path, CPUINFO_FLAGS.format(avx2=flag))
        _no_torch(monkeypatch)
        assert install.has_avx2() is expected

    def test_linux_without_cpu_flags_asks_pytorch(self, monkeypatch, tmp_path):
        _on(monkeypatch, "linux")
        _cpuinfo(monkeypatch, tmp_path, None)
        _fake_torch(monkeypatch, capability="AVX512")
        assert install.has_avx2() is True

    def test_linux_with_no_answer_anywhere_cannot_be_told(self, monkeypatch, tmp_path):
        _on(monkeypatch, "linux")
        _cpuinfo(monkeypatch, tmp_path, None)
        _fake_torch(monkeypatch, capability_error=RuntimeError("no dispatch level"))
        assert install.has_avx2() is None

    @pytest.mark.parametrize("output, expected", [
        ("RDWRFSGS TSC_THREAD_OFFSET BMI1 HLE AVX2 SMEP BMI2 ERMS INVPCID\n", True),
        ("RDWRFSGS TSC_THREAD_OFFSET SMEP ERMS\n", False),
    ])
    def test_an_intel_mac_asks_sysctl(self, monkeypatch, output, expected):
        _on(monkeypatch, "darwin")
        calls = _sysctl(monkeypatch, output)
        _no_torch(monkeypatch)
        assert install.has_avx2() is expected
        assert calls == [["sysctl", "-n", "machdep.cpu.leaf7_features"]]

    def test_an_intel_mac_with_no_sysctl_answer_asks_pytorch(self, monkeypatch):
        _on(monkeypatch, "darwin")
        _sysctl(monkeypatch, "")
        _fake_torch(monkeypatch, capability="DEFAULT")
        assert install.has_avx2() is False

# ---------------------------------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------------------------------

MODEL_BYTES = b"GGUF" + bytes(range(256)) * 16
DAMAGED_BYTES = MODEL_BYTES[::-1]  # the same size, other bytes
TEST_MODEL = ModelFile("test-model", "Test model (q8_0)", "parakeet", "wj-test/test-model-GGUF",
                       "test-model-q8_0.gguf", "0123456789abcdef0123456789abcdef01234567",
                       len(MODEL_BYTES), sha256(MODEL_BYTES), "test only")


def cache_file(places, data=MODEL_BYTES, revision=TEST_MODEL.revision) -> Path:
    """The model file where the Hugging Face cache keeps a pinned revision."""
    snapshot = places.hub / "models--wj-test--test-model-GGUF" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    path = snapshot / TEST_MODEL.filename
    path.write_bytes(data)
    return path


def users_folder(places, monkeypatch, data=MODEL_BYTES) -> Path:
    """The model file in a folder the user filled by hand, named in WHISPERJAV_CRISPASR_MODEL_DIR."""
    folder = places.tmp / "my models"
    folder.mkdir(exist_ok=True)
    path = folder / TEST_MODEL.filename
    path.write_bytes(data)
    monkeypatch.setenv(install.MODEL_DIR_ENV, str(folder))
    return path


@pytest.fixture
def hf_download(monkeypatch, places):
    """The Hugging Face download, replaced: it writes ``state.serve`` bytes (one per call) into the cache."""
    state = types.SimpleNamespace(calls=[], serve=[MODEL_BYTES])

    def download(model, force=False):
        target = places.hub / "models--wj-test--test-model-GGUF" / "snapshots" / model.revision / model.filename
        state.calls.append({"model": model.name, "force": force, "file_was_there": target.exists()})
        return cache_file(places, state.serve[min(len(state.calls), len(state.serve)) - 1])

    monkeypatch.setattr(install, "_hf_download", download)
    return state


@pytest.fixture
def hashing(monkeypatch):
    """Every SHA-256 check of a model file, counted."""
    calls = []
    real = install._sha256_of

    def counted(path):
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr(install, "_sha256_of", counted)
    return calls


class TestModelStatus:
    def test_nothing_on_this_pc(self):
        status = install.model_status(TEST_MODEL)
        assert (status.downloaded, status.path, status.source) == (False, None, "")

    def test_a_file_in_the_hugging_face_cache_is_found(self, places):
        path = cache_file(places)
        status = install.model_status(TEST_MODEL)
        assert (status.downloaded, status.path, status.source) == (True, path, "cache")

    def test_the_cache_is_asked_for_the_pinned_revision(self, monkeypatch):
        import huggingface_hub
        asked = []
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda *args, **kwargs: asked.append((args, kwargs)) or None)
        install.model_status(TEST_MODEL)
        assert asked == [((TEST_MODEL.repo_id, TEST_MODEL.filename), {"revision": TEST_MODEL.revision})]

    def test_a_cache_file_of_another_revision_is_not_used(self, places):
        cache_file(places, revision="f" * 40)
        assert not install.model_status(TEST_MODEL).downloaded

    def test_a_cache_file_of_another_size_is_not_used(self, places):
        cache_file(places, MODEL_BYTES[:-1])
        assert not install.model_status(TEST_MODEL).downloaded

    def test_a_failing_cache_lookup_counts_as_not_found(self, monkeypatch):
        import huggingface_hub

        def broken(*args, **kwargs):
            raise RuntimeError("the cache cannot be read")

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", broken)
        assert not install.model_status(TEST_MODEL).downloaded

    def test_the_users_folder_is_used_when_the_size_matches(self, places, monkeypatch):
        path = users_folder(places, monkeypatch)
        status = install.model_status(TEST_MODEL)
        assert (status.downloaded, status.path, status.source) == (True, path, "folder")

    def test_the_users_folder_comes_before_the_cache(self, places, monkeypatch):
        cache_file(places)
        users_folder(places, monkeypatch)
        assert install.model_status(TEST_MODEL).source == "folder"

    def test_a_file_of_another_size_in_the_users_folder_is_passed_over(self, places, monkeypatch):
        users_folder(places, monkeypatch, MODEL_BYTES + b"x")
        cached = cache_file(places)
        status = install.model_status(TEST_MODEL)
        assert (status.source, status.path) == ("cache", cached)


class TestEnsureModel:
    def test_offline_with_the_model_missing_stops_without_a_download(self, hf_download):
        with pytest.raises(CrispASRSetupError) as caught:
            install.ensure_model(TEST_MODEL, offline=True)
        assert TEST_MODEL.filename in str(caught.value)
        assert "--offline allows no download" in str(caught.value)
        assert hf_download.calls == []

    def test_offline_follows_the_offline_switch_when_not_given(self, hf_download, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(CrispASRSetupError, match="--offline"):
            install.ensure_model(TEST_MODEL)
        assert hf_download.calls == []

    def test_a_missing_model_is_downloaded_checked_and_remembered(self, hf_download, hashing, places):
        path = install.ensure_model(TEST_MODEL, offline=False)
        assert path.read_bytes() == MODEL_BYTES
        assert [(c["model"], c["force"]) for c in hf_download.calls] == [("test-model", False)]
        assert len(hashing) == 1
        record = json.loads((places.cache_root / "crispasr" / "verified-models.json").read_text(encoding="utf-8"))
        entry = record[os.path.realpath(path)]
        assert (entry["sha256"], entry["size"]) == (TEST_MODEL.sha256, len(MODEL_BYTES))

    def test_not_enough_disk_space_stops_before_the_download(self, hf_download, monkeypatch):
        monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(total=1, used=1, free=0))
        with pytest.raises(CrispASRSetupError, match="Not enough disk space"):
            install.ensure_model(TEST_MODEL, offline=False)
        assert hf_download.calls == []

    def test_a_cached_file_is_checked_once_and_then_remembered(self, places, hashing, hf_download):
        path = cache_file(places)
        assert install.ensure_model(TEST_MODEL, offline=False) == path
        assert install.ensure_model(TEST_MODEL, offline=False) == path
        assert len(hashing) == 1
        assert hf_download.calls == []

    def test_a_file_changed_since_its_check_is_checked_again(self, places, hashing):
        path = cache_file(places)
        install.ensure_model(TEST_MODEL, offline=False)
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        install.ensure_model(TEST_MODEL, offline=False)
        assert len(hashing) == 2

    def test_a_damaged_cache_file_is_deleted_and_downloaded_once_more(self, places, hf_download, hashing):
        cache_file(places, DAMAGED_BYTES)
        path = install.ensure_model(TEST_MODEL, offline=False)
        assert path.read_bytes() == MODEL_BYTES
        # deleted before the new download, which ignores the cache (force)
        assert [(c["force"], c["file_was_there"]) for c in hf_download.calls] == [(True, False)]
        assert len(hashing) == 2

    def test_a_cache_file_damaged_twice_stops_the_run(self, places, hf_download):
        cache_file(places, DAMAGED_BYTES)
        hf_download.serve = [DAMAGED_BYTES]
        with pytest.raises(CrispASRSetupError, match="damaged twice"):
            install.ensure_model(TEST_MODEL, offline=False)
        assert len(hf_download.calls) == 1

    def test_a_damaged_cache_file_with_offline_is_reported_and_kept(self, places, hf_download):
        damaged = cache_file(places, DAMAGED_BYTES)
        with pytest.raises(CrispASRSetupError, match="--offline"):
            install.ensure_model(TEST_MODEL, offline=True)
        assert damaged.read_bytes() == DAMAGED_BYTES
        assert hf_download.calls == []

    def test_a_damaged_file_in_the_users_folder_is_reported_and_never_deleted(self, places, monkeypatch,
                                                                             hf_download):
        path = users_folder(places, monkeypatch, DAMAGED_BYTES)
        with pytest.raises(CrispASRSetupError) as caught:
            install.ensure_model(TEST_MODEL, offline=False)
        assert str(path) in str(caught.value) and install.MODEL_DIR_ENV in str(caught.value)
        assert path.read_bytes() == DAMAGED_BYTES
        assert hf_download.calls == []

    def test_a_good_file_in_the_users_folder_is_used_even_offline(self, places, monkeypatch, hf_download):
        path = users_folder(places, monkeypatch)
        assert install.ensure_model(TEST_MODEL, offline=True) == path
        assert hf_download.calls == []


class TestTheHuggingFaceDownload:
    """install._hf_download itself, with huggingface_hub's own download function replaced (no network)."""

    @pytest.fixture
    def hub(self, monkeypatch):
        import huggingface_hub

        state = types.SimpleNamespace(calls=[], answer=None, error=None)

        def hf_hub_download(*args, **kwargs):
            state.calls.append((args, kwargs))
            if state.error is not None:
                raise state.error
            return state.answer(*args, **kwargs) if callable(state.answer) else state.answer

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        return state

    @pytest.mark.parametrize("name", sorted(manifest.MODELS))
    def test_it_asks_for_the_pinned_file_at_the_pinned_commit(self, hub, tmp_path, name):
        model = manifest.MODELS[name]
        hub.answer = str(tmp_path / "hub" / model.filename)
        assert REAL_HF_DOWNLOAD(model) == Path(hub.answer)
        assert hub.calls == [((model.repo_id, model.filename), {"revision": model.revision, "force_download": False})]
        assert len(model.revision) == 40  # a commit, never a branch name such as "main"

    def test_a_forced_download_says_so(self, hub, tmp_path):
        hub.answer = str(tmp_path / "x.gguf")
        REAL_HF_DOWNLOAD(TEST_MODEL, force=True)
        assert hub.calls == [((TEST_MODEL.repo_id, TEST_MODEL.filename),
                              {"revision": TEST_MODEL.revision, "force_download": True})]

    @pytest.mark.parametrize("error", [OSError("connection refused"), RuntimeError("401 Client Error")],
                             ids=["network", "hub"])
    def test_a_failed_download_is_a_setup_error_that_keeps_its_cause(self, hub, error):
        hub.error = error
        with pytest.raises(CrispASRSetupError) as caught:
            REAL_HF_DOWNLOAD(TEST_MODEL)
        assert str(caught.value) == f"Downloading the model {TEST_MODEL.label} failed: {error}"
        assert caught.value.__cause__ is error

    def test_ensure_model_downloads_through_it_and_checks_the_file(self, hub, places, monkeypatch, hashing):
        monkeypatch.setattr(install, "_hf_download", REAL_HF_DOWNLOAD)
        hub.answer = lambda repo_id, filename, revision=None, force_download=False: str(
            cache_file(places, revision=revision))
        path = install.ensure_model(TEST_MODEL, offline=False)
        assert path.read_bytes() == MODEL_BYTES
        assert [c[1]["revision"] for c in hub.calls] == [TEST_MODEL.revision]
        assert len(hashing) == 1

def _no_link(src, dst):
    raise OSError(17, "The system cannot move the file to a different disk drive")


def _symlink_or_skip(target: str, link: Path) -> None:
    """A relative symbolic link, as the Hugging Face cache makes them. Skipped where this account cannot make one
    (Windows without Developer Mode or administrator rights): the cache then holds plain files, and the case does
    not arise."""
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symbolic links cannot be made here: {e}")


# ---------------------------------------------------------------------------------------------------
# A plain-ASCII path for the model (route B design 4.2, owner O14)
# ---------------------------------------------------------------------------------------------------

class TestStageModel:
    @pytest.fixture
    def temp(self, tmp_path, monkeypatch):
        """The run's temp folder; %PUBLIC% and %ProgramData% moved into the test's folder."""
        public = tmp_path / "Public"
        data = tmp_path / "ProgramData"
        monkeypatch.setenv("PUBLIC", str(public))
        monkeypatch.setenv("ProgramData", str(data))
        monkeypatch.setattr(install, "_short_path", lambda path: None)  # no 8.3 names, as on the checked drive
        t = tmp_path / "temp"
        t.mkdir()
        return types.SimpleNamespace(temp=t, public=public / "WhisperJAV" / "crispasr",
                                     data=data / "WhisperJAV" / "crispasr")

    @pytest.fixture
    def model(self, tmp_path):
        folder = tmp_path / "Users" / "山田太郎" / "cache"
        folder.mkdir(parents=True)
        path = folder / TEST_MODEL.filename
        path.write_bytes(MODEL_BYTES)
        return path

    def staged_name(self):
        return f"{TEST_MODEL.sha256[:16]}.gguf"

    def test_an_ascii_path_is_used_as_it_is(self, tmp_path, temp):
        path = tmp_path / "models" / "m.gguf"
        path.parent.mkdir()
        path.write_bytes(MODEL_BYTES)
        assert install.stage_model(path, temp.temp, TEST_MODEL.sha256) == path

    @pytest.mark.skipif(sys.platform != "win32", reason="%PUBLIC% and %ProgramData% are Windows folders")
    def test_a_path_outside_ascii_is_hard_linked_into_the_first_public_folder(self, model, temp):
        staged = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        assert staged == temp.public / self.staged_name()
        assert str(staged).isascii()
        assert os.path.samefile(staged, model)

    def test_the_short_path_is_used_when_it_is_ascii(self, model, temp, monkeypatch):
        short = model.parent.parent.parent / "YAMADA~1.GGU"
        short.write_bytes(MODEL_BYTES)
        monkeypatch.setattr(install, "_short_path", lambda path: str(short))
        assert install.stage_model(model, temp.temp, TEST_MODEL.sha256) == short

    def test_without_public_folders_the_temp_folder_is_used(self, model, temp, monkeypatch):
        monkeypatch.delenv("PUBLIC")
        monkeypatch.delenv("ProgramData")
        staged = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        assert staged == temp.temp / install.STAGE_FOLDER / self.staged_name()

    def test_a_link_is_tried_in_every_folder_before_any_copy(self, model, temp, monkeypatch):
        tried = []

        real_link = os.link

        def link(src, dst):
            tried.append(Path(dst).parent)
            if Path(dst).parent == temp.temp / install.STAGE_FOLDER:
                return real_link(src, dst)
            raise OSError(17, "The system cannot move the file to a different disk drive")

        monkeypatch.setattr(install.os, "link", link)
        staged = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        expected = [p for p in (temp.public, temp.data) if sys.platform == "win32"] + [temp.temp / install.STAGE_FOLDER]
        assert tried == expected
        assert staged.parent == temp.temp / install.STAGE_FOLDER
        assert os.path.samefile(staged, model)

    def test_a_copy_is_made_only_where_no_link_can_be_made_anywhere(self, model, temp, monkeypatch):
        monkeypatch.setattr(install.os, "link", _no_link)
        staged = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        first = temp.public if sys.platform == "win32" else temp.temp / install.STAGE_FOLDER
        assert staged == first / self.staged_name()
        assert staged.read_bytes() == MODEL_BYTES and not os.path.samefile(staged, model)
        assert not list(first.glob("*.part"))

    def test_a_copy_made_before_is_reused_without_copying_again(self, model, temp, monkeypatch):
        monkeypatch.setattr(install.os, "link", _no_link)
        first = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        copies = []
        monkeypatch.setattr(install.shutil, "copy2", lambda *a, **k: copies.append(a))
        assert install.stage_model(model, temp.temp, TEST_MODEL.sha256) == first
        assert copies == []

    def test_a_file_under_the_staged_name_without_a_check_record_is_replaced(self, model, temp, monkeypatch):
        """A stale link to old data (the cache file was fetched again) is never reused."""
        monkeypatch.delenv("PUBLIC")
        monkeypatch.delenv("ProgramData")
        folder = temp.temp / install.STAGE_FOLDER
        folder.mkdir()
        (folder / self.staged_name()).write_bytes(DAMAGED_BYTES)
        staged = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        assert staged.read_bytes() == MODEL_BYTES

    def test_two_models_of_the_same_size_never_share_a_staged_file(self, model, temp, monkeypatch):
        monkeypatch.delenv("PUBLIC")
        monkeypatch.delenv("ProgramData")
        other = model.parent / "other.gguf"
        other.write_bytes(DAMAGED_BYTES)
        a = install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        b = install.stage_model(other, temp.temp, sha256(DAMAGED_BYTES))
        assert a != b
        assert a.read_bytes() == MODEL_BYTES and b.read_bytes() == DAMAGED_BYTES

    def test_a_model_reached_through_a_relative_link_is_staged_as_the_file_itself(self, model, temp, monkeypatch):
        monkeypatch.delenv("PUBLIC")
        monkeypatch.delenv("ProgramData")
        snapshot = model.parent / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        link = snapshot / "linked.gguf"
        _symlink_or_skip(os.path.join("..", "..", model.name), link)
        staged = install.stage_model(link, temp.temp, TEST_MODEL.sha256)
        assert os.path.samefile(staged, model)

    def test_no_folder_for_a_link_or_a_copy_stops_with_the_way_out(self, model, temp, monkeypatch):
        monkeypatch.setattr(install, "stage_candidates", lambda temp_dir: [])
        with pytest.raises(CrispASRSetupError) as e:
            install.stage_model(model, temp.temp, TEST_MODEL.sha256)
        assert install.MODEL_DIR_ENV in str(e.value) and TEST_MODEL.filename in str(e.value)

    def test_a_copy_without_the_disk_space_for_it_is_not_made(self, model, temp, monkeypatch):
        monkeypatch.setattr(install.os, "link", _no_link)
        monkeypatch.setattr(install.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10))
        with pytest.raises(CrispASRSetupError):
            install.stage_model(model, temp.temp, TEST_MODEL.sha256)

    def test_a_folder_whose_own_path_is_not_ascii_is_not_a_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PUBLIC", str(tmp_path / "公開"))
        monkeypatch.delenv("ProgramData", raising=False)
        folders = [f for f, _keeps in install.stage_candidates(tmp_path / "temp")]
        assert all(str(f.resolve()).isascii() for f in folders)


class TestImportsStayLight:
    def test_the_package_loads_no_torch_no_crispasr_no_asr_module_and_no_download_library(self):
        code = (
            "import sys\n"
            "import whisperjav.crispasr.install, whisperjav.crispasr.engine, whisperjav.crispasr.lines\n"
            "import whisperjav.crispasr.manifest, whisperjav.crispasr.errors, whisperjav.crispasr.pass_settings\n"
            "heavy = sorted(m for m in sys.modules\n"
            "               if m.split('.')[0] in ('torch', 'requests', 'huggingface_hub', 'transformers',\n"
            "                                      'crispasr', 'stable_whisper')\n"
            "               or m.startswith(('whisperjav.modules', 'whisperjav.pipelines', 'whisperjav.main')))\n"
            "print(','.join(heavy))\n"
        )
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO), capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == ""
