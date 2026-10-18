"""
Getting CrispASR's program and model files ready, and saying whether they are
(whisperjav/crispasr/install.py; master plan 1.2-1.7, 4.3, 4.4).

Nothing here reaches the network or a real cache. WhisperJAV's cache folder, the home folder and the
Hugging Face cache are all moved into the test's temporary folder; program downloads come from a
plain HTTP server on 127.0.0.1 that the test runs; the Hugging Face download itself is replaced; the
processor, PyTorch and the program's start check are stand-ins. No CrispASR program is started and
torch is never imported.

Run with: python -m pytest tests/test_crispasr_install.py -q
"""
import collections
import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import types
import zipfile
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants

from whisperjav.crispasr import install, manifest
from whisperjav.crispasr.install import CrispASRSetupError, CudaUnavailable, ProgramStatus, StartCheck
from whisperjav.crispasr.manifest import CRISPASR_GIT, CRISPASR_VERSION, CUDA13_RUNTIME_FILES, ModelFile, ProgramBuild
from whisperjav.crispasr.runner import exit_code_meaning

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


class _OsWithPid:
    """The os module as install.py sees it, with another process id."""

    def __init__(self, pid):
        self._pid = pid

    def getpid(self):
        return self._pid

    def __getattr__(self, name):
        return getattr(os, name)


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
# A download server on 127.0.0.1
# ---------------------------------------------------------------------------------------------------

class LocalServer:
    """Plain HTTP from memory. Each file can have a list of bodies, one per request (the last repeats)."""

    def __init__(self):
        self.bodies = {}
        self.hits = collections.Counter()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                name = self.path.lstrip("/")
                owner.hits[name] += 1
                bodies = owner.bodies.get(name)
                if not bodies:
                    self.send_error(404)
                    return
                body = bodies[min(owner.hits[name], len(bodies)) - 1]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def serve(self, name, *bodies):
        self.bodies[name] = list(bodies)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def server():
    s = LocalServer()
    yield s
    s.close()


def zip_of(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def tar_gz_of(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


FOLDER = "crispasr-windows-x86_64-cpu"
PROGRAM_FILES = {f"{FOLDER}/crispasr.exe": b"MZ stand-in program",
                 f"{FOLDER}/ggml.dll": b"MZ stand-in library",
                 f"{FOLDER}/LICENSE": b"MIT"}
TEST_ASSET = "crispasr-test-cpu.zip"


def program_build(server, payload, asset=TEST_ASSET, platform_="win32", serve=True) -> ProgramBuild:
    """A CPU build pinned to ``payload`` and served from the local server."""
    if serve:
        server.serve(asset, payload)
    return ProgramBuild("cpu", "CPU build", platform_, ("AMD64", "x86_64"), asset, len(payload),
                        sha256(payload), base_url=server.base_url)


# ---------------------------------------------------------------------------------------------------
# The program: download, check, unpack
# ---------------------------------------------------------------------------------------------------

class TestTheProgramDownload:
    def test_the_first_call_downloads_checks_unpacks_and_records(self, server, places):
        payload = zip_of(PROGRAM_FILES)
        build = program_build(server, payload)
        exe = install.ensure_program(build, offline=False)
        folder = places.cache_root / "crispasr" / CRISPASR_VERSION / "cpu"
        assert install.program_dir(build) == folder
        assert exe == folder / "crispasr.exe"
        assert exe.read_bytes() == b"MZ stand-in program"
        assert sorted(p.name for p in folder.iterdir()) == [".verified", "LICENSE", "crispasr.exe", "ggml.dll"]
        record = json.loads((folder / ".verified").read_text(encoding="utf-8"))
        assert {k: record[k] for k in ("version", "asset", "size", "sha256")} == {
            "version": CRISPASR_VERSION, "asset": TEST_ASSET, "size": len(payload), "sha256": sha256(payload)}
        assert server.hits[TEST_ASSET] == 1
        # the archive, its partial download and the unpacking folder are gone
        root = install.program_root()
        assert sorted(p.name for p in root.iterdir()) == [".download", "cpu"]
        assert list((root / ".download").iterdir()) == []
        assert install.program_status(build) == ProgramStatus(build, True, exe)

    def test_a_second_call_does_not_download_again(self, server):
        build = program_build(server, zip_of(PROGRAM_FILES))
        first = install.ensure_program(build, offline=False)
        assert install.ensure_program(build, offline=False) == first
        assert server.hits[TEST_ASSET] == 1

    def test_once_downloaded_offline_is_no_obstacle(self, server):
        build = program_build(server, zip_of(PROGRAM_FILES))
        install.ensure_program(build, offline=False)
        assert install.ensure_program(build, offline=True).is_file()
        assert server.hits[TEST_ASSET] == 1

    def test_a_damaged_download_is_fetched_once_more_then_refused(self, server):
        good = zip_of(PROGRAM_FILES)
        damaged = bytes([good[0] ^ 0xFF]) + good[1:]  # the same size, other bytes
        build = program_build(server, good, serve=False)
        server.serve(TEST_ASSET, damaged)
        with pytest.raises(CrispASRSetupError, match="damaged twice"):
            install.ensure_program(build, offline=False)
        assert server.hits[TEST_ASSET] == 2
        assert not install.program_status(build).downloaded
        assert not install.program_dir(build).exists()
        assert list((install.program_root() / ".download").iterdir()) == []

    def test_a_short_download_counts_as_damaged(self, server):
        good = zip_of(PROGRAM_FILES)
        build = program_build(server, good, serve=False)
        server.serve(TEST_ASSET, good[:-10])
        with pytest.raises(CrispASRSetupError, match="damaged twice"):
            install.ensure_program(build, offline=False)
        assert server.hits[TEST_ASSET] == 2

    def test_a_damaged_download_followed_by_a_good_one_installs(self, server):
        good = zip_of(PROGRAM_FILES)
        build = program_build(server, good, serve=False)
        server.serve(TEST_ASSET, b"\x00" * len(good), good)
        assert install.ensure_program(build, offline=False).read_bytes() == b"MZ stand-in program"
        assert server.hits[TEST_ASSET] == 2

    def test_a_failed_request_stops_without_a_retry(self, server):
        build = program_build(server, zip_of(PROGRAM_FILES), serve=False)  # the server answers 404
        with pytest.raises(CrispASRSetupError, match="failed"):
            install.ensure_program(build, offline=False)
        assert server.hits[TEST_ASSET] == 1
        assert list((install.program_root() / ".download").iterdir()) == []

    def test_offline_with_nothing_downloaded_stops_without_a_request(self, server):
        build = program_build(server, zip_of(PROGRAM_FILES))
        with pytest.raises(CrispASRSetupError, match="--offline allows no download"):
            install.ensure_program(build, offline=True)
        assert server.hits[TEST_ASSET] == 0
        assert not install.program_root().exists()

    def test_offline_follows_the_offline_switch_when_not_given(self, server, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        build = program_build(server, zip_of(PROGRAM_FILES))
        with pytest.raises(CrispASRSetupError, match="--offline"):
            install.ensure_program(build)
        assert server.hits[TEST_ASSET] == 0

    def test_not_enough_disk_space_stops_before_the_download(self, server, monkeypatch, places):
        asked = []

        def full_disk(path):
            asked.append(Path(path))
            return types.SimpleNamespace(total=10**12, used=10**12, free=0)

        monkeypatch.setattr(shutil, "disk_usage", full_disk)
        build = program_build(server, zip_of(PROGRAM_FILES))
        with pytest.raises(CrispASRSetupError, match="Not enough disk space"):
            install.ensure_program(build, offline=False)
        assert server.hits[TEST_ASSET] == 0
        # the question is asked of the nearest folder that exists on the way to the program folder
        assert asked == [places.home]

    def test_free_space_that_cannot_be_read_is_no_stop(self, server, monkeypatch):
        def unreadable(path):
            raise OSError("the device is not ready")

        monkeypatch.setattr(shutil, "disk_usage", unreadable)
        build = program_build(server, zip_of(PROGRAM_FILES))
        assert install.ensure_program(build, offline=False).is_file()

    def test_a_member_that_would_leave_the_folder_is_refused(self, server):
        build = program_build(server, zip_of({**PROGRAM_FILES, "../evil.txt": b"outside"}))
        with pytest.raises(CrispASRSetupError, match="leaves the folder"):
            install.ensure_program(build, offline=False)
        root = install.program_root()
        assert not (root / "evil.txt").exists()
        assert not install.program_dir(build).exists()
        assert sorted(p.name for p in root.iterdir()) == [".download"]
        assert list((root / ".download").iterdir()) == []

    @pytest.mark.parametrize("members, found", [
        ({f"{FOLDER}/ggml.dll": b"MZ"}, 0),
        ({"a/crispasr.exe": b"MZ", "b/crispasr.exe": b"MZ"}, 2),
    ], ids=["none", "two"])
    def test_an_archive_without_exactly_one_program_is_refused(self, server, members, found):
        build = program_build(server, zip_of(members))
        with pytest.raises(CrispASRSetupError, match=rf"exactly one crispasr\.exe \(found {found}\)"):
            install.ensure_program(build, offline=False)
        assert not install.program_dir(build).exists()
        assert not install.program_status(build).downloaded

    def test_a_program_at_the_top_of_the_archive_installs_with_its_files(self, server):
        build = program_build(server, zip_of({"crispasr.exe": b"MZ top", "ggml.dll": b"MZ"}))
        exe = install.ensure_program(build, offline=False)
        assert exe.read_bytes() == b"MZ top"
        assert (exe.parent / "ggml.dll").is_file()

    def test_a_record_that_does_not_match_is_replaced_by_a_fresh_download(self, server):
        build = program_build(server, zip_of(PROGRAM_FILES))
        exe = install.ensure_program(build, offline=False)
        record_path = exe.parent / ".verified"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["sha256"] = "0" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")
        (exe.parent / "left-over.dll").write_bytes(b"from before")
        assert install.ensure_program(build, offline=False) == exe
        assert server.hits[TEST_ASSET] == 2
        assert install.program_status(build).downloaded
        assert not (exe.parent / "left-over.dll").exists()

    def test_a_tar_archive_installs(self, server):
        payload = tar_gz_of({"crispasr-linux-x86_64/crispasr": b"\x7fELF stand-in",
                             "crispasr-linux-x86_64/libggml.so": b"\x7fELF"})
        build = program_build(server, payload, asset="crispasr-test-linux.tar.gz", platform_="linux")
        exe = install.ensure_program(build, offline=False)
        assert exe.name == "crispasr"
        assert exe.read_bytes() == b"\x7fELF stand-in"
        assert (exe.parent / "libggml.so").is_file()

    def test_a_tar_member_that_would_leave_the_folder_is_refused_like_a_zip_member(self, server):
        payload = tar_gz_of({"crispasr-linux-x86_64/crispasr": b"\x7fELF", "../evil.txt": b"outside"})
        build = program_build(server, payload, asset="crispasr-test-linux.tar.gz", platform_="linux")
        with pytest.raises(CrispASRSetupError):
            install.ensure_program(build, offline=False)
        assert not (install.program_root() / "evil.txt").exists()
        assert not install.program_dir(build).exists()


class TestTwoRunsFetchingTheSameBuild:
    """Two WhisperJAV runs (the GUI and a console, say) may fetch the same build at the same time."""

    def test_the_download_and_the_unpacking_folder_carry_this_process_id(self, server, monkeypatch):
        monkeypatch.setattr(install, "os", _OsWithPid(4242))
        seen = {}
        real_download, real_extract = install._download, install._safe_extract

        def download(url, dest, size, sha256, what):
            seen["download"] = Path(dest)
            return real_download(url, dest, size, sha256, what)

        def extract(archive, into):
            seen["archive"], seen["staging"] = Path(archive), Path(into)
            real_extract(archive, into)

        monkeypatch.setattr(install, "_download", download)
        monkeypatch.setattr(install, "_safe_extract", extract)
        build = program_build(server, zip_of(PROGRAM_FILES))
        exe = install.ensure_program(build, offline=False)
        root = install.program_root()
        assert seen["download"] == root / ".download" / f"4242-{TEST_ASSET}"
        assert seen["archive"] == seen["download"]
        assert seen["staging"] == root / ".cpu.staging-4242"
        assert not seen["download"].exists() and not seen["staging"].exists()
        assert exe == install.program_dir(build) / "crispasr.exe"
        assert install.program_status(build).downloaded

    def test_another_runs_download_and_unpacking_are_left_alone(self, server, monkeypatch):
        monkeypatch.setattr(install, "os", _OsWithPid(4242))
        root = install.program_root()
        theirs_part = root / ".download" / f"1111-{TEST_ASSET}.part"
        theirs_part.parent.mkdir(parents=True)
        theirs_part.write_bytes(b"half a download")
        theirs_staging = root / ".cpu.staging-1111" / FOLDER
        theirs_staging.mkdir(parents=True)
        (theirs_staging / "crispasr.exe").write_bytes(b"MZ theirs")
        build = program_build(server, zip_of(PROGRAM_FILES))
        assert install.ensure_program(build, offline=False).read_bytes() == b"MZ stand-in program"
        assert theirs_part.read_bytes() == b"half a download"
        assert (theirs_staging / "crispasr.exe").read_bytes() == b"MZ theirs"

    def test_a_build_another_run_finished_meanwhile_is_used_and_not_replaced(self, server, monkeypatch):
        build = program_build(server, zip_of(PROGRAM_FILES))
        real_extract = install._safe_extract
        folder = install.program_dir(build)

        def extract_while_another_run_finishes(archive, into):
            real_extract(archive, into)
            # Meanwhile another run finished the same download, and its program may already be running.
            folder.mkdir(parents=True)
            (folder / build.exe_name).write_bytes(b"MZ the other run's program")
            (folder / "in-use.dll").write_bytes(b"loaded by the other run")
            record = {"version": CRISPASR_VERSION, "asset": build.asset, "size": build.size,
                      "sha256": build.sha256}
            (folder / ".verified").write_text(json.dumps(record), encoding="utf-8")

        monkeypatch.setattr(install, "_safe_extract", extract_while_another_run_finishes)
        exe = install.ensure_program(build, offline=False)
        assert exe == folder / build.exe_name
        assert exe.read_bytes() == b"MZ the other run's program"
        assert (folder / "in-use.dll").read_bytes() == b"loaded by the other run"
        assert not (folder / "ggml.dll").exists()  # this run's unpacked files were not moved in
        root = install.program_root()
        assert sorted(p.name for p in root.iterdir()) == [".download", "cpu"]  # its unpacking folder is gone
        assert list((root / ".download").iterdir()) == []                      # and its archive
        assert server.hits[TEST_ASSET] == 1


class TestProgramStatus:
    BUILD = ProgramBuild("cpu", "CPU build", "win32", ("AMD64",), TEST_ASSET, 100, "a" * 64)

    def _install_by_hand(self, exe=True, **record_changes):
        folder = install.program_dir(self.BUILD)
        folder.mkdir(parents=True)
        if exe:
            (folder / "crispasr.exe").write_bytes(b"MZ")
        record = {"version": CRISPASR_VERSION, "asset": TEST_ASSET, "size": 100, "sha256": "a" * 64}
        record.update(record_changes)
        (folder / ".verified").write_text(json.dumps(record), encoding="utf-8")
        return folder

    def test_the_folder_is_under_whisperjavs_cache(self, places):
        assert install.program_dir(self.BUILD) == places.cache_root / "crispasr" / CRISPASR_VERSION / "cpu"

    def test_nothing_there(self):
        assert install.program_status(self.BUILD) == ProgramStatus(self.BUILD, False)

    def test_a_matching_record_and_the_program(self):
        folder = self._install_by_hand()
        assert install.program_status(self.BUILD) == ProgramStatus(self.BUILD, True, folder / "crispasr.exe")

    @pytest.mark.parametrize("field, value", [("version", "0.8.42"), ("asset", "other.zip"), ("sha256", "b" * 64)])
    def test_a_record_of_other_bytes_is_not_downloaded(self, field, value):
        self._install_by_hand(**{field: value})
        assert not install.program_status(self.BUILD).downloaded

    def test_a_record_without_the_program_is_not_downloaded(self):
        self._install_by_hand(exe=False)
        assert not install.program_status(self.BUILD).downloaded

    def test_an_unreadable_record_is_not_downloaded(self):
        folder = self._install_by_hand()
        (folder / ".verified").write_text("{not json", encoding="utf-8")
        assert not install.program_status(self.BUILD).downloaded


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
# The CUDA 13 runtime files
# ---------------------------------------------------------------------------------------------------

@pytest.fixture
def torch_lib(tmp_path):
    """PyTorch's library folder with the three CUDA 13 runtime files (small stand-ins)."""
    lib = tmp_path / "env" / "Lib" / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    for i, name in enumerate(CUDA13_RUNTIME_FILES):
        (lib / name).write_bytes(f"{name} stand-in ".encode() * (i + 3))
    return lib


@pytest.fixture
def program_folder(tmp_path):
    folder = tmp_path / "program"
    folder.mkdir()
    (folder / "crispasr.exe").write_bytes(b"MZ")
    return folder


class TestCudaRuntimeFiles:
    def test_the_files_are_hard_linked_beside_the_program(self, program_folder, torch_lib):
        assert install.place_cuda_runtime(program_folder, torch_lib) == "linked"
        for name in CUDA13_RUNTIME_FILES:
            src, dst = torch_lib / name, program_folder / name
            assert os.path.samefile(src, dst)
            assert os.stat(dst).st_nlink >= 2
            assert dst.stat().st_size == src.stat().st_size

    def test_files_already_in_place_are_left_alone(self, program_folder, torch_lib, monkeypatch):
        install.place_cuda_runtime(program_folder, torch_lib)
        linked = []
        real_link = os.link
        monkeypatch.setattr(os, "link", lambda src, dst: linked.append(dst) or real_link(src, dst))
        assert install.place_cuda_runtime(program_folder, torch_lib) == "ready"
        assert linked == []

    def test_a_file_whose_size_changed_is_linked_again(self, program_folder, torch_lib):
        install.place_cuda_runtime(program_folder, torch_lib)
        name = CUDA13_RUNTIME_FILES[1]
        (torch_lib / name).unlink()  # a PyTorch update replaces the file
        (torch_lib / name).write_bytes(b"a newer runtime, of another size")
        assert install.place_cuda_runtime(program_folder, torch_lib) == "linked"
        assert (program_folder / name).read_bytes() == b"a newer runtime, of another size"
        assert os.path.samefile(torch_lib / name, program_folder / name)

    def test_a_copy_is_made_where_a_link_is_impossible(self, program_folder, torch_lib, monkeypatch):
        before = {name: (torch_lib / name).read_bytes() for name in CUDA13_RUNTIME_FILES}

        def no_link(src, dst):
            raise OSError(17, "The system cannot move the file to a different disk drive")

        monkeypatch.setattr(os, "link", no_link)
        assert install.place_cuda_runtime(program_folder, torch_lib) == "copied"
        for name in CUDA13_RUNTIME_FILES:
            dst = program_folder / name
            assert dst.read_bytes() == before[name]
            assert not os.path.samefile(torch_lib / name, dst)
            assert (torch_lib / name).read_bytes() == before[name]  # PyTorch's own file is untouched
            assert os.stat(torch_lib / name).st_nlink == 1

    def test_a_missing_runtime_file_stops_with_its_name(self, program_folder, torch_lib):
        (torch_lib / "cublasLt64_13.dll").unlink()
        with pytest.raises(CrispASRSetupError, match="cublasLt64_13.dll"):
            install.place_cuda_runtime(program_folder, torch_lib)


class TestPyTorchsLibraryFolder:
    """Found from the package's location, never by importing torch."""

    def test_the_lib_folder_of_the_torch_package(self, monkeypatch, tmp_path):
        package = tmp_path / "site-packages" / "torch"
        asked = []

        def find_spec(name):
            asked.append(name)
            return types.SimpleNamespace(submodule_search_locations=[str(package)])

        monkeypatch.setattr(install.importlib.util, "find_spec", find_spec)
        assert install.torch_lib_dir() == package / "lib"
        assert asked == ["torch"]

    @pytest.mark.parametrize("spec", [None, types.SimpleNamespace(submodule_search_locations=None)],
                             ids=["not installed", "not a package"])
    def test_no_torch_package_means_no_folder(self, monkeypatch, spec):
        monkeypatch.setattr(install.importlib.util, "find_spec", lambda name: spec)
        assert install.torch_lib_dir() is None

    def test_a_broken_torch_entry_means_no_folder(self, monkeypatch):
        def broken(name):
            raise ValueError("torch.__spec__ is None")

        monkeypatch.setattr(install.importlib.util, "find_spec", broken)
        assert install.torch_lib_dir() is None


class TestCuda13Availability:
    """Why the CUDA 13 build cannot be tried here, and whether the CPU build is then the PC's normal set-up
    (expected: a console line only) or a fallback that goes into each file's summary (plan 1.4, 2.6).
    PyTorch is a stand-in module in sys.modules; the real one is never loaded."""

    @pytest.fixture(autouse=True)
    def lib(self, monkeypatch, torch_lib):
        monkeypatch.setattr(install, "torch_lib_dir", lambda: torch_lib)
        return torch_lib

    @pytest.mark.parametrize("version", ["13.0", "13.2"])
    def test_everything_in_place(self, monkeypatch, version):
        _fake_torch(monkeypatch, cuda=True, cuda_version=version)
        assert install.cuda13_unavailable() is None
        assert install.cuda13_unavailable_reason() is None

    def test_pytorch_that_cannot_be_loaded_is_not_expected(self, monkeypatch):
        _no_torch(monkeypatch)
        why = "PyTorch could not be loaded to look for an NVIDIA GPU"
        assert install.cuda13_unavailable() == CudaUnavailable(why, expected=False)
        assert install.cuda13_unavailable_reason() == why

    @pytest.mark.parametrize("version", [None, ""], ids=["none", "empty"])
    def test_a_cpu_only_pytorch_on_a_pc_with_an_nvidia_card(self, monkeypatch, version):
        # A CPU-only PyTorch answers "no GPU" even where an NVIDIA card is fitted. Its own build is asked
        # first, so the user is told what this install lacks, not that the PC has no NVIDIA GPU.
        _fake_torch(monkeypatch, cuda=False, cuda_version=version)
        unavailable = install.cuda13_unavailable()
        assert unavailable == CudaUnavailable("this install's PyTorch has no CUDA support", expected=True)
        assert "NVIDIA" not in unavailable.reason

    def test_a_pytorch_built_for_cuda_12(self, monkeypatch):
        _fake_torch(monkeypatch, cuda=True, cuda_version="12.8")
        assert install.cuda13_unavailable() == CudaUnavailable(
            "this install's PyTorch is built for CUDA 12.8, not CUDA 13", expected=True)
        assert install.cuda13_unavailable_reason() == "this install's PyTorch is built for CUDA 12.8, not CUDA 13"

    @pytest.mark.parametrize("cuda", [False, RuntimeError("driver too old")], ids=["none", "query fails"])
    def test_no_nvidia_gpu(self, monkeypatch, cuda):
        _fake_torch(monkeypatch, cuda=cuda)
        assert install.cuda13_unavailable() == CudaUnavailable("no NVIDIA GPU was found", expected=True)
        assert install.cuda13_unavailable_reason() == "no NVIDIA GPU was found"

    def test_missing_runtime_files_are_named_and_not_expected(self, monkeypatch, lib):
        _fake_torch(monkeypatch)
        (lib / "cublas64_13.dll").unlink()
        (lib / "cublasLt64_13.dll").unlink()
        assert install.cuda13_unavailable() == CudaUnavailable(
            "PyTorch's CUDA 13 runtime files are missing (cublas64_13.dll, cublasLt64_13.dll)", expected=False)

    def test_no_pytorch_folder_means_every_file_is_missing(self, monkeypatch):
        monkeypatch.setattr(install, "torch_lib_dir", lambda: None)
        _fake_torch(monkeypatch)
        assert install.cuda13_unavailable() == CudaUnavailable(
            "PyTorch's CUDA 13 runtime files are missing (cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll)",
            expected=False)


# ---------------------------------------------------------------------------------------------------
# The start check (crispasr --diagnostics)
# ---------------------------------------------------------------------------------------------------

CPU_DEVICE = "    [0] cpu    name=CPU desc=AMD Ryzen 7 5800X 8-Core Processor mem=0/0 MiB id=?\n"
CUDA_DEVICE = "    [1] gpu    name=CUDA0 desc=NVIDIA GeForce RTX 3060 mem=11000/12287 MiB id=0000:01:00.0\n"
IGPU_DEVICE = "    [2] igpu   name=Vulkan0 desc=AMD Radeon(TM) Graphics mem=512/16000 MiB id=?\n"


def build_info(version="0.8.43", git="eb633d90c4f2a7d1", backends="cpu,cuda"):
    lines = ["=== build info ==="]
    if version is not None:
        lines.append(f"  version       : {version}")
    lines.append(f"  git sha       : {git}")
    lines.append(f"  ggml backends : {backends}")
    return "\n".join(lines) + "\n"


def devices(*lines):
    return "=== ggml backends + devices ===\n" + "".join(lines)


def diagnostics(devices_text=devices(CPU_DEVICE, CUDA_DEVICE), **info):
    return build_info(**info) + devices_text


WIN_BUILDS = manifest.builds_for_this_machine("win32", "AMD64")
GPU_BUILD = next(b for b in WIN_BUILDS if b.gpu == "cuda13")
CPU_BUILD = next(b for b in WIN_BUILDS if b.gpu is None)


@pytest.fixture
def program_answers(monkeypatch):
    """subprocess.run replaced: the start check gets ``state.result`` or raises ``state.error``."""
    state = types.SimpleNamespace(calls=[], result=None, error=None)

    def fake_run(args, **kwargs):
        state.calls.append((list(args), kwargs))
        if state.error is not None:
            raise state.error
        return state.result

    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


def answer(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["crispasr", "--diagnostics"], returncode,
                                       stdout=stdout.encode("utf-8"), stderr=stderr.encode("utf-8"))


class TestStartCheck:
    def test_parse_the_cuda_builds_diagnostics(self):
        check = install.parse_diagnostics(diagnostics())
        assert check.version == "0.8.43"
        assert check.git == "eb633d90c4f2a7d1"
        assert check.compiled_backends == ["cpu", "cuda"]
        assert check.gpu_devices == ["CUDA0: NVIDIA GeForce RTX 3060"]
        assert (check.ok, check.reason) == (False, None)  # parsing alone never passes a build

    def test_parse_the_cpu_builds_diagnostics(self):
        check = install.parse_diagnostics(diagnostics(devices(CPU_DEVICE), backends="cpu"))
        assert check.compiled_backends == ["cpu"]
        assert check.gpu_devices == []

    def test_an_integrated_gpu_is_listed_as_a_device(self):
        check = install.parse_diagnostics(diagnostics(devices(CPU_DEVICE, IGPU_DEVICE)))
        assert check.gpu_devices == ["Vulkan0: AMD Radeon(TM) Graphics"]

    def test_parse_nothing(self):
        check = install.parse_diagnostics("")
        assert (check.version, check.git, check.compiled_backends, check.gpu_devices) == (None, None, [], [])

    def test_a_cpu_build_that_answers_is_ok(self, program_answers, tmp_path):
        exe = tmp_path / "cpu" / "crispasr.exe"
        program_answers.result = answer(diagnostics(devices(CPU_DEVICE), backends="cpu"))
        check = install.start_check(exe, CPU_BUILD)
        assert (check.ok, check.reason, check.exit_code) == (True, None, 0)
        ((args, kwargs),) = program_answers.calls
        assert args == [str(exe), "--diagnostics"]
        assert kwargs["cwd"] == str(exe.parent)
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 120

    def test_the_gpu_build_with_a_cuda_device_is_ok(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics())
        assert install.start_check(tmp_path / "crispasr.exe", GPU_BUILD).ok

    def test_the_gpu_build_without_a_cuda_device_is_refused(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(devices(CPU_DEVICE)))
        check = install.start_check(tmp_path / "crispasr.exe", GPU_BUILD)
        assert (check.ok, check.reason) == (False, "it found no CUDA device")

    def test_a_device_that_is_not_cuda_does_not_count_for_the_gpu_build(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(devices(CPU_DEVICE, IGPU_DEVICE)))
        assert install.start_check(tmp_path / "crispasr.exe", GPU_BUILD).reason == "it found no CUDA device"

    def test_devices_printed_on_stderr_count_too(self, program_answers, tmp_path):
        program_answers.result = answer(build_info(), devices(CPU_DEVICE, CUDA_DEVICE))
        assert install.start_check(tmp_path / "crispasr.exe", GPU_BUILD).ok

    def test_another_version_is_refused(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(version="0.8.42"))
        check = install.start_check(tmp_path / "crispasr.exe", GPU_BUILD)
        assert (check.ok, check.reason) == (False, "it reports version 0.8.42, not 0.8.43")

    def test_no_version_is_refused(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(version=None))
        check = install.start_check(tmp_path / "crispasr.exe", CPU_BUILD)
        assert (check.ok, check.reason) == (False, "it reports version unknown, not 0.8.43")

    def test_another_git_build_is_refused(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(git="deadbeef1234"))
        check = install.start_check(tmp_path / "crispasr.exe", GPU_BUILD)
        assert (check.ok, check.reason) == (False, f"it reports build deadbeef1234, not {CRISPASR_GIT}")

    def test_the_short_git_form_of_the_banner_is_accepted(self, program_answers, tmp_path):
        program_answers.result = answer(diagnostics(git=CRISPASR_GIT))
        assert install.start_check(tmp_path / "crispasr.exe", GPU_BUILD).ok

    @pytest.mark.parametrize("code", [0xC0000135, 0xC000001D, 12])
    def test_a_program_that_fails_gives_the_meaning_of_its_exit_code(self, program_answers, tmp_path, code):
        program_answers.result = answer(diagnostics(), returncode=code)
        check = install.start_check(tmp_path / "crispasr.exe", GPU_BUILD)
        assert (check.ok, check.reason, check.exit_code) == (False, exit_code_meaning(code), code)

    def test_a_program_that_does_not_answer_in_time(self, program_answers, tmp_path):
        program_answers.error = subprocess.TimeoutExpired(cmd=["crispasr", "--diagnostics"], timeout=5)
        check = install.start_check(tmp_path / "crispasr.exe", CPU_BUILD, timeout_s=5)
        assert (check.ok, check.reason, check.exit_code) == (False, "it did not answer within 5 s", None)

    def test_a_program_that_cannot_be_started(self, program_answers, tmp_path):
        program_answers.error = OSError(193, "%1 is not a valid Win32 application")
        check = install.start_check(tmp_path / "crispasr.exe", CPU_BUILD)
        assert check.ok is False
        assert check.reason.startswith("it could not be started (")


# ---------------------------------------------------------------------------------------------------
# Which build runs
# ---------------------------------------------------------------------------------------------------

OK_CHECK = StartCheck(True, None, version=CRISPASR_VERSION)

# Why the CUDA 13 build cannot be tried (install.cuda13_unavailable), as prepare_program receives it.
NO_NVIDIA_GPU = CudaUnavailable("no NVIDIA GPU was found", expected=True)
CPU_ONLY_PYTORCH = CudaUnavailable("this install's PyTorch has no CUDA support", expected=True)
CUDA12_PYTORCH = CudaUnavailable("this install's PyTorch is built for CUDA 12.8, not CUDA 13", expected=True)
RUNTIME_FILES_MISSING = CudaUnavailable("PyTorch's CUDA 13 runtime files are missing (cublas64_13.dll)",
                                        expected=False)
PYTORCH_NOT_LOADABLE = CudaUnavailable("PyTorch could not be loaded to look for an NVIDIA GPU", expected=False)


class Machine:
    """prepare_program's collaborators, all stand-ins, with a record of what was asked of them."""

    def __init__(self, monkeypatch, tmp_path, builds=WIN_BUILDS, avx2=True, unavailable=None, checks=None,
                 place_error=None):
        self.downloads, self.placed, self.checked = [], [], []
        self.avx2_asked = 0
        self.lib = tmp_path / "torch" / "lib"
        self.programs = tmp_path / "programs"
        checks = checks or {}

        def has_avx2():
            self.avx2_asked += 1
            return avx2

        def ensure_program(build, offline=None):
            self.downloads.append((build.key, offline))
            return self.programs / build.key / build.exe_name

        def place_cuda_runtime(folder, source):
            self.placed.append((folder, source))
            if place_error is not None:
                raise place_error
            return "linked"

        def start_check(exe, build, timeout_s=120):
            self.checked.append(build.key)
            return checks.get(build.key, OK_CHECK)

        monkeypatch.setattr(manifest, "builds_for_this_machine", lambda *args, **kwargs: list(builds))
        monkeypatch.setattr(install, "has_avx2", has_avx2)
        # Both forms answer from the stand-in, so PyTorch itself is never asked.
        monkeypatch.setattr(install, "cuda13_unavailable", lambda: unavailable)
        monkeypatch.setattr(install, "cuda13_unavailable_reason", lambda: unavailable.reason if unavailable else None)
        monkeypatch.setattr(install, "torch_lib_dir", lambda: self.lib)
        monkeypatch.setattr(install, "ensure_program", ensure_program)
        monkeypatch.setattr(install, "place_cuda_runtime", place_cuda_runtime)
        monkeypatch.setattr(install, "start_check", start_check)


class TestWhichBuildRuns:
    def test_the_gpu_build_when_everything_is_in_place(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path)
        choice = install.prepare_program(offline=False)
        assert choice.build is GPU_BUILD
        assert choice.exe == m.programs / "cuda13" / "crispasr.exe"
        assert choice.check.ok and choice.note is None
        assert choice.note_in_summary is False
        assert m.downloads == [("cuda13", False)]
        assert m.placed == [(choice.exe.parent, m.lib)]
        assert m.checked == ["cuda13"]

    def test_a_gpu_build_failing_its_start_check_falls_back_to_cpu_with_the_reason(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path, checks={"cuda13": StartCheck(False, "it found no CUDA device")})
        choice = install.prepare_program(offline=False)
        assert choice.build is CPU_BUILD
        assert choice.exe == m.programs / "cpu" / "crispasr.exe"
        assert choice.note == ("CrispASR runs on the processor (CPU build), which is slower: "
                               "the GPU build's start check failed: it found no CUDA device.")
        # Not expected on a PC whose PyTorch has everything the GPU build needs: in each file's summary.
        assert choice.note_in_summary is True
        assert m.downloads == [("cuda13", False), ("cpu", False)]
        assert m.checked == ["cuda13", "cpu"]

    @pytest.mark.parametrize("error", [
        CrispASRSetupError("PyTorch's cudart64_13.dll was not found in X"),
        PermissionError(13, "The process cannot access the file because it is being used by another process"),
    ], ids=["file missing", "file in use"])
    def test_runtime_files_that_cannot_be_placed_fall_back_to_cpu(self, monkeypatch, tmp_path, error):
        m = Machine(monkeypatch, tmp_path, place_error=error)
        choice = install.prepare_program(offline=False)
        assert choice.build is CPU_BUILD
        assert "PyTorch's CUDA 13 runtime files could not be placed" in choice.note
        assert choice.note_in_summary is True
        assert m.checked == ["cpu"]  # the GPU build was never started

    def test_without_an_nvidia_gpu_the_gpu_build_is_never_downloaded(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path, unavailable=NO_NVIDIA_GPU)
        choice = install.prepare_program(offline=False)
        assert choice.build is CPU_BUILD
        assert choice.note == "CrispASR runs on the processor (CPU build), which is slower: no NVIDIA GPU was found."
        assert choice.note_in_summary is False
        assert m.downloads == [("cpu", False)]
        assert m.placed == []

    @pytest.mark.parametrize("unavailable", [NO_NVIDIA_GPU, CPU_ONLY_PYTORCH, CUDA12_PYTORCH],
                             ids=["no nvidia gpu", "cpu-only pytorch", "cuda 12 pytorch"])
    def test_an_expected_cpu_build_is_a_console_line_only(self, monkeypatch, tmp_path, unavailable):
        # The PC's normal set-up for the CPU build: one console line per pass, no note in the summary.
        m = Machine(monkeypatch, tmp_path, unavailable=unavailable)
        choice = install.prepare_program(offline=False)
        assert choice.build is CPU_BUILD
        assert choice.note == f"CrispASR runs on the processor (CPU build), which is slower: {unavailable.reason}."
        assert choice.note_in_summary is False
        assert (m.downloads, m.placed, m.checked) == ([("cpu", False)], [], ["cpu"])

    @pytest.mark.parametrize("unavailable", [RUNTIME_FILES_MISSING, PYTORCH_NOT_LOADABLE],
                             ids=["runtime files missing", "pytorch not loadable"])
    def test_an_unexpected_cpu_build_also_goes_into_the_summary(self, monkeypatch, tmp_path, unavailable):
        m = Machine(monkeypatch, tmp_path, unavailable=unavailable)
        choice = install.prepare_program(offline=False)
        assert choice.build is CPU_BUILD
        assert choice.note == f"CrispASR runs on the processor (CPU build), which is slower: {unavailable.reason}."
        assert choice.note_in_summary is True
        assert (m.downloads, m.placed, m.checked) == ([("cpu", False)], [], ["cpu"])

    def test_a_processor_without_avx2_stops_before_any_download(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path, avx2=False)
        with pytest.raises(CrispASRSetupError, match="lacks AVX2"):
            install.prepare_program(offline=False)
        assert (m.downloads, m.checked) == ([], [])

    def test_avx2_that_cannot_be_told_is_no_stop(self, monkeypatch, tmp_path):
        Machine(monkeypatch, tmp_path, avx2=None)
        assert install.prepare_program(offline=False).build is GPU_BUILD

    def test_no_build_for_this_computer(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path, builds=[])
        with pytest.raises(CrispASRSetupError, match="no program build for this computer"):
            install.prepare_program(offline=False)
        assert m.downloads == []

    def test_a_cpu_build_that_cannot_start_stops_the_run(self, monkeypatch, tmp_path):
        reason = exit_code_meaning(0xC000001D)
        Machine(monkeypatch, tmp_path, unavailable=NO_NVIDIA_GPU, checks={"cpu": StartCheck(False, reason)})
        with pytest.raises(CrispASRSetupError) as caught:
            install.prepare_program(offline=False)
        assert str(caught.value) == f"CrispASR's CPU build cannot start on this computer: {reason}."

    def test_a_gpu_build_with_no_cpu_build_to_fall_back_on(self, monkeypatch, tmp_path):
        Machine(monkeypatch, tmp_path, builds=[GPU_BUILD], unavailable=NO_NVIDIA_GPU)
        with pytest.raises(CrispASRSetupError, match="no CPU build"):
            install.prepare_program(offline=False)

    def test_apple_silicon_runs_the_cpu_build_and_is_never_asked_about_avx2(self, monkeypatch, tmp_path):
        mac_builds = manifest.builds_for_this_machine("darwin", "arm64")
        m = Machine(monkeypatch, tmp_path, builds=mac_builds, avx2=False)
        choice = install.prepare_program(offline=False)
        assert choice.build is mac_builds[0]
        assert choice.note == "CrispASR runs on the processor (CPU build), which is slower."
        assert choice.note_in_summary is False  # its only build: the normal set-up
        assert m.avx2_asked == 0
        assert m.downloads == [("cpu", False)]

    def test_the_offline_switch_reaches_the_download(self, monkeypatch, tmp_path):
        m = Machine(monkeypatch, tmp_path)
        install.prepare_program(offline=True)
        assert m.downloads == [("cuda13", True)]


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


class TestLinkOrCopy:
    @pytest.fixture
    def src(self, tmp_path):
        path = tmp_path / "source" / "テスト.bin"
        path.parent.mkdir()
        path.write_bytes(MODEL_BYTES)
        return path

    def test_a_hard_link_where_possible(self, tmp_path, src):
        dst = tmp_path / "linked.bin"
        assert install.link_or_copy(src, dst) == "linked"
        assert os.path.samefile(src, dst)

    def test_a_copy_where_a_link_is_impossible(self, tmp_path, src, monkeypatch):
        def no_link(a, b):
            raise OSError(17, "The system cannot move the file to a different disk drive")

        monkeypatch.setattr(os, "link", no_link)
        dst = tmp_path / "copied.bin"
        assert install.link_or_copy(src, dst) == "copied"
        assert dst.read_bytes() == MODEL_BYTES
        assert not os.path.samefile(src, dst)

    @pytest.mark.parametrize("linkable", [True, False], ids=["link", "copy"])
    def test_an_existing_file_is_replaced(self, tmp_path, src, monkeypatch, linkable):
        if not linkable:
            def no_link(a, b):
                raise OSError(1, "no links on this drive")

            monkeypatch.setattr(os, "link", no_link)
        dst = tmp_path / "target.bin"
        dst.write_bytes(b"an older file")
        install.link_or_copy(src, str(dst))
        assert dst.read_bytes() == MODEL_BYTES
        assert src.read_bytes() == MODEL_BYTES

    def test_a_copy_named_for_its_purpose_is_refused_without_the_disk_space(self, tmp_path, src, monkeypatch):
        def no_link(a, b):
            raise OSError(17, "The system cannot move the file to a different disk drive")

        asked = []
        monkeypatch.setattr(os, "link", no_link)
        monkeypatch.setattr(shutil, "disk_usage",
                            lambda path: asked.append(Path(path)) or types.SimpleNamespace(total=1, used=1, free=0))
        dst = tmp_path / "run" / "audio.wav"
        dst.parent.mkdir()
        with pytest.raises(CrispASRSetupError, match="Not enough disk space for a copy of the audio"):
            install.link_or_copy(src, dst, what="a copy of the audio")
        assert not dst.exists()
        assert asked == [dst.parent]

    def test_a_link_needs_no_disk_space_and_is_not_checked(self, tmp_path, src, monkeypatch):
        def never(path):
            raise AssertionError("a hard link takes no disk space; the free space must not be asked")

        monkeypatch.setattr(shutil, "disk_usage", never)
        assert install.link_or_copy(src, tmp_path / "linked.bin", what="a copy of the audio") == "linked"

    def test_a_copy_not_named_is_not_checked(self, tmp_path, src, monkeypatch):
        def no_link(a, b):
            raise OSError(1, "no links on this drive")

        def never(path):
            raise AssertionError("no purpose was named, so the free space is not asked")

        monkeypatch.setattr(os, "link", no_link)
        monkeypatch.setattr(shutil, "disk_usage", never)
        assert install.link_or_copy(src, tmp_path / "copied.bin") == "copied"


def _no_link(src, dst):
    raise OSError(17, "The system cannot move the file to a different disk drive")


class TestStageModel:
    """What -m gets (plan 1.7): an all-ASCII path as it is; any other path staged once per pass, in the
    folder that holds every film's run folder, and named relative to the film's run folder."""

    ASCII_NAME = "parakeet-tdt-0.6b-ja-q8_0.gguf"

    @pytest.fixture
    def out(self, tmp_path):
        folder = tmp_path / "crispasr_out"  # the folder that holds every film's run folder
        folder.mkdir()
        return folder

    @pytest.fixture
    def run_dir(self, out):
        folder = out / "film_0123abcd"
        folder.mkdir()
        return folder

    def _model_in(self, tmp_path, *parts, name="テスト-q8_0.gguf"):
        folder = tmp_path.joinpath(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(MODEL_BYTES)
        return path

    def test_an_ascii_path_is_passed_as_it_is(self, run_dir, out):
        model = Path("C:/models/parakeet-tdt-0.6b-ja-q8_0.gguf")
        assert install.stage_model(model, run_dir, stage_dir=out) == str(model)
        assert install.stage_model(model, run_dir) == str(model)
        assert list(run_dir.iterdir()) == [] and list(out.iterdir()) == [run_dir]

    @pytest.mark.parametrize("parts", [("ユーザー", "モデル"), ("用户", "模型")], ids=["japanese", "chinese"])
    def test_a_path_outside_ascii_is_linked_beside_the_run_folders_and_named_from_the_film(self, tmp_path, out,
                                                                                         run_dir, parts):
        model = self._model_in(tmp_path, *parts, name=self.ASCII_NAME)
        arg = install.stage_model(model, run_dir, stage_dir=out)
        assert arg == os.path.join("..", self.ASCII_NAME)  # "..\parakeet-...gguf" on Windows
        assert arg.isascii()
        staged = out / self.ASCII_NAME
        assert os.path.samefile(run_dir / arg, staged)  # the program opens it from the film's run folder
        assert os.path.samefile(staged, model)
        assert os.stat(staged).st_nlink >= 2
        assert list(run_dir.iterdir()) == []

    def test_a_file_name_outside_ascii_is_staged_as_model_gguf(self, tmp_path, out, run_dir):
        model = self._model_in(tmp_path, "ユーザー")
        assert install.stage_model(model, run_dir, stage_dir=out) == os.path.join("..", "model.gguf")
        assert os.path.samefile(out / "model.gguf", model)

    def test_without_a_stage_folder_the_run_folder_is_used(self, tmp_path, run_dir):
        model = self._model_in(tmp_path, "ユーザー")
        assert install.stage_model(model, run_dir) == "model.gguf"
        assert os.path.samefile(run_dir / "model.gguf", model)

    def test_a_stage_folder_not_there_yet_is_made(self, tmp_path, run_dir):
        model = self._model_in(tmp_path, "ユーザー", name=self.ASCII_NAME)
        stage = tmp_path / "pass_models"
        arg = install.stage_model(model, run_dir, stage_dir=stage)
        assert (stage / self.ASCII_NAME).is_file()
        assert os.path.samefile(run_dir / arg, stage / self.ASCII_NAME)

    @pytest.mark.parametrize("linkable", [True, False], ids=["link", "copy"])
    def test_the_next_film_of_the_pass_uses_the_staged_file_without_staging_it_again(self, tmp_path, out,
                                                                                    monkeypatch, linkable):
        model = self._model_in(tmp_path, "ユーザー", name=self.ASCII_NAME)
        if not linkable:
            monkeypatch.setattr(os, "link", _no_link)
        staged_to = []
        real = install.link_or_copy
        monkeypatch.setattr(install, "link_or_copy",
                            lambda src, dst, what=None: staged_to.append(Path(dst)) or real(src, dst, what=what))
        first, second = out / "film_a_00000001", out / "film_b_00000002"
        first.mkdir()
        second.mkdir()
        a = install.stage_model(model, first, stage_dir=out)
        b = install.stage_model(model, second, stage_dir=out)
        assert a == b == os.path.join("..", self.ASCII_NAME)
        assert staged_to == [out / self.ASCII_NAME]  # one copy for the pass, not one per film
        assert (second / b).read_bytes() == MODEL_BYTES

    def test_a_copy_is_made_where_a_link_is_impossible(self, tmp_path, out, run_dir, monkeypatch):
        model = self._model_in(tmp_path, "ユーザー")
        monkeypatch.setattr(os, "link", _no_link)
        assert install.stage_model(model, run_dir, stage_dir=out) == os.path.join("..", "model.gguf")
        staged = out / "model.gguf"
        assert staged.read_bytes() == MODEL_BYTES
        assert not os.path.samefile(staged, model)

    def test_a_staged_file_of_another_size_is_replaced(self, tmp_path, out, run_dir):
        (out / "model.gguf").write_bytes(b"an older model")
        model = self._model_in(tmp_path, "ユーザー")
        install.stage_model(model, run_dir, stage_dir=out)
        assert (out / "model.gguf").read_bytes() == MODEL_BYTES

    def test_a_copy_without_the_disk_space_for_it_is_refused(self, tmp_path, out, run_dir, monkeypatch):
        model = self._model_in(tmp_path, "ユーザー", name=self.ASCII_NAME)
        asked, copied = [], []
        real_copy = shutil.copy2
        monkeypatch.setattr(os, "link", _no_link)
        monkeypatch.setattr(shutil, "disk_usage", lambda path: asked.append(Path(path)) or types.SimpleNamespace(
            total=10**12, used=10**12, free=len(MODEL_BYTES)))  # the file fits, its 10% margin does not
        monkeypatch.setattr(shutil, "copy2", lambda *args, **kwargs: copied.append(args) or real_copy(*args, **kwargs))
        with pytest.raises(CrispASRSetupError, match="Not enough disk space for a copy of the model"):
            install.stage_model(model, run_dir, stage_dir=out)
        assert asked == [out]
        assert copied == []
        assert not (out / self.ASCII_NAME).exists()


class TestImportsStayLight:
    def test_the_package_loads_no_torch_no_asr_module_and_no_download_library(self):
        code = (
            "import sys\n"
            "import whisperjav.crispasr.install, whisperjav.crispasr.runner\n"
            "import whisperjav.crispasr.ojf, whisperjav.crispasr.manifest\n"
            "heavy = sorted(m for m in sys.modules\n"
            "               if m.split('.')[0] in ('torch', 'requests', 'huggingface_hub', 'transformers')\n"
            "               or m.startswith(('whisperjav.modules', 'whisperjav.pipelines', 'whisperjav.main')))\n"
            "print(','.join(heavy))\n"
        )
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO), capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == ""
