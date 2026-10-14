"""Drift guards added in v1.9.2, after the installer shipped without faster-whisper.

In v1.9.2 the installer's requirements generator silently dropped anything
git-addressed, and the fallback template still asked PyPI for
``faster-whisper>=1.1.0`` -- a build with different Silero VAD weights than the
one the maintainer tests against.

v1.10.0 (item 2f): the generator and the fallback template are gone. Every
install takes the locked files ``whisperjav/installer/locked/requirements_<build>.txt``
-- ``uv.lock`` exported once per PyTorch build by ``installer/build_release.py`` --
which carry the git packages at the commits the lock records. These tests check
those files against ``pyproject.toml``, the registry and a fresh export.
"""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10 without tomli
    tomllib = None

PROJECT_ROOT = Path(__file__).parent.parent
LOCKED_DIR = PROJECT_ROOT / "whisperjav" / "installer" / "locked"
VARIANTS = ("cu130", "cu128", "cpu")


def _pyproject_specs() -> dict:
    """Map lower-cased package name -> its full spec string in pyproject.toml."""
    if tomllib is None:
        pytest.skip("tomllib not available")

    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    project = config["project"]
    specs = {}

    # Composite extras only re-export other extras; reading them would double up.
    composite = {"all", "colab", "kaggle", "windows", "unix"}

    for dep in project.get("dependencies", []):
        specs[re.split(r"[>=<\[@ ;]", dep)[0].strip().lower()] = dep.strip()

    for extra, deps in project.get("optional-dependencies", {}).items():
        if extra in composite:
            continue
        for dep in deps:
            if dep.startswith("whisperjav["):
                continue
            specs[re.split(r"[>=<\[@ ;]", dep)[0].strip().lower()] = dep.strip()

    return specs


def _locked_lines(variant: str) -> list:
    """The requirement lines of one locked file, comments and blank lines dropped."""
    content = (LOCKED_DIR / f"requirements_{variant}.txt").read_text(encoding="utf-8")
    return [l.strip() for l in content.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def _name(line: str) -> str:
    return re.split(r"[>=<\[@ ;]", line)[0].strip().lower()


class TestGitPinDrift:
    """A git URL must read the same in the registry and in pyproject.toml."""

    def test_registry_git_urls_match_pyproject(self):
        from whisperjav.installer.core.registry import PACKAGES, InstallSource

        pyproject = _pyproject_specs()

        git_packages = [p for p in PACKAGES if p.source == InstallSource.GIT]
        assert git_packages, "No git packages in the registry — is the registry importable?"

        mismatches = []
        for pkg in git_packages:
            name = pkg.name.lower()
            if name not in pyproject:
                mismatches.append(
                    f"  {pkg.name}: in the registry as a git package but absent from pyproject.toml"
                )
                continue

            spec = pyproject[name]
            if " @ " not in spec:
                mismatches.append(
                    f"  {pkg.name}: registry says git ('{pkg.git_url}') but pyproject.toml "
                    f"asks PyPI for '{spec}'"
                )
                continue

            pyproject_url = spec.split(" @ ", 1)[1].split(";")[0].strip()
            if pyproject_url != pkg.git_url:
                mismatches.append(
                    f"  {pkg.name}: registry='{pkg.git_url}' vs pyproject.toml='{pyproject_url}'"
                )

        assert not mismatches, (
            "Git pins differ between the registry and pyproject.toml. A source install "
            "and the Windows installer would then build from different commits:\n"
            + "\n".join(mismatches)
        )

    def test_faster_whisper_is_pinned_to_a_commit(self):
        """The pin is deliberate: master@ed9a06c carries the Silero v6.2 weights.

        Named explicitly because loosening it back to a release range is the
        exact change that would quietly restore the older VAD weights.
        """
        from whisperjav.installer.core.registry import PACKAGES, InstallSource

        fw = next((p for p in PACKAGES if p.name == "faster-whisper"), None)
        assert fw is not None, "faster-whisper is not in the registry"
        assert fw.source == InstallSource.GIT, (
            "faster-whisper must be installed from git: the PyPI release carries "
            "older Silero VAD weights and lacks the VAD parameters v1.9.2 uses"
        )
        assert re.search(r"@[0-9a-f]{40}$", fw.git_url), (
            f"faster-whisper must be pinned to a full commit hash, not a moving "
            f"branch. Current: {fw.git_url}"
        )


class TestLockedRequirements:
    """The files every install takes (item 2f): exact, complete, and current with uv.lock."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_line_is_an_exact_version_or_a_git_commit(self, variant):
        loose = [l for l in _locked_lines(variant)
                 if "==" not in l.split(";")[0] and not re.search(r" @ git\+\S+@[0-9a-f]{40}", l)]
        assert not loose, f"requirements_{variant}.txt has lines without an exact version: {loose}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_faster_whisper_is_the_commit_pyproject_pins(self, variant):
        pyproject = _pyproject_specs()["faster-whisper"]
        commit = re.search(r"@([0-9a-f]{40})", pyproject).group(1)
        lines = [l for l in _locked_lines(variant) if _name(l) == "faster-whisper"]
        assert len(lines) == 1 and lines[0].endswith(commit), (
            f"faster-whisper in requirements_{variant}.txt: {lines}; pyproject.toml pins {commit}")

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_ctranslate2_matches_the_registry(self, variant):
        """CTranslate2 4.6.2 reproduces the #125 crash on exit; the registry names the pin."""
        from whisperjav.installer.core.registry import PACKAGES

        ct2 = next(p for p in PACKAGES if p.name == "ctranslate2")
        lines = [l for l in _locked_lines(variant) if _name(l) == "ctranslate2"]
        assert lines == [ct2.pyproject_spec().split(";")[0].strip()], lines

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_torch_is_the_build_of_the_file(self, variant):
        torch = [l for l in _locked_lines(variant) if _name(l) == "torch"]
        assert any(l.startswith(f"torch==2.11.0+{variant}") for l in torch), torch

    def test_the_files_equal_a_fresh_export_of_uv_lock(self):
        """A pyproject.toml or uv.lock change without a new export is caught here."""
        import shutil
        import subprocess

        uv = shutil.which("uv")
        if not uv:
            pytest.skip("uv not on PATH")
        for variant in VARIANTS:
            r = subprocess.run(
                [uv, "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-header",
                 "--extra", "all", "--extra", variant, "--format", "requirements-txt"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, r.stderr
            fresh = [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.lstrip().startswith("#")]
            assert fresh == _locked_lines(variant), (
                f"requirements_{variant}.txt is stale: run python installer/build_release.py")
