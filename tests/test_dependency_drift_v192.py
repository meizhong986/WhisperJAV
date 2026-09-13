"""Drift guards added in v1.9.2, after the installer shipped without faster-whisper.

The Windows installer describes its dependencies in four places:

* ``pyproject.toml``                              — what pip installs from source
* ``whisperjav/installer/core/registry.py``       — the single source of truth
* ``installer/templates/post_install.py.template``— what the .exe installs, by phase
* ``installer/templates/requirements.txt.template``— the fallback list

Three of those had drifted apart by v1.9.2: faster-whisper became a commit pin
in pyproject.toml and the registry, the requirements generator silently drops
anything git-addressed, and the fallback template still asked PyPI for
``faster-whisper>=1.1.0`` — a build with different Silero VAD weights than the
one the maintainer tests against.

These tests fail when any two of those four places disagree again. The
companion test that every git-addressed dependency is actually installed by
Phase 3.5 lives in
``tests/test_installer_comprehensive.py::TestPostInstallTemplate``.
"""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10 without tomli
    tomllib = None

PROJECT_ROOT = Path(__file__).parent.parent
REQUIREMENTS_TEMPLATE = PROJECT_ROOT / "installer" / "templates" / "requirements.txt.template"


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


def _requirements_template_lines() -> list:
    """The dependency lines of the fallback template, comments stripped."""
    content = REQUIREMENTS_TEMPLATE.read_text(encoding="utf-8")
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("{{"):
            continue
        if "  #" in line:
            line = line.split("  #")[0].strip()
        lines.append(line)
    return lines


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


class TestRequirementsTemplateDrift:
    """The fallback requirements list must not fight Phase 3.5 or the registry."""

    def test_template_has_no_git_dependencies(self):
        """Git dependencies belong in Phase 3.5, never in a requirements list.

        pip resolving a git URL inside ``-r requirements.txt`` re-resolves
        dependencies and can pull a CPU torch over the CUDA one installed in
        Phase 3, which is the reason the generator strips them in the first place.
        """
        offenders = [line for line in _requirements_template_lines() if "git+" in line or " @ " in line]

        assert not offenders, (
            "The fallback requirements template lists git dependencies. They must be "
            "installed by Phase 3.5 of post_install.py.template instead:\n  "
            + "\n  ".join(offenders)
        )

    def test_template_does_not_ask_pypi_for_faster_whisper(self):
        """v1.9.2 regression guard: the template asked PyPI for faster-whisper>=1.1.0.

        That line would have installed a different build from the pinned commit
        Phase 3.5 installs, whichever ran last.
        """
        offenders = [
            line for line in _requirements_template_lines()
            if re.split(r"[>=<\[ ;]", line)[0].strip().lower() == "faster-whisper"
        ]

        assert not offenders, (
            "The fallback requirements template must not list faster-whisper: it is "
            "installed from a pinned commit in Phase 3.5. Found:\n  " + "\n  ".join(offenders)
        )

    def test_template_ctranslate2_matches_registry(self):
        """CTranslate2 4.6.2 reproduces the #125 crash on exit; 4.8.1 does not."""
        from whisperjav.installer.core.registry import PACKAGES

        ct2 = next((p for p in PACKAGES if p.name == "ctranslate2"), None)
        assert ct2 is not None, "ctranslate2 is not in the registry"

        registry_spec = ct2.pyproject_spec().split(";")[0].strip()

        template_lines = [
            line for line in _requirements_template_lines()
            if re.split(r"[>=<\[ ;]", line)[0].strip().lower() == "ctranslate2"
        ]

        assert template_lines, (
            "ctranslate2 is missing from the fallback requirements template. "
            "faster-whisper is installed there with --no-deps, so nothing else "
            f"would install it. Expected: {registry_spec}"
        )
        assert len(template_lines) == 1, f"ctranslate2 listed more than once: {template_lines}"
        assert template_lines[0] == registry_spec, (
            f"ctranslate2 pin drift: template='{template_lines[0]}' vs "
            f"registry='{registry_spec}'"
        )

class TestGeneratedRequirements:
    """The list the .exe actually installs, built from pyproject.toml.

    This is the path the real build takes. The fallback template tested above is
    used only when pyproject.toml cannot be read, so a test that only reads the
    template proves nothing about what ships. The generator is called directly
    rather than reading installer/generated/, which is gitignored and may hold a
    stale build.
    """

    @staticmethod
    def _generate() -> str:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "whisperjav_build_release", PROJECT_ROOT / "installer" / "build_release.py"
        )
        build_release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_release)

        content = build_release.ReleaseBuilder(dry_run=True).generate_requirements_from_pyproject()
        assert content, "the generator fell back to the template — pyproject.toml unreadable?"
        return content

    @staticmethod
    def _names(content: str) -> dict:
        names = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names[re.split(r"[>=<\[@ ;]", line)[0].strip().lower()] = line
        return names

    def test_the_generated_list_has_no_faster_whisper(self):
        """It is installed from a pinned commit in Phase 3.5, not from this list.

        The generator drops every git-addressed dependency, which is why Phase
        3.5 has to install it — and why it was missing from the 1.9.2 installer
        until the entry was added there.
        """
        names = self._names(self._generate())
        assert "faster-whisper" not in names, (
            f"faster-whisper must not reach the requirements file: {names.get('faster-whisper')}"
        )

    def test_the_generated_list_pins_ctranslate2(self):
        """faster-whisper is installed --no-deps, so nothing else brings it in."""
        from whisperjav.installer.core.registry import PACKAGES

        ct2 = next(p for p in PACKAGES if p.name == "ctranslate2")
        names = self._names(self._generate())

        assert "ctranslate2" in names, (
            "ctranslate2 is missing from the generated requirements file; "
            "faster-whisper would have no inference engine"
        )
        assert names["ctranslate2"] == ct2.pyproject_spec().split(";")[0].strip()

    def test_the_generated_list_contains_no_git_addresses(self):
        offenders = [
            line for line in self._generate().splitlines()
            if line.strip() and not line.strip().startswith("#")
            and ("git+" in line or " @ " in line)
        ]
        assert not offenders, (
            "pip resolving a git URL from requirements.txt can replace the CUDA "
            f"torch installed in Phase 3: {offenders}"
        )
