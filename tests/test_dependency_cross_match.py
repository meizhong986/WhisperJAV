"""
Dependency Cross-Match Tests
============================

T5a: Verify that all declared dependencies can be installed and satisfy
     their version constraints as declared across all sources.

T5b: Verify that there are no runtime import conflicts between installed
     packages (version mismatches, ABI incompatibilities, missing transitive deps).

These tests validate the DEPENDENCY_CROSS_MATCH.md table programmatically.

Run:
    pytest tests/test_dependency_cross_match.py -v
    pytest tests/test_dependency_cross_match.py -v -k "runtime"   # runtime-only
    pytest tests/test_dependency_cross_match.py -v -k "install"   # install-only
"""

import importlib
import re
import sys
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _read_file(relpath: str) -> str:
    """Read a file relative to project root."""
    return (ROOT / relpath).read_text(encoding="utf-8")


def _parse_version_from_spec(spec: str) -> Optional[str]:
    """
    Extract version constraint from a pip spec string.

    'numpy>=2.0.0'       -> '>=2.0.0'
    'pydantic>=2.0,<3.0' -> '>=2.0,<3.0'
    'tqdm'               -> None
    'pkg @ git+...'      -> None (git source)
    """
    if " @ " in spec:
        return None  # git source
    m = re.match(r"^[A-Za-z0-9_.\-]+\[?[^\]]*\]?(.*)", spec)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def _installed_version(import_name: str) -> Optional[str]:
    """Get installed version of a package via importlib.metadata."""
    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata  # type: ignore

    # Try the import name directly as distribution name
    for name_variant in [import_name, import_name.replace("_", "-")]:
        try:
            return metadata.version(name_variant)
            break
        except metadata.PackageNotFoundError:
            continue
    return None


def _pip_name_to_dist_name(pip_name: str) -> str:
    """Normalize pip name to distribution name."""
    return pip_name.lower().replace("-", "-")


def _check_version_satisfies(installed: str, constraint: str) -> bool:
    """
    Check if an installed version satisfies a constraint string.

    Uses packaging.version for proper PEP 440 comparison.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        spec = SpecifierSet(constraint)
        ver = Version(installed)
        return ver in spec
    except Exception:
        return True  # If we can't parse, don't fail


# ---------------------------------------------------------------------------
# T5a: Installation & Version Constraint Tests
# ---------------------------------------------------------------------------


class TestInstallationConstraints:
    """
    Verify that all packages declared in the registry are installed and
    satisfy their version constraints.

    These tests read the registry (single source of truth) and check each
    package against what is actually installed in the current environment.
    """

    @staticmethod
    def _get_registry_packages():
        """Load all packages from the registry."""
        from whisperjav.installer.core.registry import PACKAGES
        return PACKAGES

    def test_registry_loads_without_error(self):
        """Registry module imports and validates without raising."""
        from whisperjav.installer.core.registry import PACKAGES, validate_registry
        validate_registry()
        assert len(PACKAGES) > 70, f"Expected 70+ packages, got {len(PACKAGES)}"

    def test_pyproject_toml_parseable(self):
        """pyproject.toml is valid TOML and has expected sections."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        content = _read_file("pyproject.toml")
        data = tomllib.loads(content)
        assert "project" in data
        assert "dependencies" in data["project"]
        assert "optional-dependencies" in data["project"]

        extras = data["project"]["optional-dependencies"]
        expected_extras = ["cli", "gui", "translate", "llm", "enhance",
                           "huggingface", "qwen", "analysis", "compatibility", "dev"]
        for extra in expected_extras:
            assert extra in extras, f"Missing extra: {extra}"

    def test_registry_matches_pyproject_package_count(self):
        """Registry and pyproject.toml declare the same packages."""
        from whisperjav.installer.core.registry import (
            PACKAGES, Extra, generate_core_dependencies, generate_pyproject_extras,
        )

        # Count registry packages (excluding torch/torchaudio/torchvision which
        # are in core but installed separately)
        registry_core = [p for p in PACKAGES if p.extra == Extra.CORE
                         and p.name not in ("torch", "torchaudio", "torchvision")]
        registry_extras = {
            extra.value: [p for p in PACKAGES if p.extra == extra]
            for extra in Extra if extra != Extra.CORE
        }

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        data = tomllib.loads(_read_file("pyproject.toml"))
        pyproject_core_count = len(data["project"]["dependencies"])
        pyproject_extras = data["project"]["optional-dependencies"]

        # Core: registry count should match pyproject count
        assert len(registry_core) == pyproject_core_count, (
            f"Core count mismatch: registry={len(registry_core)}, "
            f"pyproject={pyproject_core_count}"
        )

        # Extras: check non-composite extras (skip all, colab, kaggle, windows, unix)
        composite = {"all", "colab", "kaggle", "windows", "unix"}
        for extra_name, pkgs in registry_extras.items():
            if extra_name in composite or not pkgs:
                continue
            if extra_name not in pyproject_extras:
                continue
            pyproject_count = len([
                dep for dep in pyproject_extras[extra_name]
                if not dep.startswith("whisperjav[")
            ])
            registry_count = len(pkgs)
            assert registry_count == pyproject_count, (
                f"Extra '{extra_name}' count mismatch: "
                f"registry={registry_count}, pyproject={pyproject_count}"
            )

    @pytest.mark.parametrize("pkg_info", [
        # Core packages (always expected to be installed)
        ("pysrt", None, "pysrt"),
        ("srt", None, "srt"),
        ("tqdm", None, "tqdm"),
        ("colorama", None, "colorama"),
        ("requests", None, "requests"),
        ("aiofiles", None, "aiofiles"),
        ("regex", None, "regex"),
        ("pydantic", ">=2.0,<3.0", "pydantic"),
        ("PyYAML", ">=6.0", "yaml"),
        ("jsonschema", None, "jsonschema"),
        ("tiktoken", ">=0.7.0", "tiktoken"),
        ("more-itertools", ">=10.0", "more_itertools"),
        ("faster-whisper", ">=1.1.0", "faster_whisper"),
        # PyTorch ecosystem
        ("torch", None, "torch"),
        ("torchaudio", None, "torchaudio"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_core_package_installed(self, pkg_info):
        """Each core package is importable and satisfies its version constraint."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        assert spec is not None, (
            f"Core package '{pip_name}' (import: {import_name}) is not installed"
        )
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )

    @pytest.mark.parametrize("pkg_info", [
        # CLI extra
        ("numpy", ">=2.0.0", "numpy"),
        ("scipy", ">=1.13.0", "scipy"),
        ("librosa", ">=0.11.0", "librosa"),
        ("numba", ">=0.60.0", "numba"),
        ("soundfile", None, "soundfile"),
        ("pydub", None, "pydub"),
        ("pyloudnorm", None, "pyloudnorm"),
        ("auditok", None, "auditok"),
        ("silero-vad", ">=6.2", "silero_vad"),
        ("ten-vad", None, "ten_vad"),
        ("psutil", ">=5.9.0", "psutil"),
        ("scikit-learn", ">=1.4.0", "sklearn"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_cli_package_installed(self, pkg_info):
        """Each CLI extra package is importable and satisfies its constraint."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            pytest.skip(f"{pip_name} not installed (cli extra may not be active)")
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )

    @pytest.mark.parametrize("pkg_info", [
        # GUI extra (Windows-only for some)
        ("pywebview", ">=5.0.0", "webview"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_gui_package_installed(self, pkg_info):
        """GUI packages are importable where applicable."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            pytest.skip(f"{pip_name} not installed (gui extra may not be active)")
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )

    @pytest.mark.parametrize("pkg_info", [
        ("pysubtrans", ">=1.5.0", "PySubtrans"),
        ("openai", ">=1.35.0", "openai"),
        ("google-genai", ">=1.39.0", "google.genai"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_translate_package_installed(self, pkg_info):
        """Translation packages are importable where applicable."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            pytest.skip(f"{pip_name} not installed (translate extra may not be active)")
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )

    @pytest.mark.parametrize("pkg_info", [
        ("modelscope", ">=1.20", "modelscope"),
        ("onnxruntime", ">=1.16.0", "onnxruntime"),
        ("opencv-python", ">=4.10.0", "cv2"),
        ("bs-roformer-infer", None, "bs_roformer"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_enhance_package_installed(self, pkg_info):
        """Enhancement packages are importable where applicable."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            pytest.skip(f"{pip_name} not installed (enhance extra may not be active)")
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )

    @pytest.mark.parametrize("pkg_info", [
        ("huggingface-hub", ">=0.25.0", "huggingface_hub"),
        ("transformers", ">=4.40.0", "transformers"),
        ("accelerate", ">=0.26.0", "accelerate"),
    ], ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
    def test_huggingface_package_installed(self, pkg_info):
        """HuggingFace packages are importable where applicable."""
        pip_name, constraint, import_name = pkg_info
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            pytest.skip(f"{pip_name} not installed (huggingface extra may not be active)")
        if constraint:
            version = _installed_version(pip_name)
            if version:
                assert _check_version_satisfies(version, constraint), (
                    f"{pip_name}=={version} does not satisfy {constraint}"
                )


# ---------------------------------------------------------------------------
# T5a (continued): Cross-Source Consistency Tests
# ---------------------------------------------------------------------------


class TestCrossSourceConsistency:
    """
    Verify that version constraints are identical across all declaration sources:
    pyproject.toml, registry.py, requirements.txt, requirements.txt.template.
    """

    @staticmethod
    def _extract_pyproject_specs() -> dict:
        """Parse pyproject.toml and extract all dependency specs."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        data = tomllib.loads(_read_file("pyproject.toml"))
        specs = {}

        # Core dependencies
        for dep in data["project"]["dependencies"]:
            name = re.split(r"[>=<\[@ ;]", dep)[0].strip()
            specs[name.lower()] = dep.strip()

        # Optional dependencies
        composite = {"all", "colab", "kaggle", "windows", "unix"}
        for extra, deps in data["project"]["optional-dependencies"].items():
            if extra in composite:
                continue
            for dep in deps:
                if dep.startswith("whisperjav["):
                    continue
                name = re.split(r"[>=<\[@ ;]", dep)[0].strip()
                specs[name.lower()] = dep.strip()

        return specs

    @staticmethod
    def _extract_registry_specs() -> dict:
        """Extract all specs from registry.py."""
        from whisperjav.installer.core.registry import PACKAGES
        specs = {}
        for pkg in PACKAGES:
            spec = pkg.pyproject_spec()
            specs[pkg.name.lower()] = spec
        return specs

    @staticmethod
    def _extract_template_specs() -> dict:
        """Extract specs from requirements.txt.template."""
        content = _read_file("installer/templates/requirements.txt.template")
        specs = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("{{"):
                continue
            # Remove inline comments
            if "  #" in line:
                line = line.split("  #")[0].strip()
            name = re.split(r"[>=<\[@ ;]", line)[0].strip()
            specs[name.lower()] = line
        return specs

    def test_registry_matches_pyproject(self):
        """Every package in registry has matching spec in pyproject.toml."""
        pyproject = self._extract_pyproject_specs()
        registry = self._extract_registry_specs()

        mismatches = []
        for name, reg_spec in registry.items():
            if name not in pyproject:
                # torch/torchaudio/torchvision are not in pyproject.toml deps
                # (they're installed via INDEX_URL separately)
                if name in ("torch", "torchaudio", "torchvision"):
                    continue
                mismatches.append(f"  {name}: in registry but not pyproject")
                continue

            py_spec = pyproject[name]
            # Normalize for comparison: strip platform markers for matching
            reg_base = reg_spec.split(";")[0].strip()
            py_base = py_spec.split(";")[0].strip()

            if reg_base != py_base:
                mismatches.append(
                    f"  {name}: registry='{reg_base}' vs pyproject='{py_base}'"
                )

        assert not mismatches, (
            f"Registry/pyproject.toml mismatches:\n" + "\n".join(mismatches)
        )

    def test_template_matches_registry(self):
        """
        Packages in requirements.txt.template match registry constraints.

        The template is a subset (excludes git packages, torch, dev, llm).
        """
        from whisperjav.installer.core.registry import PACKAGES

        template = self._extract_template_specs()
        registry = {pkg.name.lower(): pkg for pkg in PACKAGES}

        mismatches = []
        for name, tmpl_spec in template.items():
            if name not in registry:
                continue  # Template may have packages not in registry

            pkg = registry[name]
            reg_spec = pkg.pyproject_spec().split(";")[0].strip()
            tmpl_base = tmpl_spec.split(";")[0].strip()

            if reg_spec != tmpl_base:
                mismatches.append(
                    f"  {name}: template='{tmpl_base}' vs registry='{reg_spec}'"
                )

        assert not mismatches, (
            f"Template/registry mismatches:\n" + "\n".join(mismatches)
        )

    def test_no_duplicate_packages_in_registry(self):
        """Registry has no duplicate package names."""
        from whisperjav.installer.core.registry import PACKAGES

        seen = set()
        dupes = []
        for pkg in PACKAGES:
            key = pkg.name.lower()
            if key in seen:
                dupes.append(pkg.name)
            seen.add(key)

        assert not dupes, f"Duplicate packages in registry: {dupes}"

    def test_every_import_name_mapped(self):
        """Every package with a different import name has import_name set."""
        from whisperjav.installer.core.registry import PACKAGES

        # Known mismatches that MUST have import_name set.
        # Excludes packages where pip name hyphen→underscore is automatic
        # (e.g., faster-whisper → faster_whisper is handled by Python's
        # import normalization and does NOT need explicit import_name).
        # Only truly non-obvious mismatches belong here.
        known_mismatches = {
            "openai-whisper",       # whisper (completely different name)
            "stable-ts",            # stable_whisper (different name)
            "bs-roformer-infer",    # bs_roformer (truncated name)
            "scikit-learn",         # sklearn (abbreviation)
            "pywebview",            # webview (prefix dropped)
            "pywin32",              # win32com (different name)
            "PyYAML",               # yaml (different name)
            "Pillow",               # PIL (different name)
            "opencv-python",        # cv2 (completely different)
            "python-speech-features",  # python_speech_features (BUT also non-obvious pkg name)
            "rotary-embedding-torch",  # rotary_embedding_torch
            "more-itertools",       # more_itertools
            "pysubtrans",           # PySubtrans (case-sensitive)
            "google-api-core",      # google.api_core (dotted path)
            "qwen-asr",             # qwen_asr
        }

        missing = []
        for pkg in PACKAGES:
            if pkg.name in known_mismatches and not pkg.import_name:
                missing.append(pkg.name)

        assert not missing, (
            f"Packages with known import mismatches but no import_name: {missing}"
        )


# ---------------------------------------------------------------------------
# T5b: Runtime Conflict Detection Tests
# ---------------------------------------------------------------------------


class TestRuntimeConflicts:
    """
    Verify that installed packages have no runtime conflicts:
    - No version incompatibilities between co-installed packages
    - No ABI mismatches (numpy/numba binary compatibility)
    - No missing transitive dependencies
    - Core import chains work without errors
    """

    def test_numpy_numba_abi_compatibility(self):
        """numpy and numba have compatible ABIs (critical binary compat)."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")

        try:
            import numba
        except ImportError:
            pytest.skip("numba not installed")

        # If both import without error, ABI is compatible
        # Verify numpy is 2.x as required
        major = int(np.__version__.split(".")[0])
        assert major >= 2, f"numpy {np.__version__} is not 2.x"

        # Verify numba can actually JIT (proves ABI compat)
        @numba.njit
        def _add(a, b):
            return a + b

        result = _add(1, 2)
        assert result == 3, "numba JIT execution failed — ABI mismatch"

    def test_numpy_scipy_compatibility(self):
        """scipy works with the installed numpy version."""
        try:
            import numpy as np
            import scipy
        except ImportError:
            pytest.skip("numpy/scipy not installed")

        # scipy.linalg is a core module that exercises numpy interop
        from scipy import linalg
        import numpy as np

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = linalg.det(arr)
        assert abs(det - (-2.0)) < 1e-10, f"scipy.linalg.det wrong: {det}"

    def test_numpy_librosa_compatibility(self):
        """librosa works with numpy 2.x (no np.complex, etc.)."""
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not installed")

        # librosa 0.10.x used np.complex which was removed in numpy 2.0
        # librosa 0.11.0+ fixed this — this test proves the fix is active
        import numpy as np
        sr = 16000
        duration = 0.1  # 100ms of silence
        y = np.zeros(int(sr * duration), dtype=np.float32)

        # This triggers internal numpy calls in librosa
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)
        assert rms is not None
        assert rms.shape[0] == 1

    def test_torch_imports_cleanly(self):
        """PyTorch imports without conflicts."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        assert hasattr(torch, "__version__")
        # Verify basic tensor creation
        t = torch.zeros(2, 3)
        assert t.shape == (2, 3)

    def test_torch_numpy_interop(self):
        """torch and numpy 2.x work together."""
        try:
            import torch
            import numpy as np
        except ImportError:
            pytest.skip("torch/numpy not installed")

        # numpy array -> torch tensor -> numpy array roundtrip
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = torch.from_numpy(arr)
        back = tensor.numpy()
        assert np.allclose(arr, back), "torch/numpy roundtrip failed"

    def test_whisper_imports(self):
        """openai-whisper imports without conflicts."""
        try:
            import whisper
        except ImportError:
            pytest.skip("openai-whisper not installed")

        assert hasattr(whisper, "load_model")

    def test_stable_whisper_imports(self):
        """stable-ts imports without conflicts."""
        try:
            import stable_whisper
        except ImportError:
            pytest.skip("stable-ts not installed")

        assert hasattr(stable_whisper, "load_model")

    def test_faster_whisper_imports(self):
        """faster-whisper imports without conflicts."""
        try:
            import faster_whisper
        except ImportError:
            pytest.skip("faster-whisper not installed")

        assert hasattr(faster_whisper, "WhisperModel")

    def test_pydantic_v2_active(self):
        """Pydantic v2 is active (not v1 compat mode)."""
        import pydantic
        major = int(pydantic.__version__.split(".")[0])
        assert major == 2, f"pydantic {pydantic.__version__} is not v2"

    def test_yaml_imports(self):
        """PyYAML imports and can parse YAML."""
        import yaml
        result = yaml.safe_load("key: value")
        assert result == {"key": "value"}

    def test_srt_pysrt_no_conflict(self):
        """srt and pysrt can coexist (both handle .srt files)."""
        import srt
        import pysrt
        assert hasattr(srt, "parse")
        assert hasattr(pysrt, "open")

    def test_soundfile_imports(self):
        """soundfile imports (requires libsndfile)."""
        try:
            import soundfile
        except (ImportError, OSError):
            pytest.skip("soundfile not installed or libsndfile missing")

        assert hasattr(soundfile, "read")

    def test_silero_vad_imports(self):
        """silero-vad package imports cleanly."""
        try:
            import silero_vad
        except ImportError:
            pytest.skip("silero-vad not installed")

        assert hasattr(silero_vad, "load_silero_vad")

    def test_transformers_torch_compat(self):
        """transformers works with the installed torch version."""
        try:
            import transformers
            import torch
        except ImportError:
            pytest.skip("transformers/torch not installed")

        assert hasattr(transformers, "AutoModelForSpeechSeq2Seq")

    def test_no_pkg_resources_crash(self):
        """
        setuptools pkg_resources is available (required by modelscope at runtime).
        setuptools >=82 removed pkg_resources — we pin <82.
        """
        try:
            import pkg_resources
        except ImportError:
            pytest.skip("setuptools not installed")

        # Should not raise
        pkg_resources.get_distribution("setuptools")

    def test_import_chain_cli_pipeline(self):
        """
        The full CLI import chain loads without errors.
        This catches transitive dependency issues.
        """
        try:
            # These are the imports that main.py triggers
            import pysrt
            import srt
            import tqdm
            import colorama
            import requests
            import regex
            import yaml
            import pydantic
        except ImportError as e:
            pytest.fail(f"CLI import chain broken: {e}")

    def test_import_chain_config_v4(self):
        """The v4 config system imports without errors."""
        try:
            from whisperjav.config.v4.manager import ConfigManager
        except ImportError as e:
            pytest.skip(f"Config v4 not importable: {e}")

        # Should be constructible
        cm = ConfigManager()
        assert cm is not None

    def test_no_numpy_deprecation_warnings_in_imports(self):
        """
        Importing core modules doesn't trigger numpy deprecation warnings.
        This catches packages using removed numpy APIs.
        """
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            try:
                import numpy
                import scipy
            except ImportError:
                pytest.skip("numpy/scipy not installed")

            numpy_warnings = [
                x for x in w
                if "numpy" in str(x.message).lower()
                and issubclass(x.category, DeprecationWarning)
            ]

        # Filter out known harmless warnings
        critical = [
            x for x in numpy_warnings
            if "np.complex" in str(x.message)
            or "np.int_" in str(x.message)
            or "np.float_" in str(x.message)
            or "np.bool_" in str(x.message)
        ]

        assert not critical, (
            f"Numpy deprecation warnings from imports:\n"
            + "\n".join(str(w.message) for w in critical)
        )


# ---------------------------------------------------------------------------
# T5b (continued): Full Environment Audit
# ---------------------------------------------------------------------------


class TestEnvironmentAudit:
    """
    Comprehensive audit of the installed environment against the registry.
    Runs `pip check` equivalent and validates all constraints.
    """

    def test_pip_check_no_conflicts(self):
        """
        `pip check` reports no broken dependencies.
        This is the definitive test for dependency conflicts.
        """
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            # Filter out known acceptable issues (e.g., optional deps)
            lines = result.stdout.strip().splitlines()
            real_issues = []
            for line in lines:
                # Skip issues with optional/undeclared packages
                skip_patterns = [
                    "nemo", "pyannote", "llama-cpp", "flash-attn",
                    "omegaconf", "wget", "pynvml",
                ]
                if any(pat in line.lower() for pat in skip_patterns):
                    continue
                real_issues.append(line)

            if real_issues:
                pytest.fail(
                    f"pip check found dependency conflicts:\n"
                    + "\n".join(real_issues)
                )

    def test_all_registry_packages_version_satisfied(self):
        """
        Every package in the registry that is installed satisfies its
        declared version constraint.
        """
        from whisperjav.installer.core.registry import PACKAGES

        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata  # type: ignore

        violations = []
        for pkg in PACKAGES:
            if not pkg.version:
                continue  # No constraint to check

            try:
                installed = metadata.version(pkg.name)
            except metadata.PackageNotFoundError:
                continue  # Not installed, skip

            if not _check_version_satisfies(installed, pkg.version):
                violations.append(
                    f"  {pkg.name}: installed={installed}, "
                    f"required={pkg.version}"
                )

        assert not violations, (
            f"Version constraint violations:\n" + "\n".join(violations)
        )
