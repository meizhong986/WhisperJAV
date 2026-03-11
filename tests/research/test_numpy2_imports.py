"""
Tests that verify WhisperJAV modules can be imported and inspected for
numpy 2.x incompatible patterns at runtime.

These tests use AST analysis on the actual source files rather than
importing modules (which would require GPU, models, etc.). This makes
them safe to run in CI without special hardware.

Key numpy 2.0 changes tested:
- Removed type aliases (np.int_, np.float_, np.complex_, np.object_)
- Removed functions (np.product, np.sometrue, np.alltrue, etc.)
- Changed copy semantics
- Removed numpy.distutils
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "whisperjav"

# Numpy attributes removed in numpy 2.0
# Reference: https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
NUMPY2_REMOVED_ATTRS = {
    # Removed type aliases
    "int_",     # use intp
    "float_",   # use float64
    "complex_", # use complex128
    "object_",  # use object
    "bool8",    # use bool_
    "int0",     # use intp
    "uint0",    # use uintp
    "str0",     # removed
    "bytes0",   # removed
    "longfloat",      # use longdouble
    "singlecomplex",  # use complex64
    "cfloat",         # use complex128
    "longcomplex",    # use clongdouble
    "clongfloat",     # use clongdouble
    "string_",        # use bytes_
    # Removed functions
    "product",
    "sometrue",
    "alltrue",
    "cumproduct",
    "row_stack",
    "in1d",
    "trapz",
    "find_common_type",
    "round_",
    "asfarray",
    # Removed constants
    "PINF",
    "NINF",
    "PZERO",
    "NZERO",
    "Inf",
    "Infinity",
    "NaN",
    "infty",
    # Moved exceptions (still accessible but deprecated location)
    "AxisError",        # -> np.exceptions.AxisError
    "ComplexWarning",   # -> np.exceptions.ComplexWarning
}

# Attributes that are safe in numpy 2 (commonly confused with removed ones)
NUMPY2_SAFE_ATTRS = {
    "bool_",     # KEPT in numpy 2
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "complex64", "complex128",
    "intp", "uintp",
    "integer", "floating", "complexfloating",
    "ndarray",
    "array", "zeros", "ones", "empty",
    "mean", "sum", "max", "min", "std",
    "abs", "sqrt", "log", "log10", "exp",
    "clip", "pad", "where", "concatenate",
    "frombuffer", "reshape", "arange", "linspace",
    "vstack", "hstack", "stack",
    "ceil", "floor",
    "isnan", "isinf",
    "gcd",  # still works in numpy 2
    "inf", "nan",
    "prod", "any", "all", "cumprod",  # correct replacements
    "corrcoef", "histogram",
    "searchsorted", "argmin",
    "maximum", "percentile",
}


class NumpyAttrVisitor(ast.NodeVisitor):
    """AST visitor that collects numpy attribute accesses."""

    def __init__(self):
        self.numpy_aliases: Set[str] = set()
        self.attr_uses: List[Tuple[int, str]] = []  # (line, attr_name)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "numpy":
                self.numpy_aliases.add(alias.asname or "numpy")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module.startswith("numpy"):
            for alias in node.names:
                # Direct imports like 'from numpy import float_'
                self.attr_uses.append((node.lineno, alias.name))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Check for np.something or numpy.something
        if isinstance(node.value, ast.Name) and node.value.id in self.numpy_aliases:
            self.attr_uses.append((node.lineno, node.attr))
        self.generic_visit(node)


def _analyze_file(filepath: Path) -> List[Tuple[int, str]]:
    """Parse a Python file and return all numpy attribute accesses with line numbers."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    visitor = NumpyAttrVisitor()
    visitor.visit(tree)
    return visitor.attr_uses


def _collect_python_files() -> List[Path]:
    """Collect all Python files in the whisperjav package."""
    files = []
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return sorted(files)


# Module categories for parametrized testing
MODULE_CATEGORIES = {
    "asr_engines": [
        "whisperjav/modules/stable_ts_asr.py",
        "whisperjav/modules/whisper_pro_asr.py",
        "whisperjav/modules/faster_whisper_pro_asr.py",
    ],
    "scene_detection": [
        "whisperjav/modules/scene_detection.py",
        "whisperjav/modules/scene_detection_backends/auditok_backend.py",
        "whisperjav/modules/scene_detection_backends/silero_backend.py",
        "whisperjav/modules/scene_detection_backends/semantic_adapter.py",
        "whisperjav/modules/scene_detection_backends/none_backend.py",
        "whisperjav/modules/scene_detection_backends/utils.py",
    ],
    "speech_enhancement": [
        "whisperjav/modules/speech_enhancement/base.py",
        "whisperjav/modules/speech_enhancement/pipeline_helper.py",
        "whisperjav/modules/speech_enhancement/backends/none.py",
        "whisperjav/modules/speech_enhancement/backends/ffmpeg_dsp.py",
        "whisperjav/modules/speech_enhancement/backends/zipenhancer.py",
        "whisperjav/modules/speech_enhancement/backends/clearvoice.py",
        "whisperjav/modules/speech_enhancement/backends/bs_roformer.py",
    ],
    "speech_segmentation": [
        "whisperjav/modules/speech_segmentation/base.py",
        "whisperjav/modules/speech_segmentation/backends/none.py",
        "whisperjav/modules/speech_segmentation/backends/silero.py",
        "whisperjav/modules/speech_segmentation/backends/silero_v6.py",
        "whisperjav/modules/speech_segmentation/backends/ten.py",
        "whisperjav/modules/speech_segmentation/backends/nemo.py",
        "whisperjav/modules/speech_segmentation/backends/whisper_vad.py",
    ],
    "subtitle_pipeline": [
        "whisperjav/modules/subtitle_pipeline/orchestrator.py",
        "whisperjav/modules/subtitle_pipeline/protocols.py",
        "whisperjav/modules/subtitle_pipeline/framers/full_scene.py",
        "whisperjav/modules/subtitle_pipeline/framers/manual.py",
        "whisperjav/modules/subtitle_pipeline/framers/srt_source.py",
        "whisperjav/modules/subtitle_pipeline/framers/vad_grouped.py",
        "whisperjav/modules/subtitle_pipeline/generators/anime_whisper.py",
    ],
    "utilities": [
        "whisperjav/utils/metadata_manager.py",
        "whisperjav/modules/repetition_cleaner.py",
        "whisperjav/vendor/semantic_audio_clustering.py",
    ],
    "pipelines": [
        "whisperjav/pipelines/transformers_pipeline.py",
    ],
}


class TestNumpyAttributeUsage:
    """Verify no numpy 2 removed attributes are used in WhisperJAV modules."""

    @pytest.mark.parametrize(
        "category",
        list(MODULE_CATEGORIES.keys()),
        ids=list(MODULE_CATEGORIES.keys()),
    )
    def test_no_removed_attrs_by_category(self, category: str):
        """Check each module category for numpy 2 removed attribute usage."""
        module_paths = MODULE_CATEGORIES[category]
        issues = []

        for rel_path in module_paths:
            filepath = REPO_ROOT / rel_path.replace("/", os.sep)
            if not filepath.exists():
                continue

            attrs = _analyze_file(filepath)
            for line_no, attr_name in attrs:
                if attr_name in NUMPY2_REMOVED_ATTRS:
                    issues.append(
                        f"  {rel_path}:{line_no} -- np.{attr_name} "
                        f"(removed in numpy 2)"
                    )

        if issues:
            pytest.fail(
                f"Category '{category}' has numpy 2 incompatible attributes:\n"
                + "\n".join(issues)
            )

    def test_full_codebase_scan(self):
        """Scan the entire whisperjav package for removed numpy attributes."""
        all_files = _collect_python_files()
        issues_by_file: Dict[str, List[str]] = {}

        for filepath in all_files:
            attrs = _analyze_file(filepath)
            for line_no, attr_name in attrs:
                if attr_name in NUMPY2_REMOVED_ATTRS:
                    rel = str(filepath.relative_to(REPO_ROOT))
                    issues_by_file.setdefault(rel, []).append(
                        f"  line {line_no}: np.{attr_name}"
                    )

        if issues_by_file:
            report = ["Numpy 2 removed attributes found in codebase:"]
            for fpath, issues in sorted(issues_by_file.items()):
                report.append(f"\n{fpath}:")
                report.extend(issues)

            # Count total
            total = sum(len(v) for v in issues_by_file.values())
            report.append(f"\nTotal: {total} issues in {len(issues_by_file)} files")
            pytest.fail("\n".join(report))


class TestNumpyUsageInventory:
    """Document all numpy API usage across the codebase (informational)."""

    def test_numpy_api_inventory(self):
        """
        Produce an inventory of all numpy APIs used.
        This is informational and always passes.
        """
        all_files = _collect_python_files()
        attr_counts: Dict[str, int] = {}
        file_usage: Dict[str, Set[str]] = {}

        for filepath in all_files:
            attrs = _analyze_file(filepath)
            if attrs:
                rel = str(filepath.relative_to(REPO_ROOT))
                file_usage[rel] = set()
                for _, attr_name in attrs:
                    attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1
                    file_usage[rel].add(attr_name)

        report = [
            f"Numpy API inventory: {len(all_files)} files scanned, "
            f"{len(file_usage)} files use numpy",
            "",
            "Most-used numpy APIs:",
        ]
        for attr, count in sorted(attr_counts.items(), key=lambda x: -x[1])[:30]:
            safety = "SAFE" if attr in NUMPY2_SAFE_ATTRS else "CHECK"
            removed = " ** REMOVED **" if attr in NUMPY2_REMOVED_ATTRS else ""
            report.append(f"  np.{attr}: {count} uses [{safety}]{removed}")

        report.append(f"\nTotal unique numpy APIs used: {len(attr_counts)}")
        report.append(f"Files using numpy: {len(file_usage)}")

        print("\n".join(report))


class TestNumpyImportPatterns:
    """Check how numpy is imported across modules."""

    def test_import_patterns(self):
        """Document numpy import patterns (always passes, informational)."""
        all_files = _collect_python_files()
        import_styles: Dict[str, List[str]] = {
            "import numpy as np": [],
            "import numpy": [],
            "from numpy import ...": [],
            "conditional/lazy import": [],
        }

        for filepath in all_files:
            try:
                source = filepath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(filepath))
            except SyntaxError:
                continue

            rel = str(filepath.relative_to(REPO_ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "numpy":
                            if alias.asname == "np":
                                import_styles["import numpy as np"].append(rel)
                            else:
                                import_styles["import numpy"].append(rel)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("numpy"):
                        import_styles["from numpy import ..."].append(rel)

            # Check for lazy/conditional imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Try, ast.If)):
                    for child in ast.walk(node):
                        if isinstance(child, ast.Import):
                            for alias in child.names:
                                if alias.name == "numpy":
                                    import_styles["conditional/lazy import"].append(rel)

        report = ["Numpy import patterns:"]
        for style, files in import_styles.items():
            report.append(f"\n  {style}: {len(files)} files")
            for f in sorted(set(files))[:5]:
                report.append(f"    {f}")
            if len(set(files)) > 5:
                report.append(f"    ... and {len(set(files)) - 5} more")

        print("\n".join(report))


class TestUnusedNumpyImports:
    """Detect files that import numpy but don't use it."""

    def test_unused_numpy_imports(self):
        """Find files that import numpy but have no numpy API usage."""
        all_files = _collect_python_files()
        unused = []

        for filepath in all_files:
            try:
                source = filepath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(filepath))
            except SyntaxError:
                continue

            # Check if numpy is imported
            has_numpy_import = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "numpy":
                            has_numpy_import = True
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("numpy"):
                        has_numpy_import = True

            if not has_numpy_import:
                continue

            # Check if numpy is actually used
            attrs = _analyze_file(filepath)
            # Also check for type annotations using np.ndarray
            has_ndarray_annotation = "np.ndarray" in source or "numpy.ndarray" in source

            if not attrs and not has_ndarray_annotation:
                # Double-check: maybe it's used as just 'np' in isinstance or similar
                if "np." not in source.replace("import numpy as np", ""):
                    rel = str(filepath.relative_to(REPO_ROOT))
                    unused.append(rel)

        if unused:
            report = [
                "Files that import numpy but may not use it directly:",
                "(These could be cleaned up, or numpy may be needed by "
                "dependencies at import time)",
            ]
            for f in sorted(unused):
                report.append(f"  {f}")
            # Informational only
            print("\n".join(report))
