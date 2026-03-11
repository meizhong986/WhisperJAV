"""
Static analysis tests for numpy 2.x incompatible patterns in WhisperJAV codebase.

These tests scan the source code for known numpy 2 breaking changes without
importing any WhisperJAV modules. They work purely via AST/regex analysis
and can run on any machine with the source code.

Numpy 2.0 Migration Guide:
https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Root of the whisperjav package
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "whisperjav"


def _collect_python_files(root: Path) -> List[Path]:
    """Collect all .py files under root, excluding __pycache__ and .pyc."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return sorted(files)


PYTHON_FILES = _collect_python_files(PACKAGE_ROOT)


# ---------------------------------------------------------------------------
# Numpy 2 removed type aliases
# ---------------------------------------------------------------------------
# In numpy 2.0, these aliases were removed from the top-level namespace:
#   np.bool_  -> KEPT (still exists)
#   np.int_   -> REMOVED (use np.intp)
#   np.float_ -> REMOVED (use np.float64)
#   np.complex_ -> REMOVED (use np.complex128)
#   np.object_ -> REMOVED (use object)
#   np.str_   -> REMOVED (use np.str_... actually kept, but np.string_ -> np.bytes_)
#
# Also removed: np.bool8, np.int0, np.uint0, np.str0, np.bytes0,
#   np.longfloat, np.singlecomplex, np.cfloat, np.longcomplex, np.clongfloat

REMOVED_TYPE_ALIASES = {
    # pattern -> (replacement, severity)
    r"\bnp\.int_\b": ("np.intp", "HIGH"),
    r"\bnp\.float_\b": ("np.float64", "HIGH"),
    r"\bnp\.complex_\b": ("np.complex128", "MEDIUM"),
    r"\bnp\.object_\b": ("object", "MEDIUM"),
    r"\bnp\.bool8\b": ("np.bool_", "MEDIUM"),
    r"\bnp\.int0\b": ("np.intp", "LOW"),
    r"\bnp\.uint0\b": ("np.uintp", "LOW"),
    r"\bnp\.longfloat\b": ("np.longdouble", "LOW"),
    r"\bnp\.singlecomplex\b": ("np.complex64", "LOW"),
    r"\bnp\.cfloat\b": ("np.complex128", "LOW"),
    r"\bnp\.longcomplex\b": ("np.clongdouble", "LOW"),
    r"\bnp\.clongfloat\b": ("np.clongdouble", "LOW"),
    r"\bnp\.string_\b": ("np.bytes_", "MEDIUM"),
    r"\bnp\.unicode_\b": ("np.str_", "LOW"),
}


# ---------------------------------------------------------------------------
# Numpy 2 removed functions
# ---------------------------------------------------------------------------
REMOVED_FUNCTIONS = {
    r"\bnp\.product\b": ("np.prod", "HIGH"),
    r"\bnp\.sometrue\b": ("np.any", "HIGH"),
    r"\bnp\.alltrue\b": ("np.all", "HIGH"),
    r"\bnp\.cumproduct\b": ("np.cumprod", "HIGH"),
    r"\bnp\.row_stack\b": ("np.vstack", "MEDIUM"),
    r"\bnp\.in1d\b": ("np.isin", "MEDIUM"),
    r"\bnp\.trapz\b": ("np.trapezoid", "MEDIUM"),
    r"\bnp\.find_common_type\b": ("np.result_type or np.promote_types", "MEDIUM"),
    r"\bnp\.round_\b": ("np.round", "LOW"),
    r"\bnp\.asfarray\b": ("np.asarray with dtype=float", "MEDIUM"),
}


# ---------------------------------------------------------------------------
# Numpy 2 moved exceptions
# ---------------------------------------------------------------------------
MOVED_EXCEPTIONS = {
    r"\bnp\.AxisError\b": ("np.exceptions.AxisError", "HIGH"),
    r"\bnp\.ComplexWarning\b": ("np.exceptions.ComplexWarning", "MEDIUM"),
    r"\bnp\.DTypePromotionError\b": ("np.exceptions.DTypePromotionError", "MEDIUM"),
    r"\bnp\.VisibleDeprecationWarning\b": ("np.exceptions.VisibleDeprecationWarning", "LOW"),
    r"\bnp\.RankWarning\b": ("np.exceptions.RankWarning", "LOW"),
}


# ---------------------------------------------------------------------------
# Numpy 2 removed constants
# ---------------------------------------------------------------------------
REMOVED_CONSTANTS = {
    r"\bnp\.PINF\b": ("np.inf", "HIGH"),
    r"\bnp\.NINF\b": ("-np.inf", "HIGH"),
    r"\bnp\.PZERO\b": ("0.0", "LOW"),
    r"\bnp\.NZERO\b": ("-0.0", "LOW"),
    r"\bnp\.Inf\b": ("np.inf", "MEDIUM"),
    r"\bnp\.Infinity\b": ("np.inf", "LOW"),
    r"\bnp\.NaN\b": ("np.nan", "MEDIUM"),
    r"\bnp\.infty\b": ("np.inf", "LOW"),
}


# ---------------------------------------------------------------------------
# numpy.distutils removal
# ---------------------------------------------------------------------------
DISTUTILS_PATTERN = {
    r"numpy\.distutils": ("setuptools / meson-python", "HIGH"),
    r"from numpy\.distutils": ("setuptools / meson-python", "HIGH"),
    r"import numpy\.distutils": ("setuptools / meson-python", "HIGH"),
}


def _scan_files_for_patterns(
    patterns: Dict[str, Tuple[str, str]],
) -> List[Tuple[str, int, str, str, str]]:
    """
    Scan all Python files for regex patterns.

    Returns list of (file_path_relative, line_number, matched_text, replacement, severity).
    """
    findings = []
    compiled = {re.compile(p): (repl, sev) for p, (repl, sev) in patterns.items()}

    for fpath in PYTHON_FILES:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), start=1):
            # Skip comments
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            for regex_pat, (replacement, severity) in compiled.items():
                match = regex_pat.search(line)
                if match:
                    rel = fpath.relative_to(REPO_ROOT)
                    findings.append(
                        (str(rel), i, match.group(), replacement, severity)
                    )
    return findings


# ===========================================================================
# Tests
# ===========================================================================


class TestRemovedTypeAliases:
    """Check for numpy 2 removed type aliases (np.float_, np.int_, etc.)."""

    def test_no_removed_type_aliases(self):
        """Scan for removed numpy type aliases that will raise AttributeError in numpy 2."""
        findings = _scan_files_for_patterns(REMOVED_TYPE_ALIASES)
        if findings:
            report = ["Found numpy 2 removed type aliases:"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> use {replacement}"
                )
            # This test documents findings rather than failing hard,
            # since some patterns (like np.bool_) are actually kept in numpy 2.
            # np.int_ is the main concern.
            high_severity = [f for f in findings if f[4] == "HIGH"]
            if high_severity:
                pytest.fail(
                    "\n".join(report)
                    + f"\n\n{len(high_severity)} HIGH severity issues found."
                )
            else:
                pytest.skip(
                    "\n".join(report)
                    + "\n\nOnly LOW/MEDIUM severity issues found (informational)."
                )


class TestRemovedFunctions:
    """Check for numpy 2 removed functions (np.product, np.sometrue, etc.)."""

    def test_no_removed_functions(self):
        """Scan for removed numpy functions that will raise AttributeError in numpy 2."""
        findings = _scan_files_for_patterns(REMOVED_FUNCTIONS)
        if findings:
            report = ["Found numpy 2 removed functions:"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> use {replacement}"
                )
            pytest.fail("\n".join(report))


class TestMovedExceptions:
    """Check for numpy 2 moved exception classes."""

    def test_no_moved_exceptions(self):
        """Scan for exception classes moved to np.exceptions in numpy 2."""
        findings = _scan_files_for_patterns(MOVED_EXCEPTIONS)
        if findings:
            report = ["Found numpy 2 moved exception references:"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> use {replacement}"
                )
            pytest.fail("\n".join(report))


class TestRemovedConstants:
    """Check for numpy 2 removed constants (np.PINF, np.NINF, etc.)."""

    def test_no_removed_constants(self):
        """Scan for removed numpy constants that will raise AttributeError in numpy 2."""
        findings = _scan_files_for_patterns(REMOVED_CONSTANTS)
        if findings:
            report = ["Found numpy 2 removed constants:"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> use {replacement}"
                )
            pytest.fail("\n".join(report))


class TestDistutilsRemoval:
    """Check for numpy.distutils usage (removed in numpy 2)."""

    def test_no_numpy_distutils(self):
        """Scan for numpy.distutils imports (removed in numpy 2, replaced by meson)."""
        findings = _scan_files_for_patterns(DISTUTILS_PATTERN)
        if findings:
            report = ["Found numpy.distutils usage (removed in numpy 2):"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> use {replacement}"
                )
            pytest.fail("\n".join(report))


class TestCopySemanticChanges:
    """Check for np.array(..., copy=False) which changed behavior in numpy 2."""

    def test_no_copy_false_pattern(self):
        """
        In numpy 2, np.array(obj, copy=False) raises ValueError if a copy
        is needed. The new behavior requires copy=None for 'copy if needed'.
        """
        pattern = {r"np\.array\([^)]*copy\s*=\s*False": ("copy=None", "HIGH")}
        findings = _scan_files_for_patterns(pattern)
        if findings:
            report = [
                "Found np.array(..., copy=False) patterns (behavior changed in numpy 2):",
                "  In numpy 2, copy=False raises ValueError if a copy is actually needed.",
                "  Use copy=None for 'copy only if needed' behavior.",
            ]
            for fpath, line, matched, replacement, severity in findings:
                report.append(f"  [{severity}] {fpath}:{line} -- {matched}")
            pytest.fail("\n".join(report))


class TestNumpyIntegerOverflow:
    """Check for patterns that may be affected by numpy 2 integer overflow changes."""

    def test_document_integer_arithmetic(self):
        """
        In numpy 2, integer overflow behavior changed. Operations on numpy
        integers that overflow now raise an error instead of wrapping.
        This test documents places where integer arithmetic on numpy types occurs.
        """
        # This is informational -- we scan for .astype(np.int16) which involves
        # potential overflow from float->int conversion (but clip is usually used).
        pattern = {r"\.astype\(np\.int16\)": ("potential overflow site", "INFO")}
        findings = _scan_files_for_patterns(pattern)
        # Don't fail -- just report as informational
        if findings:
            report = [
                f"Found {len(findings)} float-to-int16 conversion sites "
                f"(potential integer overflow in numpy 2 if values exceed int16 range):"
            ]
            for fpath, line, matched, replacement, severity in findings:
                report.append(f"  {fpath}:{line}")
            # Check that clip is used before astype(np.int16)
            unprotected = []
            for fpath_str, line_no, _, _, _ in findings:
                fpath = REPO_ROOT / fpath_str
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                context_start = max(0, line_no - 3)
                context = "\n".join(lines[context_start:line_no])
                if "clip" not in context and "* 32767" in context:
                    unprotected.append(f"  {fpath_str}:{line_no} -- no np.clip before int16 cast")

            if unprotected:
                report.append("\nUnprotected conversions (no np.clip before cast):")
                report.extend(unprotected)

            pytest.skip("\n".join(report))


class TestNumpyGCDUsage:
    """Check for np.gcd usage which works on Python ints but changed for numpy scalar types."""

    def test_gcd_usage(self):
        """np.gcd with Python int arguments is fine in numpy 2. Just document usage."""
        pattern = {r"\bnp\.gcd\b": ("still works, verify scalar types", "INFO")}
        findings = _scan_files_for_patterns(pattern)
        if findings:
            report = ["Found np.gcd usage (works in numpy 2, but verify scalar types):"]
            for fpath, line, matched, replacement, severity in findings:
                report.append(f"  {fpath}:{line}")
            pytest.skip("\n".join(report))


class TestComprehensiveSummary:
    """Aggregate all findings into a single summary report."""

    def test_full_scan_summary(self):
        """Run all pattern scans and produce a summary."""
        all_patterns = {}
        all_patterns.update(REMOVED_TYPE_ALIASES)
        all_patterns.update(REMOVED_FUNCTIONS)
        all_patterns.update(MOVED_EXCEPTIONS)
        all_patterns.update(REMOVED_CONSTANTS)
        all_patterns.update(DISTUTILS_PATTERN)

        findings = _scan_files_for_patterns(all_patterns)

        high = [f for f in findings if f[4] == "HIGH"]
        medium = [f for f in findings if f[4] == "MEDIUM"]
        low = [f for f in findings if f[4] == "LOW"]

        summary = [
            f"Numpy 2 compatibility scan: {len(PYTHON_FILES)} files scanned",
            f"  HIGH severity: {len(high)}",
            f"  MEDIUM severity: {len(medium)}",
            f"  LOW severity: {len(low)}",
            f"  Total issues: {len(findings)}",
        ]

        if findings:
            summary.append("\nAll findings:")
            for fpath, line, matched, replacement, severity in sorted(findings):
                summary.append(
                    f"  [{severity}] {fpath}:{line} -- {matched} -> {replacement}"
                )

        # This test always passes -- it's a summary reporter
        print("\n".join(summary))
