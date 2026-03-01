#!/usr/bin/env python3
"""
Release Acceptance Test Suite for WhisperJAV.

Exercises key CLI pipelines against real test media to catch regressions
that unit tests with mocks cannot find. Run before every release.

Test media:
    - MIAA-432.20sec_piano.wav (20s) — works with all modes
    - MIAA-432.5sec.wav (6s) — quick smoke for faster/fast only

Run all acceptance tests:
    python -m pytest tests/test_release_acceptance.py -v -s

Run just the fast ones:
    python -m pytest tests/test_release_acceptance.py -v -s -k "faster"

Standalone mode (preferred for release gating):
    python tests/test_release_acceptance.py
"""

import os
import re
import sys
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEST_MEDIA_DIR = REPO_ROOT / "test_media"
TEST_AUDIO_5S = TEST_MEDIA_DIR / "MIAA-432.5sec.wav"
TEST_AUDIO_20S = TEST_MEDIA_DIR / "MIAA-432.20sec_piano.wav"

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.e2e]

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceCase:
    """Definition of a single acceptance test case."""
    name: str
    cli_args: List[str]
    audio_file: Path
    expected_srt_suffix: str  # e.g. ".ja.whisperjav.srt" or ".ja.merged.whisperjav.srt"
    timeout: int = 120
    slow: bool = False


ACCEPTANCE_CASES: List[AcceptanceCase] = [
    AcceptanceCase(
        name="faster_smoke_5s",
        cli_args=["--mode", "faster"],
        audio_file=TEST_AUDIO_5S,
        expected_srt_suffix=".ja.whisperjav.srt",
        timeout=120,
    ),
    AcceptanceCase(
        name="faster_20s",
        cli_args=["--mode", "faster"],
        audio_file=TEST_AUDIO_20S,
        expected_srt_suffix=".ja.whisperjav.srt",
        timeout=120,
    ),
    AcceptanceCase(
        name="fast_20s",
        cli_args=["--mode", "fast"],
        audio_file=TEST_AUDIO_20S,
        expected_srt_suffix=".ja.whisperjav.srt",
        timeout=180,
    ),
    AcceptanceCase(
        name="balanced_20s",
        cli_args=["--mode", "balanced"],
        audio_file=TEST_AUDIO_20S,
        expected_srt_suffix=".ja.whisperjav.srt",
        timeout=300,
        slow=True,
    ),
    AcceptanceCase(
        name="ensemble_faster_faster_20s",
        cli_args=[
            "--ensemble",
            "--pass1-pipeline", "faster",
            "--pass2-pipeline", "faster",
        ],
        audio_file=TEST_AUDIO_20S,
        expected_srt_suffix=".ja.merged.whisperjav.srt",
        timeout=300,
        slow=True,
    ),
]

# ---------------------------------------------------------------------------
# Result tracking (standalone mode)
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    passed: bool = False
    exit_code: int = -1
    elapsed: float = 0.0
    srt_exists: bool = False
    srt_nonempty: bool = False
    srt_valid_format: bool = False
    srt_entry_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRT_ENTRY_PATTERN = re.compile(
    r'\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
)


def run_cli(args: list, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run whisperjav CLI as a subprocess and return the result."""
    cmd = [sys.executable, "-m", "whisperjav.main"] + args
    print(f"\n>>> Running: {' '.join(cmd)}")

    # Force UTF-8 I/O in the subprocess to avoid charmap codec errors on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    # Ensure stdout/stderr are never None
    if result.stdout is None:
        result.stdout = ""
    if result.stderr is None:
        result.stderr = ""

    # Print truncated output for debugging (encode-safe for Windows charmap)
    for label, text in [("STDOUT", result.stdout), ("STDERR", result.stderr)]:
        if text:
            display = text[:5000] + "..." if len(text) > 5000 else text
            safe = display.encode("ascii", errors="replace").decode("ascii")
            print(f"{label}:\n{safe}")

    return result


def find_expected_srt(output_dir: Path, audio_file: Path, suffix: str) -> Optional[Path]:
    """Build the expected SRT path and return it if it exists."""
    basename = audio_file.stem  # e.g. "MIAA-432.5sec"
    expected = output_dir / f"{basename}{suffix}"
    return expected if expected.exists() else None


def validate_srt(srt_path: Path) -> Tuple[bool, bool, bool, int]:
    """
    Validate an SRT file.

    Returns:
        (exists, nonempty, valid_format, entry_count)
    """
    if not srt_path or not srt_path.exists():
        return False, False, False, 0

    content = srt_path.read_text(encoding="utf-8", errors="replace")
    nonempty = bool(content.strip())
    entries = SRT_ENTRY_PATTERN.findall(content)
    valid_format = len(entries) > 0
    return True, nonempty, valid_format, len(entries)


def _run_single_case(case: AcceptanceCase, output_dir: Path) -> CaseResult:
    """Execute a single acceptance test case and return the result."""
    result = CaseResult(name=case.name)
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    cli_args = [
        str(case.audio_file),
        *case.cli_args,
        "--language", "japanese",
        "--output-dir", str(case_dir),
        "--accept-cpu-mode",
    ]

    t0 = time.monotonic()
    try:
        proc = run_cli(cli_args, timeout=case.timeout)
        result.elapsed = time.monotonic() - t0
        result.exit_code = proc.returncode

        if proc.returncode != 0:
            result.error = (proc.stderr or proc.stdout or "")[:500]
            result.passed = False
            return result

        # Locate the SRT
        srt_path = find_expected_srt(case_dir, case.audio_file, case.expected_srt_suffix)

        # If exact name not found, try any SRT in the output dir as fallback
        if srt_path is None:
            srt_files = list(case_dir.glob("**/*.srt"))
            if srt_files:
                srt_path = srt_files[0]
                print(f"  NOTE: Expected SRT name not found, using fallback: {srt_path.name}")

        exists, nonempty, valid_fmt, count = validate_srt(srt_path)
        result.srt_exists = exists
        result.srt_nonempty = nonempty
        result.srt_valid_format = valid_fmt
        result.srt_entry_count = count

        # Hard fail: SRT must exist
        if not exists:
            result.error = "Output SRT file not found"
            result.passed = False
            return result

        # Soft warnings (piano audio may produce sparse output)
        if not nonempty:
            print(f"  WARNING: SRT file is empty (piano audio may produce no speech)")
        if not valid_fmt:
            print(f"  WARNING: SRT has no valid subtitle entries")

        result.passed = True

    except subprocess.TimeoutExpired:
        result.elapsed = time.monotonic() - t0
        result.error = f"Timed out after {case.timeout}s"
        result.passed = False
    except Exception as exc:
        result.elapsed = time.monotonic() - t0
        result.error = str(exc)[:500]
        result.passed = False

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def clean_output_dir(tmp_path):
    """Provide a clean temporary output directory for each test."""
    yield tmp_path


def skip_if_no_test_file(filepath: Path):
    if not filepath.exists():
        pytest.skip(f"Test file not found: {filepath}")


# ---------------------------------------------------------------------------
# Parametrized pytest tests
# ---------------------------------------------------------------------------

def _make_params():
    """Build pytest.param list with slow marker applied dynamically."""
    params = []
    for case in ACCEPTANCE_CASES:
        marks = [pytest.mark.slow] if case.slow else []
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


@pytest.mark.parametrize("case", _make_params())
def test_acceptance(case: AcceptanceCase, clean_output_dir):
    """Parametrized acceptance test for each pipeline mode."""
    skip_if_no_test_file(case.audio_file)

    result = _run_single_case(case, clean_output_dir)

    # Hard assertions
    assert result.exit_code == 0, (
        f"[{case.name}] CLI exited with code {result.exit_code}: {result.error}"
    )
    assert result.srt_exists, (
        f"[{case.name}] Output SRT file not found"
    )

    # Soft checks — warn but don't fail (piano audio is speech-sparse)
    if not result.srt_nonempty:
        warnings.warn(f"[{case.name}] SRT file is empty")
    if not result.srt_valid_format:
        warnings.warn(f"[{case.name}] SRT has no valid subtitle entries")

    print(
        f"\n  PASS: {case.name} | "
        f"exit={result.exit_code} | "
        f"entries={result.srt_entry_count} | "
        f"elapsed={result.elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _print_summary_table(results: List[CaseResult]):
    """Print a clean summary table to stdout."""
    name_w = max(len(r.name) for r in results)
    header = (
        f"{'Test':<{name_w}}  {'Status':>8}  {'Exit':>4}  "
        f"{'SRT?':>4}  {'Entries':>7}  {'Time':>8}  Error"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  RELEASE ACCEPTANCE TEST RESULTS")
    print(f"{sep}")
    print(header)
    print(sep)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        srt = "yes" if r.srt_exists else "no"
        entries = str(r.srt_entry_count) if r.srt_exists else "-"
        elapsed = f"{r.elapsed:.1f}s"
        error = r.error[:60] if r.error else ""
        print(
            f"{r.name:<{name_w}}  {status:>8}  {r.exit_code:>4}  "
            f"{srt:>4}  {entries:>7}  {elapsed:>8}  {error}"
        )

    print(sep)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.elapsed for r in results)
    verdict = "ALL PASSED" if passed == total else f"{total - passed} FAILED"
    print(f"  {passed}/{total} passed | Total: {total_time:.1f}s | {verdict}")
    print(sep)


def main():
    """Standalone runner — executes all acceptance tests outside pytest."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = REPO_ROOT / "test_results" / f"acceptance_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_base}")
    print(f"Test media dir:   {TEST_MEDIA_DIR}")

    results: List[CaseResult] = []
    for case in ACCEPTANCE_CASES:
        if not case.audio_file.exists():
            r = CaseResult(name=case.name, passed=False, error=f"Missing: {case.audio_file.name}")
            results.append(r)
            print(f"\nSKIP: {case.name} — test file not found: {case.audio_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Running: {case.name}")
        print(f"{'=' * 60}")

        r = _run_single_case(case, output_base)
        results.append(r)

        status = "PASS" if r.passed else "FAIL"
        print(f"  => {status} ({r.elapsed:.1f}s)")

    _print_summary_table(results)

    # Exit with failure code if any test failed
    failed = [r for r in results if not r.passed]
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
