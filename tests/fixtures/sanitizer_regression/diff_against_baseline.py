"""Diff current sanitizer output against the frozen baseline.

Run this AFTER applying any fix to confirm changes match the acceptance
matrix in ACCEPTANCE.md. Outputs a concise per-fixture diff report.

Usage:
    python tests/fixtures/sanitizer_regression/diff_against_baseline.py [baseline_sha]
"""

import difflib
import io
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout so Japanese text in diff output doesn't crash on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", newline="")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig

BASE_DIR = Path(__file__).parent
FROZEN_BASELINE_SHA = "e224333"  # captured 2026-04-18 — the pre-fix reference


def run_sanitizer_to_dir(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        # Clean contents but keep dir if Windows holds a handle
        for f in dst_dir.rglob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
        for d in sorted(dst_dir.rglob("*"), key=lambda p: -len(p.parts)):
            if d.is_dir():
                try:
                    d.rmdir()
                except Exception:
                    pass
    dst_dir.mkdir(parents=True, exist_ok=True)

    config = SanitizationConfig()
    config.primary_language = "ja"
    config.save_artifacts = True
    config.preserve_original_file = True
    sanitizer = SubtitleSanitizer(config=config)

    for srt in sorted(src_dir.rglob("*.srt")):
        rel = srt.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srt, target)
        try:
            sanitizer.process(target)
        except Exception as e:
            print(f"ERROR processing {rel}: {e}", file=sys.stderr)


def main():
    baseline_sha = sys.argv[1] if len(sys.argv) > 1 else FROZEN_BASELINE_SHA
    baseline_dir = BASE_DIR / f"baseline_{baseline_sha}"
    if not baseline_dir.exists():
        print(f"ERROR: baseline directory not found: {baseline_dir}", file=sys.stderr)
        sys.exit(1)

    # Compute current HEAD sha
    current_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()

    current_dir = BASE_DIR / f"current_{current_sha}"
    synthetic = BASE_DIR / "synthetic"

    print(f"Baseline:  {baseline_dir.name}")
    print(f"Current:   {current_dir.name}")
    print(f"Fixtures:  {synthetic}")
    print()

    run_sanitizer_to_dir(synthetic, current_dir)

    # Diff every sanitized file
    changes = 0
    identical = 0
    both_missing = 0
    only_baseline = 0
    only_current = 0

    all_sanitized = set()
    for p in baseline_dir.rglob("*.sanitized.srt"):
        all_sanitized.add(p.relative_to(baseline_dir))
    for p in current_dir.rglob("*.sanitized.srt"):
        all_sanitized.add(p.relative_to(current_dir))

    print("=" * 70)
    for rel in sorted(all_sanitized):
        b = baseline_dir / rel
        c = current_dir / rel
        b_text = b.read_text(encoding='utf-8') if b.exists() else None
        c_text = c.read_text(encoding='utf-8') if c.exists() else None
        if b_text is None and c_text is None:
            both_missing += 1
            continue
        if b_text is None:
            only_current += 1
            print(f"[ONLY-CURRENT] {rel}")
            continue
        if c_text is None:
            only_baseline += 1
            print(f"[ONLY-BASELINE] {rel}")
            continue
        if b_text == c_text:
            identical += 1
            continue
        changes += 1
        print(f"\n[CHANGED] {rel}")
        print("-- BASELINE --")
        print(b_text.rstrip())
        print("-- CURRENT --")
        print(c_text.rstrip())
        print()
    print("=" * 70)
    print(f"Identical:     {identical}")
    print(f"Changed:       {changes}")
    print(f"Only baseline: {only_baseline}")
    print(f"Only current:  {only_current}")
    print(f"Total:         {identical + changes + only_baseline + only_current}")


if __name__ == "__main__":
    main()
