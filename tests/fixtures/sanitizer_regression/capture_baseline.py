"""Capture current sanitizer output for every SRT in synthetic/.

Writes results to baseline_<short_sha>/ so we can diff against post-fix output
during verification (Gate G7 in V1811_SANITIZER_FIX_PLAN.md §9.7).

The output directory is named after the git HEAD short SHA so baselines from
different code states are never confused.

Usage:
    python tests/fixtures/sanitizer_regression/capture_baseline.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Ensure we're importing the repo's whisperjav, not a pip-installed one
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig


def get_short_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True
        ).strip()
        return out
    except Exception as e:
        return f"unknown_{e}"


def main():
    base_dir = Path(__file__).parent
    synthetic_dir = base_dir / "synthetic"
    sha = get_short_sha()
    out_dir = base_dir / f"baseline_{sha}"

    if out_dir.exists():
        # Delete previous baseline for this SHA to get fresh capture
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Force primary_language=ja so the CJK path is taken
    config = SanitizationConfig()
    config.primary_language = "ja"
    # Keep artifacts so we can see what was dropped and why
    config.save_artifacts = True
    config.preserve_original_file = True  # write sanitized to a sanitized SRT, not overwrite

    sanitizer = SubtitleSanitizer(config=config)

    summary = []
    errors = []

    for srt_path in sorted(synthetic_dir.rglob("*.srt")):
        rel = srt_path.relative_to(synthetic_dir)
        # Mirror directory structure under baseline_<sha>/
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        # Copy input into target location so sanitizer can write outputs next to it
        input_copy = target
        shutil.copy2(srt_path, input_copy)

        try:
            result = sanitizer.process(input_copy)
            # result.sanitized_path is the cleaned SRT
            summary.append({
                "fixture": str(rel).replace('\\', '/'),
                "original_count": result.statistics.get("original_count", 0),
                "final_count": result.statistics.get("final_count", 0),
                "sanitized_path": str(result.sanitized_path.relative_to(out_dir)).replace('\\', '/'),
            })
        except Exception as e:
            errors.append({"fixture": str(rel), "error": str(e)})
            print(f"ERROR on {rel}: {e}", file=sys.stderr)

    # Write summary
    report_path = out_dir / "BASELINE_REPORT.txt"
    with report_path.open('w', encoding='utf-8') as f:
        f.write(f"Baseline captured at HEAD: {sha}\n")
        f.write(f"Total fixtures: {len(summary) + len(errors)}\n")
        f.write(f"Successful: {len(summary)}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write("\n--- Successful captures ---\n")
        for entry in summary:
            f.write(f"  {entry['fixture']}: {entry['original_count']} -> {entry['final_count']}\n")
        if errors:
            f.write("\n--- Errors ---\n")
            for entry in errors:
                f.write(f"  {entry['fixture']}: {entry['error']}\n")

    print(f"\nBaseline captured to: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Successful: {len(summary)}, Errors: {len(errors)}")


if __name__ == "__main__":
    main()
