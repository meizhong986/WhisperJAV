"""Verify v1.8.11 Round-2 sanitizer hardening catches all in-scope escapees.

Scope cats (hard-drop expected): CAT1, CAT2, CAT3, CAT5, CAT6, CAT7, CAT11.
Out of scope for v1.8.11: CAT4 (soft rules, deferred to v1.8.12), CAT8/9
(repetition — existing cleaner already handles), CAT10 (not observed), CAT12
(soft rules, deferred), CAT13 (mis-transcription class, deferred).

Also verifies that 8 control subs (legitimate Japanese dialogue) are NOT
dropped by the new stages — catches false positives.

Data source: FORCES use of bundled filter_list_v08.json by pointing
FILTER_LIST_URL at the local file. Bypasses gist + cache for deterministic
results independent of when the user updates the gist.

Usage:
    python tests/fixtures/sanitizer_regression/round2_escapees/verify_round2.py

Exit code 0 on full pass, 1 on any mismatch.
"""

import io
import json
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", newline="")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Force bundled data for deterministic, offline-independent results.
# We use SanitizationConfig's built-in URL overrides (fields
# hallucination_exact_list_url / hallucination_regex_patterns_url). When the
# URL does not start with http://, the loader reads it as a local file —
# cache and network are bypassed. See hallucination_remover._load_json_from_url.
BUNDLED_FILTER = REPO_ROOT / "whisperjav" / "data" / "hallucination_filters" / "filter_list_v08.json"
BUNDLED_REGEXP = REPO_ROOT / "whisperjav" / "data" / "hallucination_filters" / "regexp_v09.json"

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig
import pysrt


def main() -> int:
    here = Path(__file__).parent
    input_srt = here / "input.srt"
    expected_map = json.loads((here / "expected_map.json").read_text(encoding="utf-8"))

    # Run sanitizer on a copy so we don't mutate the input
    work_dir = here / "_run_output"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    target_srt = work_dir / "input.srt"
    shutil.copy2(input_srt, target_srt)

    config = SanitizationConfig()
    config.primary_language = "ja"
    config.save_artifacts = True
    config.preserve_original_file = True
    # Override URLs to force bundled data (bypasses gist + cache).
    config.hallucination_exact_list_url = str(BUNDLED_FILTER)
    config.hallucination_regex_patterns_url = str(BUNDLED_REGEXP)
    sanitizer = SubtitleSanitizer(config=config)
    result = sanitizer.process(target_srt)

    # Read sanitized output
    out_subs = list(pysrt.open(str(result.sanitized_path), encoding="utf-8"))
    surviving_texts = {s.text.strip() for s in out_subs}

    # Build per-sub outcome
    expected_drops = expected_map["expected_drops"]
    expected_keeps = expected_map["expected_keeps"]

    drop_failures = []  # should have been dropped but survived
    keep_failures = []  # should have survived but was dropped

    for entry in expected_drops:
        text = entry["text"]
        if text in surviving_texts:
            drop_failures.append(entry)

    for entry in expected_keeps:
        text = entry["text"]
        if text not in surviving_texts:
            keep_failures.append(entry)

    # Summary
    print(f"Input subs:       {len(expected_drops) + len(expected_keeps)}")
    print(f"Surviving subs:   {len(out_subs)}")
    print(f"Expected drops:   {len(expected_drops)}  ({len(expected_drops) - len(drop_failures)} OK)")
    print(f"Expected keeps:   {len(expected_keeps)}  ({len(expected_keeps) - len(keep_failures)} OK)")
    print()

    by_cat = {}
    for entry in drop_failures:
        by_cat.setdefault(entry["cat"], []).append(entry)

    if drop_failures:
        print(f"FAIL — {len(drop_failures)} escapee(s) were NOT dropped:")
        for cat in sorted(by_cat):
            items = by_cat[cat]
            print(f"  [{cat}] {len(items)} surviving:")
            for e in items:
                print(f"    #{e['idx']:3d}  {e['text']!r}")
        print()

    if keep_failures:
        print(f"FAIL — {len(keep_failures)} control sub(s) were INCORRECTLY DROPPED:")
        for e in keep_failures:
            print(f"  #{e['idx']:3d}  [{e.get('label','?')}]  {e['text']!r}")
        print()

    # Show artifact reasons for drops (diagnostic)
    artifact_reasons = {}
    for a in sanitizer.artifact_entries:
        artifact_reasons[a.reason] = artifact_reasons.get(a.reason, 0) + 1
    if artifact_reasons:
        print("Drop reasons recorded:")
        for reason, count in sorted(artifact_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")

    if drop_failures or keep_failures:
        print()
        print("OVERALL: FAIL")
        return 1

    print()
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
