"""Generate synthetic SRT fixtures for the v1.8.11 sanitizer regression corpus.

Run once to produce the fixture files; the output files are the committed
contract. Rerunning overwrites the fixtures, which should only happen if the
plan's §7.2 test-case specification changes (requires plan revision).

Plan reference: docs/plans/V1811_SANITIZER_FIX_PLAN.md §7.2
"""

from pathlib import Path


def write_srt(path: Path, blocks):
    """Write SRT blocks to path. blocks is a list of (index, start, end, text) tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for i, (idx, start, end, text) in enumerate(blocks):
            f.write(f"{idx}\n{start} --> {end}\n{text}\n")
            if i < len(blocks) - 1:
                f.write("\n")


BASE = Path(__file__).parent / "synthetic"


# ---------- Fix 1: ${N:0:M} slice syntax ----------
#
# IMPORTANT: `あ` kana runs (e.g. "あああああああああああ") are in the curated
# filter_list_v08.json (3,501 phrases) and get caught by exact-match BEFORE the
# buggy regex partial-strip ever runs. That means `あ`-shaped inputs do NOT
# exercise Fix 1 — they mask it via step 1.1.a.ii/iii.
#
# Inputs that DO trigger the bug (verified with HEAD e224333):
#   · `いいいいいいいいいいいい?`  → OUT `?`     (residue — bug!)
#   · `いいいいいいいいいいいい。`  → OUT `。`    (residue — bug!)
#   · `うううううううううううう`   → OUT ``      (empty, dropped)
#   · `わああうううううううう...`  → OUT `わああ` (correct strip)
#   · `はぁはぁ...` / `〜〜〜...`  → OUT ``      (empty, dropped)
#
# The `い/う/え/お` runs are not in the filter list corpus, so the exact-match
# steps pass and we reach the regex partial-strip — where the bug lives.

FIX1 = BASE / "fix1_slice_syntax"

# Durations are 500ms so the post-strip residue has CPS > 1.0 and isn't
# removed by _remove_abnormally_fast_subs (MIN_SAFE_CPS=1.0). This lets the
# BEFORE/AFTER difference be observable in the final sanitized SRT, not just
# in artifact records. At 0.5s: 1 char = 2 CPS (kept), 2 chars = 4 CPS (kept),
# 3 chars = 6 CPS (kept), none above MAX_SAFE_CPS=30.

# Pure kana run that bypasses filter list → hits L142 partial strip
# Pre-fix: text → '' → dropped (empty text check)
# Post-fix: group(1)[:2] = 'いい' → kept
write_srt(FIX1 / "01_vowel_run_alone.srt", [
    (1, "00:00:00,000", "00:00:00,500", "いいいいいいいいいいいい"),
])

# #287 reproducer: kana run + single non-repeated punctuation
# Pre-fix: '?' (kana stripped to empty, punct survives — the #287 symptom)
# Post-fix: 'いい?' (first 2 kana kept, punct preserved)
write_srt(FIX1 / "02_vowel_run_with_punct.srt", [
    (1, "00:00:00,000", "00:00:00,500", "いいいいいいいいいいいい?"),
])

# Variant: different punctuation, verifying bug isn't specific to '?'
# Pre-fix: '。'  Post-fix: 'うう。'
write_srt(FIX1 / "02b_vowel_run_with_period.srt", [
    (1, "00:00:00,000", "00:00:00,500", "うううううううううううう。"),
])

# L156 pattern: (はぁ|ハァ|...){10,} — partial strip → group(1) = "はぁ" (last match)
# Pre-fix: empty → dropped
# Post-fix: group(1)[:2] = 'はぁ' → kept
write_srt(FIX1 / "03_vocalization_hagh.srt", [
    (1, "00:00:00,000", "00:00:00,500", "はぁはぁはぁはぁはぁはぁはぁはぁはぁはぁ"),
])

# L162 pattern: (あっ|アッ|...){10,} partial strip
# Uses 'いっ' to bypass filter list
write_srt(FIX1 / "04_small_tsu_run.srt", [
    (1, "00:00:00,000", "00:00:00,500", "いっいっいっいっいっいっいっいっいっいっ"),
])

# L170 pattern: ([～〜ー]{10,}) → partial strip
# Pre-fix: '' → dropped
# Post-fix: '〜〜〜' (3 chars — no kana/kanji/alnum → Fix 2 will drop)
write_srt(FIX1 / "05_long_vowel_mark.srt", [
    (1, "00:00:00,000", "00:00:00,500", "〜〜〜〜〜〜〜〜〜〜〜〜"),
])

# Control: plain text with no trigger
write_srt(FIX1 / "06_non_slice_replacement.srt", [
    (1, "00:00:00,000", "00:00:02,000", "こんにちは"),
])

# Mixed content: kana run embedded in other kana
# Pre-fix: 'わああ' (kana run stripped to empty, surrounding kept)
# Post-fix: 'わああうう' (kana run trimmed to 2 chars)
write_srt(FIX1 / "07_kana_in_context.srt", [
    (1, "00:00:00,000", "00:00:00,500", "わああうううううううううううう"),
])


# ---------- Fix 2: symbol-only purge ----------
FIX2 = BASE / "fix2_symbol_purge"

write_srt(FIX2 / "01_bang_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "!!"),
])

write_srt(FIX2 / "02_question_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "???"),
])

write_srt(FIX2 / "03_ellipsis_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "……"),
])

write_srt(FIX2 / "04_music_notes_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "♪♫♩"),
])

write_srt(FIX2 / "05_animal_emoji_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "🐈🐈🐈"),
])

write_srt(FIX2 / "06_fullwidth_punct_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "！！！"),
])

write_srt(FIX2 / "07_whitespace_plus_punct.srt", [
    (1, "00:00:00,000", "00:00:02,000", "  !!  "),
])

write_srt(FIX2 / "08_mixed_kana_and_punct.srt", [
    (1, "00:00:00,000", "00:00:02,000", "こんにちは!!"),
])

write_srt(FIX2 / "09_kanji_only.srt", [
    (1, "00:00:00,000", "00:00:02,000", "家"),
])

write_srt(FIX2 / "10_fullwidth_alnum.srt", [
    (1, "00:00:00,000", "00:00:02,000", "ｈｅｌｌｏ"),
])

# Simulates the #287 shape end-to-end: kana run + trailing single punctuation.
# Under Fix 1, this becomes "いい?" (kana residue kept). Fix 2 keeps it (has kana).
# Uses 'い' to bypass the filter list and actually reach the buggy regex path.
# 0.5s duration so residue survives CPS filter for observable BEFORE/AFTER.
write_srt(FIX2 / "11_fix1_residue_kana_plus_q.srt", [
    (1, "00:00:00,000", "00:00:00,500", "いいいいいいいいいいいい?"),
])


# ---------- Fix 3: CPS newline exclusion ----------
FIX3 = BASE / "fix3_newline_cps"

write_srt(FIX3 / "01_multiline_normal.srt", [
    (1, "00:00:00,000", "00:00:02,000", "こんにちは\n今日は"),
])

# Duration 0.1s — any CPS calc gives very high rate; removed regardless.
write_srt(FIX3 / "02_multiline_fast.srt", [
    (1, "00:00:00,000", "00:00:00,100", "はい\nはい\nはい"),
])

# Duration 10s with 4 kana + 3 newlines:
#   Pre-fix:  text_len=7, cps=0.7  → KEEP (len > 4, so low-CPS check doesn't apply)
#   Post-fix: text_len=4, cps=0.4  → REMOVE (len ≤ 4 AND cps < 1.0)
# This is the intended correctness improvement for Fix 3.
write_srt(FIX3 / "03_multiline_slow_short.srt", [
    (1, "00:00:00,000", "00:00:10,000", "あ\nい\nう\nえ"),
])

write_srt(FIX3 / "04_single_line_baseline.srt", [
    # Single-line: no newlines to exclude; Fix 3 should produce identical output
    # Uses Japanese text to avoid being caught by the English-only regex at L233
    (1, "00:00:00,000", "00:00:01,000", "あいう"),
])


# ---------- Fix 5a: SDH patterns ----------
FIX5A = BASE / "fix5a_sdh"

write_srt(FIX5A / "01_tortoise_shell.srt", [
    (1, "00:00:00,000", "00:00:03,000", "〔効果音〕こんにちは"),
])

write_srt(FIX5A / "02_new_music_symbols.srt", [
    (1, "00:00:00,000", "00:00:03,000", "♫♩♬🎵🎶こんにちは"),
])

write_srt(FIX5A / "03_sdh_bgm.srt", [
    (1, "00:00:00,000", "00:00:03,000", "[BGM]おはよう"),
])

write_srt(FIX5A / "04_sdh_breathing.srt", [
    (1, "00:00:00,000", "00:00:03,000", "(喘息)"),
])

write_srt(FIX5A / "05_sdh_noise.srt", [
    (1, "00:00:00,000", "00:00:03,000", "【ノイズ】"),
])

write_srt(FIX5A / "06_regression_existing_music_symbol.srt", [
    (1, "00:00:00,000", "00:00:03,000", "♪★こんにちは"),
])


# ---------- Combined cases ----------
COMBINED = BASE / "combined"

# Exercises all four fixes in one SRT. Each block demonstrates a specific fix.
write_srt(COMBINED / "01_all_four_fixes_together.srt", [
    # Fix 1: vowel run + single punct — becomes "いい?" (bypasses filter list via 'い')
    #        Short duration so residue survives CPS filter
    (1, "00:00:00,000", "00:00:00,500", "いいいいいいいいいいいい?"),
    # Fix 2: pure bang — dropped
    (2, "00:00:06,000", "00:00:08,000", "!!"),
    # Fix 3: multiline slow short — dropped (post-fix)
    (3, "00:00:09,000", "00:00:19,000", "あ\nい\nう\nえ"),
    # Fix 5a: SDH tortoise shell — becomes "おはよう"
    (4, "00:00:20,000", "00:00:23,000", "〔効果音〕おはよう"),
    # Normal content — passes through
    (5, "00:00:24,000", "00:00:27,000", "こんにちは"),
])

# Negative control: 20+ legitimate JAV-style subs that MUST all be preserved.
# Any drop or mutation here is a false-positive regression and blocks release.
write_srt(COMBINED / "02_negative_control_all_legitimate.srt", [
    (1,  "00:00:00,000", "00:00:02,000", "こんにちは"),
    (2,  "00:00:03,000", "00:00:05,000", "今日はいい天気ですね"),
    (3,  "00:00:06,000", "00:00:08,000", "ありがとう"),
    (4,  "00:00:09,000", "00:00:11,000", "すみません"),
    (5,  "00:00:12,000", "00:00:13,500", "はい"),
    (6,  "00:00:14,000", "00:00:15,500", "いいえ"),
    (7,  "00:00:16,000", "00:00:17,500", "そうですね"),
    (8,  "00:00:18,000", "00:00:20,000", "気持ちいい"),
    (9,  "00:00:21,000", "00:00:23,000", "もっと"),
    (10, "00:00:24,000", "00:00:26,000", "だめ"),
    (11, "00:00:27,000", "00:00:29,000", "やめて"),
    (12, "00:00:30,000", "00:00:33,000", "お父さん"),
    (13, "00:00:34,000", "00:00:36,000", "お母さん"),
    (14, "00:00:37,000", "00:00:40,000", "ちょっと待って"),
    (15, "00:00:41,000", "00:00:44,000", "分かりました"),
    (16, "00:00:45,000", "00:00:48,000", "どうしたの"),
    (17, "00:00:49,000", "00:00:52,000", "大丈夫ですか"),
    (18, "00:00:53,000", "00:00:56,000", "ゆっくりして"),
    (19, "00:00:57,000", "00:01:00,000", "少し休みましょう"),
    (20, "00:01:01,000", "00:01:04,000", "また明日"),
    (21, "00:01:05,000", "00:01:08,000", "さようなら"),
    (22, "00:01:09,000", "00:01:12,000", "また会いたいね"),
])


if __name__ == "__main__":
    print("Synthetic fixtures generated under:", BASE)
    count = 0
    for p in BASE.rglob("*.srt"):
        count += 1
        print(f"  {p.relative_to(BASE)}")
    print(f"Total: {count} fixture files")
