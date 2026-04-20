# Sanitizer Regression Corpus — Acceptance Matrix

> Contract document for v1.8.11 sanitizer fixes.
> Baseline captured at HEAD: **e224333** (2026-04-18)
> Plan reference: `docs/plans/V1811_SANITIZER_FIX_PLAN.md`

## Acceptance Categories

Every fixture must produce an output that falls into exactly one of these categories:

- **A. EXPECTED CHANGE** — output differs from baseline; the difference is an expected consequence of Fix 1, 2, 3, or 5a
- **B. BYTE-IDENTICAL** — output is byte-identical to baseline
- **C. EXPECTED DROP** — subtitle removed; artifact record shows a valid reason code (`hallucination`, `repetition`, `abnormally_fast_cps_*`, `abnormally_slow_cps_*`, `symbol_only_residue`)

Any output not fitting A/B/C blocks the v1.8.11 release.

---

## Fix 1 — `${N:0:M}` slice syntax

Durations are 0.5s so post-strip residues have CPS > 1.0 and survive the CPS filter.

| Fixture | Input | Baseline (pre-fix) | Expected (post-fix) | Category |
|---|---|---|---|---|
| 01_vowel_run_alone.srt | `いいいいいいいいいいいい` | (dropped — empty) | `いい` (kept) | **A** |
| 02_vowel_run_with_punct.srt | `いいいいいいいいいいいい?` | `?` (kept — #287 symptom) | `いい?` (kept) | **A** |
| 02b_vowel_run_with_period.srt | `うううううううううううう。` | `。` | `うう。` | **A** |
| 03_vocalization_hagh.srt | `はぁはぁはぁはぁはぁはぁはぁはぁはぁはぁ` | (dropped — empty) | `はぁ` (kept) | **A** |
| 04_small_tsu_run.srt | `いっいっいっいっいっいっいっいっいっいっ` | (dropped — empty) | `いっ` (kept) | **A** |
| 05_long_vowel_mark.srt | `〜〜〜〜〜〜〜〜〜〜〜〜` | (dropped — empty) | (dropped by Fix 2 — `〜〜〜` is symbol-only) | **C** |
| 06_non_slice_replacement.srt | `こんにちは` | `こんにちは` | `こんにちは` | **B** |
| 07_kana_in_context.srt | `わああうううううううううううう` | `わああ` | `わああうう` | **A** |

---

## Fix 2 — Symbol-only purge

| Fixture | Input | Baseline (pre-fix) | Expected (post-fix) | Category |
|---|---|---|---|---|
| 01_bang_only.srt | `!!` | (dropped — L135 `nonsensical`) | (dropped — `symbol_only_residue` if reaches Fix 2, or same reason as baseline) | **B** or **C** |
| 02_question_only.srt | `???` | (dropped — L135) | (dropped) | **B** or **C** |
| 03_ellipsis_only.srt | `……` | `……` (kept — L135 doesn't match ellipsis) | (dropped — `symbol_only_residue`) | **A** |
| 04_music_notes_only.srt | `♪♫♩` | `♫♩` (kept — ♪ stripped by L268) | (dropped — `symbol_only_residue` after Fix 5a also strips ♫♩) | **A** |
| 05_animal_emoji_only.srt | `🐈🐈🐈` | `🐈🐈🐈` (kept) | (dropped — `symbol_only_residue`) | **A** |
| 06_fullwidth_punct_only.srt | `！！！` | (dropped) | (dropped) | **B** or **C** |
| 07_whitespace_plus_punct.srt | `  !!  ` | (dropped — L135) | (dropped) | **B** or **C** |
| 08_mixed_kana_and_punct.srt | `こんにちは!!` | (dropped — `こんにちは` in filter list, `!!` stripped) | (dropped — same) | **B** |
| 09_kanji_only.srt | `家` | `家` (kept — has kanji) | `家` (kept) | **B** |
| 10_fullwidth_alnum.srt | `ｈｅｌｌｏ` | `ｈｅｌｌｏ` (kept) | `ｈｅｌｌｏ` (kept) | **B** |
| 11_fix1_residue_kana_plus_q.srt | `いいいいいいいいいいいい?` | `?` (bug residue) | `いい?` (Fix 1 keeps kana; Fix 2 keeps — has kana) | **A** |

Notes:
- For cases where baseline already drops via a different path, post-fix may either (a) drop via the same path (B) or (b) drop via `symbol_only_residue` if control flow changes slightly (C). Either is acceptable provided the final outcome matches.

---

## Fix 3 — CPS newline exclusion

| Fixture | Input | Baseline text_len | Post-fix text_len | Expected post-fix | Category |
|---|---|---|---|---|---|
| 01_multiline_normal.srt | `こんにちは\n今日は` @ 2s | 9 (incl. `\n`) | 8 | kept (both CPS in safe range) | **B** |
| 02_multiline_fast.srt | `はい\nはい\nはい` @ 0.1s | 8 (incl. 2x`\n`) | 6 | dropped (both: CPS > 30) | **B** |
| 03_multiline_slow_short.srt | `あ\nい\nう\nえ` @ 10s | 7 | 4 | **dropped (new behavior)** — slow CPS + len ≤ 4 | **A** |
| 04_single_line_baseline.srt | `あいう` @ 1s | 3 | 3 | kept (no newlines) | **B** |

---

## Fix 5a — SDH patterns

| Fixture | Input | Baseline | Expected post-fix | Category |
|---|---|---|---|---|
| 01_tortoise_shell.srt | `〔効果音〕こんにちは` | `〔効果音〕こんにちは` (unchanged — no pattern for 〔〕) | `こんにちは` | **A** |
| 02_new_music_symbols.srt | `♫♩♬🎵🎶こんにちは` | `♫♩♬🎵🎶こんにちは` (unchanged — ♫♩♬🎵🎶 not in current L268) | `こんにちは` | **A** |
| 03_sdh_bgm.srt | `[BGM]おはよう` | `[BGM]おはよう` (unchanged — specific `[音楽]` only) | `おはよう` | **A** |
| 04_sdh_breathing.srt | `(喘息)` | (dropped — L65 strips halfwidth parens) | (dropped — same via L65 or new pattern) | **B** |
| 05_sdh_noise.srt | `【ノイズ】` | (dropped — L86 strips fullwidth square) | (dropped — same) | **B** |
| 06_regression_existing_music_symbol.srt | `♪★こんにちは` | `こんにちは` (♪ and ★ stripped by L268) | `こんにちは` (unchanged) | **B** |

---

## Combined fixtures

### `01_all_four_fixes_together.srt`

Exercises all four fixes in one SRT. Expected post-fix output:

| Block | Input | Expected |
|---|---|---|
| 1 | `いいいいいいいいいいいい?` @ 0.5s | `いい?` (Fix 1 produces residue, Fix 2 keeps) |
| 2 | `!!` @ 2s | dropped |
| 3 | `あ\nい\nう\nえ` @ 10s | dropped (Fix 3: text_len=4, CPS=0.4) |
| 4 | `〔効果音〕おはよう` @ 3s | `おはよう` (Fix 5a strips tortoise shell) |
| 5 | `こんにちは` @ 3s | dropped (filter list hit — pre-existing) |

Final SRT should contain subs 1 and 4 only (renumbered 1 and 2).

### `02_negative_control_all_legitimate.srt`

**Criterion**: post-fix output must be **BYTE-IDENTICAL to baseline**.

Note: baseline already drops 6 of 22 subs due to filter-list inclusions (pre-existing behavior, not in scope for v1.8.11). The remaining 16 kept subs must stay kept.

Pre-existing drops (informational only — not our fixes' concern):
- `こんにちは`, `すみません`, `そうですね`, `気持ちいい`, `お母さん`, and one more (indexes 1, 4, 7, 8, 13 per baseline audit).

Any NEW drop introduced by Fix 1/2/3/5a is a false positive and blocks release.

---

## Verification Protocol (Phase 6)

1. Run `python tests/fixtures/sanitizer_regression/generate_synthetic.py` — regenerates fixtures (idempotent).
2. Run `python tests/fixtures/sanitizer_regression/capture_baseline.py` — captures current-state output. Named after current HEAD SHA.
3. After each code fix, re-run step 2 — creates new baseline directory.
4. Diff the post-fix baseline against `baseline_e224333/` directory:
   - For every fixture, classify diff as A / B / C per this document.
   - Any fixture output not classifiable blocks the release.
5. Archive diffs as evidence in plan document appendix.
