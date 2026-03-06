# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-06 — v1.8.6 RELEASED | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | 31 open on GitHub

---

## Quick Stats

| Category | Count |
|----------|-------|
| Open on GitHub | 31 |
| Closed this cycle (on GitHub) | 46+ |
| **New since v1.8.6 release** | 3 (#195, #196, #197) |
| **Closeable now** (fix shipped or answered) | 14 |
| Fixed in v1.8.6 (released 2026-03-04) | 5 (#190, #183, #179, #194, #143) |
| Fixed in hotfix3 (released 2026-03-01) | 6 |
| Fixed in hotfix2 (released) | 7 |
| Fixed in hotfix1 (released) | 3 |
| **Active bugs (need work)** | 3 (#195, #196, #197) |
| Feature requests (open) | 17 |
| Questions / docs / support | 4 |
| Needs more info | 2 |
| Deferred (v1.9+) | 8 |

---

## NEW: Post-v1.8.6 Issues

### #197 — Installation problem on v1.8.6

**Reporter:** KenZP12 | **Opened:** 2026-03-06 | **Comments:** 2

**Symptom:** `urllib3 (2.6.3) or chardet (7.0.1)/charset_normalizer (3.4.4) doesn't match a supported version!` warning at startup. User reports process failure during translation.

**Analysis:** This is a `requests` library compatibility warning — not an error. It fires when installed `urllib3`/`chardet` versions are newer than what `requests` was tested against. The warning itself is harmless. User ran `pip install --upgrade requests` and confirmed all packages are satisfied, but issue persists.

**Root cause hypothesis:** The warning is a red herring. The real failure (if any) is likely unrelated — possibly the same local LLM translation issue as #196. Need the user to provide the actual error traceback, not just the warning.

**Recommendation:** Ask for actual error output. The warning alone doesn't cause failures. May be a duplicate of #196.

**Priority:** Medium | **Target:** v1.8.7

---

### #196 — Local Translation Errors (v1.8.6)

**Reporter:** destinyawaits | **Opened:** 2026-03-05 | **Comments:** 1 (zhstark confirmed same on Ubuntu 22.04)

**Symptom:** Local LLM translation with gemma-9b fails with:
1. "Hit API token limit, retrying batch without context..."
2. "No matches found in translation text using patterns" (7 regex patterns fail)

**Details:** Model: `gemma-2-9b-it-abliterated-Q4_K_M.gguf` (5.37GB), context size 8192 tokens. Batch auto-reduced from 30→17 by v1.8.6 auto-cap, but still hits token limit. Input: 1,038 lines across 23 scenes.

**Analysis:** The v1.8.6 auto-cap batch size fix (commit `24faa3b`) IS working (it reduced batch from 30→17), but the cap is still too generous for this model/content combination. Two sub-issues:

1. **Batch size still too large:** 17 lines with context may exceed 8192 tokens for long Japanese subtitle lines. The auto-cap formula likely needs a safety margin reduction.
2. **Regex pattern failure:** After the retry-without-context fallback, the model's output format doesn't match any of the 7 expected patterns. This is a PySubtrans response parsing issue — the model may be producing malformed output when pushed near its context limit.

**This is the SAME user as #183** — destinyawaits. The v1.8.6 fix improved but didn't fully resolve their workflow.

**Recommendation:** Investigate the auto-cap formula in `whisperjav/translate/`. The cap should be more conservative (maybe 60-70% of model context rather than raw limit). The regex failure is a downstream symptom.

**Priority:** High — regression report from a user whose issue was supposed to be fixed | **Target:** v1.8.7

---

### #195 — UnicodeDecodeError + AttributeError during "Transforming audio"

**Reporter:** v2lmmj04 | **Opened:** 2026-03-03 | **Comments:** 2

**Symptom:** Processing M4B-to-M4A converted files fails with:
1. `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 2382`
2. `AttributeError: 'NoneType' object has no attribute 'split'`

**Root cause (confirmed by user):** The M4A files had Japanese metadata (track names, album titles, performer info). FFmpeg outputs this metadata to stderr. `audio_extraction.py:86` uses `subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')` — this crashes on non-UTF-8 metadata bytes. When the decode fails, `result.stderr` is `None`, and line 89 (`result.stderr.split('\n')`) throws `AttributeError`.

**User workaround:** Strip metadata first: `ffmpeg -i input.m4b -map_metadata -1 -vn -acodec copy output.m4a`

**Fix:** Add `errors='replace'` to the subprocess call: `subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')`. This safely replaces non-UTF-8 bytes instead of crashing. Should also add a null guard on `result.stderr` for defense in depth.

**Priority:** Medium — clear fix, user has workaround | **Target:** v1.8.7

---

## Issues to Close (fix shipped in v1.8.6 or answered)

| # | Title | Reason | Action |
|---|-------|--------|--------|
| [#190](https://github.com/meizhong986/WhisperJAV/issues/190) | GBK codec crash | **Fixed v1.8.6** — process-wide UTF-8 (commit `68d1644`) | Close |
| [#194](https://github.com/meizhong986/WhisperJAV/issues/194) | M4B file support | **Done v1.8.6** — .m4b added (commit `5769688`) | Close |
| [#143](https://github.com/meizhong986/WhisperJAV/issues/143) | VTT output + custom models | **Done v1.8.6** — VTT format + custom provider | Close |
| [#179](https://github.com/meizhong986/WhisperJAV/issues/179) | Per-video pass ordering | **Done v1.8.6** — `--ensemble-serial` (commit `8bb5cb0`) | Close |
| [#183](https://github.com/meizhong986/WhisperJAV/issues/183) | Ubuntu LLM translation | **Partially fixed v1.8.6** — auto-cap batch + Linux upgrade. Note: #196 shows residual issue. | Close with note pointing to #196 |
| [#191](https://github.com/meizhong986/WhisperJAV/issues/191) | Pass2 missing (Chinese) | **Answered** — SSL cert error, not a bug | Close |
| [#193](https://github.com/meizhong986/WhisperJAV/issues/193) | How to update packages | **Support question** — urllib3 warning, not a bug | Close |
| [#188](https://github.com/meizhong986/WhisperJAV/issues/188) | Unknown provider: Gemini | **Fixed v1.8.6** — added google-api-core dependency | Close |
| [#150](https://github.com/meizhong986/WhisperJAV/issues/150) | Xiaomi Mimo API | **Done** — custom provider covers this | Close |
| [#162](https://github.com/meizhong986/WhisperJAV/issues/162) | Local model ggml.dll | **Fixed** in v1.8.4. Stale. | Close |
| [#146](https://github.com/meizhong986/WhisperJAV/issues/146) | Local server error | **Fixed** in v1.8.3. Stale 5+ weeks. | Close |
| [#174](https://github.com/meizhong986/WhisperJAV/issues/174) | GUI settings reset | **Duplicate** of #96. Translation settings fixed. | Close as dup |
| [#184](https://github.com/meizhong986/WhisperJAV/issues/184) | GUI save configurations | **Duplicate** of #96. | Close as dup |
| [#186](https://github.com/meizhong986/WhisperJAV/issues/186) | UnicodeEncodeError subprocess | **Fixed** in hotfix1. | Close |
| [#177](https://github.com/meizhong986/WhisperJAV/issues/177) | cp950 codec translation | **Fixed** in hotfix1 + v1.8.6. | Close |
| [#158](https://github.com/meizhong986/WhisperJAV/issues/158) | Linux LLM build regression | **Fixed** in hotfix3. | Close |
| [#178](https://github.com/meizhong986/WhisperJAV/issues/178) | Custom endpoint 404 | **Fixed** in hotfix3. | Close |

**Total closeable: 17** (14 from previous list + 3 newly confirmed from v1.8.6)

---

## Active Issues — Master Table

### Bugs

| # | Title | Status | Summary | Priority | Target |
|---|-------|--------|---------|----------|--------|
| [#197](https://github.com/meizhong986/WhisperJAV/issues/197) | Installation problem v1.8.6 | **NEW** | urllib3/chardet warning + reported failure. Need actual error traceback. Likely #196 or unrelated. | Medium | v1.8.7 |
| [#196](https://github.com/meizhong986/WhisperJAV/issues/196) | Local translation errors v1.8.6 | **NEW** | Auto-cap batch (30→17) still too generous for 8K context. Regex pattern failure on model output. Same user as #183. | **High** | v1.8.7 |
| [#195](https://github.com/meizhong986/WhisperJAV/issues/195) | UnicodeDecodeError audio extraction | **NEW** | FFmpeg metadata contains non-UTF-8 bytes. `audio_extraction.py:86` needs `errors='replace'`. User confirmed stripping metadata works. | Medium | v1.8.7 |
| [#189](https://github.com/meizhong986/WhisperJAV/issues/189) | Smart Merge clears one pass | Fixed hotfix2 | Quality-aware tiebreaker. Awaiting retest. | Low | — |
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate subtitles | **UNCLEAR** | User weifu8435. No new error logs. Possibly user confusion. | Low | — |
| [#185](https://github.com/meizhong986/WhisperJAV/issues/185) | v1.8.4+ regression | Fixed hotfix2 | Same user as #187. Same root cause as #173. | Low | — |
| [#176](https://github.com/meizhong986/WhisperJAV/issues/176) | Translation settings lost | Fixed hotfix2 | Translation persistence done. Full persistence tracked in #96. | Low | — |
| [#44](https://github.com/meizhong986/WhisperJAV/issues/44) | GUI drag-drop filename only | Open | Drag-drop sends filename not path. | Low | v1.8.7+ |
| [#90](https://github.com/meizhong986/WhisperJAV/issues/90) | Source build dep conflict | Open | Dependency version conflicts. | Low | v1.8.7+ |

---

### Feature Requests

| # | Title | Status | Summary | Target |
|---|-------|--------|---------|--------|
| [#96](https://github.com/meizhong986/WhisperJAV/issues/96) | Full GUI settings persistence | Partial | Translation + ensemble presets done. Full pipeline tab persistence remaining. | v1.9 |
| [#181](https://github.com/meizhong986/WhisperJAV/issues/181) | Frameless window | Open | PyWebView `frameless=True`. Cosmetic. | v1.9+ |
| [#180](https://github.com/meizhong986/WhisperJAV/issues/180) | Multi-language GUI (i18n) | Open | Full i18n framework. High effort. | v1.9+ |
| [#175](https://github.com/meizhong986/WhisperJAV/issues/175) | Chinese language | Open | Subset of #180. | v1.9+ |
| [#164](https://github.com/meizhong986/WhisperJAV/issues/164) | MPEG-TS + Google Drive | Open | (1) MPEG-TS remux. (2) Kaggle Drive integration. | v1.8.7 |
| [#159](https://github.com/meizhong986/WhisperJAV/issues/159) | CLI --vad-threshold | Open | Simple float arg. | v1.8.7 |
| [#142](https://github.com/meizhong986/WhisperJAV/issues/142) | AMD Radeon detection | Open | DirectML/ROCm. | v1.9+ |
| [#126](https://github.com/meizhong986/WhisperJAV/issues/126) | Recursive directory | Open | Walk subdirectories. | v1.9+ |
| [#114](https://github.com/meizhong986/WhisperJAV/issues/114) | DirectML for AMD/Intel | Open | Major platform enablement. | v1.9+ |
| [#99](https://github.com/meizhong986/WhisperJAV/issues/99) | 4GB VRAM solutions | Open | VRAM check + model suggestion. | v1.8.7 |
| [#71](https://github.com/meizhong986/WhisperJAV/issues/71) | Google Translate (free) | Open | Fragile unofficial API. | v1.9+ |
| [#69](https://github.com/meizhong986/WhisperJAV/issues/69) | Grok translation | Open | Covered by custom provider. | Defer |
| [#59](https://github.com/meizhong986/WhisperJAV/issues/59) | Feature plans for 1.x | Open | Meta-issue. | Keep |
| [#51](https://github.com/meizhong986/WhisperJAV/issues/51) | Batch translate wildcard | Open | Glob/directory in translate CLI. | v1.8.7 |
| [#49](https://github.com/meizhong986/WhisperJAV/issues/49) | Output SRT to source folder | Open | `--output-dir source` exists. UX refinement. | v1.8.7 |
| [#43](https://github.com/meizhong986/WhisperJAV/issues/43) | DeepL provider | Open | Non-LLM adapter. | v1.9+ |
| [#33](https://github.com/meizhong986/WhisperJAV/issues/33) | Linux pyaudio dependency | Open | Documentation gap. macOS docs now updated. | v1.8.7 |

---

### Questions / Support / Needs Info

| # | Title | Status | Action |
|---|-------|--------|--------|
| [#197](https://github.com/meizhong986/WhisperJAV/issues/197) | Installation problem v1.8.6 | Need actual error | Ask for traceback beyond the warning |
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate | Stale | User should upgrade to v1.8.6 |
| [#185](https://github.com/meizhong986/WhisperJAV/issues/185) | v1.8.4+ regression | Fixed hotfix2 | Same user as #187 |
| [#161](https://github.com/meizhong986/WhisperJAV/issues/161) | Colab translate error | Stale since Feb 7 | Close as stale |

---

## v1.8.6 Release Summary

**Released 2026-03-04** | 17 commits on branch `v156-settings-persistence` | [Full changelog](https://github.com/meizhong986/WhisperJAV/compare/v1.8.5...v1.8.6)

| Commit | Issues | Summary |
|--------|--------|---------|
| `c2310b3` | #96 (partial) | GUI settings persistence, CLI quality knobs, longest merge strategy |
| `eb80105` | #96 (partial) | Settings hardening: backup rotation, atomic writes |
| `11b7d40` | — | Ensemble parameter presets: save, load, delete |
| `5b976bf` | — | Preset UX: cross-pipeline loading, badge names |
| `191ed49` | — | Anime-Whisper generator with ChronosJAV branding |
| `c58af04` | — | Fix AnimeWhisperGenerator crash |
| `c75b992` | — | Rewrite AnimeWhisperGenerator: low-level API |
| `da2a5c0` | — | Fix anime-whisper: drop forced max_new_tokens |
| `8bb5cb0` | **#179** | `--ensemble-serial`: per-file serial processing |
| `43f60d1` | — | Anime-whisper defaults: greedy decoding, TEN VAD |
| `68d1644` | **#190** | Process-wide UTF-8 mode for non-UTF-8 Windows |
| `24faa3b` | **#183** | Auto-cap batch size for local LLM context window |
| `5769688` | **#194** | M4B audiobook file support |
| `b7e794f` | **#143** | VTT output format: `--output-format srt/vtt/both` |
| `2e402f3` | **#183** | Linux/macOS upgrade tool cross-platform support |
| + | **#188** | google-api-core dependency for Gemini on Linux |
| + | — | TEN VAD tuning, Kotoba v2.0/v2.1, banner fix |

---

## v1.8.7 Planning — Recommended Priorities

| Priority | Issue | Description | Effort |
|----------|-------|-------------|--------|
| **High** | #196 | Local LLM batch cap too generous — reduce to ~60% of context window | Small (formula tweak) |
| **Medium** | #195 | FFmpeg metadata UTF-8 crash — add `errors='replace'` to audio_extraction.py | Small (1 line + guard) |
| **Medium** | #197 | Investigate if real failure behind urllib3 warning | Triage (need info) |
| Medium | #159 | `--vad-threshold` CLI override | Small |
| Medium | #99 | VRAM check + model suggestion | Medium |
| Medium | #51 | Batch translate wildcard | Medium |
| Low | #164 | MPEG-TS auto-remux | Small |
| Low | #49 | Output folder UX | Small |
| Low | #44 | Drag-drop path resolution | Small |
| Low | #33 | Linux pyaudio docs (macOS already updated) | Small (docs) |

**Recommended v1.8.7 focus:** Fix #196 (high — user regression) and #195 (clear root cause, easy fix), triage #197, then ship the small feature requests (#159, #51).

---

## Duplicate / Related Issue Clusters

| Cluster | Issues | Status |
|---------|--------|--------|
| **Windows encoding (GBK/cp950/UTF-8)** | ~~#190~~, ~~#186~~, ~~#177~~ | **All fixed v1.8.6.** |
| **Local LLM translation** | ~~#183~~, **#196**, ~~#162~~, ~~#149~~, ~~#148~~, ~~#146~~ | #196 is residual of #183. Auto-cap not aggressive enough. |
| **FFmpeg metadata/encoding** | **#195** | New. Non-UTF-8 metadata in M4B→M4A files. |
| **Dependency warnings** | **#197**, ~~#193~~ | urllib3/chardet version warning. Harmless but confusing. |
| **GUI settings persistence** | #96, ~~#174~~, ~~#184~~, ~~#176~~ | #96 ongoing (v1.9). Rest closed. |
| **Ensemble UX** | ~~#179~~, ~~#189~~, ~~#191~~ | All resolved in v1.8.6. |
| **VAD parameter crashes** | ~~#173~~, ~~#185~~, ~~#187~~ | Resolved in hotfix2. |
| **AMD/non-NVIDIA GPU** | #142, #114 | Deferred to v1.9+. |
| **CLI batch workflow** | #49, #51, #126 | #49/#51 for v1.8.7. #126 deferred. |
| **i18n** | #180, #175 | Deferred to v1.9+. |

---

## v1.9+ (Deferred)

| # | Issue | Reason |
|---|-------|--------|
| #96 | Full GUI settings persistence (all tabs) | Major UX redesign |
| #180/#175 | Multi-language GUI (i18n) | Full i18n framework — high effort |
| #114/#142 | DirectML / ROCm for AMD/Intel | Major platform enablement |
| #126 | Recursive directory + mirror output | Medium effort |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL provider | Non-LLM adapter complexity |
| #71 | Google Translate (no API key) | Fragile unofficial APIs |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
