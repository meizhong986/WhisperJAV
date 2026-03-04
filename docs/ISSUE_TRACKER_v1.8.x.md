# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-03 (evening) — v1.8.5-hotfix3 RELEASED | v1.8.6 dev in progress (14 commits) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | 40 open on GitHub

---

## Quick Stats

| Category | Count |
|----------|-------|
| Open on GitHub | 40 |
| Closed this cycle (on GitHub) | 37+ |
| **Closeable now** (fix confirmed or answered) | 14 |
| **Reopened / still broken** | 0 (#190 and #183-secondary now fixed in v1.8.6 dev) |
| Fixed in hotfix3 (released 2026-03-01) | 6 (#190, #178, #158/#183, #192, #166-partial) |
| Fixed in hotfix2 (released) | 7 (#189, #185, #187, #173, #176, #188, #178-partial) |
| Fixed in hotfix1 (released) | 3 (#186, #177, #143/#150/#69) |
| **NEW issues** (since last update) | 2 (#193, #194) |
| Active bugs (need work) | 1 (#187-unclear) |
| Feature requests (open) | 18 |
| Questions / docs / support | 5 |
| Needs more info | 3 |
| Deferred (v1.9+) | 8 |

---

## CRITICAL: Issues Requiring Immediate Attention

### #190 — GBK Codec Error — FIXED in v1.8.6 dev (commit `68d1644`)

**Status:** ~~User MatteoHugo confirmed on 2026-03-02 that the error persisted after hotfix3.~~ **Now fixed:** commit `68d1644` forces UTF-8 mode for the entire Python process on non-UTF-8 Windows systems. This sets `PYTHONIOENCODING=utf-8` process-wide, which covers both WhisperJAV's own subprocess calls AND PySubtrans's internal encoding. Awaiting user verification.

**Previous status:** Hotfix3 only fixed WhisperJAV's subprocess encoding. The error originated in PySubtrans's own I/O path. The new fix covers the entire process tree.

### #183 — Ubuntu LLM Translation Fails (Secondary) — FIXED in v1.8.6 dev (commit `24faa3b`)

**Status:** ~~Translation failed with "API token limit" + "No matches".~~ **Now fixed:** commit `24faa3b` auto-caps batch size when the local LLM's context window is small. The root cause was PySubtrans sending batches larger than the model's context could handle. The fix detects the model's max context and reduces batch_size accordingly. Also: the upgrade tool now works on Linux (commit `2e402f3`).

---

## GitHub Housekeeping — Issues to Close

| # | Title | Reason | Action |
|---|-------|--------|--------|
| [#191](https://github.com/meizhong986/WhisperJAV/issues/191) | Pass2 missing (Chinese) | **Answered** — SSL cert error, not a bug. User on v1.8.3. | Close with comment |
| [#193](https://github.com/meizhong986/WhisperJAV/issues/193) | How to update packages | **Support question** — urllib3/chardet version warning. Not a bug. | Answer + close |
| [#188](https://github.com/meizhong986/WhisperJAV/issues/188) | Unknown provider: Gemini | **Answered** — missing `google-genai` SDK, not a code bug. | Close with install instructions |
| [#150](https://github.com/meizhong986/WhisperJAV/issues/150) | Xiaomi Mimo API | **Done** — `custom` provider in hotfix1 covers this. | Close |
| [#162](https://github.com/meizhong986/WhisperJAV/issues/162) | Local model translation (ggml.dll) | **Fixed** in v1.8.4. User hasn't retested. | Close with upgrade note |
| [#146](https://github.com/meizhong986/WhisperJAV/issues/146) | Local server error | **Fixed** in v1.8.3/v1.8.4. Stale 5+ weeks. | Close |
| [#174](https://github.com/meizhong986/WhisperJAV/issues/174) | GUI settings reset | **Duplicate** of #96. Translation part fixed in hotfix2. | Close as dup of #96 |
| [#184](https://github.com/meizhong986/WhisperJAV/issues/184) | GUI save configurations | **Duplicate** of #96/#174. | Close as dup of #96 |
| [#186](https://github.com/meizhong986/WhisperJAV/issues/186) | UnicodeEncodeError subprocess | **Fixed** in hotfix1. No user follow-up. | Close with note |
| [#177](https://github.com/meizhong986/WhisperJAV/issues/177) | cp950 codec AI translation | **Fixed** in hotfix1. No user follow-up. | Close with note |

---

## Closed This Cycle (on GitHub)

### Newly Closed Since Last Update (2026-03-01 → 2026-03-03)

| # | Title | Resolution | Date Closed |
|---|-------|------------|-------------|
| [#192](https://github.com/meizhong986/WhisperJAV/issues/192) | Upgrade to latest dev failed | **Fixed hotfix3** — PEP 508 syntax. | 2026-03-02 |
| [#172](https://github.com/meizhong986/WhisperJAV/issues/172) | Ranking AI translation | Answered — community info. | 2026-03-01 |
| [#170](https://github.com/meizhong986/WhisperJAV/issues/170) | Which modes for highest accuracy | Answered — community info. | 2026-03-01 |
| [#166](https://github.com/meizhong986/WhisperJAV/issues/166) | Kaggle fail to run | CLI args fixed in hotfix3. | 2026-03-02 |
| [#163](https://github.com/meizhong986/WhisperJAV/issues/163) | Sync issue (~100ms ahead) | Fixed — timestamp architecture. | 2026-03-01 |
| [#160](https://github.com/meizhong986/WhisperJAV/issues/160) | Backend performance rationale | Answered — technical explanation. | 2026-03-01 |
| [#152](https://github.com/meizhong986/WhisperJAV/issues/152) | Kaggle incomplete pass1 | Addressed. | 2026-03-01 |
| [#137](https://github.com/meizhong986/WhisperJAV/issues/137) | Can't detect PyTorch | Addressed. | 2026-03-01 |
| [#134](https://github.com/meizhong986/WhisperJAV/issues/134) | Colab setup error | Addressed. | 2026-03-01 |
| [#132](https://github.com/meizhong986/WhisperJAV/issues/132) | Local LLM in Colab | Addressed. | 2026-03-01 |
| [#125](https://github.com/meizhong986/WhisperJAV/issues/125) | Error 3221226505 | Fixed — os._exit. | 2026-03-01 |
| [#118](https://github.com/meizhong986/WhisperJAV/issues/118) | OOM 4GB VRAM | Addressed. | 2026-03-01 |
| [#108](https://github.com/meizhong986/WhisperJAV/issues/108) | macOS pywebview | Addressed. | 2026-03-01 |

### Previously Closed This Cycle

| # | Title | Resolution | Date Closed |
|---|-------|------------|-------------|
| [#173](https://github.com/meizhong986/WhisperJAV/issues/173) | TEN VAD TypeError | **Fixed hotfix2.** | 2026-02-25 |
| [#169](https://github.com/meizhong986/WhisperJAV/issues/169) | Colab not working (SOLVED) | Self-resolved. | 2026-02-25 |
| [#167](https://github.com/meizhong986/WhisperJAV/issues/167) | Qwen-ASR hang | Fixed v1.8.5 timestamp arch. | 2026-02-25 |
| [#157](https://github.com/meizhong986/WhisperJAV/issues/157) | Binary compat (llama-cpp) | Fixed — AVX2 detection. | 2026-02-20 |
| [#153](https://github.com/meizhong986/WhisperJAV/issues/153) | Standalone Translator tab | **Shipped v1.8.4.** | 2026-02-18 |
| [#165](https://github.com/meizhong986/WhisperJAV/issues/165) | Windows Defender flags exe | User communication. | 2026-02-18 |
| [#168](https://github.com/meizhong986/WhisperJAV/issues/168) | ZipEnhancer setup | Answered. | 2026-02-18 |
| [#171](https://github.com/meizhong986/WhisperJAV/issues/171) | Fresh install 1.8.4 | User confirmed. | 2026-02-19 |
| [#149](https://github.com/meizhong986/WhisperJAV/issues/149) | Local server timeout | Fixed v1.8.3. | 2026-02-27 |
| [#148](https://github.com/meizhong986/WhisperJAV/issues/148) | Local LLM 502 errors | Fixed v1.8.3. | 2026-02-27 |
| [#103](https://github.com/meizhong986/WhisperJAV/issues/103) | Cache cleanup | User confirmed. | 2026-02-25 |
| And ~13 more earlier in cycle (see previous tracker version for full list) | | | |

---

## Active Issues — Master Table

### Bugs

| # | Title | Status | Summary | Rec | Target |
|---|-------|--------|---------|-----|--------|
| [#190](https://github.com/meizhong986/WhisperJAV/issues/190) | GBK codec crash (Chinese translation) | **FIXED** (v1.8.6 dev) | Process-wide UTF-8 mode (`68d1644`). Covers PySubtrans internal encoding. Awaiting user verification. | **Close when confirmed** | v1.8.6 |
| [#183](https://github.com/meizhong986/WhisperJAV/issues/183) | Ubuntu LLM translation fails | **FIXED** (v1.8.6 dev) | Auto-cap batch size for small context LLMs (`24faa3b`). Upgrade tool Linux fix (`2e402f3`). | **Close when confirmed** | v1.8.6 |
| [#189](https://github.com/meizhong986/WhisperJAV/issues/189) | Smart Merge clears one pass | Fixed hotfix2 | Quality-aware tiebreaker. Awaiting retest. | Close when confirmed | — |
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate subtitles | **UNCLEAR** | User weifu8435 claims on hotfix2. Last exchange: owner sent exe link. No new error logs. Possibly user confusion. | Info-needed | — |
| [#185](https://github.com/meizhong986/WhisperJAV/issues/185) | v1.8.4+ regression (VAD params) | Fixed hotfix2 | Same root cause as #173. User is same as #187 (weifu8435). | Dup of #187 | — |
| [#178](https://github.com/meizhong986/WhisperJAV/issues/178) | Custom endpoint → /responses 404 | Fixed hotfix3 | Custom Server bypass. Upstream bug filed. | Close when confirmed | — |
| [#158](https://github.com/meizhong986/WhisperJAV/issues/158) | Linux LLM build regression | Fixed hotfix3 | Import path fixed. | Close | — |
| [#44](https://github.com/meizhong986/WhisperJAV/issues/44) | GUI drag-drop filename only | Open | Old. Drag-drop sends filename not path. | Accept | v1.8.6+ |
| [#90](https://github.com/meizhong986/WhisperJAV/issues/90) | Source build dep conflict | Open | Dependency version conflicts. | Accept | v1.8.6+ |

---

### Feature Requests

| # | Title | Status | Summary | Rec | Target |
|---|-------|--------|---------|-----|--------|
| [#179](https://github.com/meizhong986/WhisperJAV/issues/179) | Per-video pass ordering | **DONE** | Implemented as `--ensemble-serial` (commit `8bb5cb0`). Each file completes Pass1→Pass2→Merge before next. GUI checkbox "Finish each file" in merge row. | **Close** | v1.8.6 |
| [#194](https://github.com/meizhong986/WhisperJAV/issues/194) | M4B file support | **DONE** | Added `.m4b` to allowed extensions (`5769688`). | **Close** | v1.8.6 |
| [#96](https://github.com/meizhong986/WhisperJAV/issues/96) | Full GUI settings persistence | Open | Core request. Translation part done (hotfix2). Pipeline/ensemble settings need serialization framework. | Accept | v1.9 |
| [#181](https://github.com/meizhong986/WhisperJAV/issues/181) | Frameless window / hide title bar | Open | Cosmetic. PyWebView `frameless=True`. | Defer | v1.9+ |
| [#180](https://github.com/meizhong986/WhisperJAV/issues/180) | Multi-language GUI (i18n) | Open | Full i18n framework. High effort. | Defer | v1.9+ |
| [#175](https://github.com/meizhong986/WhisperJAV/issues/175) | i18n Chinese | Open | Dup of #180. | Defer | v1.9+ |
| [#164](https://github.com/meizhong986/WhisperJAV/issues/164) | Format compat + Google Drive | Open | (1) MPEG-TS remux. (2) Google Drive Kaggle. | Accept (1) | v1.8.6 |
| [#159](https://github.com/meizhong986/WhisperJAV/issues/159) | CLI --vad-threshold override | Open | Simple float CLI arg. | Accept | v1.8.6+ |
| [#143](https://github.com/meizhong986/WhisperJAV/issues/143) | Custom local models + VTT | **DONE** | Ollama done (custom provider). VTT output done (`b7e794f`) — `--output-format srt/vtt/both`, GUI dropdown. | **Close** | v1.8.6 |
| [#142](https://github.com/meizhong986/WhisperJAV/issues/142) | AMD Radeon GPU detection | Open | Requires DirectML/ROCm. | Defer | v1.9+ |
| [#126](https://github.com/meizhong986/WhisperJAV/issues/126) | Recursive directory processing | Open | Walk subdirectories. | Defer | v1.9+ |
| [#114](https://github.com/meizhong986/WhisperJAV/issues/114) | DirectML for AMD/Intel | Open | Major platform enablement. | Defer | v1.9+ |
| [#99](https://github.com/meizhong986/WhisperJAV/issues/99) | 4GB VRAM solutions | Open | VRAM check + model suggestion. Related to #118 (closed). | Accept | v1.8.6 |
| [#71](https://github.com/meizhong986/WhisperJAV/issues/71) | Google Translate (free) | Open | Fragile unofficial API. | Defer | v1.9+ |
| [#69](https://github.com/meizhong986/WhisperJAV/issues/69) | Grok translation | Open | Partially covered by `custom` provider. | Defer | — |
| [#51](https://github.com/meizhong986/WhisperJAV/issues/51) | Batch translate wildcard | Open | Glob/directory in translate CLI. | Accept | v1.8.6 |
| [#49](https://github.com/meizhong986/WhisperJAV/issues/49) | Output SRT to source folder | Open | `--output-dir source` already works. May need UX refinement. | Accept | v1.8.6 |
| [#43](https://github.com/meizhong986/WhisperJAV/issues/43) | DeepL provider | Open | Non-LLM adapter. | Defer | v1.9+ |

---

### Questions / Documentation / Support

| # | Title | Status | Rec |
|---|-------|--------|-----|
| [#193](https://github.com/meizhong986/WhisperJAV/issues/193) | How to update packages | NEW — urllib3/chardet version warning. Not a bug. | Answer + close |
| [#161](https://github.com/meizhong986/WhisperJAV/issues/161) | Colab translate error | Stale. Screenshot-only. | Close as stale |
| [#33](https://github.com/meizhong986/WhisperJAV/issues/33) | Linux pyaudio dependency docs | Old documentation gap. | Accept (docs) |
| [#59](https://github.com/meizhong986/WhisperJAV/issues/59) | Feature plans for 1.x | Meta-issue. Keep open. | Keep |
| [#99](https://github.com/meizhong986/WhisperJAV/issues/99) | 3050/4GB VRAM feasibility | Q&A + feature overlap with VRAM check. | Keep |

---

### Needs More Info

| # | Title | Action Needed |
|---|-------|---------------|
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate | Confirm user has hotfix2/hotfix3 exe. Request debug log with `--debug`. |
| [#185](https://github.com/meizhong986/WhisperJAV/issues/185) | v1.8.4+ regression | Same user as #187. Likely resolves together. |
| [#161](https://github.com/meizhong986/WhisperJAV/issues/161) | Colab translate error | Stale since Feb 7. No useful info. |

---

## Hotfix Tracking

### v1.8.5-hotfix1 (3 commits, RELEASED)

| Commit | Issues | Summary |
|--------|--------|---------|
| `8fe643a` | **#186, #177** | UTF-8 subprocess encoding (13 calls) |
| `e783723` | **D#101** | Language-neutral instruction prompts |
| `404176b` | **#143, #150, #69, D#131** | Generic `custom` provider + `os.getenv(None)` crash fix |

### v1.8.5-hotfix2 (6 commits, RELEASED)

| Commit | Issues | Summary |
|--------|--------|---------|
| `5b6d79e` | **#185, #187, #173** | Factory-level param validation gate for all speech segmenters |
| `d00bfb8` | **#189** | Smart Merge quality-aware text tiebreaker |
| `8d0deae` | **#178 (partial), #188, #176** | URL normalization, provider error hints, settings persistence |
| `77c1a31` | (internal) | Ensemble multi-file output dir fix |
| `24b31cb` | (internal) | Merge fallback metadata fix |
| `8fa8ca3` | (cosmetic) | EOF newline fix |

### v1.8.5-hotfix3 (10 commits, RELEASED 2026-03-01)

| Commit | Issues | Summary |
|--------|--------|---------|
| `4d9b2a9` | **#190** (partial) | GBK codec fix for WhisperJAV subprocess — **PySubtrans-side error persists** |
| `19f6785` | (optimization) | Enhancement passthrough: skip I/O when enhancer is "none" |
| `f35b17a` | (housekeeping) | Notebook consolidation: rename/delete/redirect |
| `c538330` | **#166 (partial), #132** | Fix 4 fatal CLI arg bugs in Kaggle parallel notebook + Colab expert fixes |
| `55635a8` | (docs) | README: add PySubtrans acknowledgment, fix stale badges |
| `3abbfcb` | **#158, #183** | Fix Linux llama_build_utils import regression |
| `4cc4425` | **#178** | Fix PySubtrans /responses routing — Custom Server bypass for non-GPT providers |
| `c33ec97` | **#192** | Fix whisperjav-upgrade egg fragment error — PEP 508 syntax |
| `9f6c3b1` | (version) | Set display_label to hotfix3 |
| `552231c` | (version) | Fix VERSION for build, add acceptance tests |

---

## v1.8.6 Development (in progress, branch `v156-settings-persistence`)

| Commit | Issues | Summary |
|--------|--------|---------|
| `c2310b3` | **#96 (partial)** | GUI settings persistence (initial), CLI quality knobs, longest merge strategy |
| `eb80105` | **#96 (partial)** | Harden settings: backup rotation, atomic writes, 55 unit tests |
| `11b7d40` | — | Ensemble parameter presets: save, load, delete named configs |
| `5b976bf` | — | Preset UX: cross-pipeline loading, badge names, persistence |
| `191ed49` | — | Anime-Whisper generator backend with ChronosJAV GUI branding |
| `c58af04` | — | Fix AnimeWhisperGenerator crash |
| `c75b992` | — | Rewrite AnimeWhisperGenerator: low-level API |
| `da2a5c0` | — | Fix anime-whisper: drop forced max_new_tokens |
| `8bb5cb0` | **#179** | **--ensemble-serial**: per-file serial ensemble processing |
| `43f60d1` | — | Fix anime-whisper defaults: greedy decoding, safe token cap, TEN VAD |
| `68d1644` | **#190** | **Force UTF-8 mode** for entire process on non-UTF-8 Windows |
| `24faa3b` | **#183** (secondary) | **Auto-cap batch size** for local LLM context window |
| `5769688` | **#194** | **M4B audiobook file support** — add .m4b to allowed extensions |
| `b7e794f` | **#143** | **VTT output format** — `--output-format srt/vtt/both`, GUI dropdown, all pipelines |
| `2e402f3` | **#183** (upgrade) | **Linux upgrade tool** — cross-platform detection, pip helpers, platform messages |

---

## Roadmap Summary

### v1.8.5 — RELEASED (base + 3 hotfixes, 19 commits)

All hotfixes released. 15+ issues fixed. Major themes: encoding, VAD validation, settings persistence, PySubtrans routing.

### v1.8.6 — Next Feature Release (in development)

| Priority | Items | Status |
|----------|-------|--------|
| **Critical** | #190 GBK codec — process-wide UTF-8 | **DONE** (commit `68d1644`) |
| **Critical** | #183 LLM translation — auto-cap batch size | **DONE** (commit `24faa3b`) |
| **High** | #179 Per-video serial ensemble | **DONE** (commit `8bb5cb0`) |
| **High** | #96 GUI settings persistence (initial framework) | **IN PROGRESS** (commits `c2310b3`, `eb80105`) |
| High | Anime-Whisper generator backend | **DONE** (4 commits) |
| High | Ensemble parameter presets | **DONE** (2 commits) |
| Medium | #194 M4B file support | **DONE** (commit `5769688`) |
| Medium | #51 Batch translate wildcard | Glob support in translate CLI |
| Medium | #143 VTT output format | **DONE** (commit `b7e794f`) |
| Medium | #99 VRAM check + model suggestion | Auto-detect + warn |
| Low | #159 --vad-threshold CLI override | Simple float arg |
| Low | #164 MPEG-TS auto-remux | FFmpeg format probe |
| Low | #44 Drag-drop path resolution | GUI fix |
| Low | #49 Output folder UX | Refinement (source mode exists) |

### v1.9+ (Deferred)

| # | Issue | Reason |
|---|-------|--------|
| #96 | Full GUI settings persistence (all tabs) | Major UX redesign — file-backed state for every control |
| #180/#175 | Multi-language GUI (i18n) | Full i18n framework — high effort |
| #114/#142 | DirectML / ROCm for AMD/Intel | Major platform enablement |
| #126 | Recursive directory + mirror output | Medium effort |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL provider | Non-LLM adapter complexity |
| #71 | Google Translate (no API key) | Fragile unofficial APIs |

---

## Duplicate / Related Issue Clusters

| Cluster | Issues | Status |
|---------|--------|--------|
| **Windows encoding (GBK/cp950/UTF-8)** | ~~#190~~, ~~#186~~, ~~#177~~ | **All fixed.** #190 process-wide UTF-8 (`68d1644`). |
| **PySubtrans bugs** | ~~#190~~, ~~#178~~, ~~#183-secondary~~ | All addressed. Encoding, routing, batch size all fixed. |
| **ggml.dll / llama-cpp** | ~~#162~~, ~~#149~~, ~~#148~~, ~~#157~~, ~~#146~~, ~~#139~~ | **Fully resolved.** All closeable. |
| **VAD parameter type crashes** | ~~#173~~, ~~#185~~, ~~#187~~ | **Fully resolved in hotfix2.** |
| **GUI settings persistence** | #96, ~~#174~~, ~~#184~~, ~~#176~~ | #176 done. #96 in progress (v1.8.6 dev). #174/#184 are dups. |
| **Ensemble UX** | ~~#179~~, ~~#189~~, ~~#191~~ | #179 DONE (serial mode). #189 fixed. #191 answered (SSL). |
| **Colab/Kaggle** | ~~#166~~, ~~#152~~, ~~#134~~, ~~#132~~, ~~#161~~ | Mostly closed. #161 stale. |
| **macOS** | ~~#108~~, ~~#155~~, ~~#100~~, ~~#110~~ | All closed. |
| **Local LLM server** | ~~#149~~, ~~#148~~, ~~#158~~, ~~#183~~ (import part) | Import issues all fixed. Translation quality (#183) separate. |
| **AMD/non-NVIDIA GPU** | #142, #114 | Deferred to v1.9+. |
| **CLI batch workflow** | #49, #51, #126 | #49/#51 for v1.8.6. #126 deferred. |
| **Translation provider** | ~~#178~~, ~~#188~~, ~~#143~~, ~~#150~~, #69 | Most fixed. #69 covered by custom provider. |

---

## GitHub Discussions (10 open)

| # | Title | Notes |
|---|-------|-------|
| D#24 | Welcome | Pinned |
| D#101 | LLM prompt slot suggestion | Instruction prompts shipped in hotfix1 |
| D#121 | Kaggle support suggestion | Kaggle notebook shipped |
| D#131 | Local Models thank you | Custom provider covers this |
| D#133 | --instructions-file override? | Documentation question |
| D#145 | Local LLM discussion | General discussion |
| D#147 | AMD GPU discussion | Deferred |
| D#154 | Initial prompt to reduce hallucination | Feature consideration |
| D#156 | --max-workers CPU flag | Documentation question |
| D#182 | Can't find whisperjav in folder | Installation support |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
