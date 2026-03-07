# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-07 — v1.8.7b0 pre-release ready | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | 41 open on GitHub

---

## Quick Stats

| Category | Count |
|----------|-------|
| Open on GitHub | 41 |
| **Fixed in v1.8.7b0** | 3 (#196, #198 Bug1, #159) |
| **Active bugs (need work)** | 2 (#195 already fixed, #197 awaiting info) |
| **New since v1.8.6** | 4 (#195, #196, #197, #198) |
| **Closeable now** (fix shipped or answered) | 17 |
| Fixed in v1.8.6 (released 2026-03-04) | 5 (#190, #183, #179, #194, #143) |
| Fixed in hotfix3 | 6 | Fixed in hotfix2 | 7 |
| Fixed in hotfix1 | 3 |
| Feature requests (open) | 17 |
| Questions / docs / support | 4 |
| Deferred (v1.9+) | 8 |

---

## NEW: Post-v1.8.6 Issues

### #198 — Transformers mode: no subtitles, MPS not used on M1 Mac

**Reporter:** francetoastVN | **Opened:** 2026-03-06 | **Comments:** 2

**Symptom:** Two distinct problems on M1 Pro MacBook:
1. Transformers pipeline runs on CPU — MPS (Metal) not used
2. Zero subtitles produced after 53 minutes of processing

**Root cause analysis (confirmed in code):**

**Bug 1 — CPU not MPS:** `TransformersASR._detect_device()` only checked CUDA → CPU. MPS (`torch.backends.mps.is_available()`) was never checked. On Apple Silicon, `auto` always fell through to `"cpu"`.

**Bug 2 — 0 subtitles:** `kotoba-whisper-bilingual-v1.0` with `task='translate'` failed to produce segment timestamps (`return_timestamps=True`). The HF pipeline returned 1 single text blob in Japanese (not English) with no boundaries. The post-processor discarded it → 0 subtitles. This is a model+task mismatch: kotoba-bilingual does not reliably honour the Whisper translate task.

**Fix shipped:** `transformers_asr.py` `_detect_device()` now checks MPS after CUDA miss; explicit `mps` device request also handled; `_detect_dtype()` returns `float16` for MPS (bfloat16 unsupported). Commit `10fbf30`.

**Remaining (Bug 2):** Not a code bug — model limitation. Documented in GitHub response. Workaround: use `--subs-language native` + separate `--translate`, or use `openai/whisper-large-v3`.

**Status:** Bug 1 fixed v1.8.7-beta. Bug 2 is a known model limitation. | **Priority:** Medium

---

### #197 — Worker crash (exit code 0xC0000005) — needs GPU info

**Reporter:** KenZP12 | **Opened:** 2026-03-06 | **Comments:** 5

**Symptom:** Process fails with `urllib3` warning at startup and worker crash during ASR.

**Analysis:** The urllib3/chardet warning is a red herring — purely cosmetic version mismatch.

**Real failure from log:**
```
21:09:08 - FasterWhisperProASR initialized with task='transcribe'
21:09:19 - Pass 1 worker died with exit code 3221225477
```
Exit code `3221225477` = `0xC0000005` = Windows `STATUS_ACCESS_VIOLATION` — native C crash, uncatchable by Python try/except. Crash occurs 11 seconds into `WhisperModel('large-v2', device='cuda', compute_type='auto')` loading. "Worker produced no Drop-Box" confirms the crash bypassed all Python error handlers.

**Command used:** `--pass1-model large-v2 --pass1-pipeline balanced --pass1-speech-segmenter silero-v6.2`

**Root cause:** Unknown — awaiting `nvidia-smi` output from user. Asked in GitHub comment. Hypothesis: GPU VRAM insufficient for large-v2 causing ctranslate2 to crash native rather than throw a catchable Python exception. Cannot confirm without GPU specs.

**Status:** Awaiting user response. | **Priority:** Medium

---

### #196 — Local LLM translation: token limit + "No matches found"

**Reporter:** destinyawaits | **Opened:** 2026-03-05 | **Comments:** 2 (zhstark confirmed same on Ubuntu 22.04)

**Symptom:**
1. "Hit API token limit, retrying batch without context..."
2. "No matches found in translation text using patterns"

**Details:** Model: `gemma-2-9b-it-abliterated-Q4_K_M.gguf`, 8192-token context. v1.8.6 auto-cap reduced batch 30→17, but still fails.

**Root cause (analyzed):** The v1.8.6 auto-cap formula (`cap_batch_size_for_context()` in `translate/core.py`) only accounts for **input tokens**. It ignores that `n_ctx` is **shared between input AND output** in llama-cpp-python. For Japanese→Chinese translation, output costs ~150-200 tokens/line on top of input. True safe capacity for 8K context is ~11 lines (not 17).

**Failure chain:** `finish_reason=="length"` → PySubtrans retries without context → retry ALSO hits token limit → `ProcessBatchTranslation` receives truncated output → `TranslationParser` tries 7 regex patterns → all fail → "No matches found".

**Fix needed:** Revise `cap_batch_size_for_context()` to account for combined input+output token cost. Current formula: `(n_ctx - 2500) // 500 = 11 for 8K`. This is already the tighter v1.8.7 formula (overhead=2500, per_line=500). Root problem may be the retry path not re-checking the limit.

**Workaround posted:** Reduce batch size manually in GUI settings (screenshot provided in issue).

**Status:** Fix in progress. | **Priority:** High

---

### #195 — UnicodeDecodeError + AttributeError during "Transforming audio"

**Reporter:** v2lmmj04 | **Opened:** 2026-03-03 | **Comments:** 2

**Symptom:**
1. `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 2382`
2. `AttributeError: 'NoneType' object has no attribute 'split'`

**Root cause (confirmed):** M4A files with Japanese metadata. FFmpeg outputs non-UTF-8 bytes to stderr. `audio_extraction.py` uses `subprocess.run(..., text=True, encoding='utf-8')` without `errors='replace'` → decode crash → `result.stderr` is `None` → `AttributeError` on `.split()`.

**Fix:** Add `errors='replace'` to subprocess call + null guard on `result.stderr`.

**User workaround:** Strip metadata: `ffmpeg -i input.m4b -map_metadata -1 -vn -acodec copy output.m4a`

**Status:** Fix ready to implement. | **Priority:** Medium

---

## Issues to Close (fix shipped in v1.8.6 or answered, still open on GitHub)

| # | Title | Reason | Action |
|---|-------|--------|--------|
| [#190](https://github.com/meizhong986/WhisperJAV/issues/190) | GBK codec crash | **Fixed v1.8.6** commit `68d1644` | Close |
| [#194](https://github.com/meizhong986/WhisperJAV/issues/194) | M4B file support | **Done v1.8.6** commit `5769688` | Close |
| [#143](https://github.com/meizhong986/WhisperJAV/issues/143) | VTT output + custom models | **Done v1.8.6** | Close |
| [#179](https://github.com/meizhong986/WhisperJAV/issues/179) | Per-video pass ordering | **Done v1.8.6** `--ensemble-serial` | Close |
| [#183](https://github.com/meizhong986/WhisperJAV/issues/183) | Ubuntu LLM translation | **Partially fixed v1.8.6** — residual tracked in #196 | Close with note |
| [#191](https://github.com/meizhong986/WhisperJAV/issues/191) | Pass2 missing (Chinese) | **Answered** — SSL cert error, not a bug | Close |
| [#193](https://github.com/meizhong986/WhisperJAV/issues/193) | How to update packages | **Support question** — urllib3 warning harmless | Close |
| [#188](https://github.com/meizhong986/WhisperJAV/issues/188) | Unknown provider: Gemini | **Fixed v1.8.6** | Close |
| [#150](https://github.com/meizhong986/WhisperJAV/issues/150) | Xiaomi Mimo API | **Answered** — custom provider covers this | Close |
| [#162](https://github.com/meizhong986/WhisperJAV/issues/162) | Local model ggml.dll | **Fixed v1.8.4** — stale | Close |
| [#146](https://github.com/meizhong986/WhisperJAV/issues/146) | Local server error | **Fixed v1.8.3** — stale 5+ weeks | Close |
| [#174](https://github.com/meizhong986/WhisperJAV/issues/174) | GUI settings reset | **Duplicate** of #96 | Close as dup |
| [#184](https://github.com/meizhong986/WhisperJAV/issues/184) | GUI save configurations | **Duplicate** of #96 | Close as dup |
| [#186](https://github.com/meizhong986/WhisperJAV/issues/186) | UnicodeEncodeError subprocess | **Fixed hotfix1** | Close |
| [#177](https://github.com/meizhong986/WhisperJAV/issues/177) | cp950 codec translation | **Fixed hotfix1 + v1.8.6** | Close |
| [#158](https://github.com/meizhong986/WhisperJAV/issues/158) | Linux LLM build regression | **Fixed hotfix3** | Close |
| [#178](https://github.com/meizhong986/WhisperJAV/issues/178) | Custom endpoint 404 | **Fixed hotfix3** | Close |

**Total closeable: 17**

---

## Active Issues — Master Table

### Bugs

| # | Title | Status | Summary | Priority | Target |
|---|-------|--------|---------|----------|--------|
| [#198](https://github.com/meizhong986/WhisperJAV/issues/198) | Transformers: no MPS + 0 subs | Bug 1 **FIXED** `10fbf30` | MPS fix shipped. 0-subs is model limitation (kotoba+translate). | Medium | v1.8.7 |
| [#197](https://github.com/meizhong986/WhisperJAV/issues/197) | Worker crash 0xC0000005 | **AWAITING INFO** | Native crash during large-v2 load. Asked for nvidia-smi. | Medium | v1.8.7 |
| [#196](https://github.com/meizhong986/WhisperJAV/issues/196) | Local LLM token limit + no matches | **FIXED** `7407ab6` `e5328c8` `090fd43` | 3 root causes fixed: max_tokens cap, stream=True forced, supports_streaming=True in CustomClient. 28/28 E2E tests pass. | **High** | v1.8.7 |
| [#195](https://github.com/meizhong986/WhisperJAV/issues/195) | UnicodeDecodeError audio extraction | **FIXED v1.8.6** `55df512` | `errors='replace'` + null guard already shipped in v1.8.6. | Medium | — |
| [#189](https://github.com/meizhong986/WhisperJAV/issues/189) | Smart Merge clears one pass | Fixed hotfix2 | Quality-aware tiebreaker. | Low | — |
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate subtitles | **STALE** | User should upgrade to v1.8.6. No response to upgrade suggestion. | Low | — |
| [#185](https://github.com/meizhong986/WhisperJAV/issues/185) | v1.8.4+ regression | Fixed hotfix2 | Same user as #187. | Low | — |

---

### Feature Requests

| # | Title | Status | Summary | Target |
|---|-------|--------|---------|--------|
| [#96](https://github.com/meizhong986/WhisperJAV/issues/96) | Full GUI settings persistence | Partial | Translation + ensemble presets done. Full pipeline tab remaining. | v1.9 |
| [#181](https://github.com/meizhong986/WhisperJAV/issues/181) | Frameless window | Open | PyWebView `frameless=True`. Cosmetic. | v1.9+ |
| [#180](https://github.com/meizhong986/WhisperJAV/issues/180) | Multi-language GUI (i18n) | Open | Full i18n framework. High effort. | v1.9+ |
| [#175](https://github.com/meizhong986/WhisperJAV/issues/175) | Chinese language GUI | Open | Subset of #180. | v1.9+ |
| [#164](https://github.com/meizhong986/WhisperJAV/issues/164) | MPEG-TS + Google Drive | Open | MPEG-TS remux + Kaggle Drive integration. | v1.8.7 |
| [#159](https://github.com/meizhong986/WhisperJAV/issues/159) | CLI --vad-threshold | **FIXED** `c092db9` | `--vad-threshold` + `--speech-pad-ms` + per-pass overrides shipped. | v1.8.7 |
| [#142](https://github.com/meizhong986/WhisperJAV/issues/142) | AMD Radeon detection | Open | DirectML/ROCm. | v1.9+ |
| [#126](https://github.com/meizhong986/WhisperJAV/issues/126) | Recursive directory | Open | Walk subdirectories. | v1.9+ |
| [#114](https://github.com/meizhong986/WhisperJAV/issues/114) | DirectML for AMD/Intel | Open | Major platform enablement. | v1.9+ |
| [#99](https://github.com/meizhong986/WhisperJAV/issues/99) | 4GB VRAM guidance | Open | Log GPU VRAM info before model load (diagnostic). | v1.8.7 |
| [#71](https://github.com/meizhong986/WhisperJAV/issues/71) | Google Translate (free) | Open | Fragile unofficial API. | v1.9+ |
| [#69](https://github.com/meizhong986/WhisperJAV/issues/69) | Grok translation | Open | Covered by custom provider already. | Defer/Close |
| [#59](https://github.com/meizhong986/WhisperJAV/issues/59) | Feature plans for 1.x | Open | Meta-issue. | Keep open |
| [#51](https://github.com/meizhong986/WhisperJAV/issues/51) | Batch translate wildcard | Open | Glob/directory in translate CLI. | v1.8.7 |
| [#49](https://github.com/meizhong986/WhisperJAV/issues/49) | Output SRT to source folder | Open | `--output-dir source` exists. UX docs gap. | v1.8.7 |
| [#43](https://github.com/meizhong986/WhisperJAV/issues/43) | DeepL provider | Open | Non-LLM adapter. | v1.9+ |
| [#44](https://github.com/meizhong986/WhisperJAV/issues/44) | GUI drag-drop filename only | Open | Drag-drop sends filename not full path. | v1.8.7+ |
| [#33](https://github.com/meizhong986/WhisperJAV/issues/33) | Linux pyaudio dependency | Open | Documentation gap. macOS docs updated. Linux still missing. | v1.8.7 |

---

### Questions / Support / Needs Info

| # | Title | Status | Action |
|---|-------|--------|--------|
| [#197](https://github.com/meizhong986/WhisperJAV/issues/197) | Worker crash v1.8.6 | Awaiting nvidia-smi | Asked for GPU specs in comment |
| [#187](https://github.com/meizhong986/WhisperJAV/issues/187) | v1.8.5 can't generate | Stale — upgrade suggested | Close if no response |
| [#161](https://github.com/meizhong986/WhisperJAV/issues/161) | Colab translate error | Stale since Feb 7 | Close as stale |

---

## v1.8.7 — Fixes Shipped (branch: dev-1.8.7-beta)

| Commit | Issue | Summary |
|--------|-------|---------|
| `10fbf30` | **#198** | Fix MPS device detection in TransformersASR — Apple Silicon now uses Metal GPU |
| `55df512` | #183, #196, #197 | Batch cap formula tightened (overhead=2500, per_line=500) |

---

## v1.8.7 Planning — Recommended Priorities

### Bug Fixes

| Priority | Issue | Description | Effort | Status |
|----------|-------|-------------|--------|--------|
| **Done** | #198 | MPS fix in TransformersASR | Small | ✅ `10fbf30` |
| **High** | #196 | Local LLM: fix batch cap for combined input+output token cost. Also need to handle the retry-without-context path that re-hits the limit | Small | Pending |
| **Medium** | #195 | `audio_extraction.py`: add `errors='replace'` + null guard on stderr | Small (2 lines) | Pending |
| **Medium** | #197 | Awaiting GPU info from user. If VRAM-related: log GPU VRAM at INFO level before model load (diagnostic, not a decision gate) | Small | Awaiting |

### Enhancements

| Priority | Issue | Description | Effort |
|----------|-------|-------------|--------|
| **Medium** | #99 | Log free/total GPU VRAM at INFO level before `WhisperModel()` loads — gives users diagnostic context when crashes occur without guessing thresholds | Tiny |
| **Medium** | #159 | `--vad-threshold` CLI flag: pass float directly to speech segmenter, overriding sensitivity preset | Small |
| **Low** | #198 (partial) | When HF pipeline returns 0 segments after full processing, emit a specific WARNING with model+task context rather than silently reporting "0 subtitles" | Small |

### Feature Requests (candidates this round)

| Priority | Issue | Description | Effort |
|----------|-------|-------------|--------|
| **Medium** | #164 | MPEG-TS: auto-remux `.ts` files via FFmpeg before processing (common format for broadcast captures) | Small |
| **Medium** | #51 | Batch translate: accept wildcard/directory in `whisperjav-translate`, glob SRT files | Medium |
| **Low** | #49 | Output folder UX: document `--output-dir source` behaviour more clearly; possibly add GUI hint | Tiny (docs) |
| **Low** | #33 | Linux pyaudio install docs: add `apt install portaudio19-dev` note matching macOS docs | Tiny (docs) |

### Defer to v1.9+

- #96 (full GUI persistence), #180/#175 (i18n), #114/#142 (AMD/DirectML), #126 (recursive dir), #181 (frameless window), #43 (DeepL), #71 (free Google Translate)
- #69 (Grok): already covered by custom provider — close with note

---

## Duplicate / Related Issue Clusters

| Cluster | Issues | Status |
|---------|--------|--------|
| **Windows encoding (GBK/cp950/UTF-8)** | ~~#190~~, ~~#186~~, ~~#177~~ | All fixed v1.8.6 |
| **Local LLM translation** | ~~#183~~, **#196**, ~~#162~~, ~~#146~~ | #196 residual — fix pending v1.8.7 |
| **FFmpeg metadata/encoding** | **#195** | Fix pending v1.8.7 |
| **Dependency warnings** | **#197**, ~~#193~~ | urllib3/chardet cosmetic. #197 real issue is worker crash |
| **Apple Silicon / MPS** | **#198** | MPS fix done v1.8.7-beta |
| **GUI settings persistence** | #96, ~~#174~~, ~~#184~~, ~~#176~~ | #96 ongoing (v1.9) |
| **Ensemble UX** | ~~#179~~, ~~#189~~, ~~#191~~ | All resolved v1.8.6 |
| **VAD parameter crashes** | ~~#173~~, ~~#185~~, ~~#187~~ | Resolved hotfix2 |
| **AMD/non-NVIDIA GPU** | #142, #114 | Deferred v1.9+ |
| **CLI batch workflow** | #49, #51, #126 | #49/#51 for v1.8.7, #126 deferred |
| **i18n** | #180, #175 | Deferred v1.9+ |

---

## v1.8.6 Release Summary

**Released 2026-03-04** | [Full changelog](https://github.com/meizhong986/WhisperJAV/compare/v1.8.5...v1.8.6)

| Commit | Issues | Summary |
|--------|--------|---------|
| `c2310b3` | #96 (partial) | GUI settings persistence, CLI quality knobs, longest merge strategy |
| `eb80105` | #96 (partial) | Settings hardening: backup rotation, atomic writes |
| `8bb5cb0` | **#179** | `--ensemble-serial`: per-file serial processing |
| `68d1644` | **#190** | Process-wide UTF-8 mode for non-UTF-8 Windows |
| `24faa3b` | **#183** | Auto-cap batch size for local LLM context window |
| `5769688` | **#194** | M4B audiobook file support |
| `b7e794f` | **#143** | VTT output format: `--output-format srt/vtt/both` |
| `2e402f3` | **#183** | Linux/macOS upgrade tool cross-platform support |
| + | **#188** | google-api-core dependency for Gemini on Linux |
| + | — | TEN VAD tuning, Kotoba v2.0/v2.1, Anime-Whisper generator |

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
