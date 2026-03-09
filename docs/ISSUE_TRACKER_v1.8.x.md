# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-09 (rev2 — full comment sweep) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **25 open** on GitHub

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **25** | Was 19 at last update; 5 new (#205-#209), 1 self-closed (#208) |
| New issues since last update | 5 | #205, #206, #207, #208, #209 |
| Self-closed by reporters | 1 | #208 (LLM server, user installed NVIDIA Toolkit) |
| **Active bugs (need code work)** | 3 | #209 (repetition hallucination), #196 (LLM token limit STILL broken), #132 (Local LLM on Kaggle) |
| **Active bugs (awaiting user confirmation)** | 2 | #204 (SSL fallback, v1.8.7b1 released), #203 (serial mode reply) |
| **New bug reports on CLOSED issues** | 3 | #196 (zhstark: LLM still broken on Ubuntu/v1.8.7b1), #198 (francetoastVN: Mac LLM+Transformers failures), #143 (OdinShiva: Custom+Ollama NoneType) |
| **New duplicates** | 1 | #207 (dup of #96, settings persistence) |
| **New feature requests** | 2 | #205 (VibeVoice ASR), #206 (grey out incompatible options) |
| Fixed in v1.8.7b1 (all committed on main) | 9 | #198-MPS, #195, #204, #159, setuptools pin, zipenhancer GUI, installer diagnostics, China mirror fallback |
| Feature requests (open) | 13 | Added #205, #206 |
| Deferred to v1.9+ | 10 | Added #205, #206 |

### ALERT: Closed Issues With New Bug Reports (2026-03-09)

These 3 closed issues received new comments today reporting **unresolved or new bugs**. They need attention and may need to be reopened:

| # | Reporter | Platform | Error | Key Detail |
|---|----------|----------|-------|------------|
| **#196** | zhstark | Ubuntu CLI, 5090 32GB, v1.8.7b1 | "No matches found in translation text" | Batch auto-reduced to 11 (fix IS working), but still hits token limit. `--max-batch-size 10` doesn't help either. |
| **#198** | francetoastVN | macOS M1, v1.8.7b1 | LLM: "No matches found" / Transformers: no subtitle generated | Two separate failures: (1) Local LLM same token limit issue, (2) Transformers mode produces no output (attached log) |
| **#143** | OdinShiva | Unknown platform | "Connection to custom failed: str expected, not NoneType" | Custom (OpenAI-compatible) provider with Ollama endpoint `http://localhost:11434/v1`, model `qwen3.5:4b`, API Key field **empty** → NoneType error |

---

## Cluster Analysis

Issues are grouped by root cause / theme rather than by number. This reveals the real problem areas.

### Cluster 1: Network / SSL / Model Download Failures (5 issues)

The most impactful cluster for Chinese users. HuggingFace model downloads fail through proxies.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#204** | VPN users (v2rayN) SSL failures | yangming2027 | **OPEN** | HF hub tries online model validation even when cache exists. SSL errors through proxies kill the run. | **FIXED** v1.8.7b1 — 3-step fallback (normal → cache → hf-mirror.com). Awaiting user confirmation. |
| **#201** | Install SSL cert error on fresh Win10 | jl6564 | CLOSED | Missing root CA certs on fresh Windows. | Closed — env issue |
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | CLOSED | Dual-GPU Optimus: NVML can't load in non-GPU context. | Closed — documented |
| ~~#191~~ | Pass2 missing — SSL error | yangming2027 | CLOSED | Same SSL/proxy issue as #204. | Closed as dup of #204 |
| ~~#193~~ | How to update packages | Faraway-D | CLOSED | Support question — urllib3 warning. | Closed — answered |

**Status:** v1.8.7b1 shipped with comprehensive fix. Two comments posted pointing users to the release. Awaiting confirmation from yangming2027 and weifu8435.

---

### Cluster 2: Local LLM Translation (6 issues) — PARTIALLY UNRESOLVED

Recurring theme across versions. Token limits, server timeouts, parsing failures. **The v1.8.7b0/b1 fix did NOT fully resolve #196 — new reports confirm the "No matches found" error persists.**

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#196** | Token limit + "No matches" | destinyawaits | CLOSED | Original: batch too large for 8K context. v1.8.7b1 fix: max_tokens cap + streaming + auto-batch. | **FIX INCOMPLETE** — zhstark reports same error on v1.8.7b1 Ubuntu, 5090 GPU. See details below. |
| **#208** | LLM server AssertionError | Dark-Lord-9 | CLOSED | `llama_cpp` model failed to load — `assert self.model is not None`. User had missing NVIDIA Toolkit. | **Self-resolved** — user installed NVIDIA Toolkit, works now. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | **OPEN** | User reports partial translation success on Kaggle. Says "it used to work previously." Posted new debug_log.txt (2026-03-07) and Kaggle notebook. Uses `--provider local --model gemma-9b`. | **Active — needs investigation of new debug log** |
| ~~#146~~ | Local server error | leefowan | CLOSED | Fixed v1.8.3. Stale 5+ weeks. | Closed — stale |
| ~~#162~~ | Local model ggml.dll | aaxz886 | CLOSED | Fixed v1.8.4. Stale. | Closed — stale |
| ~~#158~~ | Linux LLM build | parheliamm | CLOSED | Fixed hotfix3. | Closed — fixed |

**#196 — New report from zhstark (2026-03-09, Ubuntu, v1.8.7b1, 5090 32GB):**

Screenshots examined. Key observations from the logs:
- Version confirmed: WhisperJAV 1.8.7b1
- Platform: Ubuntu CLI, conda env, Python 3.12
- GPU: 33 layers on CUDA (full offload), 221.7 tokens/sec (very fast 5090)
- Batch auto-reduced from 30 → 11 (auto-batch fix IS working)
- Output token limit: 2692 (JAV/CJK-tuned)
- **Error**: `WARNING: Hit API token limit, retrying batch without context...`
- **Error**: `Error translating scene X batch Y: No matches found in translation text using patterns: [...]`
- User tried `--max-batch-size 10` manually — same error remains
- Target language: Chinese, tone: pornify

**Diagnosis (preliminary — needs verification):**
The auto-batch reduction IS working (30→11), and the max_tokens cap IS applied (2692). But the translation still fails. Two hypotheses:
1. Even batch size 11 produces output exceeding the 8K context with Chinese pornify tone (longer instructions, longer output)
2. The retry-without-context path generates output that doesn't match PySubtrans regex patterns (the LLM response format is garbled or non-standard after the retry)

This means **#196 is NOT fully fixed** and needs further investigation. The v1.8.7b1 fix improved the situation (auto-batch works) but didn't eliminate the root cause.

**#198 — New report from francetoastVN (2026-03-09, macOS M1, v1.8.7b1):**

Two separate failures reported in comments on the closed MPS issue:
1. **Local LLM translation**: Same "No matches found" error as #196. Server starts fine (27.4 tps CPU only), batch size 10, 637 lines in 58 scenes, but immediately fails on scene 1 batch 1.
2. **Transformers mode**: No subtitle generated at all. Attached log file `TransformerTest.txt` (not yet examined).

**Analysis:** #196 is now the most critical open bug. The v1.8.7b1 fix was partial — auto-batch and max_tokens work, but the underlying "No matches found" parsing error still occurs. This affects at least 2 users on different platforms (Ubuntu 5090, macOS M1). #132 is a separate Kaggle-specific issue. #208 was self-resolved.

---

### Cluster 3: GUI Settings Persistence (5 issues)

Users repeatedly report settings lost on restart. Growing cluster.

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#96** | Full settings persistence | sky9639 | OPEN | Translation + ensemble presets done. Full pipeline tab remaining. | **v1.9 — partial** |
| **#207** | 1.86版本不能保存设置 | q864310563 | **OPEN** | "How to save settings in v1.8.6? Every restart resets all to defaults." Windows .exe installer. | **Duplicate of #96** — needs response referencing #96 |
| ~~#176~~ | Translation settings lost | Ywocp | CLOSED | Fixed hotfix2 (file-based persistence). | Closed — fixed |
| ~~#174~~ | Settings reset every time | wazzur1 | CLOSED | Duplicate of #96. | Closed as dup |
| ~~#184~~ | Save configurations request | aikotanaka6699 | CLOSED | Duplicate of #96. | Closed as dup |

**Analysis:** #207 is the 4th duplicate of #96. This keeps coming up — users find it frustrating. Translation and ensemble settings ARE saved (hotfix2+), but pipeline tab settings (model, sensitivity, scene detector, etc.) are not persisted yet. This is now the most frequently reported UX issue. Consider prioritizing for v1.8.7 or v1.8.8 rather than v1.9.

---

### Cluster 4: Encoding / Unicode Crashes (4 issues) — RESOLVED

All stem from non-UTF-8 data in subprocess I/O. **All fixed.**

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#195~~ | UnicodeDecodeError in audio extraction | v2lmmj04 | CLOSED | M4A/M4B with Japanese metadata. Fixed `55df512`. | Closed |
| ~~#190~~ | GBK codec crash | MatteoHugo | CLOSED | Fixed v1.8.6 — process-wide UTF-8 mode. | Closed |
| ~~#186~~ | UnicodeEncodeError subprocess | teijiIshida | CLOSED | Fixed hotfix1. | Closed |
| ~~#177~~ | cp950 codec translation | stonecfc | CLOSED | Fixed hotfix1 + v1.8.6. | Closed |

**Status:** Cluster fully resolved. No remaining issues.

---

### Cluster 5: Ensemble Mode Issues (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#203** | Serial mode not working / logic complaint | yangming2027 | **OPEN** | User wants per-file serial processing. Feature exists (`--ensemble-serial` / "Finish each file"). | **REPLIED** — confirmed feature exists in v1.8.6. Awaiting user confirmation. |
| ~~#189~~ | Smart Merge clears one pass | Ywocp | CLOSED | Fixed hotfix2. | Closed |
| ~~#179~~ | Pass ordering request | yangming2027 | CLOSED | Fixed v1.8.6 `--ensemble-serial`. | Closed |

**Status:** #203 awaiting user reply. Same user as #179 — the feature they requested was built but they didn't realize it.

---

### Cluster 6: Platform Support — Apple Silicon / AMD (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| ~~#198~~ | No MPS on M1 Mac | francetoastVN | CLOSED | Fixed v1.8.7b0 — MPS detection in TransformersASR. | Done |
| **#142** | AMD Radeon 9600XT not detected | MatthaisUK | OPEN | DirectML/ROCm not supported. | **Defer v1.9+** |
| **#114** | DirectML for AMD/Intel | SingingDalong | OPEN | Major platform enablement. | **Defer v1.9+** |

---

### Cluster 7: Translation Provider Requests (5 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#143** | Custom local models | ymilv | CLOSED | VTT + custom model shipped v1.8.6. **BUT**: New comment from OdinShiva (2026-03-09) reports Custom+Ollama broken. | **NEW BUG on closed issue** — see below |
| ~~#178~~ | Custom endpoint ignored | Ywocp | CLOSED | Fixed hotfix3. | Closed |
| ~~#69~~ | Grok translation | lingyunlxh | CLOSED | Covered by custom provider. | Closed |
| **#71** | Google Translate (free, no key) | x8086 | OPEN | Fragile unofficial API. | **Defer v1.9+** |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter. | **Defer v1.9+** |

**#143 — New bug report from OdinShiva (2026-03-09):**

Screenshots examined:
- GUI: AI SRT Translate tab, Provider = "Custom (OpenAI-compatible)"
- Model = "Default", Override/Custom Model = `qwen3.5:4b`
- API Key field: **empty** (placeholder text visible)
- Custom Endpoint URL: `http://localhost:11434/v1` (standard Ollama endpoint)
- "Test Connection" button shows **Failed** badge
- Error message: `"Connection to custom failed: str expected, not NoneType"`
- Second screenshot shows Ollama running with `qwen3.5:4b` model loaded

**Diagnosis (preliminary):**
The error `str expected, not NoneType` strongly suggests the code passes the API key as `None` to the OpenAI client constructor. Ollama doesn't require API keys, but the code likely does `openai.OpenAI(api_key=api_key)` where `api_key` is None from the empty GUI field. The OpenAI SDK expects a string (even if it's a dummy like `"ollama"`). This is a **simple input validation bug** — the GUI should pass a default dummy key (e.g., `"not-needed"`) when the API key field is empty and provider is Custom.

---

### NEW — Cluster 9: Whisper Hallucination / Repetition (#209)

**This is a new cluster identified 2026-03-09.**

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#209** | Single subtitle often very long | weifu8435 | **OPEN** | Whisper repetition hallucinations not caught by RepetitionCleaner due to 4 architectural gaps in pattern matching. | **Active — root cause identified, needs code fix** |

**Reporter**: weifu8435 (repeat user, previously #187, #185)
**Platform**: Windows, Ensemble mode (Fidelity+Balanced, Large V2, "Finish each file")
**Version**: v1.8.x (likely v1.8.6 based on GUI screenshot)

**Evidence (5 screenshots + settings + 6 comments examined):**

The user's SRT output contains extremely long subtitle entries with repetitive text. Four distinct patterns observed:

| Example | SRT Line | Duration | Pattern Type | Text |
|---------|----------|----------|-------------|------|
| 1 | #279 | ~10s | Char+dakuten repetition | `あ゛あ゛あ゛あ゛あ゛あ゛...` (dozens of repetitions) |
| 2 | #164 | ~7s | Short phrase + wave dash | `あ〜、あ〜、あ〜、あ〜...` (dozens of repetitions) |
| 3 | #211 | ~7s | Medium phrase repetition (10+ chars) | `お腹が空いているときは、お腹が空いているときは...` (15+ repetitions) |
| 4 | #9 | ~7s | Long phrase repetition (16+ chars) | `お母さんがお腹を張ってくれているので、お母さんがお腹を...` (10+ repetitions) |

User asks: "Why is this happening? Is it due to some kind of interference?"

**Root Cause Analysis — 4 Architectural Gaps in RepetitionCleaner:**

The sanitizer IS called in all pipelines (including ensemble). The call chain is:
```
Pipeline.process() → SRTPostProcessor.process() → SubtitleSanitizer.process()
  → HallucinationRemover (exact/regex/fuzzy phrase matching)
  → RepetitionCleaner (6 regex patterns) ← GAPS HERE
  → CPS check (>20 chars/sec)
```

**Gap 1 — Dakuten marks break regex backreferences:**
- File: `whisperjav/modules/repetition_cleaner.py` line 89
- Pattern `single_char_flood`: `([ぁ-んァ-ン])\1{3,}` captures only base char `あ` (U+3042)
- `あ゛` = U+3042 + U+309B (combining dakuten) — two codepoints
- Backreference `\1` expects exact `あ` but finds `゛` → pattern never matches
- **Result**: `あ゛あ゛あ゛あ゛...` passes through undetected

**Gap 2 — Max phrase length too short (10 chars):**
- File: `whisperjav/modules/repetition_cleaner.py` lines 59, 71
- `phrase_with_comma` pattern: `{1,10}` char limit
- `お腹が空いているときは` = 10+ chars, `お母さんがお腹を張ってくれているので` = 16 chars → exceed limit
- **Result**: Long phrase repetitions pass through undetected

**Gap 3 — Separator patterns too rigid:**
- `vowel_extension` (line 95) expects consecutive dashes: `VOWEL+DASH+DASH+DASH`
- Actual data: `あ〜、あ〜、あ〜` has comma between repetitions → NO MATCH
- `phrase_with_separator` (line 59) expects separator WITHIN captured group, not between
- **Result**: Mixed separator patterns pass through

**Gap 4 — No general-purpose repetition safety net:**
- A method `_is_all_repetition()` exists in code but is disabled (always returns True)
- CPS check at 20 chars/sec doesn't trigger: 100-char text over 10s = 10 CPS (under threshold)
- No "if >50% of text is a repeated substring, flag it" rule
- **Result**: When specific patterns miss, nothing catches the extreme cases

**Assessment**: This is an **architectural gap**, not a simple code error. The RepetitionCleaner was designed for short, known patterns but lacks a general-purpose detector for the long-tail of repetition variants that Whisper hallucinates. The fix needs to be at the safety-net level, not just adding more specific regex patterns.

**Fix approach (3 layers):**
1. **Generic substring repetition detector** — safety net that catches any repeated substring dominating the text
2. **Fix specific pattern gaps** — dakuten support, increase phrase length limit to 30 chars
3. **Absolute length sanity check** — Japanese subtitle lines are normally 10-40 chars; >200 chars is almost certainly hallucination

**Files to modify:**
- `whisperjav/modules/repetition_cleaner.py` — patterns + new safety net
- `whisperjav/config/sanitization_constants.py` — add MAX_SUBTITLE_LENGTH

---

### Cluster 8: Feature Requests (Unclustered)

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#206** | Grey out incompatible options | techguru0 | **OPEN** | Block/grey out options incompatible with each other or user's hardware. | **v1.9+** (feature request) |
| **#205** | VibeVoice ASR support | kylesskim-sys | **OPEN** | Microsoft VibeVoice in transformers v5.3.0. Owner responded: VRAM too high (18GB std, 13GB FP16, FP4 poor quality). | **v1.9+** (feature request, owner responded) |
| ~~#194~~ | M4B file support | v2lmmj04 | CLOSED | Done v1.8.6 commit `5769688`. | Closed |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic. PyWebView `frameless=True`. | v1.9+ |
| **#180** | Multi-language GUI (i18n) | QQ804218 | OPEN | Full i18n framework. High effort. | v1.9+ |
| **#175** | Chinese language GUI | yangming2027 | OPEN | Subset of #180. | v1.9+ |
| **#164** | MPEG-TS + Google Drive | hosmallming | OPEN | MPEG-TS remux + Kaggle Drive. | v1.8.7 |
| ~~#159~~ | CLI --vad-threshold | SingingDalong | CLOSED | **Fixed** `c092db9`. | Closed |
| ~~#150~~ | Xiaomi Mimo API | rr79510 | CLOSED | Covered by custom provider. | Closed |
| ~~#143~~ | Custom local models + VTT | ymilv | CLOSED | VTT done v1.8.6. Custom model path done. | Closed |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirectories. | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log GPU VRAM before model load. | v1.8.7 |
| **#59** | Feature plans for 1.x | meizhong986 | OPEN | Meta-issue (roadmap). | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob/directory in translate CLI. | v1.8.7 |
| **#49** | Output SRT to source folder | meizhong986 | OPEN | `--output-dir source` exists. Docs gap. | v1.8.7 |
| **#44** | GUI drag-drop filename only | lingyunlxh | OPEN | Drag-drop sends filename not full path. | v1.8.7+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation gap. | v1.8.7 |

---

### Remaining Unclustered

| # | Title | Reporter | State | Category | Status |
|---|-------|----------|-------|----------|--------|
| ~~#187~~ | v1.8.5 can't generate | weifu8435 | CLOSED | Stale — fixed in hotfix2/v1.8.6. | Closed — stale |
| ~~#185~~ | v1.8.4 regression | weifu8435 | CLOSED | Stale — fixed hotfix2. Same user as #187. | Closed — stale |
| ~~#161~~ | Colab translate error | kokor594-ai | CLOSED | Stale since Feb. No response to info request. | Closed — stale |
| **#132** | Local LLM on Kaggle | TinyRick1489 | **OPEN** | Active — user posted new debug log 2026-03-07. Partial success: "few lines getting translated." Needs investigation. | **Active** |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **Active bugs needing code work** | 4 | #196 (LLM token limit STILL broken), #209 (repetition hallucination), #143 (Ollama NoneType), #132 (Local LLM on Kaggle) |
| **Active bugs needing investigation** | 1 | #198 (Mac Transformers no output — need to examine log) |
| **Active bugs awaiting user confirmation** | 2 | #204 (SSL fallback), #203 (serial mode) |
| **New — needs response only** | 2 | #207 (dup of #96), #206 (feature request ack) |
| **Feature requests (open, not done)** | 13 | #205, #206, #181, #180, #175, #164, #126, #99, #71, #51, #49, #44, #33 |
| **Meta / roadmap** | 1 | #59 |
| **Deferred to v1.9+** | 10 | #96, #205, #206, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work Only)

| Priority | # | Issue | Effort | Why |
|----------|---|-------|--------|-----|
| **HIGH** | #196 | LLM "No matches found" STILL broken on v1.8.7b1 | Medium | Fix claimed shipped but 2 users on different platforms report failure. Credibility issue. |
| **HIGH** | #209 | Repetition hallucination not caught by sanitizer | Medium | Visible output quality issue; 4 architectural gaps identified |
| **MEDIUM** | #143 | Custom+Ollama NoneType API key | Tiny | Simple input validation — pass default dummy key when empty |
| **MEDIUM** | #198 | Mac: Transformers mode no output | Unknown | Need to examine attached log; Mac is 2nd priority platform |
| **MEDIUM** | #207 | Settings persistence (dup of #96) | Tiny (reply) | Respond referencing #96, explain what IS saved |
| **LOW** | #132 | Local LLM on Kaggle | Unknown | Needs investigation of new debug log; Kaggle is 4th priority platform |
| **WAITING** | #204 | SSL/China fallback | — | v1.8.7b1 shipped, awaiting user test |
| **WAITING** | #203 | Serial mode exists | — | Replied, awaiting user confirmation |

---

## v1.8.7 — Summary of Fixes (all committed on main)

| Commit | Issue | Summary |
|--------|-------|---------|
| `10fbf30` | #198 | MPS device detection for Apple Silicon |
| `55df512` | #195 | `errors='replace'` in audio extraction subprocess |
| `c092db9` | #159 | `--vad-threshold` and `--speech-pad-ms` CLI flags |
| `7407ab6` `e5328c8` `090fd43` | #196 | Local LLM: max_tokens cap, streaming, CustomClient |
| `f81d278` | — | zipenhancer GUI option fix |
| `2abb3b3` | — | setuptools>=61.0,<82 pin (pkg_resources fix) |
| `ffde153`+ | #204 | SSL/network 3-step fallback (normal → cache → hf-mirror.com) |

### v1.8.7 Remaining Work

| Priority | # | Description | Effort | Status |
|----------|---|-------------|--------|--------|
| **HIGH** | #196 | LLM "No matches found" — v1.8.7b1 fix incomplete | Medium | Need to investigate why auto-batch + max_tokens still fails |
| **HIGH** | #209 | Repetition hallucination safety net in RepetitionCleaner | Medium | Root cause identified, fix planned |
| **MEDIUM** | #143 | Custom+Ollama NoneType API key | Tiny | Pass dummy key when API key field empty |
| **MEDIUM** | #198 | Mac Transformers mode no subtitle output | Unknown | Need to examine attached log |
| **MEDIUM** | #99 | Log GPU VRAM at INFO before model load | Tiny | Not started |
| **MEDIUM** | #164 | MPEG-TS auto-remux | Small | Not started |
| **LOW** | #51 | Batch translate wildcard/directory | Medium | Not started |
| **LOW** | #49 | Document `--output-dir source` | Tiny (docs) | Not started |
| **LOW** | #33 | Linux pyaudio install docs | Tiny (docs) | Not started |
| **LOW** | #44 | GUI drag-drop full path | Small | Not started |

---

## v1.9+ Backlog

| # | Issue | Category |
|---|-------|----------|
| #96 | Full GUI settings persistence (all pipeline tabs) | Enhancement |
| #205 | VibeVoice ASR (pending VRAM improvements) | Feature |
| #206 | Grey out incompatible GUI options | Feature |
| #180/#175 | Multi-language GUI (i18n) — Chinese first | Enhancement |
| #114/#142 | DirectML / ROCm for AMD/Intel GPUs | Platform |
| #126 | Recursive directory + mirror output structure | Feature |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL translation provider | Feature |
| #71 | Google Translate (no API key) | Feature |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved — user installed NVIDIA Toolkit |
| #201 | Install SSL cert error on fresh Win10 | 2026-03-07 | Environment issue — missing root CAs |
| #200 | NVML dual-GPU Optimus detection | 2026-03-07 | Documented — Optimus limitation |
| #198 | MPS not used on M1 Mac | 2026-03-07 | Fixed v1.8.7b0 — MPS detection added |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user after v1.8.7b0 comment |
| #196 | Local LLM token limit | 2026-03-07 | Fixed v1.8.7b0 — 3 root causes addressed |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Network/SSL/HF download** | #204, ~~#201~~, ~~#200~~, ~~#191~~, ~~#193~~ | **#204** | Fixed v1.8.7b1, awaiting confirmation |
| **Local LLM translation** | **#196**, ~~#208~~, #132, ~~#162~~, ~~#146~~, ~~#158~~ | **#196 REOPENED** (fix incomplete), #132 active | **NOT resolved** — 2 users report failure on v1.8.7b1 |
| **GUI settings** | #96, **#207**, ~~#176~~, ~~#174~~, ~~#184~~ | **#96** | Partial (v1.9); #207 is 4th dup |
| **Encoding/Unicode** | ~~#195~~, ~~#190~~, ~~#186~~, ~~#177~~ | — | **Fully resolved** |
| **Ensemble mode** | #203, ~~#189~~, ~~#179~~ | **#203** | Awaiting user reply |
| **Repetition/Hallucination** | **#209** | **#209** | **NEW — needs code fix** |
| **Apple Silicon / MPS** | ~~#198~~ | Done | Resolved |
| **AMD/non-NVIDIA** | #142, #114 | Deferred | v1.9+ |
| **Translation providers** | **#143** (new bug), ~~#178~~, ~~#69~~, #71, #43 | **#143 new bug** (Ollama NoneType) | #143 needs fix, rest deferred v1.9+ |
| **i18n** | #180, #175 | Deferred | v1.9+ |

---

## Pending GitHub Actions

### Needs Response (not yet responded)

| # | Action | Priority |
|---|--------|----------|
| #196 | Respond to zhstark: acknowledge fix is incomplete, ask for full server log and the SRT file being translated. Need to understand why batch 11 still hits token limit. | **HIGH** |
| #198 | Respond to francetoastVN: (1) LLM "No matches" is same issue as #196, being investigated. (2) Download and examine TransformerTest.txt log for Transformers mode failure. | **HIGH** |
| #143 | Respond to OdinShiva: acknowledge bug, explain workaround (put any dummy text in API Key field, e.g., "ollama"). Fix incoming. | **MEDIUM** |
| #209 | Acknowledge bug report, explain we've identified the root cause (repetition cleaner pattern gaps). Ask for version number and whether all examples come from same video or different videos. | **MEDIUM** |
| #207 | Respond: reference #96, explain what IS saved (translation, ensemble) vs what isn't (pipeline settings). | **LOW** |
| #206 | Respond: acknowledge feature request, realistic about effort (needs compatibility matrix). | **LOW** |

### Awaiting User Response

| # | Last Response | Waiting Since |
|---|--------------|---------------|
| #204 | Pointed to v1.8.7b1 release | 2026-03-08 |
| #203 | Confirmed serial mode exists | 2026-03-08 |

---

## Cross-Cutting Analysis — 6 Resolution Groups

> Added: 2026-03-09 | Based on full evidence review: 14 screenshots, 3 log files (1111 lines), 50+ comments across 10 issues

Issues are re-grouped by **shared root cause and resolution path**, cutting across the per-topic clusters above. This determines the order of work.

### Group A (CRITICAL): Translation "No Matches Found" — Cascading Token Overflow

**Issues**: #196 (zhstark, Ubuntu 5090), #198-LLM (francetoastVN, Mac M1), #132 (TinyRick1489, Kaggle)
**Severity**: HIGH — affects all platforms, all local LLM users
**v1.8.7b1 fix status**: PARTIAL — auto-batch and max_tokens work, but root cause persists

**Evidence summary:**
- #196: Ubuntu v1.8.7b1, 5090 32GB, 221.7 tps. Batch auto-reduced 30→11 (fix works), max_tokens 2692 (fix works). Still hits `WARNING: Hit API token limit` → `No matches found`. Even `--max-batch-size 10` fails.
- #198: Mac M1 v1.8.7b1, 27.4 tps CPU. Batch 10, 637 lines/58 scenes. Fails immediately on scene 1 batch 1.
- #132: Kaggle pre-v1.8.7b1, gemma-9b, 29.5 tps. `'stream': False` (pre-fix). Batch 1 times out (300s), retries, succeeds 9/10 lines. Batch 2: LLM response is **a wall of `###########`** (thousands of `#` chars). 9/23 lines translated, 14 lost.

**Cascading failure chain:**
```
1. Prompt too large for 8K context window
   ↓
2. LLM hits token limit mid-generation
   ↓
3. Retry-without-context: removes history but keeps same batch size
   ↓
4. LLM generates malformed output (wall of ###, truncated, garbled format)
   ↓
5. PySubtrans regex patterns can't match the garbage
   ↓
6. "No matches found" → batch fails → lines untranslated
```

**Why v1.8.7b1 fix is incomplete:**
The auto-batch formula uses a **static calculation** that doesn't account for:
- Variable instruction length per tone (pornify is much longer than standard)
- Conversation history growth (accumulated context from prior batches)
- Target language token expansion (Japanese→Chinese ≈ 1.5x)
- The retry-without-context path keeps the same batch size — if batch was already too large, retry still fails

**Fix plan (4 items):**
1. **Dynamic batch sizing**: Count actual prompt tokens before sending (use tokenizer, not static formula)
2. **Retry with half batch**: If retry-without-context fails, halve the batch size before giving up
3. **Garbage output detection**: Check if LLM response is >50% non-alphanumeric before parsing. If so, report meaningful error instead of regex failure
4. **Partial success acceptance**: If 9/10 lines match, accept the 9 instead of forcing full retranslation

**Files to investigate/modify:**
- `whisperjav/translate/core.py` — batch size calculation, retry logic
- `whisperjav/translate/local_backend.py` — token counting, context management
- PySubtrans library integration point — where regex matching happens

---

### Group B (MEDIUM): Custom Provider API Key Handling

**Issues**: #143 (OdinShiva — Ollama)
**Severity**: MEDIUM — blocks all Ollama/LM Studio users who don't set an API key
**Fix effort**: Tiny (1-2 lines)

**Evidence**: GUI screenshot shows Custom provider, empty API Key field, endpoint `http://localhost:11434/v1`, model `qwen3.5:4b`. Error: `"Connection to custom failed: str expected, not NoneType"`. Second screenshot confirms Ollama is running with model loaded.

**Root cause**: `openai.OpenAI(api_key=None)` — SDK requires string, gets None from empty GUI field.

**Fix**: When provider is "custom" and API key is empty/None, default to `"not-needed"`.

**Files**: `whisperjav/translate/core.py` or `whisperjav/webview_gui/api.py` — wherever OpenAI client is constructed for custom provider.

**Workaround to tell user**: Put any text (e.g., `"ollama"`) in the API Key field.

---

### Group C (HIGH): Whisper Repetition Hallucination

**Issues**: #209 (weifu8435)
**Severity**: HIGH — visible output quality degradation for all transcription users
**Fix effort**: Medium

**Evidence**: 5 SRT screenshots with 4 distinct repetition patterns:
1. `あ゛あ゛あ゛あ゛...` — char+dakuten, regex backreference broken by combining diacritics
2. `あ〜、あ〜、あ〜...` — short phrase+wave dash+comma, misses all patterns
3. `お腹が空いているときは...` repeated 15+ times — 10+ chars exceeds pattern limit of 10
4. `お母さんがお腹を張ってくれているので...` repeated 10+ times — 16 chars, far over limit

**Root cause**: 4 architectural gaps in `whisperjav/modules/repetition_cleaner.py`:
1. **Dakuten** (U+309B combining mark) breaks `([ぁ-んァ-ン])\1{3,}` backreference
2. **Max phrase length 10 chars** in `{1,10}` quantifiers — Japanese phrases routinely exceed this
3. **Rigid separator patterns** — miss comma+wave-dash combos
4. **No safety net** — no general "is >50% of text repeated?" detector; `_is_all_repetition()` exists but is disabled

**Fix plan (3 layers):**
1. **Generic substring repetition detector** — safety net: try substring lengths 2 to len//2, if any repeated 3+ times covers >50% of text → clean
2. **Fix specific patterns** — dakuten support (`[゙゚〜ー]*` after base char), increase `{1,10}` to `{1,30}`
3. **Absolute length limit** — >200 chars for Japanese subtitle ≈ certainly hallucination → truncate with warning

**Files to modify:**
- `whisperjav/modules/repetition_cleaner.py` — patterns (lines 53-97) + new safety net method
- `whisperjav/config/sanitization_constants.py` — add `MAX_SUBTITLE_LENGTH`

---

### Group D (MEDIUM): MPS Beam Search Crash (Upstream Bug)

**Issues**: #198-Transformers (francetoastVN — Mac M1)
**Severity**: MEDIUM — blocks Transformers mode on Apple Silicon
**Fix effort**: Small (defensive)

**Evidence**: Full 166-line traceback. Error: `IndexError: index 1077827584 is out of bounds for dimension 0 with size 40` in `transformers/models/whisper/generation_whisper.py:1172` → `split_by_batch_index`. The value `1077827584` (0x40404040) is a classic uninitialized memory pattern.

**Root cause**: HuggingFace Transformers library bug on MPS backend. Beam search indices come back as garbage from Metal GPU. Known class of MPS backend issues in PyTorch.

Key observations:
- Model loads on MPS successfully, audio extracts fine (7348.9s ≈ 2hr movie)
- Crash occurs on first transcription chunk's beam search postprocessing
- Uses kotoba-whisper-bilingual-v1.0, float16, batch 8

**Fix plan:**
1. **Immediate workaround**: Tell user `--hf-device cpu` (bypasses MPS, slower but works)
2. **Defensive fix**: In `transformers_asr.py`, catch `IndexError` during transcription → retry with `device="cpu"` + log warning. Follows existing MPS→CPU fallback pattern used in speech enhancement backends.
3. **Upstream**: File issue with HuggingFace Transformers for beam search on MPS with long audio

**Files to modify:**
- `whisperjav/modules/transformers_asr.py` — add try/except around `self.pipe()` call (line ~285)

---

### Group E (LOW-MEDIUM): GUI Settings & Usability

**Issues**: #207 (q864310563), #96 (sky9639), #206 (techguru0)
**Severity**: LOW-MEDIUM — most frequently reported UX frustration (4 duplicates of #96)
**Fix effort**: Reply only for now; pipeline persistence is Small effort for partial fix

**Evidence**: #207 is the 4th independent duplicate of #96. Chinese user on .exe installer asks "how to save settings?" #206 is separate: feature request to grey out incompatible options.

**What IS saved** (since hotfix2): Translation tab settings, Ensemble tab settings.
**What is NOT saved**: Pipeline tab (model, sensitivity, scene detector, speech enhancer, speech segmenter).

**Actions:**
1. **#207**: Respond explaining what's saved vs not, reference #96
2. **#206**: Acknowledge feature request, defer v1.9+
3. **Consider**: Partial pipeline persistence in v1.8.8 — save dropdowns to existing settings file

---

### Group F (WAITING): Network/SSL — China Users

**Issues**: #204 (yangming2027)
**Severity**: HIGH for Chinese user base, but fix already shipped
**Status**: v1.8.7b1 released with 3-step fallback. Two comments pointing users to release. No user confirmation yet.

**Action**: Wait. No code work needed.

---

## Resolution Execution Order

This is the recommended order for processing these groups:

| Step | Group | Action | Effort | Blocks |
|------|-------|--------|--------|--------|
| 1 | **B** | Fix #143 Ollama NoneType API key | Tiny (1-2 lines) | Quick win, unblocks Ollama users |
| 2 | **E** | Respond to #207, #206 on GitHub | Tiny (comments) | Clears response backlog |
| 3 | **D** | Respond to #198 with `--hf-device cpu` workaround + add defensive MPS→CPU fallback | Small | Unblocks Mac Transformers user |
| 4 | **C** | Fix #209 repetition hallucination (3-layer safety net) | Medium | Output quality for all users |
| 5 | **A** | Fix #196 translation token overflow cascade (dynamic batch + retry + garbage detection) | Medium | Most impactful fix — all LLM translation users |
| 6 | **F** | Wait for #204 confirmation | None | — |

**Rationale**: Start with quick wins (B, E) to clear the queue, then handle the two medium-effort fixes (C, D, A) in order of isolation (C and D are independent; A is complex and may need PySubtrans investigation).

---

## Evidence Inventory

All evidence examined during this analysis:

| Source | Type | Lines/Size | Key Finding |
|--------|------|------------|-------------|
| #209 screenshot 1 (main) | SRT output | — | Line 279: `あ゛あ゛あ゛...` dakuten repetition |
| #209 screenshot 2 (settings) | GUI | — | Ensemble mode, Fidelity+Balanced, Large V2 |
| #209 screenshot 3 (line 164) | SRT output | — | `あ〜、あ〜...` wave-dash repetition |
| #209 screenshot 4 (line 211) | SRT output | — | `お腹が空いているときは...` phrase repetition (10+ chars) |
| #209 screenshot 5 (line 9) | SRT output | — | `お母さんがお腹を張って...` phrase repetition (16 chars) |
| #196 screenshot 1 | Terminal | — | v1.8.7b1 confirmed, 221.7 tps, batch 30→11 |
| #196 screenshot 2 | Terminal | — | LOCAL LLM ASSESSMENT: 33 layers CUDA, READY |
| #196 screenshot 3 | Terminal | — | "No matches found" errors per scene |
| #198 TransformerTest.txt | Log file | 166 lines | `IndexError: index 1077827584` — MPS beam search crash |
| #198 LLM comment | Terminal paste | ~50 lines | Same "No matches found" as #196 |
| #143 screenshot 1 (settings) | GUI | — | Custom provider, empty API key, Ollama endpoint |
| #143 screenshot 2 (Ollama) | Ollama UI | — | qwen3.5:4b model running |
| #132 debug_log.txt | Log file | 945 lines | Batch 2 response = wall of `###`, streaming=False, 9/23 translated |
| GitHub comments sweep | API query | 60+ comments | All comments since 2026-03-08 00:00 |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
