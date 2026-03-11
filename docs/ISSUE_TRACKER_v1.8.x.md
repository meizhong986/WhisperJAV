# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-11 (rev4 — v1.8.7 RELEASED, 3 new issues) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **28 open** on GitHub

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **28** | Was 25 at last update; 3 new (#210, #211, #212) |
| New issues since last update (rev3) | 3 | #210 (install DNS), #211 (startup warning), #212 (local LLM regex) |
| **Active bugs (need code work)** | 2 | #196/#212 (LLM token overflow — CRITICAL), #198 (MPS beam search — still crashing on v1.8.7) |
| **Active bugs (awaiting user confirmation)** | 3 | #204 (SSL fallback), #203 (serial mode), #209 (repetition fix) |
| **Cosmetic / informational** | 2 | #211 (urllib3 warning — benign), #210 (China DNS — already responded) |
| **New duplicates** | 1 | #212 (dup of #196 — same "No matches" LLM error) |
| Fixed in v1.8.7 (all shipped) | 12 | See v1.8.7 release notes |
| Feature requests (open) | 13 | Unchanged from rev3 |
| Deferred to v1.9+ | 10 | Unchanged |

### CRITICAL: MPS Beam Search Still Broken (#198)

The v1.8.7 fix (defer generate_kwargs to model defaults, commit `f0ad57a`) did **NOT** resolve the MPS crash. francetoastVN tested v1.8.7 on Mac M1 (2026-03-11) and gets the **same error**:

```
torch.AcceleratorError: index -4785543516806487491 is out of bounds: 0, range 0 to 4
```

**This is NOT a WhisperJAV bug.** The traceback shows the crash inside HuggingFace `transformers` library (`generation_whisper.py → _beam_search → logits_process.py:2021`). Our fix correctly deferred to model defaults — the model itself uses beam search in its `generation_config.json`. The bug is in HF Transformers + MPS + beam search interaction.

**Evidence**: v1.8.7 log shows `generate_kwargs` no longer forces `num_beams=5` — it's using model defaults. The crash still occurs because kotoba-whisper-bilingual-v1.0's own config specifies beam search, and HF's beam search code has an MPS-specific tensor indexing bug.

**Action needed**: Either (a) force `num_beams=1` (greedy) when device is MPS as a workaround, or (b) wait for HF Transformers upstream fix.

---

## Cluster Analysis

### Cluster 1: Local LLM Translation — Token Overflow (CRITICAL)

**Issues**: #196, #212 (NEW), #198-LLM, #132
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**v1.8.7 status**: FIX INCOMPLETE — 3 new reports confirm failure persists

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090? | **OPEN (NEW)** | Clean v1.8.7, llama-8b, 65 tps, batch 10. Fails scene 1 batch 1 immediately. |
| **#196** | Local Translation Errors | destinyawaits | Ubuntu, 5090 32GB | CLOSED | zhstark report: v1.8.7b1, batch 30→11 (auto-batch works), still fails. |
| **#198-LLM** | LLM on Mac M1 | francetoastVN | macOS M1 | CLOSED | v1.8.7, 31.7 tps CPU, batch 10, 637 lines/58 scenes. Fails scene 1 batch 1. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | Pre-v1.8.7. Partial success: 9/23 lines. Wall of `###` in batch 2. |

**Key observations from #212 log (v1.8.7, destinyawaits):**
- Version: v1.8.7 confirmed (not beta)
- Speed: 65.0 tps (fast GPU), 33 layers CUDA
- max_tokens: 2692 (JAV/CJK-tuned, fix IS applied)
- Streaming: True (fix IS applied)
- Batch size: 10 (reasonable)
- **Failure**: `WARNING: Hit API token limit, retrying batch without context...` → `No matches found` → batch fails → translation cancelled
- **Resume**: `.subtrans` file present (was retrying from previous cancelled attempt)
- Target: English, Tone: standard

**Key observations from #198-LLM log (v1.8.7, francetoastVN):**
- Same exact failure pattern on Mac M1 CPU-only (31.7 tps)
- Target: English, Tone: standard
- Batch 10, 637 lines/58 scenes
- Fails immediately on scene 1 batch 1 AND scene 1 batch 2

**Root cause analysis (refined):**

The v1.8.7b0/b1 fixes (max_tokens cap, streaming, auto-batch) are all working correctly. The problem is deeper:

1. **Prompt is too large for 8K context even with batch=10**: The translation prompt includes system instructions + conversation history + 10 subtitle lines. For standard tone (shorter instructions), this still overflows 8K. The `max_tokens=2692` cap reserves output space, but the INPUT prompt itself exceeds the remaining ~5.3K tokens.

2. **Retry-without-context doesn't reduce batch size**: When token limit is hit, PySubtrans retries the SAME batch with context stripped. If the batch itself (10 lines of Japanese subtitle) + system instructions > context window, removing context doesn't help.

3. **The "No matches" error is a symptom, not the cause**: The LLM's response after hitting the token limit is garbage/truncated, so PySubtrans regex patterns can't parse it.

**This is the #1 priority bug for v1.8.8.**

---

### Cluster 2: MPS / Apple Silicon (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#198** | Transformers MPS crash | francetoastVN | CLOSED | HF Transformers beam search + MPS = tensor index corruption | **FIX INCOMPLETE** — v1.8.7 deferred kwargs but model's own beam search still crashes on MPS. Need MPS-specific workaround. |
| ~~#198~~ | MPS not detected | francetoastVN | CLOSED | Fixed v1.8.7b0 (MPS detection). | Done |

**v1.8.7 test result (2026-03-11):** francetoastVN tested v1.8.7 Transformers mode on Mac M1 with kotoba-whisper-bilingual-v1.0. MPS IS detected (`Device: mps`), model loads fine, transcription starts, but crashes at ~4.5 minutes into a 2-hour file with `torch.AcceleratorError: index -4785543516806487491 is out of bounds`.

**The crash is in `transformers/generation/logits_process.py:2021`** — inside `sampled_tokens.tolist()` during beam search. This is a known MPS numerical instability: beam search scores on MPS produce corrupted tensor indices.

**Fix options:**
1. **Force greedy decoding on MPS**: `num_beams=1` when device is MPS — simple, reliable, slight quality loss
2. **CPU fallback for generation**: Run encoder on MPS, decoder on CPU — complex, slower
3. **Wait for HF upstream fix** — unknown timeline

---

### Cluster 3: Network / Installation (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | `Could not resolve host: github.com` — bundled git can't resolve DNS through proxy. | **RESPONDED** — meizhong986 pointed to v1.8.7 (proxy detection fix). yangming2027 commented: "network issue, try stable VPN." |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors through proxy. | **FIXED** v1.8.7 — 3-step fallback. meizhong986 responded. Awaiting confirmation. |
| **#201** | Install SSL cert error | jl6564 | **OPEN** | Missing root CA certs on fresh Windows. | **RESPONDED** — v1.8.7 includes broadened error detection. |

**Status**: All three have been responded to pointing to v1.8.7 fixes. No further code work needed — awaiting user confirmation.

---

### Cluster 4: GUI Settings Persistence (3 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#96** | Full settings persistence | sky9639 | OPEN | Translation + ensemble done. Pipeline tab remaining. | **v1.9** |
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | "Settings reset on restart" | **Dup of #96** — needs response |
| ~~#176~~ | Translation settings lost | Ywocp | CLOSED | Fixed hotfix2. | Done |

**Status**: #207 has NO responses. Needs a reply explaining what IS saved (translation, ensemble) vs what isn't (pipeline tab settings). Reference #96.

---

### Cluster 5: Startup Warning (1 issue — COSMETIC)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#211** | 启动报错 | WillChengCN | **OPEN (NEW)** | `RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.0.1)/charset_normalizer (3.4.5) doesn't match a supported version!` | **BENIGN** — cosmetic warning from requests library. justantopair-ai confirms: "meizhong986 said this warning is nothing serious." |

**Analysis**: This is a dependency version mismatch warning from the `requests` library. It has zero functional impact. The `urllib3` and `charset_normalizer` versions bundled by the installer are newer than what `requests` was tested with, but they work fine.

**Action**: Respond explaining the warning is cosmetic and doesn't affect functionality. Can consider pinning requests or suppressing the warning in a future release if it causes confusion.

---

### Cluster 6: Whisper Repetition Hallucination (#209) — FIXED, BEING TESTED

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#209** | Single subtitle very long | weifu8435 | **OPEN** | RepetitionCleaner regex gaps. | **FIXED** `9317695` — shipped in v1.8.7. User is actively testing. |

**User update (2026-03-11):** weifu8435 responded positively:
- "I started testing with the official 1.8.7 version today to see if any other hallucinations occur."
- Claims to process "four subtitles a day, over 100 a month" — will provide comprehensive real-world testing.
- Also posted additional example SRT files from v1.8.6 showing more repetition patterns.

**Status**: Actively being validated. User is engaged and testing.

---

### Cluster 7: Encoding / Unicode — FULLY RESOLVED

All 4 issues fixed. No new reports.

---

### Cluster 8: Ensemble Mode (1 remaining)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#203** | Serial mode request | yangming2027 | **OPEN** | Replied — feature exists. Awaiting confirmation. |

---

### Cluster 9: Feature Requests (Unclustered)

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#206** | Grey out incompatible options | techguru0 | **OPEN** | Block incompatible GUI choices. | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | **OPEN** | Microsoft VibeVoice. Owner: VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic. | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n. | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180. | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud. | v1.8.8 |
| **#142** | AMD Radeon | MatthaisUK | OPEN | DirectML/ROCm. | v1.9+ |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirs. | v1.9+ |
| **#114** | DirectML | SingingDalong | OPEN | AMD/Intel GPU. | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log VRAM. | v1.8.8 |
| **#71** | Google Translate (free) | x8086 | OPEN | Fragile API. | v1.9+ |
| **#59** | Feature plans | meizhong986 | OPEN | Meta roadmap. | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob in translate CLI. | v1.8.8 |
| **#49** | Output to source folder | meizhong986 | OPEN | Docs gap. | v1.8.8 |
| **#44** | GUI drag-drop | lingyunlxh | OPEN | Filename vs path. | v1.8.8+ |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter. | v1.9+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation. | v1.8.8 |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **CRITICAL bugs needing code work** | 1 | #196/#212 (LLM token overflow — affects all local LLM users) |
| **HIGH bugs needing code work** | 1 | #198 (MPS beam search still crashes on v1.8.7) |
| **Active bugs awaiting user confirmation** | 3 | #204 (SSL), #203 (serial), #209 (repetition) |
| **Cosmetic — needs response only** | 3 | #211 (urllib3 warning), #210 (install DNS), #207 (settings dup) |
| **Feature requests (open)** | 17 | See Cluster 9 |
| **Meta / roadmap** | 1 | #59 |
| **Deferred to v1.9+** | 10 | #96, #205, #206, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Effort | Why |
|----------|---|-------|--------|-----|
| **CRITICAL** | #196/#212 | LLM "No matches found" — token overflow | Medium-Large | 4 reporters, 3 platforms (Linux/Mac/Kaggle), all fail. #1 user pain point. |
| **HIGH** | #198 | MPS beam search crash on v1.8.7 | Small | Straightforward workaround (force greedy on MPS). Single platform but user has tested thoroughly. |
| **MEDIUM** | #211 | urllib3 warning (cosmetic) | Tiny (reply) | 2 users confused by it. Benign. |
| **MEDIUM** | #207 | Settings persistence (dup #96) | Tiny (reply) | 4th duplicate — clearly frustrating users |
| **LOW** | #210 | Install DNS through proxy | Tiny (reply) | Already responded, v1.8.7 fix applies |
| **DONE** | #209 | Repetition hallucination | — | Shipped v1.8.7, user actively testing |
| **DONE** | #143 | Custom provider NoneType | — | Shipped v1.8.7 |
| **WAITING** | #204 | SSL/China fallback | — | v1.8.7 shipped, awaiting confirmation |
| **WAITING** | #203 | Serial mode | — | Replied, awaiting confirmation |

---

## v1.8.7 — Summary of Shipped Fixes

| Commit | Issue | Summary |
|--------|-------|---------|
| `10fbf30` | #198 | MPS device detection for Apple Silicon |
| `55df512` | #195 | `errors='replace'` in audio extraction |
| `c092db9` | #159 | `--vad-threshold` and `--speech-pad-ms` CLI flags |
| `7407ab6` `e5328c8` `090fd43` | #196 | Local LLM: max_tokens cap, streaming, CustomClient |
| `f81d278` | — | ZipEnhancer GUI option fix |
| `2abb3b3` | — | setuptools>=61.0,<82 pin |
| `ffde153`+ | #204 | SSL/network 3-step fallback |
| `cd72de4` | #143 | Custom provider os.getenv(None) crash |
| `9317695` | #209 | Repetition cleaner: 4 regex gaps + safety net |
| `f0ad57a` | #198 | HF Transformers: defer generate_kwargs to model defaults |
| (installer) | #210 | Proxy detection for bundled git |
| (installer) | — | GPU constraint numpy fix, torchvision Phase 3 |

---

## v1.8.8 — Planned Work

| Priority | # | Description | Effort | Status |
|----------|---|-------------|--------|--------|
| **CRITICAL** | #196/#212 | LLM token overflow — dynamic batch sizing, retry with smaller batch, garbage output detection | Medium-Large | Not started |
| **HIGH** | #198 | MPS beam search — force greedy decoding (`num_beams=1`) on MPS device | Small | Not started |
| **MEDIUM** | #211 | Suppress or fix urllib3/charset_normalizer version warning | Tiny | Not started |
| **MEDIUM** | #99 | Log GPU VRAM at INFO before model load | Tiny | Not started |
| **LOW** | #164 | MPEG-TS auto-remux | Small | Not started |
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

## Pending GitHub Actions

### Needs Response (not yet responded)

| # | Action | Priority |
|---|--------|----------|
| #212 | Respond: acknowledge this is the same root cause as #196 (LLM token overflow). Being investigated for v1.8.8. Workaround: use cloud translation provider for now. | **HIGH** |
| #211 | Respond: urllib3/charset_normalizer warning is cosmetic and doesn't affect functionality. Safe to ignore. | **MEDIUM** |
| #207 | Respond: reference #96, explain what IS saved (translation, ensemble) vs what isn't (pipeline). | **MEDIUM** |

### Needs Follow-up

| # | Action | Priority |
|---|--------|----------|
| #198 | Respond to francetoastVN: MPS beam search crash is an upstream HF Transformers bug. Our fix deferred kwargs correctly but the model's own beam search still triggers the MPS tensor bug. Working on MPS-specific workaround (force greedy decoding). LLM issue = same as #196. | **HIGH** |

### Awaiting User Response

| # | Last Response | Waiting Since |
|---|--------------|---------------|
| #204 | Pointed to v1.8.7 release | 2026-03-10 |
| #203 | Confirmed serial mode exists | 2026-03-08 |
| #209 | User actively testing v1.8.7 | 2026-03-11 |
| #210 | meizhong986 pointed to v1.8.7 | 2026-03-10 |
| #201 | meizhong986 pointed to v1.8.7 | 2026-03-10 |

---

## Cross-Cutting Analysis — Resolution Groups (Updated)

### Group A (CRITICAL): Translation "No Matches Found" — Token Overflow

**Issues**: #196 (zhstark), #212 (destinyawaits, NEW), #198-LLM (francetoastVN), #132 (TinyRick1489)
**Status**: **NOT FIXED** — 4 reporters on 3 platforms confirm failure on v1.8.7

**v1.8.7 fixes that ARE working:**
- max_tokens cap: 2692 ✅
- Streaming: True ✅
- Auto-batch reduction: works (30→11) ✅
- Custom server client: works ✅

**Why it still fails (deeper analysis):**

The core problem: **10 subtitle lines + system instructions + conversation history > 8K context window**

Even with batch=10, the prompt sent to the LLM looks like:
```
System: [~2000 tokens of translation instructions]
History: [~1000+ tokens of prior conversation context]
User: [~1500+ tokens of 10 Japanese subtitle lines with formatting]
= ~4500+ input tokens
```
With max_tokens=2692 reserved for output, total = ~7200 tokens. Add overhead and the first 8192-token request already overflows.

**The retry-without-context path removes history but keeps batch=10**. If batch=10 itself + instructions > context, the retry also fails.

**Fix plan (4 items for v1.8.8):**
1. **Pre-flight token estimation**: Count approximate prompt tokens before sending. If estimated > 70% of context, reduce batch size proactively.
2. **Retry with half batch**: When "Hit API token limit" occurs, halve batch size on retry instead of only removing context.
3. **Minimum viable batch**: Allow batch=1 as final fallback. A single subtitle line should always fit in 8K.
4. **Garbage output detection**: If LLM response is >50% non-alphanumeric or empty, report "model output was garbled" instead of cryptic regex failure.

---

### Group B (FIXED): Custom Provider API Key — SHIPPED v1.8.7

**Issues**: #143
**Status**: **FIXED** `cd72de4`. OdinShiva reported after closure with Ollama endpoint. Fix adds `custom` to early-return group.
**Note**: OdinShiva's latest comment (2026-03-09) quotes our fix note — appears to have seen the response but hasn't confirmed fix works yet.

---

### Group C (FIXED, BEING TESTED): Repetition Hallucination — SHIPPED v1.8.7

**Issues**: #209
**Status**: **FIXED** `9317695`. weifu8435 actively testing v1.8.7 (2026-03-11). Processing 4 videos/day, 100+/month. Will report any remaining hallucinations.

---

### Group D (PARTIALLY FIXED): MPS Beam Search — NEEDS v1.8.8 WORKAROUND

**Issues**: #198
**Status**: v1.8.7 fix deferred generate_kwargs to model defaults (correct approach). **Still crashes** because kotoba-whisper's own `generation_config.json` uses beam search, and HF Transformers has a bug in beam search on MPS.

**Fix for v1.8.8**: Force `num_beams=1` when device is MPS. This disables beam search entirely on Apple Silicon — slight quality impact but prevents crash.

---

### Group E (LOW-MEDIUM): GUI Settings & Usability

**Issues**: #207, #96, #206, #211
**Status**: #207 and #211 need responses. No code work for v1.8.8.

---

### Group F (WAITING): Network/SSL — China Users

**Issues**: #204, #210, #201
**Status**: All responded to. v1.8.7 shipped with fixes. Waiting for confirmation.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM token overflow** | **#196**, **#212** (NEW), #198-LLM, #132 | **#196** | **CRITICAL — NOT resolved** |
| **MPS/Apple Silicon** | **#198** | **#198** | **PARTIAL** — detection done, beam search crash remains |
| **Network/SSL/Install** | #204, #210, #201 | **#204** | Responded, awaiting confirmation |
| **GUI settings** | #96, **#207** | **#96** | Partial (v1.9); #207 is 4th dup |
| **Repetition/Hallucination** | **#209** | **#209** | **FIXED** v1.8.7 — actively tested |
| **Encoding/Unicode** | ~~all closed~~ | — | **Fully resolved** |
| **Ensemble** | #203 | **#203** | Awaiting reply |
| **Startup warning** | **#211** | **#211** | Cosmetic — needs reply |
| **AMD/non-NVIDIA** | #142, #114 | Deferred | v1.9+ |
| **Translation providers** | #71, #43 | Deferred | v1.9+ |
| **i18n** | #180, #175 | Deferred | v1.9+ |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved — user installed NVIDIA Toolkit |
| #198 | MPS not used on M1 Mac | 2026-03-07 | Fixed v1.8.7b0 — MPS detection. **But beam search still crashes (new report)** |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user after v1.8.7b0 |
| #196 | Local Translation Errors | 2026-03-07 | Fixed v1.8.7b0 — partial. **New reports confirm failure persists** |
| #195 | UnicodeDecodeError audio extraction | 2026-03-08 | Fixed `55df512` |
| #194 | M4B file support | 2026-03-08 | Fixed `5769688` |
| #193 | Package update question | 2026-03-08 | Answered |
| #192 | Update to latest dev | 2026-03-02 | Resolved |
| #191 | Pass2 missing SSL | 2026-03-08 | Dup of #204 |
| #190 | GBK codec crash | 2026-03-04 | Fixed v1.8.6 |
| #189 | Smart Merge clears content | 2026-03-08 | Fixed hotfix2 |
| #188 | Unknown provider Gemini | 2026-03-04 | Fixed hotfix2 |

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-03-11 | **v1.8.7 RELEASED.** 3 new issues (#210, #211, #212). #212 is dup of #196 (LLM token overflow, STILL broken — now 4 reporters). #198 MPS beam search STILL crashes on v1.8.7 (francetoastVN tested). #211 is cosmetic urllib3 warning. #209 user actively testing v1.8.7 (positive so far). Updated all clusters and priorities. |
| 2026-03-09 | Groups B, C, D committed. #143, #209, #198 fixes shipped. |
| 2026-03-09 | Full evidence review: 14 screenshots, 3 log files, 50+ comments. 6 resolution groups identified. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
