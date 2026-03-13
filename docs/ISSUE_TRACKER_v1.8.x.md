# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-13 (rev6 — v1.8.8 code complete, 3 new issues since rev5) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **35 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `FIX IMPLEMENTED` | Code written and tested locally. NOT yet verified by user or in production. |
| `FIX VERIFIED` | User confirmed the fix resolves their issue. Safe to close the GitHub issue. |
| `AWAITING CONFIRMATION` | Fix shipped in a release. Waiting for user to confirm. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **35** | Was 32 at rev5; 3 new (#218, #219, #220) |
| New issues since rev5 | 3 | #218 (uv cu118 wheel), #219 (MossFormer2 stereo crash), #220 (install stalls) |
| **Active bugs — FIX IMPLEMENTED (not yet verified)** | 6 | LLM cluster (#196/#212/#214/#132), #200 (Optimus), #209 (repetition), #218 (cu118 wheel), #219 (MossFormer2 SS) |
| **Shipped fixes — AWAITING CONFIRMATION** | 2 | #198 (MPS beam search), #211 (urllib3 warning) |
| **Responded (awaiting user reply)** | 7 | #220, #219, #218, #217, #215, #214, #207 — all replied 2026-03-13 |
| **Awaiting user response** | 4 | #204, #203, #201, #210 |
| Feature requests (open) | 16 | Including #213 Intel GPU, #198-FR English dropdown |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

---

## v1.8.8 Stable — Implementation Status

All Track A, B, C code is complete. Changes are uncommitted pending verification.

| ID | Issue(s) | Description | Status | Notes |
|----|----------|-------------|--------|-------|
| A1 | #212/#214/#132 | Diagnostic token logging (batch stats, "No matches" handler) | `FIX IMPLEMENTED` | Needs user validation via diagnostic data |
| A2 | #212 | .subtrans stale settings override (version-stamp, auto-delete) | `FIX IMPLEMENTED` | Needs validation from zhstark's scenario |
| A3 | **#218** | cu118 wheel version mismatch (`UV_SKIP_WHEEL_FILENAME_CHECK=1`) | `FIX IMPLEMENTED` | Exact error from WillChengCN's log. Needs #218 user to confirm. |
| A4 | #212/#214 | Reduce max_tokens 2x→1x multiplier (2392→1820 for 8K ctx) | `FIX IMPLEMENTED` | Needs real-world validation via A1 diagnostic data |
| A5 | #214 | pornify.txt → sectioned format with format example | `DONE (no issue)` | Internal improvement |
| A6 | — | `--provider ollama` preview (auto-detect, num_ctx=8192, streaming) | `FIX IMPLEMENTED` | Needs manual testing with Ollama installed |
| B1 | **#200** | NVML Optimus laptop fallback (`--force-cuda` flag, guidance text) | `FIX IMPLEMENTED` | Needs Ywocp to confirm |
| C1 | **#209** | Repetition cleaner pattern #8 (`sentence_phrase_repetition`) | `FIX IMPLEMENTED` | Tested: 192→42 chars. Needs weifu8435 to confirm on v1.8.8 |

### Shipped in v1.8.8b1 (2026-03-12)

| Commit | Issue | Summary | Status |
|--------|-------|---------|--------|
| `9dc8f1f` | #198 | Force greedy decoding (`num_beams=1`) on MPS | `AWAITING CONFIRMATION` |
| `d5b8296` | — | numpy 2.x migration (all internal code) | `DONE` |
| `dea0fa2` | — | Remove `np.int_` usage (numpy 2.x crash) | `DONE` |
| `32c3711` | — | Phase 1a installer: env detection, `--local`, `--inexact` | `DONE` |
| `18be938` | — | GUI launcher creation in install scripts | `DONE` |
| `5cc5ad2` | — | Version bump to 1.8.8b1 | `DONE` |
| — | #211 | urllib3/chardet warning suppressed | `AWAITING CONFIRMATION` (justantopair-ai: "nothing serious") |

---

## Cluster Analysis

### Cluster A: Local LLM Translation — CRITICAL (4 issues)

**Issues**: #196 (closed), #212, #214, #132
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**Status**: `FIX IMPLEMENTED` (A1 diagnostics, A2 .subtrans fix, A4 max_tokens reduction) — NOT YET VERIFIED

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **OPEN — NEEDS RESPONSE** | gemma-9b, 12.3 tps, batch auto-reduced 30→11. Server 502 error immediately. `max_tokens: 2392`. |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | llama-8b AND gemma-9b both fail. User switched to DeepSeek cloud. Latest comment confirms same issue with gemma-9b. |
| **#196** | Local Translation Errors | destinyawaits | Ubuntu, 5090 32GB | CLOSED | Related to #212. zhstark: batch 30→11, still fails. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | TinyRick1489 posted debug log (2026-03-07). meizhong986 responded "working on fix." |

**Two distinct failure modes:**
1. **502 Server Error** (#214): Server crashes/times out during translation.
2. **"No matches found"** (#212, #196, #132): Server responds but output is garbled/unparseable.

**v1.8.8 tactical fixes** (bridge to v1.9.0 Ollama migration):
- A1: Diagnostic logging to capture actual token usage and raw model output
- A2: .subtrans stale settings fix (prevents old settings from overriding CLI args)
- A4: max_tokens 2x→1x reduction (less room for verbose/garbled output)
- A6: `--provider ollama` preview as an immediate alternative for local LLM users

**Strategic direction**: v1.9.0 will deprecate llama-cpp-python entirely and complete Ollama migration.

---

### Cluster B: MPS / Apple Silicon (1 issue) — FIX SHIPPED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | CLOSED | `AWAITING CONFIRMATION` — fixed in v1.8.8b1 (`num_beams=1` on MPS, commit `9dc8f1f`) |

francetoastVN also requested (2026-03-11) adding English translation option to Ensemble and AI SRT Translate dropdown menus — tracked as feature request #198-FR.

---

### Cluster C: Network / Installation (5 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#220** | 安装卡着不动 (install stalls) | libinghui20001231-debug | **OPEN (NEW)** | Install stalls during PyTorch download. GTX 1650, driver 462, cu118 correct. Likely slow network (~2GB wheels). | `NEEDS RESPONSE` — guidance only, no code fix |
| **#218** | 安装错误 (uv cu118 wheel) | WillChengCN | **OPEN (NEW)** | uv rejects cu118 llama-cpp wheel: internal version `0.2.26+cu118` ≠ filename `0.2.26` | `FIX IMPLEMENTED` (A3: `UV_SKIP_WHEEL_FILENAME_CHECK=1`) |
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM | **OPEN** | Can't find GUI exe after v1.8.7 install | `NEEDS RESPONSE` — explain exe location |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy | Responded, `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors | Responded, `AWAITING CONFIRMATION` |
| **#201** | Install SSL cert error | jl6564 | **OPEN** | Missing root CA certs | Responded, `AWAITING CONFIRMATION` |

---

### Cluster D: GPU Detection (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Dual GPU (AMD iGPU + NVIDIA dGPU). NVML fails on Optimus config. | `FIX IMPLEMENTED` (B1: `--force-cuda` flag + improved guidance text) |
| **#213** | Intel GPU support | DDXDB | **OPEN** | Requests `torch.xpu` support via PyTorch XPU wheels | `DEFERRED` to v1.9+ |

---

### Cluster E: GUI Settings Persistence (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | `NEEDS RESPONSE` — 0 comments. Dup of #96. |
| **#96** | Full settings persistence | sky9639 | OPEN | `DEFERRED` to v1.9 — translation + ensemble done, pipeline tab remaining |

---

### Cluster F: Whisper Output Quality (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | `FIX IMPLEMENTED` (C1: pattern #8 `sentence_phrase_repetition`). Tested on FC2-PPV-4025269 line 52: 192→42 chars. Needs weifu8435 to confirm on v1.8.8. |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN** | `NEEDS RESPONSE` — Qwen3-ASR produces fewer/smaller subtitles. Expected behavior (different architecture). |

---

### Cluster G: Speech Enhancement — NEW BUG

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#219** | MossFormer2_SS_16K "too many dimensions" error | anon12642 | **OPEN (NEW)** | `FIX IMPLEMENTED` (C2) — Speech separation model outputs 3D tensor `(num_sources, batch, length)`. Fix takes first separated source. `NEEDS RESPONSE` on GitHub. |

**Root cause**: MossFormer2_SS_16K is a speech **separation** model (not enhancement). It outputs multiple audio streams as `(num_sources, 1, length)`. The code only handled 2D output, so the 3D array propagated to `sf.write()` which raised "Invalid shape". Fix: take first separated source (primary speaker) from the 3D output.

---

### Cluster H: Startup Warning — COSMETIC

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#211** | 启动报错 (urllib3 warning) | WillChengCN | **OPEN** | `AWAITING CONFIRMATION` — Suppressed in v1.8.8b1. justantopair-ai confirms "nothing serious." |

---

### Cluster I: Ensemble Mode — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#203** | Serial mode request | yangming2027 | **OPEN** | Feature already exists. meizhong986 confirmed. `AWAITING CONFIRMATION` from user. |

---

### Cluster J: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#213** | Intel GPU (XPU) support | DDXDB | OPEN | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | DirectML/ROCm. New comment from FishYu-OWO re AMD/ROCm on Windows. | v1.9+ |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirs | v1.9+ |
| **#114** | DirectML | SingingDalong | OPEN | AMD/Intel GPU | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log VRAM | Backlog |
| **#71** | Google Translate (free) | x8086 | OPEN | Fragile API | v1.9+ |
| **#59** | Feature plans | meizhong986 | OPEN | Meta roadmap | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob in translate CLI | Backlog |
| **#49** | Output to source folder | meizhong986 | OPEN | Docs gap | Backlog |
| **#44** | GUI drag-drop | lingyunlxh | OPEN | Filename vs path | Backlog |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter | v1.9+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation | Backlog |
| **#198-FR** | English in Ensemble/Translate dropdown | francetoastVN | (comment) | Feature request from #198 | Backlog |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **FIX IMPLEMENTED — needs user verification** | 6 | LLM cluster (#212/#214/#132), #200 (Optimus), #209 (repetition), #218 (cu118 wheel), #219 (MossFormer2 SS) |
| **Shipped, AWAITING CONFIRMATION** | 2 | #198 (MPS), #211 (urllib3 warning) |
| **AWAITING user response (responded)** | 4 | #204, #203, #201, #210 |
| **Responded, AWAITING user reply** | 7 | #220, #219, #218, #217, #215, #214, #207 — all replied 2026-03-13 |
| **Feature requests (open)** | 19 | See Cluster J |
| **DEFERRED to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Status | Why |
|----------|---|-------|--------|-----|
| **CRITICAL** | #212/#214/#132 | LLM translation (502 + "No matches") | `FIX IMPLEMENTED` (A1,A2,A4,A6) | 5 reporters, 3 platforms. Tactical fixes + Ollama preview. |
| **HIGH** | #200 | NVML Optimus laptop GPU detection | `FIX IMPLEMENTED` (B1) | `--force-cuda` flag + improved guidance. Needs Ywocp confirmation. |
| **HIGH** | #209 | Repetition cleaner new pattern | `FIX IMPLEMENTED` (C1) | Pattern #8 added. Needs weifu8435 confirmation. |
| **HIGH** | #218 | uv cu118 wheel version mismatch | `FIX IMPLEMENTED` (A3) | Exact match for WillChengCN's error. |
| **HIGH** | #219 | MossFormer2 speech separation 3D output | `FIX IMPLEMENTED` (C2) | Takes first separated source from 3D output. Needs user confirmation. |
| **MEDIUM** | #220 | Install stalls during PyTorch download | `NEEDS RESPONSE` | Guidance only, no code fix. Slow network. |
| **MEDIUM** | #217 | Can't find GUI exe after install | `NEEDS RESPONSE` | Explain exe location. |
| **MEDIUM** | #215 | Qwen3-ASR subtitle quality question | `NEEDS RESPONSE` | Expected behavior — different model architecture. |
| **MEDIUM** | #214 | Local LLM 502 server error | `NEEDS RESPONSE` | Acknowledge + link to v1.8.8 fixes. |
| **MEDIUM** | #207 | Settings persistence question | `NEEDS RESPONSE` | 0 responses — dup of #96. |
| **DONE** | #198 | MPS beam search | `AWAITING CONFIRMATION` | Fixed v1.8.8b1 (`9dc8f1f`). |
| **DONE** | #211 | urllib3 warning | `AWAITING CONFIRMATION` | Suppressed in v1.8.8b1. |

---

## Pending GitHub Actions

### Responded (Track D1 — DONE, 7 issues replied 2026-03-13)

| # | Response Summary | Status |
|---|-----------------|--------|
| **#220** | Explained PyTorch download stall, confirmed cu118 is correct | `AWAITING CONFIRMATION` |
| **#219** | Acknowledged speech separation bug, fix ready for v1.8.8 | `FIX IMPLEMENTED` |
| **#218** | Explained cu118 wheel issue, provided env var workaround | `FIX IMPLEMENTED` |
| **#217** | Explained exe location (`%LOCALAPPDATA%\WhisperJAV`), asked for install log | `AWAITING CONFIRMATION` |
| **#215** | Explained Qwen3-ASR generates fewer subtitles by design | `AWAITING CONFIRMATION` |
| **#214** | Acknowledged 502 error, linked to #212/#196, mentioned v1.8.8 improvements | `FIX IMPLEMENTED` |
| **#207** | Referenced #96, explained partial settings save, planned for future | `AWAITING CONFIRMATION` |

### Awaiting User Response

| # | Last Response | Waiting Since |
|---|--------------|---------------|
| #204 | Pointed to v1.8.7 release | 2026-03-10 |
| #203 | meizhong986 confirmed feature exists | 2026-03-08 |
| #201 | Pointed to v1.8.7 release | 2026-03-10 |
| #210 | meizhong986 + community responded | 2026-03-11 |
| #198 | Fixed in v1.8.8b1 | 2026-03-12 |

### Candidates for Closing (Track D2)

| # | Condition | Notes |
|---|-----------|-------|
| #211 | justantopair-ai confirms "nothing serious" + suppressed in v1.8.8b1 | Can close with note |
| #203 | Feature confirmed to exist, user asked "so it's already done?" | One more confirmation, then close |
| #204 | If yangming2027 confirms v1.8.7 fix works | Waiting since 2026-03-10 |
| #201 | If jl6564 confirms v1.8.7 fix works | Waiting since 2026-03-10 |
| #210 | If iop335577 confirms DNS issue resolved | Waiting since 2026-03-11 |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | #196 (closed), **#212**, **#214**, #132 | **#212** | `FIX IMPLEMENTED` — tactical fixes A1/A2/A4 + Ollama preview A6 |
| **MPS/Apple Silicon** | **#198** | **#198** | `AWAITING CONFIRMATION` |
| **Network/SSL/Install** | **#220** (NEW), **#218** (NEW), #217, #204, #210, #201 | **#204** | Mixed — #218 `FIX IMPLEMENTED`, #220/#217 need response, others awaiting |
| **GPU detection** | **#200**, **#213** | **#200** | `FIX IMPLEMENTED` (B1); #213 deferred |
| **GUI settings** | #96, **#207** | **#96** | `DEFERRED` (v1.9); #207 needs response |
| **Whisper quality** | **#209**, **#215** | **#209** | `FIX IMPLEMENTED` (C1); #215 needs response |
| **Speech enhancement** | **#219** (NEW) | **#219** | **NEW BUG** — needs investigation |
| **Startup warning** | **#211** | **#211** | `AWAITING CONFIRMATION` |
| **Ensemble** | #203 | **#203** | Resolved, awaiting final confirmation |
| **AMD/Intel GPU** | #142, #114, **#213** | Deferred | v1.9+ |
| **Translation providers** | #71, #43 | Deferred | v1.9+ |
| **i18n** | #180, #175 | Deferred | v1.9+ |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved — user installed NVIDIA Toolkit |
| #198 | MPS not used on M1 Mac | 2026-03-07 | Fixed v1.8.7b0 (detection) + v1.8.8b1 (beam search) |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user after v1.8.7b0 |
| #196 | Local Translation Errors | 2026-03-07 | Partial fix v1.8.7. **Still broken — see #212/#214** |
| #195 | UnicodeDecodeError audio extraction | 2026-03-08 | Fixed `55df512` |
| #194 | M4B file support | 2026-03-08 | Fixed `5769688` |

---

## v1.9+ Backlog

| # | Issue | Category |
|---|-------|----------|
| #96 | Full GUI settings persistence (pipeline tab) | Enhancement |
| #213 | Intel GPU (XPU) support | Platform |
| #205 | VibeVoice ASR | Feature |
| #206 | Grey out incompatible GUI options | Feature |
| #180/#175 | Multi-language GUI (i18n) | Enhancement |
| #114/#142 | DirectML / ROCm for AMD/Intel GPUs | Platform |
| #126 | Recursive directory + mirror output | Feature |
| #181 | Frameless window | Cosmetic |
| #43 | DeepL translation provider | Feature |
| #71 | Google Translate (no API key) | Feature |
| #198-FR | English in Ensemble/Translate dropdown | Feature |

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-03-13 | **rev6.** All Track A/B/C code complete (9 items). 3 new issues (#218 cu118 wheel, #219 MossFormer2 stereo, #220 install stall). Added status legend (FIX IMPLEMENTED vs FIX VERIFIED vs AWAITING CONFIRMATION). #218 already fixed by A3. #219 new bug (stereo→mono). #220 guidance only. Updated all clusters. Added Cluster G (Speech Enhancement). 7 issues need response (Track D1). |
| 2026-03-12 | rev5. v1.8.8b1 pre-release. 5 new issues (#213-#217). #198 MPS FIXED. #214 new LLM failure mode (502). #209 new hallucination sample. |
| 2026-03-11 | rev4. v1.8.7 RELEASED. 3 new issues (#210, #211, #212). #212 dup of #196. |
| 2026-03-09 | Groups B, C, D committed. #143, #209, #198 fixes shipped. |
| 2026-03-09 | Full evidence review: 14 screenshots, 3 log files, 50+ comments. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
