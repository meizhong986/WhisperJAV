# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-12 (rev5 — v1.8.8b1 pre-release, 5 new issues since rev4) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **32 open** on GitHub

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **32** | Was 28 at rev4; 4 new (#213, #214, #215, #217), #200 still open |
| New issues since rev4 | 5 | #213 (Intel GPU), #214 (local LLM 502), #215 (Qwen3 subtitle quality), #217 (can't find GUI exe) |
| **Active bugs (need code work)** | 2 | #196/#212/#214 (LLM — CRITICAL), #198 (MPS beam search — FIX COMMITTED in v1.8.8b1) |
| **Active bugs (awaiting user confirmation)** | 3 | #204 (SSL fallback), #209 (repetition fix), #200 (NVML Optimus) |
| **Cosmetic / informational** | 3 | #211 (urllib3 warning), #207 (settings dup), #215 (Qwen3 question) |
| **New issues needing response** | 3 | #214 (local LLM 502), #215 (Qwen3 quality), #217 (GUI exe) |
| Fixed in v1.8.8b1 | 2 | #198 MPS beam search (greedy on MPS), numpy 2.x migration |
| Feature requests (open) | 15 | +1 new (#213 Intel GPU) |
| Deferred to v1.9+ | 11 | +1 (#213) |

### v1.8.8b1 Shipped Fixes
- **MPS beam search (#198)**: Force `num_beams=1` when device is MPS — commit `9dc8f1f`
- **numpy 2.x migration**: Hard requirement numpy>=2.0.0, all internal code updated — commits `d5b8296` + `dea0fa2`
- **np.int_ crash fix**: Remove `np.int_` usage in metadata_manager.py — commit `dea0fa2`
- **Installation improvements**: GUI launcher creation, env detection, `--local` flag
- **Suppress RequestsDependencyWarning**: urllib3/chardet version mismatch warning silenced

---

## Cluster Analysis

### Cluster A: Local LLM Translation — CRITICAL (4 issues)

**Issues**: #196, #212, #214 (NEW), #132
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**Status**: NOT FIXED — new report #214 confirms 502 server error on v1.8.7

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **OPEN (NEW)** | gemma-9b, 12.3 tps, batch auto-reduced 30→11, 3281 lines/8 scenes. Server 502 error immediately. |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | llama-8b AND gemma-9b both fail. User switched to DeepSeek cloud. Confirms CUDA doesn't work on Windows 11 either. |
| **#196** | Local Translation Errors | destinyawaits | Ubuntu, 5090 32GB | CLOSED | zhstark: v1.8.7b1, batch 30→11, still fails. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | TinyRick1489 posted debug log (2026-03-07). meizhong986 responded "working on fix." |

**Key new data from #214 (KenZP12, v1.8.7):**
- Server starts fine, model loads, speed measured at 12.3 tps
- Auto-batch reduction working (30→11)
- **But: 502 server error on first actual translation request**
- This is a DIFFERENT failure mode from #212 (which gets "No matches found")
- 502 = server crashed or timed out during the actual translation batch
- `max_tokens: 2392` (JAV/CJK-tuned cap IS applied)
- Streaming: True
- 3281 lines is a VERY large file (8 scenes) — but fails on first batch

**Two distinct failure modes identified:**
1. **502 Server Error** (#214): Server crashes/times out during translation. Possibly context window overflow crashing llama.cpp server.
2. **"No matches found"** (#212, #196, #132): Server responds but output is garbled/unparseable by PySubtrans regex.

Both trace back to the same root: **8K context window is too small for the translation prompt + batch**.

**This remains the #1 priority for v1.8.8 stable.**

---

### Cluster B: MPS / Apple Silicon (1 issue) — FIX SHIPPED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | CLOSED | **FIXED in v1.8.8b1** — `num_beams=1` on MPS (commit `9dc8f1f`). Awaiting user confirmation. |

**Additional**: francetoastVN asked (2026-03-11) about adding English translation option to Ensemble and AI SRT Translate dropdown menus. This is a **feature request**, not a bug.

---

### Cluster C: Network / Installation (4 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM | **OPEN (NEW)** | Can't find GUI exe after v1.8.7 install | **NEEDS RESPONSE** — likely user confusion about exe location vs launcher |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy | Responded, v1.8.7 fix |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors | Responded, v1.8.7 fix, awaiting confirmation |
| **#201** | Install SSL cert error | jl6564 | **OPEN** | Missing root CA certs | Responded, v1.8.7 fix |

---

### Cluster D: GPU Detection (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Laptop dual GPU (AMD iGPU + NVIDIA dGPU). NVML can't initialize in Optimus config. | **NEEDS WORK** — need nvidia-smi fallback in installer GPU detection |
| **#213** | Intel GPU support | DDXDB | **OPEN (NEW)** | Requests `torch.xpu` support via PyTorch XPU wheels | **Feature request** — deferred to v1.9+ |

**#200 analysis**: Detailed report from Ywocp. NVIDIA driver works fine for other AI apps, but NVML specifically fails with "Driver Not Loaded" on Optimus architecture. User suggests fallback detection (nvidia-smi, Device Manager). This is a legitimate installer bug — our GPU detection relies solely on NVML.

**Fix for v1.8.8 stable**: Add `nvidia-smi` fallback when NVML fails. Also add `--force-cuda` CLI flag for manual override.

---

### Cluster E: GUI Settings Persistence (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | **0 comments — NEEDS RESPONSE**. Dup of #96. |
| **#96** | Full settings persistence | sky9639 | OPEN | v1.9.0 — translation + ensemble done, pipeline tab remaining |

---

### Cluster F: Whisper Output Quality (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | **FIXED v1.8.7, actively testing.** User posted NEW hallucination example (2026-03-12): FC2-PPV-4025269 line 52. Needs analysis. |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN (NEW)** | Qwen3-ASR produces very few/small subtitles, bad for ensemble pass2 accuracy. **NEEDS RESPONSE.** |

**#209 update**: weifu8435 is stress-testing v1.8.7 (4 videos/day, 100+/month). Posted new hallucination sample on 2026-03-12. This may be a pattern not yet caught by the repetition cleaner — needs investigation.

**#215 analysis**: yangming2027 asking why Qwen3-ASR generates fewer subtitles than Whisper. This could be by design (Qwen3 is more conservative) or a configuration issue. Needs a knowledgeable response.

---

### Cluster G: Startup Warning — COSMETIC

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#211** | 启动报错 (urllib3 warning) | WillChengCN | **OPEN** | Benign. justantopair-ai confirms "nothing serious." **Suppressed in v1.8.8b1.** |

---

### Cluster H: Ensemble Mode — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#203** | Serial mode request | yangming2027 | **OPEN** | Feature already exists ("Finish each file"). meizhong986 confirmed (2026-03-08). yangming2027 asked "so it's already done?" — **may need one more confirmation reply.** |

---

### Cluster I: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#213** | Intel GPU (XPU) support | DDXDB | **OPEN (NEW)** | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | DirectML/ROCm | v1.9+ |
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
| **#198-FR** | English in Ensemble/Translate dropdown | francetoastVN | (comment) | Feature request from #198 comment | Backlog |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **CRITICAL bugs needing code work** | 1 cluster | #196/#212/#214/#132 (LLM translation — 5 reporters, 3 platforms) |
| **Installer bug needing code work** | 1 | #200 (NVML Optimus laptop detection) |
| **Fixed in v1.8.8b1, awaiting confirmation** | 2 | #198 (MPS), #211 (urllib3 warning suppressed) |
| **Fixed in v1.8.7, actively testing** | 1 | #209 (repetition — new sample posted) |
| **Awaiting user confirmation** | 3 | #204 (SSL), #203 (serial mode), #201 (SSL) |
| **Needs response (no comments)** | 3 | #214 (LLM 502), #215 (Qwen3 quality), #217 (GUI exe location) |
| **Needs response (has comments)** | 1 | #207 (settings dup — 0 responses) |
| **Feature requests (open)** | 18 | See Cluster I |
| **Cosmetic / DNS / responded** | 1 | #210 |
| **Deferred to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Effort | Why |
|----------|---|-------|--------|-----|
| **CRITICAL** | #196/#212/#214/#132 | LLM translation failures (502 + "No matches") | Medium-Large | 5 reporters, 3 platforms. #1 user pain point. Two failure modes. |
| **HIGH** | #200 | NVML Optimus laptop GPU detection | Small-Medium | Detailed report from Ywocp. Blocks GPU install on all Optimus laptops. |
| **HIGH** | #209 | New hallucination sample (v1.8.7) | Small | weifu8435 posted new example. May need additional regex pattern. |
| **MEDIUM** | #215 | Qwen3-ASR subtitle quality question | Tiny (reply) | yangming2027 — active community member. |
| **MEDIUM** | #217 | Can't find GUI exe after install | Tiny (reply) | Likely user confusion. Needs response. |
| **MEDIUM** | #214 | Local LLM 502 server error | Tiny (reply) | Same cluster as #212. Acknowledge + link to known issue. |
| **MEDIUM** | #207 | Settings persistence question | Tiny (reply) | 0 responses — dup of #96. |
| **DONE** | #198 | MPS beam search | — | Fixed v1.8.8b1 (`9dc8f1f`). Awaiting confirmation. |
| **DONE** | #211 | urllib3 warning | — | Suppressed in v1.8.8b1. |
| **WAITING** | #204 | SSL/China fallback | — | v1.8.7 shipped, awaiting confirmation |
| **WAITING** | #203 | Serial mode | — | Confirmed exists. May need one more reply. |
| **WAITING** | #201 | SSL cert error | — | v1.8.7 shipped, awaiting confirmation |

---

## v1.8.8b1 — Shipped Fixes

| Commit | Issue | Summary |
|--------|-------|---------|
| `9dc8f1f` | #198 | Force greedy decoding (`num_beams=1`) on MPS |
| `d5b8296` | — | numpy 2.x migration (all internal code) |
| `dea0fa2` | — | Remove `np.int_` usage (numpy 2.x crash) |
| `32c3711` | — | Phase 1a installer: env detection, `--local`, `--inexact` |
| `18be938` | — | GUI launcher creation in install scripts |
| `5cc5ad2` | — | Version bump to 1.8.8b1 |

---

## v1.8.8 Stable — Planned Work

| Priority | # | Description | Effort | Status |
|----------|---|-------------|--------|--------|
| **CRITICAL** | #196/#212/#214/#132 | LLM translation: dynamic batch sizing, retry with half batch, garbage output detection, 502 handling | Medium-Large | **Not started** — #1 priority |
| **HIGH** | #200 | NVML Optimus fallback: nvidia-smi detection + `--force-cuda` flag | Small-Medium | Not started |
| **HIGH** | #209 | Investigate new hallucination sample from weifu8435 | Small | Not started |
| **LOW** | #99 | Log GPU VRAM at INFO before model load | Tiny | Not started |

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

## Pending GitHub Actions

### Needs Response (not yet responded)

| # | Action | Priority |
|---|--------|----------|
| **#217** | Respond: explain where GUI exe is located (`%LOCALAPPDATA%\WhisperJAV`), or if source install, explain launcher file. Ask for install log. | **MEDIUM** |
| **#215** | Respond: explain Qwen3-ASR behavior — fewer subtitles is expected (different model architecture). Suggest using Whisper for pass2 if ensemble accuracy is the goal. | **MEDIUM** |
| **#214** | Respond: acknowledge 502 error, same root cause family as #212/#196. Being investigated for v1.8.8. Workaround: use cloud provider (DeepSeek). | **MEDIUM** |
| **#207** | Respond: reference #96, explain what IS saved (translation, ensemble) vs what isn't (pipeline). Planned for v1.9. | **MEDIUM** |

### Needs Follow-up

| # | Action | Priority |
|---|--------|----------|
| **#209** | Investigate new hallucination sample (FC2-PPV-4025269 line 52) posted 2026-03-12. | **HIGH** |
| **#203** | yangming2027 asked "so it's already done?" — may need one final confirmation. | **LOW** |

### Awaiting User Response

| # | Last Response | Waiting Since |
|---|--------------|---------------|
| #204 | Pointed to v1.8.7 release | 2026-03-10 |
| #201 | Pointed to v1.8.7 release | 2026-03-10 |
| #209 | User actively testing v1.8.7 | 2026-03-12 (new sample posted) |
| #210 | meizhong986 + community responded | 2026-03-11 |
| #198 | meizhong986 posted MPS fix coming | 2026-03-11 |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | **#196**, **#212**, **#214** (NEW), #132 | **#212** | **CRITICAL — 2 failure modes (502 + regex)** |
| **MPS/Apple Silicon** | **#198** | **#198** | **FIXED v1.8.8b1** — awaiting confirmation |
| **Network/SSL/Install** | #204, #210, #201 | **#204** | Responded, awaiting confirmation |
| **GPU detection** | **#200**, **#213** | **#200** | #200 needs fallback code; #213 is feature request |
| **GUI settings** | #96, **#207** | **#96** | Partial (v1.9); #207 needs response |
| **Whisper quality** | **#209**, **#215** | **#209** | #209 testing, new sample; #215 is question |
| **Startup warning** | **#211** | **#211** | **Suppressed in v1.8.8b1** |
| **Ensemble** | #203 | **#203** | Resolved, needs final confirmation |
| **Installer location** | **#217** | **#217** | Needs response |
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

## Changelog

| Date | Changes |
|------|---------|
| 2026-03-12 | **rev5.** v1.8.8b1 pre-release. 5 new issues (#213 Intel GPU, #214 LLM 502, #215 Qwen3 quality, #217 GUI exe location, #200 Optimus). #198 MPS FIXED (v1.8.8b1). #214 new LLM failure mode (502 vs regex). #209 new hallucination sample. Updated all clusters. |
| 2026-03-11 | rev4. v1.8.7 RELEASED. 3 new issues (#210, #211, #212). #212 dup of #196. #198 MPS still crashes. |
| 2026-03-09 | Groups B, C, D committed. #143, #209, #198 fixes shipped. |
| 2026-03-09 | Full evidence review: 14 screenshots, 3 log files, 50+ comments. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
