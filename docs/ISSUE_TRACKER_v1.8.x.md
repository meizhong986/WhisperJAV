# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-03-15 (rev8.1 — responded to #223, #225, #227) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **37 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX VERIFIED` | User confirmed the fix resolves their issue. Safe to close the GitHub issue. |
| `AWAITING CONFIRMATION` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **37** | 3 new since rev7 (#224, #225, #227) |
| New issues since rev7 | 3 | #224 (vocal separation analysis), #225 (GUI white screen), #227 (MPS hang on Transformers) |
| **NEEDS RESPONSE (no reply yet)** | 4 | #224, #225, #227, #223 |
| **KEY USER FEEDBACK RECEIVED** | 4 | #198 (MPS benchmark results), #132 (Kaggle Ollama confirmed), #128 (Gemma 3 proposal), #201 (self-resolved) |
| **AWAITING LOG (asked for install log)** | 3 | #217, #220, #221 |
| **Awaiting user response** | 9 | #200, #204, #207, #209, #210, #212, #214, #215, #218 |
| **Closable now** | 2 | #201 (self-resolved), #222 (responded, can close if no follow-up) |
| Feature requests (open) | 18 | See Cluster J |
| Deferred to v1.9+ | 11 | See v1.9+ Backlog |

---

## v1.8.8 Stable — RELEASED (2026-03-13)

All fixes shipped. Commit `b0f9d9b release: v1.8.8 stable`.

| ID | Issue(s) | Description | Status | Notes |
|----|----------|-------------|--------|-------|
| A1 | #212/#214/#132 | Diagnostic token logging (batch stats, "No matches" handler) | `SHIPPED` | #132: `--provider local` confirmed working on Kaggle |
| A2 | #212 | .subtrans stale settings override (version-stamp, auto-delete) | `SHIPPED` | Needs validation from destinyawaits/zhstark |
| A3 | **#218** | cu118 wheel version mismatch (`UV_SKIP_WHEEL_FILENAME_CHECK=1`) | `SHIPPED` | Needs WillChengCN to confirm |
| A4 | #212/#214 | Reduce max_tokens 2x→1x multiplier (2392→1820 for 8K ctx) | `SHIPPED` | Needs real-world validation |
| A5 | #214 | pornify.txt → sectioned format with format example | `SHIPPED` | Internal improvement |
| A6 | — | `--provider ollama` preview (auto-detect, num_ctx=8192, streaming) | `SHIPPED` | #132: confirmed working on Kaggle (after manual ollama install) |
| B1 | **#200** | NVML Optimus laptop fallback (`--force-cuda` flag, guidance text) | `SHIPPED` | Needs Ywocp to confirm |
| C1 | **#209** | Repetition cleaner pattern #8 (`sentence_phrase_repetition`) | `SHIPPED` | Needs weifu8435 to confirm |
| C2 | **#219** | MossFormer2_SS_16K 3D tensor crash | `SHIPPED` | Issue closed 2026-03-13 |
| — | #198 | Force greedy decoding (`num_beams=1`) on MPS | `SHIPPED` | MPS working but NOT accelerating Whisper (see Cluster B) |
| — | — | numpy 2.x migration | `SHIPPED` | |
| — | #211 | urllib3/chardet warning suppressed | `SHIPPED` | |
| — | — | Post-install verification timeout 30s→120s | `SHIPPED` | `63936e1` |
| — | — | RequestsDependencyWarning suppression | `SHIPPED` | `168e8b3` |

---

## Cluster Analysis

### Cluster A: Local LLM Translation — CRITICAL (4 issues + 1 contributor proposal)

**Issues**: #196 (closed), #212, #214, #132, #128 (contributor PR)
**Severity**: CRITICAL — affects ALL local LLM translation users across all platforms
**Status**: v1.8.8 tactical fixes shipped. #132 partially verified. **#128 has new Gemma 3 proposal from hyiip.**

| # | Title | Reporter | Platform | State | Detail |
|---|-------|----------|----------|-------|--------|
| **#214** | 1.8.7 localLLM fail | KenZP12 | Windows, cu128 | **OPEN** | gemma-9b, 502 error. `AWAITING CONFIRMATION` |
| **#212** | Regex Error - Local Translation v1.8.7 | destinyawaits | Linux, 5090 | **OPEN** | llama-8b AND gemma-9b both fail. `AWAITING CONFIRMATION` |
| **#196** | Local Translation Errors | zhstark | Ubuntu, 5090 32GB | CLOSED | Related to #212. |
| **#132** | Local LLM on Kaggle | TinyRick1489 | Kaggle | **OPEN** | **UPDATE 2026-03-15**: `--provider local` NOW WORKS on Kaggle v1.8.8. `--provider ollama` also works after manual ollama install (`curl -fsSL https://ollama.com/install.sh \| sh` + `ollama pull gemma3:12b`). Ollama not auto-installed — user had to do it manually. |
| **#128** | LLM context/batch sizing | hyiip | (contributor) | **OPEN** | **UPDATE 2026-03-14**: hyiip proposes replacing gemma-9b (8K context cap) with **Gemma 3 models** (128K context). Offers gemma-3-4b (~2.5GB Q4_K_M, beats Gemma 2 27B on benchmarks) and gemma-3-12b (~8.1GB Q4_K_M). Also proposes model-specific n_ctx in MODEL_REGISTRY and dynamic batch sizing. **Offers to implement.** |

**Key new developments:**
1. **#132 PARTIAL FIX VERIFIED**: `--provider local` works on Kaggle v1.8.8 (was broken before). `--provider ollama` works but requires manual ollama installation.
2. **#128 GEMMA 3 PROPOSAL**: hyiip (original LLM contributor) offers to upgrade from Gemma 2 9B (8K context, the root cause of most LLM failures) to Gemma 3 4B/12B (128K context). This would fix the context overflow issue at the source. **Decision needed: accept hyiip's contribution or do it ourselves.**

**Strategic direction update**: The Gemma 3 upgrade (#128) + Ollama migration together would solve the LLM translation cluster comprehensively. hyiip's proposal aligns perfectly with our v1.9.0 plans.

---

### Cluster B: MPS / Apple Silicon — CONFIRMED NOT ACCELERATING (3 issues)

**Issues**: #198 (closed), #227 (NEW)
**Severity**: HIGH — MPS detection works but provides NEGATIVE acceleration for Whisper
**Status**: Benchmark results confirm the problem. New issue #227 reports Transformers mode hangs on M1 MAX.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#198** | Transformers MPS crash | francetoastVN | **CLOSED** | MPS fix shipped v1.8.8. **BENCHMARK RESULTS (2026-03-15)**: MPS matmul 2.5x faster than CPU, but **Whisper transcription 0.2x** (MPS: 17.1s vs CPU: 2.8s for 15s audio). PyTorch 2.10.0. **VERDICT: MPS is NOT providing meaningful acceleration for Whisper.** MPS backend limitations for this workload type. |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN (NEW)** | `--mode transformers` hangs silently on Apple Silicon after model loads. Model: kotoba-whisper-bilingual. No output, no error, must force-kill. **NEEDS RESPONSE** |

**MPS analysis update (2026-03-15):**
- MPS **works** for basic tensor operations (2.5x matmul speedup)
- MPS **fails to accelerate** Whisper inference — 6x SLOWER than CPU
- Root cause: PyTorch MPS backend not optimized for attention-heavy transformer workloads (known limitation)
- #227 may be a different issue — model hangs entirely (not just slow), could be MPS-specific deadlock
- **Recommendation**: Default Transformers mode to CPU on Apple Silicon until PyTorch MPS matures. MPS for Whisper is a net negative.

---

### Cluster C: Network / Installation (8 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#225** | 白屏 (GUI white screen) | github3C | **OPEN (NEW)** | Program launches but shows white screen. Screenshots attached. | `RESPONDED` (2026-03-15) — asked for install log in Chinese. `AWAITING LOG` |
| **#222** | 字幕是日语... (how to get Chinese?) | libinghui20001231-debug | **OPEN** | User doesn't know about translation feature. Pointed to docs (2026-03-14). `AWAITING CONFIRMATION` |
| **#221** | 安装完后报错 (cublas64_12.dll missing) | libinghui20001231-debug | **OPEN** | GTX 1650, old driver 462. Community helped (yangming2027). | `AWAITING LOG` |
| **#220** | 安装卡着不动 (install stalls) | libinghui20001231-debug | **OPEN** | Install stalls during PyTorch download. Slow network. | `AWAITING LOG` |
| **#218** | 安装错误 (uv cu118 wheel) | WillChengCN | **OPEN** | uv rejects cu118 llama-cpp wheel | `SHIPPED` (A3). `AWAITING CONFIRMATION` |
| **#217** | 找不到WhisperJAV-GUI.exe | loveGEM + vimbackground | **OPEN** | 2 reporters, persists on v1.8.8. loveGEM installed to non-C drive. vimbackground tried admin. Both can't find GUI exe. | `AWAITING LOG` — **ESCALATED** |
| **#210** | 安装失败 DNS error | iop335577 | **OPEN** | DNS through proxy | `AWAITING CONFIRMATION` |
| **#204** | VPN/v2rayN SSL failures | yangming2027 | **OPEN** | HF hub SSL errors | `AWAITING CONFIRMATION` |
| **#201** | Install SSL cert error | jl6564 | **OPEN** | **SELF-RESOLVED (2026-03-15)**: `pip install pip-system-certs` fixed it. **CLOSABLE.** |

**#217 update**: loveGEM confirms: installed to non-C drive, `%localappdata%\temp\uv` has 5GB+ PyTorch files, installer CMD exits normally, but no GUI exe in install dir. Non-default drive install may be the trigger.

**#225 is NEW**: Different from #217. The program launches (window appears) but renders a white screen. Could be WebView2 issue or a slow first-load problem.

---

### Cluster D: GPU Detection (2 issues)

| # | Title | Reporter | State | Root Cause | Status |
|---|-------|----------|-------|------------|--------|
| **#200** | NVML "Driver Not Loaded" on Optimus laptop | Ywocp | **OPEN** | Dual GPU. | `SHIPPED` (B1: `--force-cuda`). `AWAITING CONFIRMATION` |
| **#213** | Intel GPU support | DDXDB | **OPEN** | Requests `torch.xpu` | `DEFERRED` to v1.9+ |

---

### Cluster E: GUI Settings Persistence (2 issues)

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#207** | 1.86不能保存设置 | q864310563 | **OPEN** | Responded (dup of #96). `AWAITING CONFIRMATION` |
| **#96** | Full settings persistence | sky9639 | OPEN | `DEFERRED` to v1.9 |

---

### Cluster F: Whisper Output Quality — EXPANDED (5 issues)

**Issues**: #223, #224 (NEW), #209, #215, #227 (NEW)
**Severity**: HIGH — competitive quality gap confirmed by multiple users + detailed technical analysis
**Status**: Quality improvement plan drafted (`docs/plans/V189_QUALITY_IMPROVEMENT_PLAN.md`). Not yet implemented.

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#224** | 人声分离分析 (vocal separation analysis) | yangming2027 | **OPEN (NEW)** | Detailed technical breakdown of why XXL beats WhisperJAV. Argues vocal separation is #1 gap. User's analysis is partially incorrect. | `DEFERRED RESPONSE` — will respond later with corrected analysis |
| **#223** | Faster Whisper XXL comparison | weifu8435 | **OPEN** | Claims XXL more accurate. 10 comments with screenshots. | `RESPONDED` (2026-03-15) — acknowledged quality issue (int8→fp16 fix coming), XXL ensemble integration planned, asked for usage tips |
| **#209** | Single subtitle very long (repetition) | weifu8435 | **OPEN** | `SHIPPED` (C1: pattern #8). `AWAITING CONFIRMATION` |
| **#215** | Qwen3-ASR subtitle quality | yangming2027 | **OPEN** | Responded — expected behavior. `AWAITING CONFIRMATION` |
| **#227** | M1 MAX Transformers hang | dadlaugh | **OPEN (NEW)** | Transformers mode hangs on Apple Silicon. | `RESPONDED` (2026-03-15) — CPU workaround (`--hf-device cpu`), MPS test script referenced (needs manual attachment to issue). `AWAITING CONFIRMATION` |

**Quality gap analysis — synthesized from #223, #224, and our AP1/AP2 audits:**

The quality gap vs Faster Whisper XXL has **two layers**:

**Layer 1 — Parameter defaults (fixable now, ~70% of gap):**
- Q1: Model `large-v2` → XXL uses `large-v3` (10-20% better Japanese)
- Q2: `compute_type="auto"` (int8_float16) → XXL uses `float16` (lossless)
- Q3: `beam_size=2, best_of=1` → XXL uses `beam_size=5, best_of=5`
- Q4-Q6: VAD failover, scene detection, CPS filter issues
- **Fix plan**: `docs/plans/V189_QUALITY_IMPROVEMENT_PLAN.md` — ~45 lines of changes

**Layer 2 — Vocal separation (architectural, longer-term):**
- yangming2027 (#224) argues this is the #1 factor
- XXL integrates UVR MDX-Net (Kim_vocal_v2) for vocal/BGM separation before ASR
- WhisperJAV has speech enhancement backends (ClearVoice, BS-RoFormer, ZipEnhancer) but none are vocal separation models
- WhisperJAV's `--speech-enhancer` is denoising/enhancement, not source separation
- **Gap**: No equivalent to UVR MDX-Net in current architecture
- **However**: WhisperJAV already has `bs-roformer` backend which IS a vocal isolation model. Need to verify if it's being used effectively.

---

### Cluster G: Speech Enhancement — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#219** | MossFormer2_SS_16K 3D tensor crash | anon12642 | **CLOSED** (2026-03-13) | `SHIPPED` in v1.8.8. |

---

### Cluster H: Startup Warning — COSMETIC

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#211** | 启动报错 (urllib3 warning) | WillChengCN | **OPEN** | `SHIPPED` in v1.8.8. Can close. |

---

### Cluster I: Ensemble Mode — RESOLVED

| # | Title | Reporter | State | Status |
|---|-------|----------|-------|--------|
| **#203** | Serial mode request | yangming2027 | **OPEN** | Already exists. Can close. |

---

### Cluster J: Feature Requests

| # | Title | Reporter | State | Summary | Target |
|---|-------|----------|-------|---------|--------|
| **#224** | Vocal separation (UVR MDX-Net) | yangming2027 | OPEN (NEW) | Detailed analysis of XXL's vocal separation advantage | **Investigate** |
| **#223** | Faster Whisper XXL comparison | weifu8435 | OPEN | Quality gap with XXL | **v1.8.9 (params) + future (vocal sep)** |
| **#213** | Intel GPU (XPU) support | DDXDB | OPEN | torch.xpu via PyTorch XPU wheels | v1.9+ |
| **#206** | Grey out incompatible options | techguru0 | OPEN | Block incompatible GUI choices | v1.9+ |
| **#205** | VibeVoice ASR | kylesskim-sys | OPEN | Microsoft VibeVoice. VRAM too high. | v1.9+ |
| **#181** | Frameless window | QQ804218 | OPEN | Cosmetic | v1.9+ |
| **#180** | Multi-language GUI | QQ804218 | OPEN | Full i18n | v1.9+ |
| **#175** | Chinese GUI | yangming2027 | OPEN | Subset of #180 | v1.9+ |
| **#164** | MPEG-TS + Drive | hosmallming | OPEN | Format + cloud | Backlog |
| **#142** | AMD Radeon | MatthaisUK | OPEN | FishYu-OWO AMD workaround (CTranslate2 ROCm wheels) | v1.9+ |
| **#128** | Gemma 3 model upgrade | hyiip | OPEN | 128K context Gemma 3 4B/12B models. **Contributor offers to implement.** | **v1.8.9 or v1.9.0** |
| **#126** | Recursive directory | jl6564 | OPEN | Walk subdirs, mirror output | v1.9+ |
| **#114** | DirectML | SingingDalong | OPEN | AMD/Intel GPU via torch-directml | v1.9+ |
| **#99** | 4GB VRAM guidance | hosmallming | OPEN | Log VRAM, recommend settings | Backlog |
| **#71** | Google Translate (free) | x8086 | OPEN | Fragile API | v1.9+ |
| **#59** | Feature plans | meizhong986 | OPEN | Meta roadmap | Keep open |
| **#51** | Batch translate wildcard | lingyunlxh | OPEN | Glob in translate CLI | Backlog |
| **#49** | Output to source folder | meizhong986 | OPEN | Docs gap | Backlog |
| **#44** | GUI drag-drop | lingyunlxh | OPEN | Filename vs path | Backlog |
| **#43** | DeepL provider | teijiIshida | OPEN | Non-LLM adapter | v1.9+ |
| **#33** | Linux pyaudio docs | org0ne | OPEN | Documentation | Backlog |
| **#198-FR** | English in Ensemble/Translate dropdown | francetoastVN | (comment) | Feature request | Backlog |

---

## Tally

### By Resolution Status

| Status | Count | Issues |
|--------|------:|--------|
| **NEEDS RESPONSE (new/unresponded)** | 1 | #224 (deferred — response planned later) |
| **AWAITING LOG (install log requested)** | 4 | #217, #220, #221, #225 |
| **AWAITING CONFIRMATION** | 11 | #200, #204, #207, #209, #210, #212, #214, #215, #218, #223, #227 |
| **CLOSABLE NOW** | 2 | #201 (self-resolved), #222 (responded) |
| **KEY FEEDBACK RECEIVED (action needed)** | 3 | #198 (MPS benchmark), #132 (Kaggle works), #128 (Gemma 3 proposal) |
| **Feature requests (open)** | 21 | See Cluster J |
| **DEFERRED to v1.9+** | 11 | #96, #205, #206, #213, #180, #175, #181, #142, #114, #126, #43 |

### By Priority (Active Work)

| Priority | # | Issue | Status | Why |
|----------|---|-------|--------|-----|
| **CRITICAL** | #217 | GUI exe missing after install | `ESCALATED` | 2 reporters, persists on v1.8.8, non-C drive install. Potential installer bug. |
| **CRITICAL** | #223/#224 | Quality gap vs Faster Whisper XXL | `RESPONDED + PLAN DRAFTED` | #223 responded (2026-03-15). Plan at `docs/plans/V189_QUALITY_IMPROVEMENT_PLAN.md`. |
| **HIGH** | #128 | Gemma 3 model upgrade proposal | `CONTRIBUTOR READY` | hyiip (original LLM author) offers to implement. Fixes root cause of LLM context overflow. |
| **HIGH** | #227 | MPS Transformers hang on M1 MAX | `RESPONDED` | CPU workaround given. MPS test script needs attachment. `AWAITING CONFIRMATION` |
| **HIGH** | #225 | GUI white screen after install | `RESPONDED` | Asked for install log. `AWAITING LOG` |
| **HIGH** | #198 | MPS NOT accelerating Whisper | `CONFIRMED` | Benchmark: 0.2x CPU speed. Need to decide: disable MPS for Whisper or document limitation. |
| **MEDIUM** | #132 | Kaggle LLM — works but ollama not auto-installed | `PARTIAL FIX` | `--provider local` works. `--provider ollama` needs manual install. |
| **MEDIUM** | #201 | SSL cert — self-resolved | `CLOSABLE` | User fixed with `pip install pip-system-certs`. |
| **LOW** | #211/#203 | Already resolved | `CLOSABLE` | Can close. |

---

## Pending GitHub Actions

### Issues Needing Response

| # | Action Needed | Priority |
|---|--------------|----------|
| **#224** | Deferred. User's analysis is partially incorrect (scientifically). Will respond later with corrected analysis. | MEDIUM |

### Issues Responded (2026-03-15)

| # | Action Taken |
|---|-------------|
| **#227** | CPU workaround (`--hf-device cpu`), MPS test script referenced. **NOTE**: test script not pushed to repo — needs manual file attachment. |
| **#225** | Asked for install log in Chinese. |
| **#223** | Acknowledged quality issue (int8→fp16 fix coming in next release), XXL ensemble integration planned, asked for usage tips. |

### Candidates for Closing

| # | Condition | Notes |
|---|-----------|-------|
| **#201** | Self-resolved — `pip install pip-system-certs` | Can close with acknowledgment |
| **#222** | Responded with docs + translation instructions | Can close if no further questions |
| **#211** | Already closed | Done |
| **#203** | Already closed | Done |

### Decisions Needed

| # | Decision | Options |
|---|----------|---------|
| **#128** | Accept hyiip's Gemma 3 contribution? | (a) Accept PR for v1.8.9, (b) Coordinate for v1.9.0, (c) Do it ourselves |
| **#198/#227** | MPS strategy for Whisper? | (a) Disable MPS for Whisper, default to CPU, (b) Keep MPS but warn about performance, (c) Investigate further |
| **#224** | Vocal separation integration? | (a) Document BS-RoFormer as existing alternative, (b) Add UVR MDX-Net, (c) Defer |

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Local LLM translation** | #196 (closed), **#212**, **#214**, **#132**, **#128** | **#128** | #132 partial fix verified. #128 Gemma 3 proposal pending. |
| **MPS/Apple Silicon** | #198 (closed), **#227** (NEW) | **#227** | MPS confirmed not accelerating Whisper. #227 is new hang issue. |
| **Network/SSL/Install** | **#225** (NEW), **#222**, **#221**, **#220**, **#218**, **#217**, #204, #210, #201 | **#217** | #217 ESCALATED. #225 NEW (white screen). #201 self-resolved. |
| **GPU detection** | **#200**, **#213** | **#200** | `SHIPPED` v1.8.8; #213 deferred |
| **GUI settings** | #96, **#207** | **#96** | `DEFERRED` (v1.9); #207 responded |
| **Whisper quality** | **#224** (NEW), **#223**, **#209**, **#215** | **#223** | Quality plan drafted. #224 adds vocal separation analysis. |
| **Speech enhancement** | #219 (closed) | — | Resolved |
| **AMD/Intel GPU** | #142, #114, **#213** | Deferred | v1.9+ — FishYu-OWO AMD workaround on #142 |
| **Translation providers** | #71, #43 | Deferred | v1.9+ |
| **i18n** | **#222**, #180, #175 | **#180** | v1.9+ |
| **Onboarding** | **#220**, **#221**, **#222** | — | Same user, 3 issues across install→run→use journey |

---

## Recently Closed Issues

| # | Title | Closed | Resolution |
|---|-------|--------|------------|
| **#219** | MossFormer2_SS_16K 3D tensor crash | 2026-03-13 | Fixed v1.8.8 |
| #208 | LLM server AssertionError | 2026-03-09 | Self-resolved |
| #198 | MPS beam search + detection | 2026-03-07 | Fixed v1.8.8. MPS working but not accelerating Whisper (confirmed 2026-03-15). |
| #197 | Installation problem v1.8.6 | 2026-03-08 | Closed by user |
| #196 | Local Translation Errors | 2026-03-07 | Partial fix. See #212/#214. |
| #195 | UnicodeDecodeError audio extraction | 2026-03-08 | Fixed |
| #194 | M4B file support | 2026-03-08 | Fixed |

---

## Analysis & Recommendations (rev8 — 2026-03-15)

### Current Situation

Since rev7 (2026-03-14), four significant pieces of user feedback arrived:

1. **#198 MPS benchmark**: francetoastVN ran the standalone test. **MPS is 6x SLOWER than CPU for Whisper** (17.1s vs 2.8s). MPS matmul works (2.5x speedup) but the Whisper attention workload is a net negative on MPS. This is a PyTorch MPS backend limitation, not a WhisperJAV bug.

2. **#132 Kaggle confirmation**: TinyRick1489 confirms `--provider local` works on v1.8.8 Kaggle. `--provider ollama` also works after manual ollama installation. The auto-detection says "not installed" because ollama isn't bundled — user must install it themselves.

3. **#128 Gemma 3 proposal**: hyiip (the original LLM contributor) proposes upgrading from Gemma 2 9B (8K context, the root cause of most "No matches" failures) to Gemma 3 4B (128K context, 2.5GB, beats Gemma 2 27B). Also proposes model-specific n_ctx and dynamic batch sizing. **Offers to do the work.**

4. **#224 vocal separation analysis**: yangming2027 posted a detailed technical breakdown arguing WhisperJAV's #1 quality gap vs XXL is the lack of built-in vocal separation (UVR MDX-Net). Partially correct — but overstates it, as WhisperJAV has BS-RoFormer vocal isolation and the parameter fixes address most of the gap.

Three new issues opened: #225 (GUI white screen), #227 (MPS hang on M1 MAX), #224 (vocal separation analysis/feature request).

### Key Themes Emerging

**Theme 1: Quality is the #1 user concern.** Three issues (#223, #224, #209) from two power users (weifu8435, yangming2027) all point to transcription quality. This is no longer a single complaint — it's a pattern. The quality improvement plan addresses the parameter layer; the vocal separation gap is a bigger architectural question.

**Theme 2: Apple Silicon is problematic.** Two issues (#198, #227) confirm MPS is not working well for Whisper. The MPS backend is a net negative for inference speed. Need a strategy: either disable MPS for Whisper or clearly document the limitation.

**Theme 3: LLM translation has a path forward.** Between the Gemma 3 upgrade (#128), Ollama migration, and the `--provider local` fix in v1.8.8, the LLM cluster is close to resolution. hyiip's contribution could accelerate this significantly.

**Theme 4: Installation issues continue.** #217 (GUI exe missing), #225 (white screen), #220/#221 (new user struggles) — the installer remains the biggest source of friction for new users.

### Immediate Action Items (rev8)

| ID | Issue | Action | Priority | Status |
|----|-------|--------|----------|--------|
| **R8** | #227 | Respond: suggest `--hf-device cpu` workaround, ask for full log. Likely MPS deadlock. | **HIGH** | NEW |
| **R9** | #225 | Respond in Chinese: ask for Windows version, WebView2 status, install log. White screen = WebView2 issue. | **HIGH** | NEW |
| **R10** | #224/#223 | Respond: acknowledge quality analysis, mention v1.8.9 parameter fix plan, note BS-RoFormer exists for vocal isolation. | **MEDIUM** | NEW |
| **R11** | #128 | **Decision**: accept hyiip's Gemma 3 contribution? This would fix the LLM context overflow root cause. | **HIGH** | DECISION NEEDED |
| **R12** | #201 | Close with thank-you. Self-resolved. | **LOW** | CLOSABLE |
| **R13** | #198 | **Decision**: MPS strategy — disable for Whisper, or document limitation? | **MEDIUM** | DECISION NEEDED |
| R1 | #217 | Investigate installer bug. Still awaiting logs from users. | **CRITICAL** | AWAITING LOG |
| R7 | Various | Shipped fixes awaiting confirmation. | **LOW** | AWAITING |

### Strategic Recommendations (updated)

| ID | Recommendation | Rationale | Target | Status |
|----|---------------|-----------|--------|--------|
| **S5** | **Accept hyiip's Gemma 3 contribution (#128)** — Coordinate with hyiip on Gemma 3 4B/12B model integration. Fix the 8K context overflow at the source. Reduces LLM failure rate dramatically. | Root cause fix for #212/#214/#132. Contributor-driven = low effort. | **v1.8.9** | DECISION NEEDED |
| **S6** | **Implement v1.8.9 quality parameter fixes** — Execute `V189_QUALITY_IMPROVEMENT_PLAN.md` (model→large-v3, compute_type→float16, beam_size→5, etc.). ~45 lines of changes. | #223/#224 quality gap. Addresses ~70% of the gap vs XXL. | **v1.8.9** | PLAN DRAFTED |
| **S7** | **Disable MPS for Whisper, default to CPU on Apple Silicon** — MPS is 6x slower for Whisper. Enabling it hurts users. Keep MPS for future PyTorch improvements but don't use it now. | #198 benchmark, #227 hang. Net negative for users. | **v1.8.9** | NEW |
| **S8** | **Investigate BS-RoFormer as vocal separation answer to #224** — WhisperJAV already has BS-RoFormer backend. If it performs vocal isolation effectively, the #224 criticism about "no vocal separation" is already addressed — just needs documentation and promotion. | #224 vocal separation analysis. May already be solved. | **v1.8.9** | NEW |
| S1 | **Prioritize i18n (Chinese UI)** | Largest user segment. Reduces support burden. | v1.9.0 | NOT STARTED |
| S2 | **Audit compute_type defaults** | Subsumed by S6 (quality plan). | v1.8.9 | SUBSUMED |
| S3 | **Document AMD ROCm workaround** | FishYu-OWO workaround. | v1.9.0 | NOT STARTED |
| S4 | **First-run guided setup** | Onboarding pain. | v1.9.0+ | NOT STARTED |

### Issue Velocity Trend

| Period | New Issues | Closed | Net | Notes |
|--------|-----------|--------|-----|-------|
| 2026-03-08 to 2026-03-10 | 6 (#204-#210) | 5 | +1 | v1.8.7 release cycle |
| 2026-03-11 to 2026-03-13 | 8 (#211-#220) | 3 | +5 | v1.8.8 beta + stable |
| 2026-03-14 | 3 (#221-#223) | 1 | +2 | Post-release influx |
| 2026-03-15 | 3 (#224-#227) | 0 | +3 | Quality + MPS + install issues |

**Trend**: 3-4 new issues/day continues. Quality concerns are now the dominant theme, overtaking installation issues.

---

## Release & Roadmap Analysis (rev8 — 2026-03-15)

### v1.8.9 — Proposed Scope

**Theme: Quality + LLM reliability**

| Priority | Item | Issues | Effort | Impact |
|----------|------|--------|--------|--------|
| **P0** | Quality parameter fixes (large-v3, float16, beam_size=5) | #223, #224 | ~45 LOC | HIGH — closes ~70% of quality gap |
| **P0** | MPS → CPU default for Whisper on Apple Silicon | #198, #227 | ~10 LOC | HIGH — stops hurting Mac users |
| **P1** | Gemma 3 model upgrade (if hyiip contributes) | #128, #132, #212, #214 | External PR | HIGH — fixes LLM context root cause |
| **P1** | OllamaManager smart integration (already coded) | #132, #212, #214 | Testing only | HIGH — escape hatch from llama-cpp |
| **P2** | Installer investigation (#217 GUI exe) | #217, #225 | Investigation | HIGH for new users |
| **P2** | VAD failover + scene detection fixes (Q4, Q5) | #223 | ~15 LOC | MEDIUM — quality improvement |
| **P3** | CPS threshold fix (Q6) | #223 | 1 LOC | LOW — edge case |

**Estimated scope**: Small release focused on parameter tuning + MPS fix + LLM improvements. No new architecture.

### v1.9.0 — Proposed Scope

**Theme: Platform expansion + UX**

| Priority | Item | Issues | Notes |
|----------|------|--------|-------|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #128, #132, #212, #214 | Remove ~1500 LOC fragile code |
| **P0** | Chinese UI (i18n, at least partial) | #175, #180, #222 | Biggest support burden reducer |
| **P1** | AMD ROCm support (document + partial integration) | #142, #114, #213 | FishYu-OWO proved it works |
| **P1** | First-run setup wizard | #217, #220, #221, #225 | Onboarding quality |
| **P2** | GUI settings persistence (pipeline tab) | #96, #207 | Long-standing request |
| **P2** | Vocal separation investigation (BS-RoFormer or UVR) | #224 | Quality gap Layer 2 |
| **P3** | DirectML for Intel GPUs | #213 | Niche |

### v2.0+ — Backlog

| Item | Issues |
|------|--------|
| UVR MDX-Net native integration | #224 |
| Google Translate (no key) | #71 |
| DeepL provider | #43 |
| VibeVoice ASR | #205 |
| Frameless window | #181 |
| MPEG-TS + Google Drive | #164 |
| Batch translate wildcard | #51 |
| Recursive directory | #126 |

### Decision Matrix — What to Do Next

| Option | Description | Blocked? | Impact | Effort |
|--------|------------|----------|--------|--------|
| **A: Respond to new issues (#227, #225, #224, #223)** | Send responses to 4 unresponded issues | No | Immediate user goodwill | 30 min |
| **B: Close #201, respond to #128** | Housekeeping + coordinate with hyiip on Gemma 3 | No | Unblocks contributor | 15 min |
| **C: Implement v1.8.9 quality parameter fixes** | Execute V189 plan (large-v3, float16, beam_size, etc.) | No | Highest quality impact | 2-3 hrs |
| **D: Fix MPS → CPU default for Whisper** | Based on #198 benchmark results | No | Stops hurting Mac users | 30 min |
| **E: Investigate #217 installer bug** | Audit post-install script for non-C drive path issues | Partially (awaiting logs) | New user experience | 1-2 hrs |
| **F: Investigate #225 GUI white screen** | WebView2 detection, screenshots analysis | Need screenshots | New user experience | 1 hr |

**Recommended sequence**: A → B → C → D → E (respond first, then build)

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
| 2026-03-15 | **rev8.1** Responded to #223 (quality fix int8→fp16 coming, XXL ensemble planned), #225 (asked install log), #227 (CPU workaround, MPS test script — needs manual attachment since not pushed to repo). #224 response deferred. NOTE: `tests/standalone/test_mps_whisper.py` exists locally but is NOT committed/pushed — curl link in #227 won't work until pushed. |
| 2026-03-15 | **rev8.** 3 new issues (#224 vocal separation analysis, #225 GUI white screen, #227 MPS hang on M1 MAX). Key feedback: #198 MPS benchmark confirms Whisper 6x SLOWER on MPS than CPU. #132 `--provider local` works on Kaggle v1.8.8, `--provider ollama` works after manual install. #128 hyiip proposes Gemma 3 4B/12B upgrade (128K context). #201 self-resolved (`pip-system-certs`). Added Release & Roadmap Analysis section. |
| 2026-03-14 | **rev7.** v1.8.8 RELEASED. 3 new issues (#221-#223). #217 ESCALATED. #142 AMD ROCm workaround. Follow-ups sent on all stale issues. |
| 2026-03-13 | rev6. All Track A/B/C code complete (9 items). 3 new issues (#218-#220). |
| 2026-03-12 | rev5. v1.8.8b1 pre-release. 5 new issues (#213-#217). |
| 2026-03-11 | rev4. v1.8.7 RELEASED. 3 new issues (#210-#212). |
| 2026-03-09 | Groups B, C, D committed. Fixes shipped. |
| 2026-03-08 | v1.8.7b1 released with China network fixes. |
| 2026-02-27 | v1.8.5-hotfix2 released. |

---

*This tracker is a point-in-time analysis. For live status, see [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues).*
