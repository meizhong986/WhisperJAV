# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-12 (rev33) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **54 open** on GitHub

---

## Status Legend

| Label | Meaning |
|-------|---------|
| `SHIPPED` | Fix released in a stable version. Waiting for user to test/confirm. |
| `FIX CODED` | Fix committed but not yet released. |
| `AWAITING REPLY` | Response given. Waiting for user reply. |
| `NEEDS RESPONSE` | Issue has no response or needs a follow-up reply. |
| `NEEDS FOLLOW-UP` | User replied after our response — needs another look. |
| `DEFERRED` | Moved to a future release. |

---

## Quick Stats

| Category | Count | Notes |
|----------|------:|-------|
| Total open on GitHub | **54** | +1 new: #287. |
| **v1.8.10.post3 RELEASED** | — | 2026-04-09. 25 commits since v1.8.10. Tagged + pushed + GitHub release published. |
| **v1.8.11 dev branch** | — | 2026-04-12. Ollama curated list fix + `--ollama-max-tokens` flag coded. |
| **NEEDS RESPONSE** | **1** | **#287** (all subtitles "!!" — new, needs investigation). |
| **NEEDS FOLLOW-UP** | **0** | All followed up. |
| **AWAITING REPLY** | 15 | #263, #286, #271, #268, #262, #280, #282, #281, #284, #259, #217, #265, #274, #279 |
| **FIX CODED (v1.8.11)** | 2 | #271 (curated model list + `--ollama-max-tokens`). |
| **Closed this session** | **1** | **#267** (user confirmed fix) |
| **Closed last session** | 5 | #234, #251, #258, #260, #285 (closed by reporter) |
| **SHIPPED in post3** | 6 | #272, #269, #265 (filter), #259, #267 (partial), #231 |
| Feature requests (open) | 30 | See Feature Requests section. |

---

## v1.8.10.post3 — RELEASED (2026-04-09)

See `installer/RELEASE_NOTES_v1.8.10.post3.md` for full details.

**Key changes**: 3 crash fixes (device="auto", length_penalty, single-segment), full ASR/VAD retune (hf1→hf3), Silero v3.1/v4.0 anti mega-group presets, post-processing overhaul (regex, punct, dedup), GUI customize modal audit (11 fixes), TEN VAD backend hardening, post-model logprob gate toggle, Silero v3.1 as default segmenter for balanced/fidelity, resolver_v3 versioned Silero name support, numba cache redirect, XXL model flag fix, hallucination filter bundling, translation overwrite fix, Colab numpy2 migration.

**Default changes users will notice**:
- Default speech segmenter for balanced/fidelity: silero (v4.0) → **Silero v3.1**
- Default sensitivity in GUI ensemble: aggressive (unchanged but now with retuned params)
- ASR parameters significantly retuned across all 3 sensitivity levels

---

## Open Issues by Status

### Bugs / Active Issues

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#287** | All subtitles are "!!" with latest version | zoqapopita93 | `NEEDS RESPONSE` | **NEW 04-12.** All output subtitles contain only "!!". Screenshot attached showing config. Likely sanitizer/hallucination issue or ASR regression. P1 — needs investigation. |
| **#286** | CUDA kernel error on GTX 1050 Ti | techguru0 | `AWAITING REPLY` | Responded 04-11: humble acknowledgment of variant installer idea, noted backlog, re-asked if 2.5.1+cu118 worked. |
| **#285** | Batch/Scene length translation questions | oceanseamountain | `CLOSED` | **Closed by reporter 04-10** after my answer. |
| **#284** | v1.8.10 install stuck during PyTorch phase | qq73-maker | `AWAITING REPLY` | Responded 04-09: suggested post3 installer, asked about network/proxy. Two users affected. |
| **#282** | Why does Ollama need GitHub connection? | KenZP12 | `AWAITING REPLY` | Responded 04-09: explained gist fetch for instructions file, bundled fallback works offline. |
| **#281** | Ollama not detected in GUI | wlee15 | `AWAITING REPLY` | Responded 04-09: confirmed @vicecity2930 workaround, explained Server URL field. |
| **#280** | Qwen3-ASR TypeError: `check_model_inputs()` | zoqapopita93 | `AWAITING REPLY` | Responded 04-09: upstream `transformers` mismatch, suggested `pip install --no-deps transformers==4.49.0`. |
| **#274** | Pipeline aggregation mode question | cuixiaopi | `AWAITING REPLY` | Responded 04-09: explained 2-pass ensemble mode. |
| **#271** | Ollama translation model issues | justantopair-ai | `FIX CODED` | **v1.8.11**: curated model list fixed (removed thinking model, new defaults: gemma3:12b, qwen2.5:7b-instruct). `--ollama-max-tokens` CLI flag added. Responded 04-12 with fix confirmation. |
| **#268** | Thai + Korean translation targets | yedkung69-ctrl | `AWAITING REPLY` | Responded 04-11: acknowledged GUI gap, noted for v1.9.0 i18n work, mentioned CLI workaround. |
| **#267** | Stuck on Streaming features (Qwen+Semantic) | OrangeFarmHorse | `CLOSED` | **CLOSED 04-11.** User confirmed `NUMBA_DISABLE_JIT=1` workaround works. Numba post3 fix was insufficient — v1.8.11 needs deeper investigation. |
| **#265** | Hallucination + post-processing suggestions | yangming2027 | `AWAITING REPLY` | Responded 04-09: answered @zoqapopita93 (v2 vs v3, custom params), noted post3 post-processing improvements. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | Responded 04-11: explained Python doesn't auto-honor Windows system proxy (need HTTP_PROXY env var or TUN mode), pushed silero-v6.2 as cleanest bypass. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | User found cmd workaround. |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues | destinyawaits | `AWAITING REPLY` | Responded 04-09: overwrite fix shipped in post3, suggested upgrade. |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Pointed to v1.8.10 Ollama GUI. |
| **#251** | post2 launch failure (fastgit SSL) | zoqapopita93 | `AWAITING REPLY` | fastgit mirror SSL. 3 solutions. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | |
| **#233** | Translation error (local LLM) | WillChengCN | `RESPONDED` | Ollama recommended. |
| **#231** | Kaggle notebook error | fzfile | `SHIPPED` post3 | Colab fix shipped. |
| **#225** | GUI white screen | github3C | `STALE` | |
| **#217** | GUI.exe not found (China network) | loveGEM | `AWAITING REPLY` | Responded 04-09: gave @vimbackground pip install instructions for pre-downloaded whl. |

### Feature Requests (30)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#279** | Stash integration | LOW | Backlog |
| **#268** | Thai + Korean translation targets | LOW | v1.9+ |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 |
| **#264** | Model download location customization | LOW | Responded |
| **#262** | Cohere Transcribe model | LOW | **NEW (04-11):** @anon12642 confirms Cohere demo good on anime, hallucinates on non-speech, would benefit from VAD. v1.9.x scope. |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | Docs / FAQ |
| **#248** | Diarization | MEDIUM | v1.9+ |
| **#247** | Docker support | LOW | Backlog |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#230** | Standalone merge module | HIGH | v1.9.0 |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
| **#128** | Gemma 3 models | HIGH | v1.9.0 |
| **#126** | Recursive directory | LOW | Backlog |
| **#114** | DirectML | MEDIUM | v1.9+ |
| **#99** | 4GB VRAM guidance | LOW | Backlog |
| **#96** | Settings persistence | MEDIUM | v1.9.0 |
| **#71** | Google Translate (free) | LOW | Backlog |
| **#59** | Feature plans (meta) | — | Keep open |
| **#51** | Batch translate | LOW | Backlog |
| **#49** | Output to source folder | LOW | Backlog |
| **#44** | GUI drag-drop | LOW | Backlog |
| **#43** | DeepL provider | LOW | v1.9+ |
| **#33** | Linux pyaudio docs | LOW | Backlog |

---

## Immediate Actions — Post-Release Priority

### Completed (rev26)

All P0 and P1 responses posted on 2026-04-09:
- #284 (install stuck), #280 (Qwen3-ASR), #267 (numba follow-up)
- #281 (Ollama UX), #282 (Ollama GitHub), #271 (thinking models), #259 (overwrite fix)

### Outstanding Actions (rev33)

**v1.8.11 dev branch started (2026-04-12):**
- ✓ #271 **FIX CODED**: curated Ollama model list fixed — removed Shisa-v2.1-Qwen3-8B (thinking model), new defaults: gemma3:12b (#1), qwen2.5:7b-instruct (#2). Responded to thread.
- ✓ #271 **FIX CODED**: `--ollama-max-tokens` CLI flag added (both entry points + service layer). Responded to thread.
- **#287 NEEDS RESPONSE** — new bug (04-12): all subtitles are "!!". Needs investigation.

**Remaining v1.8.11 candidates** (from ranked list): #263 torch.hub China fix, #267 numba architectural fix, #280 Qwen3-ASR pin, sanitizer hardening (#287 investigation), #265 hallucination improvements.

**15 issues AWAITING REPLY from users.** Normal turnaround 1-3 days.

### Architectural concerns

1. **#267 RESOLVED — numba JIT cache fix in post3 was insufficient.** OrangeFarmHorse confirms `NUMBA_DISABLE_JIT=1` env var works. So the post3 `ensure_numba_cache_dir()` fix did NOT fully solve the problem — the workaround is to disable JIT entirely. **Action for v1.8.11**: detect admin-installed scenarios and set `NUMBA_DISABLE_JIT=1` automatically, or invoke `setup_console()` even earlier in the import chain so the cache redirect actually takes effect before librosa imports. Need to investigate WHY the redirect didn't work.

2. **#267 and #263 had DIFFERENT root causes** — despite being initially cross-referenced as the same hang. #267 was numba JIT (resolved by NUMBA_DISABLE_JIT=1). #263 is Chinese network/GFW (torch.hub Silero VAD → api.github.com → 403). Two distinct problems that produced similar symptoms.

3. **torch.hub Silero VAD fragility — Chinese network issue (#263)** — silero-v3.1 and silero-v4.0 backends both call `torch.hub.load()` which hits `api.github.com/repos/snakers4/silero-vad/branches/...`. **GitHub API is blocked/throttled from China**, causing HTTP 403. PyTorch's `_validate_not_a_forked_repo` has a latent `KeyError` bug that masks the real network error. Solutions for v1.8.11:
   - Pre-bundle Silero v3.1/v4.0 model files with the installer (no runtime GitHub call)
   - OR switch default to silero-v6.2 (pip-installed `silero-vad` package, no GitHub API)
   - OR catch torch.hub failure and auto-fallback to silero-v6.2
   - Add `skip_validation=True` to torch.hub.load() if PyTorch version supports it
   - OR set HTTP_PROXY/HTTPS_PROXY env vars from Windows system proxy settings (auto-detect)

4. **Default segmenter recently changed to silero-v3.1** — may have **increased Chinese-user exposure** to the torch.hub problem. Worth reconsidering for v1.8.11.

5. **Install hardening track for v1.9.0 must explicitly cover China-network scenarios** — #284, #217, #251, #261, #263 all touch this. It's a coherent cluster that needs systematic treatment, not piecemeal fixes.

6. ~~**Ollama `--max-tokens` not exposed to CLI**~~ — **FIXED in v1.8.11 dev** (commit 275adb5). `--ollama-max-tokens` flag added to both CLI entry points.

7. **Translate tab source language dropdown hard-coded to CJK** — needs GUI fix to pass through all Whisper languages. v1.9.0.

8. **Per-GPU-arch installer variants question (#286)** — techguru0 referenced VideOCR's approach of separate cu118 (10-series) and cu129 (16-50 series) installers. Worth considering for v1.9.0 install hardening, but adds maintenance burden.

9. ~~**Curated Ollama model list is broken (#271)**~~ — **FIXED in v1.8.11 dev** (commit 275adb5). Removed `Shisa-v2.1-Qwen3-8B` (thinking model). New curated list: `gemma3:12b` (#1), `qwen2.5:7b-instruct` (#2), `qwen2.5-abliterate` (#3), `dolphin-llama3` (#4). All instruct-only.

10. **Discussion #257 — community translation knowledge** — referenced by @justantopair-ai in #271. Active community thread about translation models, settings, and findings. Worth mining for v1.9.0 hardening insights and Ollama model recommendations. **Future hardening reference.**

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Ollama** | #282, #281, #271, #259, #255, #233, #132 | #132 | #259 fix shipped. #271 root cause known. #281/#282 need responses. |
| **Install stuck** | #284, #217 | #284 | 2 users. post3 installer may help. |
| **Semantic hang / numba** | #267, #263 | #267 | Fix shipped in post3 but reported insufficient. Needs testing on post3. |
| **Qwen3-ASR** | #280, #215 | #280 | Upstream transformers mismatch. |
| **Diarization** | #248, #252 | #248 | v1.9+. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Hallucination** | #265, #246 | #265 | Active community. post3 has improved sanitizer. |
| **Merge module** | #230, #265 | #230 | v1.9.0 P1. |

---

## Release Roadmap

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. |
| **P0** | Unified CLI override layer | #269 | Small | Standard/ensemble divergence. |
| **P1** | Ollama model recommendation fix | #271 | Small | Only recommend instruct models in GUI. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed rules + v2 word list. |
| **P1** | Post-processing polish (trailing period, leading dash, line wrapping) | #265 | Small | 3 concrete suggestions. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment. |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean. |
| **P2** | Semantic scene detection hang (deeper investigation) | #267 | Medium | numba redirect may be insufficient. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation | #224, #254 | Medium | BS-RoFormer / UVR. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combos. |
| **P2** | Qwen3-ASR transformers compatibility | #280 | Medium | Pin or adapt to API change. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR. |
| **P3** | Stash integration | #279 | Medium | Niche request. |

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Community contributions |
| **Architecture** | Web-based UI (replace pywebview) | Persistent issues (#225, #240) |
| **Translation** | Translation model compatibility layer | #271 — detect thinking models |
| **Scale** | Batch processing dashboard | Queue, progress, ETA |
| **Platform** | China mirror support | #217, #251 — PyTorch behind GFW |
| **Platform** | Linux native installer | AppImage or deb |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-04-12** | **rev33.** 53→54 open (+1 new: #287). **v1.8.11 dev branch started.** Two fixes coded: (1) Ollama curated model list — removed thinking model, new instruct-only defaults. (2) `--ollama-max-tokens` CLI flag added to both entry points. #271 responded with fix confirmation. #287 new bug identified (all subtitles "!!", NEEDS RESPONSE). |
| 2026-04-11 | **rev32.** 54→53 open. **#267 CLOSED** (user confirmed `NUMBA_DISABLE_JIT=1` works). All 6 follow-ups responded: #263 (proxy + silero-v6.2), #286 (humble variant ack), #271 (CLI flag commit + Shisa-Qwen3 curated list bug), #268 (GUI gap), #262 (Cohere thanks). **Found: curated Ollama #1 model is a Qwen3 thinking model — bug for v1.8.11.** Discussion #257 noted as future hardening reference. |
| 2026-04-11 | **rev31.** 54 open. #267 user confirmed fix. #263 user replied with proxy question. 5 new follow-ups identified. |
| 2026-04-10 | **rev30.** 55→54 open. #285 closed by reporter. #263 corrected diagnosis (GFW). Posted silero-v6.2 workaround. |
| 2026-04-10 | **rev29.** 55 open. All P0/P1/P2 responded. Board clean. |
| 2026-04-10 | **rev28.** 53→55 open (+2). 2 new issues: #286, #285. New comments on #271, #268. |
| 2026-04-09 | **rev27.** 57→53 open. All P2 responses posted. Closed 4 stale: #234, #251, #258, #260. |
| 2026-04-09 | **rev26.** All P0/P1 responses posted: #284, #280, #267, #281, #282, #271, #259. Found missed #217 reply (10 days stale). |
| 2026-04-09 | **rev25.** v1.8.10.post3 RELEASED — 25 commits, 3 crash fixes, full ASR/VAD retune, Silero v3.1 default, post-processing overhaul, GUI audit, TEN backend, forensic tools. Tagged, pushed, GitHub release published (stable). |
| 2026-04-09 | **rev24.** 57 open (+2). 7 new issues. 14 unpushed commits identified. |
| 2026-04-01 | **rev23.** 55 open. #267 ROOT CAUSE FOUND. #259 overwrite fix. #271 instruct models work. |
| 2026-04-01 | **rev22.** 55 open. XXL fix. Translation overwrite fix. |
| 2026-03-31 | **rev21.** 54→55 open. New: #271, #272. Closed: #269. |
| 2026-03-31 | **rev20.** 55→54 open. New: #265, #267, #268, #269. Closed: #218, #221, #236, #244. |
| 2026-03-30 | **rev19.** v1.8.10 released. |
| 2026-03-29 | **rev18.** 55 open. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |

---
