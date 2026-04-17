# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-17 (rev36) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **61 open** on GitHub

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
| Total open on GitHub | **61** | +5 new: #289, #290, #291, #292, #293. +4 previously untracked: #223, #227, #232, #246. |
| **v1.8.10.post3 RELEASED** | — | 2026-04-09. Stable. |
| **v1.8.11 dev branch** | — | 2026-04-12. Ollama curated list fix + `--ollama-max-tokens` flag coded. No new commits since rev33. |
| **NEEDS RESPONSE** | **0** | All responded 04-17. |
| **NEEDS FOLLOW-UP** | **2** | #217 (user gave up after whisper install also failed — template posted, awaiting), #247 (foxfire881 new Docker NAS ask 04-17). |
| **AWAITING REPLY** | 18 | **+5 posted 04-17 (China-network bilingual template)**: #289, #284, #217, #263, #261. Plus: #286, #287, #268, #262, #280, #282, #259, #265, #274, #279, #271, #258, #260. Normal turnaround 1-3 days. |
| **FIX CODED (v1.8.11)** | 3 | #271 (curated model list + `--ollama-max-tokens`), #291 (install_colab.sh `llm` extra). |
| **Closed this session** | **1** | **#281** (reporter fixed via uninstall+reinstall with admin privileges 04-16). |
| **SHIPPED in post3** | 6 | #272, #269, #265 (filter), #259, #267 (partial), #231. |
| Feature requests (open) | 34 | See Feature Requests section. |

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
| **#291** | Google Colab Step 3 Translation — `ModuleNotFoundError: starlette_context` | ktrankc | `FIX CODED` | **Root cause verified**: `install_colab.sh` line 176 did not include `[llm]` extra, so `starlette-context` + `sse-starlette` + `uvicorn` + `fastapi` + `pydantic-settings` were missing. **Fix: added `llm` to extras list for v1.8.11.** Responded 04-17 with manual workaround + migration-to-ollama path. |
| **#290** | Google Colab Pass 2 Error — "Killed" after MossFormer2 load | ktrankc | `AWAITING REPLY` | Responded 04-17: "Killed" is ambiguous (OOM killer vs Colab runtime kill vs cgroup) — asked for Colab tier, file size, resource panel observation; recommended trying `zipenhancer` or `ffmpeg-dsp` to test ClearVoice memory hypothesis. |
| **#289** | PyTorch CUDA 12.8 install timeouts | coco7887 | `AWAITING REPLY` | Responded 04-17 with bilingual China-network template (community ask + mirror roadmap + whl workaround). |
| **#287** | All subtitles are "!!" with latest version | zoqapopita93 | `AWAITING REPLY` | Responded 04-12 asking user to switch segmenter VAD to TEN-VAD or Silero v3.1. Awaiting user reply. |
| **#286** | CUDA kernel error on GTX 1050 Ti | techguru0 | `AWAITING REPLY` | User replied 04-14 "will try it" re: PyTorch 2.5.1 downgrade. Waiting on result. |
| **#284** | v1.8.10 install stuck during PyTorch phase | qq73-maker | `AWAITING REPLY` | Follow-up 04-17: posted bilingual China-network template (community ask + mirror roadmap). |
| **#282** | Why does Ollama need GitHub connection? | KenZP12 | `AWAITING REPLY` | Responded 04-09: gist fetch for instructions, bundled fallback works offline. |
| **#280** | Qwen3-ASR TypeError: `check_model_inputs()` | zoqapopita93 | `AWAITING REPLY` | Responded 04-09: upstream `transformers` mismatch, suggested `pip install --no-deps transformers==4.49.0`. |
| **#274** | Pipeline aggregation mode question | cuixiaopi | `AWAITING REPLY` | Responded 04-09: explained 2-pass ensemble mode. |
| **#271** | Ollama translation model issues | justantopair-ai | `FIX CODED` | **v1.8.11**: curated model list fixed. `--ollama-max-tokens` CLI flag added. Responded 04-12. |
| **#268** | Thai + Korean translation targets | yedkung69-ctrl | `AWAITING REPLY` | Responded 04-11: GUI gap noted for v1.9.0. |
| **#265** | Hallucination + post-processing suggestions | yangming2027 | `AWAITING REPLY` | Responded 04-09: answered @zoqapopita93, noted post3 improvements. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template (earlier user found cmd workaround). |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues | destinyawaits | `AWAITING REPLY` | Responded 04-09: overwrite fix shipped in post3. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Pointed to v1.8.10 Ollama GUI. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | |
| **#233** | Translation error (local LLM) | WillChengCN | `RESPONDED` | Ollama recommended. |
| **#231** | Kaggle notebook error | fzfile | `SHIPPED` post3 | Colab fix shipped. Reopened by #290/#291 patterns. |
| **#227** | M1 Max transformer mode issues | dadlaugh | `STALE` | From 2026-03-17. Apple Silicon transformer mode. Not actively tracked. |
| **#225** | GUI white screen | github3C | `STALE` | |
| **#217** | GUI.exe not found (China network) | loveGEM | `AWAITING REPLY` | Follow-up 04-17: bilingual China-network template posted (community ask + mirror roadmap + whl workaround). User vimbackground had given up after whisper install also failed. |

### Feature Requests (34)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#293** | Translation context / Whisper prompt feature | LOW | v1.9+ — **NEW 04-16.** User asks if movie descriptions/names as prompt would improve accuracy. Whisper and Qwen3-ASR both support initial prompt. Valid P2 ask. |
| **#292** | Low GPU utilization with higher-end models | LOW | v1.9+ — **NEW 04-17.** User running supergemma4-26b MoE on RTX 4090. Community (@justantopair-ai) already answered with good technical explanation. Probably just needs a thumbs-up from owner. |
| **#279** | Stash integration | LOW | Backlog |
| **#268** | Thai + Korean translation targets | LOW | v1.9+ |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 |
| **#264** | Model download location customization | LOW | Responded |
| **#262** | Cohere Transcribe model | LOW | v1.9.x |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | Docs / FAQ |
| **#248** | Diarization | MEDIUM | v1.9+ |
| **#247** | Docker support | LOW | Backlog — **foxfire881 new comment 04-17** requesting docker-compose for NAS deployment (CPU-only, 7x24 use case). |
| **#246** | Serverless GPU pipeline + anime-whisper hallucination | LOW | v1.9+ — from 2026-03-31. Cluster with #265. |
| **#242** | XXL in Pass 1 | MEDIUM | v1.9 |
| **#239** | AMD GPU | MEDIUM | v1.9+ |
| **#232** | Whisper-ja-anime-v0.1 model support | LOW | v1.9+ — from 2026-03-16. Cluster with #205, #246. |
| **#230** | Standalone merge module | HIGH | v1.9.0 |
| **#224** | Vocal separation | MEDIUM | v1.9 |
| **#223** | Subtitle edit to PurfView Faster Whisper for Pass 2 | LOW | Backlog — from 2026-03-25. Cluster with #230 (merge module). |
| **#215** | Qwen3-ASR quality | LOW | Expected behavior |
| **#213** | Intel GPU (XPU) | LOW | v1.9+ |
| **#206** | Grey out incompatible options | MEDIUM | v1.9 |
| **#205** | VibeVoice ASR | LOW | v1.9+ |
| **#181** | Frameless window | LOW | Backlog |
| **#180** | Multi-language GUI (i18n) | HIGH | v1.9.0 |
| **#175** | Chinese GUI | HIGH | v1.9.0 |
| **#164** | MPEG-TS + Drive | LOW | Backlog |
| **#142** | AMD Radeon ROCm | MEDIUM | v1.9+ |
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

*Note: #128 (Gemma 3 models) was merged as a PR, not an issue — removed from tracker.*

---

## Immediate Actions — Post-Release Priority

### Completed (rev26)

All P0 and P1 responses posted on 2026-04-09:
- #284 (install stuck), #280 (Qwen3-ASR), #267 (numba follow-up)
- #281 (Ollama UX), #282 (Ollama GitHub), #271 (thinking models), #259 (overwrite fix)

### Outstanding Actions (rev34)

**NEEDS RESPONSE queue (5):**
- **#289** CUDA 12.8 install timeouts (P0, Traditional Chinese) — same pattern as #284. Guide on pre-download whl + `Scripts\pip.exe install --no-deps` path.
- **#290** Colab Pass 2 "Killed" after MossFormer2_SE_48K load (P1) — almost certainly Colab free-tier RAM exhaustion. Recommend: switch speech-enhancer to `zipenhancer` or `ffmpeg-dsp`, or disable, to fit Colab's ~12GB RAM ceiling.
- **#291** Colab local-LLM `starlette_context` missing (P1) — known-broken deprecated path. Recommend: switch to `--provider ollama` as deprecation warning already states. Note llama-cpp removal in v1.9.0.
- **#292** Low GPU utilization with gemma4-26b MoE (P2) — community already answered well. Quick owner ack + point to `qwen2.5:7b-instruct` / `translategemma:12b` recommended defaults.
- **#293** Translation context prompt feature (P2) — short answer: not currently exposed, but Whisper ASR does support initial_prompt internally. Log as v1.9+ feature.

**NEEDS FOLLOW-UP queue (2):**
- **#217** loveGEM/vimbackground — user gave up after OpenAI Whisper install also failed post manual torch install. Candidate for close with a summary of "complete manual install path" or a pointer to future China-mirror installer roadmap (v1.9.0 install hardening).
- **#247** Docker support — foxfire881 follow-up for NAS/CPU use case (04-17). Low priority, brief acknowledgment sufficient.

**v1.8.11 dev status (unchanged since rev33):**
- ✓ #271 **FIX CODED**: curated Ollama model list fixed + `--ollama-max-tokens` flag.
- No new commits since 04-12.

**Remaining v1.8.11 candidates**: #263 torch.hub China fix, #267 numba architectural fix, sanitizer hardening (#287 investigation — dependent on user reply re: VAD swap), #265 hallucination improvements, Colab hardening (#290/#291 regression pair).

**14 issues AWAITING REPLY from users.** Normal turnaround 1-3 days.

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

11. **Colab Pass 2 OOM regression (#290)** — ClearVoice 48kHz (MossFormer2_SE_48K) appears to push Colab free-tier memory past the kill threshold. Pass 1 completes 7908s of audio fine; Pass 2 dies during speech enhancement load. Options: (a) lower speech_enhancer default on Colab to `zipenhancer` (16kHz, much smaller), (b) add Colab memory detection and auto-downgrade, (c) document Colab memory constraint in Colab notebook header. Likely regression from post3 ClearVoice-48kHz default.

12. **Colab local-LLM break (#291)** — `starlette_context` not pulled in as transitive dep by current `llama-cpp-python[server]`. The deprecation warning already says migrate to Ollama. Since v1.9.0 will remove llama-cpp entirely, fix options: (a) pin `starlette-context` in Colab extras, (b) surface explicit error guiding to `--provider ollama`, (c) no-op (deprecated path, will be gone in v1.9.0).

13. **#281 closure pattern (uninstall+reinstall with admin)** — wlee15 resolved Ollama detection by reinstalling as all-users with admin. Similar to other "all-users-install fixes it" reports. Worth investigating why user-local installs have more Ollama detection friction. Could be a PATH / env var inheritance issue.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Status |
|---------|--------|---------|--------|
| **Ollama / Translation** | #293, #292, #282, #271, #259, #255, #233, #132 | #132 | #281 CLOSED by reporter (uninstall+reinstall as admin). #271 FIX CODED. Community-saturated cluster. |
| **Install: PyTorch download / China network** | #289, #284, #217, #263, #261, #243 | #284 | **Growing cluster** — 6 issues touching PyTorch download timeouts / TLS resets / GFW. v1.9.0 install hardening imperative. |
| **Install: GPU architecture mismatch** | #286, #243 | #286 | GTX 10-series Pascal kernel dropped from recent torch wheels. Downgrade to 2.5.1+cu118 workaround. |
| **Colab regressions** | #291, #290, #231 | #290 | post3 fixed #231 for numpy2 but Pass 2 OOM (#290) and local LLM (#291) now broken. |
| **Semantic hang / numba** | #267 (closed), #263 | #267 | Fix shipped in post3 but reported insufficient. #263 distinct root cause (GFW). |
| **Qwen3-ASR** | #280, #215 | #280 | Upstream transformers mismatch. |
| **Quality / Sanitizer regression** | #287, #265, #246 | #287 | #287 "!!" only — possible silero-v3.1-default regression. Awaiting user VAD-swap test. |
| **Diarization** | #248, #252 | #248 | v1.9+. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Merge module** | #230, #223, #265 | #230 | v1.9.0 P1. |
| **Alt ASR models** | #205, #232, #246 | #232 | Anime-specific models, VibeVoice. |

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
| **2026-04-17** | **rev36.** Posted 4 replies: #291 (root cause identified — Colab install missing `[llm]` extra; workaround given + v1.8.11 fix promised), #290 (factual: "Killed" is ambiguous, asked for Colab tier/file size/resource panel, recommended zipenhancer to test ClearVoice hypothesis), #292 (owner ack to community-answered thread, pointed to translategemma:12b / qwen2.5:7b-instruct), #293 (logged Whisper initial_prompt exposure as v1.9+ candidate). **#291 FIX CODED for v1.8.11**: `installer/install_colab.sh` line 176 — added `llm` to extras list (pulls in `starlette-context`, `sse-starlette`, `uvicorn`, `fastapi`, `pydantic-settings`). Both Colab notebooks reference this script, so the fix propagates. NEEDS RESPONSE queue now empty. |
| 2026-04-17 | **rev35.** Posted bilingual China-network template (Chinese + English) to 5 issues: #289, #284, #217, #263, #261. Template asks community to share working VPN / DNS / CUDA+PyTorch combos, flags **v1.9.1 China-mirror exploration** as a goal (without timeline commitment), and repeats the pre-downloaded whl + `Scripts\pip.exe install --no-deps` + silero-v6.2 interim workarounds. All 5 issues now AWAITING REPLY. |
| 2026-04-17 | **rev34.** 54→61 open. **5 new issues**: #293 (translation context), #292 (low GPU util), #291 (Colab local-LLM starlette_context), #290 (Colab Pass 2 OOM), #289 (CUDA 12.8 timeouts). **4 previously-untracked** added: #223, #227, #232, #246. **1 closed**: #281 (user resolved via admin reinstall). **Status flips**: #287 NEEDS RESPONSE → AWAITING REPLY (owner asked about VAD swap). #217 AWAITING REPLY → NEEDS FOLLOW-UP (user gave up after whisper install also failed). **Removed**: #128 (was PR, merged, not issue). **New cluster**: Colab regressions (#290, #291) suggests post3 introduced a regression pair. Install/PyTorch/China cluster grew to 6 issues. |
| 2026-04-12 | **rev33.** 53→54 open (+1 new: #287). **v1.8.11 dev branch started.** Two fixes coded: (1) Ollama curated model list — removed thinking model, new instruct-only defaults. (2) `--ollama-max-tokens` CLI flag added to both entry points. #271 responded with fix confirmation. #287 new bug identified (all subtitles "!!", NEEDS RESPONSE). |
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
