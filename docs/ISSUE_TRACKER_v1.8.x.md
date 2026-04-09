# WhisperJAV Issue Tracker — v1.8.x Cycle

> Updated: 2026-04-09 (rev24) | Source: [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) | **57 open** on GitHub

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
| Total open on GitHub | **57** | +2 since rev23 (55→57). 7 new issues (#274-#284), 5 closed (#269, #272, #275, #276, +1). |
| **v1.8.10 RELEASED** | — | 2026-03-30. 39 commits in release. |
| **Unpushed commits on main** | **14** | hf1→hf3 retune, TEN backend, forensic tools, post-processing, GUI audit. See below. |
| **NEEDS RESPONSE** | **6** | #280, #282, #284, #274, #279, #268. No maintainer reply at all. |
| **NEEDS FOLLOW-UP** | **3** | #267 (fix didn't work), #271 (thinking model problem + GUI model recs), #265 (community questions). |
| **AWAITING REPLY** | 5 | #234, #251, #258, #260, #263 |
| Feature requests (open) | 30 | +1 (#279 Stash). See Feature Requests section. |

---

## CRITICAL: 14 Unpushed Commits

These 14 commits are on local main but NOT on origin/main. Users cannot access them yet.

| Commit | Date | Description | Risk |
|--------|------|-------------|------|
| `fd5407b` | 04-09 | retune: v1.8.10-hf3 — Whisper ASR defaults, Silero v3.1/v4.0 presets, neg_threshold cleanup | MEDIUM — major parameter changes |
| `78ad155` | 04-05 | tools: add forensic CSV generator and FW diagnostic suite | LOW — tools only |
| `6135f49` | 04-05 | feat: make post-model logprob gate optional with pipeline-specific defaults | MEDIUM — behavior change |
| `7375400` | 04-05 | feat: TEN VAD backend — silence merging, max_speech splitting, long-segment control | MEDIUM — new backend feature |
| `e679275` | 04-05 | retune: Silero v6.2 and TEN VAD thresholds per forensic analysis | MEDIUM — parameter changes |
| `75b8e12` | 04-05 | retune: relax logprob/no_speech thresholds, fix full-audio filter crash | HIGH — crash fix |
| `6109c0c` | 04-04 | fix: apply sensitivity presets to non-Silero segmenters in legacy pipelines | MEDIUM — bug fix |
| `ccd469f` | 04-03 | fix: revert length_penalty=-0.5 to None — OpenAI Whisper requires 0-1 | HIGH — crash fix |
| `756da9d` | 04-03 | fix: add slow CPS removal for short hallucination labels | LOW — sanitizer |
| `80a2ef6` | 04-02 | fix: add regex patterns for closing variants and multi-word English | LOW — sanitizer |
| `9c2dc8a` | 04-02 | fix: post-processing pipeline — regex, punct norm, post-merge dedup | MEDIUM — behavior change |
| `a0eb82e` | 04-02 | fix: GUI customize modal audit — 11 fixes + config standardization | MEDIUM — GUI |
| `17bc2e0` | 04-02 | fix: resolve device="auto" in OpenAI Whisper ASR modules | HIGH — crash fix |
| `d67e82e` | 04-02 | retune: ASR + VAD default parameters for hallucination reduction (hf1) | MEDIUM — parameter changes |

**Note**: The numba cache fix (`28dff5b`, 04-01) IS on origin/main — it was pushed previously. But the user (#267) reports it did not resolve their issue.

---

## v1.8.10 — RELEASED (2026-03-30)

See `installer/RELEASE_NOTES_v1.8.10.md` for full details.

---

## Fixes on origin/main (pushed, not in a release tag)

These fixes are on origin/main after v1.8.10 was tagged. Colab/pip users get them via `pip install --upgrade`.

| Commit | Item | Issues | Risk |
|--------|------|--------|------|
| `28dff5b` | numba cache dir redirect — `ensure_numba_cache_dir()` | #267, #263 | MEDIUM — user reports fix insufficient |
| `e50df15` | XXL model flag not forwarded to subprocess; semantic diagnostics v7.1.0 | #272, #267 | LOW |
| `1bfe548` | Hallucination filter: bundled JSON missing from wheel, stale gist URLs | #265 | LOW |
| `4a600b2` | `--scene-detection-method` ignored in standard pipeline modes | #269 (closed) | LOW |
| `2533438` | Docs: GUI user guide updated for Ollama + enhance-for-VAD | — | LOW |
| `a6b473d` | Colab install: remove venv, reuse system torch, fix llama-cpp filename | #231 | LOW |
| `cfcbddf` | Translation overwrite fix | #259 | LOW |

---

## Open Issues by Status

### Bugs / Active Issues (22)

| # | Title | Reporter | Status | Notes |
|---|-------|----------|--------|-------|
| **#284** | v1.8.10 install stuck during PyTorch phase | qq73-maker | `NEEDS RESPONSE` | **NEW.** Install hangs at Phase 3 PyTorch install (uv, CUDA 12.8, RTX 4060). @mapei1992 confirms same — fell back to v1.8.9. **Two users affected.** |
| **#282** | Why does Ollama need GitHub connection? | KenZP12 | `NEEDS RESPONSE` | **NEW.** User noticed Ollama translation connects to GitHub. This is the `instructions_standard.txt` gist fetch. |
| **#281** | Ollama not detected in GUI | wlee15 | `NEEDS RESPONSE` | **NEW.** Ollama installed but GUI doesn't show it. Ensemble mode AI-translate works fine. Community workaround by @vicecity2930: fill Ollama Server URL with `http://127.0.0.1:11434`. |
| **#280** | Qwen3-ASR TypeError: `check_model_inputs()` | zoqapopita93 | `NEEDS RESPONSE` | **NEW.** Pass 2 crash loading Qwen3-ASR. Error in third-party `qwen_asr` lib — `check_model_inputs()` decorator API mismatch with installed `transformers` version. Full traceback provided. |
| **#274** | Pipeline aggregation mode question | cuixiaopi | `NEEDS RESPONSE` | **NEW.** Asks about combining 3 pipelines. Needs clarification response. |
| **#272** | XXL ignores --pass2-model large-v2 flag | TinyRick1489 | `CLOSED` | Fixed on origin/main (`e50df15`). User confirmed fix works. **Closed 04-02.** |
| **#271** | Ollama translation model issues | justantopair-ai | `NEEDS FOLLOW-UP` | **ACTIVE (15 comments).** Key finding by @TinyRick1489 (04-04): thinking models (gemma4, qwen3.5) break translation format because chain-of-thought mixed into response. @l34240013 (04-04): GUI recommended models are NOT instruct models — same format error. Debug logs attached. **Action needed: update GUI model recommendations.** |
| **#268** | Thai + Korean translation targets | yedkung69-ctrl | `NEEDS RESPONSE` | @seminice86 also requested Korean + custom target support. No maintainer reply. |
| **#267** | Stuck on Streaming features (Qwen+Semantic) | OrangeFarmHorse | `NEEDS FOLLOW-UP` | **FIX DID NOT WORK.** User upgraded (04-01 16:28) but same hang persists. numba cache redirect (`28dff5b`) on origin/main but user still hangs. Did not try manual folder permissions. **Needs deeper investigation.** |
| **#265** | Hallucination + post-processing suggestions | yangming2027 | `NEEDS FOLLOW-UP` | **ACTIVE (18 comments).** @yangming2027 shared v2 hallucination word list (04-03). @zoqapopita93 (04-06) asking about large-v2 vs v3 and custom settings — unanswered. |
| **#263** | GPU not utilized / stuck at VAD | herlong6529424-dot | `AWAITING REPLY` | Likely same root cause as #267. Diagnostic script sent 04-01. No response from user. |
| **#261** | Network check: unknown url type https | henry99a | `AWAITING REPLY` | NSIS SSL context issue. User found cmd workaround. |
| **#260** | Uninstall leaves 6GB | hawai-stack | `AWAITING REPLY` | Responded with cache paths. |
| **#259** | Local Translation Issues (hotfix2) | destinyawaits | `NEEDS FOLLOW-UP` | Ollama working (04-01). **NEW BUG**: save path not respected — temp file overwrites existing subtitles in same folder. Translation overwrite fix coded (`cfcbddf`) but only on origin/main. |
| **#258** | 我遇到的问题 (vague) | Uillike | `AWAITING REPLY` | Asked for logs. No response. |
| **#255** | 如何用ollama进行翻译 | cheny7918 | `RESPONDED` | Pointed to v1.8.10 Ollama GUI. |
| **#251** | post2 launch failure (fastgit SSL) | zoqapopita93 | `AWAITING REPLY` | fastgit mirror SSL failure. 3 solutions provided. |
| **#243** | Install verify fails (RTX 3050) | Trenchcrack | `AWAITING REPLY` | Community member helping. |
| **#240** | GUI access violation Win11 | m739566004-svg | `SHIPPED` post2 | private_mode=True. |
| **#237** | XXL model questions | yangming2027 | `COMMUNITY` | Active discussion. |
| **#234** | CUDA version confusion | techguru0 | `AWAITING REPLY` | Corrected. |
| **#233** | Translation error (local LLM) | WillChengCN | `RESPONDED` | Recommended Ollama + v1.8.10 upgrade. |
| **#231** | Kaggle notebook error | fzfile | `RESPONDED` | llvmlite fix + translate cmd fix on main. |
| **#225** | GUI white screen | github3C | `STALE` | Exhausted hypotheses. |
| **#217** | GUI.exe not found (China network) | loveGEM | `NEEDS FOLLOW-UP` | Root cause: PyTorch download fails. @vimbackground (03-30) asked how to pre-place downloaded whl. No response to that. |

### Recently Closed (since rev23)

| # | Title | Closed | Notes |
|---|-------|--------|-------|
| **#276** | Drag-and-drop to reorder video queue | 04-03 | Feature request. Closed. |
| **#275** | UI freeze remaining time when removing video | 04-03 | Bug report. Closed. |
| **#272** | XXL ignores --pass2-model large-v2 | 04-02 | Fixed. Confirmed by user. |

### Stale / Low Activity (1)

| # | Title | Last Activity | Recommendation |
|---|-------|---------------|----------------|
| #232 | whisper-ja-anime model | 03-16 | Evaluate model. Keep open. |

### Feature Requests (30)

| # | Title | Priority | Target |
|---|-------|----------|--------|
| **#279** | Stash integration | LOW | Backlog. **NEW.** |
| **#268** | Thai + Korean translation targets | LOW | v1.9+ (Korean requested by @seminice86) |
| **#265** | Post-translation hallucination filter (Chinese) | MEDIUM | v1.9 — user contributed 130-rule XML + v2 word list |
| **#264** | Model download location customization | LOW | Responded. HF_HOME workaround. |
| **#262** | Cohere Transcribe model | LOW | Responded. No timestamps — needs integration work. |
| **#254** | Remove non-speech sounds | MEDIUM | v1.9 |
| **#252** | Multi-speaker / diarization | MEDIUM | v1.9+ |
| **#250** | Model folder documentation | LOW | Docs / FAQ |
| **#248** | Diarization | MEDIUM | v1.9+ (dup #252). |
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

## Immediate Actions

| Priority | Action | Issues | Status |
|----------|--------|--------|--------|
| **P0** | Respond to #284 (v1.8.10 install stuck — 2 users) | #284 | **TODO** |
| **P0** | Respond to #280 (Qwen3-ASR TypeError) | #280 | **TODO** |
| **P0** | Follow up #267 (numba fix didn't work) | #267 | **TODO** |
| **P1** | Respond to #281 (Ollama detection UX) | #281 | **TODO** — community workaround posted |
| **P1** | Respond to #282 (Ollama GitHub connection explanation) | #282 | **TODO** |
| **P1** | Follow up #271 (thinking model problem + GUI model recs) | #271 | **TODO** — @l34240013 and @TinyRick1489 identified root cause |
| **P1** | Respond to #268 (Thai + Korean) | #268 | **TODO** |
| **P2** | Respond to #274 (pipeline aggregation question) | #274 | **TODO** |
| **P2** | Respond to #279 (Stash integration) | #279 | **TODO** |
| **P2** | Follow up #265 (@zoqapopita93 questions unanswered) | #265 | **TODO** |
| **P2** | Follow up #217 (@vimbackground whl question) | #217 | **TODO** |

---

## Emerging Patterns

### v1.8.10 Installer Stuck (#284) — NEW, 2 USERS

**Symptom**: v1.8.10 conda-constructor install hangs during Phase 3 (PyTorch install via uv). RTX 4060, CUDA 12.8, driver 576.88. @mapei1992 also hit this and fell back to v1.8.9.
**Status**: No investigation yet. Could be uv timeout, network, or dependency resolution issue. Post-install script uses `uv pip install` which may behave differently than pip.
**Impact**: Blocks new user adoption of v1.8.10.

### Semantic Scene Detection Hang (#267) — FIX INSUFFICIENT

**Root cause hypothesis (numba PermissionError)**: Confirmed by Ctrl+C traceback. Fix implemented: `ensure_numba_cache_dir()` sets `NUMBA_CACHE_DIR` to user-writable location. Fix is on origin/main.
**But**: User upgraded (04-01 16:28) and same hang persists. Did not try manual folder permissions.
**Possible explanations**:
1. `setup_console()` may not be called before the code path that triggers librosa import
2. The user's upgrade may not have picked up the fix (pip cache?)
3. The problem may not be ONLY numba cache — could be a deeper numba/llvmlite JIT issue
**Status**: Needs deeper investigation. Need to verify the user actually got the fix (check version), and whether `NUMBA_CACHE_DIR` was actually set at runtime.

### Qwen3-ASR Third-Party Crash (#280) — NEW

**Error**: `TypeError: check_model_inputs() missing 1 required positional argument: 'func'` during `from qwen_asr import Qwen3ASRModel`.
**Analysis**: The `@check_model_inputs()` decorator in `qwen_asr` lib's `modeling_qwen3_asr.py:986` is called with `()` (no args), but the `check_model_inputs` function from `transformers` expects `func` as the first positional arg. This is a **transformers version incompatibility** — the decorator API changed between versions.
**Impact**: Blocks Qwen3-ASR usage entirely.

### Ollama Translation: Thinking Models Break Format (#271) — ROOT CAUSE IDENTIFIED

**Root cause**: Thinking models (gemma4, qwen3.5, qwen3) include chain-of-thought reasoning in their response, which breaks PySubtrans's expected format. Only instruct-tuned models (qwen2.5-instruct, translategemma, gemma3:12b-it) work correctly.
**Additional problem (reported by @l34240013)**: The GUI's recommended Ollama model list includes non-instruct models that produce format errors.
**Impact**: Users follow GUI recommendations → translation fails → frustration.
**Fix needed**: Update GUI model recommendations to only suggest instruct models. Consider adding a model compatibility note.

### Ollama GitHub Connection (#282) — EXPECTED BEHAVIOR

**Explanation**: WhisperJAV fetches `instructions_standard.txt` from a GitHub Gist to provide translation instructions to the LLM. This is by design — the instructions file is the prompt template. Fallback to bundled file exists if offline.
**Action**: Respond with explanation. Consider making this more transparent in the UI.

### Translation Save Path Bug (#259) — FIX CODED, NOT PUSHED

**Symptom**: AI SRT Translate creates temp file in same folder as input SRT, overwrites existing English subtitles. User-specified save path not used until task completes.
**Fix**: `cfcbddf` on origin/main. But 14 subsequent commits are NOT pushed, so user-accessible state is the fix.

---

## Duplicate / Related Issue Map

| Cluster | Issues | Primary | Action |
|---------|--------|---------|--------|
| **Local LLM / Ollama** | #282, #281, #271, #259, #255, #233, #132 | #132 | #271 thinking model root cause found. #281 UX issue. #282 gist fetch. |
| **Install stuck** | #284, #217 | #284 | Two users on v1.8.10. Different root cause from #217 (network vs phase 3 hang). |
| **Semantic hang / numba** | #267, #263 | #267 | Fix insufficient. Deeper investigation needed. |
| **Qwen3-ASR** | #280, #215 | #280 | Transformers version mismatch. #215 is quality (different). |
| **Diarization** | #248, #252 | #248 | v1.9+. |
| **AMD/Intel GPU** | #239, #142, #114, #213 | #142 | v1.9+. |
| **i18n** | #175, #180 | #180 | v1.9.0 P0. |
| **Speech enhancement** | #254, #224 | #224 | v1.9. |
| **XXL** | #242, #237, #223 | #242 | Feature (Pass 1). |
| **Install/Network** | #261, #251, #243, #240, #225 | — | Individual. |
| **Model management** | #264, #250 | #250 | Docs / FAQ. |
| **Translation targets** | #268 | #268 | Thai + Korean. v1.9+. |
| **Hallucination** | #265, #246 | #265 | Post-translation filter for v1.9. Active community input. |
| **Merge module** | #230, #265 | #230 | v1.9.0 P1. |

---

## Release Roadmap

### v1.8.10 Post-Release — 14 Unpushed Commits

**Decision needed**: Push these as-is, or bundle into a v1.8.11 hotfix release?

Arguments for push-only (no version bump):
- Colab/pip users get fixes automatically
- Lower overhead, faster turnaround

Arguments for v1.8.11 release:
- 14 commits is substantial (3 crash fixes, 2 major retunes, new TEN backend)
- .exe installer users need a new installer to get crash fixes
- Includes HIGH-risk crash fixes (length_penalty, device="auto", full-audio filter)
- Clean versioning for support ("are you on v1.8.11?")

### v1.8.11 — Proposed Hotfix Release

**Scope**: All 14 unpushed commits + issue responses + targeted fixes.

| Item | Commit/Status | Risk |
|------|---------------|------|
| ASR + VAD retune (hf1 through hf3) | `d67e82e`..`fd5407b` | MEDIUM |
| device="auto" crash fix | `17bc2e0` | HIGH — crash prevention |
| length_penalty revert | `ccd469f` | HIGH — crash prevention |
| Full-audio filter crash fix | `75b8e12` | HIGH — crash prevention |
| Post-processing pipeline (regex, punct, dedup) | `9c2dc8a`, `80a2ef6`, `756da9d` | MEDIUM |
| GUI customize modal audit (11 fixes) | `a0eb82e` | MEDIUM |
| Sensitivity presets for non-Silero segmenters | `6109c0c` | MEDIUM |
| TEN VAD backend features | `7375400` | MEDIUM — new feature |
| Post-model logprob gate toggle | `6135f49` | MEDIUM — behavior change |
| Forensic tools | `78ad155` | LOW — tools only |
| **NEW**: GUI Ollama model recommendations update | TODO | LOW — #271 |
| **NEW**: Respond to #267 (deeper numba investigation) | TODO | MEDIUM |
| **NEW**: Respond to #280 (Qwen3-ASR version pin guidance) | TODO | LOW |
| **NEW**: Respond to #284 (install stuck diagnosis) | TODO | MEDIUM |

### v1.9.0 — Next Major Release

**Theme: Platform Expansion + Translation Overhaul + UX**

| Priority | Item | Issues | Effort | Notes |
|---|---|---|---|---|
| **P0** | Ollama full migration (deprecate llama-cpp-python) | #132, #233, #255, #259 | Large | Remove ~1500 LOC. |
| **P0** | Chinese UI (partial i18n) | #175, #180 | Medium | 40%+ issues are Chinese. |
| **P0** | Unified CLI override layer | #269 | Small | Standard/ensemble divergence. |
| **P1** | Post-translation hallucination filter (Chinese) | #265 | Medium | User contributed rules + v2 word list. |
| **P1** | Post-processing polish (trailing period, leading dash, line wrapping) | #265 | Small | 3 concrete suggestions. |
| **P1** | Standalone merge CLI | #230 | Medium | `whisperjav-merge`. |
| **P1** | Speaker diarization | #248, #252 | Large | Need quality solution. |
| **P1** | XXL in Pass 1 | #242 | Medium | Needs forced alignment. |
| **P1** | AMD ROCm support | #142, #114, #239 | Medium | Document + test. |
| **P1** | Additional translation targets | #268 | Small | Thai, Korean. |
| **P1** | Full dual-track enhance-for-VAD | — | Medium | Separate VAD/ASR audio paths. |
| **P1** | Ollama model recommendation fix | #271 | Small | Only recommend instruct models in GUI. |
| **P2** | Semantic scene detection hang (deeper investigation) | #267 | Medium | numba redirect insufficient. |
| **P2** | GUI settings persistence | #96 | Medium | Long-standing request. |
| **P2** | Vocal separation | #224, #254 | Medium | BS-RoFormer / UVR. |
| **P2** | Grey out incompatible options | #206 | Small | Prevent invalid GUI combos. |
| **P2** | Uninstall cleanup tool | #260 | Small | Script to remove cached models. |
| **P2** | Model cache paths docs | #250 | Small | FAQ / docs. |
| **P2** | MPS selective policy | #227 | Small | Force CPU for whisper, allow MPS for kotoba. |
| **P2** | Qwen3-ASR transformers compatibility | #280 | Medium | Pin or adapt to transformers API change. |
| **P3** | Docker support | #247 | Medium | Dockerfile + compose. |
| **P3** | Gemma 3 model configs | #128 | Small | Contributor PR. |
| **P3** | whisper-ja-anime model | #232 | Small | If standard HF format. |
| **P3** | Stash integration | #279 | Medium | Niche request. |

### v2.0 — Strategic Vision

**Theme: Architecture + Scale + Ecosystem**

| Area | Item | Why |
|---|---|---|
| **Architecture** | Plugin system for ASR backends | Hardcoded pipeline classes → community contributions |
| **Architecture** | Separate installer from runtime | Enables Docker, cloud, headless |
| **Architecture** | Web-based UI (replace pywebview) | PyWebView has persistent issues (#225, #240) |
| **Scale** | Batch processing dashboard | Queue, progress, ETA |
| **Scale** | GPU memory management overhaul | Proper memory budget for multi-model workflows |
| **Ecosystem** | Public API / SDK | REST API for remote processing |
| **Quality** | Ground truth test framework | Formalize forensic analysis. CI/CD regression detection |
| **Platform** | Linux native installer | AppImage or deb package |
| **Platform** | China mirror support | #217, #251 — PyTorch download fails behind GFW |
| **Translation** | Translation model compatibility layer | #271 — detect and handle thinking models vs instruct models |

---

## Changelog

| Date | Changes |
|------|---------|
| **2026-04-09** | **rev24.** 57 open (+2). 7 new issues: #274 (pipeline question), #275 (UI freeze, closed), #276 (drag-drop, closed), #279 (Stash), #280 (Qwen3-ASR crash), #281 (Ollama not found), #282 (Ollama GitHub), #284 (install stuck). #272 closed (user confirmed fix). 14 unpushed commits on main. #267 fix insufficient (user reports same hang after upgrade). #271 root cause: thinking models break translation; GUI recommends wrong models. #265 v2 hallucination word list shared. |
| 2026-04-01 | **rev23.** 55 open. #267 ROOT CAUSE FOUND: numba PermissionError. Fix coded: `ensure_numba_cache_dir()`. #259 overwrite fix. #271 resolved — instruct models work. |
| 2026-04-01 | **rev22.** 55 open. `e50df15`: XXL fix. `cfcbddf`: translation overwrite fix. |
| 2026-03-31 | **rev21.** 54→55 open. New: #271, #272. Closed: #269. |
| 2026-03-31 | **rev20.** 55→54 open. New: #265, #267, #268, #269. Closed: #218, #221, #236, #244. |
| 2026-03-30 | **rev19.** v1.8.10 released. 54 open. #253 closed. |
| 2026-03-29 | **rev18.** 55 open. |
| 2026-03-28 | **rev17.** Closed 11. Responded 7. |
| 2026-03-28 | **rev16.** Full refresh. |

---
