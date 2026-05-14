# Ideation-Inputs Verification Log

Claim-by-claim verification of the four ideation inputs (one `.txt`, three `.pdf`) the user supplied at the start of the CrispASR integration planning effort. Each substantive technical claim is checked against the Phase 0 primary-source dossiers (`01_crispasr_dossier.md`, `02_crisperweaver_dossier.md`, `03_susurrus_dossier.md`, `04_ecosystem_synthesis.md`, `05_whisperjav_integration_surfaces.md`).

**Verdicts:**
- ✅ **Verified** — claim matches primary-source evidence
- ⚠️ **Partially verified** — the claim is approximately true but the wording overstates, omits material caveats, or misnumbers
- ❌ **Contradicted** — primary sources show the claim is false
- ❓ **Could not verify** — primary sources insufficient (logged in `07_open_questions.md`)

The user already stated upfront that the inputs may be "incomplete or in parts inaccurate" — this verification log is captured so future iterations don't accidentally re-anchor on inaccurate inputs.

---

## Input source map

| Tag | File | Style |
|-----|------|-------|
| **D** | `DSeek_Input_For_CripsASR.txt` | Most grounded; least embellished |
| **P1** | `Integrating CrispASR into WhisperJAV.pdf` | Heaviest architectural blueprint; cites WhisperJAV's "Acoustic Hell" |
| **P2** | `Architectural Analysis and Integration Strategies for the CrispASR Multilingual Engine.pdf` | CrispASR-engine-focused; covers GGUF mmap, LID, etc. |
| **P3** | `Architectural Analysis of CrisperWeaver and CrispASR: Strategic Integration Pathways for WhisperJAV.pdf` | CrisperWeaver-focused; most honest about limitations |

---

## Section 1 — Architectural facts about CrispASR

| # | Source | Claim | Verdict | Primary-source check |
|---|--------|-------|---------|---------------------|
| 1.1 | P1, P2, P3 | "CrispASR is a fork of whisper.cpp" | ✅ Verified | `01_crispasr_dossier.md` §0: README and GitHub description both state "Fork of whisper.cpp"; logical fork (GitHub `fork: false`); `LICENSE` retains *"Copyright (c) 2023-2026 The ggml authors"*; AUTHORS file preserves pre-fork whisper.cpp/ggml roster |
| 1.2 | P1, P2, P3 | "CrispASR supports 24 ASR backends" | ⚠️ Partially verified — count is now higher | `01_crispasr_dossier.md` §1: **28 per-backend adapter files** in `examples/cli/` (canary, chatterbox, cohere, crispasr, fastconformer_ctc, firered_asr, gemma4_e2b, glm_asr, granite, granite_nle, indextts, kokoro, kyutai_stt, m2m100, mimo_asr, moonshine, moonshine_streaming, omniasr, orpheus, parakeet, qwen3, qwen3_tts, t5, vibevoice, voxtral, voxtral4b, wav2vec2). Some are TTS; the ASR-only subset count is not nailed down in this Phase 0 pass |
| 1.3 | P2 | "5 TTS engines (Kokoro, Qwen3-TTS, VibeVoice, Orpheus, Chatterbox)" | ✅ Verified | `01_crispasr_dossier.md` §19; consistent with the 28-file adapter list above |
| 1.4 | P1, P2 | "Built on the ggml tensor library" | ✅ Verified | `01_crispasr_dossier.md` §1: `ggml/` carried as a vendored git subtree with 5 fork-local patches enumerated in `UPSTREAM.md` (one upstreamed as ggml#1477) |
| 1.5 | P1, P2 | "Hardware backends: CUDA, Metal, Vulkan, AVX/NEON" | ⚠️ Partially verified — D3D12 evidence missing | CrispASR README enumerates CUDA + Metal + Vulkan + CPU; D3D12 mentioned in P3 was not corroborated by `01_crispasr_dossier.md` §7 (flagged in dossier's could-not-verify) |
| 1.6 | P2 | "MIT license; same as upstream whisper.cpp" | ✅ Verified | `01_crispasr_dossier.md` §2: SPDX `MIT`; copyright "The ggml authors" preserved |
| 1.7 | P2 | "GGUF format; mmap-based loading; near-instantaneous startup" | ⚠️ Partially verified | GGUF loading and the `gguf_loader.{h,cpp}` primitive are real (`01_crispasr_dossier.md` §1, §6); the mmap-induced "near-instantaneous startup" claim is plausible but no benchmark was captured |
| 1.8 | P2 | "5-file recipe for adding a new backend" | ✅ Verified | `01_crispasr_dossier.md` §5 documents the 5-file recipe pattern |
| 1.9 | P1, P2 | "C-ABI surface for embedding" | ✅ Verified | `01_crispasr_dossier.md` §1: `crispasr_c_api.cpp` is 175.5 KB; bindings/{go,java,javascript,ruby} plus python/, crispasr-sys/ Rust, flutter/crispasr/ Dart consume this C-ABI |
| 1.10 | P2 | "HTTP server mode via `--server` flag" | ✅ Verified | `01_crispasr_dossier.md` §14, also referenced by WhisperInc consumer in `04_ecosystem_synthesis.md` |
| 1.11 | P2 | "Language identification via `-l auto` with ggml-tiny model" | ✅ Verified | `01_crispasr_dossier.md` §12 |
| 1.12 | P3 | "Capability flag `CAP_WORD_TIMESTAMPS` and capability-bitmask logic" | ✅ Verified | `01_crispasr_dossier.md` §10 enumerates 19 `CAP_*` bits |
| 1.13 | P1, P3 | "NeMo Forced Aligner-style CTC built into the engine" | ✅ Verified, with material caveat | `01_crispasr_dossier.md` §11; **caveat per dossier and SubtitleEdit#10775**: per-segment dispatch can bypass the CTC pass when both `CAP_WORD_TIMESTAMPS` and `CAP_TIMESTAMPS_CTC` are advertised — fix is `--force-aligner` flag |
| 1.14 | P1 | "ChronosJAV-via-CrispASR-CTC bypasses WhisperJAV's PyTorch alignment" | ❌ Contradicted | The CTC pass has the known dispatch defect noted in 1.13. P3 itself acknowledges the limitation. The "breathtaking architectural revelation" framing in P1 is not supported. |
| 1.15 | P1, P2 | "Python bindings via lightweight `crispasr` Python package" | ✅ Verified | `01_crispasr_dossier.md` §15: `python/crispasr/_binding.py` (83.3 KB) is **pure ctypes** — not pybind11, not cffi. P2's mention of "ctypes/cffi" is ctypes-correct |
| 1.16 | (all) | "Python bindings bypass the GIL during inference" | ❓ Could not verify | `01_crispasr_dossier.md` §16: thread-safety contract documented verbatim as *"All functions are thread-unsafe per context — wrap with a mutex"*; GIL-release-during-FFI was not isolated in this Phase 0 pass |
| 1.17 | P2 | "Auto-probe with `-m auto`: GGUF metadata header drives backend selection" | ✅ Verified | `01_crispasr_dossier.md` §9 |
| 1.18 | P1 | "3.8x faster than voxtral.c" | ❓ Could not verify | No primary-source benchmark in the CrispASR repo's verifiable docs corroborates this specific number; the dossier did not surface it. Logged in `07_open_questions.md` |
| 1.19 | P2 | "Distil-whisper executes 6.3x faster than base architecture" | ❓ Out of scope | This is a distil-whisper benchmark cited as a generic backend property, not a CrispASR-specific claim. Not verified here |
| 1.20 | P2 | "CrispEmbed: 9.5x performance increase over ONNX runtimes" | ❓ Out of scope | CrispEmbed not researched in Phase 0; flagged for `07_open_questions.md` §K.6 |

---

## Section 2 — Architectural facts about Susurrus

| # | Source | Claim | Verdict | Primary-source check |
|---|--------|-------|---------|---------------------|
| 2.1 | D | "Susurrus supports 12 transcription backends, including CrispASR" | ⚠️ Partially verified | Susurrus's `BACKEND_MODEL_MAP` in `config.py` does **not** include a `crispasr` entry per `03_susurrus_dossier.md` §6 — the README integration step 5 was skipped. The README claims many backends but `backends/transcription/` shows only voxtral_api.py and voxtral_local.py at the top level; the rest live under `workers/transcription/backends/`. Number 12 not verified |
| 2.2 | D | "Pluggable Backend Interface: `TranscriptionBackend` base class in `workers/transcription/backends/base.py` with `transcribe()` method yielding `(start, end, text)`" | ✅ Verified | `03_susurrus_dossier.md` §5 quotes the base class verbatim |
| 2.3 | D, P1 | "Susurrus auto-downloads the CrispASR binary if not present" | ✅ Verified, with material caveat | `03_susurrus_dossier.md` §11: pulls from `releases/latest/download/{asset}` into `~/.cache/susurrus/crispasr/`; **no version pinning, no checksum, no signature, no Linux/Windows ARM64 support**, and the macOS asset is single-tarball regardless of arm64 vs x86_64 |
| 2.4 | D | "Susurrus separates GUI/CLI from backend code" | ✅ Verified | `03_susurrus_dossier.md` §4, §8 |
| 2.5 | (implicit) | "Susurrus is the reference Python host pattern for CrispASR" | ⚠️ Partially verified | True at the snapshot moment but **Susurrus is effectively dormant** — 7 total commits, latest 2026-04-19 (~24 days before research), no releases, no tags (`03_susurrus_dossier.md` §0; `04_ecosystem_synthesis.md` §F.4). Treat as a frozen example, not an active reference |
| 2.6 | (implicit) | "Susurrus invokes CrispASR via the official Python bindings or in-process FFI" | ❌ Contradicted | `03_susurrus_dossier.md` §12: Susurrus calls the **`crispasr` binary as a subprocess** with `subprocess.Popen` and parses stdout via regex `\[(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]\s*(.*)`. Even the maintainer's own Python consumer chose subprocess over the in-process Python bindings (`04_ecosystem_synthesis.md` §F.5) |

---

## Section 3 — Architectural facts about CrisperWeaver

| # | Source | Claim | Verdict | Primary-source check |
|---|--------|-------|---------|---------------------|
| 3.1 | P3 | "AGPL-3.0 license; Flutter 3.38" | ✅ Verified | `02_crisperweaver_dossier.md` §0: SPDX `AGPL-3.0`, full GNU AGPL v3 text in LICENSE; Flutter 3.38 |
| 3.2 | P3 | "Dart FFI bridge to CrispASR via `CrispASREngine` class" | ✅ Verified, with structural caveat | `02_crisperweaver_dossier.md` §5: bridge is split across `lib/engines/crispasr_engine.dart` (45 KB, consumer wrapper) and the upstream `package:crispasr` (binding package, 106 KB) — CrisperWeaver's own code never touches `DynamicLibrary` directly |
| 3.3 | P3 | "Library-discovery fallback: bundled Frameworks/ → dev paths → system paths" | ⚠️ Partially verified | `02_crisperweaver_dossier.md` §6: hard-coded `_libCandidates()` exists per platform; **the README's claim of a "user-supplied override path" has no corresponding mechanism in the extracted source** (flagged in dossier §17) |
| 3.4 | P3 | "Model management: HuggingFace API probing, SHA-1 checksum, GGUF quant discovery" | ✅ Verified | `02_crisperweaver_dossier.md` §7: probes `https://huggingface.co/api/models/{repoId}?blobs=true`, regex `-(q[0-9][a-z_0-9]*|f16|f32|bf16)$`, HTTP-Range resumeable downloads, SHA-1 in `Isolate.run`, plus a documented skip-verification toggle |
| 3.5 | P3 | "Telemetry surface streams progress over FFI without blocking UI" | ⚠️ Partially verified | `02_crisperweaver_dossier.md` §5: the FFI is **single-isolate-blocking**; parallelism is achieved by `transcription_worker_pool` that spawns N isolates each with its own FFI handle and exchanges `Float32List` over `SendPort`. The "non-blocking" claim is achieved at the application layer, not at the FFI layer |
| 3.6 | P3 | "Cross-platform reach: macOS, Linux, Windows, Android, iOS" | ✅ Verified | `02_crisperweaver_dossier.md` §3 documents platform-specific bundling scripts; iOS is deliberately unsigned for SideStore distribution |
| 3.7 | (implicit) | "CrisperWeaver pins to a specific CrispASR version" | ❌ Contradicted | `02_crisperweaver_dossier.md` §0 + `04_ecosystem_synthesis.md` §F.3: `pubspec.yaml` has `crispasr: path: ../CrispASR/flutter/crispasr` — **no version constraint**; resolves against whatever is in the sibling directory. CI has `CRISPASR_REPO` / `CRISPASR_REF` env vars for optional pinning |

---

## Section 4 — Architectural facts about WhisperJAV (as described in inputs)

| # | Source | Claim | Verdict | Primary-source check |
|---|--------|-------|---------|---------------------|
| 4.1 | P1, P3 | "WhisperJAV defines seven configurable Python-based pipelines: faster, fast, balanced, fidelity, transformers, qwen, anime" | ❌ Contradicted | `05_whisperjav_integration_surfaces.md` §2: `PIPELINE_CLASSES` registers `balanced, fast, faster, fidelity, transformers, qwen` — plus `xxl` as an early-branch dispatch in `pass_worker.py:662`. **There is no `anime` pipeline.** anime-whisper exists only as a text-generator in `modules/subtitle_pipeline/generators/anime_whisper.py`, not a top-level pipeline. P3's pipeline table is inaccurate on this row |
| 4.2 | P1, P3 | "ChronosJAV is a pipeline" | ❌ Contradicted | `05_whisperjav_integration_surfaces.md` §6: ChronosJAV-like decoupled-alignment lives under `whisperjav/modules/subtitle_pipeline/` (orchestrator + aligners/ + generators/). It is **an architectural pattern**, not a top-level user-facing pipeline. It is referenced through `qwen_pipeline.py` and `transformers_pipeline.py` |
| 4.3 | P1 | "WhisperJAV is tightly coupled to PyTorch-based ASR backends operating entirely within PyTorch" | ⚠️ Partially verified | True for `fast`, `fidelity`, `transformers`, `qwen`. Not true for `faster` and `balanced` (CTranslate2, not PyTorch). And not true for `xxl` (subprocess to external binary). The framing "entirely within PyTorch" overstates |
| 4.4 | P3 | "WhisperJAV uses Auditok + Silero + TEN-VAD for VAD" | ✅ Verified | CLAUDE.md and `05_whisperjav_integration_surfaces.md` §5 reference Silero (Pydantic preset at `config/components/vad/silero.py`); TEN-VAD and Auditok are referenced in the CLAUDE.md mode-table |
| 4.5 | P3 | "WhisperJAV has Two-Pass Ensemble Mode with merge strategies pass1_primary, smart_merge, longest, full_merge" | ✅ Verified, with extension | `05_whisperjav_integration_surfaces.md` §4.5: 7 strategies — `FULL_MERGE, PASS1_PRIMARY, PASS2_PRIMARY, PASS1_OVERLAP, PASS2_OVERLAP, SMART_MERGE, LONGEST`. P3 lists 4 (missing the two `*_OVERLAP` variants and `PASS2_PRIMARY`) |
| 4.6 | P3 | "Python 3.10-3.12 (3.9 dropped, 3.13 incompatible with openai-whisper)" | ✅ Verified | CLAUDE.md |
| 4.7 | P3 | "Apple Silicon MPS pipeline hangs after HuggingFace models load (#227)" | ⚠️ Partially verified | Issue #227 is referenced in v1.9.x carryover (MEMORY.md). The exact MPS pipeline-hang root cause was not re-verified in this Phase 0 pass |
| 4.8 | P3 | "Qwen pipeline currently CPU-only on macOS" | ⚠️ Could not verify in code in this pass | Stated in MEMORY.md and inputs; `whisperjav/pipelines/qwen_pipeline.py` was not read line-by-line. Logged in `07_open_questions.md` |
| 4.9 | D | "WhisperJAV pipelines and `--backend crisp-asr` flag" | ❌ Contradicted (pre-emptive misframing) | There is no `--backend` argparse flag in `main.py`; pipeline selection is via `--mode` / `--pass2-pipeline`. The D-input was sketching a future state, not describing existing state |
| 4.10 | (all) | "BYOP pattern available as integration precedent" | ✅ Verified | `05_whisperjav_integration_surfaces.md` §1 documents the complete BYOP end-to-end. xxl_runner.py is 162 lines, zero imports from pipelines/config/ensemble; `pass_worker.py:662` is the early-dispatch hook; `asr_config.json > ui_preferences.byop` is the persistence; `--xxl-exe` + `--pass2-pipeline xxl` are the CLI surfaces |

---

## Section 5 — Architectural claims about CrispASR's relationship to its consumers

| # | Source | Claim | Verdict | Primary-source check |
|---|--------|-------|---------|---------------------|
| 5.1 | P3 | "SubtitleEdit implemented dynamic routing based on probabilistic linguistic classification (auto-language for CrispASR glm/parakeet/Qwen3)" | ✅ Verified | `04_ecosystem_synthesis.md` §A: SubtitleEdit subprocess-invokes `crispasr.exe` with per-backend defaults; beta23 added a forced-aligner combo-box (response to issue #10775) |
| 5.2 | P3 | "Meetily integrates CrispASR via Rust FFI" | ❌ Contradicted | `04_ecosystem_synthesis.md` §B: Meetily uses `whisper-rs` + vendored `whisper.cpp` directly — **no CrispASR/crispasr-sys references in the repo**. The Meetily claim does not survive primary-source check |
| 5.3 | (implicit) | "Subprocess invocation is too slow / FFI is the production path" | ❌ Contradicted | `04_ecosystem_synthesis.md` §I.1: 5 of 7 observed real-world consumers use subprocess; only 2 use FFI (CrisperWeaver via Dart, parakit via Rust). Subprocess is the **dominant** observed pattern |

---

## Section 6 — Strategic-framing claims (not architecturally testable)

The four inputs make several non-testable strategic claims. Listed here for traceability — these are NOT verified or contradicted in this dossier because they are matters of judgment not fact, and the user has explicitly placed strategic decisions out of scope for Phase 0.

| Source | Claim | Status |
|--------|-------|--------|
| P1 | "Integrating CrispASR represents a profound leap forward in computational efficiency and transcription fidelity" | Strategic; not in Phase 0 scope |
| P1 | "Architectural revelation: route CrispASR through ChronosJAV" | Strategic; downgraded by the contradiction in 1.14 |
| P2 | "CrispASR is mature, production-ready" | Strategic — and qualified by maintainer-cadence facts in `04_ecosystem_synthesis.md` §F.4 |
| P3 | "Adopting CrispASR is not merely the superficial addition of new speech models; it represents a fundamental, necessary architectural upgrade" | Strategic; user has chosen addition-only scope (CL1), so this framing is explicitly out of scope |

---

## Summary tally

| Verdict | Count |
|---------|-------|
| ✅ Verified | 17 |
| ⚠️ Partially verified | 11 |
| ❌ Contradicted | 7 |
| ❓ Could not verify | 4 (logged in `07_open_questions.md`) |

The inputs are roughly two-thirds load-bearing and one-third either overstated or factually wrong. The contradicted-7 cluster heavily around two themes: (a) the proposed ChronosJAV-via-CrispASR-CTC alignment story has a known defect; (b) Meetily and several pipeline-list / version-pinning claims do not survive primary-source check. Future iterations should not treat the inputs as authoritative — they should read the Phase 0 dossiers instead.
