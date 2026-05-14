# CrispASR Integration — Phase 0 Research Dossier

Foundational primary-source research for the strategic-planning effort to integrate the CrispASR ecosystem into WhisperJAV (v1.9.x scope).

**Status**: Phase 0 **complete** (2026-05-13).
**Vendor decision**: Already made (CrispASR selected). This phase did NOT relitigate that decision.
**Output character**: Descriptive only — no recommendations, no integration design, no curation. Those belong to later phases.

---

## Scope

Five research streams covered:

1. **CrispASR** — github.com/CrispStrobe/CrispASR — C++ ggml runtime hub
2. **CrisperWeaver** — github.com/CrispStrobe/CrisperWeaver — Flutter consumer + Dart-FFI reference
3. **Susurrus** — github.com/CrispStrobe/Susurrus — Python multi-backend GUI that already wraps CrispASR
4. **Cross-ecosystem + real-world consumers** — SubtitleEdit, parakit, WyomingCrispAsrServer, whatdeysay, WhisperInc, others; plus the version-pinning, whisper.cpp posture, and licensing relationships across the three ecosystem repos
5. **WhisperJAV-side surfaces** — BYOP, pipelines, ensemble, config, alignment, installer — verified against actual code

---

## Table of contents

| # | File | Size | Subject |
|---|------|------|---------|
| 00 | [`00_index.md`](00_index.md) | this file | Table of contents, scope, glossary, upstream versions, Phase 0 summary |
| 01 | [`01_crispasr_dossier.md`](01_crispasr_dossier.md) | 112 KB / 1,658 lines | CrispASR architectural review across 25 sections |
| 02 | [`02_crisperweaver_dossier.md`](02_crisperweaver_dossier.md) | 75 KB | CrisperWeaver review (Flutter + Dart-FFI emphasis) |
| 03 | [`03_susurrus_dossier.md`](03_susurrus_dossier.md) | 59 KB | Susurrus review (TranscriptionBackend contract + CrispASR invocation pattern, §12 highest value) |
| 04 | [`04_ecosystem_synthesis.md`](04_ecosystem_synthesis.md) | 60 KB | Part 1: real-world consumer patterns. Part 2: cross-ecosystem relationships, version pinning, whisper.cpp posture, license boundaries |
| 05 | [`05_whisperjav_integration_surfaces.md`](05_whisperjav_integration_surfaces.md) | 27 KB | BYOP plumbing end-to-end, pipeline base class, ensemble dispatch, config layering, alignment, installer, GUI |
| 06 | [`06_ideation_inputs_verification.md`](06_ideation_inputs_verification.md) | (verification log) | Claim-by-claim verification of the four user-supplied ideation inputs against primary sources |
| 07 | [`07_open_questions.md`](07_open_questions.md) | (forward-looking) | Aggregated could-not-verify items across all dossiers, grouped by category |

---

## Upstream repo versions at research time

| Repo | URL | Default branch | Snapshot commit | Date | License | Stars |
|------|-----|----------------|-----------------|------|---------|-------|
| CrispASR | github.com/CrispStrobe/CrispASR | `main` | `bac5f8f` (tip of `main`); last tagged release `v0.6.6` | 2026-05-13 (commit), 2026-05-12 (tag) | MIT | 176 |
| CrisperWeaver | github.com/CrispStrobe/CrisperWeaver | `main` | `9b93f86` ("chore: bump version to 0.5.0") | 2026-05-12 | AGPL-3.0 | 12 |
| Susurrus | github.com/CrispStrobe/Susurrus | `main` | `7073a77` | 2026-04-19 (no activity since) | MIT | 16 |

WhisperJAV side: branch `dev_v1.9.0`, last commit `225b471` at research start (2026-05-13). `v1.8.14` is the latest released tag.

---

## Phase 0 completion summary

### Highlights

- **CrispASR is a logical fork of `ggerganov/whisper.cpp`** (GitHub `fork: false` but README and AUTHORS preserve provenance), MIT-licensed, very active (1,330 commits in last 4 weeks, 40 releases in ~6.5 weeks), single-maintainer-dominant with Claude AI co-authorship visible throughout. Pure-ctypes Python bindings exist at `python/crispasr/_binding.py` (83.3 KB). C-ABI lives in `crispasr_c_api.cpp` (175.5 KB). 28 per-backend adapter files in `examples/cli/`.

- **CrisperWeaver is AGPL-3.0** (different license from CrispASR — does NOT propagate to CrispASR consumers). Consumes CrispASR via local `path:` dependency in pubspec.yaml — no version pinning. FFI bridge is split: `lib/engines/crispasr_engine.dart` (consumer wrapper) + upstream `package:crispasr` (binding package). Single-isolate-blocking FFI; parallelism via `transcription_worker_pool` spawning N isolates.

- **Susurrus calls CrispASR as a subprocess** (NOT FFI), parses stdout regex `\[hh:mm:ss.fff --> hh:mm:ss.fff\] text`. Auto-downloads `releases/latest` with **no version pinning, no checksum, no signature**. 7 total commits, effectively dormant since 2026-04-19. `BACKEND_MODEL_MAP` does not include `crispasr` — README integration step 5 was skipped. **Even the maintainer's own Python project chose subprocess over in-process bindings.**

- **Subprocess is the dominant integration pattern**: 5 of 7 observed real-world consumers (SubtitleEdit, Susurrus, WyomingCrispAsrServer, whatdeysay, plus forks) use subprocess; only 2 use FFI (CrisperWeaver via Dart, parakit via Rust); 1 uses HTTP server mode (WhisperInc). SubtitleEdit is the most production-mature integration — tag-pinned to `v0.6.6` with SHA-256 verification, per-backend CLI defaults, 9 distinct CrispASR backends wrapped.

- **SubtitleEdit issue #10775 is real and ongoing** — the per-segment CTC dispatch in `examples/cli/crispasr_run.cpp` bypasses the aligner when a backend advertises both `CAP_WORD_TIMESTAMPS` and `CAP_TIMESTAMPS_CTC`. Documented workaround is the `--force-aligner` flag (shipped in CrispASR v0.6.0). SubtitleEdit beta23 added a forced-aligner picker UI; the original reporter says the picker reverts to "built-in" after each run.

- **WhisperJAV's existing BYOP path is a direct precedent for external-binary integration**. `whisperjav/byop/xxl_runner.py` (162 lines, zero imports from pipelines/config/ensemble) wraps `faster-whisper-xxl` as a subprocess. `pass_worker.py:662` has an explicit `pipeline=='xxl'` early-dispatch. `asr_config.json > ui_preferences.byop` persists `xxl_exe_path` + `xxl_extra_args`. `--xxl-exe` + `--pass2-pipeline xxl` are the CLI surfaces. **Crash-tolerance pattern is already in place**: SRT existence is checked before exit code, accommodating C++-destructor-shutdown crashes that happen post-write.

- **Speech-enhancement subsystem (v1.7.3+) is the cleanest multi-backend factory pattern in the WhisperJAV codebase** — lazy-loaded string-keyed registry with per-backend dependency declarations. Five backends. The pattern is reusable shape for any future multi-backend ASR factory.

- **The four ideation inputs are roughly 2/3 load-bearing**. Verification log in `06_ideation_inputs_verification.md`: 17 verified, 11 partially verified, 7 contradicted, 4 could-not-verify. Notable contradictions: anime is NOT a pipeline in WhisperJAV; ChronosJAV is NOT a pipeline (architectural pattern only); Meetily does NOT use CrispASR (uses whisper-rs directly); ChronosJAV-via-CrispASR-CTC alignment has the known defect in 1.13; Susurrus does NOT use Python bindings (subprocess).

### Decisions to defer to later phases (per user directive)

- Backend curation (CL1 — user decides)
- WER / empirical evaluation (CL2 — explicitly out of scope)
- Integration boundary choice (Phase 1+ — BYOP-style subprocess is the simplest end of the option space)
- Model distribution strategy
- License-AGPL boundary engineering for any direct CrisperWeaver code reference

### What this dossier delivers to the next iteration

- Five complete primary-source dossiers (01-05) with file:line citations throughout
- A verification log (06) that lets future iterations reject inaccurate ideation-input framings without re-relitigating
- An open-questions log (07) organized by where the answer most likely lives — read-on-demand when specific integration questions arise
- This index (00) as the single entry point

### What this dossier does NOT deliver

- Any recommendation about how to integrate CrispASR
- Any curation of backends
- Any benchmark, WER number, or empirical comparison
- Any architectural design proposal
- Any Phase 1 plan

These are the next iteration's work.

---

## Glossary

- **ggml** — C tensor library for ML inference, optimized for commodity hardware. Vendored as a git subtree in CrispASR with 5 fork-local patches.
- **GGUF** — GPT-Generated Unified Format. The model serialization standard CrispASR consumes.
- **C-ABI** — C Application Binary Interface. CrispASR exposes its surface in `crispasr_c_api.cpp` (175.5 KB) for FFI consumers.
- **FFI** — Foreign Function Interface. CrispASR's official bindings: Python (pure ctypes), Rust (`crispasr-sys` + `crispasr`), Dart (`flutter/crispasr/`), plus `bindings/{go,java,javascript,ruby}/`.
- **BYOP** — Bring Your Own Provider. WhisperJAV's existing extension pattern for external ASR binaries (`whisperjav/byop/xxl_runner.py`).
- **TranscriptionBackend** — Susurrus's abstract base class (`workers/transcription/backends/base.py`) with `transcribe()` generator + `cleanup()` contract.
- **LID** — Language Identification. CrispASR ships a fast pre-step via `-l auto` using the ggml-tiny model.
- **CTC** — Connectionist Temporal Classification. The alignment algorithm class used by CrispASR's NeMo-style forced aligner.
- **TDT** — Token-and-Duration Transducer. The decoder pattern used by NVIDIA Parakeet.
- **CAP_*** — Per-backend capability bitmask flags (19 enumerated in `01_crispasr_dossier.md` §10). `CAP_WORD_TIMESTAMPS` and `CAP_TIMESTAMPS_CTC` are the two most relevant for alignment-integration design.
- **5-file recipe** — CrispASR's documented pattern for contributing a new backend with minimal touchpoints to shared primitives.

---

## Phase 0 conventions (followed throughout)

- **Primary sources only.** Every substantive claim cites a URL + file path (and line number when applicable).
- **Descriptive, not prescriptive.** No "should", "recommend", or "would be best to" framing.
- **Acknowledge uncertainty.** Could-not-verify is captured per-dossier and aggregated in 07.
- **No code changes.** Phase 0 was research and documentation only.

---

_Phase 0 complete. Awaiting direction on Phase 1._
