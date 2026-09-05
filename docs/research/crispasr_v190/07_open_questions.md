# Open Questions Log — Phase 0

Aggregated list of items where primary-source research could not produce a definitive answer. Compiled across all Phase 0 dossiers:

- `01_crispasr_dossier.md` §25
- `02_crisperweaver_dossier.md` §17
- `03_susurrus_dossier.md` §20
- `04_ecosystem_synthesis.md` §K
- `05_whisperjav_integration_surfaces.md` §11
- `06_ideation_inputs_verification.md` (4 ❓ entries)

This is a **forward-looking artifact**. Each question is something a subsequent planning iteration (Phase 1+) may need to resolve. Items are grouped by where the answer most likely lives. Cross-references to source dossiers given inline.

---

## A. CrispASR — engine internals

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| A.1 | What is the full `whisper_full_params` field set? | Configuration mapping for any in-process consumer | Read `include/crispasr.h` (35.3 KB) in full |
| A.2 | What is the exact aligner-dispatch logic in `examples/cli/crispasr_run.cpp` (111.7 KB)? Per-segment branching when both `CAP_WORD_TIMESTAMPS` and `CAP_TIMESTAMPS_CTC` are set | The known defect class in SubtitleEdit#10775 lives here | Read `crispasr_run.cpp` in detail |
| A.3 | Whether SubtitleEdit#10775 is fundamentally a CrispASR-side defect or a SubtitleEdit flag-mapping bug | Determines whether a future integrator inherits the bug | Read both sides; `--aligner-model` vs `-am` flag-name discrepancy is a known clue |
| A.4 | `Session.stream_open()` iteration interface — what generator semantics, what callback shape | Required for streaming integrations | Read `python/crispasr/_binding.py` (83.3 KB) |
| A.5 | macOS code-signing posture for the CrispASR Mach-O binaries shipped in releases | Bundling decisions on macOS | Inspect a downloaded release artifact with `codesign -dvv` |
| A.6 | D3D12 backend — is it actually implemented or just a planned acceleration path? | Windows hardware-accel coverage | Search CMakeLists.txt / src/ggml-cuda / src/ggml-vulkan |
| A.7 | Whether `CrispASR` Python bindings release the GIL during inference | In-process Python concurrency | Inspect ctypes flag usage in `_binding.py`; verify with `py-spy` |
| A.8 | Per-backend Japanese language coverage beyond `whisper`, `qwen3`, `omniasr` | Curation criterion (b) — Japanese support | Per-backend README sections; check `audit-hf-licenses.py` and `gen-feature-matrix.py` outputs |
| A.9 | Whether the C-ABI is committed to stability across point releases (v0.6.x → v0.7.x etc.) | Pinning strategy | Look for an `ABI_STABILITY.md` or equivalent; ask in an issue if absent |
| A.10 | "3.8x faster than voxtral.c" claim from PDF P1 — provenance | If repeating any benchmark numbers in user-facing docs | Search CrispASR PERFORMANCE.md (54.9 KB, not read in Phase 0) |

## B. CrispASR — distribution + consumption

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| B.1 | What's actually inside the ~648 KB of `HISTORY.md`, `PLAN.md`, `TODO.md`, `LEARNINGS.md` | May contain ABI commitments, roadmap signals, known issues | Direct reads, possibly via subagent |
| B.2 | Per-platform binary sizes and dependency tree (what each .zip / .tar.gz actually contains) | Installer impact assessment | Download one release per platform; inspect contents |
| B.3 | Whether prebuilt binaries are dynamically linked against system libraries (e.g., glibc, libstdc++) or fully self-contained | Linux/distro compatibility | `ldd` on a downloaded binary |
| B.4 | Quantization-format catalog per backend (which backends ship which q-levels) | Model-management UX | Browse the HF model pages referenced in CrispASR's README |
| B.5 | Whether HTTP server mode supports concurrent requests / multi-tenant model holding | Server-mode integration design | Read `examples/server/server.cpp` (54.7 KB) |
| B.6 | What the CLI's progress output (ANSI ESC[K / `\r` re-draw) actually looks like in long-running transcription | Subprocess stdout-parser design | Run a sample transcription locally and capture output |

## C. CrisperWeaver — implementation details

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| C.1 | Whether a user-supplied library override path actually exists despite README claim | Library-discovery design reference | Read the full `_libCandidates()` chain; the dossier flagged this as a documentation/code mismatch |
| C.2 | Exact default values of persisted settings keys | Settings-persistence pattern reference | Read settings storage code |
| C.3 | How `NativeCallable.listener` handles backpressure for the microphone streaming callback | Streaming design pattern | Read the audio-capture + FFI streaming code |
| C.4 | Whether AGPL-3.0 obligations propagate to a host that consumes `package:crispasr` (the upstream binding package, MIT) but does NOT consume CrisperWeaver itself | License compliance | Standard AGPL FAQ; legal-flavored, not code-flavored |

## D. Susurrus — limitations to be aware of

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| D.1 | Whether Susurrus's CrispASR backend was abandoned, parked, or is being re-spun elsewhere | Whether to treat it as a live reference or a frozen example | Direct contact or repo-activity wait-and-see |
| D.2 | Why the `BACKEND_MODEL_MAP` registration step (README step 5) was skipped for the CrispASR backend | If we adopt the Susurrus pattern, do we need this step or not | Inspect README vs `config.py` vs `crispasr_backend.py` |
| D.3 | Whether the README's `backends/diarization/manager.py` and `workers/diarize_worker.py` would actually import on a clean checkout against `main` | If we copy any Susurrus diarization code | Try a clean clone + import |

## E. Cross-ecosystem

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| E.1 | Is CrispASR's release cadence (~6.5 releases/week) sustainable, or will it slow? | Pinning strategy and upgrade-burden | Watch release feed over time; no answer available now |
| E.2 | What does the placeholder author identity `crispasr integration <crispasr-dev@localhost>` represent — a script, a CI bot, a delegated identity? | Bus-factor and provenance | Ask the maintainer; check CI workflows |
| E.3 | Is there a `CrispEmbed` v0.x repo worth dossier'ing as part of the ecosystem? (Phase 0 explicitly excluded it.) | Future scope expansion | Out of scope for Phase 0; revisit if RAG / embedding becomes relevant |
| E.4 | What does the FFI vs subprocess performance differential actually look like for a typical JAV-length file? | Integration-pathway design (Phase 1+) | Build a tiny benchmark when an integration-boundary decision is on the table |
| E.5 | Is there a documented or de-facto C-ABI stability guarantee that supports pinning at the .so / .dylib level rather than re-building per CrispASR version? | Library-management story | Read `crispasr-sys/build.rs` (10.1 KB) and check for soversioning |

## F. WhisperJAV-side gaps deliberately not closed in Phase 0

(These are deferred deliberately: they are read-on-demand when a specific integration vector is being evaluated.)

| # | Question | Likely resolution path |
|---|----------|------------------------|
| F.1 | `balanced_pipeline.py` end-to-end audio-segmentation → ASR → output flow | Read the file |
| F.2 | `merge.py` strategy semantics — overlap thresholds, deduplication rules per strategy | Read the file |
| F.3 | `safety_caps.py` (`conditional_sensitivity_cap`) — when it triggers, what it does | Read the file |
| F.4 | `pass_worker.py` lines 1-650 (the `_build_pipeline()` for non-XXL pipelines) | Read the file |
| F.5 | `whisperjav/webview_gui/assets/app.js` full BYOP wiring (inputs, events, validations) | Read the file |
| F.6 | `config/v4/manager.py` ConfigManager priority-chain merge logic | Read the file |
| F.7 | `config/legacy.py` (`LEGACY_PIPELINES`, `resolve_legacy_pipeline`) | Read the file |
| F.8 | `installer/core/registry.py` full `PackageEntry` dataclass surface and 11-extra enum | Read the file |
| F.9 | `modules/subtitle_pipeline/orchestrator.py` decoupled-alignment contract | Read the file |
| F.10 | `modules/subtitle_pipeline/aligners/factory.py` aligner-registration pattern | Read the file |
| F.11 | Conda-constructor flow under `installer/templates/`, `installer/generated/`, post-install hooks | Read these directories |
| F.12 | Whether `asr_config.json` shape has a Pydantic/JSON-Schema validator | Grep for validators |
| F.13 | `whisperjav/pipelines/qwen_pipeline.py` body — to confirm the "CPU-only on macOS" claim and locate the MPS-detection gap | Read the file |

## G. Real-world consumer specifics

| # | Question | Why it matters | Likely resolution path |
|---|----------|----------------|------------------------|
| G.1 | Exact lines of SubtitleEdit's `GetWhisperProcess` for the CrispASR branch (WebFetch truncated the method body) | Most directly portable invocation reference | Local-clone SubtitleEdit; read `src/ui/Features/Video/SpeechToText/SpeechToTextViewModel.cs` |
| G.2 | Whether SubtitleEdit beta23+ now appends `--force-aligner` conditionally from the new combo-box value | If we want to follow that pattern | Read the same view-model post-beta23 |
| G.3 | parakit's exact `Engine::open_with_threads` thread-pool model | Reference for any Rust-FFI-style integration | Read parakit source |
| G.4 | Whether any production consumer has solved the streaming-via-subprocess case cleanly (Wyoming's `AnsiLineReader` looks like a workaround) | Streaming integration reference | Read WyomingCrispAsrServer source |

## H. Strategic / scope-bracketing questions

(These belong to Phase 1+ vision-iteration, NOT to Phase 0. Captured here only so they aren't forgotten.)

| # | Question | Phase to address |
|---|----------|------------------|
| H.1 | Which curated CrispASR backends to ship (curation criteria per CL1) | Phase 1+ — user decides per CL1 |
| H.2 | Whether to follow BYOP-style (subprocess) or another integration boundary | Phase 1+ |
| H.3 | Whether to bundle GGUF models in the installer or first-run-download | Phase 1+ |
| H.4 | Whether to expose CrispASR backends as a single new pipeline (e.g. `--mode crisp`), as additional `--pass2-pipeline` choices, or both | Phase 1+ |
| H.5 | Whether to mirror CrispASR releases on our own infrastructure (for distribution-link stability) | Phase 1+ |

---

## Status of the open-questions list

This list is **not exhaustive**. It captures the gaps that Phase 0 research could observe. Additional gaps will surface during Phase 1 design — they should be added here as they appear.

This list is also **not blocking**. Phase 0's purpose was to build a factual foundation, not to answer every imaginable question. The dossiers are ready to be consumed; the open questions can be resolved on demand when a specific integration decision needs them.
