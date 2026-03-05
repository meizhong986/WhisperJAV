# ADR-006 Audit A: Vision vs. Implementation Plan Gap Analysis

| Field       | Value                                        |
|-------------|----------------------------------------------|
| **Date**    | 2026-02-17                                   |
| **Scope**   | ADR-006 (Vision) vs. ADR-006-IMPLEMENTATION-PLAN |
| **Auditor** | Claude Opus 4.6 (architectural gap analysis) |

---

## Executive Summary

The Implementation Plan is a **strong, high-fidelity translation** of the ADR-006 vision into actionable phases. The plan covers the majority of the ADR's architectural elements: all four protocols, all shared data types, the orchestrator design, the hardening stage, the reconstruction function, and the VRAM lifecycle management. The phasing strategy (additive-first, existing-code-modifications-last) faithfully implements the ADR's migration strategy from Section 11.2.

However, the audit identifies **23 discrete gaps** between the two documents. The most significant cluster of gaps relates to **deferred scope that the ADR presents as core architecture but the plan explicitly defers**: the `whisper-segment` framer, the `VADProportionalAligner`, and the `HybridAligner` are all described as part of the proposed architecture (Section 6.1 block diagram, Section 7.4 implementations table) but are deferred to "future" in the plan. The plan is transparent about this (Section 10.3), but the ADR does not distinguish between "core v1" and "future expansion" in its architectural diagrams, creating a perception gap.

The second cluster involves **minor type/contract discrepancies**: the ADR's `harden_scene_result()` signature (Section 9.3) differs from the plan's implementation (Section 2.3) in parameter structure; the orchestrator's `process_scenes()` parameter naming diverges between the two documents; and the plan adds a `Phase 7` that has no corresponding ADR section. These are all individually low-severity but collectively indicate the plan evolved beyond the ADR during implementation.

Overall severity assessment: **No CRITICAL gaps. 3 HIGH, 8 MEDIUM, 12 LOW.** The architecture is sound and the plan is a faithful-enough translation that no fundamental redesign is needed to close the gaps.

---

## Methodology

1. Read both documents in their entirety, noting every architectural element (protocol, type, method, workflow, constant, design decision) in the ADR.
2. For each element, searched the implementation plan for corresponding coverage, recording which phase/step addresses it.
3. Read each phase of the implementation plan and cross-referenced back to the ADR to identify additions, omissions, or modifications.
4. Checked the actual git commits (Phase 0 through Phase 7) to understand whether the implementation plan was itself faithfully executed, which informed whether deviations are "plan-only" or "plan+implementation."
5. Classified all findings by severity (CRITICAL/HIGH/MEDIUM/LOW) based on impact on architectural integrity, correctness, and maintainability.

---

## A. Coverage Matrix

### A.1 Protocol Specifications (ADR Section 7)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `TemporalFramer` protocol definition | 7.1 | Phase 0, Step 2.2 | COVERED | Exact match on method signatures |
| `TemporalFrame` dataclass | 7.1 | Phase 0, Step 2.1 | COVERED | All fields present |
| `FramingResult` dataclass | 7.1 | Phase 0, Step 2.1 | COVERED | Plan adds `field(default_factory=dict)` for metadata; ADR omits this detail |
| `TextGenerator` protocol definition | 7.2 | Phase 0, Step 2.2 | COVERED | Exact match |
| `TranscriptionResult` dataclass | 7.2 | Phase 0, Step 2.1 | COVERED | Exact match |
| `TextCleaner` protocol definition | 7.3 | Phase 0, Step 2.2 | COVERED | Exact match |
| `TextAligner` protocol definition | 7.4 | Phase 0, Step 2.2 | COVERED | Exact match |
| `WordTimestamp` dataclass | 7.4 | Phase 0, Step 2.1 | COVERED | Exact match |
| `AlignmentResult` dataclass | 7.4 | Phase 0, Step 2.1 | COVERED | Plan adds `field(default_factory=dict)`; ADR omits |

### A.2 TemporalFramer Backends (ADR Section 5.4 / 7.1)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `vad-grouped` backend | 5.4, 7.1 | Phase 2, Step 4.3 | COVERED | |
| `whisper-segment` backend | 5.4, 7.1 | Phase 2 factory note | **DEFERRED** | Plan Section 4.1.2 explicitly defers. Plan Section 10.3 documents deferral. ADR Section 6.1 includes it in the block diagram without distinction. |
| `srt-source` backend | 5.4, 7.1 | Phase 2, Step 4.4 | COVERED | |
| `full-scene` backend | 5.4, 7.1 | Phase 2, Step 4.2 | COVERED | |
| `manual` backend | 5.4, 7.1 | Phase 2, Step 4.5 | COVERED | |

### A.3 TextGenerator Implementations (ADR Section 7.2)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `Qwen3TextGenerator` | 7.2 | Phase 1, Step 3.1.3 | COVERED | |
| `TransformersGenerator` | 7.2 | — | **DEFERRED** | Plan Section 10.3 documents. ADR lists in Section 7.2 table. |
| `VLLMGenerator` | 7.2 | — | **DEFERRED** | Plan Section 10.3 documents. ADR Section 14.4 references ADR-005. |
| `WhisperTextGenerator` | 7.2 | — | **DEFERRED** | Plan Section 10.3 documents. |

### A.4 TextAligner Implementations (ADR Section 7.4)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `Qwen3ForcedAlignerAdapter` | 7.4 | Phase 1, Step 3.2.3 | COVERED | |
| `CTCAligner` | 7.4 | — | **DEFERRED** | Plan Section 10.3 documents. |
| `VADProportionalAligner` | 7.4 | — | **DEFERRED** | Plan Section 10.3 states "aligner-free path in orchestrator Step 9 covers the core logic inline." ADR lists as a distinct backend. |
| `NoneAligner` | 7.4 | Phase 1, Step 3.2.4 | COVERED | |
| `HybridAligner` | 7.4 | — | **DEFERRED** | Plan Section 10.3 documents. Needs 2+ working aligners first. |

### A.5 TextCleaner Implementations (ADR Section 7.3)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `AssemblyTextCleaner` wrapper | 7.3 | Phase 1, Step 3.3.3 | COVERED | Plan names it `Qwen3TextCleaner` |
| `PassthroughCleaner` | 7.3 | Phase 1, Step 3.3.4 | COVERED | |

### A.6 Orchestrator Design (ADR Section 9)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| `DecoupledSubtitlePipeline` class | 9.1 | Phase 3, Step 5.1 | COVERED | |
| `__init__()` signature | 9.1 | Phase 3, Step 5.1 | **MODIFIED** | See Deviation B.3 — parameter naming differs |
| `process_scenes()` signature | 9.1 | Phase 3, Step 5.1 | **MODIFIED** | See Deviation B.4 — parameter naming differs |
| 9-step internal flow | 9.2 | Phase 3, Step 5.2 | COVERED | Plan provides detailed pseudocode for all 9 steps |
| VRAM swap pattern | 9.2 (Steps 4, 8) | Phase 3, Steps 2-4, 5-8 | COVERED | |
| Hardening stage | 9.3 | Phase 0, Step 2.3 + Phase 3 Step 9 | COVERED | |
| Sentinel integration | 9.4 | Phase 3, Step 5.3 | COVERED | |
| Sentinel stats accumulation | — | Phase 3, Step 5.4 | COVERED | Plan adds explicit detail not in ADR |

### A.7 Data Flow & Contracts (ADR Section 8)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| Inter-component data flow diagram | 8.1 | Phase 3, Step 5.2 | COVERED | Implicit via orchestrator pseudocode |
| Package location (`subtitle_pipeline/`) | 8.2 | Phase 0 + all phases | COVERED | Exact directory match |
| Phase 5 -> Phase 6 contract (WhisperResult) | 8.3 | Phase 0 (reconstruction.py) + Phase 4 | COVERED | |

### A.8 Audit Finding Resolutions (ADR Section 10)

| ADR Finding | ADR Section 10 Resolution | Plan Coverage | Status | Notes |
|---|---|---|---|---|
| C1: OOM stale closure | Eliminated by lifecycle | Phase 5, Step 7.1.1 | COVERED | Plan provides targeted fix for existing code too |
| C2: Wrong post-processor | TextCleaner protocol | Phase 5, Step 7.1.2 | COVERED | |
| H1: VAD_ONLY overlapping | Fixed in hardening | Phase 0, Step 2.3 | COVERED | Explicit rewrite documented |
| H2: Assembly no timestamps | Fixed in hardening | Phase 0, Step 2.3 | COVERED | |
| H3: suppress_silence conflict | Design decision in reconstruction | Phase 0, Step 2.4 | COVERED | |
| H4: Wrong diagnostics | Fixed in hardening | Phase 0, Step 2.3 | COVERED | |
| M1: Stale 300s comment | Resolved by extraction | Phase 5, Step 7.2 | COVERED | |
| M2: end=0.0 as NULL | Explicit null in AlignmentResult | Phase 0 (types.py) | **PARTIAL** | ADR says "explicit null representation"; plan's WordTimestamp still uses float 0.0. See Gap C.4. |
| M3: No batch_size override | TextGenerator.load() config | Phase 1, Step 3.1.3 | COVERED | Constructor accepts independent batch_size |
| M4: Private _exact_lists | Encapsulated in TextCleaner | Phase 1, Step 3.3.3 | COVERED | |
| M5: Redundant VRAM cleanup | safe_cuda_cleanup() after unload | Phase 1, Step 3.1.3 + Phase 3 | COVERED | |
| M6: Redundant audio checks | Duration from TemporalFrame | Phase 2 (framers) | COVERED | |
| M7: Inconsistent timestamp modes | Single hardening path | Phase 0, Step 2.3 | COVERED | |
| M8: suppress_silence coupled | Same as H3 | Phase 0, Step 2.4 | COVERED | |
| L1-L7: Dead code/cosmetic | Fresh implementation | Phase 6 | COVERED | |

### A.9 Mode Mapping (ADR Section 11)

| ADR Element | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| ASSEMBLY mapping | 11.1 | Phase 4 | COVERED | Full-scene + qwen3 + assembly cleaner + qwen-forced-aligner |
| VAD_SLICING mapping | 11.1 | — | **DEFERRED** | Plan Section 10.3 explicitly defers migration |
| CONTEXT_AWARE mapping | 11.1 | — | **DEFERRED** | Plan Section 10.3: "needs design work" |
| Migration strategy Phase 0-3 | 11.2 | Phases 0-3 | COVERED | |
| Migration strategy Phase 4 | 11.2 | Phase 4 | COVERED | Assembly only, as specified |
| Migration strategy Phase 5 "Future" | 11.2 | — | **DEFERRED** | ADR says "Deprecate mode-specific code as orchestrator proves equivalent" |

### A.10 New Workflows (ADR Section 12)

| ADR Workflow | ADR Section | Plan Phase/Step | Status | Notes |
|---|---|---|---|---|
| Whisper-Guided Qwen | 12.1 | — | **NOT COVERED** | Depends on `whisper-segment` framer (deferred) |
| SRT Re-Transcription | 12.2 | Phase 2 (srt-source framer) | **PARTIAL** | Framer exists, but no integration guidance. Workflow is "enabled" but not exercised. |
| Two-Pass Refinement | 12.3 | — | **NOT COVERED** | Enabled by architecture but no plan step addresses wiring it |
| Aligner-Free Fast Mode | 12.4 | Phase 3, Step 9 (aligner=None path) | COVERED | Orchestrator handles aligner-free explicitly |
| Cross-Model Benchmarking | 12.5 | — | **DEFERRED** | Plan Section 10.3: "Depends on stable protocol contracts" |

### A.11 Appendices & Constants (ADR Sections 13, 14, Appendices)

| ADR Element | ADR Section | Plan Coverage | Status | Notes |
|---|---|---|---|---|
| What doesn't change (Phases 1-4, 6-8) | 13 | Plan Section 1.4 | COVERED | Files modified list is minimal |
| Future TextGenerator implementations | 14.1 | Plan 10.3 (deferred) | DEFERRED | |
| Future TextAligner implementations | 14.2 | Plan 10.3 (deferred) | DEFERRED | |
| Diagnostic/Benchmarking utility | 14.3 | Plan 10.3 (deferred) | DEFERRED | |
| vLLM integration | 14.4 | Plan 10.3 (deferred) | DEFERRED | |
| Alignment Sentinel reference | App A | Phase 3, Step 5.3 | COVERED | |
| AssemblyTextCleaner reference | App B | Phase 1, Step 3.3.3 | COVERED | |
| Key Constants & Limits | App C | Not explicitly addressed | **GAP** | See Gap C.5 |

---

## B. Deviation Register

### B.1 `harden_scene_result()` Signature Difference

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 0, Step 2.3 |
| **ADR Reference** | Section 9.3 |
| **Type** | MODIFICATION |
| **Severity** | LOW |

**ADR specifies:**
```python
def harden_scene_result(
    result: WhisperResult,
    scene_duration_sec: float,
    timestamp_mode: TimestampMode,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[WhisperResult, Dict]:
```

**Plan specifies:**
```python
def harden_scene_result(
    result: WhisperResult,
    config: HardeningConfig,
) -> HardeningDiagnostics:
```

The plan wraps the individual parameters into a `HardeningConfig` dataclass and returns a typed `HardeningDiagnostics` instead of a generic `Tuple[WhisperResult, Dict]`. This is an **improvement** over the ADR's design (stronger typing, config aggregation), but it is a deviation. The ADR's signature is more explicit about what goes in; the plan's is more extensible.

### B.2 Plan Adds `HardeningConfig` and `HardeningDiagnostics` Types

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 0, Step 2.1 |
| **ADR Reference** | Not present in Section 8 types |
| **Type** | ADDITION |
| **Severity** | LOW |

The ADR does not define `HardeningConfig` or `HardeningDiagnostics` as data types. They appear only implicitly in the `harden_scene_result()` signature (Section 9.3). The plan formalizes them as explicit dataclasses in `types.py`. This is a beneficial addition for type safety and is consistent with the ADR's spirit.

### B.3 `DecoupledSubtitlePipeline.__init__()` Parameter Naming

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 3, Step 5.1 |
| **ADR Reference** | Section 9.1 |
| **Type** | MODIFICATION |
| **Severity** | LOW |

The ADR and plan use identical parameter names in `__init__()`. No deviation on this specific point upon closer inspection.

### B.4 `process_scenes()` Parameter Naming

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 3, Step 5.1 |
| **ADR Reference** | Section 9.1 |
| **Type** | MODIFICATION |
| **Severity** | LOW |

**ADR specifies:**
```python
def process_scenes(
    self,
    scene_paths: List[Path],
    scene_durations: List[float],
    speech_regions_per_scene: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[Optional[WhisperResult], Dict]]:
```

**Plan specifies:**
```python
def process_scenes(
    self,
    scene_audio_paths: List[Path],
    scene_durations: List[float],
    scene_speech_regions: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Tuple[Optional[WhisperResult], Dict[str, Any]]]:
```

Two naming differences: `scene_paths` vs `scene_audio_paths`, and `speech_regions_per_scene` vs `scene_speech_regions`. The plan also adds `Dict[str, Any]` type annotation where the ADR uses bare `Dict`. These are cosmetic but could cause confusion when cross-referencing documents.

### B.5 Plan Adds Phase 7 (Not in ADR)

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 7 (git commit 18341b6) |
| **ADR Reference** | Not present |
| **Type** | ADDITION |
| **Severity** | MEDIUM |

The actual implementation includes a Phase 7 (`_phase5_assembly()` removal + `japanese_postprocess` deprecation across all entry points) that is not described in either the ADR or the implementation plan. The plan's Phase 6 says "Do NOT remove the old `_phase5_assembly()` method" but Phase 7 does exactly that. This represents implementation evolving beyond the plan.

The ADR was updated to "IMPLEMENTED (Phases 0-6 complete)" in its status field, but the actual git log shows Phase 7 also completed. The ADR status is stale/inaccurate relative to actual commits.

### B.6 Plan Adds `TimestampMode` as New Enum

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 0, Step 2.1 |
| **ADR Reference** | Section 9.3 (uses `TimestampMode` by name, no definition) |
| **Type** | ADDITION |
| **Severity** | LOW |

The ADR uses `TimestampMode` in the `harden_scene_result()` signature (Section 9.3) but never defines it. The plan creates it as a new `str, Enum` in `types.py` with four values. The plan also specifies the migration strategy: the old enum stays in `qwen_pipeline.py` during Phases 0-3, then `qwen_pipeline.py` imports from the new location in Phase 4. This is a necessary gap fill by the plan.

### B.7 Plan Adds Batch Fallback Error Handling

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 3, Steps 2-4 |
| **ADR Reference** | Section 9.2 (no error handling detail) |
| **Type** | ADDITION |
| **Severity** | LOW |

The ADR's orchestrator flow (Section 9.2) does not describe error handling for generation or alignment failures. The plan adds explicit error handling: batch failure falls back to per-frame generation, per-frame failure logs error and sets text to empty string. This is a beneficial addition consistent with the ADR's general resilience goals.

### B.8 Plan Changes ADR Migration Phase Numbering

| Attribute | Value |
|---|---|
| **Plan Phase** | Section 1.2 (6 phases, later 7) |
| **ADR Reference** | Section 11.2 (4 phases) |
| **Type** | MODIFICATION |
| **Severity** | LOW |

The ADR describes migration in 4 phases (0-3 + "Future"). The plan expands to 7 phases (0-6, with Phase 7 added during implementation). The mapping is:

| ADR Phase | Plan Phases |
|---|---|
| Phase 0 (types, protocols, hardening, reconstruction) | Phase 0 |
| Phase 1 (Qwen3 adapters) | Phase 1 |
| Phase 2 (orchestrator, sentinel) | Phases 2 + 3 (framers separated from orchestrator) |
| Phase 3 (wire into QwenPipeline) | Phase 4 |
| Future (deprecate) | Phases 5, 6, 7 |

This is a refinement, not a deviation. The plan correctly separates framers (Phase 2) from the orchestrator (Phase 3), which the ADR lumps into a single phase. And the plan breaks "Future" into concrete Phase 5 (bug fixes), Phase 6 (dead code), and Phase 7 (deprecation).

### B.9 Plan Defers `whisper-segment` Framer with Explicit Justification

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 2, Step 4.1.2 (note) |
| **ADR Reference** | Section 5.4, 7.1, 12.1 |
| **Type** | OMISSION (intentional) |
| **Severity** | MEDIUM |

The ADR describes `whisper-segment` as one of the five core backends (Section 5.4), includes it in the block diagram (Section 6.1), specifies its behavior in the backend table (Section 7.1), and builds an entire workflow around it (Section 12.1 "Whisper-Guided Qwen"). The plan explicitly defers it with justification: "requires loading a Whisper model which adds complexity (VRAM management, model download)."

This is the single largest functional gap between the documents. The `whisper-segment` backend is the enabler for the most novel workflow the ADR proposes. Without it, the architecture is proven for existing Assembly-equivalent flows but has not yet demonstrated its cross-model composability promise.

### B.10 Plan Defers `VADProportionalAligner` with Inline Alternative

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 3, Step 9 (inline logic) |
| **ADR Reference** | Section 7.4 |
| **Type** | MODIFICATION |
| **Severity** | LOW |

The ADR lists `VADProportionalAligner` as a distinct backend in the TextAligner implementations table (Section 7.4), described as "No neural model -- VAD + proportional char distribution." The plan covers this functionality inline in the orchestrator's Step 9 (aligner-free path: word timestamps derived from frame boundaries) rather than as a separate backend class. The plan explicitly notes this tradeoff in Section 10.3: "dedicated backend adds marginal value."

This is a reasonable simplification. The inline logic achieves the same result. A dedicated backend could be extracted later if needed.

### B.11 Plan Specifies `qwen3` Cleaner Naming (ADR Uses `assembly`)

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 1, Step 3.3 |
| **ADR Reference** | Section 11.1 |
| **Type** | MODIFICATION |
| **Severity** | LOW |

The ADR's mode mapping table (Section 11.1) shows the TextCleaner as `assembly (Qwen3)` for all three modes. The plan names the implementation `Qwen3TextCleaner` and the factory key is `"qwen3"`. The renaming is reasonable (it IS a Qwen3-specific cleaner that wraps AssemblyTextCleaner), but creates a minor naming inconsistency with the ADR.

### B.12 Plan Adds Debug Artifact Details

| Attribute | Value |
|---|---|
| **Plan Phase** | Phase 3, Step 5.5 |
| **ADR Reference** | Not specified in detail |
| **Type** | ADDITION |
| **Severity** | LOW |

The ADR mentions debug artifacts only in the comparison table (Section 4.2, Row R) as an "Assembly advantage." The plan provides specific artifact file naming conventions and four artifact types per scene. This is a beneficial addition.

### B.13 Plan Adds Explicit Test File Structure

| Attribute | Value |
|---|---|
| **Plan Phase** | Appendix C |
| **ADR Reference** | Not present |
| **Type** | ADDITION |
| **Severity** | LOW |

The ADR contains no testing guidance. The plan adds Appendix C with a complete test directory structure and per-phase verification criteria. This is a beneficial addition.

---

## C. Completeness Gaps

### C.1 ADR Section 5.5 "The Aligner Becomes Optional" — No Dedicated Plan Coverage

**Severity: LOW**

ADR Section 5.5 describes this as a "key architectural win" and explains why aligner-free workflows are significant. The plan implements this as the `aligner=None` path in the orchestrator (Phase 3, Step 9) and the `NoneAligner` backend (Phase 1). However, there is no dedicated plan section that explicitly maps Section 5.5's vision to implementation steps. The coverage is implicit and adequate, but a cross-reference would improve traceability.

### C.2 ADR Section 6.2 "SpeechEnhancer Placement" — No Plan Reference

**Severity: LOW**

ADR Section 6.2 explains why SpeechEnhancer runs before temporal framing. The plan does not reference this section or the SpeechEnhancer at all, because the plan correctly scopes itself to Phase 5 (ASR) only, and SpeechEnhancer is Phase 3 (pre-existing, unchanged). However, the plan's orchestrator pseudocode (Phase 3, Steps 1-9) shows `framer.frame(scene_audio)` as the entry point, implying the audio is already enhanced. This assumption is correct but undocumented in the plan.

### C.3 ADR Section 6.3 "Existing Protocol + Factory Pattern" — Implicit in Plan

**Severity: LOW**

ADR Section 6.3 documents the existing codebase pattern (SceneDetector, SpeechSegmenter, SpeechEnhancer factories) and states "New domains will follow the same pattern." The plan follows this pattern exactly (three new factories with lazy-import registries) but does not explicitly reference Section 6.3 or state that it is following the existing pattern. The alignment is clear from reading both documents but a cross-reference would improve traceability.

### C.4 ADR Finding M2 Resolution — WordTimestamp Null Representation

**Severity: MEDIUM**

ADR Section 10 states M2 is "Resolved -- sentinel operates on `AlignmentResult.words` which has explicit null representation, not overloaded `0.0`." However, the plan's `WordTimestamp` dataclass (Phase 0, Step 2.1) uses `start: float` and `end: float` with no explicit null representation. There is no `Optional[float]`, no sentinel value, and no documented convention for "no timestamp available."

The ADR claims the design resolves M2 by having "explicit null representation" but neither document actually defines what that representation is. The plan's `WordTimestamp` uses the same `float` type as the old code, meaning `0.0` could still be misinterpreted as "no timestamp."

### C.5 ADR Appendix C "Key Constants & Limits" — No Plan Coverage

**Severity: LOW**

ADR Appendix C documents 10 key constants (ForcedAligner hard limit 180s, assembly max scene duration 120s, TEN VAD max group 29s, sentinel thresholds, etc.). The plan does not address where these constants should live in the new architecture or whether they should be extracted/centralized. Some are referenced implicitly (e.g., the plan's `VadGroupedFramer` uses `max_group_duration_s: float = 29.0`), but there is no systematic treatment.

### C.6 ADR "What Doesn't Change" Section (13) — No Explicit Plan Verification

**Severity: LOW**

ADR Section 13 lists 8 components that should remain unchanged. The plan's "Files Modified" list (Section 1.4) implicitly confirms this by listing only 2 modified files. However, Phase 7 (added during implementation) modifies `pass_worker.py`, `main.py`, and `webview_gui/api.py`, which are in the "Ensemble system", "CLI interfaces", and "GUI interfaces" categories listed in Section 13 as "Untouched." These modifications are minor (changing a default parameter), but they technically violate the ADR's "What Doesn't Change" commitment.

### C.7 No Plan Coverage for ADR Section 4 (Architectural Comparison)

**Severity: LOW**

ADR Section 4 provides a detailed 18-row feature comparison between Assembly and VAD_SLICING. The plan uses this comparison as motivation but does not map each comparison row to specific implementation steps. For example:
- Row A (Adaptive Step-Down): Not addressed in the plan at all. The plan defers VAD_SLICING migration.
- Row G (Sentinel Step-Down Deferral): Not addressed.
- Row K (Per-Group Diagnostics): Partially addressed by plan's per-frame diagnostics.
- Row M (Min Duration Guard): Addressed in VadGroupedFramer.

Since the plan is explicitly scoped to Assembly mode first, these gaps are expected and acceptable. But the ADR's comparison table creates an expectation that the plan should address all 18 rows, which it does not.

### C.8 Plan Phase 4 (Integration) Does Not Address Scene Speech Regions Conversion

**Severity: MEDIUM**

The plan's Phase 4 (Section 6.1.3) shows calling `self._subtitle_pipeline.process_scenes()` with `scene_speech_regions`. However, the plan does not detail how `speech_regions_per_scene` (available from Phase 4 speech segmentation) is converted into the format expected by the orchestrator. The ADR's `process_scenes()` expects `List[List[Tuple[float, float]]]`, but the pipeline's Phase 4 may produce speech regions in a different format (e.g., `SegmentationResult.groups`).

The actual implementation commit (Phase 4, 188602c) shows 166 insertions, suggesting this conversion was handled, but the plan document does not specify the mapping.

### C.9 No Plan Coverage for Orchestrator `cleanup()` Method

**Severity: LOW**

The ADR's `TemporalFramer`, `TextGenerator`, and `TextAligner` protocols all include `cleanup()` methods. The plan documents individual adapter cleanup (e.g., "cleanup() calls unload() if loaded") but does not describe the orchestrator's own `cleanup()` method — specifically, in what order it calls cleanup on its four components, or how it handles cleanup errors. This is an implementation detail but important for VRAM leak prevention.

---

## D. Contract/Interface Gaps

### D.1 `harden_scene_result()` Return Type Mismatch

**Severity: LOW**

| Attribute | ADR (Section 9.3) | Plan (Section 2.3) |
|---|---|---|
| Return type | `Tuple[WhisperResult, Dict]` | `HardeningDiagnostics` |
| Mutation | Implicit (returns new result?) | Explicit ("Mutates result in-place. Returns diagnostics.") |

The ADR implies the function returns a (modified result, diagnostics dict) tuple. The plan specifies mutation in-place with a typed diagnostics return. These are incompatible APIs. The plan's approach is architecturally cleaner (no ambiguity about whether the input is modified), but consumers must know to read the mutated input.

### D.2 `AlignmentResult.words` Type — WordTimestamp vs Dict

**Severity: MEDIUM**

The ADR's `AlignmentResult.words` (Section 7.4) contains `List[WordTimestamp]` (typed dataclass). But the ADR's orchestrator pseudocode (Section 9.2, Step 9) says "merge text + timestamps -> word dicts", and the existing `merge_master_with_timestamps()` function returns `List[Dict[str, Any]]`.

The plan's `Qwen3ForcedAlignerAdapter` (Phase 1, Step 3.2.3) says it "wraps in `AlignmentResult`" after calling `merge_master_with_timestamps()`. This means the adapter must convert from `List[Dict]` to `List[WordTimestamp]`. The plan does not document this conversion step explicitly.

Furthermore, `reconstruct_from_words()` (Phase 0, Step 2.4) accepts `List[Dict[str, Any]]` (word dicts), not `List[WordTimestamp]`. This means there must be a conversion back from `WordTimestamp` to dict before reconstruction. Neither document addresses this bidirectional conversion.

### D.3 Sentinel API Operates on Word Dicts, Not WordTimestamp

**Severity: MEDIUM**

The existing `alignment_sentinel.py` functions (`assess_alignment_quality`, `redistribute_collapsed_words`) operate on `List[Dict]` (word dicts with 'word', 'start', 'end' keys). The ADR's types module introduces `WordTimestamp` dataclass. The plan's orchestrator Step 9 calls the sentinel with "all_words" but does not specify whether these are `WordTimestamp` instances or dicts.

If `AlignmentResult.words` contains `WordTimestamp` objects, they must be converted to dicts before passing to the sentinel (and back again after recovery). Neither document addresses this.

### D.4 `TemporalFramer.frame()` — Audio Input Format

**Severity: LOW**

Both documents agree that `frame()` takes `np.ndarray` audio. However, the ADR (Section 7.1) and plan (Section 2.2) both specify `audio: np.ndarray` without specifying expected shape, dtype, or sample rate conventions. The `sample_rate` parameter is provided but the convention for mono vs stereo, float32 vs int16, etc., is not documented in either place.

This is acceptable for an internal protocol but could cause issues when third-party backends are added.

---

## E. Workflow Gaps

### E.1 ASSEMBLY Workflow

**Severity: NONE (fully covered)**

The ADR's ASSEMBLY mapping (Section 11.1: full-scene + qwen3 + assembly cleaner + qwen-forced-aligner) is fully covered by the plan's Phase 4 integration (Section 6.1.2-6.1.3). The wiring is explicit and the plan provides verification criteria.

### E.2 VAD_SLICING Workflow

**Severity: HIGH (explicitly deferred)**

The ADR maps VAD_SLICING to `vad-grouped` + `qwen3-text-only` + `assembly` + `qwen-forced-aligner` (Section 11.1). The plan explicitly defers this migration (Section 10.3: "Step-down logic doesn't fit cleanly; needs design work"). The `vad-grouped` framer is built (Phase 2), but the orchestrator is never wired to use it with the full VAD_SLICING workflow.

This means the architecture's most complex workflow path (the one with adaptive step-down, per-group sentinel, two-tier retry) remains in the old monolithic code. The ADR's vision of a unified orchestrator handling all modes is not realized by the plan.

### E.3 CONTEXT_AWARE Workflow

**Severity: HIGH (explicitly deferred)**

The ADR maps CONTEXT_AWARE to `full-scene` + `qwen3-coupled` + `assembly` + built-in (Section 11.1, noting "May remain as legacy path initially"). The plan defers migration entirely (Section 10.3: "Coupled mode has different semantics; needs design work").

The ADR is partially prepared for this by noting it may "remain as legacy path initially," but the plan's deferral means the coupled-mode architecture is untouched.

### E.4 Whisper-Guided Qwen Workflow (ADR Section 12.1)

**Severity: HIGH (blocked by deferred framer)**

This is the ADR's showcase workflow demonstrating cross-model composability. It requires the `whisper-segment` framer, which the plan defers. Without this framer, the workflow cannot be executed and the architecture's most compelling value proposition is undemonstrated.

### E.5 SRT Re-Transcription Workflow (ADR Section 12.2)

**Severity: MEDIUM (partially enabled)**

The `srt-source` framer is built (Phase 2), so the technical path exists. However, the plan provides no integration guidance for this workflow. There is no CLI flag, no GUI option, and no documented way for a user to invoke this workflow. It is "architecturally enabled" but "functionally unreachable" from the user's perspective.

### E.6 Error Handling — Per-Scene vs Per-Batch

**Severity: LOW**

The ADR (Section 9.2) does not address error handling beyond stating "Result is None if scene processing failed." The plan (Phase 3, Step 9) provides detailed per-scene error isolation with try/except. The plan also adds batch-level error handling (Step 2-4: batch failure falls back to per-frame). This is a plan addition that improves on the ADR's sparse error handling specification.

### E.7 VRAM Lifecycle — Explicit Swap Points

**Severity: NONE (fully covered)**

Both documents describe the same VRAM lifecycle: generator.load() -> generate -> generator.unload() -> safe_cuda_cleanup() -> aligner.load() -> align -> aligner.unload() -> safe_cuda_cleanup(). The plan adds try/finally guards, which is a beneficial addition.

---

## F. Severity Classification

| ID | Finding | Severity | Rationale |
|---|---|---|---|
| E.4 | `whisper-segment` framer deferred, blocking Whisper-Guided Qwen workflow | **HIGH** | The ADR's most novel workflow is unimplementable. This is the primary architectural differentiator from the old monolithic approach. |
| E.2 | VAD_SLICING migration deferred | **HIGH** | The most feature-rich mode (step-down, per-group sentinel, 4-mode timestamps) remains in monolithic code. The unified orchestrator vision is only partially realized. |
| E.3 | CONTEXT_AWARE migration deferred | **HIGH** | Coupled-mode workflow remains untouched. Less impactful than E.2 since CONTEXT_AWARE is simpler, but still means the orchestrator handles only 1 of 3 modes. |
| C.4 | WordTimestamp has no explicit null representation (ADR M2 resolution incomplete) | **MEDIUM** | The ADR claims M2 is resolved, but the plan's types still use `float` with no null convention. Could lead to the same 0.0 misinterpretation the ADR identified. |
| D.2 | WordTimestamp vs Dict conversion undocumented | **MEDIUM** | Bidirectional conversion between typed dataclass and dict is required but not specified. Implementation may work but contract is ambiguous. |
| D.3 | Sentinel API type mismatch with WordTimestamp | **MEDIUM** | Related to D.2. The sentinel functions expect dicts; the protocol types use dataclasses. Conversion path not documented. |
| B.9 | `whisper-segment` framer deferred (plan vs ADR block diagram) | **MEDIUM** | ADR presents 5 framers as core architecture; plan implements 4. The ADR's block diagram does not distinguish core from future. |
| C.8 | Speech regions format conversion unspecified | **MEDIUM** | Phase 4 integration must convert pipeline speech regions to orchestrator format. The plan does not document this mapping. |
| E.5 | SRT Re-Transcription workflow unreachable | **MEDIUM** | Framer exists but no user-facing path to invoke it. |
| B.5 | Phase 7 not in plan or ADR | **MEDIUM** | Implementation evolved beyond documents. _phase5_assembly() removed despite plan saying "keep for reference." ADR status stale. |
| C.6 | Phase 7 violates ADR "What Doesn't Change" | **MEDIUM** | pass_worker.py, main.py, api.py modified despite ADR listing them as "Untouched." Changes are minor (default parameter) but violate the stated scope. |
| B.1 | `harden_scene_result()` signature differs | LOW | Plan improves on ADR design (typed config/diagnostics). Beneficial deviation. |
| B.2 | Plan adds HardeningConfig/HardeningDiagnostics types | LOW | Not in ADR but needed. Beneficial addition. |
| B.4 | `process_scenes()` parameter naming differs | LOW | Cosmetic. `scene_paths` vs `scene_audio_paths`. |
| B.6 | `TimestampMode` enum defined in plan, undefined in ADR | LOW | Necessary gap fill. |
| B.7 | Plan adds batch fallback error handling | LOW | Beneficial addition not in ADR. |
| B.8 | Migration phase numbering differs (ADR 4 phases, plan 7) | LOW | Refinement, not deviation. |
| B.10 | VADProportionalAligner deferred, covered inline | LOW | Reasonable simplification. |
| B.11 | Cleaner naming: ADR "assembly", plan "qwen3" | LOW | Minor naming inconsistency. |
| B.12 | Plan adds debug artifact details | LOW | Beneficial addition. |
| B.13 | Plan adds test file structure | LOW | Beneficial addition. |
| C.1 | ADR Section 5.5 — no dedicated plan section | LOW | Coverage is implicit and adequate. |
| C.5 | ADR Appendix C constants — no systematic plan coverage | LOW | Some constants referenced implicitly in framer defaults. |
| C.9 | Orchestrator cleanup() method not specified | LOW | Implementation detail, but important for VRAM. |
| D.1 | harden_scene_result() return type differs | LOW | Plan improves on ADR design. |
| D.4 | Audio input format conventions undocumented | LOW | Acceptable for internal protocol. |

---

## G. Recommendations

### Priority 1: Document Alignment (address perception gaps)

1. **Update ADR-006 status field** to reflect Phase 7 and the actual final state. Current status says "Phases 0-6 complete" but Phase 7 exists.

2. **Add a "Scope Boundary" section to the ADR** (or annotate Section 6.1's block diagram) distinguishing "v1 core" backends from "future" backends. The block diagram currently shows `whisper-segment`, `ctc-aligner`, `vllm-backend`, and `TransformersGenerator` without distinguishing them from implemented backends. This creates false expectations.

3. **Update ADR Section 13 ("What Doesn't Change")** to acknowledge that Phase 7 made minor changes to `pass_worker.py`, `main.py`, and `webview_gui/api.py`.

### Priority 2: Type Contract Clarification (address MEDIUM gaps)

4. **Resolve the WordTimestamp null representation gap** (C.4). Either:
   - (a) Make `WordTimestamp.start` and `.end` `Optional[float]` with `None` meaning "no timestamp", or
   - (b) Document a convention (e.g., `start == end == -1.0` means null), or
   - (c) Acknowledge that M2 is NOT fully resolved and the sentinel's existing 0.0 heuristic persists.

5. **Document the WordTimestamp <-> Dict conversion** (D.2, D.3). Specify:
   - `Qwen3ForcedAlignerAdapter.align()` converts from `Dict` to `WordTimestamp` after `merge_master_with_timestamps()`.
   - The orchestrator converts back from `WordTimestamp` to `Dict` before calling sentinel functions and `reconstruct_from_words()`.
   - Or: modify `reconstruct_from_words()` and sentinel to accept `WordTimestamp` natively.

6. **Document the speech regions format conversion** (C.8) in Phase 4 integration — how pipeline Phase 4 `SegmentationResult` maps to `List[List[Tuple[float, float]]]`.

### Priority 3: Deferred Scope Tracking (acknowledge gaps, create future plan)

7. **Create a tracking issue or ADR addendum** for the `whisper-segment` framer (E.4, B.9). This is the gateway to the most compelling new workflows. Document:
   - What VRAM management challenges it introduces
   - Whether it should load a persistent Whisper model or create/destroy per scene
   - How it interacts with the existing faster-whisper pipeline

8. **Create a design note** for VAD_SLICING migration (E.2). The plan correctly identifies "step-down logic doesn't fit cleanly," but does not elaborate. Key questions for a future design:
   - Does step-down become a framer concern (re-frame on collapse) or an orchestrator concern (retry loop)?
   - Does per-group sentinel integrate into the existing Step 9 or need a per-frame loop?
   - Does the two-tier retry pattern require a new protocol method (e.g., `framer.reframe()`)?

9. **Document a CLI/GUI path for SRT Re-Transcription** (E.5). The `srt-source` framer exists. To make it user-reachable, define:
   - A CLI flag (e.g., `--temporal-framer srt-source --srt-path existing.srt`)
   - How it integrates with the existing `--qwen-input-mode` parameter or replaces it

### Priority 4: Minor Cleanup (LOW severity)

10. **Harmonize parameter naming** between ADR and plan (`scene_paths` vs `scene_audio_paths`, `speech_regions_per_scene` vs `scene_speech_regions`). Update whichever document is considered the source of truth.

11. **Add audio format conventions** to the `TemporalFramer` protocol docstring (D.4): expected dtype (float32), channel count (mono), sample rate (parameterized).

12. **Harmonize cleaner naming** (B.11): decide whether the factory key should be `"qwen3"` or `"assembly"` and update the non-matching document.

---

*End of Gap Analysis*
