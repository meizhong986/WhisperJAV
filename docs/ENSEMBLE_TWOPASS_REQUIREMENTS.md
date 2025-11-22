# Two-Pass Ensemble Pipeline Requirements Document

**Version**: 1.0
**Date**: 2025-11-22
**Status**: Draft
**Supersedes**: ENSEMBLE_PIPELINE_REQUIREMENTS.md (preserved for reference)

---

## 1. Executive Summary

Two-Pass Ensemble is a simplified approach to ensemble processing that combines existing pipelines rather than individual components. Users select two pipeline configurations (with customization), run both on the same media, and merge results using a selected strategy.

**Key Insight**: Pipelines already work. Instead of building complex dynamic component instantiation, we orchestrate existing pipelines and add merge capability.

**Value Proposition**: Higher transcription quality through consensus/comparison of two different approaches, with significantly less implementation complexity.

---

## 2. Concept Overview

### 2.1 What Changes from Original Vision

| Aspect | Original Vision | Pivot Vision |
|--------|-----------------|--------------|
| Selection unit | Individual components (ASR, VAD, Scene) | Whole pipelines (Balanced, Fast, etc.) |
| Architecture | Dynamic component instantiation | Pipeline orchestration + merge |
| Customization | Per-component parameters | Per-pipeline sensitivity + overrides |
| Complexity | High (new EnsemblePipeline class) | Medium (orchestrator + merge engine) |
| Code reuse | Build new | Leverage existing pipelines |

### 2.2 User Workflow

1. Select Pipeline for Pass 1 (e.g., Balanced)
2. Set Sensitivity for Pass 1 (e.g., Aggressive)
3. Optionally customize Pass 1 parameters
4. Select Pipeline for Pass 2 (e.g., Fidelity)
5. Set Sensitivity for Pass 2 (e.g., Conservative)
6. Optionally customize Pass 2 parameters
7. Select Merge Strategy
8. Process → Get merged SRT output

---

## 3. Use Cases

### 3.1 Core Use Cases

#### UC-1: Speed vs Quality Ensemble
**Goal**: Get fast results with quality validation
**Flow**:
- Pass 1: Faster pipeline (quick initial transcription)
- Pass 2: Fidelity pipeline (high-quality validation)
- Merge: Use Fidelity where confidence higher

#### UC-2: Different Sensitivity Comparison
**Goal**: Catch both subtle and obvious speech
**Flow**:
- Pass 1: Balanced + Aggressive sensitivity
- Pass 2: Balanced + Conservative sensitivity
- Merge: Union of both (more complete coverage)

#### UC-3: Backend Comparison
**Goal**: Compare Faster-Whisper vs OpenAI Whisper
**Flow**:
- Pass 1: Balanced (uses Faster-Whisper)
- Pass 2: Fidelity (uses OpenAI Whisper)
- Merge: Confidence-based selection

#### UC-4: Customized Pipeline Comparison
**Goal**: Test specific parameter variations
**Flow**:
- Pass 1: Balanced + beam_size=2
- Pass 2: Balanced + beam_size=5
- Merge: Compare and select best

#### UC-5: Single Pass Fallback
**Goal**: Use ensemble UI with one pass
**Flow**:
- Pass 1: Fidelity
- Pass 2: Disabled
- Result: Standard single-pass output

### 3.2 Edge Cases

#### EC-1: Identical Pipeline Selection
**Trigger**: User selects same pipeline for both passes
**Behavior**: Warn but allow (different sensitivity/overrides still valid)

#### EC-2: Pass 2 Disabled Mid-Process
**Trigger**: User unchecks Pass 2 after starting
**Behavior**: Complete Pass 1 only, skip merge

#### EC-3: One Pass Fails
**Trigger**: Pass 1 succeeds, Pass 2 fails
**Behavior**: Offer to use Pass 1 result only

### 3.3 Negative Use Cases

#### NC-1: No Pipeline Selected
**Expected**: Error - "At least Pass 1 pipeline is required"

#### NC-2: Invalid Merge Strategy
**Expected**: Error - "Unknown merge strategy"

---

## 4. Functional Requirements

### FR-1: Pipeline Selection
System SHALL allow selection from existing pipelines for each pass:
- Balanced
- Fast
- Faster
- Fidelity

### FR-2: Sensitivity Selection
System SHALL allow sensitivity selection per pass:
- Conservative
- Balanced
- Aggressive

### FR-3: Pipeline Customization
System SHALL allow parameter overrides per pass using existing `resolve_legacy_pipeline()` override mechanism.

### FR-4: Pass 2 Enable/Disable
System SHALL allow enabling/disabling Pass 2 via checkbox.

### FR-5: Independent Execution
System SHALL run Pass 1 and Pass 2 as independent pipeline executions.

### FR-6: Result Storage
System SHALL store both SRT results before merging:
- `{filename}_pass1.srt`
- `{filename}_pass2.srt`
- `{filename}_merged.srt` (final output)

### FR-7: Merge Strategy Selection
System SHALL provide merge strategy options:
- Confidence-based (prefer higher confidence scores)
- Timing-based (align by timestamps, pick best)
- Union (include all unique segments)
- Intersection (only matching segments)

### FR-8: Progress Reporting
System SHALL report progress for:
- Pass 1 (0-45%)
- Pass 2 (45-90%)
- Merge (90-100%)

### FR-9: Single Pass Mode
When Pass 2 disabled, system SHALL skip merge and output Pass 1 result directly.

---

## 5. Non-Functional Requirements

### NFR-1: Performance
- Two-pass processing time ≈ 2x single pass (expected)
- Merge operation < 5 seconds for typical SRT

### NFR-2: Storage
- Intermediate SRT files retained for debugging
- Option to clean up intermediate files

### NFR-3: Backward Compatibility
- Existing Transcription Mode tab unchanged
- Existing CLI --mode argument unchanged

---

## 6. Architecture

### 6.1 Component Diagram

```
┌─────────────────────────────────────────────────┐
│              EnsembleOrchestrator               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐         ┌─────────────┐       │
│  │ Pass1Config │         │ Pass2Config │       │
│  │ - pipeline  │         │ - pipeline  │       │
│  │ - sensitiv. │         │ - sensitiv. │       │
│  │ - overrides │         │ - overrides │       │
│  └──────┬──────┘         └──────┬──────┘       │
│         │                       │               │
│         ▼                       ▼               │
│  ┌─────────────┐         ┌─────────────┐       │
│  │  Pipeline   │         │  Pipeline   │       │
│  │  Instance   │         │  Instance   │       │
│  │ (existing)  │         │ (existing)  │       │
│  └──────┬──────┘         └──────┬──────┘       │
│         │                       │               │
│         ▼                       ▼               │
│  ┌─────────────┐         ┌─────────────┐       │
│  │  SRT Pass1  │         │  SRT Pass2  │       │
│  └──────┬──────┘         └──────┬──────┘       │
│         │                       │               │
│         └───────────┬───────────┘               │
│                     ▼                           │
│            ┌─────────────────┐                  │
│            │  MergeEngine    │                  │
│            └────────┬────────┘                  │
│                     │                           │
└─────────────────────┼───────────────────────────┘
                      ▼
              ┌──────────────┐
              │ Merged SRT   │
              └──────────────┘
```

### 6.2 New Modules Required

| Module | Location | Purpose |
|--------|----------|---------|
| EnsembleOrchestrator | `whisperjav/ensemble/orchestrator.py` | Coordinate two-pass execution |
| MergeEngine | `whisperjav/ensemble/merge.py` | Merge SRT results |
| MergeStrategies | `whisperjav/ensemble/strategies.py` | Strategy implementations |

### 6.3 Reused Modules

- `whisperjav/pipelines/*` - All existing pipelines
- `whisperjav/config/legacy.py` - `resolve_legacy_pipeline()`
- `whisperjav/utils/progress_aggregator.py` - Progress tracking

---

## 7. Data Flow

### 7.1 Configuration Structure

```python
ensemble_config = {
    "pass1": {
        "pipeline": "balanced",
        "sensitivity": "aggressive",
        "overrides": {"beam_size": 5}
    },
    "pass2": {
        "enabled": True,
        "pipeline": "fidelity",
        "sensitivity": "conservative",
        "overrides": {}
    },
    "merge": {
        "strategy": "confidence"
    },
    "input": "video.mp4",
    "output_dir": "/output"
}
```

### 7.2 Execution Flow

```
1. EnsembleOrchestrator receives config
2. Resolve Pass 1: resolve_legacy_pipeline("balanced", "aggressive", overrides)
3. Instantiate Pass 1 pipeline (BalancedPipeline)
4. Execute Pass 1 → pass1.srt
5. If pass2.enabled:
   a. Resolve Pass 2: resolve_legacy_pipeline("fidelity", "conservative", overrides)
   b. Instantiate Pass 2 pipeline (FidelityPipeline)
   c. Execute Pass 2 → pass2.srt
   d. Merge results → merged.srt
6. Return merged.srt (or pass1.srt if single pass)
```

---

## 8. Merge Strategies

### 8.1 Confidence-Based
- Parse confidence/probability from each segment
- For overlapping timestamps, pick higher confidence
- Requires: word_timestamps=True for confidence data

### 8.2 Timing-Based
- Align segments by timestamp overlap
- Pick segment with better timing precision
- Useful when one pass has better VAD

### 8.3 Union
- Include all unique segments from both passes
- Remove duplicates by content similarity
- Results in more complete but potentially noisy output

### 8.4 Intersection
- Only include segments present in both passes
- Higher precision, lower recall
- Good for high-confidence output

### 8.5 Manual Review (Future)
- Output diff-style comparison
- User selects preferred segments
- Interactive merge UI

---

## 9. GUI Changes

### 9.1 Ensemble Tab Layout

Replace current component-based layout with:

```
┌─────────────────────────────────────────────────┐
│  Two-Pass Ensemble                              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────── Pass 1 ───────────┐              │
│  │ Pipeline:   [Balanced     ▼] │              │
│  │ Sensitivity:[Aggressive   ▼] │              │
│  │ [⚙ Customize Parameters]     │              │
│  └──────────────────────────────┘              │
│                                                 │
│  ☑ Enable Pass 2                               │
│  ┌─────────── Pass 2 ───────────┐              │
│  │ Pipeline:   [Fidelity     ▼] │              │
│  │ Sensitivity:[Conservative ▼] │              │
│  │ [⚙ Customize Parameters]     │              │
│  └──────────────────────────────┘              │
│                                                 │
│  Merge Strategy: [Confidence-based ▼]          │
│                                                 │
│  [Process]                                      │
└─────────────────────────────────────────────────┘
```

### 9.2 Customization Modal

Same modal pattern as original, but populated with pipeline's resolved parameters:

```python
# Get parameters for customization
config = resolve_legacy_pipeline("balanced", "aggressive")
# Display config['params']['decoder'] + config['params']['provider']
```

---

## 10. API Changes

### 10.1 New API Methods (api.py)

```python
def get_pipeline_defaults(self, pipeline: str, sensitivity: str) -> Dict:
    """Get resolved parameters for a pipeline+sensitivity combination."""

def start_ensemble_twopass(self, config: Dict) -> Dict:
    """Start two-pass ensemble processing."""

def get_merge_strategies(self) -> List[Dict]:
    """Return available merge strategies with descriptions."""
```

### 10.2 CLI Arguments

```
--ensemble              Enable two-pass ensemble mode
--pass1-pipeline        Pipeline for pass 1
--pass1-sensitivity     Sensitivity for pass 1
--pass1-overrides       JSON overrides for pass 1
--pass2-pipeline        Pipeline for pass 2
--pass2-sensitivity     Sensitivity for pass 2
--pass2-overrides       JSON overrides for pass 2
--merge-strategy        Merge strategy name
```

---

## 11. Success Criteria

### SC-1: Core Functionality
- [ ] Pass 1 executes with selected pipeline
- [ ] Pass 2 executes with selected pipeline
- [ ] Both passes use correct sensitivity
- [ ] Overrides are applied correctly
- [ ] Merge produces valid SRT

### SC-2: GUI
- [ ] Pipeline dropdowns populate correctly
- [ ] Sensitivity dropdowns work
- [ ] Customize button shows pipeline parameters
- [ ] Pass 2 checkbox enables/disables
- [ ] Progress bar shows all stages

### SC-3: Merge Strategies
- [ ] Confidence-based produces reasonable output
- [ ] At least 2 strategies implemented for v1

### SC-4: Edge Cases
- [ ] Single pass mode works
- [ ] Failed pass handled gracefully
- [ ] Intermediate files retained

---

## 12. Implementation Phases

### Phase 1: Core Orchestration
- EnsembleOrchestrator class
- Sequential pass execution
- Basic merge (simple confidence-based)

### Phase 2: GUI Integration
- Update Ensemble tab layout
- Pipeline/sensitivity dropdowns
- Customize modal for pipeline params

### Phase 3: Merge Strategies
- Implement remaining strategies
- Strategy selection in GUI/CLI

### Phase 4: Polish
- Progress reporting refinement
- Error handling
- Intermediate file management

---

## 13. Out of Scope

1. **Component-level selection** - Preserved in ENSEMBLE_PIPELINE_REQUIREMENTS.md for future
2. **Parallel pass execution** - Sequential for v1
3. **More than 2 passes** - Future enhancement
4. **Interactive merge UI** - Future enhancement
5. **Automatic strategy selection** - User chooses

---

## Appendix: Preserved Original Vision

The original component-based ensemble vision is preserved in:
`docs/ENSEMBLE_PIPELINE_REQUIREMENTS.md`

This may be revisited in future versions after two-pass ensemble proves value.

---

*End of Requirements Document*
