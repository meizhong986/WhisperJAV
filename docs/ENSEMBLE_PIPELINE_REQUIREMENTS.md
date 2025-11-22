# EnsemblePipeline Requirements Document

**Version**: 1.0
**Date**: 2025-11-22
**Status**: Draft

---

## 1. Executive Summary

EnsemblePipeline is a dynamic pipeline implementation that enables runtime component selection for WhisperJAV's Ensemble Mode. Unlike existing pipelines (Balanced, Fast, Faster, Fidelity) which hardcode their components, EnsemblePipeline reads component selections from configuration and instantiates them at runtime.

**Problem Statement**: Current pipelines ignore user's ASR/VAD/Scene selections from the Ensemble Mode GUI. The `args.mode` defaults to "balanced", and BalancedPipeline always instantiates FasterWhisperProASR regardless of user selection.

**Solution**: Create EnsemblePipeline that dynamically instantiates components based on `asr_name`, `vad_name`, and `features` from the resolved configuration.

---

## 2. Use Cases

### 2.1 Core Use Cases

#### UC-1: Basic Component Selection
**Actor**: User
**Precondition**: User opens Ensemble Mode tab
**Flow**:
1. User selects ASR engine (e.g., stable_ts)
2. User selects VAD engine (e.g., silero)
3. User selects Scene detection (e.g., auditok_scene_detection)
4. User clicks Process
5. System creates EnsemblePipeline with selected components
6. System processes media and generates subtitles

**Postcondition**: Subtitles generated using exact components selected

#### UC-2: Component Customization
**Actor**: User
**Precondition**: User has selected a component
**Flow**:
1. User clicks gear icon (⚙) next to ASR engine
2. Modal displays ASR-specific parameters (beam_size, temperature, etc.)
3. User modifies beam_size from 2 to 5
4. User clicks Apply
5. Override is stored in configuration
6. System uses customized parameter during processing

**Postcondition**: Component uses customized parameters

#### UC-3: Minimal Configuration
**Actor**: User
**Precondition**: User wants quick transcription
**Flow**:
1. User selects only ASR engine (faster_whisper)
2. User leaves VAD as "None"
3. User leaves Scene detection as "None"
4. User clicks Process
5. System creates EnsemblePipeline with ASR only

**Postcondition**: Direct transcription without preprocessing

#### UC-4: Maximum Feature Configuration
**Actor**: User
**Precondition**: User wants highest quality output
**Flow**:
1. User selects ASR engine (openai_whisper)
2. User selects VAD engine (silero)
3. User selects Scene detection (auditok_scene_detection)
4. User customizes all three components
5. User clicks Process

**Postcondition**: Full pipeline with all preprocessing stages

#### UC-5: Batch Processing with Ensemble
**Actor**: User
**Precondition**: User has multiple media files
**Flow**:
1. User selects multiple files
2. User configures Ensemble settings
3. User clicks Process
4. System applies same component configuration to all files
5. Progress shown for each file

**Postcondition**: All files processed with identical configuration

### 2.2 Edge Cases

#### EC-1: Invalid Component Combination
**Actor**: System
**Trigger**: User selects incompatible components
**Flow**:
1. User selects stable_ts ASR (has built-in VAD)
2. User also selects silero VAD
3. System detects potential conflict
4. System warns user but allows processing
5. stable_ts internal VAD takes precedence

**Postcondition**: Processing completes with warning logged

#### EC-2: Missing Required Parameter
**Actor**: System
**Trigger**: Component requires parameter not provided
**Flow**:
1. User customizes ASR but leaves required field empty
2. System validates configuration before processing
3. System returns validation error with specific field
4. User corrects configuration

**Postcondition**: User informed of missing parameter

#### EC-3: Component Initialization Failure
**Actor**: System
**Trigger**: Component fails to initialize (e.g., missing model)
**Flow**:
1. User selects openai_whisper with large-v3 model
2. Model not downloaded
3. System attempts initialization
4. Initialization fails with clear error
5. System suggests downloading model

**Postcondition**: Graceful failure with actionable error message

#### EC-4: Override Type Mismatch
**Actor**: System
**Trigger**: User provides wrong type for parameter
**Flow**:
1. User enters "five" for beam_size (expects int)
2. System validates override types
3. System rejects with type error
4. Form shows validation error on field

**Postcondition**: User corrects to valid type

#### EC-5: Empty Feature List
**Actor**: System
**Trigger**: No features selected
**Flow**:
1. User deselects all features (Scene detection = None)
2. System passes empty features list
3. EnsemblePipeline skips feature processing stage
4. Processing continues with ASR only

**Postcondition**: Valid minimal pipeline execution

### 2.3 Negative Use Cases

#### NC-1: No ASR Selected
**Actor**: System
**Trigger**: User attempts processing without ASR
**Expected**: System prevents processing
**Message**: "ASR engine is required for transcription"

#### NC-2: Invalid Component Name
**Actor**: System
**Trigger**: API receives unknown component name
**Expected**: System rejects with clear error
**Message**: "Unknown ASR component: 'invalid_asr'. Available: faster_whisper, stable_ts, openai_whisper"

---

## 3. Forward-Looking Features

### F-1: Second Pass Processing
**Status**: Greyed out in GUI
**Description**: Run two different configurations and merge results
**Future Work**: Implement merge strategies (confidence-based, timing-based)

### F-2: Audio Preprocessing Component
**Status**: Greyed out in GUI
**Description**: Noise reduction, normalization before ASR
**Future Work**: Integrate denoiser, demucs options

### F-3: SRT Postprocessing Component
**Status**: Greyed out in GUI
**Description**: Post-ASR subtitle refinement
**Future Work**: Timing adjustment, line breaking, sanitization as selectable component

### F-4: Translation Tool Integration
**Status**: Greyed out in GUI
**Description**: Translate subtitles after generation
**Future Work**: Integrate whisperjav-translate with provider selection

### F-5: Custom Component Registration
**Description**: Allow users to register custom components
**Future Work**: Plugin system with @register_custom decorator

### F-6: Configuration Presets
**Description**: Save/load ensemble configurations
**Future Work**: Export to JSON, import, preset library

### F-7: A/B Testing Mode
**Description**: Run two configs and compare results
**Future Work**: Diff viewer, quality metrics comparison

---

## 4. Functional Requirements

### FR-1: Dynamic Component Instantiation
EnsemblePipeline SHALL instantiate components at runtime based on configuration values:
- `asr_name` → ASR component class
- `vad_name` → VAD component class
- `features` → Feature component classes

### FR-2: Component Registry Access
EnsemblePipeline SHALL use existing component registries:
- `ASR_REGISTRY` for ASR components
- `VAD_REGISTRY` for VAD components
- `FEATURE_REGISTRY` for feature components

### FR-3: Parameter Passing
EnsemblePipeline SHALL pass resolved parameters to components:
- Decoder parameters to ASR
- VAD parameters to VAD component
- Feature parameters to feature components

### FR-4: Override Application
EnsemblePipeline SHALL apply user overrides from `config['params']` to component initialization.

### FR-5: Fallback to Defaults
When parameter not specified, EnsemblePipeline SHALL use component's default from Options class.

### FR-6: Pipeline Stage Execution
EnsemblePipeline SHALL execute stages in order:
1. Audio extraction (always)
2. Scene detection (if feature selected)
3. VAD preprocessing (if vad_name != 'none')
4. ASR transcription (always)
5. Post-processing (always)

### FR-7: Progress Reporting
EnsemblePipeline SHALL report progress through existing ProgressAggregator interface.

### FR-8: Error Propagation
EnsemblePipeline SHALL propagate component errors with context (which component, which stage).

---

## 5. Non-Functional Requirements

### NFR-1: Performance
- Component instantiation overhead SHALL be < 100ms
- No performance regression vs hardcoded pipelines

### NFR-2: Memory
- EnsemblePipeline SHALL not load unused components
- Memory footprint comparable to existing pipelines

### NFR-3: Maintainability
- Adding new component SHALL require only registry entry
- No changes to EnsemblePipeline class for new components

### NFR-4: Backward Compatibility
- Existing pipeline modes (balanced, fast, faster, fidelity) SHALL continue to work
- CLI `--mode` argument SHALL remain functional

### NFR-5: Testability
- Each component combination SHALL be unit testable
- Integration tests for common configurations

---

## 6. Architectural Requirements

### AR-1: Current Architecture Gap

**Current Flow** (Broken):
```
GUI → API → CLI args → main.py → resolve_ensemble_config() → config dict
                                                                    ↓
                                     IGNORED ←──────────────────────┘
                                        ↓
args.mode="balanced" → BalancedPipeline → HARDCODED FasterWhisperProASR
```

**Required Flow**:
```
GUI → API → CLI args → main.py → resolve_ensemble_config() → config dict
                                                                    ↓
                              EnsemblePipeline ← config['asr_name'] ─┘
                                    ↓
                         Dynamic ASR instantiation
```

### AR-2: Component Registry Pattern
EnsemblePipeline SHALL use existing registry pattern:
```python
from whisperjav.modules import ASR_REGISTRY, VAD_REGISTRY, FEATURE_REGISTRY

asr_class = ASR_REGISTRY[config['asr_name']]
asr_instance = asr_class(**config['params']['asr'])
```

### AR-3: Pipeline Interface Compliance
EnsemblePipeline SHALL extend BasePipeline and implement:
- `process(input_path, output_path, **kwargs)`
- `get_progress()`
- `cancel()`

### AR-4: Configuration Structure
EnsemblePipeline SHALL accept legacy structure from `resolve_ensemble_config()`:
```python
{
    'pipeline_name': 'ensemble',
    'workflow': {'model': '...', 'vad': '...', 'backend': '...'},
    'model': {'model_name': '...', 'device': '...', 'compute_type': '...'},
    'params': {
        'decoder': {...},
        'provider': {...},
        'vad': {...}
    },
    'features': {...}
}
```

---

## 7. Data Flow Specification

### 7.1 GUI to Backend Flow

```
[Ensemble Tab]
     ↓
EnsembleManager.collectConfig()
     ↓
{
  inputs: [...],
  asr: "faster_whisper",
  vad: "silero",
  features: ["auditok_scene_detection"],
  overrides: {"asr.beam_size": 5}
}
     ↓
pywebview.api.start_ensemble_process()
     ↓
_build_ensemble_args()
     ↓
["--asr", "faster_whisper", "--vad", "silero",
 "--features", "auditok_scene_detection",
 "--overrides", '{"asr.beam_size": 5}']
     ↓
subprocess: python -m whisperjav.main ... --asr faster_whisper
```

### 7.2 CLI to Pipeline Flow

```
main.py: parse_args()
     ↓
if args.asr:  # Ensemble mode detected
     ↓
resolve_ensemble_config(asr, vad, features, overrides)
     ↓
{
  'asr_name': 'faster_whisper',
  'vad_name': 'silero',
  'features': {'auditok_scene_detection': {...}},
  'params': {'decoder': {...}, 'provider': {...}, 'vad': {...}}
}
     ↓
_map_to_legacy_structure()
     ↓
EnsemblePipeline(config)  # NEW: dynamic instantiation
     ↓
pipeline.process(input, output)
```

### 7.3 Pipeline Internal Flow

```
EnsemblePipeline.__init__(config)
     ↓
1. Read config['asr_name'] → "faster_whisper"
2. Lookup ASR_REGISTRY["faster_whisper"] → FasterWhisperProASR
3. Instantiate: self.asr = FasterWhisperProASR(**config['params']['provider'])
     ↓
4. Read config['vad_name'] → "silero"
5. Lookup VAD_REGISTRY["silero"] → SileroVAD
6. Instantiate: self.vad = SileroVAD(**config['params']['vad'])
     ↓
7. For each feature in config['features']:
   - Lookup FEATURE_REGISTRY[name]
   - Instantiate with params
     ↓
EnsemblePipeline.process()
     ↓
Execute stages with instantiated components
```

---

## 8. API Contracts

### 8.1 CLI Arguments (Ensemble Mode)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --asr | string | Yes | - | ASR component name |
| --vad | string | No | "none" | VAD component name |
| --features | string | No | "" | Comma-separated feature names |
| --task | string | No | "transcribe" | "transcribe" or "translate" |
| --overrides | JSON | No | null | Parameter overrides |

### 8.2 resolve_ensemble_config() Contract

**Input**:
```python
resolve_ensemble_config(
    asr: str,                          # Required
    vad: str = "none",
    task: str = "transcribe",
    features: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Output**: Legacy-compatible configuration dictionary

### 8.3 EnsemblePipeline Contract

```python
class EnsemblePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        """Initialize with resolved config containing asr_name, vad_name, features."""

    def process(self, input_path: Path, output_path: Path, **kwargs) -> Path:
        """Process media file and return SRT path."""

    def get_progress(self) -> float:
        """Return 0.0-1.0 progress."""

    def cancel(self) -> None:
        """Cancel processing."""
```

---

## 9. Constraints & Dependencies

### 9.1 Dependencies
- Existing component registries (ASR_REGISTRY, VAD_REGISTRY, FEATURE_REGISTRY)
- BasePipeline abstract class
- resolve_ensemble_config() from legacy.py
- ProgressAggregator for progress tracking

### 9.2 Constraints
- Must work with existing GUI without changes
- Must preserve backward compatibility with --mode argument
- Cannot break existing pipeline tests
- Must use existing config resolution (v3 system)

### 9.3 Assumptions
- Component registries are populated at import time
- All components follow consistent Options pattern
- Legacy structure mapping is correct and tested

---

## 10. Success Criteria

### SC-1: Functional Correctness
- [ ] User selects stable_ts → stable_ts is used (not FasterWhisperProASR)
- [ ] User selects silero VAD → silero VAD is used
- [ ] User selects auditok scene detection → auditok is used
- [ ] Overrides are applied to components

### SC-2: Integration
- [ ] GUI Process button triggers EnsemblePipeline
- [ ] CLI --asr argument triggers EnsemblePipeline
- [ ] Progress reporting works
- [ ] Cancellation works

### SC-3: Error Handling
- [ ] Invalid component name returns clear error
- [ ] Missing model returns download suggestion
- [ ] Type mismatches are caught at validation

### SC-4: Backward Compatibility
- [ ] --mode balanced still works
- [ ] --mode fast still works
- [ ] All existing tests pass

### SC-5: Performance
- [ ] No measurable performance regression
- [ ] Startup time < 100ms overhead

---

## 11. Out of Scope

The following are explicitly NOT in scope for initial EnsemblePipeline implementation:

1. **Second pass processing** - GUI element exists but is disabled
2. **Audio preprocessing component** - Future feature
3. **SRT postprocessing component** - Future feature
4. **Translation tool integration** - Separate workflow
5. **Configuration presets** - Future enhancement
6. **Custom component plugins** - Future enhancement
7. **A/B testing mode** - Future enhancement
8. **Batch-specific configurations** - All files use same config

---

## Appendix A: Component Registry Reference

### ASR Components
| Name | Class | Description |
|------|-------|-------------|
| faster_whisper | FasterWhisperProASR | Faster-Whisper backend |
| stable_ts | StableTSASR | Stable-TS backend |
| openai_whisper | WhisperProASR | OpenAI Whisper backend |

### VAD Components
| Name | Class | Description |
|------|-------|-------------|
| none | - | No VAD |
| silero | SileroVAD | Silero VAD |

### Feature Components
| Name | Class | Description |
|------|-------|-------------|
| auditok_scene_detection | AuditokSceneDetector | Audio-based scene splitting |
| silero_scene_detection | SileroSceneDetector | VAD-based scene splitting |

---

## Appendix B: Related Documents

- `CLAUDE.md` - Project overview and architecture
- `whisperjav/config/README.md` - Configuration system v3 documentation
- `whisperjav/pipelines/` - Existing pipeline implementations

---

*End of Requirements Document*
