# WhisperJAV Configuration Architecture - Authoritative Reference

## Overview

WhisperJAV maintains **THREE DISTINCT CONFIGURATION SYSTEMS**:

| System | Format | Status | Use Case |
|--------|--------|--------|----------|
| **v4 YAML** | YAML files | Current (v1.7.0+) | New development, GUI, patchable |
| **Legacy JSON** | asr_config.json | Stable | Existing pipelines, backward compat |
| **CLI/GUI Args** | Runtime | Active | User overrides |

---

## 1. V4 YAML-DRIVEN SYSTEM (Recommended)

### Directory Structure
```
whisperjav/config/v4/
├── manager.py              # ConfigManager (entry point)
├── gui_api.py              # Frontend API for GUI
├── README.md               # Authoritative documentation
├── errors.py               # Custom exceptions
├── loaders/
│   ├── yaml_loader.py      # Safe YAML parsing
│   └── merger.py           # Deep merge strategies
├── registries/
│   ├── model_registry.py
│   ├── tool_registry.py
│   ├── ecosystem_registry.py
│   ├── preset_registry.py
│   └── base_registry.py
├── schemas/
│   ├── base.py             # ConfigBase, MetadataBlock, GUIHint
│   ├── model.py            # ModelConfig schema
│   ├── ecosystem.py        # EcosystemConfig schema
│   ├── tool.py             # ToolConfig schema
│   └── preset.py           # PresetConfig schema
└── ecosystems/             # YAML config files (SOURCE OF TRUTH)
    ├── transformers/
    │   ├── ecosystem.yaml
    │   └── models/*.yaml
    ├── tools/*.yaml
    └── presets/*.yaml
```

### Key Concepts

#### Ecosystems
Family of related models sharing common defaults and provider.
```yaml
# ecosystems/transformers/ecosystem.yaml
schemaVersion: v1
kind: Ecosystem
metadata:
  name: transformers
  displayName: "HuggingFace Transformers"
defaults:
  model.device: auto
  chunk.batch_size: 16
  decode.language: ja
provider:
  module: whisperjav.modules.transformers_asr
  class: TransformersASR
compatible_tools:
  - auditok-scene-detection
```

#### Models
Specific ASR model with parameters and presets.
```yaml
# ecosystems/transformers/models/kotoba-whisper-v2.yaml
schemaVersion: v1
kind: Model
metadata:
  name: kotoba-whisper-v2
  ecosystem: transformers
spec:
  model.id: "kotoba-tech/kotoba-whisper-v2.0"
  chunk.length_s: 15
  decode.beam_size: 5
presets:
  conservative:
    decode.beam_size: 3
  balanced: {}
  aggressive:
    decode.beam_size: 7
gui:
  decode.beam_size:
    widget: spinner
    min: 1
    max: 20
```

#### Parameter Naming (Dot-Prefix)
```
model.*      → Model loading (model.id, model.device, model.dtype)
chunk.*      → Chunking (chunk.length_s, chunk.batch_size)
decode.*     → Decoding (decode.beam_size, decode.temperature)
quality.*    → Thresholds (quality.no_speech, quality.logprob)
timestamps.* → Timestamp mode
vad.*        → VAD parameters
```

### ConfigManager API
```python
from whisperjav.config.v4 import ConfigManager
manager = ConfigManager()

manager.get_model_config(name, sensitivity, overrides)  # Resolved config
manager.list_models(ecosystem=None)
manager.get_ecosystem(name)
manager.get_tool(name)
manager.get_preset(name)
manager.get_gui_schema(model_name)  # For UI generation
```

### Tools (Scene Detection, VAD)

Tools are reusable auxiliary components. **Scene detection** is configured here:

**File locations:**
```
whisperjav/config/v4/ecosystems/tools/
├── auditok-scene-detection.yaml   # Energy-based (default)
└── silero-scene-detection.yaml    # Neural network-based
```

**Example: auditok-scene-detection.yaml**
```yaml
schemaVersion: v1
kind: Tool
metadata:
  name: auditok-scene-detection
tool_type: scene_detection
spec:
  core.max_duration_s: 29.0
  pass1.energy_threshold: 32
  pass2.energy_threshold: 38
  pass1.max_silence_s: 2.5
  pass2.max_silence_s: 1.8
presets:
  conservative:
    pass1.energy_threshold: 40
    pass2.energy_threshold: 45
  aggressive:
    pass1.energy_threshold: 28
    pass2.energy_threshold: 32
```

**Access via API:**
```python
manager.get_tool("auditok-scene-detection")
manager.get_tool_config("auditok-scene-detection", sensitivity="balanced")
```

---

## 2. LEGACY JSON SYSTEM

### Files
```
whisperjav/config/
├── asr_config.json             # Main config (v4.4 format)
├── manager.py                  # Legacy ConfigManager
├── transcription_tuner.py      # Configuration resolver
├── legacy.py                   # Pipeline mappings
└── sanitization_config.py      # Language rules
```

### asr_config.json Structure
```json
{
  "version": "4.4",
  "pipeline_parameter_map": {
    "balanced": {
      "common": ["common_transcriber_options", "common_decoder_options"],
      "engine": "faster_whisper_engine_options",
      "vad": "silero_vad_options"
    }
  },
  "models": { ... },
  "pipelines": { ... },
  "common_decoder_options": {
    "conservative": { "beam_size": 3 },
    "balanced": { "beam_size": 5 },
    "aggressive": { "beam_size": 7 }
  },
  "silero_vad_options": { ... },
  "feature_configs": { ... }
}
```

### TranscriptionTuner
```python
from whisperjav.config.transcription_tuner import TranscriptionTuner
tuner = TranscriptionTuner()

resolved = tuner.resolve_params(
    pipeline_name="balanced",
    sensitivity="conservative",
    task="transcribe"
)
# Returns: { 'params': {...}, 'workflow': {...}, 'features': {...} }
```

### Feature Configs (Scene Detection - Legacy)

**Location:** `whisperjav/config/asr_config.json` → `feature_configs` section

```json
{
  "feature_configs": {
    "scene_detection": {
      "default_method": "auditok",
      "auditok": {
        "max_duration_s": 29.0,
        "pass1_energy_threshold": 32,
        "pass2_energy_threshold": 38,
        "pass1_max_silence_s": 2.5,
        "pass2_max_silence_s": 1.8,
        "bandpass_low_hz": 200,
        "bandpass_high_hz": 4000,
        "brute_force_fallback": true
      },
      "silero": { ... }
    }
  }
}
```

**Pipeline workflow references:**
```json
{
  "pipelines": {
    "balanced": {
      "workflow": {
        "features": {
          "scene_detection": "default"  // Uses default_method
        }
      }
    }
  }
}
```

---

## 3. PARAMETER RESOLUTION CASCADE

### Precedence Order (Later Wins)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Ecosystem Defaults        (lowest priority)              │
│ 2. Model/Tool Base Spec                                     │
│ 3. Model Preset[sensitivity]                                │
│ 4. Global Preset[sensitivity]                               │
│ 5. User CLI/GUI Overrides    (highest priority)             │
└─────────────────────────────────────────────────────────────┘
```

### Flow Diagram
```
CLI Arguments / GUI Form
         ↓
Sensitivity Selection (conservative/balanced/aggressive)
         ↓
┌──────────────────────────────────────┐
│  v4: ConfigManager.get_model_config  │
│  OR                                  │
│  Legacy: TranscriptionTuner.resolve  │
└──────────────────────────────────────┘
         ↓
Merge: ecosystem → spec → preset → overrides
         ↓
Type Validation + None Filtering
         ↓
Pipeline Execution
```

---

## 4. KEY PARAMETERS & VALUES

### Sensitivity Profiles

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| beam_size | 3 | 5 | 7 |
| temperature | 0.0 | 0.0 | 0.1 |
| logprob_threshold | -0.8 | -1.0 | -1.5 |
| no_speech_threshold | 0.7 | 0.6 | 0.4 |

### Model Parameters

| Parameter | Type | Values | Default |
|-----------|------|--------|---------|
| model.device | enum | auto, cuda, cpu | auto |
| model.dtype | enum | auto, float16, float32 | auto |
| chunk.length_s | int | 5-30 | 15 |
| chunk.batch_size | int | 1-64 | 16 |
| decode.beam_size | int | 1-20 | 5 |
| decode.temperature | float | 0.0-1.0 | 0.0 |
| quality.no_speech | float | 0.0-1.0 | 0.6 |

### VAD Parameters

| Parameter | Type | Default |
|-----------|------|---------|
| threshold | float | 0.5 |
| min_speech_duration_ms | int | 100 |
| max_speech_duration_s | float | 30 |
| min_silence_duration_ms | int | 300 |

### Scene Detection Parameters (Auditok)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| core.max_duration_s | float | 29.0 | Max scene length (seconds) |
| core.min_duration_s | float | 0.2 | Min scene length |
| pass1.energy_threshold | int | 32 | Coarse pass energy (dB) |
| pass2.energy_threshold | int | 38 | Fine pass energy (dB) |
| pass1.max_silence_s | float | 2.5 | Max silence before split (P1) |
| pass2.max_silence_s | float | 1.8 | Max silence before split (P2) |
| audio.bandpass_low_hz | int | 200 | Low freq filter |
| audio.bandpass_high_hz | int | 4000 | High freq filter |
| fallback.brute_force | bool | true | Use fixed chunking as fallback |

**By sensitivity:**

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| pass1.energy_threshold | 40 | 32 | 28 |
| pass2.energy_threshold | 45 | 38 | 32 |
| pass1.max_silence_s | 3.0 | 2.5 | 2.0 |
| pass2.max_silence_s | 2.5 | 1.8 | 1.2 |

---

## 5. CLI ARGUMENTS MAPPING

```bash
# Mode selection (routes to different systems)
--mode balanced     → Legacy TranscriptionTuner
--mode transformers → Direct CLI args (v4-style)

# Common overrides
--sensitivity       → Preset selection
--model            → Model override
--language         → decode.language
--task             → transcribe/translate

# Ensemble mode
--ensemble         → Two-pass workflow
--pass1-pipeline   → Pass 1 config
--pass2-pipeline   → Pass 2 config
```

---

## 6. GUI INTEGRATION

### GUIAPI (Frontend)
```python
from whisperjav.config.v4.gui_api import GUIAPI
api = GUIAPI()

api.get_ecosystems_summary()           # Discovery
api.get_model_schema(model_name)       # UI generation
api.get_resolved_config(name, sens)    # Resolved values
api.validate_overrides(name, dict)     # Validation
```

### GUI Hints in YAML
```yaml
gui:
  decode.beam_size:
    widget: spinner
    min: 1
    max: 20
    step: 1
    group: decode
    label: "Beam Size"
    description: "Higher = more accurate, slower"
```

---

## 7. MERGE STRATEGIES

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| STRATEGIC | Dicts merged, lists replaced | Default (K8s-style) |
| OVERRIDE | Later completely replaces | Simple overrides |
| ADDITIVE | Lists concatenated | Feature accumulation |

---

## 8. FILE CONVENTIONS

| Type | Location | Format |
|------|----------|--------|
| Ecosystem | `v4/ecosystems/<name>/ecosystem.yaml` | YAML |
| Model | `v4/ecosystems/<eco>/models/<name>.yaml` | YAML |
| Tool | `v4/ecosystems/tools/<name>.yaml` | YAML |
| Preset | `v4/ecosystems/presets/<level>.yaml` | YAML |
| Legacy | `config/asr_config.json` | JSON |

---

## 9. ADDING NEW CONFIGURATIONS

### New Model (v4 - Recommended)
1. Create `ecosystems/<eco>/models/<name>.yaml`
2. Follow schema: schemaVersion, kind, metadata, spec, presets, gui
3. No Python code changes required
4. Test: `ConfigManager().list_models()`

### New to Legacy
1. Add to `models` section in asr_config.json
2. Add sensitivity profiles to relevant sections
3. Update `pipeline_parameter_map` if needed
4. Validate: `TranscriptionTuner().validate_configuration()`

---

## 10. VALIDATION

### v4 System
- Pydantic schema validation at load
- Custom exceptions: ModelNotFoundError, EcosystemNotFoundError
- Circular dependency detection

### Legacy System
- JSON schema validation
- Type conversion per backend
- None value filtering

---

## Quick Reference

| Task | v4 Command | Legacy Command |
|------|------------|----------------|
| List models | `ConfigManager().list_models()` | `TranscriptionTuner().list_pipelines()` |
| Get config | `manager.get_model_config(name, sens)` | `tuner.resolve_params(pipe, sens)` |
| GUI schema | `manager.get_gui_schema(name)` | N/A |
| Validate | `manager.validate_config(name, cfg)` | `tuner.validate_configuration()` |
