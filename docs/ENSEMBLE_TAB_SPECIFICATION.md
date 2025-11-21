# Ensemble Tab Feature Specification

**Version:** 1.0
**Date:** 2025-01-21
**Status:** Design Complete, Ready for Implementation
**Author:** Architecture Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Use Cases](#3-use-cases)
4. [Design Approach](#4-design-approach)
5. [Architecture Overview](#5-architecture-overview)
6. [Data Structures](#6-data-structures)
7. [API Contracts](#7-api-contracts)
8. [Component Registry](#8-component-registry)
9. [Flow Examples](#9-flow-examples)
10. [UI Specifications](#10-ui-specifications)
11. [Implementation Plan](#11-implementation-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Future Considerations](#13-future-considerations)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. Executive Summary

### 1.1 Purpose

The Ensemble Tab is a new GUI feature that enables power users to build custom processing pipelines by mixing and matching individual components (ASR engines, VAD systems, features) and fine-tuning their parameters, without being constrained by preset sensitivity profiles.

### 1.2 Key Benefits

- **Fine-grained control**: Users adjust individual parameters instead of using presets
- **Mix-and-match**: Choose any combination of ASR + VAD + Features
- **Discoverability**: Users can explore all available components and their parameters
- **Separation of concerns**: Power users use Ensemble; casual users use simple Main tab

### 1.3 Scope

- New "Ensemble" tab in PyWebView GUI
- Component selection dropdowns
- Parameter customization popups
- Backend API extensions
- No changes to existing Main tab functionality

### 1.4 Prerequisites

This feature builds on the v3.0 Component-Based Configuration Architecture, which provides:
- Self-contained component definitions
- Auto-registration system
- Introspection APIs for GUI population
- Parameter override support

---

## 2. Problem Statement

### 2.1 Current Limitation

In the current WhisperJAV GUI:
- Users select a **mode** (balanced, faster, fast, fidelity)
- Users select a **sensitivity** preset (conservative, balanced, aggressive)
- All parameters are determined by these two choices
- No way to adjust individual parameters (e.g., VAD threshold, beam size)

### 2.2 User Pain Points

1. **Lack of fine-tuning**: "I want aggressive sensitivity but with a slightly higher VAD threshold"
2. **No component choice**: "I want to use Faster-Whisper but without VAD"
3. **No discovery**: "What parameters exist? What do they do?"
4. **Edit source code**: Currently users must edit Python files to change values

### 2.3 Root Cause

The original architecture coupled component selection with preset sensitivity profiles. The v3.0 architecture decoupled these but the GUI was not updated to expose this flexibility.

### 2.4 Success Criteria

A successful implementation allows users to:
- Select individual components independently
- View all parameters for each component
- Adjust any parameter value
- Process media with custom configuration
- Save and load custom configurations (future phase)

---

## 3. Use Cases

### 3.1 Primary Use Cases

#### UC-1: Custom VAD Threshold

**Actor:** Power user
**Precondition:** User has media that requires sensitive voice detection
**Flow:**
1. User opens WhisperJAV GUI
2. User navigates to Ensemble tab
3. User selects ASR: "Faster Whisper"
4. User selects VAD: "Silero"
5. User clicks [Customize] next to VAD
6. Popup shows Silero parameters with defaults
7. User adjusts threshold from 0.5 to 0.25
8. User clicks [Apply]
9. User adds input files
10. User clicks [Start Processing]
11. System processes with custom threshold

**Postcondition:** Transcription uses threshold=0.25
**Validation:** Check resolved config shows `params.vad.threshold = 0.25`

---

#### UC-2: Disable VAD Entirely

**Actor:** User with high-quality studio audio
**Precondition:** Audio has no background noise, VAD is unnecessary
**Flow:**
1. User opens Ensemble tab
2. User selects ASR: "Faster Whisper"
3. User selects VAD: "None"
4. VAD customize button is disabled/hidden
5. User processes media

**Postcondition:** Processing runs without VAD preprocessing
**Validation:** Check resolved config shows `vad_name = "none"`

---

#### UC-3: Enable Scene Detection with Custom Settings

**Actor:** User processing long-form content
**Precondition:** User wants longer scene segments
**Flow:**
1. User opens Ensemble tab
2. User selects ASR: "Stable-TS"
3. User selects VAD: "None"
4. User checks [x] Scene Detection
5. User clicks [Customize] for Scene Detection
6. User adjusts max_duration_s from 180 to 300
7. User processes media

**Postcondition:** Scenes split at 300s max instead of 180s
**Validation:** Check resolved config shows `features.scene_detection.max_duration_s = 300`

---

#### UC-4: High-Quality Transcription Settings

**Actor:** User prioritizing accuracy over speed
**Precondition:** User has time and wants best possible output
**Flow:**
1. User opens Ensemble tab
2. User selects ASR: "Faster Whisper"
3. User clicks [Customize] for ASR
4. User adjusts:
   - beam_size: 10 (from 5)
   - patience: 2.0 (from 1.2)
   - best_of: 10 (from 5)
5. User processes media

**Postcondition:** Higher quality transcription (slower)
**Validation:** Check resolved config shows adjusted ASR parameters

---

#### UC-5: Switch ASR Engine

**Actor:** User wanting to try Stable-TS
**Flow:**
1. User opens Ensemble tab
2. User changes ASR dropdown from "Faster Whisper" to "Stable-TS"
3. Customize button now shows Stable-TS parameters
4. User can adjust Stable-TS-specific settings

**Postcondition:** Processing uses Stable-TS backend
**Validation:** Check resolved config shows `asr_name = "stable_ts"`

---

### 3.2 Edge Cases

#### EC-1: Invalid Parameter Values

**Scenario:** User enters threshold = 5.0 (valid range is 0.0-1.0)
**Expected:** Validation error shown, [Apply] disabled
**Implementation:** Frontend validation from parameter schema

#### EC-2: Incompatible Component Combination

**Scenario:** Future component with limited compatibility
**Expected:** Incompatible options disabled or warning shown
**Implementation:** Use `compatible_vad` / `compatible_asr` from component metadata

#### EC-3: Component Has No Customizable Parameters

**Scenario:** Future simple component with no options
**Expected:** [Customize] button disabled or hidden
**Implementation:** Check if component Options class has any fields

---

### 3.3 Negative Use Cases

#### NC-1: Casual User on Ensemble Tab

**Scenario:** Casual user accidentally opens Ensemble tab
**Expected:** Can still process with defaults (no customization required)
**Implementation:** All components have sensible defaults

#### NC-2: Missing Required Selection

**Scenario:** User clicks [Start] without selecting ASR
**Expected:** Validation error: "Please select an ASR engine"
**Implementation:** Frontend validation before start_process

---

## 4. Design Approach

### 4.1 Design Principles

1. **Progressive Disclosure**: Show complexity only when requested
2. **Sensible Defaults**: Everything works without customization
3. **Clear Feedback**: Show current values, validation errors
4. **Isolation**: Ensemble tab is independent of Main tab
5. **Consistency**: Use same UI patterns as existing GUI

### 4.2 UI Pattern: Popup for Settings

**Rationale:**
- Keeps main tab uncluttered
- Groups related parameters
- Clear action buttons (Apply/Cancel)
- Can be modal (focused interaction)

**Alternative Considered:** Inline expansion
- Rejected: Makes tab too long, harder to scan

### 4.3 No Sensitivity Presets

**Rationale:**
- Ensemble users want full control
- Presets would override their customizations
- Simplifies mental model: "I set everything myself"

**Consequence:**
- Each component shows "balanced" defaults initially
- User can adjust from there

### 4.4 Component Selection Pattern

```
[Dropdown to select component] [Customize button]
```

- Dropdown: Choose which component (or "None")
- Customize: Opens parameter popup (disabled if "None")

---

## 5. Architecture Overview

### 5.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│                     WhisperJAV GUI                          │
│  ┌─────────┐  ┌─────────────┐  ┌──────────┐                │
│  │ Main    │  │ Ensemble    │  │ Settings │                │
│  │ Tab     │  │ Tab (NEW)   │  │ Tab      │                │
│  └────┬────┘  └──────┬──────┘  └──────────┘                │
│       │              │                                      │
│       └──────┬───────┘                                      │
│              │                                              │
│         PyWebView API                                       │
└──────────────┼──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Backend (Python)                          │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ WhisperJAVAPI   │  │ Config System   │                   │
│  │ (api.py)        │  │ v3.0            │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│           └────────┬───────────┘                             │
│                    │                                         │
│           Component Registry                                 │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐                       │
│  │ ASR     │ │ VAD     │ │ Features │                       │
│  │ Registry│ │ Registry│ │ Registry │                       │
│  └─────────┘ └─────────┘ └──────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 File Structure

```
whisperjav/
├── webview_gui/
│   ├── api.py                    # Backend API (extend)
│   └── assets/
│       ├── index.html            # Add Ensemble tab HTML
│       ├── style.css             # Add Ensemble styles
│       └── app.js                # Add Ensemble logic
│
├── config/
│   ├── resolver_v3.py            # Core resolution (exists)
│   ├── legacy.py                 # Legacy mappings (exists)
│   └── components/
│       ├── base.py               # Component base classes (exists)
│       ├── __init__.py           # Registry access (exists)
│       ├── asr/
│       │   ├── faster_whisper.py # ASR component (exists)
│       │   └── stable_ts.py      # ASR component (exists)
│       ├── vad/
│       │   └── silero.py         # VAD component (exists)
│       └── features/
│           └── scene_detection.py # Feature component (exists)
```

### 5.3 Existing API Methods (Already Implemented)

These methods in `api.py` already support the Ensemble tab:

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_available_components()` | List all registered components | `{asr: [...], vad: [...], features: [...]}` |
| `get_component_schema(type, name)` | Get parameter schema for component | `{schema: [...], metadata: {...}}` |
| `get_legacy_pipelines()` | List legacy pipeline names | `{pipelines: [...]}` |
| `start_process(options)` | Start processing with options | `{success: bool, message: str}` |

### 5.4 New API Methods Needed

| Method | Purpose | Parameters | Returns |
|--------|---------|------------|---------|
| `start_ensemble_process(options)` | Process with Ensemble config | See Section 7.2 | `{success: bool, ...}` |
| `validate_ensemble_config(options)` | Validate before processing | Same as above | `{valid: bool, errors: [...]}` |
| `get_component_defaults(type, name)` | Get default values | `type`, `name` | `{defaults: {...}}` |

---

## 6. Data Structures

### 6.1 Component Metadata (Existing)

Located in component files (e.g., `faster_whisper.py`):

```python
@register_asr
class FasterWhisperASR(ASRComponent):
    name = "faster_whisper"
    display_name = "Faster Whisper"
    description = "Fast and accurate ASR using CTranslate2"
    provider = "faster_whisper"
    model_id = "large-v2"
    supported_tasks = ["transcribe", "translate"]
    compatible_vad = ["silero", "faster_whisper_vad", "none"]

    Options = FasterWhisperOptions  # Pydantic model
    presets = {
        "conservative": FasterWhisperOptions(...),
        "balanced": FasterWhisperOptions(...),
        "aggressive": FasterWhisperOptions(...),
    }
```

### 6.2 Options Schema (Existing)

Pydantic models define parameters:

```python
class FasterWhisperOptions(BaseModel):
    """Faster-Whisper ASR parameters."""

    task: str = Field(default="transcribe", description="Task type")
    language: str = Field(default="ja", description="Source language")
    beam_size: int = Field(default=5, ge=1, le=20, description="Beam search size")
    patience: float = Field(default=1.2, ge=0.0, le=3.0, description="Beam search patience")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    # ... more fields
```

### 6.3 Ensemble Configuration (Frontend → Backend)

When user clicks [Start Processing] in Ensemble tab:

```javascript
// JavaScript object sent to backend
const ensembleConfig = {
    // Component selections
    asr: "faster_whisper",
    vad: "silero",           // or "none"
    features: ["auditok_scene_detection"],  // array, can be empty

    // Parameter overrides (only non-default values)
    overrides: {
        "asr.beam_size": 10,
        "asr.patience": 2.0,
        "vad.threshold": 0.25,
        "features.scene_detection.max_duration_s": 300
    },

    // Standard options
    inputs: ["C:/videos/video1.mp4"],
    output_dir: "C:/output",
    task: "transcribe",
    language: "ja"
};
```

### 6.4 Resolved Configuration (Backend Output)

After `resolve_config_v3()` processes the Ensemble config:

```python
{
    'asr_name': 'faster_whisper',
    'vad_name': 'silero',
    'sensitivity_name': 'custom',  # Indicates Ensemble mode
    'task': 'transcribe',
    'language': 'ja',

    'model': {
        'provider': 'faster_whisper',
        'model_name': 'large-v2',
        'device': 'cuda',
        'compute_type': 'float16',
        'supported_tasks': ['transcribe', 'translate']
    },

    'params': {
        'asr': {
            'task': 'transcribe',
            'language': 'ja',
            'beam_size': 10,      # Overridden
            'patience': 2.0,      # Overridden
            'temperature': 0.0,
            # ... all parameters with values
        },
        'vad': {
            'threshold': 0.25,    # Overridden
            'min_speech_duration_ms': 250,
            'min_silence_duration_ms': 100,
            # ... all parameters with values
        }
    },

    'features': {
        'scene_detection': {
            'max_duration_s': 300,  # Overridden
            'min_silence_len': 700,
            # ... all parameters with values
        }
    }
}
```

### 6.5 Parameter Schema for UI

Returned by `get_component_schema()`:

```python
{
    "success": True,
    "schema": [
        {
            "name": "threshold",
            "type": "float",
            "default": 0.5,
            "description": "Speech detection threshold",
            "constraints": {
                "ge": 0.0,
                "le": 1.0
            }
        },
        {
            "name": "min_speech_duration_ms",
            "type": "int",
            "default": 250,
            "description": "Minimum speech duration in milliseconds",
            "constraints": {
                "ge": 0,
                "le": 5000
            }
        }
        // ... more parameters
    ],
    "metadata": {
        "name": "silero",
        "display_name": "Silero VAD",
        "description": "Voice Activity Detection using Silero models",
        "compatible_asr": ["faster_whisper", "stable_ts", "openai_whisper"]
    }
}
```

---

## 7. API Contracts

### 7.1 Existing API: get_available_components

**Purpose:** Populate component dropdowns in Ensemble tab

**Request:** None (no parameters)

**Response:**
```python
{
    "success": True,
    "components": {
        "asr": [
            {
                "name": "faster_whisper",
                "display_name": "Faster Whisper",
                "description": "Fast and accurate ASR using CTranslate2"
            },
            {
                "name": "stable_ts",
                "display_name": "Stable-TS",
                "description": "OpenAI Whisper with stable timestamps"
            }
        ],
        "vad": [
            {
                "name": "silero",
                "display_name": "Silero VAD",
                "description": "Voice Activity Detection using Silero models"
            }
        ],
        "features": [
            {
                "name": "auditok_scene_detection",
                "display_name": "Scene Detection",
                "description": "Audio-based scene splitting"
            }
        ]
    }
}
```

**Error Response:**
```python
{
    "success": False,
    "error": "Failed to load components: <error message>"
}
```

### 7.2 New API: start_ensemble_process

**Purpose:** Start processing with Ensemble configuration

**Request:**
```python
{
    # Required
    "inputs": ["path/to/file.mp4"],
    "asr": "faster_whisper",

    # Optional with defaults
    "vad": "none",
    "features": [],
    "overrides": {},
    "output_dir": "<default>",
    "task": "transcribe",
    "language": "ja",
    "temp_dir": "",
    "keep_temp": False,
    "verbosity": "summary"
}
```

**Response (Success):**
```python
{
    "success": True,
    "message": "Process started successfully",
    "command": "whisperjav.main <args>"
}
```

**Response (Validation Error):**
```python
{
    "success": False,
    "message": "Invalid configuration: beam_size must be between 1 and 20"
}
```

**Response (Process Error):**
```python
{
    "success": False,
    "message": "Failed to start process: <error>"
}
```

### 7.3 New API: validate_ensemble_config

**Purpose:** Validate configuration before processing (for real-time UI feedback)

**Request:** Same as `start_ensemble_process`

**Response (Valid):**
```python
{
    "valid": True,
    "warnings": []  # Optional warnings
}
```

**Response (Invalid):**
```python
{
    "valid": False,
    "errors": [
        {
            "field": "overrides.asr.beam_size",
            "message": "Value 25 exceeds maximum of 20"
        },
        {
            "field": "vad",
            "message": "Unknown VAD component: invalid_vad"
        }
    ]
}
```

### 7.4 New API: get_component_defaults

**Purpose:** Get default parameter values for a component (to populate UI initially)

**Request:**
```python
{
    "component_type": "vad",
    "component_name": "silero"
}
```

**Response:**
```python
{
    "success": True,
    "defaults": {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30
    }
}
```

---

## 8. Component Registry

### 8.1 Registry Architecture

Components self-register via decorators:

```python
# In whisperjav/config/components/vad/silero.py

from whisperjav.config.components.base import VADComponent, register_vad

@register_vad
class SileroVAD(VADComponent):
    name = "silero"
    # ... rest of definition
```

When the module is imported, the decorator adds the class to `_vad_registry`.

### 8.2 Registry Access Functions

Located in `whisperjav/config/components/__init__.py`:

```python
def get_all_components() -> Dict[str, List[Dict]]:
    """Get all registered components for GUI population."""

def get_asr(name: str) -> Type[ASRComponent]:
    """Get ASR component class by name."""

def get_vad(name: str) -> Type[VADComponent]:
    """Get VAD component class by name."""

def get_feature(name: str) -> Type[FeatureComponent]:
    """Get feature component class by name."""

def get_component(component_type: str, name: str):
    """Get any component by type and name."""
```

### 8.3 Current Registered Components

| Type | Name | Display Name | File Location |
|------|------|--------------|---------------|
| ASR | faster_whisper | Faster Whisper | `components/asr/faster_whisper.py` |
| ASR | stable_ts | Stable-TS | `components/asr/stable_ts.py` |
| VAD | silero | Silero VAD | `components/vad/silero.py` |
| Feature | auditok_scene_detection | Scene Detection | `components/features/scene_detection.py` |

### 8.4 Adding New Components

To add a new component (e.g., new VAD):

1. Create file: `components/vad/new_vad.py`
2. Define Options class (Pydantic model)
3. Define component class with `@register_vad`
4. Import in `components/vad/__init__.py`

The component automatically appears in:
- `get_available_components()` response
- Ensemble tab dropdowns (after GUI refresh)

No other files need modification.

---

## 9. Flow Examples

### 9.1 Flow: User Opens Ensemble Tab

```
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│  User   │     │ Frontend │     │   API.py    │     │ Registry │
└────┬────┘     └────┬─────┘     └──────┬──────┘     └────┬─────┘
     │               │                  │                 │
     │ Click         │                  │                 │
     │ Ensemble Tab  │                  │                 │
     │──────────────>│                  │                 │
     │               │                  │                 │
     │               │ get_available_   │                 │
     │               │ components()     │                 │
     │               │─────────────────>│                 │
     │               │                  │                 │
     │               │                  │ get_all_        │
     │               │                  │ components()    │
     │               │                  │────────────────>│
     │               │                  │                 │
     │               │                  │ {asr:[...],     │
     │               │                  │  vad:[...],     │
     │               │                  │  features:[...]}│
     │               │                  │<────────────────│
     │               │                  │                 │
     │               │ {success: true,  │                 │
     │               │  components:...} │                 │
     │               │<─────────────────│                 │
     │               │                  │                 │
     │  Populate     │                  │                 │
     │  dropdowns    │                  │                 │
     │<──────────────│                  │                 │
     │               │                  │                 │
```

### 9.2 Flow: User Clicks Customize Button

```
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│  User   │     │ Frontend │     │   API.py    │     │ Registry │
└────┬────┘     └────┬─────┘     └──────┬──────┘     └────┬─────┘
     │               │                  │                 │
     │ Click         │                  │                 │
     │ [Customize]   │                  │                 │
     │ for Silero    │                  │                 │
     │──────────────>│                  │                 │
     │               │                  │                 │
     │               │ get_component_   │                 │
     │               │ schema("vad",    │                 │
     │               │        "silero") │                 │
     │               │─────────────────>│                 │
     │               │                  │                 │
     │               │                  │ get_vad         │
     │               │                  │ ("silero")      │
     │               │                  │────────────────>│
     │               │                  │                 │
     │               │                  │ SileroVAD class │
     │               │                  │<────────────────│
     │               │                  │                 │
     │               │                  │ Extract schema  │
     │               │                  │ from Options    │
     │               │                  │                 │
     │               │ {schema: [...],  │                 │
     │               │  metadata: {...}}│                 │
     │               │<─────────────────│                 │
     │               │                  │                 │
     │  Show popup   │                  │                 │
     │  with sliders │                  │                 │
     │<──────────────│                  │                 │
     │               │                  │                 │
```

### 9.3 Flow: User Starts Processing

```
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌───────────┐
│  User   │     │ Frontend │     │   API.py    │     │ Resolver  │
└────┬────┘     └────┬─────┘     └──────┬──────┘     └─────┬─────┘
     │               │                  │                  │
     │ Click         │                  │                  │
     │ [Start]       │                  │                  │
     │──────────────>│                  │                  │
     │               │                  │                  │
     │               │ Collect UI       │                  │
     │               │ values into      │                  │
     │               │ ensembleConfig   │                  │
     │               │                  │                  │
     │               │ start_ensemble_  │                  │
     │               │ process(config)  │                  │
     │               │─────────────────>│                  │
     │               │                  │                  │
     │               │                  │ resolve_         │
     │               │                  │ config_v3(...)   │
     │               │                  │─────────────────>│
     │               │                  │                  │
     │               │                  │ Merged config    │
     │               │                  │<─────────────────│
     │               │                  │                  │
     │               │                  │ Build CLI args   │
     │               │                  │ Start subprocess │
     │               │                  │                  │
     │               │ {success: true}  │                  │
     │               │<─────────────────│                  │
     │               │                  │                  │
     │  Processing   │                  │                  │
     │  started      │                  │                  │
     │<──────────────│                  │                  │
     │               │                  │                  │
```

### 9.4 Flow: Complete User Journey

1. **User opens GUI** → Main tab shown by default
2. **User clicks Ensemble tab** → `get_available_components()` called
3. **Dropdowns populate** with ASR, VAD, Features options
4. **User selects ASR** → "Faster Whisper"
5. **User selects VAD** → "Silero"
6. **User clicks [Customize] for VAD** → `get_component_schema("vad", "silero")`
7. **Popup shows** with threshold slider, etc.
8. **User adjusts threshold** to 0.25
9. **User clicks [Apply]** → Popup closes, value stored in state
10. **User adds input files** via file dialog
11. **User clicks [Start Processing]** → `start_ensemble_process(config)`
12. **Processing begins** with custom threshold
13. **Logs stream** to output panel
14. **Processing completes** → Success message

---

## 10. UI Specifications

### 10.1 Tab Structure

```html
<div class="tabs">
    <button class="tab-btn" data-tab="main">Main</button>
    <button class="tab-btn" data-tab="ensemble">Ensemble</button>
    <button class="tab-btn" data-tab="settings">Settings</button>
</div>

<div id="main-tab" class="tab-content">
    <!-- Existing Main tab content -->
</div>

<div id="ensemble-tab" class="tab-content">
    <!-- New Ensemble tab content -->
</div>
```

### 10.2 Ensemble Tab Layout

```html
<div id="ensemble-tab" class="tab-content">
    <!-- ASR Selection -->
    <div class="component-row">
        <label>ASR Engine:</label>
        <select id="ensemble-asr">
            <option value="faster_whisper">Faster Whisper</option>
            <option value="stable_ts">Stable-TS</option>
        </select>
        <button id="customize-asr" class="btn-customize">Customize</button>
    </div>

    <!-- VAD Selection -->
    <div class="component-row">
        <label>VAD:</label>
        <select id="ensemble-vad">
            <option value="none">None</option>
            <option value="silero">Silero VAD</option>
        </select>
        <button id="customize-vad" class="btn-customize">Customize</button>
    </div>

    <!-- Features -->
    <div class="features-section">
        <label>Features:</label>
        <div class="feature-item">
            <input type="checkbox" id="feature-scene-detection">
            <label for="feature-scene-detection">Scene Detection</label>
            <button id="customize-scene" class="btn-customize">Customize</button>
        </div>
    </div>

    <!-- Input/Output (reuse from Main tab) -->
    <div class="io-section">
        <!-- Input files list -->
        <!-- Output directory -->
    </div>

    <!-- Action buttons -->
    <div class="actions">
        <button id="ensemble-start" class="btn-primary">Start Processing</button>
    </div>
</div>
```

### 10.3 Customize Popup (Modal)

```html
<div id="customize-modal" class="modal">
    <div class="modal-content">
        <div class="modal-header">
            <h3 id="modal-title">Silero VAD Settings</h3>
            <button class="modal-close">&times;</button>
        </div>

        <div class="modal-body" id="modal-parameters">
            <!-- Dynamically generated parameter controls -->
        </div>

        <div class="modal-footer">
            <button id="modal-reset" class="btn-secondary">Reset to Defaults</button>
            <button id="modal-apply" class="btn-primary">Apply</button>
        </div>
    </div>
</div>
```

### 10.4 Parameter Control Templates

**Slider (for float/int with range):**
```html
<div class="param-control">
    <label>Threshold</label>
    <input type="range" min="0" max="1" step="0.01" value="0.5">
    <span class="param-value">0.5</span>
    <p class="param-description">Speech detection threshold (0.0-1.0)</p>
</div>
```

**Number input (for int without tight range):**
```html
<div class="param-control">
    <label>Beam Size</label>
    <input type="number" min="1" max="20" value="5">
    <p class="param-description">Beam search width</p>
</div>
```

**Dropdown (for enum/choices):**
```html
<div class="param-control">
    <label>Task</label>
    <select>
        <option value="transcribe">Transcribe</option>
        <option value="translate">Translate</option>
    </select>
    <p class="param-description">Processing task type</p>
</div>
```

### 10.5 State Management (JavaScript)

```javascript
// Ensemble tab state
const ensembleState = {
    asr: {
        name: "faster_whisper",
        overrides: {}  // Only non-default values
    },
    vad: {
        name: "silero",
        overrides: {
            threshold: 0.25  // User customized
        }
    },
    features: {
        scene_detection: {
            enabled: true,
            overrides: {}
        }
    },
    inputs: [],
    output_dir: ""
};

// When user applies popup changes
function applyCustomization(componentType, values) {
    const defaults = await pywebview.api.get_component_defaults(
        componentType,
        ensembleState[componentType].name
    );

    // Store only non-default values as overrides
    const overrides = {};
    for (const [key, value] of Object.entries(values)) {
        if (value !== defaults[key]) {
            overrides[key] = value;
        }
    }

    ensembleState[componentType].overrides = overrides;
}

// When user clicks Start
async function startEnsembleProcessing() {
    const config = buildEnsembleConfig();
    const result = await pywebview.api.start_ensemble_process(config);
    // Handle result
}
```

### 10.6 CSS Classes

```css
/* Component row */
.component-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
}

.component-row label {
    width: 120px;
    font-weight: bold;
}

.component-row select {
    flex: 1;
}

.btn-customize {
    padding: 5px 10px;
    background: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 4px;
    cursor: pointer;
}

.btn-customize:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Modal */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 1000;
}

.modal.active {
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-content {
    background: white;
    border-radius: 8px;
    width: 500px;
    max-height: 80vh;
    overflow-y: auto;
}

/* Parameter controls */
.param-control {
    margin-bottom: 20px;
}

.param-control label {
    display: block;
    font-weight: bold;
    margin-bottom: 5px;
}

.param-description {
    font-size: 12px;
    color: #666;
    margin-top: 5px;
}
```

---

## 11. Implementation Plan

### 11.1 Phases

#### Phase 1: Backend API Extensions

**Duration:** 1-2 sessions
**Files:** `whisperjav/webview_gui/api.py`

Tasks:
1. Implement `start_ensemble_process(options)`
2. Implement `validate_ensemble_config(options)`
3. Implement `get_component_defaults(type, name)`
4. Add tests for new API methods

Deliverables:
- New API methods functional
- Unit tests passing
- Can be tested via Python REPL

#### Phase 2: Ensemble Tab HTML Structure

**Duration:** 1 session
**Files:** `whisperjav/webview_gui/assets/index.html`

Tasks:
1. Add Ensemble tab button to tab bar
2. Create Ensemble tab content structure
3. Add component selection dropdowns
4. Add feature checkboxes
5. Add Customize buttons
6. Create modal HTML structure

Deliverables:
- HTML structure in place
- Tab switching works
- Elements visible (not yet functional)

#### Phase 3: JavaScript Logic

**Duration:** 2-3 sessions
**Files:** `whisperjav/webview_gui/assets/app.js`

Tasks:
1. Tab switching logic
2. Populate dropdowns from `get_available_components()`
3. Customize button click handlers
4. Modal open/close logic
5. Dynamic parameter control generation from schema
6. State management for overrides
7. Start button handler calling `start_ensemble_process()`
8. Validation feedback

Deliverables:
- Fully functional Ensemble tab
- Customization popups working
- Processing starts with custom config

#### Phase 4: CSS Styling

**Duration:** 1 session
**Files:** `whisperjav/webview_gui/assets/style.css`

Tasks:
1. Style component rows
2. Style modal
3. Style parameter controls
4. Responsive adjustments
5. Disabled states

Deliverables:
- Professional appearance
- Consistent with existing GUI style

#### Phase 5: Testing & Polish

**Duration:** 1-2 sessions

Tasks:
1. Test all use cases from Section 3
2. Edge case handling
3. Error message improvements
4. Performance testing
5. Cross-browser testing (WebView2)

Deliverables:
- All use cases pass
- No console errors
- Smooth user experience

### 11.2 Priority Order

1. **Phase 1: Backend** - Foundation for everything
2. **Phase 2: HTML** - Structure to work with
3. **Phase 3: JavaScript** - Core functionality
4. **Phase 4: CSS** - Polish
5. **Phase 5: Testing** - Quality assurance

### 11.3 Dependencies

```
Phase 1 (Backend)
    │
    ▼
Phase 2 (HTML) ──────┐
    │                │
    ▼                ▼
Phase 3 (JavaScript)
    │
    ▼
Phase 4 (CSS)
    │
    ▼
Phase 5 (Testing)
```

---

## 12. Testing Strategy

### 12.1 Unit Tests

**Backend API tests** (`tests/webview_gui/test_api_ensemble.py`):

```python
class TestStartEnsembleProcess:
    def test_basic_ensemble_config(self):
        """Test processing with basic ensemble config."""

    def test_with_overrides(self):
        """Test overrides are applied correctly."""

    def test_invalid_component_raises(self):
        """Test unknown component returns error."""

    def test_invalid_override_value(self):
        """Test out-of-range override returns error."""

class TestValidateEnsembleConfig:
    def test_valid_config(self):
        """Test valid config passes validation."""

    def test_missing_asr_fails(self):
        """Test missing ASR returns error."""

    def test_invalid_override_type(self):
        """Test wrong type override returns error."""

class TestGetComponentDefaults:
    def test_returns_defaults(self):
        """Test defaults match component Options defaults."""
```

### 12.2 Integration Tests

**End-to-end flow tests:**

1. **Test: Complete Ensemble workflow**
   - Open GUI
   - Navigate to Ensemble
   - Select components
   - Customize parameters
   - Start processing
   - Verify output uses custom config

2. **Test: Override persistence**
   - Customize component
   - Switch to different component
   - Switch back
   - Verify customizations retained

3. **Test: Reset to defaults**
   - Customize component
   - Click Reset
   - Verify values return to defaults

### 12.3 Manual Test Checklist

#### Ensemble Tab Access
- [ ] Tab button visible
- [ ] Clicking tab shows Ensemble content
- [ ] Other tabs still work

#### Component Selection
- [ ] ASR dropdown populates with all registered ASR
- [ ] VAD dropdown populates with all registered VAD + "None"
- [ ] Features checkboxes populate with all features
- [ ] Selecting "None" for VAD disables its Customize button

#### Customize Popup
- [ ] Clicking Customize opens modal
- [ ] Modal shows correct component name
- [ ] All parameters displayed with correct types
- [ ] Sliders have correct min/max/step
- [ ] Current values shown
- [ ] Description text visible
- [ ] Close button works
- [ ] Apply button saves values and closes
- [ ] Reset button restores defaults

#### Processing
- [ ] Start button validates inputs
- [ ] Error shown if no ASR selected
- [ ] Error shown if no input files
- [ ] Processing starts with custom config
- [ ] Logs show custom parameter values
- [ ] Output correct for custom settings

#### Edge Cases
- [ ] Very long parameter values handled
- [ ] Many overrides don't break UI
- [ ] Rapid clicking doesn't cause issues
- [ ] Browser resize doesn't break layout

### 12.4 Validation Test Cases

For each use case in Section 3, verify:

| Use Case | Validation Point | How to Verify |
|----------|------------------|---------------|
| UC-1 | threshold=0.25 | Check DEBUG logs show vad.threshold=0.25 |
| UC-2 | vad=none | Check logs show "VAD: none" |
| UC-3 | max_duration_s=300 | Check scene splits at 300s |
| UC-4 | beam_size=10 | Check logs show beam_size=10 |
| UC-5 | backend=stable-ts | Check logs show "Using Stable-TS" |

---

## 13. Future Considerations

### 13.1 Save/Load Configurations

**Priority 3 from original plan**

Features:
- Save current Ensemble config to file
- Load previously saved config
- Config stored as JSON in `~/.whisperjav/configs/`
- Dropdown to select saved configs

Implementation notes:
- Add `save_ensemble_config(name, config)` API
- Add `load_ensemble_config(name)` API
- Add `list_saved_configs()` API

### 13.2 CLI Parameter Flags

**Priority 2 from original plan**

Add CLI flags for common parameters:
```bash
whisperjav video.mp4 --vad-threshold 0.25 --beam-size 10
```

Implementation:
- Add argparse arguments
- Pass as overrides to `resolve_config_v3()`

### 13.3 Component Compatibility Warnings

When user selects incompatible combination:
- Check `compatible_vad` / `compatible_asr` fields
- Show warning or disable incompatible options

### 13.4 Parameter Presets in Ensemble

Optional: Let Ensemble users apply a preset as starting point
- "Start from Aggressive preset" button
- Then customize individual values

### 13.5 Real-time Parameter Descriptions

Enhance parameter hints:
- Show impact: "Higher = slower but more accurate"
- Show recommended values for common scenarios
- Link to documentation

### 13.6 A/B Testing Support

For experimenters:
- Process same file with different configs
- Compare outputs side-by-side
- Save comparison results

### 13.7 New Components

Easy to add with current architecture:
- **New ASR:** Whisper.cpp, Kaldi, etc.
- **New VAD:** WebRTC VAD, pyannote, etc.
- **New Features:** Diarization, noise reduction, etc.

Each requires only one new file + GUI auto-updates.

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **ASR** | Automatic Speech Recognition - converts audio to text |
| **VAD** | Voice Activity Detection - identifies speech segments |
| **Component** | Self-contained module (ASR, VAD, or Feature) with metadata and options |
| **Preset** | Named set of parameter values (conservative/balanced/aggressive) |
| **Override** | User-specified parameter value that differs from default |
| **Ensemble** | Custom combination of components with user-defined parameters |
| **Schema** | Description of parameters (type, default, constraints) |
| **Registry** | Dictionary mapping component names to classes |
| **Progressive Disclosure** | UI pattern showing simple options first, advanced on demand |

---

## 15. References

### 15.1 File References

| File | Purpose |
|------|---------|
| `whisperjav/webview_gui/api.py` | Backend API class |
| `whisperjav/webview_gui/assets/index.html` | GUI HTML structure |
| `whisperjav/webview_gui/assets/app.js` | GUI JavaScript logic |
| `whisperjav/webview_gui/assets/style.css` | GUI styling |
| `whisperjav/config/components/base.py` | Component base classes |
| `whisperjav/config/components/__init__.py` | Registry access functions |
| `whisperjav/config/resolver_v3.py` | Configuration resolver |
| `whisperjav/config/components/asr/faster_whisper.py` | Faster Whisper ASR |
| `whisperjav/config/components/asr/stable_ts.py` | Stable-TS ASR |
| `whisperjav/config/components/vad/silero.py` | Silero VAD |
| `whisperjav/config/components/features/scene_detection.py` | Scene detection |

### 15.2 Existing Tests

| Test File | Coverage |
|-----------|----------|
| `tests/config/test_components_base.py` | Component base classes |
| `tests/config/test_resolver_v3.py` | v3.0 resolver |
| `tests/config/test_legacy.py` | Legacy pipeline mappings |
| `tests/config/test_introspection.py` | GUI introspection APIs |

### 15.3 Related Documentation

| Document | Location |
|----------|----------|
| Project README | `README.md` |
| Claude Code guidance | `CLAUDE.md` |
| This specification | `docs/ENSEMBLE_TAB_SPECIFICATION.md` |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-21 | Architecture Team | Initial complete specification |

---

## Approval

This specification is ready for implementation review.

**Next Steps:**
1. Review this document for completeness
2. Clarify any questions
3. Begin Phase 1 implementation

---

*End of Specification*
