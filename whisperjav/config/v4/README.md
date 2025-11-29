# WhisperJAV Configuration System v4

> **This is the authoritative configuration architecture for WhisperJAV v1.7.0+**
>
> If you're adding new models, ecosystems, or modifying configuration behavior,
> this is where you should work. The v1-v3 config systems in the parent directory
> are LEGACY and should not be extended.

## Quick Start

### Adding a New Model (YAML-only, no Python)

1. Create a YAML file in the appropriate ecosystem:
   ```
   ecosystems/<ecosystem>/models/<model-name>.yaml
   ```

2. Use this template:
   ```yaml
   schemaVersion: v1
   kind: Model

   metadata:
     name: my-new-model
     ecosystem: transformers
     displayName: "My New Model"
     description: "Description here"
     tags: [japanese, fast]

   spec:
     model.id: "huggingface/model-id"
     model.device: auto
     decode.beam_size: 5

   presets:
     conservative:
       decode.beam_size: 3
     balanced: {}
     aggressive:
       decode.beam_size: 7

   gui:
     model.device:
       widget: dropdown
       options: [auto, cuda, cpu]
   ```

3. **Done.** No Python code changes required.

### Using the Config System

```python
from whisperjav.config.v4 import ConfigManager

manager = ConfigManager()

# List available models
models = manager.list_models()

# Get resolved config
config = manager.get_model_config(
    "kotoba-whisper-v2",
    sensitivity="balanced",
    overrides={"decode.beam_size": 10}
)

# For GUI integration
from whisperjav.config.v4.gui_api import GUIAPI
api = GUIAPI()
schema = api.get_model_schema("kotoba-whisper-v2")
```

---

## Architecture Overview

```
v4/
├── __init__.py           # Package exports
├── manager.py            # Central ConfigManager
├── gui_api.py            # Frontend-friendly API
├── errors.py             # Custom exceptions
├── schemas/              # Pydantic validation models
│   ├── base.py           # ConfigBase, MetadataBlock, GUIHint
│   ├── model.py          # ModelConfig
│   ├── ecosystem.py      # EcosystemConfig
│   ├── tool.py           # ToolConfig
│   └── preset.py         # PresetConfig
├── loaders/              # YAML loading and merging
│   ├── yaml_loader.py    # Safe YAML parsing with validation
│   └── merger.py         # Deep merge, flatten/unflatten
├── registries/           # Lazy-loading registries
│   ├── model_registry.py
│   ├── tool_registry.py
│   ├── ecosystem_registry.py
│   └── preset_registry.py
└── ecosystems/           # YAML config files (SOURCE OF TRUTH)
    ├── transformers/
    │   ├── ecosystem.yaml
    │   └── models/
    │       ├── kotoba-whisper-v2.yaml
    │       └── whisper-large-v3.yaml
    ├── tools/
    │   ├── auditok-scene-detection.yaml
    │   └── silero-scene-detection.yaml
    └── presets/
        ├── conservative.yaml
        ├── balanced.yaml
        └── aggressive.yaml
```

---

## Design Principles

### 1. YAML is the Source of Truth
All configuration lives in YAML files. No Python code changes are needed to:
- Add a new model
- Modify parameter defaults
- Add new sensitivity presets
- Update GUI hints

### 2. Patchable Without Redistribution
Users and developers can update settings by editing YAML files. No need to rebuild or redistribute the package.

### 3. Flat Dot-Notation Parameters
Parameters use flat keys with dot-prefixes for grouping:
```yaml
spec:
  model.id: "kotoba-tech/kotoba-whisper-v2.0"
  model.device: auto
  decode.beam_size: 5
  quality.no_speech: 0.6
```

This is easier to merge and patch than deeply nested structures.

### 4. GUI Auto-Generation
The `gui:` block provides widget hints for automatic UI generation:
```yaml
gui:
  model.device:
    widget: dropdown
    options: [auto, cuda, cpu]
    group: model
  decode.beam_size:
    widget: slider
    min: 1
    max: 20
```

### 5. Schema Versioning
Every YAML file has `schemaVersion: v1`. This allows future schema evolution while maintaining backward compatibility.

### 6. Clear Error Messages
All errors include:
- What went wrong
- Context (file, field, value)
- Actionable suggestion

---

## Key Concepts

### Ecosystems
A family of related models sharing common defaults and a provider implementation.
```
transformers/ecosystem.yaml → defines TransformersASR provider
```

### Models
A specific ASR model with its parameters, presets, and GUI hints.
```
transformers/models/kotoba-whisper-v2.yaml
```

### Tools
Reusable auxiliary components (scene detection, VAD) with contracts.
```
tools/auditok-scene-detection.yaml
```

### Presets
Cross-cutting sensitivity configurations (conservative, balanced, aggressive).
```
presets/balanced.yaml
```

### Config Resolution Order
When getting a model config, values are merged in this order (later wins):
1. Ecosystem defaults
2. Model spec (base values)
3. Model preset[sensitivity]
4. Global preset[sensitivity]
5. User overrides

---

## Adding New Components

### New Model in Existing Ecosystem
1. Create `ecosystems/<ecosystem>/models/<name>.yaml`
2. Follow the model template above
3. Test: `python -c "from whisperjav.config.v4 import ConfigManager; print(ConfigManager().list_models())"`

### New Ecosystem
1. Create directory: `ecosystems/<name>/`
2. Create `ecosystem.yaml` with provider info
3. Create `models/` subdirectory
4. Add at least one model

### New Tool
1. Create `ecosystems/tools/<name>.yaml`
2. Define contract (input/output)
3. Add spec with parameters

---

## Testing

```bash
# Run v4 config tests
python -m pytest tests/test_config_v4.py -v

# Quick validation
python -c "
from whisperjav.config.v4 import ConfigManager
m = ConfigManager()
print('Ecosystems:', m.list_ecosystems())
print('Models:', m.list_models())
print('Tools:', m.list_tools())
"
```

---

## Legacy Systems (DO NOT EXTEND)

The following are LEGACY and exist only for backward compatibility:

| Path | Status | Notes |
|------|--------|-------|
| `config/legacy.py` | LEGACY | v1 pipeline mappings |
| `config/resolver.py` | LEGACY | v1 config resolution |
| `config/resolver_v3.py` | LEGACY | v3 component resolution |
| `config/components/` | LEGACY | Python-based component definitions |
| `config/schemas/` | LEGACY | Old Pydantic schemas |
| `config/asr_config.json` | LEGACY | v1 JSON config file |

**If you need to modify configuration behavior, do it in v4/.**

---

## FAQ

### Why v4 instead of extending v3?
See [ADR-001: YAML-Driven Configuration Architecture](../../../docs/adr/ADR-001-yaml-config-architecture.md)

### Can I still use the legacy config?
Yes, for existing pipelines. But new development should use v4.

### How do I migrate from v3 to v4?
The v4 system is independent. You can use both simultaneously during migration.

### Where are the Python component classes?
In v4, component behavior is defined by the ecosystem's provider class.
Parameters come from YAML, not Python class attributes.
