# ADR-001: YAML-Driven Configuration Architecture (v4)

**Status:** Accepted
**Date:** 2024-11-29
**Decision Makers:** WhisperJAV Development Team

---

## Context

WhisperJAV needed a configuration system that could support:

1. **Multiple ASR ecosystems** (Transformers, Kaldi, BERT) with different models
2. **GUI auto-generation** from configuration schemas
3. **Patchability** - update settings without rebuilding/redistributing the package
4. **Extensibility** - add new models with minimal effort ("drop and go")
5. **Sensitivity presets** across all components

The existing configuration systems (v1-v3) had limitations:

| Version | Approach | Limitation |
|---------|----------|------------|
| v1 | JSON config + Python classes | Required Python changes to add models |
| v2 | Component registry | Still Python-coupled, no GUI hints |
| v3 | Pydantic components | Better validation, but still Python-first |

### Key Question
> "In which approach can I, as the main developer, update settings (parameter values) or add parameters, without needing to repackage and redistribute the entire product?"

**Answer:** Only a **config-driven (YAML) approach** allows patching without redistribution.

---

## Decision

We adopt a **YAML-first configuration architecture** (v4) with the following characteristics:

### 1. YAML as Source of Truth
All configuration lives in YAML files under `whisperjav/config/v4/ecosystems/`.
Python code reads and validates YAML but does not define configuration structure.

### 2. Flat Dot-Notation Parameters
Parameters use flat keys with semantic prefixes:
```yaml
spec:
  model.id: "kotoba-tech/kotoba-whisper-v2.0"
  model.device: auto
  decode.beam_size: 5
```

This is easier to merge, patch, and display in GUIs than nested structures.

### 3. Schema Validation with Pydantic
YAML files are validated against Pydantic models at load time.
Invalid configurations fail fast with clear error messages.

### 4. GUI Hints in Configuration
```yaml
gui:
  model.device:
    widget: dropdown
    options: [auto, cuda, cpu]
```

The GUI reads these hints to auto-generate configuration panels.

### 5. Lazy-Loading Registries
Configurations are discovered and loaded on demand, with caching for performance.

### 6. Independence from Legacy
The v4 system has **zero dependencies** on v1-v3 code.
Both can coexist during migration.

---

## Alternatives Considered

### Alternative 1: Extend v3 Python Components
**Rejected because:**
- Still requires Python changes to add models
- Cannot patch without redistribution
- GUI hints would need to be added to Python classes

### Alternative 2: Hydra/OmegaConf
**Rejected because:**
- Heavy dependency for our use case
- Learning curve for contributors
- Overkill for single-application config

### Alternative 3: Pure JSON
**Rejected because:**
- No comments (harder to document inline)
- More verbose than YAML
- Less human-readable

---

## Consequences

### Positive
- **Zero Python changes to add a model** - just create a YAML file
- **Patchable** - users can modify YAML without rebuilding
- **GUI auto-generation** - widget hints in config enable dynamic UI
- **Clear separation** - YAML defines "what", Python defines "how"
- **Version evolution** - `schemaVersion` enables future changes

### Negative
- **Two config systems** - v4 coexists with legacy during transition
- **YAML learning curve** - contributors need basic YAML knowledge
- **Potential drift** - if legacy code isn't deprecated, patterns may diverge

### Mitigations
- Clear documentation in `whisperjav/config/v4/README.md`
- Deprecation notices in legacy code and docs
- This ADR for historical context

---

## Implementation

The v4 system is implemented in `whisperjav/config/v4/` with:

| Component | Purpose |
|-----------|---------|
| `schemas/` | Pydantic models for YAML validation |
| `loaders/` | YAML parsing with error handling |
| `registries/` | Lazy-loading config discovery |
| `manager.py` | Central API for config access |
| `gui_api.py` | Frontend-friendly API |
| `ecosystems/` | YAML config files |

See `whisperjav/config/v4/README.md` for usage details.

---

## References

- Research: HuggingFace config patterns, Kubernetes strategic merge, Hydra composition
- Domain Analysis: Identified entities (Ecosystem, Model, Tool, Preset) and relationships
- Validation: Tested against use cases (add model, patch setting, GUI generation)
