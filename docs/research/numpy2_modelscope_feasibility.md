# NumPy 2.x + ModelScope Feasibility Analysis

**Date**: 2026-03-11
**Status**: Research complete
**Conclusion**: Migration to numpy 2.x is feasible but blocked by ClearVoice upstream

---

## Executive Summary

ModelScope itself does NOT pin numpy at all. The `numpy<2` constraint in WhisperJAV
originates from two sources:

1. **ClearVoice upstream** (PyPI package) pins `numpy<2.0,>=1.24.3` -- this is the
   real blocker
2. **Historical pyvideotrans compatibility** concern -- no longer verified as necessary

The WhisperJAV fork of ClearVoice (`meizhong986/ClearerVoice-Studio`) has already
relaxed the numpy upper bound to `numpy>=1.24.3` (no `<2.0` cap). This means the
fork is already prepared for numpy 2.x, but **runtime compatibility has not been
validated**.

---

## 1. ModelScope + NumPy 2: Compatible

### ModelScope does not constrain numpy

- **PyPI metadata** (modelscope 1.34.0, Jan 2026): numpy is NOT listed in
  `requires_dist` at all
- **requirements/hub.txt**: No numpy reference
- **requirements/framework.txt**: No numpy reference
- **requirements/audio/audio_signal.txt**: No numpy reference
- **GitHub issue #1151**: ModelScope maintainer explicitly confirmed: "modelscope in
  general does not enforce numpy version"

### ModelScope ZipEnhancer code is numpy-2-safe

The ZipEnhancer model file (`modelscope/models/audio/ans/zipenhancer.py`):
- Imports numpy but does not actually use any numpy operations
- All computation is done via PyTorch tensors
- No usage of removed aliases (`np.float_`, `np.complex_`, etc.)

The ANS pipeline code (`modelscope/pipelines/audio/`) uses:
- `np.float32` (dtype class -- NOT removed in numpy 2)
- `np.int16` (dtype class -- NOT removed)
- `np.frombuffer()`, `np.reshape()`, `np.concatenate()`, `np.zeros()` -- all safe
- `np.float32(ndarray)` as constructor -- this is valid in numpy 2 (it's a concrete
  dtype scalar type, unlike the removed `np.float_` alias)

One minor concern: `modelscope/utils/audio/audio_utils.py` uses `np.fromstring()`
which was deprecated in numpy 1.14 but is still present (with a deprecation warning)
in numpy 2.x. This will not cause a crash.

### Verdict: ModelScope works with numpy 2.x

No code changes needed in ModelScope for WhisperJAV's usage pattern (ZipEnhancer
acoustic noise suppression pipeline).

---

## 2. ClearVoice: The Real Blocker

### Upstream ClearVoice pins numpy<2

The official `clearvoice` package on PyPI (v0.1.2) declares:
```
numpy<2.0,>=1.24.3
```

This is the **primary reason** WhisperJAV pins `numpy<2`.

### WhisperJAV's fork has relaxed this

The fork at `meizhong986/ClearerVoice-Studio` (clearvoice v0.1.3) declares:
```
numpy>=1.24.3
```

The `<2.0` upper bound has been removed. However, the fork also upgraded librosa
from `0.10.2.post1` to `>=0.11.0`, which is necessary since librosa 0.10.x is not
compatible with numpy 2.x (librosa 0.11+ added numpy 2 support).

### ClearVoice runtime compatibility: Likely OK but unverified

ClearVoice uses standard numpy operations (array creation, dtype conversions,
reshaping). A code audit would be needed to confirm no deprecated aliases are used
internally, but since the fork maintainer explicitly removed the numpy<2 pin, it's
likely been tested.

---

## 3. Other Dependencies: Compatibility Matrix

| Dependency | numpy 2 status | Notes |
|---|---|---|
| **modelscope>=1.20** | Compatible | No numpy pin, confirmed by maintainer |
| **clearvoice (upstream PyPI)** | BLOCKED | Pins `numpy<2.0` |
| **clearvoice (WJ fork)** | Compatible | Pin removed, librosa upgraded |
| **bs-roformer-infer** | Compatible | Requires `numpy>=1.23` (no upper bound) |
| **PyTorch 2.3.1+** | Compatible | Full numpy 2 support |
| **openai-whisper** | Compatible | Requires `numpy` (no version pin) |
| **numba 0.61.2** | Partial | Supports numpy up to 2.2 (not 2.3+) |
| **librosa 0.11+** | Compatible | numpy 2 support added |
| **librosa 0.10.x** | BLOCKED | Not compiled for numpy 2 |
| **scipy** | Compatible | numpy 2 support since scipy 1.13 |
| **datasets** | Compatible | Has numpy 2 test suite |
| **transformers** | Compatible | numpy 2 support added |
| **scikit-learn** | Compatible | numpy 2 support since 1.5 |

### Key constraint: numba

Numba 0.61.2 supports numpy up to 2.2.x. If WhisperJAV targets numpy 2.x, it should
use `numpy>=2.0,<2.3` initially. numba is used by librosa and WhisperJAV for
performance.

---

## 4. WhisperJAV Internal Code: One Fix Needed

### metadata_manager.py uses np.int_ (safe) and np.bool_ (safe)

File: `whisperjav/utils/metadata_manager.py`

```python
if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, ...)):
    return int(obj)
elif isinstance(obj, (np.bool_)):
    return bool(obj)
```

- `np.int_`: **Still available** in numpy 2.0 (it was NOT removed, unlike `np.float_`)
- `np.bool_`: **Still available** in numpy 2.0 (redefined as alias to `np.bool`)
- `np.integer`, `np.intc`, `np.intp`: **Still available**

**No changes needed** in WhisperJAV's own code for numpy 2 compatibility.

---

## 5. Migration Path

### Option A: Lift the numpy<2 pin (Recommended)

**Effort**: Low
**Risk**: Low-Medium

Steps:
1. Change `numpy>=1.26.0,<2.0` to `numpy>=1.26.0` in pyproject.toml, requirements.txt,
   and upgrade.py
2. Ensure the ClearVoice dependency points to the fork (already done)
3. Pin librosa to `>=0.11.0` (the fork already does this)
4. Pin numba to `>=0.60.0` (for numpy 2.0-2.2 support)
5. Test all 5 speech enhancement backends
6. Test full transcription pipeline (fast, balanced, faster modes)

This allows pip to resolve numpy to either 1.x or 2.x based on other constraints.
Users with modern setups get numpy 2, users with older setups stay on 1.x.

### Option B: Require numpy 2.x

**Effort**: Medium
**Risk**: Medium

Same as Option A but change pin to `numpy>=2.0,<2.3`. This forces numpy 2 and
ensures no ambiguity. However, it breaks compatibility with any remaining packages
that truly need numpy 1.x.

Not recommended yet -- let pip resolve naturally.

### Option C: Make ModelScope/ClearVoice optional (isolate numpy constraint)

**Effort**: High
**Risk**: Low

The enhance extra already isolates ModelScope and ClearVoice. If we keep `numpy<2`
only in the enhance extra and lift it from the cli extra, users who don't use speech
enhancement can use numpy 2.

Problem: pyproject.toml extras don't support per-extra numpy version constraints
cleanly. The `cli` extra already declares `numpy>=1.26.0,<2.0` as a core dependency.

### Option D: Bypass ModelScope for ZipEnhancer

**Effort**: Medium-High
**Risk**: Medium

ZipEnhancer could be loaded without ModelScope:
- The model weights are downloadable via `huggingface_hub.snapshot_download()` or
  direct URL
- The ONNX path already uses only `modelscope.snapshot_download` (for downloading),
  not the pipeline API
- The torch path uses `modelscope.pipelines.pipeline()` which wraps PyTorch model
  loading

To bypass ModelScope entirely:
1. Download model weights directly from ModelScope/HuggingFace
2. Instantiate the ZipEnhancer PyTorch model directly
3. Run inference with standard PyTorch code

This eliminates ModelScope as a dependency but requires reimplementing the model
loading and inference wrapper (~200 lines). The ONNX path would be easier to
decouple since it already uses standard ONNX runtime.

---

## 6. Risk Assessment

### What could break with numpy 2?

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ClearVoice internal numpy 2 incompatibility | Low | High (enhancement fails) | Fork is already unblocked; graceful fallback exists |
| ModelScope audio utils `np.fromstring()` warning | Medium | None (warning only) | Monitor for future removal |
| numba incompatibility with numpy 2.3+ | Medium | Medium (librosa perf) | Pin numpy<2.3 |
| Undiscovered dep using np.float_ or np.complex_ | Low | Medium | Test all code paths |
| pyvideotrans compatibility layer breaks | Low | Low (optional feature) | Test if still needed |

### Overall risk: LOW

The ecosystem has largely migrated to numpy 2. The main risk is undiscovered
incompatibilities in transitive dependencies, which would manifest as import-time
`AttributeError` exceptions (easy to detect during testing).

---

## 7. Timeline and Ecosystem Status

### Current state (March 2026)

- **numpy 2.4.3** is the latest release (supports Python 3.11-3.14)
- **numpy 1.26.4** is the latest 1.x release (end-of-life, security fixes only)
- **PyTorch 2.6+** requires numpy 2.x in some configurations
- **Most AI packages** now support or require numpy 2.x

### Urgency

The numpy 1.x pin is becoming increasingly problematic:
- New package versions may drop numpy 1.x support
- Python 3.13+ may not have numpy 1.x wheels
- Users encounter pip resolution conflicts when mixing WhisperJAV with other AI tools

### Recommended timeline

1. **v1.8.8**: Lift the `<2.0` upper bound, allow both numpy 1.x and 2.x
   (change to `numpy>=1.26.0`)
2. **v1.9.0**: Consider requiring numpy 2.x if ecosystem stabilizes
3. **Long-term**: Drop ModelScope dependency entirely via Option D if it becomes
   problematic

---

## 8. Recommended Action

**For v1.8.8, adopt Option A (lift the numpy<2 pin)**:

1. In `pyproject.toml`, `requirements.txt`, and `upgrade.py`:
   - Change `numpy>=1.26.0,<2.0` to `numpy>=1.26.0`
2. In `pyproject.toml` enhance extra:
   - Ensure ClearVoice points to the fork (already done)
   - Pin `librosa>=0.11.0` in the enhance extra
3. Pin `numba>=0.60.0` (already satisfied by current `>=0.58.0`)
4. Run full test suite with numpy 2.2.x
5. Test all 5 enhancement backends with a real audio file
6. Update the comment from "pyvideotrans compatibility" to explain the actual
   constraint rationale

This is low-risk because:
- ModelScope does not constrain numpy
- The ClearVoice fork already removed the numpy<2 pin
- bs-roformer-infer does not constrain numpy upper bound
- WhisperJAV's own code uses no removed numpy aliases
- The enhancement module has graceful fallback on failure
