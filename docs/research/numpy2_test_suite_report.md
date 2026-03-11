# NumPy 2.0 Compatibility Report for WhisperJAV

**Date**: 2026-03-11
**Current numpy constraint**: `numpy>=1.26.0,<2.0` (pyproject.toml, cli extra)
**Reason for pin**: "pyvideotrans compatibility" (legacy note, pyvideotrans not imported in codebase)

---

## Executive Summary

WhisperJAV has **low direct risk** from numpy 2 breaking changes in its own code. Only **1 file** uses a removed numpy 2 API (`np.int_` in `metadata_manager.py`). The primary migration risk comes from **dependencies** -- specifically numba, ModelScope, and the openai-whisper/stable-ts ecosystem.

**Recommended action**: Bump minimum dependency versions, fix the 1 direct code issue, then test with numpy 2 on a branch.

---

## 1. Numpy Usage Across WhisperJAV

### Files using numpy (37 files total)

| Component | Files | Primary numpy APIs used |
|-----------|-------|------------------------|
| ASR engines | 3 | `np.ndarray`, `np.mean`, `np.float32` |
| Scene detection | 6 | `np.ndarray`, `np.int16`, `np.float32`, `np.log10`, `np.abs`, `np.clip`, `np.ceil`, `np.array` |
| Speech enhancement | 7 | `np.ndarray`, `np.float32`, `np.zeros`, `np.ones`, `np.linspace`, `np.maximum`, `np.mean`, `np.gcd`, `np.frombuffer`, `np.stack` |
| Speech segmentation | 7 | `np.ndarray`, `np.float32`, `np.int16`, `np.mean`, `np.clip`, `np.arange`, `np.linspace` |
| Subtitle pipeline | 6 | `np.ndarray`, `np.float32`, `np.mean` |
| Utilities | 3 | `np.integer`, `np.int_`, `np.bool_`, `np.floating`, `np.isnan`, `np.isinf`, `np.ndarray` |
| Vendor | 1 | `np.ceil`, `np.mean`, `np.std`, `np.vstack`, `np.hstack`, `np.pad`, `np.zeros`, `np.searchsorted`, `np.argmin`, `np.any`, `np.log10`, `np.percentile`, `np.arange` |
| Tests | 3 | `np.mean`, `np.abs`, `np.sum`, `np.min`, `np.max`, `np.std`, `np.median`, `np.histogram`, `np.array`, `np.linspace`, `np.sin`, `np.pi`, `np.corrcoef` |
| Pipelines | 1 | `np.ndarray` (lazy import, type annotation only) |

### Notable patterns

- **Type annotations**: Most files use `np.ndarray` for type hints -- safe in numpy 2.
- **Audio conversion**: Heavy use of `.astype(np.float32)` and `.astype(np.int16)` -- safe in numpy 2.
- **Statistics**: `np.mean`, `np.max`, `np.min`, `np.std` -- safe in numpy 2.
- **Array creation**: `np.zeros`, `np.ones`, `np.array`, `np.arange`, `np.linspace` -- safe in numpy 2.
- **Unused import**: `repetition_cleaner.py` imports numpy but has no `np.` usage.

---

## 2. Incompatible Patterns Found

### HIGH severity (will break)

| File | Line | Pattern | Replacement |
|------|------|---------|-------------|
| `whisperjav/utils/metadata_manager.py` | 13 | `np.int_` | `np.intp` |

**Details**: `metadata_manager.py` line 13 uses `np.int_` in an `isinstance()` check for JSON serialization. In numpy 2, `np.int_` was removed (it was an alias for Python's `int` which caused confusion). The fix is to replace `np.int_` with `np.intp` or simply remove it from the isinstance check since `np.integer` (also on that line) already covers all numpy integer types.

### MEDIUM/LOW severity (informational)

| File | Line | Pattern | Status |
|------|------|---------|--------|
| `whisperjav/utils/metadata_manager.py` | 23 | `np.bool_` | **SAFE** -- `np.bool_` is kept in numpy 2 |

### No issues found for:

- Removed functions (`np.product`, `np.sometrue`, `np.alltrue`, etc.) -- none used
- Removed constants (`np.PINF`, `np.NINF`, etc.) -- none used
- Moved exceptions (`np.AxisError`, etc.) -- none used
- `numpy.distutils` -- not used
- `np.array(..., copy=False)` -- not used
- Removed type aliases (`np.float_`, `np.complex_`, `np.object_`, `np.str_`) -- none used

---

## 3. Dependency Compatibility Matrix

### Critical path dependencies (must support numpy 2 for any WhisperJAV pipeline to work)

| Package | Numpy 2 Status | Min Version for numpy 2 | WhisperJAV Constraint | Risk | Action |
|---------|---------------|------------------------|----------------------|------|--------|
| **numpy** | N/A | 2.0.0 | `>=1.26.0,<2.0` | -- | Bump upper bound |
| **torch** | Yes | 2.3.0 | not pinned | LOW | No action |
| **scipy** | Yes | 1.12.0 | `>=1.10.1` | LOW | Bump to `>=1.12.0` |
| **numba** | Yes | 0.59.0 | `>=0.58.0` | **HIGH** | Bump to `>=0.59.0` |
| **librosa** | Yes | 0.10.2 | `>=0.10.0` | LOW | Bump to `>=0.10.2` |
| **soundfile** | Yes | any | not pinned | LOW | No action |
| **faster-whisper** | Yes | 1.1.0 | `>=1.1.0` | LOW | No action |

### Enhancement dependencies

| Package | Numpy 2 Status | Risk | Notes |
|---------|---------------|------|-------|
| **modelscope** | Partial | **HIGH** | Complex numpy usage, ZipEnhancer depends on it |
| **clearvoice** | Unknown | MEDIUM | Custom fork, needs testing |
| **opencv-python** | Yes (4.9+) | MEDIUM | WhisperJAV pins `>=4.0`, need `>=4.9.0` |
| **scikit-learn** | Yes (1.4+) | LOW | WhisperJAV pins `>=1.3.0`, need `>=1.4.0` |
| **einops** | Yes (0.7+) | LOW | No action needed |

### No numpy dependency (safe)

pysubtrans, openai, google-genai, pysrt, srt, tqdm, colorama, requests, pydantic, PyYAML, pydub, pywebview, auditok (minimal usage)

---

## 4. Risk Assessment by Component

| Component | Direct Code Risk | Dependency Risk | Overall Risk | Notes |
|-----------|-----------------|-----------------|--------------|-------|
| **Balanced pipeline** | LOW | MEDIUM (numba, librosa) | MEDIUM | numba version bump needed |
| **Faster pipeline** | NONE | LOW (faster-whisper) | LOW | Clean |
| **Fast pipeline** | LOW | MEDIUM (openai-whisper, stable-ts) | MEDIUM | Depends on whisper fork |
| **Scene detection** | NONE | LOW | LOW | Uses only safe numpy APIs |
| **Speech enhancement (ffmpeg-dsp)** | NONE | NONE | NONE | No risky deps |
| **Speech enhancement (zipenhancer)** | NONE | **HIGH** (modelscope) | **HIGH** | ModelScope is the blocker |
| **Speech enhancement (clearvoice)** | NONE | MEDIUM | MEDIUM | Needs testing |
| **Speech enhancement (bs-roformer)** | NONE | MEDIUM | MEDIUM | Needs testing |
| **Speech segmentation (silero)** | NONE | LOW | LOW | Clean |
| **Speech segmentation (ten)** | NONE | MEDIUM (ten-vad unknown) | MEDIUM | Needs testing |
| **Translation module** | NONE | NONE | NONE | No numpy deps |
| **GUI** | NONE | NONE | NONE | No numpy deps |
| **Metadata/utilities** | **HIGH** (`np.int_`) | NONE | **HIGH** | 1-line fix |

---

## 5. Recommended Migration Order

### Phase 1: Zero-risk fixes (can do immediately)

1. **Fix `metadata_manager.py`**: Remove `np.int_` from the isinstance check on line 13. `np.integer` already covers all numpy integer types.
   ```python
   # Before (line 13):
   if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, ...)):
   # After:
   if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, ...)):
   ```

2. **Remove unused numpy import** from `repetition_cleaner.py`.

### Phase 2: Bump dependency minimums

Update `pyproject.toml` and `installer/core/registry.py`:
```
numpy>=1.26.0  (keep lower bound for now, remove <2.0 upper bound later)
scipy>=1.12.0  (from >=1.10.1)
numba>=0.59.0  (from >=0.58.0)
librosa>=0.10.2  (from >=0.10.0)
scikit-learn>=1.4.0  (from >=1.3.0)
opencv-python>=4.9.0  (from >=4.0)
```

### Phase 3: Test with numpy 2 on a branch

1. Create branch `feature/numpy2-compat`
2. Change constraint to `numpy>=2.0.0,<3.0`
3. Run full test suite
4. Test each pipeline mode (faster, fast, balanced)
5. Test each enhancement backend
6. Focus testing on ModelScope/ZipEnhancer (highest risk)

### Phase 4: Decide on dual support vs numpy 2 only

Option A: `numpy>=1.26.0` (no upper bound) -- supports both numpy 1.x and 2.x
Option B: `numpy>=2.0.0` -- numpy 2 only, cleaner but breaks existing installs

Recommendation: **Option A** with tested minimum versions for all dependencies.

---

## 6. Effort Estimate

| Task | Effort | Blocking? |
|------|--------|-----------|
| Fix `np.int_` in metadata_manager.py | 5 min | No |
| Remove unused numpy import in repetition_cleaner.py | 1 min | No |
| Bump dependency versions in pyproject.toml | 15 min | No |
| Bump dependency versions in registry.py | 15 min | No |
| Test core pipelines with numpy 2 | 2-4 hours | No |
| Test ModelScope/ZipEnhancer with numpy 2 | 2-4 hours | **Yes** -- highest risk |
| Test ClearVoice with numpy 2 | 1-2 hours | No |
| Test BS-RoFormer with numpy 2 | 1-2 hours | No |
| Update installer build scripts | 30 min | No |
| Regression testing full pipeline | 4-8 hours | No |

**Total estimated effort**: 1-2 days of focused testing

**Key blocker**: ModelScope numpy 2 compatibility. If ModelScope does not work with numpy 2, the ZipEnhancer backend must be isolated or ModelScope must be upgraded. This is the single largest risk factor.

---

## 7. Test Suite

The following test files were created to support ongoing numpy 2 compatibility validation:

- **`tests/research/test_numpy2_patterns.py`** -- Static regex scan for all known numpy 2 removed APIs
- **`tests/research/test_numpy2_imports.py`** -- AST-based analysis of numpy attribute usage per module category
- **`tests/research/test_numpy2_deps.py`** -- Dependency version constraint analysis and compatibility matrix

Run with:
```bash
python -m pytest tests/research/ -v
```

These tests require only the source code and standard library -- no GPU, models, or heavy dependencies needed.

---

## 8. pyvideotrans Compatibility Note

The current `numpy<2.0` pin cites "pyvideotrans compatibility" as the reason. pyvideotrans exists as an **optional** `[compatibility]` extra in pyproject.toml, not a core dependency. The numpy pin is in the `[cli]` extra, which means it constrains numpy for ALL CLI users, not just those using the pyvideotrans compatibility layer.

This coupling is questionable: the numpy constraint for all users is driven by an optional compatibility extra that most users do not install. If pyvideotrans compatibility is truly the only reason for the pin, it should be moved to the `[compatibility]` extra's own dependency list, not the `[cli]` extra.

More practically, the actual blocking dependencies for numpy 2 are:
1. **numba** ABI compatibility (need >=0.59.0)
2. **ModelScope** (uncertain numpy 2 support)
3. **openai-whisper** (main branch likely supports numpy 2, but unverified)
