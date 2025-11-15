# PIPELINE RENAME PLAN: "balanced" → "fidelity"

## Executive Summary

This document provides a comprehensive plan for renaming the "balanced" pipeline to "fidelity" throughout the WhisperJAV codebase. The rename affects 26 files across multiple subsystems including core pipelines, configuration files, user interfaces, documentation, and testing infrastructure.

**Scope**: Pipeline mode name change from "balanced" to "fidelity"
**Impact Level**: HIGH - User-facing change affecting CLI, GUI, configuration, and documentation
**Backward Compatibility**: DECISION REQUIRED - Should old "balanced" name be supported as an alias?
**Estimated Effort**: 3-4 hours for implementation + 2-3 hours for testing

---

## Impact Analysis

### 1. Core Pipeline Architecture (HIGH PRIORITY)

#### Files Requiring Changes:

**A. Pipeline Class File**
- **File**: `whisperjav/pipelines/balanced_pipeline.py`
  - **Action**: Rename to `fidelity_pipeline.py`
  - **Changes**:
    - Line 2: Update docstring "Balanced pipeline implementation" → "Fidelity pipeline implementation"
    - Line 29: Class name `BalancedPipeline` → `FidelityPipeline`
    - Line 30: Update docstring
    - Line 41: Update docstring in `__init__`
    - Line 126: Update docstring in `process` method
    - Line 375: Method `get_mode_name()` return value "balanced" → "fidelity"

**B. Pipeline Imports**
- **File**: `whisperjav/__init__.py`
  - **Line 13**: `from whisperjav.pipelines.balanced_pipeline import BalancedPipeline` → `from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline`

**C. Main Entry Point**
- **File**: `whisperjav/main.py`
  - **Line 41**: Import statement
  - **Line 112**: CLI argument choices `["balanced", "fast", "faster"]` → `["fidelity", "fast", "faster"]`
  - **Line 112**: Default value `default="balanced"` → `default="fidelity"` (DECISION: keep or change default?)
  - **Line 158**: Sensitivity argument choices (keep "balanced" for sensitivity, different from mode)
  - **Line 230**: Function signature parameter default
  - **Line 231**: Function signature parameter default
  - **Line 278**: Signature text formatting
  - **Line 396**: Pipeline selection condition `else: # balanced` → `else: # fidelity`
  - **Lines 95, 187**: Banner text and help descriptions

---

### 2. Configuration System (HIGH PRIORITY)

#### JSON Configuration Files

**A. Main Config**
- **File**: `whisperjav/config/asr_config.json`
  - **Line 29**: `pipeline_parameter_map` key `"balanced"` → `"fidelity"`
  - **Line 187**: `pipelines` section key `"balanced"` → `"fidelity"`
  - **Line 211**: UI preferences `last_mode` default value
  - **NOTE**: Sensitivity profiles still use "balanced" - these are DIFFERENT from pipeline modes

**B. Version-Specific Config**
- **File**: `whisperjav/config/asr_config_v1_5_4.json`
  - Same changes as asr_config.json

**C. Template Config**
- **File**: `whisperjav/config/config.template.json`
  - Search for "balanced" mode references and update

**D. Tuner Code**
- **File**: `whisperjav/config/transcription_tuner.py`
  - **Line 528**: `list_sensitivity_profiles()` - NOTE: Keep "balanced" here, as sensitivity is separate from mode
  - Code comments may reference the "balanced" pipeline

---

### 3. User Interfaces (HIGH PRIORITY)

#### GUI (PyWebView)

**A. HTML Interface**
- **File**: `whisperjav/webview_gui/assets/index.html`
  - **Line 140**: `<option value="balanced" selected>balanced (standard whisper)</option>` → `<option value="fidelity" selected>fidelity (standard whisper)</option>`
  - **Line 153**: Sensitivity dropdown - KEEP "balanced" (different from mode)
  - **Line 187**: Info text about modes - update description

**B. Backend API**
- **File**: `whisperjav/webview_gui/api.py`
  - **Line 129**: Default mode value `mode = options.get('mode', 'balanced')` → `mode = options.get('mode', 'fidelity')` (DECISION: keep default?)

**C. JavaScript (app.js)**
- Search for mode-related default values and validation logic

#### Documentation for GUI
- **File**: `whisperjav/webview_gui/QUICK_START.md`
  - **Line 78**: Mode description `balanced (accuracy) / fast / faster (speed)` → `fidelity (accuracy) / fast / faster (speed)`
  - **Line 134**: Recommended settings examples
  - **Line 155**: Settings examples for different scenarios
  - **Line 281**: Processing modes for accuracy

---

### 4. Documentation (MEDIUM PRIORITY)

#### User-Facing Documentation

**A. Main README**
- **File**: `README.md`
  - **Line 15**: Features description "balanced: Scene + VAD-enhanced processing"
  - **Line 23**: Quick Start example
  - **Line 95**: Banner text showing available modes
  - **Line 112**: CLI argument example `--mode balanced`
  - **Line 173**: Usage example
  - **Lines 304-308**: Processing Modes Guide table
  - **Line 313**: Content-Specific Recommendations table (multiple references)
  - **Line 347**: Sensitivity Selection section (KEEP "balanced" for sensitivity)
  - **Line 404**: Usage example
  - **Line 432**: Batch processing example
  - **Line 448**: CLI example
  - **Line 587**: Troubleshooting section - AMD GPU advice

**B. Project Instructions**
- **File**: `CLAUDE.md`
  - **Line 23**: Development command example
  - **Line 151**: Architecture section
  - **Line 95**: Banner text in code example

**C. Release Notes**
- **File**: `RELEASE_NOTES_v1.5.3.md`
  - No specific "balanced" mode references found (generic)

**D. GitHub/Copilot Instructions**
- **File**: `.github/copilot-instructions.md`
  - Search for mode references

---

### 5. Testing & Validation (MEDIUM PRIORITY)

#### Test Files

**A. Configuration Tests**
- **File**: `whisperjav/tests/test_tuner_v4_3.py`
  - Search for "balanced" in test cases
  - Update test fixtures and assertions

**B. Integration Tests**
- **File**: `tests/test_async_cancellation.py`
- **File**: `tests/test_gui_refactor.py`
  - Check for hardcoded "balanced" references

**C. Comparison Scripts**
- **File**: `scripts/compare_scene_methods.py`
  - Contains "balanced" references in documentation/examples

**D. Test Utilities**
- **File**: `scripts/test_srt_coverage.py`
- **File**: `whisperjav/webview_gui/test_api.py`

---

### 6. Build & Distribution (LOW PRIORITY)

#### Installer Components

These files don't directly reference "balanced" in code but may in documentation:

**A. Installer Scripts** (check for hardcoded defaults):
- `installer/WhisperJAV_Launcher_v1.5.1.py`
- `installer/WhisperJAV_Launcher_v1.5.3.py`
- `installer/WhisperJAV_Launcher_v1.5.4.py`
- `installer/post_install_v1.5.*.py`
- `installer/validate_installer_v1.5.*.py`

**B. README Files**:
- `installer/README.md`
- Search for example commands

**C. Batch Files**:
- Generally don't contain mode references

---

### 7. Utility Modules (LOW PRIORITY)

**A. Async Processor**
- **File**: `whisperjav/utils/async_processor.py`
  - May contain mode selection logic or defaults

**B. Configuration GUIs**
- **File**: `whisperjav/utils/config_editor_gui.py`
- **File**: `whisperjav/config/configurator_gui_ds.py`

**C. Sanitization Config**
- **File**: `whisperjav/config/sanitization_constants.py`
- **File**: `whisperjav/config/sanitization_config.py`
  - Unlikely to reference modes, but check

---

## File-by-File Change List

### Critical Path (Must Change):

1. **whisperjav/pipelines/balanced_pipeline.py** → Rename file + update 6 locations
2. **whisperjav/__init__.py** → Update import (1 location)
3. **whisperjav/main.py** → Update CLI args, imports, pipeline selection (8+ locations)
4. **whisperjav/config/asr_config.json** → Update pipeline keys (3 locations)
5. **whisperjav/webview_gui/assets/index.html** → Update dropdown + descriptions (3 locations)
6. **README.md** → Update all user-facing examples (15+ locations)
7. **CLAUDE.md** → Update developer documentation (2 locations)

### Important Secondary:

8. **whisperjav/config/asr_config_v1_5_4.json** → Same as asr_config.json
9. **whisperjav/webview_gui/api.py** → Update default mode (1 location)
10. **whisperjav/webview_gui/QUICK_START.md** → Update examples (5+ locations)
11. **whisperjav/config/config.template.json** → Update template
12. **whisperjav/tests/test_tuner_v4_3.py** → Update test cases

### Optional/Documentation:

13. **scripts/compare_scene_methods.py** → Update examples
14. **scripts/README_SRT_COVERAGE.md** → Check for references
15. **installer files** → Verify no hardcoded defaults
16. **.github/copilot-instructions.md** → Update if present

---

## Implementation Sequence

### Phase 1: Core Infrastructure (CRITICAL - DO FIRST)
**Objective**: Update pipeline class and configuration system

1. **Rename pipeline file**:
   ```bash
   git mv whisperjav/pipelines/balanced_pipeline.py whisperjav/pipelines/fidelity_pipeline.py
   ```

2. **Update fidelity_pipeline.py**:
   - Class name: `BalancedPipeline` → `FidelityPipeline`
   - Docstrings: Replace "balanced" references
   - `get_mode_name()`: Return "fidelity"

3. **Update configuration files**:
   - `whisperjav/config/asr_config.json`: Rename pipeline key
   - `whisperjav/config/asr_config_v1_5_4.json`: Same changes
   - `whisperjav/config/config.template.json`: Update template

4. **Verify configuration loads**:
   ```bash
   python -c "from whisperjav.config.transcription_tuner import TranscriptionTuner; t = TranscriptionTuner(); print(t.list_pipelines())"
   # Should show: ['faster', 'fast', 'fidelity']
   ```

**Testing**: Run config validation
```bash
python -c "from whisperjav.config.transcription_tuner import TranscriptionTuner; t = TranscriptionTuner(); t.validate_configuration()"
```

---

### Phase 2: Entry Points (CRITICAL - DO SECOND)
**Objective**: Update CLI and imports

1. **Update whisperjav/__init__.py**:
   - Import: `from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline`

2. **Update whisperjav/main.py**:
   - CLI argument choices
   - Import statement
   - Pipeline selection logic (line 396)
   - Help text and banner
   - Signature formatting function

3. **Test CLI argument parsing**:
   ```bash
   python -m whisperjav.main --help
   # Verify: --mode {fidelity,fast,faster}

   python -m whisperjav.main --check
   # Verify: No import errors
   ```

**Testing**: Dry run with `--help` and `--check` flags

---

### Phase 3: User Interfaces (HIGH PRIORITY)
**Objective**: Update GUI and ensure consistency

1. **Update GUI HTML** (`whisperjav/webview_gui/assets/index.html`):
   - Mode dropdown options
   - Description text
   - Info labels

2. **Update GUI API** (`whisperjav/webview_gui/api.py`):
   - Default mode value (if changing)

3. **Update GUI JavaScript** (`whisperjav/webview_gui/assets/app.js`):
   - Search for "balanced" in validation/defaults

4. **Test GUI**:
   ```bash
   python -m whisperjav.webview_gui.main
   # Verify: Dropdown shows "fidelity"
   # Verify: Mode selection works
   ```

**Testing**: Launch GUI, verify all dropdowns and modes

---

### Phase 4: Documentation (MEDIUM PRIORITY)
**Objective**: Update user-facing documentation

1. **Update README.md**:
   - Search and replace "balanced" → "fidelity" (carefully, preserve sensitivity references)
   - Update mode tables
   - Update example commands
   - Update recommendations

2. **Update CLAUDE.md**:
   - Update architecture section
   - Update example commands

3. **Update GUI documentation**:
   - `whisperjav/webview_gui/QUICK_START.md`
   - `whisperjav/webview_gui/API_REFERENCE.md`

4. **Review and update**:
   - `RELEASE_NOTES_v1.5.3.md` (if needed for next version)
   - `installer/README.md`

**Testing**: Read through documentation for consistency

---

### Phase 5: Testing & Validation (MEDIUM PRIORITY)
**Objective**: Update tests and verify functionality

1. **Update test files**:
   - `whisperjav/tests/test_tuner_v4_3.py`
   - Other test files with "balanced" references

2. **Run full test suite**:
   ```bash
   python -m pytest tests/ -v
   python -m pytest whisperjav/tests/ -v
   ```

3. **Integration testing**:
   ```bash
   # Test with actual file
   whisperjav test_video.mp4 --mode fidelity --sensitivity balanced

   # Verify output
   ls output/*.srt
   ```

**Testing**: Complete test suite + manual integration test

---

### Phase 6: Optional Updates (LOW PRIORITY)
**Objective**: Clean up scripts and utilities

1. **Update scripts**:
   - `scripts/compare_scene_methods.py`
   - `scripts/README_SRT_COVERAGE.md`

2. **Update installer files** (if any hardcoded defaults exist)

3. **Update utility GUIs** (if they reference modes)

**Testing**: Spot check utilities

---

## Testing Strategy

### Unit Testing (After Each Phase)

**Phase 1 Tests**:
```python
# Test config loads correctly
from whisperjav.config.transcription_tuner import TranscriptionTuner
t = TranscriptionTuner()
assert 'fidelity' in t.list_pipelines()
assert 'balanced' not in t.list_pipelines()  # Unless backward compat

# Test pipeline instantiation
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
pipeline = FidelityPipeline(output_dir=".", temp_dir=".", keep_temp_files=False,
                            subs_language="native", resolved_config={})
assert pipeline.get_mode_name() == "fidelity"
```

**Phase 2 Tests**:
```bash
# Test CLI parsing
python -m whisperjav.main --mode fidelity --help
python -m whisperjav.main --check

# Test imports
python -c "from whisperjav import FidelityPipeline; print('OK')"
```

**Phase 3 Tests**:
```bash
# Launch GUI and verify:
# 1. Mode dropdown shows "fidelity"
# 2. Can select fidelity mode
# 3. Can start process with fidelity mode
python -m whisperjav.webview_gui.main
```

### Integration Testing (After Phase 5)

**Test Scenarios**:

1. **CLI Transcription**:
   ```bash
   whisperjav test.mp4 --mode fidelity --sensitivity aggressive
   # Verify: SRT file created
   # Verify: Metadata shows mode="fidelity"
   ```

2. **GUI Transcription**:
   - Select fidelity mode in GUI
   - Process test file
   - Verify console output
   - Check output SRT file

3. **Configuration Loading**:
   ```bash
   # Test custom config
   whisperjav test.mp4 --mode fidelity --config custom_config.json
   ```

4. **Async Processing**:
   ```bash
   whisperjav *.mp4 --mode fidelity --async-processing
   ```

### Regression Testing

**Critical Workflows to Verify**:

1. All three modes work (fidelity, fast, faster)
2. Sensitivity profiles work (conservative, balanced, aggressive)
3. Language selection works (Japanese, Korean, Chinese, English)
4. Translation integration works
5. GUI file selection and processing works
6. Batch processing works
7. Progress tracking works
8. Metadata generation includes correct mode name

---

## Risk Assessment

### High-Risk Areas

**1. Configuration Key Mismatch**
- **Risk**: Old configs with "balanced" key fail to load
- **Mitigation**:
  - Add config migration logic to handle old "balanced" key
  - OR: Support both "balanced" and "fidelity" keys temporarily
  - Add validation error messages

**2. User Confusion**
- **Risk**: Existing users don't understand name change
- **Mitigation**:
  - Add deprecation notice in release notes
  - Show friendly error if "balanced" used in config
  - Update all documentation simultaneously

**3. Third-Party Scripts**
- **Risk**: User scripts break due to mode name change
- **Mitigation**:
  - Consider supporting "balanced" as alias
  - Provide migration guide
  - Version bump to indicate breaking change

**4. GUI State Persistence**
- **Risk**: GUI saved settings have "balanced" mode
- **Mitigation**:
  - Add migration logic in GUI initialization
  - Fallback to "fidelity" if "balanced" found

### Medium-Risk Areas

**1. Test Failures**
- **Risk**: Tests hardcoded with "balanced" fail
- **Mitigation**: Comprehensive test updates in Phase 5

**2. Documentation Inconsistency**
- **Risk**: Some docs not updated, causing confusion
- **Mitigation**: Comprehensive grep and systematic updates

**3. Installer Defaults**
- **Risk**: Installers hardcode "balanced" mode
- **Mitigation**: Review all installer scripts

### Low-Risk Areas

**1. Backward Compatibility**
- **Risk**: Old metadata files reference "balanced"
- **Impact**: Minimal - metadata is informational only
- **Mitigation**: Note in migration guide

**2. Log Messages**
- **Risk**: Some log messages still say "balanced"
- **Impact**: Low - logs are for debugging
- **Mitigation**: Search codebase for string literals

---

## Rollback Plan

### If Critical Issues Arise

**Emergency Rollback Steps**:

1. **Revert Git Commits**:
   ```bash
   git log --oneline  # Find pre-rename commit
   git revert <commit-hash>..HEAD
   # OR
   git reset --hard <commit-hash>
   ```

2. **Restore Pipeline File**:
   ```bash
   git mv whisperjav/pipelines/fidelity_pipeline.py whisperjav/pipelines/balanced_pipeline.py
   ```

3. **Restore Configuration**:
   - Revert asr_config.json changes
   - Restore import statements

4. **Notify Users**:
   - GitHub release notes
   - Deprecation of rename attempt

### Partial Rollback (If Some Changes Work)

**Option: Support Both Names**:

Add alias support in multiple locations:

```python
# In main.py
PIPELINE_ALIASES = {
    'balanced': 'fidelity',  # Backward compatibility
}

mode = args.mode
if mode in PIPELINE_ALIASES:
    logger.warning(f"Mode '{mode}' is deprecated, using '{PIPELINE_ALIASES[mode]}'")
    mode = PIPELINE_ALIASES[mode]
```

```json
// In asr_config.json - duplicate pipeline config
{
  "pipelines": {
    "fidelity": { ... },
    "balanced": { ... }  // Deprecated, same as fidelity
  }
}
```

---

## Backward Compatibility Recommendations

### Option A: Breaking Change (Clean Rename)
**Pros**:
- Simpler codebase
- No technical debt
- Clear new naming

**Cons**:
- Breaks existing scripts
- User confusion
- Config migration needed

**Recommended for**: Major version bump (v2.0.0)

### Option B: Graceful Migration (Support Both)
**Pros**:
- No breaking changes
- Smooth user transition
- Old configs still work

**Cons**:
- More complex code
- Temporary technical debt
- Eventual cleanup needed

**Recommended for**: Minor version bump (v1.6.0)

**Implementation**:
1. Support both "balanced" and "fidelity" in CLI/config
2. Show deprecation warning when "balanced" used
3. Update all documentation to use "fidelity"
4. Plan removal of "balanced" alias in v2.0.0

---

## Questions for User Decision

### 1. Backward Compatibility
**Question**: Should "balanced" be supported as a deprecated alias?

**Option A**: Yes, support both for 1-2 versions with deprecation warnings
**Option B**: No, clean break, require migration

**Recommendation**: Option A for user-friendliness

### 2. Default Mode
**Question**: Should the default mode remain the renamed pipeline?

**Current**: `--mode balanced` (default)
**Proposed**: `--mode fidelity` (default)

**Recommendation**: Keep default as "fidelity" (maintains same behavior)

### 3. GUI Default Selection
**Question**: Should GUI remember user's last selected mode?

**Current**: GUI remembers via `ui_preferences.last_mode`
**Issue**: Saved preference might be "balanced"

**Recommendation**: Add migration logic to convert "balanced" → "fidelity" in saved preferences

### 4. Version Bump
**Question**: What version bump should accompany this change?

**Options**:
- Patch (v1.5.4): If backward compatible
- Minor (v1.6.0): If graceful migration
- Major (v2.0.0): If breaking change

**Recommendation**: Minor (v1.6.0) with backward compatibility

### 5. Migration Guide
**Question**: Should we provide a migration guide for users?

**Recommendation**: Yes, include in release notes:
- What changed and why
- How to update scripts/configs
- Timeline for deprecation (if applicable)

---

## Summary Statistics

### Files to Modify: 26 files

**By Category**:
- Core Pipeline: 3 files
- Configuration: 4 files
- User Interfaces: 5 files
- Documentation: 8 files
- Tests: 4 files
- Installer: 2 files (check only)

**By Priority**:
- Critical: 7 files (Phases 1-2)
- High: 5 files (Phase 3)
- Medium: 8 files (Phases 4-5)
- Low: 6 files (Phase 6)

### Estimated Effort

**Development**: 3-4 hours
- Phase 1: 45 min
- Phase 2: 30 min
- Phase 3: 60 min
- Phase 4: 60 min
- Phase 5: 30 min
- Phase 6: 30 min

**Testing**: 2-3 hours
- Unit tests: 30 min
- Integration tests: 60 min
- GUI testing: 30 min
- Regression testing: 60 min

**Total**: 5-7 hours

---

## Appendix: Search Commands

### Finding All References

```bash
# Case-sensitive search for mode references
grep -r "balanced" --include="*.py" --include="*.json" --include="*.md" --include="*.html" --include="*.js"

# Find import statements
grep -r "balanced_pipeline" --include="*.py"

# Find class references
grep -r "BalancedPipeline" --include="*.py"

# Find string literals in Python
grep -r '"balanced"' --include="*.py"
grep -r "'balanced'" --include="*.py"

# Find JSON keys
grep -r '"balanced":' --include="*.json"

# Find HTML option values
grep -r 'value="balanced"' --include="*.html"
```

### Verification Commands

```bash
# After rename, verify no old references (except sensitivity)
grep -r "balanced" --include="*.py" | grep -v "sensitivity" | grep -v "#.*balanced"

# Verify new references exist
grep -r "fidelity" --include="*.py" --include="*.json"

# Test imports
python -c "from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline"

# Test CLI
python -m whisperjav.main --help | grep fidelity
```

---

## Conclusion

This rename operation is feasible but requires careful, systematic execution across 26 files. The highest risk areas are configuration key mismatches and user confusion.

**Recommended Approach**:
1. Implement with backward compatibility (support both names)
2. Add deprecation warnings
3. Comprehensive testing before release
4. Clear migration documentation
5. Plan "balanced" alias removal for v2.0.0

**Critical Success Factors**:
- Execute phases in sequence
- Test after each phase
- Update all documentation simultaneously
- Provide clear migration path for users
- Consider semantic versioning for release

---

**Report Generated**: 2025-11-15
**Codebase Version**: WhisperJAV v1.5.4 (main branch)
**Analysis Depth**: Comprehensive (26 files analyzed)
