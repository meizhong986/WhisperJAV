# WhisperJAV-GUI: Tabbed Interface Implementation Plan

**Document Status:** Living Document
**Created:** 2025-10-29
**Last Updated:** 2025-10-29
**Current Phase:** Step 1 - Initial Tab Structure

---

## Executive Summary

This document tracks the phased implementation of a tabbed UI architecture for WhisperJAV-GUI, integrating the whisperjav-translate module and preparing for future whisperjav-ensemble functionality.

### Design Philosophy

- **Progressive Disclosure**: Users see only relevant options per workflow stage
- **Modular Architecture**: Each tab maps to a distinct backend component
- **Defensive Implementation**: Phased approach with validation gates
- **Zero Regression**: Each step preserves existing functionality

---

## Phase Overview

| Phase | Description | Status | Date |
|-------|-------------|--------|------|
| Step 1 | Initial tab structure (2 tabs) | ✅ Implemented - Awaiting User Testing | 2025-10-29 |
| Step 2 | Design AI Translation Options tab | ⏳ Pending | TBD |
| Step 3 | Implement AI Translation integration | ⏳ Pending | TBD |
| Step 4 | Design Ensemble Options placeholder | ⏳ Pending | TBD |

---

## Step 1: Initial Tab Structure

### Objective

Refactor existing single-page GUI into tabbed interface with **two tabs**:
1. **Transcription Mode** - Basic transcription options
2. **Transcription Adv. Options** - Power-user features

**Critical Constraint:** Functionally equivalent to existing GUI (zero feature changes)

### Current UI Structure Analysis

**File:** `whisperjav/gui/whisperjav_gui.py`

**Layout (Grid Rows):**
- Row 0: **Source** - File/folder selection listbox
- Row 1: **Destination** - Output directory
- Row 2: **Processing profile** - Mode, sensitivity, output language
- Row 3: **Advanced options** - Collapsible section with toggle button
- Row 4: **Run controls** - Progress bar, Start/Cancel buttons
- Row 5: **Console/Log** - Output text area

### Proposed New Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Source (Always visible)                                    │
│ [File listbox with Add/Remove/Clear buttons]              │
├─────────────────────────────────────────────────────────────┤
│ Destination (Always visible)                               │
│ Output: [path entry] [Browse] [Open]                      │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────┬──────────────────────────────────┐ │
│ │ Transcription Mode  │ Transcription Adv. Options       │ │ ← Tabs
│ └─────────────────────┴──────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │         SWITCHABLE TAB CONTENT AREA                    │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ [Progress Bar═══════════════════════] Idle                │
│ [Start] [Cancel]                                           │
├─────────────────────────────────────────────────────────────┤
│ Console                                                    │
│ Ready.                                                     │
│ _                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Tab 1: "Transcription Mode" Content

**Migrated from:** Current Row 2 "Processing profile" LabelFrame

**UI Elements:**
- **Mode:** ○ balanced ○ fast ○ faster (Radio buttons)
- **Sensitivity:** [dropdown: conservative/balanced/aggressive]
- **Output language:** [dropdown: japanese/english-direct]
- **Info labels:**
  - "Speed vs. Accuracy: 'fast' and 'faster' prioritize throughput..."
  - "Details vs. Noise: 'conservative' reduces false positives..."

**Variables (unchanged):**
- `self.mode_var` (StringVar, default="balanced")
- `self.sens_var` (StringVar, default="balanced")
- `self.lang_var` (StringVar, default="japanese")

### Tab 2: "Transcription Adv. Options" Content

**Migrated from:** Current Row 3 "Advanced options" LabelFrame (collapsible)

**UI Elements:**

**Row 1 (4 columns):**
1. ☐ Adaptive classification (WIP) - disabled
2. ☐ Adaptive audio enhancements (WIP) - disabled
3. ☐ Smart postprocessing (WIP) - disabled
4. Verbosity: [dropdown: quiet/summary/normal/verbose]

**Row 2 (4 columns):**
1. ☐ Model override (enables dropdown below)
2. Model selection: [dropdown: large-v3/large-v2/turbo] - initially disabled
3. ☐ Async processing
4. Max workers: [spinner: 1-16]

**Row 3 (4 columns):**
1. Opening credit: [text entry spanning 2 columns]
   - Label: "Opening credit (Example: Produced by XXX):"
2. ☐ Keep temp files
3. Temp dir: [text entry] [Browse button]

**Variables (unchanged):**
- `self.opt_adapt_cls`, `self.opt_adapt_enh`, `self.opt_smart_post` (BooleanVar)
- `self.verbosity_var` (StringVar, default="summary")
- `self.model_override_enabled` (BooleanVar, default=False)
- `self.model_selection_var` (StringVar, default="large-v3")
- `self.async_var` (BooleanVar, default=False)
- `self.workers_var` (IntVar, default=1)
- `self.credit_var` (StringVar, default="")
- `self.keep_temp_var` (BooleanVar, default=False)
- `self.temp_var` (StringVar, default="")

---

## Implementation Steps

### 1.1 Create ttk.Notebook Widget

**Location:** Replace current row 3 (collapsible Advanced header)

**Code changes:**
```python
# Remove old code (lines ~157-162):
# adv_header = ttk.Frame(frm)
# self.adv_open = tk.BooleanVar(value=False)
# self._adv_btn = ttk.Button(adv_header, text="Show advanced ▸", ...)

# Add new code at row 2:
self.notebook = ttk.Notebook(frm)
self.notebook.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
```

### 1.2 Create Tab 1 Frame: "Transcription Mode"

**Steps:**
1. Create `ttk.Frame` as child of `self.notebook`
2. Move "Processing profile" content into this frame
3. Preserve grid layout and widget structure
4. Add frame to notebook: `self.notebook.add(tab1, text="Transcription Mode")`

### 1.3 Create Tab 2 Frame: "Transcription Adv. Options"

**Steps:**
1. Create `ttk.Frame` as child of `self.notebook`
2. Move "Advanced options" content into this frame
3. Preserve 4-column grid layout
4. Add frame to notebook: `self.notebook.add(tab2, text="Transcription Adv. Options")`

### 1.4 Update Grid Row Indices

**Changes:**
- Row 0: Source (unchanged)
- Row 1: Destination (unchanged)
- Row 2: Notebook (NEW - replaces old rows 2-3)
- Row 3: Run controls (previously row 4)
- Row 4: Console/Log (previously row 5)

**Update rowconfigure:**
```python
frm.rowconfigure(4, weight=1)  # Console expands (was row 5)
```

### 1.5 Clean Up Obsolete Code

**Remove:**
1. `self.adv_open = tk.BooleanVar(value=False)` (line ~160)
2. `self._adv_btn = ttk.Button(...)` (line ~161-162)
3. `def toggle_advanced(self):` method (lines ~439-449)
4. Any references to `adv_header.grid()`
5. Dynamic `self.adv.grid()` / `self.adv.grid_remove()` calls

**Keep:**
- All widget variables (StringVar, BooleanVar, IntVar)
- `_toggle_model_override()` method (still needed)
- `_browse_temp_dir()` method (still needed)

### 1.6 Verify build_args() Unchanged

**Validation:**
- No changes required to `build_args()` method
- All variables (`self.mode_var`, etc.) remain accessible
- CLI arguments generated must be identical

### 1.7 Testing Checklist

**Visual Validation:**
- [ ] Two tabs visible: "Transcription Mode", "Transcription Adv. Options"
- [ ] Tab 1 shows mode, sensitivity, output language
- [ ] Tab 2 shows all advanced options (3 rows × 4 columns layout)
- [ ] Source/Destination sections visible above tabs
- [ ] Progress bar and buttons visible below tabs
- [ ] Console log visible at bottom

**Functional Validation:**
- [ ] Switching between tabs works smoothly
- [ ] All widgets interactive and respond correctly
- [ ] Model override checkbox enables/disables dropdown
- [ ] Temp dir Browse button opens file dialog
- [ ] Start button builds identical CLI arguments as before
- [ ] Progress bar animates during execution
- [ ] Console log displays output correctly

**Regression Testing:**
- [ ] Run transcription with default settings
- [ ] Run with custom mode (fast/faster)
- [ ] Run with aggressive sensitivity
- [ ] Enable model override and run with large-v2
- [ ] Enable async processing with max workers > 1
- [ ] Specify custom temp directory
- [ ] Add opening credit text
- [ ] Compare CLI arguments before/after refactoring

---

## Code Implementation Status

### Modified Files

| File | Status | Lines Changed | Notes |
|------|--------|---------------|-------|
| `whisperjav/gui/whisperjav_gui.py` | ✅ Completed | ~120 | Tab refactoring complete |

### Implementation Summary (Step 1)

**Date:** 2025-10-29

**Changes Made:**
1. ✅ Created `ttk.Notebook` widget at row 2
2. ✅ Created Tab 1 "Transcription Mode" with basic options
3. ✅ Created Tab 2 "Transcription Adv. Options" with power-user features
4. ✅ Updated grid row indices (Run controls: row 3, Console: row 4)
5. ✅ Removed `toggle_advanced()` method and collapsible UI code
6. ✅ Preserved all widget variables unchanged
7. ✅ Module imports successfully (no syntax errors)
8. ✅ **Enhanced tab styling for discoverability** (see UI Enhancement section below)

**Variables Preserved:**
- `self.mode_var`, `self.sens_var`, `self.lang_var`
- `self.opt_adapt_cls`, `self.opt_adapt_enh`, `self.opt_smart_post`
- `self.verbosity_var`, `self.model_override_enabled`, `self.model_selection_var`
- `self.async_var`, `self.workers_var`, `self.credit_var`
- `self.keep_temp_var`, `self.temp_var`

**Methods Preserved:**
- `build_args()` - unchanged, generates identical CLI arguments
- `_toggle_model_override()` - still functional
- `_browse_temp_dir()` - still functional

**New Method Added:**
- `_setup_notebook_style()` - configures enhanced tab styling for better discoverability

### UI Enhancement: File Folder Tab Design

**User Feedback Addressed:**
- "Tabs not prominent enough to be discovered easily"
- "Not intuitive to know which tab is active"
- Need for better visual hierarchy and user-friendliness
- Previous design was "disappointingly worse" - needed complete redesign

**Solution Implemented: File Folder Metaphor Design**

Based on user-provided specification and external developer input, implementing a physical file folder tab design where the active tab appears to "lift" the content page.

**Design Philosophy:**
- Mimics physical file folder tabs
- Active tab appears to sit "in front" and be seamlessly connected to content
- Active tab "breaks through" the tab bar's bottom border line
- Creates illusion of depth and layering

**Color Palette (User-Specified Final Design):**

| Element | Color | Hex Code | Description |
|---------|-------|----------|-------------|
| Tab Bar Background | white | #FFFFFF | **White** horizontal bar |
| Content Area | light gray | #F0F0F0 | Light gray content background |
| Active Tab BG | light gray | #F0F0F0 | **Matches content area** |
| Inactive Tab BG | very light gray | #FAFAFA | **Lighter than active** tab |
| Active Text | sky-700 | #0369A1 | Professional blue, **bold** |
| Inactive Text | gray-600 | #4B5563 | Medium-dark gray |
| Borders | light gray | #D1D5DB | Subtle light gray borders |
| Hover State | medium-light gray | #F5F5F5 | Between inactive and active |
| Hover Text | gray-800 | #1F2937 | Darker text on hover |

**Tab Bar (Foundation):**
- Background: **White (#FFFFFF)** - clean, minimal
- Bottom border: 1px solid light gray (#D1D5DB)
- Small 1-2px gaps between tabs (tab margins)

**Inactive Tab State (Background Tabs):**
- **Shape:** Rounded top corners (~8px radius), sharp bottom corners
- **Size:** Shorter/smaller than active tab
  - Padding: 18px horizontal, 8px vertical
- **Background:** Very light gray (#FAFAFA) - **lighter than active tab**
- **Text:**
  - Color: Medium-dark gray (#4B5563)
  - Font: 10pt, normal weight
- **Borders:** 1px light gray (#D1D5DB)
- **Position:** Flush with tab bar's bottom border
- **Hover Effect:**
  - Background → medium-light gray (#F5F5F5)
  - Text → darker gray (#1F2937)

**Active Tab State (Foreground Tab - CRITICAL):**
- **Shape:** Same rounded top corners
- **Size:** Taller than inactive tabs
  - Padding: 18px horizontal, **12px top, 10px bottom** (creates "lift")
  - ~50% taller vertically than inactive tabs
- **Background:** Light gray (#F0F0F0) - **matches content area below**
- **Text:**
  - Color: Sky blue (#0369A1)
  - Font: 11pt, **bold** weight
- **Borders:**
  - Top & sides: 2px visible in light gray (#D1D5DB)
  - Bottom: Matches content background (seamless)
- **Visual Effect:** Bottom edge overlaps/hides tab bar border
- **Z-index:** Higher layer to sit on top
- **Result:** Seamless visual connection to light gray content area below

**Content Area:**
- Background: Light gray (#F0F0F0) - **matches active tab exactly**
- Appears as ONE continuous surface with active tab
- Creates the illusion that the active tab is "lifting" this page

**Key Visual Effects Achieved:**

```
════════════════════════════════════════════════════════════════ ← WHITE tab bar
  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐
  │ Transcription    │  │ Transcription  │  │ AI Translation │  ← Inactive:
  │ Mode (sky-700,   │  │ Adv. Options   │  │ Options        │    LIGHTER (#FAFAFA)
  │ BOLD, 11pt)      │  │ (gray-600,10pt)│  │ (gray-600,10pt)│    shorter, normal
  └──────────────────┴──┴────────────────┴──┴────────────────┴──
───────────────────────────────────────────────────────────────── ← Tab bar border
                                                                    (broken by active)
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓                                                          ▓▓
▓▓  LIGHT GRAY CONTENT AREA (#F0F0F0)                      ▓▓
▓▓  (seamlessly connected to active tab)                   ▓▓
▓▓                                                          ▓▓
```

**Color Relationships:**
- Tab bar: #FFFFFF (white) - brightest
- Inactive tabs: #FAFAFA (very light gray) - lighter than active
- Active tab: #F0F0F0 (light gray) - matches content, darker than inactive
- Content area: #F0F0F0 (light gray) - same as active tab

**Implementation Details:**
- Method: `_setup_notebook_style()` (~95 lines with comments)
- Uses `ttk.Style.configure()` and `ttk.Style.map()`
- Configures `TNotebook`, `TNotebook.Tab`, and `TNotebook.client` styles
- State-based styling for selected, active (hover), and default states
- Different padding for selected vs. inactive to create height difference
- Cross-platform compatible (standard ttk approach)

**Accessibility:**
- Clear visual distinction (not color-only)
- Bold font weight for active tab (typographic hierarchy)
- Good contrast ratios
- Larger clickable areas

### Commits

_No commits yet - awaiting user testing validation_

---

## Risk Assessment

### Low Risk
- ✅ No changes to business logic
- ✅ No changes to CLI argument building
- ✅ No changes to subprocess execution
- ✅ All widget variables preserved

### Medium Risk
- ⚠️ Grid layout changes could affect responsive resizing
  - **Mitigation:** Test window resize in multiple directions

### Zero Risk
- ✅ Build process unchanged
- ✅ Dependencies unchanged
- ✅ Configuration files unchanged

---

## Next Steps (Post Step 1)

### Step 2: Design AI Translation Options Tab

**Prerequisites:**
- ✅ Step 1 validated by user
- ✅ Tab architecture proven stable

**Research Required:**
1. Analyze `whisperjav/translate/cli.py` for all CLI options
2. Review `whisperjav/translate/providers.py` for AI provider configs
3. Review `whisperjav/translate/instructions.py` for preset system
4. Review `whisperjav/translate/settings.py` for persistence mechanism
5. Understand PySubtrans integration in `whisperjav/translate/core.py`

**Design Decisions Needed:**
1. Which translate options to expose in GUI?
2. Layout design for 3rd tab
3. Workflow integration: standalone vs. integrated mode
4. Settings persistence strategy

### Step 3: Implement AI Translation Tab

_Details TBD after Step 2 design phase_

### Step 4: Design Ensemble Options Placeholder

_Details TBD - reserved for future whisperjav-ensemble feature_

---

## References

### Source Files
- Main GUI: `whisperjav/gui/whisperjav_gui.py`
- Translate CLI: `whisperjav/translate/cli.py`
- Translate Core: `whisperjav/translate/core.py`
- Design Wireframe: `whisperjav/gui/screen/GUI-Design-Proposed-NEW-UI-TABS.pdf`

### Documentation
- CLAUDE.md: Project overview and development commands
- README.md: User-facing documentation

---

## Change Log

| Date | Phase | Description |
|------|-------|-------------|
| 2025-10-29 | Step 1 | Initial plan document created |
| 2025-10-29 | Step 1 | Tab-based refactoring implementation started |
| 2025-10-29 | Step 1 | ✅ Tab-based refactoring completed - awaiting user testing |
| 2025-10-29 | Step 1 | ❌ Enhanced tab styling v1 (Professional Blue) - rejected by user |
| 2025-10-29 | Step 1 | ✅ File folder tab design implemented per user specification |
| 2025-10-29 | Step 1 | 🎨 Adjusted colors: white tab bar, inactive tabs lighter than active |

---

_End of Document_
