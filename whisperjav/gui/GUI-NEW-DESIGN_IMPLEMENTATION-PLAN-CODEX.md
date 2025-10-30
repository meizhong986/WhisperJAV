# WhisperJAV-GUI New Design Implementation Plan

This plan outlines a staged migration from the current single-pane layout to the proposed tabbed interface that will eventually host transcription, translation, and ensemble controls. Each step is incremental, validated before moving forward, and designed to keep the GUI usable throughout.

---

## Step 1 – Establish Tab Framework (Status Quo Functionality)

### Step 1 Objective

- Refactor the existing `whisperjav-gui` to render a two-tab layout: `Transcription Mode` and `Transcription Adv. Options`.
- Both tabs should expose exactly the controls that are currently present in the GUI so functionality remains unchanged.
- Ensure tab switching is purely structural; no behavioural changes to command execution or data bindings.

### Step 1 Key Tasks

1. Audit current widget structure in `whisperjav/gui/whisperjav_gui.py` (and supporting modules) to map existing panels to future tabs.
2. Introduce a tab container (e.g., `ttk.Notebook`) with two tabs.
3. Move or wrap existing transcription widgets into the `Transcription Mode` tab.
4. Move advanced controls (currently grouped within the same view) into `Transcription Adv. Options` without altering their event handlers.
5. Validate that applying settings, running transcription, and displaying progress still work identically.

### Step 1 Exit Criteria

- GUI launches and looks visually similar aside from tab headers.
- All commands work as before; no features missing or broken.
- Structure ready to accept additional tabs.

---

## Step 2 – Design AI Translation Tab

### Step 2 Objective

- Define which `whisperjav-translate` options should be surfaced in the new `AI Translation Options` tab.
- Produce a control layout that fits the tabbed UX revealed in the wireframe.

### Step 2 Design Considerations

- Respect CLI precedence and defaults (`settings.resolve_config`, env vars, CLI flags) while exposing the options that deliver the most value.
- Keep UX approachable: identify essential vs advanced translation settings.
- Ensure key provider metadata (API key sources, tone/model choices, sampling parameters) fits within the panel.

### Step 2 Deliverables

- A documented list of controls to expose (labels, tooltips, default behaviour).
- Preliminary wireframe sketch or annotated description showing layout within the tab.
- Integration notes describing how the GUI selections map to CLI invocation arguments.

### Step 2 Exit Criteria

- Stakeholders agree on which translation options belong in the first release.
- Layout clear enough to guide implementation without further discovery.

---

## Step 3 – Implement AI Translation Tab

### Step 3 Objective

- Build the UI defined in Step 2 and wire it to the existing translation workflow.

### Step 3 Key Tasks

1. Create controls (dropdowns, toggles, input fields) per design.
2. Bind controls to an internal configuration object mirroring `whisperjav-translate` arguments.
3. Update the orchestration in `main.py`/GUI to pass selected translation options via the CLI handoff (`whisperjav-translate …`).
4. Ensure validation, default loading (`settings.resolve_config`), and persistence align with CLI behaviour.
5. Add contextual logging/feedback for translation progress, respecting `--no-progress` and quiet modes.

### Step 3 Exit Criteria

- Translation tab operational end-to-end (selection → CLI invocation → user feedback).
- Settings reflect run output; translation path is surfaced to the user.
- Regression tests confirm transcription modes still unaffected.

---

## Step 4 – Draft Ensemble Tab Concept

### Step 4 Objective

- Produce a UI-first concept for the future `Ensemble Options` tab.
- Keep this exploratory; no implementation yet.

### Step 4 Design Activities

- Brainstorm ensemble workflows (model blending, voting schemes, metadata overlays).
- Sketch UI variants that can accommodate these concepts.
- Document dependencies or data flows needed for ensemble integration.

### Step 4 Deliverables

- Narrative describing ensemble use cases and required controls.
- Low-fidelity wireframes or textual layout plans.
- Notes on technical prerequisites (e.g., pipeline hooks, config structures) to inform future scheduling.

### Step 4 Exit Criteria

- Clear direction for ensemble feature when development is greenlit.
- Design assets ready for product review.

---

### Execution Guidelines

- After completing each step, pause for validation (functional testing, stakeholder review) before moving on.
- Keep commits incremental, focusing on one step per pull request where possible.
- Maintain parity across CLI and GUI options; avoid GUI-only behaviour unless explicitly approved.
- Document notable design decisions directly within the repo (e.g., update `copilot-instructions.md` as patterns evolve).

This plan ensures we first lock in tabbed infrastructure, then layer translation functionality, and finally prepare for ensemble expansion without rushing implementation details.
