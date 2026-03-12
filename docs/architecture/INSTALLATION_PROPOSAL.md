# WhisperJAV Installation System – Unified Proposal

**Date:** 2026-03-12
**Target Versions:** v1.8.8 (Phase 1) -> v2.0+ (Phases 2-4)
**Author:** [Expert Consultant]
**For:** Review and agreement by WhisperJAV development team.

---

## 1. Executive Summary

WhisperJAV currently has two installation paths (source install and conda-constructor Windows installer) that use different dependency management methods. This causes maintenance overhead and user confusion. The proposed solution unifies both paths around a single core: **`uv sync` with a lockfile (`uv.lock`)**. It adds a smart launcher that abstracts environment details from end users, and it separates user data from the installation folder. The result is a **reproducible, fast, and consumer-friendly** installation experience that also serves developers seamlessly.

---

## 2. Core Principles

- **Consumer-first design:** End users (Sarah, Mark, Alex) must never see or interact with Python environments, terminals, or activation commands. They only double-click an installer and then a desktop shortcut.
- **Developer convenience:** Power users (Jamie) can install into their active Conda/venv with one command, and all dependencies are exactly pinned.
- **Reproducibility:** Every installation, regardless of path, uses the same lockfile, guaranteeing identical package versions.
- **Self-healing:** The launcher automatically recreates a missing or corrupted environment.
- **Data safety:** User-generated content (models, configs) lives outside the installation folder, so upgrades never wipe personal files.

---

## 3. User Personas and Requirements

| Persona | Description | Key Requirements |
|--------|-------------|------------------|
| **Sarah** - Fresh consumer (Windows) | Non-technical, downloads .exe installer. | One-click install to default folder. Desktop shortcut that always opens GUI. No terminal, no environment knowledge. |
| **Mark** - Upgrader | Existing user of version 1.x. | New installer detects old installation and upgrades in place. Data preserved. Shortcut continues to work. |
| **Alex** - Portable user | Wants to run from USB, no admin rights. | ZIP archive with everything needed. Double-click launcher works on any Windows PC. |
| **Jamie** - Developer | Uses Conda/venv, may want editable install. | Install into active environment with one command. Exact reproducibility via lockfile. Can run `whisperjav` command directly. |
| **Returning user** | Installed weeks ago, wants to use app again. | Double-click shortcut just works. If environment broken, it auto-repairs. |

---

## 4. Proposed Solution – Components

### 4.1 Core Installation Script (`install.py`)

- **Inputs:** optional `--local`, `--upgrade`, `--dev`, `--env PATH` flags.
- **Behavior:**
  1. **Preflight checks:** Python version, disk space, network, FFmpeg, Git.
  2. **Environment detection (only if no override flag):**
     - Check `VIRTUAL_ENV` (virtual environment) -> validate path and Python.
     - Else check `CONDA_PREFIX` (Conda environment) -> validate.
     - Else use project-local `.venv` (fallback).
  3. **Set `UV_PROJECT_ENVIRONMENT`** to the target environment path.
  4. **GPU detection** -> select PyTorch index URL (cu128, cu118, cpu, metal).
  5. **Run `uv sync`** with:
     - Appropriate extras (from `--dev`, etc.)
     - `--index pytorch=<url>` (unless metal)
     - `--inexact` if installing into an existing non-empty environment (to avoid removing unrelated packages).
  6. **Optional:** Install `llama-cpp-python` (separate `uv pip install` call, refactored into a reusable function).
  7. **Verification:** Import test and CUDA test.
  8. **Write `install.json`** in the user data directory (see Section 4.4) containing:
     - Installation path
     - Version installed
     - Timestamp

- **Why this works for consumers:** When no environment variables are set (the typical consumer case), it falls back to `.venv` inside the app folder - exactly what we want.

### 4.2 Launcher

- **A Python script (`launcher.py`)** placed in the installation folder (or bundled in the `.exe`).
- **Responsibilities:**
  - Read `install.json` to locate installation and user data directories.
  - Check if the target environment (e.g., `runtime\venv` or Conda env) exists and is intact:
    - Verify that key packages can be imported (e.g., `torch`, `whisperjav`).
    - Optionally compare a hash of the lockfile with a stored hash to detect changes.
  - If missing or corrupted, run `install.py --local` (for consumer installs) or `install.py --upgrade` (for upgrades) using the bundled Python.
  - Set environment variable `WHISPERJAV_DATA_DIR` to the user data path.
  - Launch the GUI by executing the environment's Python with `-m whisperjav_gui`.
- **Platform wrappers:**
  - **Windows:** A `.bat` file that calls `pythonw.exe launcher.py` (or a tiny C# executable that hides the console).
  - **macOS:** A shell script inside the `.app` bundle.
  - **Linux:** A shell script placed in `/usr/bin/whisperjav` or used by the desktop entry.

### 4.3 Conda-Constructor Installer (Windows)

- **No change to the installer UI or build process** - same `.exe` output.
- **Changes inside the installer:**
  - Include `uv.lock` and the enhanced `install.py` in the package.
  - Replace the 2,800-line `post_install.py` with a minimal script that:
    - Detects GPU (reuse `standalone.py`).
    - If this is a fresh install (no previous `install.json`), runs `install.py --local`.
    - If an existing installation is detected (via `install.json`), runs `install.py --upgrade`.
    - Creates a desktop shortcut pointing to the launcher (not directly to Python).
  - Ensure `uv.exe` is already bundled (as currently done).
  - Write `install.json` to `%APPDATA%\WhisperJAV` (user data location).

### 4.4 User Data Separation

- **Location:**
  - Windows: `%APPDATA%\WhisperJAV`
  - macOS: `~/Library/Application Support/WhisperJAV`
  - Linux: `~/.local/share/whisperjav`
- **Contents:**
  - `models/` - downloaded AI models
  - `config.toml` - user preferences
  - `logs/` - application logs
  - `install.json` - installation metadata
- The launcher passes the data directory to the GUI via `WHISPERJAV_DATA_DIR`.

### 4.5 Portable ZIP Distribution

- **Structure:** Same as an installed folder, but without the `venv` (which is created on first run).
- **Contents:** Bundled Python, `launcher.py`, `install.py`, `uv.lock`, application code, and a `README.txt`.
- **First run:** The launcher detects no environment, runs `install.py --local`, creates `venv`, then launches GUI.
- **No registry or system-wide changes** - fully portable.

### 4.6 Upgrade System (`upgrade.py`)

- **Unified detection:** Reads `install.json` to locate the installation.
- **Two modes:**
  - **Source install (uv-based):** Runs `git pull` then `install.py --upgrade`.
  - **Conda-constructor install:** Runs the new installer (which internally calls `install.py --upgrade`).
- Preserves user data by not touching the data directory.

---

## 5. Phased Implementation Plan

### Phase 1a (v1.8.8) – Core `install.py` Enhancements (Source Path Only)

**Scope:**
- Enhance `install.py` with:
  - Environment detection: `VIRTUAL_ENV` -> `CONDA_PREFIX` -> fallback to local `.venv`.
  - `--local` flag to force local `.venv` creation.
  - Use `UV_PROJECT_ENVIRONMENT` when targeting active environment.
  - **Automatic `--inexact`** when target is an external environment (conda/venv).
  - Refactor `_install_local_llm()` into top-level `run_uv_pip_install()`.
- Update `upgrade.py` for source installs to use `install.py --upgrade` (internal).
- **Remove `.python-version`** from repo and add to `.gitignore`.
- **Spike: Test Approach A for conda-constructor** - bundle `pyproject.toml` and `uv.lock` in a minimal test installer to verify feasibility. This spike will answer:
  - Can `construct.yaml` include extra files?
  - Can `uv sync` run inside the conda environment with `UV_PROJECT_ENVIRONMENT` pointing to it?
  - Does the result match the current `.whl` + `pip` pipeline?
  - Document findings and decide Phase 1b approach.
- **Test matrix for `uv sync` in conda (to be run during spike):**
  1. Clean conda env -> `uv sync --inexact --extra cli` -> verify packages installed, conda packages untouched.
  2. Conda env with unrelated packages (pandas, scikit-learn) -> `uv sync --inexact --extra cli` -> verify unrelated packages remain.
  3. Conda env with conflicting numpy (1.26) -> `uv sync --inexact --extra cli` -> verify numpy upgraded to lockfile version or handled gracefully.
  4. Simulated v1.8.7 pip-installed env -> `uv sync --inexact --extra cli` -> verify upgrade works cleanly.

**Outcome:** Source install path fixed, reproducible, and conda-compatible. A clear decision on conda-constructor integration approach.

---

### Phase 1b (v1.9.0) – Conda-Constructor Integration

**Scope (depends on spike outcome):**

- **If Approach A feasible:**
  - Modify `build_release.py` to bundle `pyproject.toml`, `uv.lock`, and the `whisperjav/` source directory alongside the `.whl`.
  - Update post-install script to copy these files to the installation root and run `install.py --local` (or `--upgrade` if upgrading).
- **If Approach A infeasible, fallback to Approach B:**
  - Keep existing packaging, but replace the `uv pip install` calls with a single `uv pip install` that reads pinned versions from `uv.lock` (i.e., generate a temporary `requirements.txt` from the lockfile and install with `--constraint`). This still gives lockfile reproducibility, though slower than `uv sync`.

- **Regardless of approach:**
  - **Create a minimal post-install wrapper** (Python script) that handles:
    - VC++ Redist detection / installation.
    - WebView2 detection and prompt.
    - GFW proxy detection (registry scan).
    - Git timeout reconfiguration.
    - Then calls `install.py --local` (or `--upgrade`) with the bundled Python.
  - This wrapper replaces the current 2,800-line `post_install.py` with a ~300-line focused script.
  - **Add `install.json` writing** to the installation root (for now; will be migrated in Phase 2). Contains: install path, version, timestamp.
  - **Update `upgrade.py`** to detect conda-constructor installs via `install.json` and trigger the new installer (or an in-place upgrade using `install.py --upgrade` if environment is uv-compatible).

- **Testing:**
  - Full Windows installer test on clean machines, with/without VC++/WebView2.
  - Upgrade test from v1.8.7 to v1.9.0.
  - Test behind simulated GFW (e.g., with network delay/packet loss).
  - Validate that `install.json` is correctly written and read by `upgrade.py`.

**Outcome:** Conda-constructor installer now uses the same lockfile-based logic as source installs, with full reproducibility and proper handling of Windows-specific dependencies.

---

### Phase 2 (v2.0+) – Launcher and User Data Separation

**Scope:**
- Develop `launcher.py` (Python script) with:
  - Read `install.json` from data directory (new location).
  - Check environment integrity (import test, optional lockfile hash comparison).
  - If missing/corrupted, run `install.py --local` (or `--upgrade`) using bundled Python.
  - Set `WHISPERJAV_DATA_DIR` environment variable.
  - Launch GUI by executing environment's Python with `-m whisperjav_gui`.
- **Platform wrappers:**
  - Windows: `.bat` file calling `pythonw.exe launcher.py` (or compiled C# stub).
  - macOS: shell script inside `.app` bundle.
  - Linux: shell script in `/usr/bin` or desktop entry.
- **Move all user data** to platform-specific data directory:
  - Windows: `%APPDATA%\WhisperJAV`
  - macOS: `~/Library/Application Support/WhisperJAV`
  - Linux: `~/.local/share/whisperjav`
- **Update codebase** to respect `WHISPERJAV_DATA_DIR` for:
  - Model cache (currently HuggingFace cache + `.whisperjav_cache/`).
  - Config files (currently `whisperjav_config.json`).
  - Logs (currently `whisperjav.log`).
- Provide migration script for existing users to move data from old locations.
- Modify installer to create shortcut pointing to launcher, not directly to Python.
- Update `install.py` to write `install.json` to the new data directory.

**Testing:**
- Consumer workflow: install -> shortcut -> launcher starts GUI.
- Self-healing: delete environment -> launcher recreates it.
- Upgrade preserves data in new location.
- Migration from v1.9.0 works correctly.

**Outcome:** Full consumer-friendly experience; environment complexity completely hidden.

---

### Phase 3 (v2.1+) – Portable ZIP and Cross-Platform Expansion

**Scope:**
- **Portable ZIP:** Package same structure as installed version (bundled Python, `launcher.py`, `install.py`, lockfile, app code) - without pre-created `venv`. Launcher creates it on first run.
- **macOS `.dmg` installer:** Use same internal layout inside `WhisperJAV.app/Contents/Resources`. Post-install script runs `install.py --local`.
- **Linux packaging:**
  - AppImage: self-contained, uses bundled Python and launcher.
  - `.deb`/`.rpm` packages for system-wide install (optional).
- **In-app updates** (optional): launcher checks for new version, downloads installer or runs `git pull` + `install.py --upgrade`.

**Testing:** Cross-platform validation on clean macOS/Linux systems.

---

## 6. Key Design Decisions (Final)

| Decision | Rationale |
|----------|-----------|
| **All installations eventually use `uv sync` + lockfile** | Reproducibility, speed, hash verification, declarative GPU routing. |
| **Conda-constructor integration via Approach A (bundle `pyproject.toml` + `uv.lock`)** if feasible; otherwise Approach B (lockfile-aware `uv pip install`). | A spike in Phase 1a will determine feasibility. |
| **Environment detection order:** `VIRTUAL_ENV` first, then `CONDA_PREFIX`. | `VIRTUAL_ENV` is explicitly user-activated; `CONDA_PREFIX` may be base env. |
| **`--inexact` is automatic** when targeting an external environment (conda/venv). | Prevents accidental removal of unrelated packages. |
| **`--upgrade` flag is internal only** - used by `upgrade.py`; not user-facing. | Keeps responsibilities clear; `upgrade.py` handles version checks and rollback. |
| **`install.json` is introduced in Phase 1b**, not Phase 1a. | Not needed for source-only fixes; avoids temporary locations and rework. |
| **Windows-specific checks** remain in a minimal post-install wrapper, not in `install.py`. | Keeps `install.py` cross-platform; wrapper handles platform quirks. |
| **User data separation (`WHISPERJAV_DATA_DIR`) is Phase 2** - requires app-wide changes. | Avoids bloating installation logic; done after core is stable. |
| **`.python-version` file removed** from repo and added to `.gitignore`. | Prevents uv from downloading its own Python. |

---

## 7. Detailed Responses to Developer's Points

### Point 1: `install.json` in Phase 1a – Is It Premature?

**Agreed.** We will **skip `install.json` entirely in Phase 1a**. It will be introduced in **Phase 1b** when it's needed for upgrade detection. This avoids temporary locations and rework.

### Point 2: Approach A Feasibility – Needs a Concrete Spike

**Absolutely.** The spike is added as a **blocking task in Phase 1a**. It will answer:
- Can `construct.yaml` bundle extra files (`pyproject.toml`, `uv.lock`, `whisperjav/`)?
- Can `uv sync` run inside the conda env with `UV_PROJECT_ENVIRONMENT`?
- Does the result match the current `.whl` + `pip` pipeline?
- Based on the outcome, we will finalize the Phase 1b approach.

### Point 3: `--inexact` Should Be Automatic When Targeting External Envs

**Fully agreed.** The logic in `install.py` will be:
```python
if target_env is external (conda or venv detected):
    inexact = True
else:  # target is .venv we created
    inexact = False
```
No user flag needed.

### Point 4: The Minimal Post-Install Wrapper (Phase 1b)

**Agreed.** We will keep a **minimal Windows-specific wrapper** (200-300 lines) that handles VC++ Redist, WebView2, GFW proxy, and Git timeout, then calls `install.py`. This keeps `install.py` platform-agnostic and simplifies maintenance.

### Point 5: What Does `--upgrade` Actually Do?

**Clarified:** `install.py --upgrade` is an **internal flag**, not documented for end users. It will:
- Run in non-interactive mode (skip prompts).
- Imply `--inexact` (always, because it's upgrading an existing environment).
- Log that it's an upgrade.
- Update `install.json` with new version.
- **Do not** perform `git pull` - that's `upgrade.py`'s job.

`upgrade.py` will handle version checking, backup, and then call `install.py --upgrade`.

### Point 6: `uv sync` Conda Testing – What Specifically to Test

The test matrix provided is excellent and will be used during the spike in Phase 1a. It covers:
- Clean conda env.
- Conda env with unrelated packages.
- Conda env with conflicting numpy.
- Previous pip-installed env (simulating v1.8.7 migration).

These tests will inform whether `uv sync --inexact` behaves as expected in all scenarios, and if any adjustments are needed (e.g., always creating a dedicated `venv` inside the conda env if conflicts arise).
