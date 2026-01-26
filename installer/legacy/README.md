# Legacy Installer Backups

## Purpose

This directory contains backup copies of installer files before the v2 architecture refactoring.
These backups exist to:

1. **Enable safe rollback** if issues are discovered during refactoring
2. **Preserve institutional knowledge** about the original implementation
3. **Provide reference** for feature parity verification

## Backups Created

| File | Original Location | Lines | Key Features |
|------|-------------------|-------|--------------|
| `install.py.backup_v1.8.2` | `/install.py` | ~537 | Main CLI installer |
| `install_windows_v1.8.2.bat` | `/installer/install_windows.bat` | ~950 | Has retry logic, Git timeout |
| `install_linux_v1.8.2.sh` | `/installer/install_linux.sh` | ~823 | PEP 668 handling, missing retry |
| `install_colab.sh.backup_v1.8.2` | `/installer/install_colab.sh` | ~470 | Colab-specific optimizations |
| `post_install.py.template.backup_v1.8.2` | `/installer/templates/post_install.py.template` | ~1876 | Has retry, Git timeout, uv support |

## v2 Architecture Changes

The v2 architecture replaced the large shell scripts with thin wrappers:

| Script | Before (v1.8.2) | After (v2.0) | Change |
|--------|-----------------|--------------|--------|
| `install_windows.bat` | ~950 lines | ~115 lines | -88% |
| `install_linux.sh` | ~823 lines | ~191 lines | -77% |

All logic (GPU detection, retry, timeout handling) is now in:
- `install.py` (root) - Main Python installer
- `whisperjav/installer/` - Unified installer module

## Rollback Procedure

If critical issues are found, rollback using:

```bash
# Restore from git tag
git checkout pre-installer-refactor-v1.8.2

# Or restore individual files
cp installer/legacy/install.py.backup_v1.8.2 install.py
cp installer/legacy/install_windows.bat.backup_v1.8.2 installer/install_windows.bat
# etc.
```

## Feature Parity Reference

When implementing the new architecture, these files serve as the authoritative reference for:

- **Retry logic**: See `install_windows.bat:820-872` (`:run_pip_with_retry`)
- **Git timeout handling**: See `install_windows.bat:874-891` (`:configure_git_timeouts`)
- **GPU detection**: See `post_install.py.template` (`detect_nvidia_driver()`)
- **CUDA matrix**: See `post_install.py.template` (`TORCH_DRIVER_MATRIX`)
- **uv support**: See `post_install.py.template:41-50, 170-186`

## DO NOT DELETE

These backups should be retained until:
1. The v2 architecture is fully validated
2. At least one production release has been made with the new system
3. No regressions have been reported

Created: 2026-01-26
Git Tag: `pre-installer-refactor-v1.8.2`
