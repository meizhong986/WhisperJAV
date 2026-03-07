WhisperJAV Installation System - External Review Package
========================================================

Version: 1.8.2rc1
Date: 2026-01-28

This folder contains all installation-related files for external review.
All files have .txt extension added for easy ingestion.


FILE OVERVIEW
=============

1. MAIN ORCHESTRATOR
--------------------
install.py.txt
  - Main installation script run by users
  - Handles staged installation (PyTorch first for GPU lock-in)
  - Contains preflight checks, logging, Git timeout handling
  - Queries registry for package definitions (SSOT pattern)


2. SHELL WRAPPERS (Thin delegation layer)
-----------------------------------------
install_windows.bat.txt
  - Windows batch script wrapper
  - Delegates to install.py after basic environment setup

install_linux.sh.txt
  - Linux/macOS shell script wrapper
  - Handles PEP 668 externally-managed check
  - Delegates to install.py


3. INSTALLER CORE MODULE (whisperjav/installer/core/)
-----------------------------------------------------
registry.py.txt [CRITICAL - SINGLE SOURCE OF TRUTH]
  - Defines ALL packages with versions, extras, install order
  - Contains import name mappings (e.g., Pillow -> PIL)
  - Generates pyproject.toml extras and requirements.txt
  - Installation order enforces GPU lock-in pattern

standalone.py.txt
  - Self-contained utilities for standalone installer
  - MUST have ZERO imports from whisperjav.*
  - GPU detection, CUDA version selection, Git timeout handling

executor.py.txt
  - StepExecutor class for package installation
  - Retry logic, timeout handling, uv detection
  - Git timeout detection and auto-configuration

detector.py.txt
  - Platform detection (Windows, Linux, macOS, Apple Silicon)
  - GPU detection via nvidia-smi, pynvml, /proc/driver/nvidia
  - CUDA version selection based on driver matrix

config.py.txt
  - Configuration constants
  - CUDA driver matrix, timeout values, retry settings
  - Git timeout patterns for detection


4. VALIDATION MODULE (whisperjav/installer/validation/)
-------------------------------------------------------
validation_imports.py.txt
  - Import scanner using AST parsing
  - Catches "ghost dependencies" (imports not in registry)
  - Cross-references with registry import map

validation_sync.py.txt
  - Validates pyproject.toml matches registry
  - Prevents package drift between files

validation_main.py.txt
  - Validation runner for CI/CD
  - Run: python -m whisperjav.installer.validation


5. STANDALONE INSTALLER (conda-constructor)
-------------------------------------------
post_install_v1.8.2rc1.py.txt
  - Generated post-install script for .exe installer
  - Most comprehensive installation logic
  - Runs after conda-constructor creates base environment

post_install.py.template.txt
  - Template used to generate post_install_v*.py
  - Contains version placeholders


ARCHITECTURE NOTES
==================

GPU Lock-in Pattern:
  1. Install PyTorch FIRST with --index-url for CUDA wheels
  2. PyTorch is now "locked in" with GPU support
  3. Subsequent packages see torch as satisfied, don't pull CPU version

Installation Order (by registry order field):
  10-19: PyTorch (MUST BE FIRST)
  20-29: Scientific stack (numpy before numba)
  30-39: Whisper packages
  40-49: Audio/CLI packages
  50-59: GUI packages
  60-69: Translation packages
  70-79: Enhancement packages
  80-89: HuggingFace/optional

Single Source of Truth:
  - registry.py defines ALL packages
  - pyproject.toml, requirements.txt derived from registry
  - install.py queries registry instead of hardcoding
  - Validation catches drift


KEY ISSUES ADDRESSED
====================

Issue #90: CPU PyTorch resolution
  - Solved by installing torch first with --index-url

Issue #111: Git timeout on slow connections (GFW/VPN)
  - Detect 21-second TCP timeout pattern
  - Auto-configure Git with extended timeouts

ModelScope ZipEnhancer:
  - Added oss2 to ENHANCE extra (Alibaba Cloud SDK)


REVIEW FOCUS AREAS
==================

1. Security: No arbitrary code execution, proper input validation
2. Robustness: Error handling, retry logic, fallbacks
3. Correctness: Package versions, installation order
4. Completeness: All packages in registry, no ghost dependencies
5. User Experience: Clear error messages, logging, preflight checks


RUNNING VALIDATION
==================

From project root:
  python -m whisperjav.installer.validation

Expected output:
  [1/3] Checking standalone.py self-containment... OK
  [2/3] Validating pyproject.toml sync... OK
  [3/3] Scanning for untracked imports... OK
  VALIDATION PASSED: All checks passed
