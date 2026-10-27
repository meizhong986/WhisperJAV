#!/usr/bin/env python3
"""
WhisperJAV Release Build Orchestrator
======================================

Single command to generate all version-specific installer files from templates.

Usage:
    python build_release.py           # Full build
    python build_release.py --dry-run # Preview without changes
    python build_release.py --clean   # Remove generated files
    python build_release.py --validate # Validate only
    python build_release.py --check-tree          # Only: no uncommitted/untracked files?
    python build_release.py --check-wheel <wheel> # Only: every package file tracked by git?

A full build stops if the working tree is not committed (checked after phase 1)
or if the wheel holds a package file that git does not track.

v1.10.0 (item 2f; design section 22): the dependency lists come from uv.lock, one per
lock extra (cu130, cu128, vulkan, cpu), exported into whisperjav/installer/locked/ (tracked by git, so a
lock change shows up as uncommitted files and stops the build until committed).
Needs uv on PATH (uv.lock's required-version names the lowest).
"""

import os
import sys
import shutil
import subprocess
import configparser
import argparse
from pathlib import Path
from datetime import datetime

# Configure UTF-8 encoding for Windows console.
#
# Only when this file is run as a script. Doing it on import replaced the
# importer's stdout with a wrapper over the raw buffer, which broke anything
# that had redirected stdout -- pytest's output capture, for one, so the
# generator could not be tested at all ("I/O operation on closed file").
if sys.platform == 'win32' and __name__ == '__main__':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# v1.10.0 (item 2f; design 22.6.8): the uv.lock extras, one locked file each, from the one
# accelerator table (standard library only, so this import needs nothing installed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from whisperjav.installer.core.accelerators import LOCK_EXTRAS as LOCKED_VARIANTS  # noqa: E402

class ReleaseBuilder:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.installer_dir = Path(__file__).parent
        self.project_root = self.installer_dir.parent
        self.templates_dir = self.installer_dir / "templates"
        self.generated_dir = self.installer_dir / "generated"
        self.version_file = self.installer_dir / "VERSION"

        # Load version config
        self.config = self._load_version_config()
        self.version = self._get_version_string()
        self.placeholders = self._build_placeholders()

    def _load_version_config(self):
        """Load VERSION file as config."""
        if not self.version_file.exists():
            print(f"{RED}ERROR: VERSION file not found at {self.version_file}{RESET}")
            sys.exit(1)

        config = configparser.ConfigParser()
        config.read(self.version_file)
        return config

    def _get_version_string(self):
        """Build PEP 440 compliant version string for wheel/pip.

        PEP 440 prerelease format: 1.7.0b0 (no hyphen!)
        Valid suffixes: a0, a1, b0, b1, rc0, rc1, etc.
        """
        major = self.config.get('version', 'major')
        minor = self.config.get('version', 'minor')
        patch = self.config.get('version', 'patch')
        prerelease = self.config.get('version', 'prerelease', fallback='')

        version = f"{major}.{minor}.{patch}"
        if prerelease:
            # PEP 440: prerelease is concatenated directly (no hyphen)
            version += prerelease
        return version

    def _get_display_version(self):
        """Build human-readable version string for display.

        Example: 1.7.0-beta (with hyphen for readability)
        """
        major = self.config.get('version', 'major')
        minor = self.config.get('version', 'minor')
        patch = self.config.get('version', 'patch')
        display_label = self.config.get('version', 'display_label', fallback='')

        version = f"{major}.{minor}.{patch}"
        if display_label:
            version += f"-{display_label}"
        return version

    def _build_placeholders(self):
        """Build placeholder dictionary for template substitution.

        Two version formats are available:
        - {{VERSION}}: PEP 440 compliant (1.7.0b0) for wheel/pip/filenames
        - {{DISPLAY_VERSION}}: Human-readable (1.7.0-beta) for UI/documentation
        """
        display_version = self._get_display_version()
        display_label = self.config.get('version', 'display_label', fallback='stable')

        return {
            # PEP 440 version for wheel/pip/filenames
            '{{VERSION}}': self.version,
            # Human-readable version for display
            '{{DISPLAY_VERSION}}': display_version,
            '{{DISPLAY_LABEL}}': display_label,
            '{{VERSION_MAJOR}}': self.config.get('version', 'major'),
            '{{VERSION_MINOR}}': self.config.get('version', 'minor'),
            '{{VERSION_PATCH}}': self.config.get('version', 'patch'),
            '{{APP_NAME}}': self.config.get('metadata', 'app_name'),
            '{{DESCRIPTION}}': self.config.get('metadata', 'description'),
            '{{AUTHOR}}': self.config.get('metadata', 'author'),
            '{{LICENSE}}': self.config.get('metadata', 'license'),
            '{{URL}}': self.config.get('metadata', 'url'),
            '{{PYTHON_VERSION}}': self.config.get('installer', 'python_version'),
            # User-facing names use display version
            '{{SHORTCUT_NAME}}': self.config.get('installer', 'shortcut_name_template').format(version=display_version),
            '{{SHORTCUT_DESCRIPTION}}': self.config.get('installer', 'shortcut_description').format(version=display_version),
            '{{INSTALL_PREFIX}}': self.config.get('installer', 'install_prefix'),
            '{{ARCHITECTURE}}': self.config.get('version', 'architecture'),
        }

    @staticmethod
    def _strip_template_comments(content):
        """
        Drop lines that are notes to whoever maintains the template.

        A template is copied to the user almost verbatim, so there is nowhere in
        it to leave a note without the user reading it. A line whose first
        non-blank characters are ``{{!`` is for the maintainer and never ships.

        This exists because the installer README's "WHAT'S NEW" section still
        described the v1.9.0 features under every later version: there was no
        way to leave a reminder beside the text that needed rewriting.
        """
        return "".join(
            line for line in content.splitlines(keepends=True)
            if not line.lstrip().startswith("{{!")
        )

    def _substitute_placeholders(self, content):
        """Replace all placeholders in content."""
        for placeholder, value in self.placeholders.items():
            content = content.replace(placeholder, value)
        return content

    def locked_requirements_dir(self):
        """whisperjav/installer/locked/ -- the exported files, shipped in the wheel too."""
        return self.project_root / "whisperjav" / "installer" / "locked"

    def export_locked_requirements(self):
        """v1.10.0 (item 2f): export uv.lock once per PyTorch build, exact versions, no hashes.

        Every install -- this installer, install.py, whisperjav-upgrade -- installs one of
        these files, so users get the versions development tested (owner, W53-W56).
        Stops if uv.lock no longer matches pyproject.toml: run `uv lock`, commit, build again.
        """
        self.print_header("Phase 1.5: Export Locked Requirements (uv.lock)")
        uv = shutil.which("uv")
        if not uv:
            print(f"  {RED}✗{RESET} uv not found on PATH (see installer/build_env.yml for the version)")
            return False
        check = subprocess.run([uv, "lock", "--check"], cwd=self.project_root, capture_output=True, text=True)
        if check.returncode != 0:
            print(f"  {RED}✗{RESET} uv.lock does not match pyproject.toml:")
            print(f"      {(check.stderr or check.stdout).strip()[:500]}")
            print("  Run `uv lock`, commit uv.lock, then build again.")
            return False
        print(f"  {GREEN}✓{RESET} uv.lock matches pyproject.toml")

        out_dir = self.locked_requirements_dir()
        for variant in LOCKED_VARIANTS:
            command = ["export", "--frozen", "--no-hashes", "--no-emit-project", "--no-header",
                       "--extra", "all", "--extra", variant, "--format", "requirements-txt"]
            result = subprocess.run([uv, *command], cwd=self.project_root, capture_output=True,
                                    text=True, encoding="utf-8")
            if result.returncode != 0:
                print(f"  {RED}✗{RESET} uv export ({variant}) failed: {result.stderr.strip()[:500]}")
                return False
            content = (
                f"# WhisperJAV -- release-pinned dependencies, PyTorch build {variant}\n"
                f"# Exported from uv.lock by installer/build_release.py; do not edit -- change\n"
                f"# pyproject.toml, run `uv lock`, then build again.\n"
                f"# uv {' '.join(command)}\n"
                + result.stdout
            )
            path = out_dir / f"requirements_{variant}.txt"
            if self.dry_run:
                print(f"  [DRY-RUN] Would write {path.relative_to(self.project_root)}")
                continue
            old = path.read_bytes() if path.exists() else None
            out_dir.mkdir(exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            state = "unchanged" if old == path.read_bytes() else "UPDATED -- commit it"
            pins = len([l for l in content.splitlines() if l and not l.startswith((" ", "#"))])
            print(f"  {GREEN}✓{RESET} {path.relative_to(self.project_root)} ({pins} packages, {state})")
        return True

    def print_header(self, title):
        """Print formatted header."""
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def update_version_py(self):
        """Update whisperjav/__version__.py with current version."""
        self.print_header("Phase 1: Update Core Version")

        version_py = self.project_root / "whisperjav" / "__version__.py"
        major = self.config.get('version', 'major')
        minor = self.config.get('version', 'minor')
        patch = self.config.get('version', 'patch')
        arch = self.config.get('version', 'architecture')
        display_label = self.config.get('version', 'display_label', fallback='stable')
        display_version = self._get_display_version()

        content = f'''#!/usr/bin/env python3
"""Version information for WhisperJAV."""

# PEP 440 compliant version for pip/wheel
__version__ = "{self.version}"

# Human-readable version for display in UI
__version_display__ = "{display_version}"

# Version metadata
__version_info__ = {{
    "major": {major},
    "minor": {minor},
    "patch": {patch},
    "release": "{display_label}",
    "architecture": "{arch}"
}}
'''

        if self.dry_run:
            print(f"  [DRY-RUN] Would update {version_py}")
            print(f"            PEP 440 version: {self.version}")
            print(f"            Display version: {display_version}")
        else:
            version_py.write_text(content)
            print(f"  {GREEN}✓{RESET} Updated {version_py}")
            print(f"      PEP 440 version: {self.version}")
            print(f"      Display version: {display_version}")

        return True

    def generate_from_templates(self):
        """Generate all version-specific files from templates."""
        self.print_header("Phase 2: Generate Files from Templates")

        if not self.templates_dir.exists():
            print(f"{RED}ERROR: Templates directory not found: {self.templates_dir}{RESET}")
            return False

        # Create generated directory
        if not self.dry_run:
            self.generated_dir.mkdir(exist_ok=True)

        # Template to output filename mapping
        # The requirements files are not templates: they are exported from uv.lock (item 2f)
        file_mappings = {
            'construct.yaml.template': f'construct_v{self.version}.yaml',
            'post_install.bat.template': f'post_install_v{self.version}.bat',
            'post_install.py.template': f'post_install_v{self.version}.py',
            'WhisperJAV_Launcher.py.template': f'WhisperJAV_Launcher_v{self.version}.py',
            'README_INSTALLER.txt.template': f'README_INSTALLER_v{self.version}.txt',
            'build_installer.bat.template': f'build_installer_v{self.version}.bat',
            'validate_installer.py.template': f'validate_installer_v{self.version}.py',
            'custom_template.nsi.tmpl.template': f'custom_template_v{self.version}.nsi.tmpl',
            'create_desktop_shortcut.bat.template': f'create_desktop_shortcut_v{self.version}.bat',
            'uninstall.bat.template': f'uninstall_v{self.version}.bat',
        }

        generated_count = 0
        for template_name, output_name in file_mappings.items():
            template_path = self.templates_dir / template_name
            output_path = self.generated_dir / output_name

            if not template_path.exists():
                print(f"  {YELLOW}⚠{RESET} Template not found: {template_name}")
                continue

            # Read template and substitute placeholders
            content = template_path.read_text(encoding='utf-8')
            content = self._strip_template_comments(content)
            content = self._substitute_placeholders(content)

            if self.dry_run:
                print(f"  [DRY-RUN] Would generate {output_name}")
            else:
                output_path.write_text(content, encoding='utf-8')
                print(f"  {GREEN}✓{RESET} Generated {output_name}")

            generated_count += 1

        # v1.10.0 (item 2f): the locked files, under the installer's versioned names.
        for variant in LOCKED_VARIANTS:
            src = self.locked_requirements_dir() / f"requirements_{variant}.txt"
            output_name = f"requirements_v{self.version}_{variant}.txt"
            if self.dry_run:
                print(f"  [DRY-RUN] Would copy {src.name} to {output_name}")
            elif not src.exists():
                print(f"  {RED}✗{RESET} {src} missing (phase 1.5 writes it)")
                return False
            else:
                shutil.copy(src, self.generated_dir / output_name)
                print(f"  {GREEN}✓{RESET} Copied {output_name}")
            generated_count += 1

        # design 22.6.8: the one accelerator table, shipped next to post_install (which cannot
        # import WhisperJAV before its phase 5) -- the file itself, so there is no second copy.
        table = self.project_root / "whisperjav" / "installer" / "core" / "accelerators.py"
        output_name = f"whisperjav_accelerators_v{self.version}.py"
        if self.dry_run:
            print(f"  [DRY-RUN] Would copy {table.name} to {output_name}")
        else:
            shutil.copy(table, self.generated_dir / output_name)
            print(f"  {GREEN}✓{RESET} Copied {output_name}")
        generated_count += 1

        print(f"\n  Generated {generated_count} files")
        return True

    def remove_stale_build_dirs(self):
        """v1.10.0 (N1): remove setuptools' leftovers from earlier builds.

        A stale whisperjav.egg-info/SOURCES.txt and build/lib/ carry files from old
        builds into the new wheel even after the package-data rules stop matching them.
        """
        for stale in (self.project_root / "build", self.project_root / "whisperjav.egg-info"):
            if stale.exists():
                shutil.rmtree(stale)
                print(f"  Removed stale {stale.name}/")

    def build_wheel(self):
        """Build wheel package."""
        self.print_header("Phase 3: Build Wheel Package")

        wheel_name = f"whisperjav-{self.version}-py3-none-any.whl"

        if self.dry_run:
            print(f"  [DRY-RUN] Would build wheel: {wheel_name}")
            return True

        # Clean old wheels
        dist_dir = self.project_root / "dist"
        if dist_dir.exists():
            for whl in dist_dir.glob("*.whl"):
                whl.unlink()

        self.remove_stale_build_dirs()

        # Build wheel (PEP 517 via python -m build)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "build", "--wheel"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"  {RED}✗{RESET} Wheel build failed")
                print(result.stderr)
                return False

            # Copy to generated directory
            wheel_path = dist_dir / wheel_name
            if wheel_path.exists():
                dest = self.generated_dir / wheel_name
                shutil.copy(wheel_path, dest)
                print(f"  {GREEN}✓{RESET} Built and copied {wheel_name}")
            else:
                print(f"  {RED}✗{RESET} Wheel not found: {wheel_path}")
                return False

        except Exception as e:
            print(f"  {RED}✗{RESET} Error building wheel: {e}")
            return False

        return True

    def _git_lines(self, *args):
        """Run a git command in the project root; return its output lines."""
        result = subprocess.run(
            ["git", *args], cwd=self.project_root, capture_output=True, text=True, check=True
        )
        return [line for line in result.stdout.splitlines() if line]

    def check_tree(self):
        """v1.10.0 (N1): stop if the working tree holds uncommitted or untracked files.

        The wheel is built from the working folder, so anything lying there can end up
        in it. Runs after phase 1, so a __version__.py that differs from VERSION must be
        committed first. Ignored files are not listed here; check_wheel() catches those.
        """
        self.print_header("Check: Working Tree Committed")
        try:
            dirty = self._git_lines("status", "--porcelain")
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"  {RED}✗{RESET} git status failed: {e}")
            return False
        if not dirty:
            print(f"  {GREEN}✓{RESET} No uncommitted or untracked files")
            return True
        print(f"  {RED}✗{RESET} {len(dirty)} uncommitted or untracked file(s):")
        for line in dirty:
            print(f"      {line}")
        if self.dry_run:
            print(f"  [DRY-RUN] A real build would stop here")
            return True
        print("  Commit, move or delete them, then build again.")
        return False

    def check_wheel(self, wheel_path):
        """v1.10.0 (N1): stop if the wheel holds a package file that git does not track.

        Catches ignored files the package-data rules pick up (the 1.9.x wheels shipped
        50 asr_config.backup.*.json files this way).
        """
        import zipfile

        self.print_header("Check: Wheel Holds Only Tracked Files")
        wheel_path = Path(wheel_path)
        if not wheel_path.exists():
            if self.dry_run:
                print(f"  [DRY-RUN] Would check {wheel_path.name}")
                return True
            print(f"  {RED}✗{RESET} Wheel not found: {wheel_path}")
            return False
        try:
            tracked = set(self._git_lines("ls-files", "whisperjav"))
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"  {RED}✗{RESET} git ls-files failed: {e}")
            return False
        with zipfile.ZipFile(wheel_path) as zf:
            entries = [n for n in zf.namelist() if n.startswith("whisperjav/") and not n.endswith("/")]
        untracked = sorted(n for n in entries if n not in tracked)
        if not untracked:
            print(f"  {GREEN}✓{RESET} All {len(entries)} package files in {wheel_path.name} are tracked by git")
            return True
        print(f"  {RED}✗{RESET} {len(untracked)} file(s) in {wheel_path.name} are not tracked by git:")
        for name in untracked:
            print(f"      {name}")
        if self.dry_run:
            print(f"  [DRY-RUN] A real build would stop here")
            return True
        return False

    def copy_static_files(self):
        """Copy static files (LICENSE, icon) to generated directory."""
        self.print_header("Phase 4: Copy Static Files")

        static_files = ['LICENSE.txt', 'whisperjav_icon.ico']

        for filename in static_files:
            src = self.installer_dir / filename
            if src.exists():
                if self.dry_run:
                    print(f"  [DRY-RUN] Would copy {filename}")
                else:
                    dest = self.generated_dir / filename
                    shutil.copy(src, dest)
                    print(f"  {GREEN}✓{RESET} Copied {filename}")
            else:
                print(f"  {YELLOW}⚠{RESET} Not found: {filename}")

        return True

    def run_validation(self):
        """Run the validation script."""
        self.print_header("Phase 5: Validation")

        if self.dry_run:
            print("  [DRY-RUN] Would run validation")
            return True

        validator = self.generated_dir / f"validate_installer_v{self.version}.py"
        if not validator.exists():
            print(f"  {YELLOW}⚠{RESET} Validator not found, skipping")
            return True

        try:
            result = subprocess.run(
                [sys.executable, str(validator)],
                cwd=self.generated_dir,
                capture_output=True,
                text=True
            )

            # Print validation output
            print(result.stdout)

            if result.returncode != 0:
                print(f"  {RED}✗{RESET} Validation failed")
                return False

        except Exception as e:
            print(f"  {RED}✗{RESET} Error running validation: {e}")
            return False

        return True

    def clean(self):
        """Remove generated directory."""
        self.print_header("Cleaning Generated Files")

        if self.generated_dir.exists():
            if self.dry_run:
                print(f"  [DRY-RUN] Would remove {self.generated_dir}")
            else:
                shutil.rmtree(self.generated_dir)
                print(f"  {GREEN}✓{RESET} Removed {self.generated_dir}")
        else:
            print("  Nothing to clean")

    def build(self):
        """Run full build process."""
        print()
        print("=" * 70)
        print(f"  WhisperJAV Release Builder - v{self.version}")
        print("=" * 70)

        if self.dry_run:
            print(f"\n  {YELLOW}DRY-RUN MODE - No changes will be made{RESET}\n")

        steps = [
            ("Update __version__.py", self.update_version_py),
            ("Export locked requirements", self.export_locked_requirements),
            ("Generate from templates", self.generate_from_templates),
            ("Check working tree", self.check_tree),
            ("Build wheel package", self.build_wheel),
            ("Check wheel contents", lambda: self.check_wheel(
                self.project_root / "dist" / f"whisperjav-{self.version}-py3-none-any.whl")),
            ("Copy static files", self.copy_static_files),
            ("Run validation", self.run_validation),
        ]

        for step_name, step_func in steps:
            if not step_func():
                print(f"\n{RED}Build failed at: {step_name}{RESET}")
                return False

        # Summary
        self.print_header("Build Complete")
        print(f"\n  Version: {self.version}")
        print(f"  Output:  {self.generated_dir}")
        print(f"\n  {GREEN}Ready to build installer:{RESET}")
        print(f"  cd generated && build_installer_v{self.version}.bat")
        print()

        return True


def main():
    parser = argparse.ArgumentParser(description="WhisperJAV Release Builder")
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--clean', action='store_true', help='Remove generated files')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    parser.add_argument('--check-tree', action='store_true',
                        help='Only check that the working tree has no uncommitted or untracked files')
    parser.add_argument('--check-wheel', metavar='WHEEL',
                        help='Only check that every package file in WHEEL is tracked by git')
    args = parser.parse_args()

    builder = ReleaseBuilder(dry_run=args.dry_run)

    if args.check_tree:
        return 0 if builder.check_tree() else 1

    if args.check_wheel:
        return 0 if builder.check_wheel(args.check_wheel) else 1

    if args.clean:
        builder.clean()
        return 0

    if args.validate:
        builder.run_validation()
        return 0

    success = builder.build()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
