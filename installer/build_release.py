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
"""

import os
import sys
import shutil
import subprocess
import configparser
import argparse
from pathlib import Path
from datetime import datetime

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
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

    def _substitute_placeholders(self, content):
        """Replace all placeholders in content."""
        for placeholder, value in self.placeholders.items():
            content = content.replace(placeholder, value)
        return content

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
        file_mappings = {
            'construct.yaml.template': f'construct_v{self.version}.yaml',
            'post_install.bat.template': f'post_install_v{self.version}.bat',
            'post_install.py.template': f'post_install_v{self.version}.py',
            'requirements.txt.template': f'requirements_v{self.version}.txt',
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
            content = self._substitute_placeholders(content)

            if self.dry_run:
                print(f"  [DRY-RUN] Would generate {output_name}")
            else:
                output_path.write_text(content, encoding='utf-8')
                print(f"  {GREEN}✓{RESET} Generated {output_name}")

            generated_count += 1

        print(f"\n  Generated {generated_count} files")
        return True

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

        # Build wheel
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "bdist_wheel"],
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
            ("Generate from templates", self.generate_from_templates),
            ("Build wheel package", self.build_wheel),
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
    args = parser.parse_args()

    builder = ReleaseBuilder(dry_run=args.dry_run)

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
