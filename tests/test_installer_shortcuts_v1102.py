"""1.10.2: the Windows installer's shortcuts (1.10.1 install test, findings 1 and 2).

Finding 1: the uninstaller left the desktop shortcut. Finding 2: the Start-menu folder "WhisperJAV" held only "Git
Bash", the bundled git package's menu entry. Reads the templates only; nothing is built or installed here. The
uninstall command itself was run against scratch folders (2026-11-12); the real uninstaller is checked by the install
test of the release build.
"""
import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "installer" / "templates"
NSI = (TEMPLATES / "custom_template.nsi.tmpl.template").read_text(encoding="utf-8")
README = (TEMPLATES / "README_INSTALLER.txt.template").read_text(encoding="utf-8")
CONSTRUCT = (TEMPLATES / "construct.yaml.template").read_text(encoding="utf-8")


def _uninstall_section():
    start = NSI.index('Section "Uninstall"')
    return NSI[start:NSI.index("SectionEnd", start)]


def test_conda_packages_never_bring_their_menu_entries():
    installs = re.findall(r"_conda\.exe\" install --offline[^\n]*", NSI)
    assert installs, "the conda install line was not found"
    assert all("--no-shortcuts" in line for line in installs)
    assert "{{ env.shortcuts }}" not in NSI


def test_menu_packages_not_emptied():
    # An empty menu_packages would remove the "Create shortcuts" tick box, and with it the desktop shortcut.
    assert not re.search(r"^menu_packages\s*:", CONSTRUCT, re.MULTILINE)


def test_start_menu_entry_for_whisperjav_itself():
    assert 'CreateShortCut "$SMPROGRAMS\\WhisperJAV\\WhisperJAV.lnk"' in NSI
    assert 'CreateShortCut "$DESKTOP\\WhisperJAV v{{VERSION}}.lnk"' in NSI


def test_uninstall_removes_only_this_installs_links():
    section = _uninstall_section()
    m = re.search(r"nsExec::ExecToLog `(.*?)`", section)
    assert m, "the shortcut removal command is not in the Uninstall section"
    cmd = m.group(1)
    assert "'$DESKTOP', 'WhisperJAV v*.lnk'" in cmd
    assert "'$SMPROGRAMS\\WhisperJAV', '*.lnk'" in cmd
    assert "$$root = '$INSTDIR\\'" in cmd and "StartsWith($$root" in cmd
    assert 'RMDir "$SMPROGRAMS\\WhisperJAV"' in section
    # It runs before the install folder is removed.
    assert section.index("nsExec::ExecToLog `") < section.index("Removing files and folders")


def test_readme_says_what_the_uninstall_removes():
    assert "leaves the desktop shortcut" not in README
    assert "removes this install's desktop shortcut" in README
