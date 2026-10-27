"""
Is this computer supported by WhisperJAV 1.10? -- one check, run first by every installer.

WHY THIS EXISTS (design section 22.1.2; owner W232 7, W233 R6, D-F of W234)
    1.10.0 supports a fixed list of computers and refuses every other one BEFORE anything
    is installed, in the owner's form: "WhisperJAV 1.10.0 does not support ... This is
    due to ... Nothing was installed."

WHO CALLS IT
    install.py (before its preflight checks), installer/install_linux.sh and
    installer/install_colab.sh (as their first step, with whatever python3 the system
    has). install_mac.sh makes its two tests in the shell first, because on an Intel Mac
    without the Xcode tools running python3 opens Apple's install dialog. The Windows
    installer refuses Windows on ARM and Windows before 10 on its first screen (NSIS).

RULES
    Standard library only, no imports from whisperjav.*, and Python 3.6 syntax: the
    shell scripts run this file directly with the system's python3, which may be old.

    python3 supported.py   -> prints the refusal and exits 1, or prints nothing and exits 0
"""

import os
import platform
import sys

MIN_GLIBC = (2, 35)
MIN_MACOS = (14, 0)
MIN_WINDOWS_MAJOR = 10
NOTHING = "Nothing was installed."

# The owner's form (W232 7). Texts: design 22.10, T62-T66, T69.
LINUX_GLIBC_TEXT = ("WhisperJAV 1.10.0 does not support this Linux. This is due to its system library glibc "
                    "{version}: WhisperJAV 1.10.0 needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36 "
                    "or newer). " + NOTHING)
LINUX_NO_GLIBC_TEXT = ("WhisperJAV 1.10.0 does not support this Linux. This is due to its system library: it has "
                       "no glibc, and WhisperJAV 1.10.0 needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12, "
                       "Fedora 36 or newer). " + NOTHING)
MACHINE_TEXT = ("WhisperJAV 1.10.0 does not support this computer ({system}, {machine}). This is due to its "
                "packages being made for x86-64 and ARM64 processors only. " + NOTHING)
INTEL_MAC_TEXT = ("WhisperJAV 1.10.0 does not support Intel Macs. This is due to PyTorch 2.11, which WhisperJAV "
                  "1.10.0 needs, not being made for Intel Macs. " + NOTHING)
ROSETTA_TEXT = ("WhisperJAV 1.10.0 does not support this Python on an Apple silicon Mac. This is due to the "
                "Python running under Rosetta (as an Intel program): WhisperJAV 1.10.0 needs an Apple silicon "
                "(arm64) Python 3.13, for example from Homebrew. " + NOTHING)
OLD_MACOS_TEXT = ("WhisperJAV 1.10.0 does not support macOS {version}. This is due to ONNX Runtime 1.29, which "
                  "WhisperJAV 1.10.0 needs: it requires macOS 14 (Sonoma) or newer. " + NOTHING)
WINDOWS_ARM_TEXT = ("WhisperJAV 1.10.0 does not support Windows on ARM. This is due to its packages being made "
                    "for x86-64 Windows only. " + NOTHING)
OLD_WINDOWS_TEXT = ("WhisperJAV 1.10.0 does not support this version of Windows. This is due to Python 3.13, "
                    "which WhisperJAV 1.10.0 needs: it requires Windows 10 or newer. " + NOTHING)
NOTEBOOK_TEXT = ("WhisperJAV 1.10.0 does not support Google Colab or Kaggle notebooks. This is due to 1.10.0 "
                 "being made and tested for installs on your own computer only. " + NOTHING)


def _version_tuple(text):
    parts = []
    for piece in (text or "").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def glibc_version():
    """'2.35' on a glibc Linux, None where there is no glibc (or it cannot be read)."""
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")  # "glibc 2.35"
    except (AttributeError, ValueError, OSError):
        return None
    if not value or not value.startswith("glibc "):
        return None
    return value.split(" ", 1)[1]


def windows_native_arm():
    """True on Windows on ARM, also for an x86-64 Python under emulation (which reports AMD64)."""
    if platform.machine().upper() == "ARM64":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        fn = kernel32.IsWow64Process2  # Windows 10 1709 and newer
    except (AttributeError, OSError, ImportError):
        return False
    process_machine = wintypes.USHORT()
    native_machine = wintypes.USHORT()
    try:
        ok = fn(kernel32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine))
    except OSError:
        return False
    return bool(ok) and native_machine.value == 0xAA64  # IMAGE_FILE_MACHINE_ARM64


def mac_under_rosetta():
    """True when this Python is an Intel program running on an Apple silicon Mac."""
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True, timeout=10).stdout
        return out.strip() == "1"
    except Exception:  # noqa: BLE001 -- no sysctl answer means not translated
        return False


def hosted_notebook():
    """True in Google Colab or a Kaggle notebook."""
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def refusal(system=None, machine=None, glibc=None, mac_version=None, windows_major=None,
            windows_arm=None, rosetta=None, notebook=None):
    """The refusal text for this computer, or "" when 1.10 supports it.

    Every argument defaults to this computer's own value; tests pass them.
    """
    system = system if system is not None else sys.platform
    machine = (machine if machine is not None else platform.machine()).lower()
    if notebook if notebook is not None else hosted_notebook():
        return NOTEBOOK_TEXT
    if system == "win32":
        if windows_arm if windows_arm is not None else windows_native_arm():
            return WINDOWS_ARM_TEXT
        major = windows_major if windows_major is not None else sys.getwindowsversion().major
        if major < MIN_WINDOWS_MAJOR:
            return OLD_WINDOWS_TEXT
        if machine not in ("amd64", "x86_64"):
            return MACHINE_TEXT.format(system="Windows", machine=machine)
        return ""
    if system.startswith("linux"):
        if machine not in ("x86_64", "aarch64", "arm64"):
            return MACHINE_TEXT.format(system="Linux", machine=machine)
        version = glibc if glibc is not None else glibc_version()
        if not version:
            return LINUX_NO_GLIBC_TEXT
        if _version_tuple(version) < MIN_GLIBC:
            return LINUX_GLIBC_TEXT.format(version=version)
        return ""
    if system == "darwin":
        if machine != "arm64":
            if rosetta if rosetta is not None else mac_under_rosetta():
                return ROSETTA_TEXT
            return INTEL_MAC_TEXT
        version = mac_version if mac_version is not None else platform.mac_ver()[0]
        if version and _version_tuple(version) < MIN_MACOS:
            return OLD_MACOS_TEXT.format(version=version)
        return ""
    return MACHINE_TEXT.format(system=system, machine=machine)


def main():
    text = refusal()
    if text:
        print(text)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
