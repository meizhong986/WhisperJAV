"""1.10.2: `whisperjav.exe` runs in UTF-8 mode, as `python -m whisperjav.main` and the GUI's runs already did.

1.10.1 install test, finding 4: with its output going to a pipe or file, `whisperjav.exe` started the ensemble pass
worker without UTF-8 mode; the worker wrote cp1252 and logged errors on "→". The command imports whisperjav.main
(so __name__ is not '__main__'); the guard now also looks at sys.argv[0]. Checked by a run on 2026-11-12 (a startup
probe showed utf8_mode=1 in the relaunched process). Here the relaunch itself is replaced by a stand-in.
"""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the UTF-8 relaunch exists on Windows only")

PROBE = (
    "import sys\n"
    "sys.argv = [{argv0!r}, '--version']\n"
    "import whisperjav.utils.console as c\n"
    "def stand_in(module):\n"
    "    print('RELAUNCH', module)\n"
    "    sys.exit(0)\n"
    "c.relaunch_for_utf8 = stand_in\n"
    "import whisperjav.main\n"
    "print('NO RELAUNCH')\n"
)


def _run(argv0):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING", "_WHISPERJAV_UTF8")}
    r = subprocess.run([sys.executable, "-c", PROBE.format(argv0=argv0)], capture_output=True, text=True,
                       env=env, timeout=300)
    return r.stdout


@pytest.mark.parametrize("argv0", [r"C:\WJ\Scripts\whisperjav.exe", r"C:\WJ\Scripts\WhisperJAV.EXE"])
def test_the_whisperjav_command_relaunches_in_utf8_mode(argv0):
    assert "RELAUNCH whisperjav.main" in _run(argv0)


@pytest.mark.parametrize("argv0", [r"C:\WJ\Scripts\whisperjav-gui.exe", r"C:\WJ\Scripts\pytest.exe", "-c"])
def test_other_importers_do_not_relaunch(argv0):
    assert "NO RELAUNCH" in _run(argv0)
