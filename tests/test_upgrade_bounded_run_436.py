"""#436 (1.9.3 addendum): "whisperjav-upgrade hangs in the compatibility check". subprocess.run(capture_output=True,
timeout=...) waits on Windows for pipes a grandchild keeps open (reproduced: a 2 s timeout returned after 20 s).
upgrade._run_bounded writes to temporary files and kills the process tree, so the timeout holds.
"""
import subprocess
import sys
import time

import pytest

from whisperjav.upgrade import _run_bounded

GRANDCHILD = ("import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']); "
              "time.sleep(20)")


def test_timeout_holds_when_a_grandchild_keeps_running():
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded([sys.executable, "-c", GRANDCHILD], timeout=2)
    assert time.monotonic() - t0 < 12          # the plain subprocess.run took 20 s here


def test_output_and_return_code_are_kept():
    r = _run_bounded([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
                     timeout=30)
    assert r.returncode == 3 and r.stdout.strip() == "out" and r.stderr.strip() == "err"
