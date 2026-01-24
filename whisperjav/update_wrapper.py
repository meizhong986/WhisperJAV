#!/usr/bin/env python3
"""
WhisperJAV Update Wrapper

This script is spawned by the GUI to perform updates while the GUI is closed.
It handles the complete update lifecycle:

1. Wait for GUI process to exit (by PID)
2. Run the upgrade script
3. Relaunch the GUI
4. Log everything for debugging

Usage (spawned by GUI):
    pythonw.exe -m whisperjav.update_wrapper --pid <gui_pid> [--wheel-only]

The wrapper runs hidden by default. On failure, it can optionally show
a console window with error details.
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Determine installation directory
INSTALL_DIR = Path(sys.prefix)
LOG_FILE = INSTALL_DIR / "update.log"


def setup_logging():
    """Set up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def wait_for_process_exit(pid: int, timeout: int = 30) -> bool:
    """
    Wait for a process to exit.

    Args:
        pid: Process ID to wait for
        timeout: Maximum seconds to wait

    Returns:
        True if process exited, False if timeout
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Waiting for process {pid} to exit...")

    try:
        import psutil
        use_psutil = True
    except ImportError:
        use_psutil = False
        logger.warning("psutil not available, using polling method")

    start_time = time.time()

    while time.time() - start_time < timeout:
        if use_psutil:
            try:
                proc = psutil.Process(pid)
                if not proc.is_running():
                    logger.info(f"Process {pid} has exited")
                    return True
            except psutil.NoSuchProcess:
                logger.info(f"Process {pid} has exited")
                return True
        else:
            # Fallback: try to check if process exists on Windows
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                    capture_output=True,
                    text=True
                )
                if str(pid) not in result.stdout:
                    logger.info(f"Process {pid} has exited")
                    return True
            except Exception:
                pass

        time.sleep(0.5)

    logger.warning(f"Timeout waiting for process {pid}")
    return False


def run_upgrade(wheel_only: bool = False) -> tuple:
    """
    Run the upgrade script.

    Args:
        wheel_only: If True, only update the wheel (hot-patch mode)

    Returns:
        Tuple of (success, return_code, output)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting upgrade...")

    python_exe = INSTALL_DIR / "python.exe"
    if not python_exe.exists():
        python_exe = INSTALL_DIR / "python"

    # Build command
    cmd = [str(python_exe), "-m", "whisperjav.upgrade"]
    if wheel_only:
        cmd.append("--wheel-only")
    cmd.append("--yes")  # Non-interactive mode

    logger.info(f"Running: {' '.join(cmd)}")

    # Set UTF-8 encoding to prevent Unicode crashes on Windows cp1252 console
    # This is critical for upgrading from older versions that may use Unicode symbols
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=str(INSTALL_DIR),
            env=env
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"  {line}")

        success = result.returncode == 0
        if success:
            logger.info("Upgrade completed successfully")
        else:
            logger.error(f"Upgrade failed with code {result.returncode}")

        return success, result.returncode, result.stdout + result.stderr

    except subprocess.TimeoutExpired:
        logger.error("Upgrade timed out after 30 minutes")
        return False, -1, "Timeout"
    except Exception as e:
        logger.error(f"Upgrade error: {e}")
        return False, -1, str(e)


def relaunch_gui() -> bool:
    """
    Relaunch the WhisperJAV GUI.

    Returns:
        True if launch succeeded
    """
    logger = logging.getLogger(__name__)
    logger.info("Relaunching GUI...")

    # Prefer pythonw.exe for windowless launch
    pythonw = INSTALL_DIR / "pythonw.exe"
    if not pythonw.exists():
        pythonw = INSTALL_DIR / "python.exe"
    if not pythonw.exists():
        pythonw = INSTALL_DIR / "python"

    try:
        # Launch GUI as detached process
        if sys.platform == "win32":
            # Windows: use DETACHED_PROCESS
            subprocess.Popen(
                [str(pythonw), "-m", "whisperjav.webview_gui.main"],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                cwd=str(INSTALL_DIR),
                close_fds=True
            )
        else:
            # Unix: use start_new_session
            subprocess.Popen(
                [str(pythonw), "-m", "whisperjav.webview_gui.main"],
                start_new_session=True,
                cwd=str(INSTALL_DIR)
            )

        logger.info("GUI relaunched successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to relaunch GUI: {e}")
        return False


def show_error_dialog(message: str):
    """Show an error dialog to the user (Windows only)."""
    if sys.platform != "win32":
        return

    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            "WhisperJAV Update Error",
            0x10  # MB_ICONERROR
        )
    except Exception:
        pass  # Silently fail if we can't show dialog


def main():
    """Main entry point for the update wrapper."""
    parser = argparse.ArgumentParser(description="WhisperJAV Update Wrapper")
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="PID of the GUI process to wait for"
    )
    parser.add_argument(
        "--wheel-only",
        action="store_true",
        help="Hot-patch mode: only update whisperjav package"
    )
    parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Don't relaunch GUI after update"
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("WhisperJAV Update Wrapper")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Install directory: {INSTALL_DIR}")
    logger.info(f"GUI PID to wait for: {args.pid}")
    logger.info(f"Wheel-only mode: {args.wheel_only}")
    logger.info("=" * 60)

    # Step 1: Wait for GUI to exit
    if not wait_for_process_exit(args.pid, timeout=60):
        logger.warning("GUI didn't exit in time, proceeding anyway...")

    # Small delay to ensure file handles are released
    time.sleep(1)

    # Step 2: Run upgrade
    success, return_code, output = run_upgrade(wheel_only=args.wheel_only)

    if not success:
        error_msg = (
            f"WhisperJAV update failed (code {return_code}).\n\n"
            f"See log file for details:\n{LOG_FILE}\n\n"
            "You can try running the upgrade manually:\n"
            f"{INSTALL_DIR}\\python.exe -m whisperjav.upgrade"
        )
        logger.error(error_msg)
        show_error_dialog(error_msg)
        return 1

    # Step 3: Relaunch GUI
    if not args.no_relaunch:
        if not relaunch_gui():
            logger.warning("Failed to relaunch GUI, please start manually")

    logger.info("Update wrapper finished successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
