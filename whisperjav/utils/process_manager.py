"""
Process management utilities for WhisperJAV.

Provides cross-platform process tree termination to ensure
child processes (including GPU workers) are properly cleaned up
when the user cancels an operation.

v1.7.4+: Fixes orphaned GPU worker processes on cancellation.
"""

import os
import logging
from typing import List, Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("whisperjav")


def get_process_tree(pid: int) -> List[int]:
    """
    Get all descendant PIDs of a process (children, grandchildren, etc.).

    Args:
        pid: Parent process ID

    Returns:
        List of descendant PIDs (does NOT include the parent itself)
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available, cannot enumerate process tree")
        return []

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return [child.pid for child in children]
    except psutil.NoSuchProcess:
        return []
    except Exception as e:
        logger.warning(f"Failed to enumerate process tree for PID {pid}: {e}")
        return []


def terminate_process_tree(
    pid: int,
    timeout: float = 5.0,
    include_parent: bool = True
) -> Dict[str, Any]:
    """
    Terminate a process and all its descendants gracefully.

    Uses a two-phase approach:
    1. SIGTERM to all processes (graceful shutdown request)
    2. Wait for timeout
    3. SIGKILL to any survivors (force kill)

    This ensures GPU workers spawned by multiprocessing are properly
    terminated when the user cancels an operation.

    Args:
        pid: Root process ID to terminate
        timeout: Seconds to wait for graceful shutdown before force-killing
        include_parent: Whether to also terminate the parent process

    Returns:
        dict with termination results:
            {
                "success": bool,
                "terminated": [list of PIDs that were terminated gracefully],
                "killed": [list of PIDs that required force-kill],
                "already_dead": [list of PIDs that were already dead],
                "errors": [list of error messages]
            }
    """
    result = {
        "success": True,
        "terminated": [],
        "killed": [],
        "already_dead": [],
        "errors": []
    }

    if not PSUTIL_AVAILABLE:
        result["success"] = False
        result["errors"].append("psutil not available - falling back to basic termination")
        return result

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        result["already_dead"].append(pid)
        return result
    except psutil.AccessDenied as e:
        result["success"] = False
        result["errors"].append(f"Access denied to process {pid}: {e}")
        return result
    except Exception as e:
        result["success"] = False
        result["errors"].append(f"Cannot access process {pid}: {e}")
        return result

    # Collect all processes to terminate (children first, then parent)
    # Killing children first prevents orphan creation
    processes_to_kill = []

    try:
        children = parent.children(recursive=True)
        # Reverse order: deepest children first
        processes_to_kill.extend(reversed(children))
    except psutil.NoSuchProcess:
        pass  # Parent already dead
    except Exception as e:
        result["errors"].append(f"Failed to get children: {e}")

    if include_parent:
        processes_to_kill.append(parent)

    if not processes_to_kill:
        return result

    pids_to_kill = [p.pid for p in processes_to_kill]
    logger.debug(f"Terminating process tree: {pids_to_kill}")

    # Phase 1: Send SIGTERM to all (graceful termination)
    alive_processes = []
    for proc in processes_to_kill:
        try:
            proc.terminate()
            alive_processes.append(proc)
            result["terminated"].append(proc.pid)
        except psutil.NoSuchProcess:
            result["already_dead"].append(proc.pid)
        except psutil.AccessDenied:
            result["errors"].append(f"Access denied terminating PID {proc.pid}")
        except Exception as e:
            result["errors"].append(f"Failed to terminate PID {proc.pid}: {e}")

    if not alive_processes:
        return result

    # Phase 2: Wait for graceful shutdown
    gone, still_alive = psutil.wait_procs(alive_processes, timeout=timeout)

    # Update terminated list to only include those that actually died gracefully
    gracefully_terminated = [p.pid for p in gone]

    # Phase 3: Force-kill survivors
    for proc in still_alive:
        try:
            logger.warning(f"Force-killing PID {proc.pid} (did not terminate gracefully)")
            proc.kill()
            result["killed"].append(proc.pid)
            # Remove from terminated since it required force-kill
            if proc.pid in result["terminated"]:
                result["terminated"].remove(proc.pid)
        except psutil.NoSuchProcess:
            result["already_dead"].append(proc.pid)
        except psutil.AccessDenied:
            result["errors"].append(f"Access denied killing PID {proc.pid}")
            result["success"] = False
        except Exception as e:
            result["errors"].append(f"Failed to kill PID {proc.pid}: {e}")
            result["success"] = False

    # Wait briefly for kills to complete
    if still_alive:
        psutil.wait_procs(still_alive, timeout=2.0)

    total_killed = len(result["terminated"]) + len(result["killed"])
    logger.debug(
        f"Process tree termination complete: "
        f"{len(result['terminated'])} terminated gracefully, "
        f"{len(result['killed'])} force-killed, "
        f"{len(result['already_dead'])} already dead"
    )

    return result


def kill_process_tree(pid: int, include_parent: bool = True) -> Dict[str, Any]:
    """
    Immediately force-kill a process and all its descendants.

    No graceful shutdown - sends SIGKILL directly.
    Use this when immediate termination is required.

    Args:
        pid: Root process ID to kill
        include_parent: Whether to also kill the parent process

    Returns:
        dict with kill results (same format as terminate_process_tree)
    """
    result = {
        "success": True,
        "terminated": [],
        "killed": [],
        "already_dead": [],
        "errors": []
    }

    if not PSUTIL_AVAILABLE:
        result["success"] = False
        result["errors"].append("psutil not available")
        return result

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        result["already_dead"].append(pid)
        return result
    except Exception as e:
        result["success"] = False
        result["errors"].append(f"Cannot access process {pid}: {e}")
        return result

    # Collect all processes (children first, then parent)
    processes_to_kill = []

    try:
        children = parent.children(recursive=True)
        # Reverse order: deepest children first
        processes_to_kill.extend(reversed(children))
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        result["errors"].append(f"Failed to get children: {e}")

    if include_parent:
        processes_to_kill.append(parent)

    logger.debug(f"Force-killing process tree: {[p.pid for p in processes_to_kill]}")

    # Kill all immediately
    for proc in processes_to_kill:
        try:
            proc.kill()
            result["killed"].append(proc.pid)
        except psutil.NoSuchProcess:
            result["already_dead"].append(proc.pid)
        except psutil.AccessDenied:
            result["errors"].append(f"Access denied killing PID {proc.pid}")
            result["success"] = False
        except Exception as e:
            result["errors"].append(f"Failed to kill PID {proc.pid}: {e}")
            result["success"] = False

    # Wait briefly for kills to complete
    if processes_to_kill:
        psutil.wait_procs(processes_to_kill, timeout=2.0)

    return result


def is_process_alive(pid: int) -> bool:
    """
    Check if a process is still running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running, False otherwise
    """
    if not PSUTIL_AVAILABLE:
        # Fallback: try os.kill with signal 0 (doesn't actually kill)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return False


def get_process_info(pid: int) -> Dict[str, Any]:
    """
    Get information about a process and its children.

    Useful for debugging process management issues.

    Args:
        pid: Process ID to inspect

    Returns:
        dict with process information
    """
    if not PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}

    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)

        return {
            "pid": pid,
            "name": proc.name(),
            "status": proc.status(),
            "cmdline": proc.cmdline()[:3] if proc.cmdline() else [],  # First 3 args only
            "children_count": len(children),
            "children": [
                {
                    "pid": child.pid,
                    "name": child.name(),
                    "status": child.status()
                }
                for child in children
            ]
        }
    except psutil.NoSuchProcess:
        return {"error": f"Process {pid} not found"}
    except Exception as e:
        return {"error": str(e)}
