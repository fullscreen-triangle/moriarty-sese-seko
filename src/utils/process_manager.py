"""
Process Management Utilities for Moriarty Pipeline

This module provides utilities for managing processes, particularly
for long-running background tasks like LLM training.
"""

import os
import sys
import signal
import time
import logging
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def detach_process() -> int:
    """
    Fork the current process to create a detached background process.
    
    Returns:
        For parent process: PID of the child process
        For child process: 0
    """
    # Check if running on a non-Unix system (e.g., Windows)
    if not hasattr(os, 'fork'):
        logger.warning("Fork is not available on this platform. Using alternative method.")
        return _detach_process_no_fork()
    
    try:
        # First fork
        pid = os.fork()
        if pid > 0:
            # This is the parent process, return child's PID
            return pid
        
        # Decouple from parent environment
        os.setsid()
        os.umask(0)
        
        # Second fork to prevent zombie processes
        pid = os.fork()
        if pid > 0:
            # Exit the first child
            os._exit(0)
            
        # Redirect standard file descriptors
        sys.stdin.close()
        sys.stdout.flush()
        sys.stderr.flush()
        
        with open(os.devnull, 'r') as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
            
        # Keep stdout and stderr for logging purposes
        # but redirect them to files if needed
        
        # We're now in the fully detached child process
        return 0
        
    except Exception as e:
        logger.error(f"Error in detach_process: {e}")
        return -1

def _detach_process_no_fork() -> int:
    """
    Alternative method for platforms without fork support (e.g., Windows).
    Launches a new process and exits the current one.
    
    Returns:
        PID of the new process or -1 on error
    """
    try:
        # Create a file to communicate the PID back
        with tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.pid') as pid_file:
            pid_file_path = pid_file.name
        
        # Get the current script path and arguments
        script_path = sys.argv[0]
        script_args = sys.argv[1:]
        
        # Launch the new process that will write its PID to the file
        cmd = [sys.executable, script_path] + script_args + ["--detached", f"--pid-file={pid_file_path}"]
        
        # Use subprocess to create the new process
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.DETACHED_PROCESS if hasattr(subprocess, 'DETACHED_PROCESS') else 0
        )
        
        # Wait a moment for the process to start
        time.sleep(1)
        
        # Try to read the PID from the file
        try:
            with open(pid_file_path, 'r') as f:
                child_pid = int(f.read().strip())
            
            # Cleanup the temporary file
            os.unlink(pid_file_path)
            return child_pid
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error reading PID file: {e}")
            return -1
            
    except Exception as e:
        logger.error(f"Error in _detach_process_no_fork: {e}")
        return -1

def is_process_running(pid: int) -> bool:
    """
    Check if a process with the given PID is running.
    
    Args:
        pid: Process ID to check
        
    Returns:
        True if the process is running, False otherwise
    """
    if pid <= 0:
        return False
        
    try:
        # Send signal 0 to the process - doesn't actually send a signal
        # but performs error checking
        os.kill(pid, 0)
        return True
    except OSError:
        # Process doesn't exist
        return False

def terminate_process(pid: int, timeout: int = 5) -> bool:
    """
    Terminate a process gracefully, then forcefully if needed.
    
    Args:
        pid: Process ID to terminate
        timeout: Time to wait for graceful termination before force-killing
        
    Returns:
        True if the process was terminated, False otherwise
    """
    if not is_process_running(pid):
        return True
        
    try:
        # Try to terminate gracefully first
        os.kill(pid, signal.SIGTERM)
        
        # Wait for the process to terminate
        for _ in range(timeout):
            if not is_process_running(pid):
                return True
            time.sleep(1)
            
        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError as e:
        logger.error(f"Error terminating process {pid}: {e}")
        return False

def save_process_info(pid: int, process_info: Dict[str, Any], info_file: str) -> bool:
    """
    Save process information to a file.
    
    Args:
        pid: Process ID
        process_info: Process information dictionary
        info_file: File to save the information to
        
    Returns:
        True if successful, False otherwise
    """
    data = {
        "pid": pid,
        "start_time": time.time(),
        "info": process_info
    }
    
    try:
        with open(info_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving process info: {e}")
        return False

def load_process_info(info_file: str) -> Optional[Dict[str, Any]]:
    """
    Load process information from a file.
    
    Args:
        info_file: File containing process information
        
    Returns:
        Process information dictionary or None if not found/readable
    """
    try:
        with open(info_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading process info: {e}")
        return None 