#!/usr/bin/env python3
"""
Utility functions for the Moriarty CLI.

This module provides utilities for progress tracking, status reporting,
and other common operations used across the CLI commands.
"""

import os
import sys
import logging
import traceback
import datetime
import subprocess
from typing import List, Dict, Any, Optional, Callable, Union, Iterator, Tuple
from pathlib import Path
from tqdm import tqdm
import signal
import tempfile
from colorama import Fore, Style, init

# Initialize colorama
init()

# Set up logger
logger = logging.getLogger("moriarty-cli")

class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that writes to tqdm progress bar without interrupting it.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_console_logging(level: str = 'info') -> None:
    """
    Setup console logging with the specified level.
    
    Args:
        level: Logging level (debug, info, warning, error)
    """
    # Map string levels to logging constants
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }
    numeric_level = level_map.get(level.lower(), logging.INFO)
    
    # Create console handler with tqdm-compatible output
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Get root logger and add handler
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the new handler
    root_logger.addHandler(console_handler)
    
    logger.debug(f"Console logging set to level: {level}")

def setup_file_logging(log_file_path: str, level: str = 'info') -> None:
    """
    Setup file logging with the specified level.
    
    Args:
        log_file_path: Path to the log file
        level: Logging level (debug, info, warning, error)
    """
    if not log_file_path:
        return
    
    # Map string levels to logging constants
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }
    numeric_level = level_map.get(level.lower(), logging.INFO)
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"File logging set to level: {level}, file: {log_file_path}")

def handle_keyboard_interrupt(signal, frame):
    """Handle keyboard interrupt (Ctrl+C) by gracefully exiting."""
    print("\nOperation interrupted by user. Exiting...")
    sys.exit(1)

def find_video_files(directory: str, pattern: str = '*.mp4') -> List[str]:
    """
    Find video files in a directory matching the specified pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match files
        
    Returns:
        List of video file paths
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).match(pattern):
                video_files.append(os.path.join(root, file))
    
    logger.debug(f"Found {len(video_files)} video files in {directory}")
    return video_files

def ensure_output_directory(directory: str) -> str:
    """
    Ensure that the output directory exists.
    
    If the directory doesn't exist, it will be created.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to the directory
    """
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {abs_directory}")
    return abs_directory

def create_temp_directory(base_dir: Optional[str] = None) -> str:
    """
    Create a temporary directory for processing.
    
    Args:
        base_dir: Base directory for temp directory creation
        
    Returns:
        Path to the created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="moriarty_", dir=base_dir)
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir

def clean_temp_directory(temp_dir: str) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Temporary directory to clean up
    """
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned temporary directory: {temp_dir}")

def create_progress_bar(iterable: Union[List, Iterator], desc: str, total: Optional[int] = None) -> tqdm:
    """
    Create a tqdm progress bar for iterables.
    
    Args:
        iterable: Iterable to track progress for
        desc: Description for the progress bar
        total: Total number of items (optional)
        
    Returns:
        tqdm progress bar
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit="file",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

def format_time_elapsed(start_time: datetime.datetime) -> str:
    """
    Format elapsed time since start_time.
    
    Args:
        start_time: Start time
        
    Returns:
        Formatted time string (HH:MM:SS)
    """
    elapsed = datetime.datetime.now() - start_time
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def print_summary(results: Dict[str, Any], start_time: datetime.datetime) -> None:
    """
    Print a summary of processing results.
    
    Args:
        results: Dictionary of results
        start_time: Start time of processing
    """
    total_elapsed = format_time_elapsed(start_time)
    
    print("\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print("="*60)
    print(f"Total elapsed time: {total_elapsed}")
    
    if 'processed' in results:
        print(f"Files processed: {results['processed']}")
    
    if 'success' in results:
        print(f"Successful: {results['success']}")
    
    if 'failures' in results:
        print(f"Failed: {len(results.get('failures', []))}")
        
        if results.get('failures'):
            print("\nFailed files:")
            for failure in results.get('failures', []):
                print(f"  - {failure['file']}: {failure['error']}")
    
    if 'warnings' in results and results['warnings']:
        print(f"\nWarnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    print("="*60)

def safe_execute(func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely execute a function with exception handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (success flag, result or None, error message or None)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        logger.debug(tb)
        return False, None, str(e)

# Set up signal handlers for graceful shutdown
signal.signal(signal.SIGINT, handle_keyboard_interrupt)

class ProgressTracker:
    """Progress tracker with tqdm integration and status reporting."""
    
    def __init__(self, total: int, desc: str = None, unit: str = "it"):
        """
        Initialize a progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description to display in the progress bar
            unit: Unit label for the items being processed
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        self.current = 0
        self.statuses = {}
        self.start_time = datetime.datetime.now()
        
    def update(self, increment: int = 1, status: str = None) -> None:
        """
        Update progress and optionally set status.
        
        Args:
            increment: Amount to increment progress by
            status: Optional status message to display
        """
        self.current += increment
        self.pbar.update(increment)
        
        if status:
            self.set_status(status)
    
    def set_status(self, status: str, item_id: str = None) -> None:
        """
        Set status for the current operation or a specific item.
        
        Args:
            status: Status message
            item_id: Optional identifier for the specific item
        """
        if item_id:
            self.statuses[item_id] = status
            message = f"{item_id}: {status}"
        else:
            message = status
            
        self.pbar.set_postfix_str(message)
        logger.info(message)
    
    def close(self) -> None:
        """Close the progress tracker and calculate summary stats."""
        self.pbar.close()
        duration = datetime.datetime.now() - self.start_time
        
        logger.info(f"Completed {self.current}/{self.total} items in {duration}")
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the tracked progress.
        
        Returns:
            Dictionary containing summary statistics
        """
        duration = datetime.datetime.now() - self.start_time
        return {
            "total": self.total,
            "completed": self.current,
            "duration_seconds": duration.total_seconds(),
            "items_per_second": self.current / duration.total_seconds() if duration.total_seconds() > 0 else 0,
            "statuses": self.statuses.copy(),
        }

def process_items_with_progress(
    items: List[Any],
    process_func: Callable[[Any], Any],
    desc: str = "Processing",
    unit: str = "it",
    error_handler: Callable[[Any, Exception], None] = None
) -> List[Any]:
    """
    Process a list of items with progress tracking.
    
    Args:
        items: List of items to process
        process_func: Function that processes each item
        desc: Description for the progress bar
        unit: Unit label for the progress bar
        error_handler: Optional function to handle errors during processing
        
    Returns:
        List of results from processing each item
    """
    results = []
    tracker = ProgressTracker(total=len(items), desc=desc, unit=unit)
    
    for item in items:
        try:
            result = process_func(item)
            results.append(result)
            tracker.update(1)
        except Exception as e:
            if error_handler:
                error_handler(item, e)
            else:
                logger.error(f"Error processing item {item}: {str(e)}")
            tracker.set_status(f"Error: {str(e)}", str(item))
    
    tracker.close()
    return results

def create_directory_if_not_exists(directory: str) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
        
    Returns:
        True if the directory was created or already exists, False on error
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False

def print_status(message: str, status: str = "info", item: str = None) -> None:
    """
    Print a formatted status message.
    
    Args:
        message: Message to print
        status: Status type (info, success, warning, error)
        item: Optional item identifier
    """
    status_colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "debug": Fore.CYAN,
    }
    
    color = status_colors.get(status.lower(), Fore.WHITE)
    prefix = f"[{item}] " if item else ""
    
    print(f"{color}{prefix}{message}{Style.RESET_ALL}")
    
    # Also log the message
    log_level = getattr(logging, status.upper(), logging.INFO)
    logger.log(log_level, f"{prefix}{message}")

def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file exists, False otherwise
    """
    if os.path.isfile(file_path):
        return True
    logger.error(f"File not found: {file_path}")
    return False

def validate_directory_exists(directory: str) -> bool:
    """
    Validate that a directory exists.
    
    Args:
        directory: Path to the directory to check
        
    Returns:
        True if the directory exists, False otherwise
    """
    if os.path.isdir(directory):
        return True
    logger.error(f"Directory not found: {directory}")
    return False

def get_file_list(directory: str, pattern: str = "*") -> List[str]:
    """
    Get a list of files in a directory matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files against
        
    Returns:
        List of file paths
    """
    import glob
    
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    file_pattern = os.path.join(directory, pattern)
    files = glob.glob(file_pattern)
    
    return [f for f in files if os.path.isfile(f)]

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 3m 45s")
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

class StatusReporter:
    """Status reporter for long-running operations."""
    
    def __init__(self, total_steps: int = None):
        """
        Initialize a status reporter.
        
        Args:
            total_steps: Optional total number of steps in the operation
        """
        self.start_time = datetime.datetime.now()
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step = None
        self.current_status = None
    
    def start_step(self, step_name: str) -> None:
        """
        Start a new step.
        
        Args:
            step_name: Name of the step
        """
        self.current_step = step_name
        self.current_status = "running"
        
        if self.total_steps:
            progress = f"[{self.completed_steps + 1}/{self.total_steps}] "
        else:
            progress = ""
            
        print_status(f"{progress}Starting: {step_name}...", "info")
        logger.info(f"Starting step: {step_name}")
    
    def complete_step(self, status: str = "success", message: str = None) -> None:
        """
        Mark the current step as complete.
        
        Args:
            status: Status of the completed step (success, warning, error)
            message: Optional status message
        """
        if not self.current_step:
            return
            
        self.completed_steps += 1
        self.current_status = status
        
        if not message:
            message = f"Completed: {self.current_step}"
            
        print_status(message, status)
        logger.info(f"Completed step: {self.current_step} - {status}")
        
        self.current_step = None
    
    def update_status(self, message: str, status: str = "info") -> None:
        """
        Update status without changing the current step.
        
        Args:
            message: Status message
            status: Status type (info, warning, error)
        """
        print_status(message, status)
        logger.info(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the operation.
        
        Returns:
            Dictionary containing operation summary
        """
        duration = datetime.datetime.now() - self.start_time
        
        return {
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "duration_seconds": duration.total_seconds(),
            "current_step": self.current_step,
            "current_status": self.current_status,
        }
        
    def finish(self) -> None:
        """Complete the operation and print a summary."""
        duration = datetime.datetime.now() - self.start_time
        formatted_duration = format_duration(duration.total_seconds())
        
        if self.total_steps:
            message = f"Operation completed: {self.completed_steps}/{self.total_steps} steps in {formatted_duration}"
        else:
            message = f"Operation completed in {formatted_duration}"
            
        print_status(message, "success")
        logger.info(message) 