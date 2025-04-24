#!/usr/bin/env python3
"""
Progress tracking utilities for Moriarty Pipeline CLI.

This module provides functions for displaying progress bars and stage information
in the command-line interface.
"""

import sys
import time
import threading
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Iterator
from enum import Enum
from tqdm import tqdm
from tqdm.rich import tqdm as rich_tqdm
from pathlib import Path
from contextlib import contextmanager
import os

# Check if we can use colorama for colored output
try:
    from colorama import init, Fore, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ANSI color codes
class Color:
    RESET = Fore.RESET if COLORAMA_AVAILABLE else ""
    BOLD = Fore.RESET + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    RED = Fore.RED if COLORAMA_AVAILABLE else ""
    GREEN = Fore.GREEN if COLORAMA_AVAILABLE else ""
    YELLOW = Fore.YELLOW if COLORAMA_AVAILABLE else ""
    BLUE = Fore.BLUE if COLORAMA_AVAILABLE else ""
    MAGENTA = Fore.MAGENTA if COLORAMA_AVAILABLE else ""
    CYAN = Fore.CYAN if COLORAMA_AVAILABLE else ""
    LIGHT_GRAY = Fore.LIGHTBLACK_EX if COLORAMA_AVAILABLE else ""
    DARK_GRAY = Fore.DIM if COLORAMA_AVAILABLE else ""
    LIGHT_RED = Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    LIGHT_GREEN = Fore.GREEN + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    LIGHT_YELLOW = Fore.YELLOW + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    LIGHT_BLUE = Fore.BLUE + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    LIGHT_MAGENTA = Fore.MAGENTA + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    LIGHT_CYAN = Fore.CYAN + Style.BRIGHT if COLORAMA_AVAILABLE else ""
    WHITE = Fore.WHITE if COLORAMA_AVAILABLE else ""

class StageStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class PipelineProgress:
    """Manages progress reporting for pipeline stages."""
    
    def __init__(self, total_stages: int = 0, use_rich: bool = True, disable: bool = False):
        """
        Initialize progress tracking.
        
        Args:
            total_stages: Total number of pipeline stages
            use_rich: Whether to use rich formatting
            disable: Whether to disable progress displays
        """
        self.total_stages = total_stages
        self.current_stage = 0
        self.use_rich = use_rich
        self.disable = disable
        
        # Current active progress bars
        self.active_bars: Dict[str, Any] = {}
        
        # Stage information
        self.stages: List[Dict[str, Any]] = []
        for i in range(total_stages):
            self.stages.append({
                "index": i,
                "name": f"Stage {i+1}",
                "status": StageStatus.PENDING,
                "progress": 0.0,
                "start_time": None,
                "end_time": None,
                "details": {}
            })
            
        # Stage started time
        self.start_time = time.time()
    
    def start_stage(self, name: str, description: str = "", index: Optional[int] = None) -> None:
        """
        Start a new pipeline stage.
        
        Args:
            name: Stage name
            description: Stage description
            index: Stage index, defaults to incrementing the current stage
        """
        if index is None:
            index = self.current_stage
            self.current_stage += 1
        
        if index >= len(self.stages):
            # Add new stages if needed
            while index >= len(self.stages):
                self.stages.append({
                    "index": len(self.stages),
                    "name": f"Stage {len(self.stages)+1}",
                    "status": StageStatus.PENDING,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "details": {}
                })
            self.total_stages = len(self.stages)
        
        # Update stage info
        stage = self.stages[index]
        stage["name"] = name
        stage["description"] = description
        stage["status"] = StageStatus.RUNNING
        stage["start_time"] = time.time()
        
        # Print stage header
        if not self.disable:
            stage_header = f"\n{Color.BOLD}{Color.BLUE}[Stage {index+1}/{self.total_stages}] {name}{Color.RESET}"
            if description:
                stage_header += f": {description}"
            print(stage_header)
    
    def complete_stage(self, index: Optional[int] = None, status: StageStatus = StageStatus.COMPLETED, 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a stage as completed.
        
        Args:
            index: Stage index (defaults to last started stage)
            status: Completion status
            details: Additional stage details
        """
        if index is None:
            index = self.current_stage - 1
        
        if index < 0 or index >= len(self.stages):
            logger.warning(f"Invalid stage index: {index}")
            return
        
        stage = self.stages[index]
        stage["status"] = status
        stage["progress"] = 1.0
        stage["end_time"] = time.time()
        
        if details:
            stage["details"].update(details)
        
        # Print completion message
        if not self.disable:
            duration = stage["end_time"] - stage["start_time"]
            status_color = Color.GREEN if status == StageStatus.COMPLETED else \
                          Color.YELLOW if status == StageStatus.SKIPPED else \
                          Color.RED
            
            message = f"{Color.BOLD}{status_color}[{status.value}]{Color.RESET} "
            message += f"{stage['name']} completed in {duration:.2f}s"
            
            print(f"\n{message}")
            
            # Print any details as key-value pairs
            if details:
                for key, value in details.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        print(f"  {Color.DARK_GRAY}{key}:{Color.RESET} {value}")
            
            print()  # Add blank line after stage completion
    
    def update_stage_progress(self, progress: float, index: Optional[int] = None, 
                            message: Optional[str] = None) -> None:
        """
        Update progress for a stage.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            index: Stage index (defaults to current stage)
            message: Progress message
        """
        if index is None:
            index = self.current_stage - 1
        
        if index < 0 or index >= len(self.stages):
            return
        
        self.stages[index]["progress"] = max(0.0, min(1.0, progress))
        
        if message and not self.disable:
            print(f"  {Color.DARK_GRAY}Progress:{Color.RESET} {message}")
    
    def create_progress_bar(self, name: str, total: int, desc: str = "", unit: str = "it",
                          color: Optional[str] = None) -> tqdm:
        """
        Create a new progress bar.
        
        Args:
            name: Unique identifier for the progress bar
            total: Total number of items
            desc: Description for the progress bar
            unit: Unit of items
            color: Color for the progress bar
            
        Returns:
            tqdm progress bar instance
        """
        if name in self.active_bars:
            self.active_bars[name].close()
        
        if self.disable:
            bar = tqdm(total=total, desc=desc, unit=unit, disable=True)
        elif self.use_rich:
            bar = rich_tqdm(total=total, desc=desc, unit=unit)
        else:
            # Regular tqdm with color
            if color:
                desc = f"{color}{desc}{Color.RESET}"
            bar = tqdm(total=total, desc=desc, unit=unit)
        
        self.active_bars[name] = bar
        return bar
    
    def close_progress_bar(self, name: str) -> None:
        """
        Close a progress bar.
        
        Args:
            name: Name of the progress bar to close
        """
        if name in self.active_bars:
            self.active_bars[name].close()
            del self.active_bars[name]
    
    def close_all_progress_bars(self) -> None:
        """Close all active progress bars."""
        for bar in self.active_bars.values():
            bar.close()
        self.active_bars.clear()
    
    def print_summary(self) -> None:
        """Print a summary of all stages."""
        if self.disable:
            return
            
        total_duration = time.time() - self.start_time
        completed_stages = sum(1 for stage in self.stages if stage["status"] == StageStatus.COMPLETED)
        failed_stages = sum(1 for stage in self.stages if stage["status"] == StageStatus.FAILED)
        
        print(f"\n{Color.BOLD}Pipeline Execution Summary{Color.RESET}")
        print(f"{'=' * 30}")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Stages completed: {completed_stages}/{self.total_stages}")
        
        if failed_stages > 0:
            print(f"{Color.RED}Failed stages: {failed_stages}{Color.RESET}")
        
        print(f"{'=' * 30}")
        
        # Print individual stage summary
        for i, stage in enumerate(self.stages):
            if stage["start_time"] is not None:
                duration = (stage["end_time"] or time.time()) - stage["start_time"]
                status_str = stage["status"].value
                
                status_color = Color.GREEN if stage["status"] == StageStatus.COMPLETED else \
                              Color.YELLOW if stage["status"] == StageStatus.SKIPPED else \
                              Color.RED if stage["status"] == StageStatus.FAILED else \
                              Color.BLUE
                
                print(f"{i+1}. {stage['name']}: {status_color}{status_str}{Color.RESET} ({duration:.2f}s)")
    
    def get_stage_info(self, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a stage.
        
        Args:
            index: Stage index (defaults to current stage)
            
        Returns:
            Stage information dictionary
        """
        if index is None:
            index = self.current_stage - 1
        
        if index < 0 or index >= len(self.stages):
            return {}
        
        return self.stages[index].copy()
    
    def get_all_stages(self) -> List[Dict[str, Any]]:
        """
        Get information about all stages.
        
        Returns:
            List of stage information dictionaries
        """
        return [stage.copy() for stage in self.stages]

def run_with_progress(items: Union[List[Any], Iterator[Any]], 
                     process_func: Callable[[Any], Any],
                     desc: str = "Processing", 
                     unit: str = "it",
                     color: Optional[str] = None,
                     progress: Optional[PipelineProgress] = None) -> List[Any]:
    """
    Run a function over a list of items with a progress bar.
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        desc: Description for the progress bar
        unit: Unit name for the progress bar
        color: ANSI color for the progress bar
        progress: PipelineProgress instance (creates a new one if None)
        
    Returns:
        List of results from processing each item
    """
    # Use provided progress tracker or create a temporary one
    temp_progress = progress is None
    if temp_progress:
        progress = PipelineProgress(total_stages=1, use_rich=False)
    
    # Get total if possible
    total = len(items) if hasattr(items, "__len__") else None
    
    # Create progress bar
    bar_name = f"progress_{desc}"
    bar = progress.create_progress_bar(bar_name, total, desc=desc, unit=unit, color=color)
    
    results = []
    
    try:
        for item in items:
            result = process_func(item)
            results.append(result)
            bar.update(1)
    finally:
        progress.close_progress_bar(bar_name)
        
        # Clean up temporary progress object
        if temp_progress:
            progress.close_all_progress_bars()
    
    return results

def spinner_task(message: str, stop_event: threading.Event, delay: float = 0.1) -> None:
    """
    Display a spinner animation with a message.
    
    Args:
        message: Message to display
        stop_event: Event to signal when to stop the spinner
        delay: Delay between spinner updates
    """
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    
    while not stop_event.is_set():
        char = spinner_chars[i % len(spinner_chars)]
        sys.stdout.write(f"\r{Color.BLUE}{char}{Color.RESET} {message}")
        sys.stdout.flush()
        time.sleep(delay)
        i += 1
    
    # Clear the line when done
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()

class Spinner:
    """Context manager for displaying a spinner during a long-running operation."""
    
    def __init__(self, message: str):
        """
        Initialize the spinner.
        
        Args:
            message: Message to display with the spinner
        """
        self.message = message
        self.stop_event = threading.Event()
        self.spinner_thread = None
    
    def __enter__(self):
        """Start the spinner."""
        self.stop_event.clear()
        self.spinner_thread = threading.Thread(
            target=spinner_task, 
            args=(self.message, self.stop_event)
        )
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the spinner."""
        self.stop_event.set()
        if self.spinner_thread:
            self.spinner_thread.join(timeout=1.0)
        
        # Print completion or error message
        if exc_type is None:
            print(f"\r{Color.GREEN}✓{Color.RESET} {self.message}")
        else:
            print(f"\r{Color.RED}✗{Color.RESET} {self.message} - Error: {exc_val}")

# Global flag to enable/disable progress indicators
SHOW_PROGRESS = True

def set_show_progress(show: bool) -> None:
    """
    Set whether to show progress indicators.
    
    Args:
        show: True to show progress, False to hide
    """
    global SHOW_PROGRESS
    SHOW_PROGRESS = show

def colored(text: str, color: str, style: str = "") -> str:
    """
    Return colored text if colorama is available, otherwise return plain text.
    
    Args:
        text: Text to color
        color: Color to use
        style: Text style
        
    Returns:
        Colored text string
    """
    if COLORAMA_AVAILABLE:
        return f"{style}{color}{text}{Style.RESET_ALL}"
    return text

def progress_bar(iterable: Iterator, 
                 total: Optional[int] = None,
                 desc: Optional[str] = None,
                 unit: str = "it",
                 color: str = Color.BLUE) -> Iterator:
    """
    Create a progress bar for an iterable.
    
    Args:
        iterable: Iterable to track progress for
        total: Total number of items (optional)
        desc: Description of the operation
        unit: Unit name for items
        color: Color to use for the progress bar
        
    Returns:
        Iterable with progress tracking
    """
    if not SHOW_PROGRESS:
        return iterable
        
    if TQDM_AVAILABLE:
        if desc:
            desc = colored(desc, color)
        return tqdm(iterable, total=total, desc=desc, unit=unit)
    else:
        # Simple progress indicator using dots
        count = 0
        
        if desc:
            sys.stderr.write(f"\n{desc}: ")
        
        for item in iterable:
            yield item
            count += 1
            if count % 10 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()
                
        sys.stderr.write(f" Done ({count} {unit})\n")
        sys.stderr.flush()

class StatusReporter:
    """
    Reports status updates for long-running operations.
    Maintains a list of stages and their completion status.
    """
    
    def __init__(self, stages: List[str], show_progress: bool = True):
        """
        Initialize with a list of stage names.
        
        Args:
            stages: List of stage names
            show_progress: Whether to show progress
        """
        self.stages = stages
        self.status = {stage: "pending" for stage in stages}
        self.current_stage: Optional[str] = None
        self.show_progress = show_progress and SHOW_PROGRESS
        self._lock = threading.RLock()
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_active = False
        
    def start_stage(self, stage: str) -> None:
        """
        Mark a stage as started and show a spinner.
        
        Args:
            stage: Name of the stage to start
        """
        with self._lock:
            if stage not in self.stages:
                raise ValueError(f"Unknown stage: {stage}")
                
            self.current_stage = stage
            self.status[stage] = "running"
            
            if self.show_progress:
                self._print_status()
                self._start_spinner()
    
    def complete_stage(self, stage: str, success: bool = True) -> None:
        """
        Mark a stage as completed.
        
        Args:
            stage: Name of the stage to complete
            success: Whether the stage completed successfully
        """
        with self._lock:
            if stage not in self.stages:
                raise ValueError(f"Unknown stage: {stage}")
                
            self._stop_spinner()
            self.status[stage] = "success" if success else "failed"
            
            if self.show_progress:
                self._print_status()
    
    def fail_stage(self, stage: str) -> None:
        """
        Mark a stage as failed.
        
        Args:
            stage: Name of the stage to mark as failed
        """
        self.complete_stage(stage, success=False)
    
    def _print_status(self) -> None:
        """Print the current status of all stages."""
        if not self.show_progress:
            return
            
        status_symbols = {
            "pending": colored("◯", Color.DIM + Color.YELLOW),
            "running": colored("◉", Color.BRIGHT + Color.BLUE),
            "success": colored("✓", Color.BRIGHT + Color.GREEN),
            "failed": colored("✗", Color.BRIGHT + Color.RED)
        }
        
        # Clear the line
        sys.stderr.write("\r\033[K")
        
        # Print all stages with their status
        for stage in self.stages:
            symbol = status_symbols[self.status[stage]]
            name = stage
            
            # Highlight the current stage
            if stage == self.current_stage:
                name = colored(name, Color.BRIGHT + Color.CYAN)
            
            sys.stderr.write(f"{symbol} {name}  ")
        
        sys.stderr.flush()
    
    def _spinner_worker(self) -> None:
        """Background thread for the spinner animation."""
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        
        while self._spinner_active:
            with self._lock:
                if self.current_stage and self.status.get(self.current_stage) == "running":
                    # Clear the line
                    sys.stderr.write("\r\033[K")
                    
                    # Print all stages with their status
                    for stage in self.stages:
                        if stage == self.current_stage:
                            symbol = colored(spinner_chars[i], Color.BRIGHT + Color.CYAN)
                            name = colored(stage, Color.BRIGHT + Color.CYAN)
                        else:
                            symbol = colored("◯" if self.status[stage] == "pending" else 
                                           "✓" if self.status[stage] == "success" else
                                           "✗", 
                                         Color.DIM + Color.YELLOW if self.status[stage] == "pending" else
                                         Color.BRIGHT + Color.GREEN if self.status[stage] == "success" else
                                         Color.BRIGHT + Color.RED)
                            name = stage
                        
                        sys.stderr.write(f"{symbol} {name}  ")
                    
                    sys.stderr.flush()
            
            i = (i + 1) % len(spinner_chars)
            time.sleep(0.1)
    
    def _start_spinner(self) -> None:
        """Start the spinner animation."""
        if not self.show_progress:
            return
            
        with self._lock:
            if self._spinner_thread is None or not self._spinner_thread.is_alive():
                self._spinner_active = True
                self._spinner_thread = threading.Thread(target=self._spinner_worker, daemon=True)
                self._spinner_thread.start()
    
    def _stop_spinner(self) -> None:
        """Stop the spinner animation."""
        with self._lock:
            self._spinner_active = False
            
            if self._spinner_thread and self._spinner_thread.is_alive():
                self._spinner_thread.join(0.2)

@contextmanager
def stage_progress(stage: str, reporter: StatusReporter):
    """
    Context manager for tracking stage progress.
    
    Args:
        stage: Name of the stage
        reporter: StatusReporter instance
        
    Yields:
        None
    """
    reporter.start_stage(stage)
    try:
        yield
        reporter.complete_stage(stage, success=True)
    except Exception as e:
        reporter.fail_stage(stage)
        raise e

def log_step(message: str, step_type: str = "info") -> None:
    """
    Log a step in the process with appropriate formatting.
    
    Args:
        message: Message to log
        step_type: Type of step (info, success, warning, error)
    """
    if not SHOW_PROGRESS:
        return
        
    prefix_map = {
        "info": colored("ℹ", Color.BRIGHT + Color.BLUE),
        "success": colored("✓", Color.BRIGHT + Color.GREEN),
        "warning": colored("⚠", Color.BRIGHT + Color.YELLOW),
        "error": colored("✗", Color.BRIGHT + Color.RED),
        "start": colored("►", Color.BRIGHT + Color.CYAN),
        "end": colored("◼", Color.BRIGHT + Color.MAGENTA),
    }
    
    prefix = prefix_map.get(step_type, prefix_map["info"])
    sys.stderr.write(f"{prefix} {message}\n")
    sys.stderr.flush()

def spinner(desc: str = "Processing", color: str = Color.BLUE) -> Callable:
    """
    Decorator that adds a spinner while the function is running.
    
    Args:
        desc: Description to show
        color: Color to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            if not SHOW_PROGRESS:
                return func(*args, **kwargs)
                
            with spinning_indicator(desc, color):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def spinning_indicator(desc: str = "Processing", color: str = Color.BLUE) -> None:
    """
    Context manager that shows a spinner while the block executes.
    
    Args:
        desc: Description to show
        color: Color to use
        
    Yields:
        None
    """
    if not SHOW_PROGRESS:
        yield
        return
        
    if TQDM_AVAILABLE:
        with auto_tqdm(desc=colored(desc, color), total=0, bar_format='{desc}: {elapsed}') as pbar:
            yield
    else:
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        spinner_thread = None
        spinner_active = True
        
        def spin() -> None:
            nonlocal i
            while spinner_active:
                sys.stderr.write(f"\r{colored(desc, color)}: {spinner_chars[i]} ")
                sys.stderr.flush()
                i = (i + 1) % len(spinner_chars)
                time.sleep(0.1)
            sys.stderr.write(f"\r{colored(desc, color)}: Done!       \n")
            sys.stderr.flush()
        
        try:
            spinner_thread = threading.Thread(target=spin, daemon=True)
            spinner_thread.start()
            yield
        finally:
            spinner_active = False
            if spinner_thread:
                spinner_thread.join(0.2)
                
def format_time(seconds: float) -> str:
    """
    Format time in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

@contextmanager
def timed_operation(desc: str, color: str = Color.BLUE) -> None:
    """
    Context manager that times an operation and reports the elapsed time.
    
    Args:
        desc: Description of the operation
        color: Color to use
        
    Yields:
        None
    """
    if not SHOW_PROGRESS:
        yield
        return
        
    start_time = time.time()
    log_step(f"{desc}: Starting...", "start")
    
    try:
        yield
        elapsed = time.time() - start_time
        log_step(f"{desc}: Completed in {format_time(elapsed)}", "success")
    except Exception as e:
        elapsed = time.time() - start_time
        log_step(f"{desc}: Failed after {format_time(elapsed)} - {str(e)}", "error")
        raise 