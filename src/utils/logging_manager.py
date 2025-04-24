"""
Logging Manager for Moriarty Pipeline

This module provides utilities for setting up and managing logs
throughout the Moriarty application. It offers a unified logging system
with configurable verbosity levels and supports various output formats.
"""

import os
import sys
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal

# Define verbosity levels mapping
VERBOSITY_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Global logger registry to keep track of configured loggers
_logger_registry = {}

def setup_training_logs(log_dir: Union[str, Path], 
                       log_level: int = logging.INFO,
                       max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                       backup_count: int = 5) -> logging.Logger:
    """
    Set up logging for a long-running training process.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    # Create the log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("moriarty_training")
    logger.setLevel(log_level)

    # Remove any existing handlers (to avoid duplicates on reconfiguration)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler that logs all messages
    log_file = log_dir / "training.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)

    # Create separate file handler for errors
    error_log_file = log_dir / "training_errors.log"
    error_file_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_file_handler.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)

    # Create a training status file
    status_file = log_dir / "training_status.json"
    _create_status_file(status_file)

    logger.info(f"Logging configured for training. Log files in: {log_dir}")
    return logger

def setup_stream_logger(name: str = "moriarty", 
                       level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that logs to stderr.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers (to avoid duplicates)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger

def update_training_progress(log_dir: Union[str, Path], 
                           progress: Dict[str, Any]) -> bool:
    """
    Update the training progress status file.

    Args:
        log_dir: Directory containing log files
        progress: Dictionary with progress information

    Returns:
        True if successful, False otherwise
    """
    import json

    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    status_file = log_dir / "training_status.json"

    try:
        # Load existing status if any
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
        else:
            status = {"start_time": time.time()}

        # Update with new progress
        status.update({
            "last_update_time": time.time(),
            "progress": progress
        })

        # Write back to file
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        return True
    except Exception as e:
        logging.error(f"Error updating training progress: {e}")
        return False

def get_training_progress(log_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get the current training progress from the status file.

    Args:
        log_dir: Directory containing log files

    Returns:
        Progress dictionary or None if not available
    """
    import json

    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    status_file = log_dir / "training_status.json"

    try:
        if not status_file.exists():
            return None

        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error reading training progress: {e}")
        return None

def _create_status_file(status_file: Path) -> None:
    """
    Create initial training status file.

    Args:
        status_file: Path to status file
    """
    import json

    status = {
        "start_time": time.time(),
        "status": "initializing",
        "progress": {
            "epoch": 0,
            "step": 0,
            "total_epochs": 0,
            "total_steps": 0
        }
    }

    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logging.error(f"Error creating status file: {e}")

def log_exception(logger: logging.Logger, message: str = "An exception occurred") -> None:
    """
    Log an exception with traceback.

    Args:
        logger: Logger to use
        message: Message to prepend to the exception
    """
    import traceback
    logger.error(f"{message}: {traceback.format_exc()}")

def get_latest_log_entries(log_dir: Union[str, Path], n_lines: int = 20) -> Dict[str, str]:
    """
    Get the latest entries from log files.

    Args:
        log_dir: Directory containing log files
        n_lines: Number of lines to get from each log file

    Returns:
        Dictionary mapping log file names to their last n_lines
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    result = {}

    for log_file in log_dir.glob("*.log"):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                result[log_file.name] = "".join(lines[-n_lines:]) if lines else ""
        except Exception as e:
            logging.error(f"Error reading log file {log_file}: {e}")
            result[log_file.name] = f"Error reading log: {str(e)}"

    return result 

def configure_logging(
    app_name: str = "moriarty",
    verbosity: Union[str, int] = "info",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    capture_warnings: bool = True,
    propagate: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    module_levels: Optional[Dict[str, Union[str, int]]] = None
) -> logging.Logger:
    """
    Configure the unified logging system for the Moriarty application.

    This function serves as the single entry point for configuring logging
    throughout the application. It supports different verbosity levels,
    output formats, and destinations.

    Args:
        app_name: Name of the application/module
        verbosity: Logging level (can be string like "debug" or int like logging.DEBUG)
        log_file: Path to log file (if None, logs to console only)
        log_dir: Directory for log files (creates app_name.log inside this directory)
        log_format: Format string for log messages
        date_format: Format string for timestamps
        capture_warnings: Whether to capture warnings from the warnings module
        propagate: Whether to propagate logs to parent loggers
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        module_levels: Dict mapping module names to specific log levels

    Returns:
        Configured logger
    """
    # Convert string verbosity to int if needed
    if isinstance(verbosity, str):
        level = VERBOSITY_LEVELS.get(verbosity.lower(), logging.INFO)
    else:
        level = verbosity

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Default level for root

    # Clear any existing handlers on the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure the app logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file or log_dir:
        if log_dir:
            if isinstance(log_dir, str):
                log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{app_name}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Add separate error file handler
        error_log_file = Path(log_file).with_name(f"{Path(log_file).stem}_errors.log")
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    # Configure specific module levels if provided
    if module_levels:
        for module_name, module_level in module_levels.items():
            if isinstance(module_level, str):
                module_level = VERBOSITY_LEVELS.get(module_level.lower(), logging.INFO)

            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)
            module_logger.propagate = True  # Allow propagation to app logger

    # Capture warnings if requested
    if capture_warnings:
        logging.captureWarnings(True)

    # Store in registry
    _logger_registry[app_name] = logger

    logger.debug(f"Logging configured for {app_name} at level {logging.getLevelName(level)}")
    return logger

def get_logger(name: str = "moriarty") -> logging.Logger:
    """
    Get a configured logger by name.

    If the logger hasn't been configured yet, it will be configured
    with default settings.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    if name in _logger_registry:
        return _logger_registry[name]

    # If not found, configure with defaults
    return configure_logging(app_name=name)

def set_verbosity(verbosity: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Change the verbosity level of a logger.

    Args:
        verbosity: New verbosity level (string or int)
        logger_name: Name of the logger to modify (None for all loggers)
    """
    # Convert string verbosity to int if needed
    if isinstance(verbosity, str):
        level = VERBOSITY_LEVELS.get(verbosity.lower(), logging.INFO)
    else:
        level = verbosity

    if logger_name:
        # Modify specific logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Also update its handlers
        for handler in logger.handlers:
            if not isinstance(handler, logging.handlers.RotatingFileHandler) or \
               not handler.name == "error_file_handler":
                handler.setLevel(level)
    else:
        # Modify all registered loggers
        for name, logger in _logger_registry.items():
            logger.setLevel(level)

            # Update handlers except error handlers
            for handler in logger.handlers:
                if not isinstance(handler, logging.handlers.RotatingFileHandler) or \
                   not handler.name == "error_file_handler":
                    handler.setLevel(level)
