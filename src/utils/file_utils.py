"""
File and Directory Utilities for Moriarty Pipeline

This module provides utility functions for file and directory operations
needed by the Moriarty pipeline orchestrator.
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path (string or Path)
        
    Returns:
        Path object for the directory
    """
    if isinstance(directory, str):
        directory = Path(directory)
        
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_json(data: Any, filepath: Union[str, Path], pretty: bool = True) -> Path:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save the JSON file
        pretty: Whether to format with indentation
        
    Returns:
        Path object for the saved file
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
        
    # Ensure parent directory exists
    ensure_dir_exists(filepath.parent)
    
    try:
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
        return filepath
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise

def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
        
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise

def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        List of Path objects for matching files
    """
    if isinstance(directory, str):
        directory = Path(directory)
        
    try:
        return list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Error listing files in {directory} with pattern {pattern}: {e}")
        return []

def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing destination
        
    Returns:
        True if successful, False otherwise
    """
    if isinstance(src, str):
        src = Path(src)
    if isinstance(dst, str):
        dst = Path(dst)
        
    try:
        if dst.exists() and not overwrite:
            logger.warning(f"Destination file {dst} exists and overwrite=False")
            return False
            
        # Ensure destination directory exists
        ensure_dir_exists(dst.parent)
        
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False

def find_latest_file(directory: Union[str, Path], pattern: str = "*") -> Path:
    """
    Find the most recently modified file in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        Path to the most recent file, or None if no files found
    """
    files = list_files(directory, pattern)
    
    if not files:
        return None
        
    return max(files, key=lambda p: p.stat().st_mtime)

def get_file_size(filepath: Union[str, Path], human_readable: bool = False) -> Union[int, str]:
    """
    Get the size of a file.
    
    Args:
        filepath: Path to the file
        human_readable: Whether to return a human-readable size (e.g., "3.5 MB")
        
    Returns:
        Size in bytes, or human-readable string if requested
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
        
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    size_bytes = filepath.stat().st_size
    
    if not human_readable:
        return size_bytes
        
    # Convert to human-readable format
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0 or unit == "TB":
            break
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} {unit}" 