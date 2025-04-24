"""
Checkpoint Validator for Moriarty Pipeline

This module provides utilities for validating and managing model checkpoints,
particularly for the long-running LLM training pipeline.
"""

import os
import logging
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def validate_checkpoints(checkpoint_dir: Union[str, Path]) -> List[str]:
    """
    Validate checkpoint files and return a list of valid checkpoint paths.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of paths to valid checkpoints, sorted by newest first
    """
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
        
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []
    
    # Get all potential checkpoint directories
    checkpoints = []
    
    # First look for HuggingFace format checkpoints (directories like "checkpoint-123")
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    for item in checkpoint_dir.glob("checkpoint-*"):
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step = int(match.group(1))
                checkpoints.append({
                    "path": str(item),
                    "step": step,
                    "time": item.stat().st_mtime,
                    "valid": _is_valid_hf_checkpoint(item)
                })
    
    # Filter to only valid checkpoints
    valid_checkpoints = [c["path"] for c in checkpoints if c["valid"]]
    
    # Sort by recency (newest first)
    valid_checkpoints.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    
    if not valid_checkpoints:
        logger.warning(f"No valid checkpoints found in {checkpoint_dir}")
    else:
        logger.info(f"Found {len(valid_checkpoints)} valid checkpoints in {checkpoint_dir}")
        
    return valid_checkpoints

def _is_valid_hf_checkpoint(checkpoint_path: Path) -> bool:
    """
    Check if a checkpoint directory contains a valid HuggingFace model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        True if valid, False otherwise
    """
    # Check for essential files that indicate a complete checkpoint
    required_files = ["config.json", "pytorch_model.bin"]
    
    # For newer HF checkpoints with sharded weights
    alternative_required = ["config.json"]
    has_sharded = False
    
    for item in checkpoint_path.glob("pytorch_model-*.bin"):
        has_sharded = True
        break
    
    # Check if all required files exist
    if has_sharded:
        for file in alternative_required:
            if not (checkpoint_path / file).exists():
                return False
        return True
    else:
        for file in required_files:
            if not (checkpoint_path / file).exists():
                return False
        return True

def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get information about a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Dictionary with checkpoint information or None if invalid
    """
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
        
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint {checkpoint_path} does not exist")
        return None
    
    try:
        # Extract step number from the checkpoint name
        step = 0
        match = re.search(r"checkpoint-(\d+)", str(checkpoint_path))
        if match:
            step = int(match.group(1))
        
        # Get metadata from the trainer_state.json if it exists
        trainer_state_path = checkpoint_path / "trainer_state.json"
        trainer_state = None
        
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
        
        # Get basic file information
        stat = checkpoint_path.stat()
        
        info = {
            "path": str(checkpoint_path),
            "step": step,
            "creation_time": stat.st_ctime,
            "modification_time": stat.st_mtime,
            "size_bytes": _get_dir_size(checkpoint_path),
            "trainer_state": trainer_state
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting checkpoint info for {checkpoint_path}: {e}")
        return None

def _get_dir_size(path: Path) -> int:
    """
    Get the total size of a directory.
    
    Args:
        path: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = Path(dirpath) / f
            if not fp.is_symlink():
                total_size += fp.stat().st_size
    return total_size

def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    """
    Find the latest valid checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint or None if no valid checkpoints found
    """
    valid_checkpoints = validate_checkpoints(checkpoint_dir)
    
    if not valid_checkpoints:
        return None
        
    return valid_checkpoints[0]  # Already sorted newest first

def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], keep_top_n: int = 3) -> List[str]:
    """
    Cleanup old checkpoints, keeping only the N most recent valid ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_top_n: Number of recent checkpoints to keep
        
    Returns:
        List of paths to checkpoints that were deleted
    """
    valid_checkpoints = validate_checkpoints(checkpoint_dir)
    
    if len(valid_checkpoints) <= keep_top_n:
        logger.info(f"Not enough checkpoints to clean up. Found {len(valid_checkpoints)}, keeping {keep_top_n}")
        return []
    
    # Keep the first keep_top_n checkpoints (already sorted by newest first)
    to_keep = valid_checkpoints[:keep_top_n]
    to_delete = valid_checkpoints[keep_top_n:]
    
    deleted = []
    for checkpoint in to_delete:
        try:
            import shutil
            shutil.rmtree(checkpoint)
            deleted.append(checkpoint)
            logger.info(f"Deleted old checkpoint: {checkpoint}")
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint}: {e}")
    
    return deleted 