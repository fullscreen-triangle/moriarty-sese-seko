"""
Progress and status reporting for the Moriarty CLI.

This module provides utilities for showing progress bars and status updates
during video analysis and knowledge distillation operations.
"""

import sys
import time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
import logging
from tqdm import tqdm

class StageStatus(Enum):
    """Status of a pipeline stage."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

class StatusReporter:
    """
    Reports progress and status information for pipeline operations.
    
    Provides both terminal progress bars and structured logging of
    pipeline stages and their statuses.
    """
    
    def __init__(self, total_stages: int = 0, verbose: bool = False):
        """
        Initialize the status reporter.
        
        Args:
            total_stages: Total number of stages in the pipeline
            verbose: Whether to print verbose output
        """
        self.total_stages = total_stages
        self.current_stage = 0
        self.verbose = verbose
        self.stages: Dict[str, StageStatus] = {}
        self.stage_start_times: Dict[str, float] = {}
        self.stage_end_times: Dict[str, float] = {}
        self.logger = logging.getLogger("moriarty.status")
        
        # Main progress bar for overall pipeline
        self.main_progress = None
        
        # Current active progress bar for stage
        self.current_progress = None
    
    def start_pipeline(self, name: str, total_stages: Optional[int] = None) -> None:
        """
        Start tracking a new pipeline run.
        
        Args:
            name: Name of the pipeline
            total_stages: Total number of stages (if different from initialization)
        """
        if total_stages is not None:
            self.total_stages = total_stages
            
        self.current_stage = 0
        self.stages = {}
        self.stage_start_times = {}
        self.stage_end_times = {}
        
        self.logger.info(f"Starting pipeline: {name}")
        
        if self.verbose:
            self.main_progress = tqdm(
                total=self.total_stages,
                desc=f"Pipeline: {name}",
                unit="stage",
                position=0,
                leave=True
            )
    
    def end_pipeline(self, success: bool = True) -> None:
        """
        End the current pipeline run.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        if self.main_progress:
            self.main_progress.close()
            self.main_progress = None
            
        if success:
            self.logger.info("Pipeline completed successfully")
        else:
            self.logger.error("Pipeline failed")
            
        # Log timing summary
        for stage_name, status in self.stages.items():
            if status == StageStatus.COMPLETED:
                duration = self.stage_end_times.get(stage_name, 0) - self.stage_start_times.get(stage_name, 0)
                self.logger.info(f"Stage '{stage_name}' completed in {duration:.2f} seconds")
            elif status == StageStatus.FAILED:
                self.logger.error(f"Stage '{stage_name}' failed")
    
    def start_stage(self, name: str, total: Optional[int] = None) -> None:
        """
        Start a new pipeline stage.
        
        Args:
            name: Name of the stage
            total: Total number of items to process in this stage (for progress bar)
        """
        self.current_stage += 1
        self.stages[name] = StageStatus.IN_PROGRESS
        self.stage_start_times[name] = time.time()
        
        self.logger.info(f"Starting stage: {name}")
        
        if self.main_progress:
            self.main_progress.update(1)
            
        if self.verbose and total is not None:
            # Create a progress bar for this stage
            self.current_progress = tqdm(
                total=total,
                desc=f"Stage {self.current_stage}/{self.total_stages}: {name}",
                unit="item",
                position=1,
                leave=False
            )
    
    def update_stage_progress(self, amount: int = 1) -> None:
        """
        Update progress for the current stage.
        
        Args:
            amount: Number of items completed
        """
        if self.current_progress:
            self.current_progress.update(amount)
    
    def complete_stage(self, name: str, success: bool = True) -> None:
        """
        Mark a pipeline stage as completed.
        
        Args:
            name: Name of the stage
            success: Whether the stage completed successfully
        """
        if name in self.stages:
            self.stages[name] = StageStatus.COMPLETED if success else StageStatus.FAILED
            self.stage_end_times[name] = time.time()
            
            duration = self.stage_end_times[name] - self.stage_start_times.get(name, self.stage_end_times[name])
            
            if success:
                self.logger.info(f"Stage '{name}' completed in {duration:.2f} seconds")
            else:
                self.logger.error(f"Stage '{name}' failed after {duration:.2f} seconds")
        
        if self.current_progress:
            self.current_progress.close()
            self.current_progress = None
    
    def skip_stage(self, name: str, reason: str) -> None:
        """
        Mark a pipeline stage as skipped.
        
        Args:
            name: Name of the stage
            reason: Reason for skipping
        """
        if name in self.stages:
            self.stages[name] = StageStatus.SKIPPED
            
        self.logger.info(f"Stage '{name}' skipped: {reason}")
        
        if self.main_progress:
            self.main_progress.update(1)
    
    def log_info(self, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
        if self.verbose:
            tqdm.write(message)
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
        if self.verbose:
            tqdm.write(f"WARNING: {message}")
    
    def log_error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
        if self.verbose:
            tqdm.write(f"ERROR: {message}")
    
    def create_processing_callback(self, total: int) -> Callable[[int, str], None]:
        """
        Create a callback function for processing progress updates.
        
        Args:
            total: Total number of items to process
            
        Returns:
            Callback function for progress updates
        """
        progress_bar = tqdm(total=total, desc="Processing", position=1, leave=False) if self.verbose else None
        
        def callback(step: int, message: str = "") -> None:
            if progress_bar:
                progress_bar.update(step)
                if message:
                    progress_bar.set_description(message)
            
            if message:
                self.logger.debug(message)
        
        return callback 