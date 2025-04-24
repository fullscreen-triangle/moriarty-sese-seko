"""
Unified Configuration Management System for Moriarty

This module provides a centralized configuration system that can be accessed
throughout the application. It supports:
- Default configurations
- Loading from environment variables
- Loading from configuration files (YAML, JSON)
- Hierarchical configuration with overrides
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Setup logging
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Unified configuration management system for Moriarty.
    
    This class implements the Singleton pattern to ensure only one configuration
    instance exists throughout the application.
    """
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance or return existing one."""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        # Only initialize once
        if getattr(self, '_initialized', False):
            return
            
        # Initialize with default configuration
        self._config = self._get_default_config()
        self._config_sources = ["default"]
        self._initialized = True
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration settings."""
        return {
            "system": {
                "memory_limit_fraction": 0.8,
                "workers": None,  # Auto-detect
                "batch_size": 32,
                "device": "cuda" if self._is_cuda_available() else "cpu",
            },
            "paths": {
                "output_dir": "output",
                "model_dir": "models",
                "llm_training_dir": "llm_training_data",
                "llm_model_dir": "llm_models",
            },
            "pose": {
                "model_complexity": 1,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            "video": {
                "extract_fps": 30,
                "output_video_fps": 30,
                "output_width": 1280,
                "output_height": 720,
            },
            "pipeline": {
                "use_ray": True,
                "use_dask": True,
                "enable_visualizations": True,
                "enable_progress_bars": True,
            },
            "logging": {
                "level": "INFO",
                "file": "moriarty.log",
                "enable_console": True,
            }
        }
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_prefix = "MORIARTY_"
        
        # Only process environment variables starting with the prefix
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(env_prefix):].lower()
                
                # Split by double underscore to get hierarchical keys
                parts = config_key.split("__")
                
                # Navigate to the right place in the config
                config = self._config
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                
                # Set the value (convert to appropriate type)
                try:
                    # Try to parse as JSON first (for complex values)
                    config[parts[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string if not valid JSON
                    config[parts[-1]] = value
        
        self._config_sources.append("environment")
        logger.info("Configuration loaded from environment variables")
    
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file (YAML or JSON).
        
        Args:
            file_path: Path to the configuration file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {file_path.suffix}")
                    return
            
            # Deep merge with existing configuration
            self._deep_update(self._config, file_config)
            self._config_sources.append(str(file_path))
            logger.info(f"Configuration loaded from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Deep update of nested dictionaries.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Update or set the value
                target[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value (e.g., 'system.workers')
            default: Default value to return if the path doesn't exist
            
        Returns:
            The configuration value or the default
        """
        parts = path.split('.')
        
        # Navigate through the nested dictionary
        config = self._config
        for part in parts:
            if part not in config:
                return default
            config = config[part]
        
        return config
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value by path.
        
        Args:
            path: Dot-separated path to the configuration value (e.g., 'system.workers')
            value: Value to set
        """
        parts = path.split('.')
        
        # Navigate through the nested dictionary
        config = self._config
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._config.copy()
    
    def get_sources(self) -> List[str]:
        """Get the list of configuration sources that have been loaded."""
        return self._config_sources.copy()

# Global instance for easy access
config = ConfigurationManager() 