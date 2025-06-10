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
    Manages configuration settings for the computer vision pipeline.
    
    This class handles loading configuration from files and environment variables,
    provides default values, and validates settings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if getattr(self, '_initialized', False):
            return
            
        self.config_path = Path("config.json")
        self.config = self._load_config()
        self._initialized = True
        
        logger.info("Configuration manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return self._merge_with_defaults(config)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Return default config if file doesn't exist or couldn't be loaded
        config = self._get_default_config()
        self._save_config(config)
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration settings."""
        return {
            "system": {
                "memory_limit_fraction": 0.25,  # Conservative 25% instead of 80%
                "workers": None,  # Auto-detect (conservative)
                "batch_size": 5,  # Small batch size for stability  
                "device": "cuda" if self._is_cuda_available() else "cpu",
            },
            "paths": {
                "output_dir": "output",
                "model_dir": "models",
                "llm_training_dir": "llm_training_data",
                "llm_model_dir": "llm_models",
                "cache_dir": "cache",  # For model caching
            },
            "pose": {
                "model_complexity": 1,  # Lower complexity for faster processing
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            "mediapipe": {
                "cache_models": True,  # Enable model caching
                "default_complexity": 1,  # Conservative complexity
                "max_cached_models": 3,  # Limit cached models to save memory
                "model_cache_dir": "cache/mediapipe_models",
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
                "conservative_mode": True,  # Enable conservative resource usage
                "max_workers": 4,  # Hard cap on workers
                "max_batch_size": 10,  # Hard cap on batch size
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
    
    def _merge_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with defaults."""
        default_config = self._get_default_config()
        
        def deep_merge(default: Dict, user: Dict) -> Dict:
            merged = default.copy()
            for key, value in user.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        return deep_merge(default_config, user_config)
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except IOError as e:
            logger.warning(f"Failed to save config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config(self.config)
    
    def get_mediapipe_config(self) -> Dict[str, Any]:
        """Get MediaPipe-specific configuration."""
        return self.get('mediapipe', {})
    
    def get_conservative_limits(self) -> Dict[str, Any]:
        """Get conservative resource limits to prevent system crashes."""
        return {
            "memory_limit_fraction": min(0.25, self.get('system.memory_limit_fraction', 0.25)),
            "max_workers": min(4, self.get('pipeline.max_workers', 4)),
            "max_batch_size": min(10, self.get('pipeline.max_batch_size', 10)),
            "conservative_mode": self.get('pipeline.conservative_mode', True),
        }
    
    def should_cache_models(self) -> bool:
        """Check if model caching is enabled."""
        return self.get('mediapipe.cache_models', True)
    
    def get_model_cache_dir(self) -> Path:
        """Get the model cache directory."""
        cache_dir = Path(self.get('mediapipe.model_cache_dir', 'cache/mediapipe_models'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

# Global configuration instance
config_manager = ConfigurationManager()

# Export the global config instance for easier access
config = config_manager 