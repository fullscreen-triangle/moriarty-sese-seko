"""
Caching System for Moriarty Pipeline

This module provides a caching system for storing and retrieving intermediate
results to avoid redundant computations in the pipeline.
"""

import os
import json
import pickle
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Caching system for intermediate results to avoid redundant computations.
    
    This class implements the Singleton pattern to ensure only one cache manager
    instance exists throughout the application.
    """
    _instance = None
    
    def __new__(cls, cache_dir: str = ".cache", expiration: float = 24*60*60):
        """Create a singleton instance or return existing one."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: str = ".cache", expiration: float = 24*60*60):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            expiration: Cache expiration time in seconds (default: 24 hours)
        """
        # Only initialize once
        if getattr(self, '_initialized', False):
            return
            
        self.cache_dir = Path(cache_dir)
        self.expiration = expiration
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "stores": 0}
        self._initialized = True
        logger.info(f"Cache initialized in {self.cache_dir} with {self.expiration}s expiration")
    
    @classmethod
    def get_instance(cls, cache_dir: str = ".cache", expiration: float = 24*60*60):
        """Get or create the singleton instance."""
        return cls(cache_dir, expiration)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from the arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key as a string
        """
        # Combine all arguments into a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_str = "|".join(key_parts)
        
        # Create a hash of the key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}.pickle"
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key not found or expired
            
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self._stats["misses"] += 1
            return default
        
        # Check if cache has expired
        if time.time() - cache_path.stat().st_mtime > self.expiration:
            self._stats["misses"] += 1
            return default
        
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Error reading cache for key {key}: {e}")
            self._stats["misses"] += 1
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            self._stats["stores"] += 1
            logger.debug(f"Cached value for key: {key}")
        except Exception as e:
            logger.error(f"Error caching value for key {key}: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            key: Specific key to clear, or None to clear all cache
        """
        if key:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cleared cache for key: {key}")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.pickle"):
                cache_file.unlink()
            logger.info("Cleared all cache entries")
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pickle"):
            if time.time() - cache_file.stat().st_mtime > self.expiration:
                cache_file.unlink()
                count += 1
        
        logger.info(f"Cleared {count} expired cache entries")
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._stats.copy()


def cached(expiration: Optional[float] = None):
    """
    Decorator for caching function results.
    
    Args:
        expiration: Optional custom expiration time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            cache_manager = CacheManager.get_instance()
            
            # Use custom expiration if provided
            if expiration is not None:
                cache_manager.expiration = expiration
            
            # Generate cache key from function name and arguments
            key_base = f"{func.__module__}.{func.__name__}"
            key = cache_manager._generate_key(key_base, *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(key)
            
            # If not in cache, compute and store
            if result is None:
                result = func(*args, **kwargs)
                cache_manager.set(key, result)
            
            return result
        
        return wrapper
    
    return decorator


# Global instance for easy access
cache = CacheManager.get_instance() 