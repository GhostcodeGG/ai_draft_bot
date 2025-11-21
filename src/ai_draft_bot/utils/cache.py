"""Caching utilities for expensive feature computations.

This module provides caching decorators and utilities to dramatically
speed up feature extraction by avoiding redundant computations.
"""

from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable


def hash_card_name(card_name: str) -> str:
    """Create a hash key for a card name.

    Args:
        card_name: Card name

    Returns:
        MD5 hash of the card name
    """
    return hashlib.md5(card_name.encode()).hexdigest()


def numpy_cache(maxsize: int = 1000) -> Callable:
    """LRU cache decorator that works with numpy arrays.

    Args:
        maxsize: Maximum cache size

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @lru_cache(maxsize=maxsize)
        def cached_wrapper(hashable_args):
            # Convert back from hashable format
            args, kwargs = pickle.loads(hashable_args)
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args/kwargs to hashable format
            hashable = pickle.dumps((args, kwargs))
            return cached_wrapper(hashable)

        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


class FeatureCache:
    """Persistent cache for card-level features.

    This cache stores computed features to disk for reuse across sessions.
    """

    def __init__(self, cache_dir: Path | str = "cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: dict[str, Any] = {}

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{hash_card_name(key)}.pkl"

    def get(self, key: str) -> Any | None:
        """Get cached value.

        Args:
            key: Cache key (usually card name)

        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    value = pickle.load(f)
                    # Add to memory cache
                    self.memory_cache[key] = value
                    return value
            except Exception:
                # Corrupted cache file
                cache_path.unlink(missing_ok=True)
                return None

        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Store in memory
        self.memory_cache[key] = value

        # Store to disk
        cache_path = self._get_cache_path(key)
        try:
            with cache_path.open("wb") as f:
                pickle.dump(value, f)
        except Exception:
            # If disk write fails, at least we have memory cache
            pass

    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        disk_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in disk_files)

        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(disk_files),
            "total_size_mb": total_size / (1024 * 1024),
        }


# Global feature cache instance
_feature_cache = FeatureCache()


def get_feature_cache() -> FeatureCache:
    """Get the global feature cache instance."""
    return _feature_cache


def cached_card_features(func: Callable) -> Callable:
    """Decorator for caching card-level features.

    Usage:
        @cached_card_features
        def extract_features(card: CardMetadata) -> np.ndarray:
            # Expensive computation
            ...
    """

    @wraps(func)
    def wrapper(card, *args, **kwargs):
        cache = get_feature_cache()

        # Create cache key from card name + function name
        cache_key = f"{func.__name__}:{card.name}"

        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Compute and cache
        result = func(card, *args, **kwargs)
        cache.set(cache_key, result)

        return result

    return wrapper
