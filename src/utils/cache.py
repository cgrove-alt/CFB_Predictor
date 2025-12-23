"""
Caching Layer for Sharp Sports Predictor.

Provides:
- In-memory caching with TTL
- Disk-based caching for API responses
- Decorator for easy function caching
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T")


class CacheEntry:
    """Represents a cached value with metadata."""

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.created_at


class Cache:
    """
    Multi-tier caching system with memory and disk support.

    Features:
    - In-memory cache with TTL
    - Optional disk persistence
    - Automatic cleanup of expired entries
    - Thread-safe operations
    """

    def __init__(
        self,
        default_ttl: int = 300,
        cache_dir: Optional[str] = None,
        enable_disk: bool = False,
        max_memory_items: int = 1000,
    ):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds
            cache_dir: Directory for disk cache
            enable_disk: Whether to persist to disk
            max_memory_items: Maximum items in memory cache
        """
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._enable_disk = enable_disk
        self._max_memory_items = max_memory_items

        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path(__file__).parent.parent.parent / ".cache"

        if self._enable_disk:
            self._cache_dir.mkdir(exist_ok=True)

    def _make_key(self, key: str) -> str:
        """Create a safe cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _disk_path(self, key: str) -> Path:
        """Get disk path for a cache key."""
        safe_key = self._make_key(key)
        return self._cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Check memory first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                return entry.value
            else:
                del self._memory_cache[key]

        # Check disk cache
        if self._enable_disk:
            disk_path = self._disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        entry = pickle.load(f)
                    if not entry.is_expired:
                        # Promote to memory cache
                        self._memory_cache[key] = entry
                        return entry.value
                    else:
                        disk_path.unlink()  # Delete expired
                except Exception:
                    pass

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        ttl = ttl or self._default_ttl
        entry = CacheEntry(value, ttl)

        # Check memory limit
        if len(self._memory_cache) >= self._max_memory_items:
            self._evict_oldest()

        self._memory_cache[key] = entry

        # Write to disk
        if self._enable_disk:
            try:
                disk_path = self._disk_path(key)
                with open(disk_path, "wb") as f:
                    pickle.dump(entry, f)
            except Exception:
                pass  # Disk cache is optional

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        deleted = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True

        if self._enable_disk:
            disk_path = self._disk_path(key)
            if disk_path.exists():
                disk_path.unlink()
                deleted = True

        return deleted

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of items cleared
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        if self._enable_disk and self._cache_dir.exists():
            for cache_file in self._cache_dir.glob("*.cache"):
                cache_file.unlink()
                count += 1

        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        removed = 0

        # Clean memory cache
        expired_keys = [
            k for k, v in self._memory_cache.items() if v.is_expired
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            removed += 1

        # Clean disk cache
        if self._enable_disk and self._cache_dir.exists():
            for cache_file in self._cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)
                    if entry.is_expired:
                        cache_file.unlink()
                        removed += 1
                except Exception:
                    cache_file.unlink()  # Remove corrupted files
                    removed += 1

        return removed

    def _evict_oldest(self) -> None:
        """Evict the oldest entry from memory cache."""
        if not self._memory_cache:
            return

        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].created_at,
        )
        del self._memory_cache[oldest_key]

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_items = len(self._memory_cache)
        expired_memory = sum(
            1 for v in self._memory_cache.values() if v.is_expired
        )

        disk_items = 0
        if self._enable_disk and self._cache_dir.exists():
            disk_items = len(list(self._cache_dir.glob("*.cache")))

        return {
            "memory_items": memory_items,
            "memory_expired": expired_memory,
            "disk_items": disk_items,
            "disk_enabled": self._enable_disk,
            "max_memory_items": self._max_memory_items,
            "default_ttl": self._default_ttl,
        }


# Global cache instance
_global_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = Cache(
            default_ttl=300,
            enable_disk=True,
        )
    return _global_cache


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    cache_instance: Optional[Cache] = None,
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys
        cache_instance: Optional specific cache to use

    Usage:
        @cached(ttl=300, key_prefix="api")
        def fetch_data(year: int, week: int):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = cache_instance or get_cache()

            # Build cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        # Add method to invalidate cache for this function
        def invalidate(*args, **kwargs):
            cache = cache_instance or get_cache()
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))
            return cache.delete(cache_key)

        wrapper.invalidate = invalidate
        return wrapper

    return decorator


class APICache:
    """
    Specialized cache for API responses with rate limiting awareness.
    """

    def __init__(
        self,
        schedule_ttl: int = 300,
        lines_ttl: int = 300,
        stats_ttl: int = 3600,
    ):
        """
        Initialize API cache with different TTLs for different data types.

        Args:
            schedule_ttl: TTL for schedule data
            lines_ttl: TTL for betting lines
            stats_ttl: TTL for team stats
        """
        self._cache = Cache(enable_disk=True)
        self._schedule_ttl = schedule_ttl
        self._lines_ttl = lines_ttl
        self._stats_ttl = stats_ttl

    def get_schedule(self, season: int, week: int) -> Optional[Any]:
        """Get cached schedule."""
        key = f"schedule:{season}:{week}"
        return self._cache.get(key)

    def set_schedule(self, season: int, week: int, data: Any) -> None:
        """Cache schedule data."""
        key = f"schedule:{season}:{week}"
        self._cache.set(key, data, self._schedule_ttl)

    def get_lines(self, season: int, week: int) -> Optional[Any]:
        """Get cached betting lines."""
        key = f"lines:{season}:{week}"
        return self._cache.get(key)

    def set_lines(self, season: int, week: int, data: Any) -> None:
        """Cache betting lines."""
        key = f"lines:{season}:{week}"
        self._cache.set(key, data, self._lines_ttl)

    def get_team_stats(self, team: str, season: int) -> Optional[Any]:
        """Get cached team stats."""
        key = f"stats:{team}:{season}"
        return self._cache.get(key)

    def set_team_stats(self, team: str, season: int, data: Any) -> None:
        """Cache team stats."""
        key = f"stats:{team}:{season}"
        self._cache.set(key, data, self._stats_ttl)

    def clear_all(self) -> int:
        """Clear all API cache."""
        return self._cache.clear()
