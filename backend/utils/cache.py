"""
backend/utils/cache.py
Thread-safe in-memory TTL cache for GenAI responses and embedding vectors.
"""

import time
import threading
from typing import Any, Optional


class TTLCache:
    """
    Thread-safe dictionary-based cache with per-key time-to-live expiry.

    Usage:
        cache = TTLCache(default_ttl=300)   # 5-minute default
        cache.set("key", value)
        val = cache.get("key")              # None if expired
    """

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock  = threading.Lock()
        self._ttl   = default_ttl

    # ── Public API ──────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if it exists and has not expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with optional TTL override (seconds)."""
        expires_at = time.time() + (ttl if ttl is not None else self._ttl)
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns number of evicted keys."""
        now = time.time()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# ── Module-level singleton instances ────────────────────────────────

# Cache for Ollama model responses (longer TTL — AI responses are expensive)
response_cache = TTLCache(default_ttl=600)   # 10 minutes

# Cache for embedding vectors (very long TTL — embeddings don't change)
embedding_cache = TTLCache(default_ttl=3600)  # 1 hour

# Cache for strategy/agent results (medium TTL)
strategy_cache = TTLCache(default_ttl=300)    # 5 minutes
