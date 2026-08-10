"""
app/services/local_cache.py
────────────────────────────
In-process TTL cache and rate limiter — the replacement for the Redis layer.

WHAT CHANGED, AND WHAT IT COSTS
───────────────────────────────
Redis held three things: a "which conversation is this rider in" pointer, a cache of
the regional alerts response, and the per-rider rate-limit counter. Only the first was
load-bearing, and it was already mirrored durably in Firestore — the fallback existed
precisely because Redis kept being unavailable. The other two are rebuildable in a few
hundred milliseconds. So the dependency bought very little and cost an outage mode.

The trade is real and worth stating plainly: this state lives in ONE process.

  • Rate limiting is per instance. An N-instance deployment therefore allows up to
    N x `rate_limit_per_minute` per rider. If a hard global limit matters, enforce it
    at the gateway or load balancer, which can see all instances — do not reintroduce
    a shared cache just to count requests.
  • The alerts cache is per instance, so the upstream weather API is called once per
    instance per TTL rather than once overall. At six destinations on a 15-minute TTL
    that is negligible.
  • Everything resets on deploy. For a counter and a cache, that is correct behaviour.

Conversation state is unaffected: it lives in Firestore and always did.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.core.logging import get_logger

log = get_logger(__name__)


class TTLCache:
    """A small dict with per-entry expiry. Not shared between processes — by design."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._entries[key] = (time.monotonic() + ttl_seconds, value)
        self._evict_expired()

    def clear(self) -> None:
        self._entries.clear()

    def _evict_expired(self) -> None:
        """Drop expired entries so an unbounded key space can't leak memory.

        Cheap because the caches here hold single-digit numbers of keys; if one ever
        grows, this is the line to replace with a bounded LRU.
        """
        now = time.monotonic()
        for key in [k for k, (expires_at, _) in self._entries.items() if expires_at <= now]:
            self._entries.pop(key, None)


class RateLimiter:
    """Fixed-window request counter, keyed by rider.

    A fixed window (rather than a sliding one) is deliberate: it matches the
    `Retry-After: 60` the API promises, so a caller that waits the advertised time is
    always let back in. See the module docstring on the per-instance caveat.
    """

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: dict[str, tuple[int, int]] = {}   # key → (window index, count)

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns (allowed, count_in_this_window)."""
        window = int(time.time() // self.window_seconds)
        current_window, count = self._windows.get(key, (window, 0))

        if current_window != window:
            current_window, count = window, 0

        count += 1
        self._windows[key] = (current_window, count)

        if len(self._windows) > 10_000:
            self._prune(window)

        return count <= self.limit, count

    def reset(self) -> None:
        self._windows.clear()

    def _prune(self, current_window: int) -> None:
        """Forget riders whose window has already rolled over."""
        stale = [k for k, (w, _) in self._windows.items() if w != current_window]
        for key in stale:
            self._windows.pop(key, None)
        log.debug("rate_limiter_pruned", removed=len(stale))
