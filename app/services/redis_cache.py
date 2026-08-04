"""
app/services/redis_cache.py
────────────────────────────
Redis cache service — the hot cache layer sitting in front of Firestore.
Used for:
  - Active session caching (TTL: 2h)
  - Weather data caching (TTL: 30 min)
  - Places/hotel data caching (TTL: 24h)
  - Rate limiting counters
"""

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Cache key prefixes for namespacing
KEYS = {
    "session": "session:{session_id}",
    "weather": "weather:{destination}:{date}",
    "places": "places:{destination}:{category}",
    "alerts": "alerts:himachal",
    "rate_limit": "ratelimit:{user_id}",
    "crowd": "crowd:{place}:{month}",
}


class RedisCacheService:
    """
    Async Redis cache service.
    All methods gracefully degrade — cache misses fall back to source of truth.
    """

    def __init__(self) -> None:
        self._pool: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        if self._pool is None:
            self._pool = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
            )
        return self._pool

    # ── Generic helpers ───────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Get a JSON-serialised value from Redis."""
        try:
            client = await self._get_client()
            raw = await client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            log.warning("redis_get_failed", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: int) -> bool:
        """Set a JSON-serialised value with TTL (seconds)."""
        try:
            client = await self._get_client()
            await client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            log.warning("redis_set_failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(key)
        except Exception as e:
            log.warning("redis_delete_failed", key=key, error=str(e))

    # ── Session cache ─────────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Optional[dict]:
        key = KEYS["session"].format(session_id=session_id)
        return await self.get(key)

    async def set_session(self, session_id: str, session_dict: dict) -> None:
        key = KEYS["session"].format(session_id=session_id)
        await self.set(key, session_dict, ttl=settings.redis_ttl_session)

    async def invalidate_session(self, session_id: str) -> None:
        key = KEYS["session"].format(session_id=session_id)
        await self.delete(key)

    # ── Weather cache ─────────────────────────────────────────────────────────

    async def get_weather(self, destination: str, date: str = "today") -> Optional[dict]:
        key = KEYS["weather"].format(destination=destination.lower(), date=date)
        return await self.get(key)

    async def set_weather(self, destination: str, data: dict, date: str = "today") -> None:
        key = KEYS["weather"].format(destination=destination.lower(), date=date)
        await self.set(key, data, ttl=settings.redis_ttl_weather)

    # ── Places cache ──────────────────────────────────────────────────────────

    async def get_places(self, destination: str, category: str) -> Optional[list]:
        key = KEYS["places"].format(destination=destination.lower(), category=category)
        return await self.get(key)

    async def set_places(self, destination: str, category: str, data: list) -> None:
        key = KEYS["places"].format(destination=destination.lower(), category=category)
        await self.set(key, data, ttl=settings.redis_ttl_places)

    # ── Alerts cache ──────────────────────────────────────────────────────────

    async def get_himachal_alerts(self) -> Optional[dict]:
        return await self.get(KEYS["alerts"])

    async def set_himachal_alerts(self, data: dict, ttl: int = 900) -> None:
        await self.set(KEYS["alerts"], data, ttl=ttl)

    # ── Crowd cache ───────────────────────────────────────────────────────────

    async def get_crowd_data(self, place: str, month: str) -> Optional[dict]:
        key = KEYS["crowd"].format(place=place.lower(), month=month)
        return await self.get(key)

    async def set_crowd_data(self, place: str, month: str, data: dict) -> None:
        key = KEYS["crowd"].format(place=place.lower(), month=month)
        await self.set(key, data, ttl=86400 * 7)  # 1 week

    # ── Rate limiting ─────────────────────────────────────────────────────────

    async def check_rate_limit(self, user_id: str) -> tuple[bool, int]:
        """
        Sliding window rate limiter.
        Returns (is_allowed, current_count).
        """
        try:
            client = await self._get_client()
            key = KEYS["rate_limit"].format(user_id=user_id)
            current = await client.incr(key)
            if current == 1:
                await client.expire(key, 60)  # 1-minute window
            allowed = current <= settings.rate_limit_per_minute
            return allowed, current
        except Exception as e:
            log.warning("rate_limit_check_failed", user_id=user_id, error=str(e))
            return True, 0  # fail open

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._pool:
            await self._pool.aclose()
            self._pool = None

    async def ping(self) -> bool:
        """Health check for Redis connectivity."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception:
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────

_redis_cache: Optional[RedisCacheService] = None


def get_redis_cache() -> RedisCacheService:
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCacheService()
    return _redis_cache
