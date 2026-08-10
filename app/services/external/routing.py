"""
app/services/external/routing.py
─────────────────────────────────
How far and how long the drive actually is, from a routing engine.

This replaces a table of 80 surveyed routes plus two sets of calibrated fudge factors
(a great-circle multiplier per terrain band, and an average speed per distance band).
The table only covered routes somebody had measured; the factors filled the gap with
arithmetic that had a median error of 10% and a worst case of 170%. Both were guesses
about roads, made without asking anything that knows about roads.

A real engine follows the actual road graph, so:

  • the distance is the distance driven, not a straight line scaled by terrain
  • the drive time comes from the road classes and speed limits on the route
  • a mountain road that doubles back is measured, not approximated

DEFAULTS AND SUBSTITUTION
─────────────────────────
The default is OSRM (`router.project-osrm.org`): no API key, and its car profile
already accounts for road class. The public instance is fine for development and rate-
limited for production, so point `settings.routing_base_url` at a self-hosted OSRM (or
anything speaking the same `/route/v1` shape) when this carries real traffic.

`route()` returns None when the engine cannot find a road between the two points, and
None is a meaningful answer: the fare endpoint needs a distance, so no route means no
price — and the reply says the fare is confirmed at booking rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.services.local_cache import TTLCache

log = get_logger(__name__)


@dataclass(frozen=True)
class RoadRoute:
    """A driveable route between two points, as the engine measured it."""

    distance_km: float
    duration_minutes: int

    @property
    def duration_text(self) -> str:
        """Drive time the way a rider would say it: "4h 30m", "45m"."""
        hours, minutes = divmod(int(round(self.duration_minutes)), 60)
        if hours and minutes:
            return f"{hours}h {minutes}m"
        return f"{hours}h" if hours else f"{minutes}m"


class _Retryable(Exception):
    """A transient engine failure worth retrying (5xx, timeout, connection reset)."""


class RoutingService:
    """Live road routing, pooled and cached. Never raises to its callers."""

    _TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._cache = TTLCache()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                headers={"User-Agent": settings.geocoding_user_agent},
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((_Retryable, httpx.ConnectError, httpx.ReadTimeout)),
    )
    async def _get(self, url: str, params: dict) -> object:
        client = await self._get_client()
        response = await client.get(url, params=params)
        if response.status_code >= 500:
            raise _Retryable(str(response.status_code))
        response.raise_for_status()
        return response.json()

    async def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> Optional[RoadRoute]:
        """Drive the roads between two (lat, lon) points. None when there is no route."""
        # ~100 m resolution on the cache key: the same journey asked twice in a
        # conversation must not cost two round trips, and GPS jitter is not a new route.
        key = (
            f"{round(origin[0], 3)},{round(origin[1], 3)}"
            f";{round(destination[0], 3)},{round(destination[1], 3)}"
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached or None      # a cached miss is stored as False

        try:
            payload = await self._fetch_route(origin, destination)
        except Exception as e:
            log.warning(
                "routing_failed",
                origin=origin,
                destination=destination,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

        route = self._parse_route(payload)
        self._cache.set(key, route or False, settings.routing_cache_ttl_seconds)
        if route is None:
            log.info("routing_no_road_found", origin=origin, destination=destination)
        return route

    # ══════════════════════════════════════════════════════════════════════════
    # ENGINE BOUNDARY — only the two methods below know OSRM's URL and JSON shape.
    #
    # GET /route/v1/driving/{lon},{lat};{lon},{lat}?overview=false
    #   → {"code": "Ok", "routes": [{"distance": <metres>, "duration": <seconds>}]}
    #
    # Swap engine by reimplementing these two and nothing else.
    # ══════════════════════════════════════════════════════════════════════════

    async def _fetch_route(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> object:
        # OSRM takes lon,lat — the reverse of every other coordinate in this codebase.
        coordinates = (
            f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        )
        url = (
            f"{settings.routing_base_url.rstrip('/')}"
            f"{settings.routing_path.rstrip('/')}/{coordinates}"
        )
        return await self._get(url, {"overview": "false", "alternatives": "false"})

    def _parse_route(self, payload: object) -> Optional[RoadRoute]:
        if not isinstance(payload, dict) or payload.get("code") != "Ok":
            return None
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            return None
        best = routes[0]
        if not isinstance(best, dict):
            return None

        metres, seconds = best.get("distance"), best.get("duration")
        if not isinstance(metres, (int, float)) or not isinstance(seconds, (int, float)):
            return None
        if metres <= 0:
            return None

        # Rounded to 5 minutes: the engine reports seconds, but quoting "4h 07m" to a
        # rider implies it knows about today's traffic, and it does not.
        minutes = max(5, int(round((seconds / 60) / 5) * 5))
        return RoadRoute(distance_km=round(metres / 1000, 1), duration_minutes=minutes)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ── Singleton ─────────────────────────────────────────────────────────────────
_routing_service: Optional[RoutingService] = None


def get_routing_service() -> RoutingService:
    global _routing_service
    if _routing_service is None:
        _routing_service = RoutingService()
    return _routing_service
