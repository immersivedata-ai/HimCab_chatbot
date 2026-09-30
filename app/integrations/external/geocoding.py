"""
app/integrations/external/geocoding.py
───────────────────────────────────
Turning places into coordinates, and coordinates into places — live, over the wire.

This replaces a hand-maintained table of 71 town coordinates and a hand-drawn outline
of the Himachal border. Both worked until they didn't: a town nobody had listed had no
coordinates, so it could not be routed, priced, or recognised as being in the state,
and the rider silently got no fare. The set of places people ask to go is not a set
anyone can finish typing out.

Two operations, both from the same provider:

  geocode(name)      → where is "Jibhi"?          (name → coordinates + state)
  reverse(lat, lon)  → what is the rider next to? (GPS fix → town + state)

`Place.in_himachal` comes from the provider's own administrative data, so "is this
rider a resident?" is answered by the boundary the mapping authority publishes rather
than by a polygon we drew.

DEFAULTS AND SUBSTITUTION
─────────────────────────
The default provider is Nominatim (OpenStreetMap): no API key, global coverage,
and it returns the administrative state directly. Its usage policy asks for an
identifying User-Agent and at most one request a second, both of which this client
honours — see `_MIN_REQUEST_INTERVAL_SECONDS` and `settings.geocoding_user_agent`.
Point `settings.geocoding_base_url` at a self-hosted instance (or any Nominatim-
compatible service) for production traffic; nothing here assumes the public server.

Results are cached in memory for `settings.geocoding_cache_ttl_seconds`. That is not
a table being maintained: a cache entry expires, holds only what a rider actually
asked for, and is rebuilt from the provider. It exists because the coordinates of a
town do not change between two messages, and because the rate limit above makes
re-asking on every turn both slow and impolite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.services.local_cache import TTLCache

log = get_logger(__name__)

#: The state whose residents are "local". Compared case-insensitively against the
#: provider's administrative name, so no boundary geometry is needed here.
HIMACHAL_STATE = "himachal pradesh"

#: Nominatim's usage policy caps anonymous clients at one request per second. Enforced
#: process-wide rather than per-call, because a single chat turn can geocode both ends
#: of a journey at once.
_MIN_REQUEST_INTERVAL_SECONDS = 1.1

#: Administrative keys a town's name can arrive under, most specific first. Reverse
#: geocoding a hill road returns no `city` but usually a `village` or a `county`.
_PLACE_NAME_KEYS = (
    "city", "town", "village", "municipality", "suburb",
    "county", "state_district", "district", "region",
)


@dataclass(frozen=True)
class Place:
    """A resolved location: what it is called, where it is, and which state it is in."""

    name: str
    lat: float
    lon: float
    state: Optional[str] = None
    country: Optional[str] = None
    #: The provider's full display string, kept for logs and debugging only.
    display_name: Optional[str] = None

    @property
    def coords(self) -> tuple[float, float]:
        return (self.lat, self.lon)

    @property
    def in_himachal(self) -> bool:
        """True when the provider places this inside Himachal Pradesh.

        This is the persona decision — local rider or visitor — and it comes from
        published administrative boundaries rather than a polygon of our own, which
        is why a fix 2 km from the state line no longer needs a caveat.
        """
        return (self.state or "").strip().lower() == HIMACHAL_STATE


class _RetryableError(Exception):
    """A transient provider failure worth retrying (5xx, timeout, connection reset)."""


class GeocodingService:
    """Live place lookups, pooled and rate-limited. Never raises to its callers.

    Every method returns None on failure. That is deliberate: an unresolvable place
    must leave the agent asking the rider where they mean, not pricing a guess.
    """

    _TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._cache = TTLCache()
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                headers={
                    # Nominatim rejects anonymous traffic; identify the service.
                    "User-Agent": settings.geocoding_user_agent,
                    "Accept": "application/json",
                    "Accept-Language": "en",
                },
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def _throttle(self) -> None:
        """Space requests out to honour the provider's rate limit."""
        async with self._rate_lock:
            wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((_RetryableError, httpx.ConnectError, httpx.ReadTimeout)),
    )
    async def _get(self, path: str, params: dict) -> object:
        await self._throttle()
        client = await self._get_client()
        url = f"{settings.geocoding_base_url.rstrip('/')}{path}"
        response = await client.get(url, params=params)
        if response.status_code >= 500:
            raise _RetryableError(str(response.status_code))
        response.raise_for_status()
        return response.json()

    # ══════════════════════════════════════════════════════════════════════════
    # PROVIDER BOUNDARY — only the two methods below know the response shape.
    # Nominatim: /search and /reverse, format=jsonv2, addressdetails=1.
    # Swap provider by reimplementing these two and nothing else.
    # ══════════════════════════════════════════════════════════════════════════

    def _to_place(self, row: object, fallback_name: Optional[str] = None) -> Optional[Place]:
        if not isinstance(row, dict):
            return None
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (KeyError, TypeError, ValueError):
            return None

        address = row.get("address") if isinstance(row.get("address"), dict) else {}
        name = next(
            (address[key] for key in _PLACE_NAME_KEYS if address.get(key)),
            None,
        ) or row.get("name") or fallback_name
        if not name:
            # A coordinate with no nameable place is still useful for measuring a
            # distance, but there is nothing to call it — the caller decides.
            display = row.get("display_name") or ""
            name = display.split(",")[0].strip() or None
        if not name:
            return None

        return Place(
            name=str(name),
            lat=lat,
            lon=lon,
            state=address.get("state"),
            country=address.get("country"),
            display_name=row.get("display_name"),
        )

    async def geocode(self, query: str) -> Optional[Place]:
        """Where is this place? None when the provider cannot place it."""
        cleaned = " ".join((query or "").split())
        if not cleaned:
            return None

        key = f"fwd:{cleaned.lower()}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached or None      # a cached miss is stored as False

        try:
            payload = await self._get("/search", {
                "q": cleaned,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 1,
                # Him Cab operates in India; scoping the search stops "Manali" matching
                # a hamlet on another continent.
                "countrycodes": settings.geocoding_country_codes,
            })
        except Exception as e:
            log.warning("geocode_failed", query=cleaned, error=str(e), error_type=type(e).__name__)
            return None

        rows = payload if isinstance(payload, list) else []
        place = self._to_place(rows[0], fallback_name=cleaned) if rows else None
        self._cache.set(key, place or False, settings.geocoding_cache_ttl_seconds)
        if place is None:
            log.info("geocode_no_match", query=cleaned)
        return place

    async def reverse(self, lat: float, lon: float) -> Optional[Place]:
        """What is at these coordinates? None when the provider cannot say."""
        # Rounded to ~100 m for the cache key: a rider shifting a few metres between
        # messages is the same place, and a key per GPS jitter would never hit.
        key = f"rev:{round(lat, 3)},{round(lon, 3)}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached or None

        try:
            payload = await self._get("/reverse", {
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "addressdetails": 1,
                # Town/city level. Higher zoom returns a street address, which is not
                # what a pickup city should be called.
                "zoom": 10,
            })
        except Exception as e:
            log.warning("reverse_geocode_failed", lat=lat, lon=lon, error=str(e))
            return None

        place = self._to_place(payload)
        self._cache.set(key, place or False, settings.geocoding_cache_ttl_seconds)
        return place

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ── Singleton ─────────────────────────────────────────────────────────────────
_geocoding_service: Optional[GeocodingService] = None


def get_geocoding_service() -> GeocodingService:
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service
