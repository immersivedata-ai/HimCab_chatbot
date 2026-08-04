"""
app/services/external/weather.py
──────────────────────────────────
OpenWeatherMap API wrapper with Redis caching.
Falls back to cached data if API call fails.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.exceptions import WeatherServiceError
from app.core.logging import get_logger
from app.models.alerts import AlertSeverity, AlertType, TravelAlert, WeatherReport
from app.services.redis_cache import get_redis_cache

log = get_logger(__name__)

# Himachal Pradesh destination → lat/lon mapping
HIMACHAL_COORDS: dict[str, tuple[float, float]] = {
    "dharamshala":    (32.2190, 76.3234),
    "mcleod ganj":   (32.2431, 76.3214),
    "manali":         (32.2396, 77.1887),
    "shimla":         (31.1048, 77.1734),
    "kullu":          (31.9579, 77.1095),
    "kasol":          (32.0109, 77.3160),
    "spiti":          (32.2461, 78.0338),
    "dalhousie":      (32.5383, 75.9738),
    "chail":          (30.9632, 77.2085),
    "kufri":          (31.0985, 77.2675),
    "bir billing":    (32.0392, 76.7125),
    "khajjiar":       (32.5447, 76.0527),
    "kinnaur":        (31.5800, 78.3000),
    "chitkul":        (31.3530, 78.4370),
    "tirthan valley": (31.6300, 77.4000),
    "jibhi":          (31.6200, 77.3400),
    "sangla":         (31.4200, 78.2500),
    "narkanda":       (31.2600, 77.4500),
    "manikaran":      (32.0050, 77.3450),
    "solang valley":  (32.3200, 77.1700),
    "palampur":       (32.1100, 76.5400),
    "mandi":          (31.7100, 76.9300),
}


def _normalize_destination(destination: str) -> tuple[str, tuple[float, float]]:
    """Find the closest matching Himachal destination and its coordinates."""
    dest_lower = destination.lower().strip()

    # Exact prefix/suffix match
    for key, coords in HIMACHAL_COORDS.items():
        if key in dest_lower or dest_lower in key:
            return key, coords

    # Partial word match (e.g. "mcleodganj" → "mcleod ganj")
    for key, coords in HIMACHAL_COORDS.items():
        if any(word in dest_lower for word in key.split() if len(word) > 3):
            return key, coords

    # Default to Shimla for unknown destinations
    log.warning("weather_destination_fallback", destination=destination, fallback="shimla")
    return "shimla", HIMACHAL_COORDS["shimla"]


def _parse_weather_response(data: dict, destination: str) -> WeatherReport:
    """
    Parse OpenWeatherMap 2.5 API response into WeatherReport.
    data = {"current": <weather endpoint json>, "forecast": <forecast endpoint json>}
    """
    alerts: list[TravelAlert] = []
    current = data.get("current", {})
    forecast_raw = data.get("forecast", {}).get("list", [])

    # Build 5-day forecast (2.5 gives 3-hour intervals; sample one per day)
    forecast_7day = []
    seen_dates: set[str] = set()
    for entry in forecast_raw:
        date_str = datetime.fromtimestamp(
            entry["dt"], tz=timezone.utc
        ).strftime("%Y-%m-%d")
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)
        rain_mm = entry.get("rain", {}).get("3h", 0) if isinstance(entry.get("rain"), dict) else 0
        snow_mm = entry.get("snow", {}).get("3h", 0) if isinstance(entry.get("snow"), dict) else 0
        forecast_7day.append({
            "date": date_str,
            "min_c": round(entry["main"]["temp_min"], 1),
            "max_c": round(entry["main"]["temp_max"], 1),
            "condition": entry["weather"][0]["description"] if entry.get("weather") else "N/A",
            "rain_mm": rain_mm,
            "snow_mm": snow_mm,
        })

    main = current.get("main", {})
    wind = current.get("wind", {})
    weather_list = current.get("weather", [{}])

    # Check for snow in current conditions
    condition_desc = weather_list[0].get("description", "").lower() if weather_list else ""
    if "snow" in condition_desc or "blizzard" in condition_desc:
        alerts.append(TravelAlert(
            alert_id=str(uuid4()),
            type=AlertType.SNOWFALL,
            severity=AlertSeverity.HIGH,
            title="Snowfall Detected",
            description=f"Current conditions: {condition_desc}",
            affected_areas=[destination],
            source="OpenWeatherMap",
            issued_at=datetime.now(timezone.utc),
        ))

    # Check for heavy rain
    if "heavy rain" in condition_desc or "thunderstorm" in condition_desc:
        alerts.append(TravelAlert(
            alert_id=str(uuid4()),
            type=AlertType.RAIN,
            severity=AlertSeverity.MEDIUM,
            title="Heavy Rain Warning",
            description=f"Current conditions: {condition_desc}. Risk of landslides on hill roads.",
            affected_areas=[destination],
            source="OpenWeatherMap",
            issued_at=datetime.now(timezone.utc),
            action_recommended="Avoid night driving on mountain roads.",
        ))

    return WeatherReport(
        destination=destination,
        current_temp_c=round(main.get("temp", 0), 1),
        feels_like_c=round(main.get("feels_like", 0), 1),
        humidity_percent=main.get("humidity", 0),
        wind_speed_kmh=round(wind.get("speed", 0) * 3.6, 1),
        condition=condition_desc or "N/A",
        condition_icon=weather_list[0].get("icon") if weather_list else None,
        forecast_7day=forecast_7day,
        alerts=alerts,
        fetched_at=datetime.now(timezone.utc),
    )


class WeatherService:
    """
    Async weather service with Redis caching layer.
    Uses OpenWeatherMap free 2.5 endpoint — no paid subscription needed.
    Free tier: 1,000 calls/day, sufficient for production with Redis caching.
    """

    CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    _TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._cache = get_redis_cache()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
    )
    async def _fetch_from_api(self, lat: float, lon: float) -> dict:
        """Fetch current weather + 5-day forecast from free OWM 2.5 API."""
        params = {"lat": lat, "lon": lon, "appid": settings.openweather_api_key, "units": "metric"}
        client = await self._get_client()
        current_resp, forecast_resp = await asyncio.gather(
            client.get(self.CURRENT_URL, params=params),
            client.get(self.FORECAST_URL, params={**params, "cnt": 7}),
        )
        current_resp.raise_for_status()
        forecast_resp.raise_for_status()
        return {"current": current_resp.json(), "forecast": forecast_resp.json()}

    async def get_weather(self, destination: str) -> WeatherReport:
        """
        Fetch weather for a Himachal destination.
        Checks Redis cache first (30-min TTL), falls back to API.
        """
        dest_key, coords = _normalize_destination(destination)

        # 1. Check Redis cache
        cached = await self._cache.get_weather(dest_key)
        if cached:
            log.debug("weather_cache_hit", destination=dest_key)
            return WeatherReport.model_validate(cached)

        # 2. Fetch from API
        try:
            raw = await self._fetch_from_api(*coords)
            report = _parse_weather_response(raw, dest_key)

            # 3. Cache the result
            await self._cache.set_weather(dest_key, report.model_dump(mode="json"))
            log.info("weather_fetched", destination=dest_key)
            return report

        except httpx.HTTPStatusError as e:
            log.error(
                "weather_api_error",
                status=e.response.status_code,
                destination=dest_key,
                response_text=e.response.text[:300],
            )
            raise WeatherServiceError(f"Weather API error for {destination}: {e.response.status_code}")
        except Exception as e:
            log.error("weather_fetch_failed", destination=dest_key, error=str(e), error_type=type(e).__name__)
            raise WeatherServiceError(f"Could not fetch weather for {destination}")

    async def get_snowfall_risk(self, destination: str) -> dict:
        """
        Assess snowfall risk for a destination.
        Returns: {"risk": "high/medium/low/none", "expected_mm": float, "message": str}
        """
        report = await self.get_weather(destination)
        total_snow = sum(d.get("snow_mm", 0) for d in report.forecast_7day[:3])

        if total_snow > 20:
            return {"risk": "high", "expected_mm": total_snow, "message": "Heavy snowfall expected. Road closures possible."}
        elif total_snow > 5:
            return {"risk": "medium", "expected_mm": total_snow, "message": "Moderate snowfall expected. Drive carefully."}
        elif total_snow > 0:
            return {"risk": "low", "expected_mm": total_snow, "message": "Light snowfall possible."}
        return {"risk": "none", "expected_mm": 0.0, "message": "No snowfall expected."}

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ── Singleton ─────────────────────────────────────────────────────────────────
_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service
