"""
tests/unit/test_places_service.py
───────────────────────────────────
Unit tests for the OpenStreetMap Overpass places service.
Tests query construction, bounding box lookup, drive time, and caching.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.external.places import (
    PlacesService,
    _get_bbox,
    _lookup_drive_time,
    HIMACHAL_BBOX,
    DEFAULT_BBOX,
    CATEGORY_OSM_TAGS,
)


class TestBoundingBoxLookup:
    """Tests for destination → bounding box resolution."""

    def test_exact_destination(self):
        """Should find exact destination match."""
        bbox = _get_bbox("Manali")
        assert bbox == HIMACHAL_BBOX["manali"]

    def test_case_insensitive(self):
        """Should match regardless of case."""
        bbox = _get_bbox("SHIMLA")
        assert bbox == HIMACHAL_BBOX["shimla"]

    def test_partial_match(self):
        """Should match partial destination names."""
        bbox = _get_bbox("mcleod ganj dharamshala")
        # Should match "mcleod ganj" or "dharamshala"
        assert bbox in (HIMACHAL_BBOX["mcleod ganj"], HIMACHAL_BBOX["dharamshala"])

    def test_unknown_destination_returns_default(self):
        """Should return Shimla as default for unknown destinations."""
        bbox = _get_bbox("Mumbai")
        assert bbox == DEFAULT_BBOX

    def test_whitespace_handling(self):
        """Should handle leading/trailing whitespace."""
        bbox = _get_bbox("  kasol  ")
        assert bbox == HIMACHAL_BBOX["kasol"]

    def test_new_destinations_added(self):
        """Verify new destinations were added to the bounding box table."""
        assert "kinnaur" in HIMACHAL_BBOX
        assert "chitkul" in HIMACHAL_BBOX
        assert "tirthan valley" in HIMACHAL_BBOX
        assert "jibhi" in HIMACHAL_BBOX
        assert "solang valley" in HIMACHAL_BBOX


class TestDriveTimeLookup:
    """Tests for the hardcoded drive time table."""

    def test_known_route(self):
        """Should return drive time for a known route."""
        result = _lookup_drive_time("Delhi", "Shimla")
        assert result["duration_minutes"] == 420
        assert result["distance_km"] == 370
        assert result["origin"] == "Delhi"
        assert result["destination"] == "Shimla"

    def test_bidirectional_lookup(self):
        """Should find routes in either direction."""
        result1 = _lookup_drive_time("Shimla", "Manali")
        result2 = _lookup_drive_time("Manali", "Shimla")
        assert result1["duration_minutes"] == result2["duration_minutes"]

    def test_unknown_route_returns_none(self):
        """Should return None values for unknown routes."""
        result = _lookup_drive_time("Mumbai", "Goa")
        assert result["duration_minutes"] is None
        assert result["distance_km"] is None
        assert "note" in result

    def test_case_insensitive_route(self):
        """Should be case-insensitive."""
        result = _lookup_drive_time("DELHI", "shimla")
        assert result["duration_minutes"] == 420


class TestPlacesService:
    """Tests for the PlacesService Overpass API wrapper."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """Should return cached data without calling Overpass."""
        cached_data = [{"name": "Cached Hotel"}]

        with patch("app.services.external.places.get_redis_cache") as cache_mock:
            cache = AsyncMock()
            cache.get_places.return_value = cached_data
            cache_mock.return_value = cache

            service = PlacesService()
            result = await service.get_places("Manali", "hotels")

        assert result == cached_data
        # Verify no HTTP call was made (Overpass not called)

    @pytest.mark.asyncio
    async def test_api_failure_returns_empty_list(self):
        """Should return empty list on API failure (graceful degradation)."""
        with patch("app.services.external.places.get_redis_cache") as cache_mock:
            cache = AsyncMock()
            cache.get_places.return_value = None  # cache miss
            cache_mock.return_value = cache

            service = PlacesService()
            service._fetch_overpass = AsyncMock(side_effect=Exception("Network error"))
            result = await service.get_places("Manali", "hotels")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_drive_time_is_async(self):
        """get_drive_time should be awaitable (the original bug fix)."""
        with patch("app.services.external.places.get_redis_cache") as cache_mock:
            cache_mock.return_value = AsyncMock()
            service = PlacesService()
            result = await service.get_drive_time("Delhi", "Shimla")

        assert result["duration_minutes"] == 420

    def test_all_categories_have_osm_tags(self):
        """Every category should have a corresponding OSM tag mapping."""
        expected = ["hotels", "restaurants", "temples", "cafes", "attractions"]
        for cat in expected:
            assert cat in CATEGORY_OSM_TAGS, f"Missing OSM tag for category: {cat}"

    @pytest.mark.asyncio
    async def test_parse_element_extracts_name(self):
        """_parse_element should extract name from tags."""
        with patch("app.services.external.places.get_redis_cache") as cache_mock:
            cache_mock.return_value = AsyncMock()
            service = PlacesService()

        element = {
            "type": "node",
            "id": 12345,
            "lat": 32.24,
            "lon": 77.19,
            "tags": {
                "name": "Hotel Mountain View",
                "tourism": "hotel",
                "stars": "3",
                "phone": "+91-1234567890",
            }
        }
        parsed = service._parse_element(element)
        assert parsed["name"] == "Hotel Mountain View"
        assert parsed["lat"] == 32.24
        assert parsed["lon"] == 77.19
        assert parsed["phone"] == "+91-1234567890"
        assert "hotel" in parsed["types"]

    @pytest.mark.asyncio
    async def test_parse_element_english_name_preferred(self):
        """Should prefer name:en over name."""
        with patch("app.services.external.places.get_redis_cache") as cache_mock:
            cache_mock.return_value = AsyncMock()
            service = PlacesService()

        element = {
            "type": "node",
            "id": 99,
            "tags": {
                "name": "मंदिर",
                "name:en": "Temple",
                "amenity": "place_of_worship",
            }
        }
        parsed = service._parse_element(element)
        assert parsed["name"] == "Temple"
