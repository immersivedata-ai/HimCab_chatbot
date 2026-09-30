"""
app/integrations/himcab/fare_service.py
────────────────────────────────────
Async client for the Him Cab fare endpoint. Follows the maintained "Style A" pattern
(pooled httpx.AsyncClient + tenacity retry + module singleton), copied from
app/integrations/external/weather.py.

Everything above the "CONTRACT BOUNDARY" fence is contract-independent. Only
`_build_quote_request` and `_parse_quote_response` depend on the Spring Swagger — both
are confirmed against POST /api/rider/fare/estimate (see the fence below).
"""

import re
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.enums import TripType
from app.core.logging import get_logger
from app.integrations.himcab.models import FareCategoryFare, FareQuote, FareQuoteInput
from app.integrations.himcab.quote_builder import select_headline_category
from app.integrations.himcab.vehicles import select_cabs_for_group

log = get_logger(__name__)

#: How many cabs a reply may put in front of the rider. Three is a recommendation; a
#: rental is a CHOICE — the app's Rent a Car screen lists every cab and will not book
#: until one is selected, so trimming the list here would hide a cab the rider is
#: entitled to pick and then tell them it was not available.
_MAX_OPTIONS = 3
_MAX_RENTAL_OPTIONS = 6

#: Digits inside whatever the backend put in `seatCapacity`.
_SEAT_DIGITS = re.compile(r"\d+")


def _parse_seat_capacity(value: Any) -> Optional[int]:
    """Seats a category holds, from whatever `seatCapacity` the estimate endpoint sent.

    This exists because the real backend returns `seatCapacity` as a HUMAN RANGE STRING
    like "4 - 6" for some categories, not an integer — and `FareCategoryFare.seat_capacity`
    is an int. Handing "4 - 6" straight to the model raised a pydantic ValidationError
    that took down the WHOLE parse (every category, not just that row): it propagated out
    of `get_quote`, `tool_executor._safe_call` swallowed it to None, and the rider got a
    ride reply with NO cabs at all. It went unseen for a long time because a valid rider
    JWT is the only thing that ever reaches a real response — every stub and every expired
    token failed earlier, so the estimate body was never actually parsed until now.

    The UPPER bound is the capacity: a "4 - 6" cab seats up to six, so a party of six
    still fits it, and `vehicles.select_cabs_for_group` keeps cabs whose seats meet the
    group. None when the field is absent or carries no number, which the seat inference
    from the category name (`vehicles.seat_capacity_for`) then covers.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    nums = [int(n) for n in _SEAT_DIGITS.findall(str(value))]
    return max(nums) if nums else None


class _RetryableFareError(Exception):
    """Raised for transient backend failures (5xx) so tenacity retries them.
    4xx (bad JWT / bad contract) are NOT wrapped in this — they propagate immediately."""


class FareService:
    """Fetches real cab fares from the Him Cab Spring backend, authenticated with the
    rider's forwarded JWT. Never logs the token."""

    _TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    @property
    def _fare_url(self) -> str:
        return f"{settings.java_backend_url.rstrip('/')}{settings.himcab_fare_path}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        # ConnectError (refused) and ConnectTimeout/PoolTimeout (never got a socket) all
        # mean the fare host was not reachable, and all deserve the same retry. This tuple
        # used to carry ConnectError but not the timeouts, so a refused connection retried
        # three times while a connect timeout failed on the first attempt — the same
        # "backend is down" outcome reached two different ways, one of them flakier. A
        # fare miss is swallowed to None either way (no invented price), so the only thing
        # the asymmetry changed was how often a transient network blip cost the rider cabs.
        retry=retry_if_exception_type((
            _RetryableFareError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.ReadTimeout,
        )),
    )
    async def _post(self, url: str, body: dict, headers: dict) -> dict:
        """POST with retry on transient errors only. 4xx propagate immediately (no retry)."""
        client = await self._get_client()
        resp = await client.post(url, json=body, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Retry only server-side/transient failures. Client errors (401/403/422/404)
            # mean an expired JWT or a contract mismatch — retrying won't help.
            if status >= 500:
                log.warning("fare_api_5xx_retrying", status=status, url=url)
                raise _RetryableFareError(str(status)) from e
            log.warning("fare_api_client_error", status=status, response_text=e.response.text[:300])
            raise
        return resp.json()

    async def get_quote(self, quote_input: FareQuoteInput, rider_jwt: str) -> FareQuote:
        """Fetch a fare quote for the given input on behalf of the rider.

        Raises on failure (network, 4xx, exhausted 5xx retries) — the caller wraps this
        in tool_executor._safe_call, which turns any exception into None (graceful
        degradation to no price at all). The reply is never blocked by a fare miss.
        """
        # Defensive: strip an accidentally-included 'Bearer ' prefix so we never send a
        # doubled 'Bearer Bearer <token>' header (silently rejected as 401 by the backend).
        # chat.py already normalizes this at ingestion; this is belt-and-suspenders for
        # any other caller of get_quote.
        token = rider_jwt.strip()
        if token.lower().startswith("bearer "):
            log.warning("fare_jwt_had_bearer_prefix_stripped")
            token = token[len("bearer "):].strip()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        body = self._build_quote_request(quote_input)

        # NEVER log rider_jwt or the Authorization header.
        log.info(
            "fare_api_call",
            url=self._fare_url,
            trip_type=quote_input.trip_type,
            distance_km=quote_input.distance_km,
            has_jwt=bool(rider_jwt),
        )
        raw = await self._post(self._fare_url, body, headers)
        quote = self._parse_quote_response(raw, quote_input)
        log.info(
            "fare_api_success",
            base_fare_inr=quote.base_fare_inr,
            category_name=quote.category_name,
            options=len(quote.alternatives),
            group_size=quote_input.group_size,
        )
        return quote

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ══════════════════════════════════════════════════════════════════════════════
    # CONTRACT BOUNDARY — both methods CONFIRMED against POST /api/rider/fare/estimate.
    #
    # Request:  {"bookingTime": "HH:mm", "distanceInMeters": int, "rentHours": int,
    #            "tripType": "LOCAL"|"OUTSTATION"|"RENTAL"}
    # Response: {"message","statusCode","errorMessage","data": [ {categoryId, name,
    #            categoryImage, totalFare, seatCapacity}, ... ]}  (one row per category)
    # ══════════════════════════════════════════════════════════════════════════════

    def _build_quote_request(self, q: FareQuoteInput) -> dict:
        """Map FareQuoteInput → POST /api/rider/fare/estimate body (CONFIRMED shape).

        distanceInMeters is required; callers must not invoke this without a distance.
        bookingTime drives the night-charge window; rentHours applies only to RENTAL.
        """
        meters = round((q.distance_km or 0) * 1000)
        return {
            "bookingTime": q.booking_time,
            "distanceInMeters": meters,
            "rentHours": q.rent_hours,
            "tripType": q.trip_type,
        }

    def _parse_quote_response(self, raw: dict, q: FareQuoteInput) -> FareQuote:
        """Map the estimate response → FareQuote (headline + all category options).

        Envelope: {message, statusCode, errorMessage, data: [ per-category rows ]}.
        """
        data = raw.get("data") if isinstance(raw, dict) else None
        rows = data if isinstance(data, list) else []

        # The endpoint prices EVERY category, including ones this group cannot use.
        # Narrowing here means the app, the prompt and the headline all see the same
        # short list — nothing downstream can reintroduce a minibus for four people.
        priced = self._extract_categories(rows)
        options = select_cabs_for_group(
            priced,
            q.group_size,
            max_options=(
                _MAX_RENTAL_OPTIONS if q.trip_type == TripType.RENTAL else _MAX_OPTIONS
            ),
        )
        headline = select_headline_category(options, q.group_size, q.budget_range)

        if headline is None:
            # Nothing parseable → base 0; caller/writer will treat as unavailable.
            return FareQuote(
                base_fare_inr=0,
                trip_type=q.trip_type,
                distance_km=q.distance_km,
                alternatives=options,
                raw=raw if isinstance(raw, dict) else {},
            )

        return FareQuote(
            base_fare_inr=headline.total_fare_inr,
            currency="INR",
            category_id=headline.category_id,
            category_name=headline.category_name,
            vehicle_label=headline.category_name,   # category name doubles as the vehicle label
            seat_capacity=headline.seat_capacity,   # per-category, often null
            trip_type=q.trip_type,
            distance_km=q.distance_km,
            offer_applied_server_side=False,        # estimate takes no offerId
            alternatives=options,
            raw=raw if isinstance(raw, dict) else {},
        )

    @staticmethod
    def _extract_categories(rows: list) -> list[FareCategoryFare]:
        """Pull per-category (id, name, image, totalFare, seatCapacity) out of the rows.

        A row that will not parse is SKIPPED and logged, never raised. One malformed
        category used to take the whole quote down with it — the exception propagated out
        of get_quote, _safe_call turned it into None, and the rider got a ride reply with
        no cabs at all instead of the other categories that parsed perfectly well. A
        dropped category costs one option; a raised one costs them every option.
        """
        out: list[FareCategoryFare] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            fare = r.get("totalFare")
            if fare is None:
                continue
            try:
                out.append(FareCategoryFare(
                    category_id=r.get("categoryId"),
                    category_name=r.get("name"),
                    category_image=r.get("categoryImage"),
                    total_fare_inr=float(fare),
                    # The backend sends this as a range string ("4 - 6") for some tiers,
                    # not an int — parsed to the upper bound here so the strict model
                    # never sees a shape it will reject. See `_parse_seat_capacity`.
                    seat_capacity=_parse_seat_capacity(r.get("seatCapacity")),
                ))
            except (ValueError, TypeError) as e:
                # pydantic's ValidationError is a ValueError; float() raises ValueError.
                log.warning(
                    "fare_category_skipped",
                    category=r.get("name"),
                    total_fare_raw=r.get("totalFare"),
                    seat_capacity_raw=r.get("seatCapacity"),
                    error=str(e),
                )
        return out


# ── Singleton ─────────────────────────────────────────────────────────────────
_fare_service: Optional[FareService] = None


def get_fare_service() -> FareService:
    global _fare_service
    if _fare_service is None:
        _fare_service = FareService()
    return _fare_service
