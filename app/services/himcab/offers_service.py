"""
app/services/himcab/offers_service.py
──────────────────────────────────────
Async client for the Him Cab offer endpoints. Same "Style A" shape as its neighbour
`fare_service.py` (pooled httpx.AsyncClient + tenacity retry + module singleton), and
the same rule: the rider's own JWT is forwarded, and never logged.

WHY THE ASSISTANT FETCHES THESE
───────────────────────────────
The reply used to carry a standing instruction never to mention a coupon, because the
only coupons it could produce were invented ones. That is the right rule when there is
no source of truth and the wrong one now there is: a rider quoted Rs 4,200 for Shimla →
Manali should be told about the Rs 300 coupon sitting in their account, not left to find
it on the payment screen two taps later.

So offers come from the backend or they do not appear at all. Nothing here generates,
rounds or "helpfully" adjusts a discount.

TWO ENDPOINTS, ONE FALLBACK
───────────────────────────
  • GET /api/rider/offer/getOffersByTripType/{riderId}?tripType=…  — personalised, and
    the one to use whenever the rider's numeric id is known
  • GET /api/rider/offer/get-all?status=true                       — everything live,
    filtered by trip type here instead

The fallback exists because the rider id is not guaranteed: it is read from the JWT
claims, and whether the Spring token carries one is up to the Spring team. Without the
fallback, a token without an id claim would mean no offers at all — a silent downgrade
that is invisible until someone asks why nobody uses coupons from the chat.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.services.himcab.models import Offer

log = get_logger(__name__)


class _RetryableOfferError(Exception):
    """Transient backend failure (5xx) — retried. 4xx propagate immediately."""


def _parse_moment(value: Any) -> Optional[datetime]:
    """Read a backend timestamp, in any of the three shapes it arrives in.

    The Swagger says date-time, the Android client's model says Long, and the wire
    carries whichever the serialiser felt like. Rather than pick one and be wrong for
    half the offers, accept epoch millis, epoch seconds and ISO-8601 alike. An
    unparseable value becomes None, which reads as "no bound on this side" — a coupon
    is never hidden because its date failed to parse.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000.0 if value > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _trip_types(row: dict) -> list[str]:
    """Trip types an offer covers, from either field name the backend uses.

    `offerTripTypes` is a list (the OfferDTO in Swagger); `offerTripType` is a single
    string (what the Android model reads). Both appear, so both are read.
    """
    raw = row.get("offerTripTypes") or row.get("offerTripType") or []
    values = raw if isinstance(raw, list) else [raw]
    return [str(v).strip().upper() for v in values if v not in (None, "")]


def _parse_offer(row: dict) -> Optional[Offer]:
    """One backend row → Offer, or None when it carries no usable discount."""
    if not isinstance(row, dict):
        return None
    try:
        value = float(row.get("couponValue") or 0)
    except (TypeError, ValueError):
        return None
    code = row.get("couponCode")
    # A coupon with no code cannot be applied, and one worth nothing is not an offer.
    if value <= 0 or not code:
        return None

    try:
        minimum = float(row.get("minimumValue") or 0)
    except (TypeError, ValueError):
        minimum = 0.0

    return Offer(
        offer_id=row.get("id"),
        coupon_code=str(code).strip().upper(),
        coupon_name=row.get("couponName"),
        coupon_value_inr=value,
        minimum_value_inr=max(0.0, minimum),
        trip_types=_trip_types(row),
        user_type=row.get("offerUserType"),
        terms_and_conditions=row.get("termsAndConditions"),
        image=row.get("image"),
        live_date=_parse_moment(row.get("liveDate")),
        expire_date=_parse_moment(row.get("expireDate")),
    )


def _rows_from(payload: Any) -> list[dict]:
    """Pull offer rows out of either envelope the backend returns.

    `getOffersByTripType` answers with `data: [...]`; the paginated `get-all` answers
    with `data: {content: [...]}`. One reader for both keeps the caller from having to
    care which endpoint served it.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            return [r for r in content if isinstance(r, dict)]
    return []


def usable_offers(
    offers: list[Offer],
    trip_type: str,
    fare_inr: Optional[float] = None,
    *,
    now: Optional[datetime] = None,
) -> list[Offer]:
    """The offers this rider can actually put on THIS trip, best saving first.

    Three filters, each removing a coupon that would otherwise be mentioned and then
    rejected at checkout: wrong trip type, outside its date window, or a minimum spend
    above the quoted fare. With no fare yet (the quote failed), the minimum-spend check
    is skipped rather than guessed at — the offer is real, we just cannot say whether
    this trip clears the bar.
    """
    moment = now or datetime.now(timezone.utc)
    eligible = [
        offer for offer in offers
        if offer.applies_to(trip_type)
        and offer.is_live(moment)
        and (fare_inr is None or fare_inr >= offer.minimum_value_inr)
    ]
    eligible.sort(key=lambda o: o.coupon_value_inr, reverse=True)
    return eligible


class OffersService:
    """Fetches live Him Cab coupons on behalf of a rider, using their forwarded JWT."""

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

    def _url(self, path: str) -> str:
        return f"{settings.java_backend_url.rstrip('/')}{path}"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(
            (_RetryableOfferError, httpx.ConnectError, httpx.ReadTimeout)
        ),
    )
    async def _get(self, url: str, params: dict, headers: dict) -> dict:
        """GET with retry on transient errors only.

        One retry rather than the fare service's two: offers are an enhancement to the
        reply, and holding the whole turn open to chase them costs the rider more than
        the coupon is worth.
        """
        client = await self._get_client()
        resp = await client.get(url, params=params, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status >= 500:
                log.warning("offers_api_5xx_retrying", status=status, url=url)
                raise _RetryableOfferError(str(status)) from e
            log.warning(
                "offers_api_client_error", status=status, response_text=e.response.text[:300]
            )
            raise
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    async def get_offers(
        self,
        rider_jwt: str,
        trip_type: str = "ALL",
        rider_id: Optional[int] = None,
    ) -> list[Offer]:
        """Live coupons for this rider, personalised when their id is known.

        Raises on failure; the caller wraps this in `tool_executor._safe_call`, which
        turns any exception into None. A missing offer costs the rider a discount they
        can still find in the app — it must never cost them the reply.
        """
        token = rider_jwt.strip()
        if token.lower().startswith("bearer "):
            token = token[len("bearer "):].strip()

        headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        wanted = (trip_type or "ALL").upper()

        if rider_id is not None:
            url = self._url(settings.himcab_offers_by_trip_type_path.format(riderId=rider_id))
            params: dict[str, Any] = {"tripType": wanted}
        else:
            # No rider id in the token — fall back to the catalogue and filter here.
            url = self._url(settings.himcab_offers_all_path)
            params = {"page": 0, "size": settings.himcab_offers_page_size, "status": True}

        # NEVER log rider_jwt or the Authorization header.
        log.info(
            "offers_api_call",
            url=url,
            trip_type=wanted,
            personalised=rider_id is not None,
        )
        raw = await self._get(url, params, headers)
        offers = [o for o in (_parse_offer(r) for r in _rows_from(raw)) if o is not None]
        log.info("offers_api_success", count=len(offers), trip_type=wanted)
        return offers

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ── Singleton ─────────────────────────────────────────────────────────────────
_offers_service: Optional[OffersService] = None


def get_offers_service() -> OffersService:
    global _offers_service
    if _offers_service is None:
        _offers_service = OffersService()
    return _offers_service
