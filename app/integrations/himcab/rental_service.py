"""
app/integrations/himcab/rental_service.py
──────────────────────────────────────
The rental packages Him Cab actually sells, read from the rate card.

WHY THIS EXISTS
───────────────
A rental is priced off a RentalChart, and the backend matches one by an EXACT hour
comparison (`TripService.findMatchingRentalChart`: `rentHours == chart.getMaxHour()`).
So "how many hours would you like?" is the wrong question — a rider who answers 5 when
the rate card defines 4, 8 and 12 matches no chart and gets no fare at all, with
nothing in the reply to explain why.

The assistant therefore offers the packages that are configured, and this module is
where it learns them: `GET /api/rider/fare/get-all`. Nothing about a package is stored
here — hours, included kilometres and the category they belong to all come from the
rate card, so changing a package there changes what the chatbot offers with no code
change and no redeploy.

WHAT IS SOLD
────────────
The rate card says what the backend can PRICE; `settings.rental_hour_options` says what
the product SELLS. It is an allowlist, and it wins: the rider is offered those durations
and no others (`_is_sold`). The card supplies the kilometres against them and nothing
more — a 6- or 12-hour chart left configured in the backend must not reappear as a chip
on a screen that sells 4 and 8.

There is no separate min/max window any more. Two settings describing one product
statement could disagree, and the one that lost was whichever the reader did not check;
anything wanting a range takes min()/max() of the same list.

When the rate card cannot be read at all, those same durations are offered anyway — see
`default_packages`. That still buys the rider a PRICE: the fare endpoint matches the
chart itself from `rentHours`, so a duration we could not parse a package for can still
be quoted (see `tool_executor._rental_tools`). What is lost is only what the package
INCLUDES, which nothing then claims.

WHAT THIS DOES NOT DO
─────────────────────
It does not price anything. `included_km` and the per-km rates are carried only so the
reply can say what a package INCLUDES; the fare itself always comes from
POST /fare/estimate (`fare_service.py`). Computing a price here from `farePerKilometer`
would produce a number that disagrees with the one the rider is charged at booking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.services.local_cache import TTLCache

log = get_logger(__name__)


class _RetryableRentalError(Exception):
    """A transient backend failure (5xx) worth retrying. 4xx propagate immediately."""


@dataclass(frozen=True)
class RentalPackage:
    """One bookable rental block, exactly as the rate card defines it."""

    hours: int
    #: Kilometres included in the package. None when the rate card does not say.
    included_km: Optional[float] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    #: False for a duration offered from product config because the rate card could not
    #: be read (`default_packages`). It is still choosable AND still priceable — the
    #: backend matches the chart from `rentHours` on its own — but `included_km` is
    #: unknown, so the reply may not say what the package covers.
    from_rate_card: bool = True

    @property
    def label(self) -> str:
        """How a rider would refer to it: "8 hrs / 80 km"."""
        if self.included_km:
            return f"{self.hours} hrs / {round(self.included_km)} km"
        return f"{self.hours} hrs"


def _as_int(value: Any) -> Optional[int]:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _first(row: dict, *keys: str) -> Any:
    """First present value among `keys` — the backend mixes camelCase and snake_case."""
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def distinct_packages(packages: list[RentalPackage]) -> list[RentalPackage]:
    """One entry per hour block, shortest first.

    The rate card defines a chart PER CATEGORY, so an 8-hour package appears once for
    every cab type. The rider is choosing a duration, not a duration-and-a-car — they
    pick the car afterwards from the priced options — so the duplicates are collapsed
    here rather than read out.
    """
    by_hours: dict[int, RentalPackage] = {}
    for package in packages:
        existing = by_hours.get(package.hours)
        # Keep whichever entry actually states its included kilometres.
        if existing is None or (existing.included_km is None and package.included_km):
            by_hours[package.hours] = package
    return [by_hours[h] for h in sorted(by_hours)]


def _is_sold(hours: int) -> bool:
    """True when `hours` is a duration the product sells. See WHAT IS SOLD.

    Membership, not a range: the configured list IS the product, so a chart the backend
    happens to price in between two of them is still not something a rider may choose.
    """
    return hours in settings.rental_hour_options


def default_packages() -> list[RentalPackage]:
    """The configured durations, for when the rate card cannot be read.

    Marked `from_rate_card=False` and carrying no included kilometres, because the card
    is where both come from. They exist so the rider still chooses from the durations Him
    Cab actually sells instead of being asked an open question that invites an answer no
    chart matches — and the fare endpoint can still price whichever one they pick.
    """
    return [
        RentalPackage(hours=h, from_rate_card=False)
        for h in settings.rental_hour_options
        if _is_sold(h)
    ]


def offered_packages(rate_card: list[RentalPackage]) -> list[RentalPackage]:
    """What the rider may choose from: the rate card, or the configured durations.

    THE one definition of "offered", so the chips on the asking turn, the durations the
    quote recognises and the list the reply reads out can never disagree.
    """
    return list(rate_card) if rate_card else default_packages()


def offered_hours(packages: list[dict]) -> list[int]:
    """The durations in a `packages_payload` list — what an answer is checked against."""
    return [p["hours"] for p in packages if p.get("hours")]


async def available_packages(rider_jwt: Optional[str]) -> list[dict]:
    """The rental blocks the rider may choose from right now.

    The one entry point for "what may we offer?", used by every place that asks the
    rider for a duration — the opening greeting, the clarifying question and the quote.
    They must agree: a package named in the welcome and unrecognised two turns later is
    worse than never having offered it.

    Never raises and never blocks a turn. Without a readable rate card (no token, a dead
    backend, a shape we cannot parse) this falls back to the configured durations, which
    the rider can still choose, be quoted for and be handed off with. The rate card
    itself is TTL-cached, so asking repeatedly is cheap.
    """
    # The feature flag is the one case that offers NOTHING: with rentals switched off
    # there is no product to fall back to. A missing token is different — the durations
    # are product config, not rider data, so they can still be offered unpriced.
    if not settings.enable_rentals:
        return []
    if not rider_jwt:
        return packages_payload(default_packages())
    try:
        rate_card = await get_rental_service().get_packages(rider_jwt=rider_jwt)
    except Exception as e:  # get_packages already swallows; this is the last net
        log.warning(
            "rental_packages_unavailable", error=str(e), error_type=type(e).__name__
        )
        rate_card = []
    return packages_payload(offered_packages(rate_card))


def packages_payload(packages: list[RentalPackage]) -> list[dict]:
    """Packages as the agent state and the `rental_packages` event carry them.

    One function so the chips the app renders while ASKING and the packages quoted
    while ANSWERING are the same objects — a rider must never be offered a duration on
    one turn that the next turn does not recognise.

    `from_rate_card` is false for a duration that came from product config rather than
    the rate card. Offer it as a tap like any other — it can still be priced — but treat
    its `included_km` as unknown rather than zero.
    """
    return [
        {
            "hours": p.hours,
            "included_km": p.included_km,
            "label": p.label,
            "from_rate_card": p.from_rate_card,
        }
        for p in packages
    ]


class RentalService:
    """Reads the rental rate card. Never raises to its callers."""

    _TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._cache = TTLCache()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    @property
    def _url(self) -> str:
        return f"{settings.java_backend_url.rstrip('/')}{settings.himcab_fare_getall_path}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(
            (_RetryableRentalError, httpx.ConnectError, httpx.ReadTimeout)
        ),
    )
    async def _get(self, url: str, headers: dict) -> dict:
        client = await self._get_client()
        response = await client.get(url, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status >= 500:
                log.warning("rental_api_5xx_retrying", status=status)
                raise _RetryableRentalError(str(status)) from e
            log.warning(
                "rental_api_client_error", status=status, response_text=e.response.text[:300]
            )
            raise
        return response.json()

    async def get_packages(self, rider_jwt: str) -> list[RentalPackage]:
        """The rate card's rental packages, shortest first. Empty on any failure.

        Narrowed to the bookable window, so everything downstream — the chips, the
        question, the quote, the reply — sees exactly the durations the product sells.

        Empty is a usable answer: the caller falls back to `default_packages`, so a
        rider still picks a real duration rather than being asked an open question.
        """
        cached = self._cache.get("packages")
        if cached is not None:
            return list(cached)

        token = rider_jwt.strip()
        if token.lower().startswith("bearer "):
            token = token[len("bearer "):].strip()

        try:
            # NEVER log the token.
            log.info("rental_api_call", url=self._url)
            raw = await self._get(
                self._url,
                {"Accept": "application/json", "Authorization": f"Bearer {token}"},
            )
        except Exception as e:
            log.warning(
                "rental_packages_unavailable", error=str(e), error_type=type(e).__name__
            )
            return []

        parsed = distinct_packages(self._parse_packages(raw))
        packages = [p for p in parsed if _is_sold(p.hours)]
        dropped = [p.hours for p in parsed if not _is_sold(p.hours)]
        if dropped:
            # Not an error — the backend prices these, the product does not sell them.
            # Logged because a rate card offering NOTHING we sell looks exactly like an
            # unreadable one from here on, and this line is the difference.
            log.info(
                "rental_packages_not_sold",
                dropped_hours=dropped,
                sold_hours=list(settings.rental_hour_options),
            )
        if packages:
            self._cache.set(
                "packages", packages, settings.rental_packages_cache_ttl_seconds
            )
        else:
            # Deliberately NOT cached. An empty parse means the backend answered with
            # something we could not read — a shape we do not handle, or a rate card
            # with no charts configured. Caching that would keep the assistant unable
            # to quote for an hour AFTER the cause was fixed, and would make the retry
            # that proves it fixed silently impossible.
            log.warning(
                "rental_rate_card_unreadable",
                url=self._url,
                envelope_keys=sorted(raw.keys()) if isinstance(raw, dict) else None,
                rows=len(self._fare_rows(raw)),
                hint="run scripts/check_rental.py --jwt <token> --raw",
            )
        log.info(
            "rental_packages_loaded",
            count=len(packages),
            hours=[p.hours for p in packages],
        )
        return list(packages)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ══════════════════════════════════════════════════════════════════════════
    # CONTRACT BOUNDARY — the only place that knows the rate card's JSON shape.
    #
    # GET /api/rider/fare/get-all → {"data": [ {categoryId, name/categoryName,
    #     rentalChartList: [ {maxHour, maxKilometer, farePerKilometer,
    #                         fareGreaterKilometer, gst}, ... ]}, ... ]}
    #
    # Field names mirror the Fare/RentalChart entities in the Spring service (see
    # docs/reference/java_backend/TripService.java). Both envelopes the backend uses
    # elsewhere are accepted — a bare list under `data`, and a paginated `content` —
    # and every key is read case-tolerantly, because this endpoint has not been
    # exercised against a live backend yet. A shape we do not recognise yields an
    # empty list, which degrades to asking the rider rather than failing the turn.
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_packages(self, payload: Any) -> list[RentalPackage]:
        packages: list[RentalPackage] = []
        for row in self._fare_rows(payload):
            charts = _first(row, "rentalChartList", "rentalCharts", "rental_chart_list")
            if not isinstance(charts, list):
                continue
            category_id = _as_int(_first(row, "categoryId", "id", "category_id"))
            category_name = _first(row, "categoryName", "name", "category_name")
            for chart in charts:
                if not isinstance(chart, dict):
                    continue
                hours = _as_int(_first(chart, "maxHour", "max_hour", "hours"))
                if hours is None:
                    continue
                packages.append(RentalPackage(
                    hours=hours,
                    included_km=_as_float(
                        _first(chart, "maxKilometer", "max_kilometer", "maxKm")
                    ),
                    category_id=category_id,
                    category_name=str(category_name) if category_name else None,
                ))
        return packages

    @staticmethod
    def _fare_rows(payload: Any) -> list[dict]:
        """Rate-card rows out of either envelope the backend uses."""
        data = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in ("content", "fares", "fareList"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [r for r in rows if isinstance(r, dict)]
            # A single fare object, not a collection.
            return [data]
        return []


# ── Singleton ─────────────────────────────────────────────────────────────────
_rental_service: Optional[RentalService] = None


def get_rental_service() -> RentalService:
    global _rental_service
    if _rental_service is None:
        _rental_service = RentalService()
    return _rental_service
